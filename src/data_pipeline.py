#!/usr/bin/env python3
import logging
import math
import os
import shutil

import hydra
import pandas as pd
import rootutils
from dotenv import load_dotenv
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.db import InfluxDBWrapper
from src.data.measurement_file import MeasurementFile
from src.extraction.fft import extract_fft_features
from src.extraction.moving_average import calculate_moving_average, calc_window_size
from src.extraction.denoising import apply_wavelet_denoising, apply_butterworth_filter
from src.processing.time_series import crop_signal, resample_signal, create_segments
from src.utils import get_partition_paths

load_dotenv()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool):
    """Set up logging configuration based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    logger.disabled = not verbose


def load_single_file_data(zip_file_path: str):
    """Load a single measurement file from a zip archive."""
    file = MeasurementFile(zip_file_path)
    file_data = file.get_sensor_data()
    file_data['file_hash'] = file.generate_file_hash()
    file_data['label'] = file.get_label()
    file_data.rename(columns={'time': '_time'}, inplace=True)
    file_data['_time'] = pd.to_datetime(file_data['_time'], unit='ns')
    file_data.reset_index(drop=True, inplace=True)
    return file_data


def query_raw_data(cfg: dict) -> pd.DataFrame:
    """Query raw data from InfluxDB and cache it if necessary."""
    cache_dir = cfg.paths.cache_data_dir
    cache_path = os.path.join(cache_dir, 'raw_data_db_cache.parquet')
    if cfg.database.use_cache and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            logger.info("Loaded data from cache.")
            return df
        except FileNotFoundError:
            logger.warning("Cache file not found. Querying data from InfluxDB...")

    with InfluxDBWrapper() as client:
        query_api = client.query_api
        query = f'''
            from(bucket: "{cfg.database.bucket}")
              |> range(start: 0, stop: now())
              |> filter(fn: (r) => r._measurement == "measurement")
              |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        data_frames = query_api.query_data_frame(query, org=client.org)
        df = pd.concat(data_frames) if isinstance(data_frames, list) else data_frames

    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(cache_path)

    logger.debug(f"Data shape: {df.shape}"
                 f"\nColumns: {df.columns}"
                 f"\nUnique Labels: {df['label'].unique()}"
                 f"\nUnique File Hashes: {df['file_hash'].nunique()}")
    return df


def get_session_lengths(df: pd.DataFrame, resample_rate_hz: float) -> pd.DataFrame:
    return (df.groupby("session_id")
            .agg({"session_id": "count"})
            .rename(columns={"session_id": "count"}) / resample_rate_hz)


def truncate_sessions(segments_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Truncate sessions to a maximum length."""
    assert 'session_id' in segments_df.columns, "Session ID column not found in the DataFrame."
    assert 'segment_id' in segments_df.columns, "Segment ID column not found in the DataFrame."

    resample_rate_hz = cfg.preprocessing.resample_rate_hz
    segment_size_seconds = cfg.preprocessing.segment_size_seconds
    max_session_length_s = cfg.preprocessing.get("max_session_length_s", None)

    logger.info(
        f"Session lengths before truncation: {get_session_lengths(segments_df, resample_rate_hz).describe()}")

    logger.info(f"Truncating sessions to a maximum length of {max_session_length_s} seconds.")

    max_count_segments = math.floor(max_session_length_s / segment_size_seconds)
    truncated_segments_df = segments_df.groupby('session_id').apply(
        lambda x: x[x['segment_id'].isin(
            x['segment_id'].drop_duplicates().sample(
                n=min(x['segment_id'].drop_duplicates().shape[0], max_count_segments))
        )]
    ).reset_index(drop=True)

    logger.info(
        f"Session lengths after truncation: {get_session_lengths(truncated_segments_df, resample_rate_hz).describe()}")

    return truncated_segments_df


def prepare_time_series_segments(data: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    assert '_time' in data.columns, "Time column not found in the DataFrame."
    assert 'file_hash' in data.columns, "File hash column not found in the DataFrame."
    assert 'label' in data.columns, "Label column not found in the DataFrame."

    unwanted_columns = ['result', 'table', '_start', '_stop', '_measurement',
                        'device_id', 'measurement_group', 'platform',
                        'device_name', 'app_version', 'recording_time']
    data.drop(columns=unwanted_columns, inplace=True, errors='ignore')

    data['_time'] = pd.to_datetime(data['_time'], unit='ns')
    data = data.set_index('_time', drop=True)

    segments_df = pd.DataFrame()
    skipped_files = 0
    for file_hash, file_data in tqdm(data.groupby('file_hash'), desc="Processing Files", unit="file"):
        if file_data.isnull().values.any() or file_data.isna().values.any():
            columns_with_nan = file_data.columns[file_data.isnull().any()].tolist()
            nan_percentage_per_column = file_data[columns_with_nan].isnull().mean() * 100

            # logger.warning(
            #    f"File {file_hash} contains NaN values: ({', '.join([f'{k}: {v:.2f}%' for k, v in nan_percentage_per_column.items()])}).")
            # logger.warning(f"Skipping file {file_hash}.")
            skipped_files += 1
            continue

        label = file_data['label'].iloc[0]

        float_columns = file_data.select_dtypes(include=['float64']).columns
        file_data = file_data[float_columns]

        try:
            cropped_file_data = crop_signal(file_data,
                                            cfg.preprocessing.crop.start_seconds,
                                            cfg.preprocessing.crop.end_seconds)
        except Exception as e:
            logger.warning(f"Error processing file {file_hash}. Skipping file: {e}")
            skipped_files += 1
            continue

        try:
            resampled_file_data = resample_signal(cropped_file_data,
                                                  cfg.preprocessing.resample_rate_hz)
        except Exception as e:
            logger.warning(f"Error resampling file {file_hash}. Skipping file: {e}")
            skipped_files += 1
            continue

        try:
            file_segments = create_segments(resampled_file_data,
                                            cfg.preprocessing.segment_size_seconds,
                                            cfg.preprocessing.overlap_seconds)
        except Exception as e:
            logger.warning(f"Error segmenting file {file_hash}. Skipping file: {e}")
            skipped_files += 1
            continue

        segment_df = pd.DataFrame()
        for i, segment in enumerate(file_segments):
            segment = segment.copy()
            segment.reset_index(inplace=True)  # Reset index to get the time column
            segment.loc[:, 'segment_id'] = file_hash + '_' + str(i)
            segment.loc[:, 'session_id'] = file_hash
            segment.loc[:, 'label'] = label
            segment_df = pd.concat([segment_df, segment], axis=0, ignore_index=True)

        segments_df = pd.concat([segments_df, segment_df], axis=0, ignore_index=True)

    if segments_df.empty:
        raise ValueError("No segments were created for the given data. Check the input data and configuration.")

    segments_df.reset_index(inplace=True, drop=True)

    logger.info(f"Segments created: {len(segments_df['segment_id'].unique())}")
    logger.info(f"Files processed: {len(data['file_hash'].unique()) - skipped_files}, Skipped files: {skipped_files}")
    return segments_df


def extract_segment_features(df: pd.DataFrame, segment_id: str, source_cols: list, cfg: dict) -> pd.DataFrame:
    """Extract features for a specific segment with optimized pandas operations."""
    segment = df[df['segment_id'] == segment_id]
    segment_features = {}

    for column in source_cols:
        # FFT features
        if cfg.preprocessing.feature_extraction.get("use_fft", None):
            fft_features = extract_fft_features(segment[column], cfg.preprocessing.resample_rate_hz)
            segment_features.update({
                f'{column}_{k}': pd.Series([v] * len(segment), index=segment.index)
                for k, v in fft_features.items()
            })

        # Smoothing
        smoothing = cfg.preprocessing.feature_extraction.get("smoothing", None)
        if smoothing:
            if smoothing in ('butterworth', True):
                segment_features[f'{column}_butterworth_smoothed'] = apply_butterworth_filter(segment[column],
                                                                                              order=4,
                                                                                              cutoff=0.1,
                                                                                              sampling_rate=cfg.preprocessing.resample_rate_hz)
            elif smoothing == 'wavelet':
                segment_features[f'{column}_wavelet_smoothed'] = apply_wavelet_denoising(segment[column],
                                                                                         wavelet='db4',
                                                                                         level=1)

        # Moving average
        moving_window_size_s = cfg.preprocessing.feature_extraction.get("moving_window_size_s", None)
        if moving_window_size_s:
            moving_window_size = calc_window_size(moving_window_size_s, cfg.preprocessing.resample_rate_hz)
            segment_features[f'{column}_moving_avg'] = calculate_moving_average(segment, column, moving_window_size)

        # Pearson correlation
        if cfg.preprocessing.feature_extraction.get("use_pears_corr", None):
            for other_column in source_cols:
                if other_column != column:
                    segment_features[f'{column}_{other_column}_correlation'] = segment[column].corr(
                        segment[other_column])

    return pd.DataFrame(segment_features, index=segment.index)


def extract_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Extract features from DataFrame using optimized parallel processing."""
    segment_ids = df['segment_id'].unique()
    source_cols = df.select_dtypes(include=['float64']).columns
    len_before_extraction = len(df)

    n_jobs = cfg.get("n_jobs", None)
    if n_jobs:
        results = Parallel(n_jobs=n_jobs)(delayed(extract_segment_features)(
            df, segment_id, source_cols, cfg) for segment_id in tqdm(segment_ids, desc="Extracting features"))
    else:
        results = [
            extract_segment_features(df, segment_id, source_cols, cfg)
            for segment_id in tqdm(segment_ids, desc="Extracting features")]

    features_df = pd.concat(results, axis=0)
    df = pd.concat([df, features_df], axis=1)
    assert len(df) == len_before_extraction, \
        f"Data length mismatch after feature extraction: {len(df)} vs {len_before_extraction}"
    return df


def split_data(df: pd.DataFrame, split_by: str, val_size: float, stratify: bool, k_folds: int = None) -> tuple | list:
    """Split data into train and validation sets or perform K-fold split."""
    assert split_by in df.columns, "Split column not found in the DataFrame."

    grouped = df.groupby(split_by).first().reset_index()
    assert grouped[split_by].nunique() == len(grouped), "Split column contains duplicate values."

    if k_folds:
        folds = []

        if stratify:
            kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=1337)
            splits = kf.split(grouped, grouped['label'])
        else:
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=1337)
            splits = kf.split(grouped)

        for train_idx, val_idx in splits:
            train_ids = grouped.loc[train_idx, split_by]
            val_ids = grouped.loc[val_idx, split_by]
            train_mask = df[split_by].isin(train_ids)
            val_mask = df[split_by].isin(val_ids)
            folds.append((df[train_mask], df[val_mask]))

        return folds
    else:
        train_idxs, val_idxs = train_test_split(grouped,
                                                test_size=val_size,
                                                stratify=grouped['label'] if stratify else None,
                                                shuffle=True, random_state=1337)
        train_ids = train_idxs[split_by]
        val_ids = val_idxs[split_by]
        train_mask = df[split_by].isin(train_ids)
        val_mask = df[split_by].isin(val_ids)
        return df[train_mask], df[val_mask]


def get_scaler(scaler_type: str) -> object:
    """Select the appropriate scaler based on the input type."""
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }
    return scalers.get(scaler_type, StandardScaler())


def scale_data(train_data: pd.DataFrame,
               val_data: pd.DataFrame,
               cfg: dict) -> tuple:
    """Apply scaling to the training and validation data."""
    scaler = get_scaler(cfg.preprocessing.scaling.type)
    float_columns = train_data.select_dtypes(include=['float64']).columns
    train_data.loc[:, float_columns] = scaler.fit_transform(train_data[float_columns])
    val_data.loc[:, float_columns] = scaler.transform(val_data[float_columns])
    return train_data, val_data


def transform_data(train_data: pd.DataFrame,
                   val_data: pd.DataFrame,
                   cfg: dict) -> tuple:
    """Apply PCA to the training and validation data."""
    if cfg.preprocessing.get("pca", None) and cfg.preprocessing.pca.get("components", None):
        float_columns = train_data.select_dtypes(include=['float64']).columns

        X_train_pca = train_data[float_columns].copy().fillna(0)
        X_val_pca = val_data[float_columns].copy().fillna(0)

        pca_components = cfg.preprocessing.pca.components

        pca = PCA(n_components=pca_components)
        X_train_pca = pca.fit_transform(X_train_pca)
        X_val_pca = pca.transform(X_val_pca)
        pca_columns = [f'pca_{i}' for i in range(pca_components)]
        train_data = pd.concat([train_data, pd.DataFrame(X_train_pca, columns=pca_columns)], axis=1)
        val_data = pd.concat([val_data, pd.DataFrame(X_val_pca, columns=pca_columns)], axis=1)

    return train_data, val_data


def save_partitions(train_data: pd.DataFrame, val_data: pd.DataFrame, paths: dict) -> None:
    """Save the training and validation data to disk."""
    for path in paths.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)

    train_data.to_parquet(paths['train'])
    val_data.to_parquet(paths['validate'])


def scale_and_transform_data(train_data: pd.DataFrame, val_data: pd.DataFrame, cfg: dict) -> tuple:
    """Process data by scaling and possibly transforming with PCA."""
    train_data, val_data = scale_data(train_data, val_data, cfg)
    train_data, val_data = transform_data(train_data, val_data, cfg)
    return train_data, val_data


@hydra.main(version_base="1.3", config_path="../configs", config_name="pipeline.yaml")
def pipeline(cfg):
    print(OmegaConf.to_yaml(cfg))

    setup_logging(cfg.get("verbose", True))

    if cfg.get("pca_components", None):
        raise ValueError("PCA components not implemented yet.")

    if not cfg.get("n_jobs", None):
        logger.warning("Using single-threaded processing. This may take a while for more features.")

    measurement_file_path = cfg.get("measurement_file_path", None)
    if measurement_file_path:
        logger.info("Loading single measurement file...")
        df = load_single_file_data(measurement_file_path)
    else:
        logger.info("Querying data...")
        df = query_raw_data(cfg)

    logger.info("Preprocessing data...")
    segments_df = prepare_time_series_segments(df, cfg)

    if cfg.preprocessing.get("max_session_length_s", None):
        segments_df = truncate_sessions(segments_df, cfg)

    logger.info("Extracting features...")
    segments_df = extract_features(segments_df, cfg)

    if measurement_file_path:
        logger.info("Done.")
        return segments_df

    partitioned_data_dir = cfg.paths.get("partitioned_data_dir")
    if os.path.exists(partitioned_data_dir):
        logger.info(f"Clearing existing partitioned data directory: {partitioned_data_dir}")
        shutil.rmtree(partitioned_data_dir)
        os.makedirs(partitioned_data_dir, exist_ok=True)

    logger.info("Splitting and processing data...")

    split_by = cfg.partitioning.split_by
    logger.debug(
        f"Segments label counts by {split_by}: {segments_df.groupby(split_by).first()['label'].value_counts()}")

    logger.debug(
        f"Segments label counts by {split_by} (%): {segments_df.groupby(split_by).first()['label'].value_counts(normalize=True)}")

    k_folds = cfg.partitioning.k_folds
    validation_size = cfg.partitioning.validation_size
    stratify = cfg.partitioning.stratify
    if k_folds:
        paths = get_partition_paths(partitioned_data_dir, k_folds=k_folds)
        folds = split_data(segments_df, split_by, validation_size, stratify, k_folds)

        for i, (train_data, val_data) in enumerate(folds):
            logger.debug(f"Fold {i + 1}: Train: {len(train_data)} ({len(train_data) / len(segments_df) * 100:.2f}%)"
                         f" - Validate: {len(val_data)} ({len(val_data) / len(segments_df) * 100:.2f}%)")

            train_data, val_data = scale_and_transform_data(train_data, val_data, cfg)
            save_partitions(train_data, val_data, paths[i])
    else:
        train_data, val_data = split_data(segments_df, split_by, validation_size, stratify)
        logger.debug(f"Train: {len(train_data)} ({len(train_data) / len(segments_df) * 100:.2f}%)"
                     f" - Validate: {len(val_data)} ({len(val_data) / len(segments_df) * 100:.2f}%)")

        train_data, val_data = scale_and_transform_data(train_data, val_data, cfg)
        save_partitions(train_data, val_data, get_partition_paths(partitioned_data_dir, k_folds=k_folds))

    logger.info("Data preparation completed.")


if __name__ == "__main__":
    pipeline()
