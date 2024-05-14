#!/usr/bin/env python3
import logging
import os
import shutil

import pandas as pd
import rootutils
import typer
from dotenv import load_dotenv
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.db import InfluxDBWrapper
from src.data.measurement_file import MeasurementFile
from src.extraction.fft import extract_fft_features
from src.extraction.moving_average import calculate_moving_average, calc_window_size
from src.processing.time_series import crop_signal, resample_signal, create_segments
from src.utils import get_env_variable, get_partition_paths

load_dotenv()

app = typer.Typer()
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool):
    """Set up logging configuration based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    logger.disabled = not verbose


def load_single_file_data(zip_file_path):
    """Load a single measurement file from a zip archive."""
    file = MeasurementFile(zip_file_path)
    file_data = file.get_sensor_data()
    file_data['file_hash'] = file.generate_file_hash()
    file_data['label'] = file.get_label()
    file_data.rename(columns={'time': '_time'}, inplace=True)
    file_data['_time'] = pd.to_datetime(file_data['_time'], unit='ns')
    file_data.reset_index(drop=True, inplace=True)
    return file_data


def query_raw_data(use_cache: bool) -> pd.DataFrame:
    """Query raw data from InfluxDB and cache it if necessary."""
    cache_dir = os.path.join(get_env_variable("DATA_DIR"), "cache")
    cache_path = os.path.join(cache_dir, 'raw_data_db_cache.parquet')
    if use_cache:
        try:
            df = pd.read_parquet(cache_path)
            logger.info("Loaded data from cache.")
            return df
        except FileNotFoundError:
            logger.warning("Cache file not found. Querying data from InfluxDB...")

    influxdb_bucket = get_env_variable('INFLUXDB_INIT_BUCKET')
    with InfluxDBWrapper() as client:
        query_api = client.query_api
        query = f'''
            from(bucket: "{influxdb_bucket}")
              |> range(start: -1y)
              |> filter(fn: (r) => r._measurement == "measurement")
              |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        data_frames = query_api.query_data_frame(query, org=client.org)
        df = pd.concat(data_frames) if isinstance(data_frames, list) else data_frames

    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(cache_path)
    return df


def prepare_time_series_segments(data: pd.DataFrame,
                                 start_crop_seconds: float,
                                 end_crop_seconds: float,
                                 resample_rate_hz: float,
                                 segment_size_seconds: float,
                                 overlap_seconds: float) -> pd.DataFrame:
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
            logger.warning(f"File {file_hash} contains NaN values. Skipping file.")
            skipped_files += 1
            continue

        label = file_data['label'].iloc[0]

        float_columns = file_data.select_dtypes(include=['float64']).columns
        file_data = file_data[float_columns]

        try:
            cropped_file_data = crop_signal(file_data, start_crop_seconds, end_crop_seconds)
        except Exception as e:
            logger.error(f"Error processing file {file_hash}. Skipping file: {e}")
            skipped_files += 1
            continue

        try:
            resampled_file_data = resample_signal(cropped_file_data, resample_rate_hz)
        except Exception as e:
            logger.error(f"Error resampling file {file_hash}. Skipping file: {e}")
            skipped_files += 1
            continue

        try:
            file_segments = create_segments(resampled_file_data, segment_size_seconds, overlap_seconds)
        except Exception as e:
            logger.error(f"Error segmenting file {file_hash}. Skipping file: {e}")
            skipped_files += 1
            continue

        segment_df = pd.DataFrame()
        for i, segment in enumerate(file_segments):
            segment = segment.copy()
            segment.reset_index(inplace=True)  # Reset index to get the time column
            segment.loc[:, 'segment_id'] = file_hash + '_' + str(i)
            segment.loc[:, 'label'] = label
            segment_df = pd.concat([segment_df, segment], axis=0, ignore_index=True)

        segments_df = pd.concat([segments_df, segment_df], axis=0, ignore_index=True)

    if segments_df.empty:
        raise ValueError("No segments were created. Check the data preprocessing steps.")

    segments_df.reset_index(inplace=True, drop=True)

    logger.info(f"Segments created: {len(segments_df['segment_id'].unique())}")
    logger.info(f"Files processed: {len(data['file_hash'].unique()) - skipped_files}, Skipped files: {skipped_files}")
    return segments_df


def extract_segment_features(df, segment_id, columns, moving_window_size, use_fft, use_pears_corr):
    """Extract features for a specific segment with optimized pandas operations."""
    segment = df[df['segment_id'] == segment_id]
    segment_features = {}

    for column in columns:
        # FFT features
        if use_fft:
            fft_features = extract_fft_features(segment[column])
            segment_features.update({
                f'{column}_{k}': pd.Series([v] * len(segment), index=segment.index)
                for k, v in fft_features.items()
            })

        # Moving average
        if moving_window_size:
            segment_features[f'{column}_moving_avg'] = calculate_moving_average(segment, column, moving_window_size)

        # Pearson correlation
        if use_pears_corr:
            for other_column in columns:
                if other_column != column:
                    segment_features[f'{column}_{other_column}_correlation'] = segment[column].corr(
                        segment[other_column])

    return pd.DataFrame(segment_features, index=segment.index)


def extract_features(df: pd.DataFrame, multi_processing: bool, n_jobs: int,
                     moving_window_size_s: float, use_fft: bool, use_pears_corr: bool) -> pd.DataFrame:
    """Extract features from DataFrame using optimized parallel processing."""
    float_columns = df.select_dtypes(include=['float64']).columns
    moving_window_size = calc_window_size(moving_window_size_s) if moving_window_size_s else None
    segment_ids = df['segment_id'].unique()

    len_before = len(df)

    if multi_processing:
        results = Parallel(n_jobs=n_jobs)(delayed(extract_segment_features)(
            df, segment_id, float_columns, moving_window_size, use_fft, use_pears_corr)
                                          for segment_id in tqdm(segment_ids, desc="Extracting features"))
    else:
        results = [
            extract_segment_features(df, segment_id, float_columns, moving_window_size, use_fft, use_pears_corr)
            for segment_id in tqdm(segment_ids, desc="Extracting features")]

    features_df = pd.concat(results, axis=0)
    df = pd.concat([df, features_df], axis=1)
    assert len(df) == len_before, f"Data length mismatch after feature extraction: {len(df)} vs {len_before}"
    return df


def split_data(df: pd.DataFrame, val_size: float, k_folds: int = None) -> tuple | list:
    """Split data into train and validation sets or perform K-fold split."""
    grouped = df.groupby('segment_id').first().reset_index()
    assert grouped['segment_id'].nunique() == len(grouped), "Duplicate segment IDs found in the dataset."

    if k_folds:
        folds = []
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=1337)
        for train_idx, val_idx in kf.split(grouped):
            train_ids = grouped.loc[train_idx, 'segment_id']
            val_ids = grouped.loc[val_idx, 'segment_id']
            train_mask = df['segment_id'].isin(train_ids)
            val_mask = df['segment_id'].isin(val_ids)
            folds.append((df[train_mask], df[val_mask]))

        return folds
    else:
        train_ids, val_ids = train_test_split(grouped['segment_id'], test_size=val_size, stratify=grouped['label'])
        train_mask = df['segment_id'].isin(train_ids)
        val_mask = df['segment_id'].isin(val_ids)
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
               scaler_type: str = 'standard'):
    """Apply scaling to the training and validation data."""
    scaler = get_scaler(scaler_type)
    float_columns = train_data.select_dtypes(include=['float64']).columns
    train_data.loc[:, float_columns] = scaler.fit_transform(train_data[float_columns])
    val_data.loc[:, float_columns] = scaler.transform(val_data[float_columns])
    return train_data, val_data


def transform_data(train_data: pd.DataFrame,
                   val_data: pd.DataFrame,
                   pca_components: int):
    """Apply PCA to the training and validation data."""
    float_columns = train_data.select_dtypes(include=['float64']).columns

    X_train_pca = train_data[float_columns].copy().fillna(0)
    X_val_pca = val_data[float_columns].copy().fillna(0)

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


def scale_and_transform_data(train_data, val_data, scaler_type, pca_components):
    """Process data by scaling and possibly transforming with PCA."""
    train_data, val_data = scale_data(train_data, val_data, scaler_type)
    if pca_components:
        train_data, val_data = transform_data(train_data, val_data, pca_components)

    return train_data, val_data


@app.command()
def pipeline(crop_start_s: float = typer.Option(get_env_variable('START_CROP_SECONDS'),
                                                help="Seconds to crop from the start of each signal"),
             crop_end_s: float = typer.Option(get_env_variable('END_CROP_SECONDS'),
                                              help="Seconds to crop from the end of each signal"),
             resample_rate_hz: float = typer.Option(get_env_variable('RESAMPLE_RATE_HZ'),
                                                    help="Resample rate in Hz"),
             segment_size_s: float = typer.Option(get_env_variable('SEGMENT_SIZE_SECONDS'),
                                                  help="Segment size in seconds"),
             overlap_s: float = typer.Option(get_env_variable('OVERLAP_SECONDS'),
                                             help="Overlap between segments in seconds"),

             moving_window_size_s: float = typer.Option(None, help="Moving window size in seconds"),
             fft: bool = typer.Option(False, help="Calculate FFT features"),
             pearson_corr: bool = typer.Option(False, help="Calculate Pearson correlation features"),

             validation_size: float = typer.Option(0.2, help="Validation set size in proportion to the data set"),
             k_folds: int = typer.Option(None, help="Number of splits for cross-validation"),
             scaler_type: str = typer.Option('standard', help="Type of scaler to scale numerical features"),
             pca_components: int = typer.Option(None, help="Number of PCA components to use"),

             measurement_file_path: str = typer.Option(None,
                                                       help="Path to a single measurement file to process. If "
                                                            "provided, this will override the DB query."),
             use_db_cache: bool = typer.Option(False,
                                               help="Use cached DB data if available to avoid querying InfluxDB"),
             clear_output: bool = typer.Option(True,
                                               help="Clear the output directory before saving new splits/folds data"),
             multi_processing: bool = typer.Option(False, help="Enable multi-processing"),
             n_jobs: int = typer.Option(-1, help="Number of jobs to run in parallel for multi-processing"),
             verbose: bool = typer.Option(False, help="Enable verbose logging")):
    setup_logging(verbose)

    output_dir = os.path.join(get_env_variable('DATA_DIR'), 'partitions')

    logger.info("Configuration Parameters:")
    logger.info("=== Timing and Sampling ===")
    logger.info(f"Crop Start [s]: {crop_start_s}")
    logger.info(f"Crop End [s]: {crop_end_s}")
    logger.info(f"Resample Rate [Hz]: {resample_rate_hz}")
    logger.info(f"Segment Size [s]: {segment_size_s}")
    logger.info(f"Overlap [s]: {overlap_s}\n")

    logger.info("=== Feature Extraction Settings ===")
    logger.info(f"Extract FFT Features: {'Yes' if fft else 'No'}")
    logger.info(f"Extract Pearson Correlation: {'Yes' if pearson_corr else 'No'}")
    logger.info(f"Moving Window Features - Window Size [s]: {moving_window_size_s}\n")

    logger.info("=== Model Validation and Data Splitting ===")
    logger.info(f"Validation Set Size: {validation_size}")
    logger.info(f"Number of Folds for CV: {k_folds if k_folds else 'No Cross-Validation'}\n")

    logger.info("=== Data Processing Options ===")
    logger.info(f"Scaler Type: {scaler_type}")
    logger.info(f"PCA Components: {pca_components if pca_components else 'None'}\n")

    logger.info("=== System Configuration ===")
    if measurement_file_path:
        logger.info(f"Measurement File Path: {measurement_file_path}")
    else:
        logger.info(f"Use DB Cache: {'Enabled' if use_db_cache else 'Disabled'}")
    logger.info(f"Clear Output: {'Yes' if clear_output else 'No'}")
    logger.info(f"Output Path: {output_dir}")
    logger.info(f"Multi-processing: {'Enabled' if multi_processing else 'Disabled'}")
    if multi_processing:
        logger.info(f"Number of Jobs: {n_jobs}")
    logger.info(f"Verbose Logging: {'Yes' if verbose else 'No'}")
    logger.info("====================================\n")

    if pca_components:
        raise ValueError("PCA components not implemented yet.")

    if not multi_processing:
        logger.warning("Using single-threaded processing. This may take a while for more features.")

    if measurement_file_path:
        logger.info("Loading single measurement file...")
        df = load_single_file_data(measurement_file_path)
    else:
        logger.info("Querying data...")
        df = query_raw_data(use_db_cache)

    logger.info("Preprocessing data...")
    segments_df = prepare_time_series_segments(df,
                                               crop_start_s,
                                               crop_end_s,
                                               resample_rate_hz,
                                               segment_size_s,
                                               overlap_s)

    logger.info("Extracting features...")
    segments_df = extract_features(segments_df,
                                   multi_processing,
                                   n_jobs,
                                   moving_window_size_s,
                                   fft,
                                   pearson_corr)

    if measurement_file_path:
        logger.info("Done.")
        return segments_df

    if clear_output and os.path.exists(output_dir):
        logger.info(f"Clearing output directory: {output_dir}")
        shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    logger.info("Splitting and processing data...")
    if k_folds:
        paths = get_partition_paths(k_folds=k_folds)
        folds = split_data(segments_df, validation_size, k_folds)

        for i, (train_data, val_data) in enumerate(folds):
            train_data, val_data = scale_and_transform_data(train_data, val_data, scaler_type, pca_components)
            save_partitions(train_data, val_data, paths[i])
    else:
        train_data, val_data = split_data(segments_df, validation_size)
        train_data, val_data = scale_and_transform_data(train_data, val_data, scaler_type, pca_components)

        paths = get_partition_paths(k_folds=k_folds)
        save_partitions(train_data, val_data, paths)

    logger.info("Data preparation completed.")


if __name__ == "__main__":
    app()
