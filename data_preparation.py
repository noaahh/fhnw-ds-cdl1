#!/usr/bin/env python3
import logging
import os
import shutil

import pandas as pd
import typer
from dotenv import load_dotenv
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from tqdm import tqdm

from src.data.db import InfluxDBWrapper
from src.data.partition_helper import get_partition_paths
from src.extraction.fft import extract_fft_features
from src.extraction.moving_average import calculate_moving_average, calc_window_size
from src.helper import get_env_variable

app = typer.Typer()
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool):
    """Set up logging configuration based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    logger.disabled = not verbose


load_dotenv()


def query_segments(use_cache: bool) -> pd.DataFrame:
    """Query segment data from InfluxDB or load from cache."""
    cache_dir = 'cache'
    cache_path = os.path.join(cache_dir, 'segments.parquet')
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
              |> filter(fn: (r) => r._measurement == "segment")
              |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        data_frames = query_api.query_data_frame(query, org=client.org)
        df = pd.concat(data_frames) if isinstance(data_frames, list) else data_frames

    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(cache_path)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the DataFrame by removing unwanted columns and handling missing values."""
    initial_length = len(df)
    df.dropna(inplace=True)
    dropped_rows = initial_length - len(df)
    logger.info(f"Dropped {dropped_rows} rows with missing values.")

    # Creating a unique ID combining segment_id and file_hash
    df['id'] = df['segment_id'].astype(str) + "_" + df['file_hash']
    unwanted_columns = ['result', 'table', '_start', '_stop', '_measurement',
                        'device_id', 'segment_id', 'measurement_group', 'platform',
                        'device_name', 'file_hash', 'app_version', 'recording_time']
    df.drop(columns=unwanted_columns, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def extract_features_for_segment(df, segment_id, columns, moving_window_size, use_fft, use_pears_corr):
    """Extract features for a specific segment with optimized pandas operations."""
    segment = df[df['id'] == segment_id]
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
    segment_ids = df['id'].unique()

    len_before = len(df)

    if multi_processing:
        results = Parallel(n_jobs=n_jobs)(delayed(extract_features_for_segment)(
            df, segment_id, float_columns, moving_window_size, use_fft, use_pears_corr)
                                          for segment_id in tqdm(segment_ids, desc="Extracting features"))
    else:
        results = [
            extract_features_for_segment(df, segment_id, float_columns, moving_window_size, use_fft, use_pears_corr)
            for segment_id in tqdm(segment_ids, desc="Extracting features")]

    features_df = pd.concat(results, axis=0)
    df = pd.concat([df, features_df], axis=1)
    assert len(df) == len_before, f"Data length mismatch after feature extraction: {len(df)} vs {len_before}"
    return df


def split_data(df: pd.DataFrame, val_size: float, n_splits: int = None) -> tuple | list:
    """Split data into train and validation sets or perform K-fold split."""
    grouped = df.groupby('id').first().reset_index()
    assert grouped['id'].nunique() == len(grouped), "Duplicate IDs found in the dataset."

    if n_splits:
        folds = []
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=1337)
        for train_idx, val_idx in kf.split(grouped):
            train_ids = grouped.loc[train_idx, 'id']
            val_ids = grouped.loc[val_idx, 'id']
            train_mask = df['id'].isin(train_ids)
            val_mask = df['id'].isin(val_ids)
            folds.append((df[train_mask], df[val_mask]))

        return folds
    else:
        train_ids, val_ids = train_test_split(grouped['id'], test_size=val_size, stratify=grouped['label'])
        train_mask = df['id'].isin(train_ids)
        val_mask = df['id'].isin(val_ids)
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
def prep_data(validation_size: float = 0.2,
              n_splits: int = typer.Option(None),
              scaler_type: str = typer.Option('standard'),
              pca_components: int = typer.Option(None),
              moving_window_size_s: float = typer.Option(None),
              fft: bool = typer.Option(False),
              pearson_corr: bool = typer.Option(False),
              use_cache: bool = typer.Option(False),
              clear_output: bool = typer.Option(False),
              output_path: str = typer.Option('./data/splits'),
              multi_processing: bool = typer.Option(False),
              n_jobs: int = typer.Option(-1),
              verbose: bool = typer.Option(False)):
    setup_logging(verbose)
    logger.info("--- PARAMETERS ---")
    logger.info(f"Validation set size: {validation_size}")
    logger.info(f"Number of splits: {n_splits if n_splits else 'No Cross-Validation'}")
    logger.info(f"Scaler type: {scaler_type}")
    logger.info(f"Extract FFT features: {fft}")
    logger.info(f"Extract Pearson correlation: {pearson_corr}")
    logger.info(f"PCA components: {pca_components}")
    logger.info(f"Moving window size: {moving_window_size_s}s")
    logger.info(f"Use cache: {use_cache}")
    logger.info(f"Clear output: {clear_output}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Multi-processing: {multi_processing}")
    logger.info(f"Number of jobs: {n_jobs}")
    logger.info(f"Verbose: {verbose}")
    logger.info("--------------------\n")

    if pca_components:
        raise ValueError("PCA components not implemented yet.")

    if not multi_processing:
        logger.warning("Using single-threaded processing. This may take a while for more features.")

    logger.info("Querying data...")
    df = query_segments(use_cache)

    logger.info("Preprocessing data...")
    df = preprocess_data(df)

    logger.info("Extracting features...")
    df = extract_features(df, multi_processing, n_jobs, moving_window_size_s, fft, pearson_corr)

    if clear_output and os.path.exists(output_path):
        logger.info(f"Clearing output directory: {output_path}")
        shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)

    logger.info("Splitting and processing data...")
    if n_splits:
        paths = get_partition_paths(output_path, n_splits)
        folds = split_data(df, validation_size, n_splits)

        for i, (train_data, val_data) in enumerate(folds):
            train_data, val_data = scale_and_transform_data(train_data, val_data, scaler_type, pca_components)
            save_partitions(train_data, val_data, paths[i])
    else:
        train_data, val_data = split_data(df, validation_size)
        train_data, val_data = scale_and_transform_data(train_data, val_data, scaler_type, pca_components)

        paths = get_partition_paths(output_path)
        save_partitions(train_data, val_data, paths)

    logger.info("Data preparation completed.")


if __name__ == "__main__":
    app()
