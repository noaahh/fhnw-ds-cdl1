import logging
import os
import shutil

import pandas as pd
import typer
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from src.data.db import InfluxDBWrapper
from src.data.partition_helper import get_partition_paths

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

    influxdb_bucket = os.getenv('INFLUXDB_INIT_BUCKET', '')
    assert influxdb_bucket, "InfluxDB bucket not set in environment variables."

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
    """Preprocess the data by dropping unnecessary columns and handling duplicates."""
    initial_length = len(df)
    df.dropna(inplace=True)
    dropped_rows = initial_length - len(df)
    logger.info(f"Dropped {dropped_rows} rows with missing values.")

    df['id'] = (df['segment_id'].astype(str) +
                "_" + df['file_hash'])  # Combine segment_id and file_hash to create unique segment IDs across files

    df.drop(
        columns=['result', 'table', '_start', '_stop', '_measurement', 'device_id', 'segment_id', 'measurement_group',
                 'platform', 'device_name', 'file_hash', 'app_version', 'recording_time'],
        inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def split_data(df: pd.DataFrame, test_size: float, n_splits: int = None) -> tuple | list:
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
        train_ids, val_ids = train_test_split(grouped['id'], test_size=test_size, stratify=grouped['label'])
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


def standardize_data(X_train: pd.DataFrame,
                     X_val: pd.DataFrame,
                     scaler_type: str = 'standard',
                     pca_components: int = None):
    """Apply scaling and optional PCA to the training and validation data."""
    scaler = get_scaler(scaler_type)
    float_columns = X_train.select_dtypes(include=['float64']).columns
    X_train.loc[:, float_columns] = scaler.fit_transform(X_train[float_columns])
    X_val.loc[:, float_columns] = scaler.transform(X_val[float_columns])

    if pca_components:
        pca = PCA(n_components=pca_components)
        X_train_pca = pca.fit_transform(X_train[float_columns])
        X_val_pca = pca.transform(X_val[float_columns])

        pca_columns = [f'pca_{i}' for i in range(pca_components)]
        X_train = pd.concat([X_train, pd.DataFrame(X_train_pca, columns=pca_columns)], axis=1)
        X_val = pd.concat([X_val, pd.DataFrame(X_val_pca, columns=pca_columns)], axis=1)

    return X_train, X_val


def save_partitions(X_train: pd.DataFrame, X_val: pd.DataFrame, paths: dict) -> None:
    """Save the training and validation data to disk."""
    for path in paths.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)

    X_train.to_parquet(paths['train'])
    X_val.to_parquet(paths['validate'])

    X_train['label'].to_frame().to_parquet(paths['train_labels'])
    X_val['label'].to_frame().to_parquet(paths['validate_labels'])


def prep_data(test_size: float = 0.2,
              n_splits: int = typer.Option(None),
              scaler_type: str = typer.Option('standard'),
              pca_components: int = typer.Option(None),
              use_cache: bool = typer.Option(False),
              clear_output: bool = typer.Option(False),
              output_path: str = typer.Option('./data/splits'),
              verbose: bool = typer.Option(False)):
    logger.disabled = not verbose
    df = query_segments(use_cache)
    df = preprocess_data(df)

    if clear_output and os.path.exists(output_path):
        shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)

    if n_splits:
        paths = get_partition_paths(output_path, n_splits)
        folds = split_data(df, test_size, n_splits)
        for i, (X_train, X_val) in enumerate(folds):
            X_train, X_val = standardize_data(X_train, X_val, scaler_type, pca_components)
            save_partitions(X_train, X_val, paths[i])
    else:
        X_train, X_val = split_data(df, test_size)
        X_train, X_val = standardize_data(X_train, X_val, scaler_type, pca_components)
        paths = get_partition_paths(output_path)
        save_partitions(X_train, X_val, paths)


if __name__ == "__main__":
    typer.run(prep_data)
