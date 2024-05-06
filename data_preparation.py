import logging
import os

import pandas as pd
import typer
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.data.db import InfluxDBWrapper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()


def query_data():
    """Query segment data from InfluxDB and return as a DataFrame."""
    INFLUXDB_BUCKET = os.getenv('INFLUXDB_INIT_BUCKET')
    with InfluxDBWrapper() as client:
        query_api = client.query_api
        query = f'''
            from(bucket: "{INFLUXDB_BUCKET}")
              |> range(start: -1y)
              |> filter(fn: (r) => r._measurement == "segment")
              |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        data_frames = query_api.query_data_frame(query, org=client.org)
        return pd.concat(data_frames) if isinstance(data_frames, list) else data_frames


def preprocess_data(df):
    """Preprocess the data by dropping unnecessary columns and duplications."""
    return df.drop(columns=['result', 'table', '_start', '_stop', '_measurement'])


def split_data(df, test_size, n_splits=None):
    """Split data into train and validation sets or perform K-fold split."""
    df['id'] = df['segment_id'].astype(str) + "_" + df['file_hash']
    grouped = df.groupby('id').first().reset_index()

    if n_splits:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=1337)
        return [(train_index, test_index) for train_index, test_index in kf.split(grouped, grouped['label'])]
    else:
        train_ids, val_ids = train_test_split(grouped['id'], test_size=test_size, stratify=grouped['label'],
                                              shuffle=True, random_state=1337)
        train_mask = df['id'].isin(train_ids)
        val_mask = df['id'].isin(val_ids)
        return df.loc[train_mask], df.loc[val_mask]


def standardize_data(X_train, X_val, scaler_type='standard', pca_components=None):
    """Apply scaling and optional PCA."""
    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    float_columns = X_train.select_dtypes(include=['float64']).columns
    X_train[float_columns] = scaler.fit_transform(X_train[float_columns])
    X_val[float_columns] = scaler.transform(X_val[float_columns])

    if pca_components:
        pca = PCA(n_components=pca_components)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
    return X_train, X_val


def run_all(test_size: float = 0.2, n_splits: int = typer.Option(None),
            scaler_type: str = typer.Option('standard', help="Type of scaler: 'standard' or 'minmax'"),
            pca_components: int = typer.Option(None, help="Number of PCA components to retain"),
            verbose: bool = typer.Option(False, help="Enable verbose logging"),
            output_path: str = typer.Option('./data/splits', help="Output path for the data files")):
    """Execute the full data preparation pipeline with options for scaling and PCA."""
    logger.info("--- PARAMETERS ---")
    logger.info(f"Test size: {test_size}")
    logger.info(f"Number of splits: {n_splits}")
    logger.info(f"Scaler type: {scaler_type}")
    logger.info(f"PCA components: {pca_components if pca_components else 'No PCA'}")
    logger.info(f"Verbose: {verbose}")
    logger.info(f"Output path: {output_path}")
    logger.info("-------------------------\n")

    df = query_data()
    df = preprocess_data(df)

    if n_splits:
        folds = split_data(df, test_size, n_splits)
        for i, (train_idx, val_idx) in enumerate(folds):
            train_path = os.path.join(output_path, f'fold_{i}_train.parquet')
            validate_path = os.path.join(output_path, f'fold_{i}_validate.parquet')

            df.iloc[train_idx].drop(columns='label').to_parquet(train_path)
            df.iloc[val_idx].drop(columns='label').to_parquet(validate_path)

            df.iloc[train_idx]['label'].to_frame().to_parquet(
                os.path.join(output_path, f'fold_{i}_train_labels.parquet'))
            df.iloc[val_idx]['label'].to_frame().to_parquet(
                os.path.join(output_path, f'fold_{i}_validate_labels.parquet'))
        logger.info(f"Saved {n_splits} folds to disk at {output_path}.")
    else:
        X_train, X_val = split_data(df, test_size)
        X_train, X_val = standardize_data(X_train, X_val, scaler_type, pca_components)

        train_path = os.path.join(output_path, 'X_train.parquet')
        val_path = os.path.join(output_path, 'X_val.parquet')
        X_train.to_parquet(train_path)
        X_val.to_parquet(val_path)

        y_train_path = os.path.join(output_path, 'y_train.parquet')
        y_val_path = os.path.join(output_path, 'y_val.parquet')
        X_train['label'].to_frame().to_parquet(y_train_path)
        X_val['label'].to_frame().to_parquet(y_val_path)

    logger.info("Data preparation completed successfully.")


if __name__ == "__main__":
    typer.run(run_all)
