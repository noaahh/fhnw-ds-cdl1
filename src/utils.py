import logging
import os

import pandas as pd
import typer
from dotenv import load_dotenv

load_dotenv()


def get_env_variable(variable_name):
    value = os.getenv(variable_name)
    if value is None:
        raise ValueError(f"Environment variable {variable_name} must be set")

    if value.isdigit():
        return int(value)

    return value


PARTITION_PATHS_KEYS = ['train', 'validate']


def get_partition_paths(root_partition_dir, k_folds=None):
    partition_paths = []
    test_path = os.path.join(root_partition_dir, 'test.parquet')

    if k_folds is not None:
        for i in range(k_folds):
            fold_dir = os.path.join(root_partition_dir, f'fold_{i}')

            partition_paths.append({
                'train': os.path.join(fold_dir, 'train.parquet'),
                'validate': os.path.join(fold_dir, 'val.parquet'),
                'test': test_path
            })
    else:
        partition_paths = {
            'train': os.path.join(root_partition_dir, 'train.parquet'),
            'validate': os.path.join(root_partition_dir, 'val.parquet'),
            'test': test_path
        }

    return partition_paths


def get_partitioned_data(partition_paths):
    if isinstance(partition_paths, dict):
        assert all([key in partition_paths for key in PARTITION_PATHS_KEYS]), "Missing keys in partition paths."
        return {key: pd.read_parquet(path) for key, path in partition_paths.items()}
    elif isinstance(partition_paths, list):
        folds = []
        for fold in partition_paths:
            assert all([key in fold for key in PARTITION_PATHS_KEYS]), "Missing keys in partition paths."
            folds.append({key: pd.read_parquet(path) for key, path in fold.items()})
        return folds
    else:
        raise ValueError("Invalid partition paths format.")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def validate_smoothing(value: str):
    if value.lower() not in ('butterworth', 'wavelet', 'true', 'false'):
        raise typer.BadParameter("Smoothing must be either 'butterworth', 'wavelet', 'true', or 'false'")
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False
    return value.lower()
