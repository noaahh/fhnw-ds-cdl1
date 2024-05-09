import os

import pandas as pd

PARTITION_PATHS_KEYS = ['train', 'validate']


def get_partition_paths(root_output_dir, k_folds=None):
    partition_paths = []

    if k_folds is not None:
        for i in range(k_folds):
            fold_dir = os.path.join(root_output_dir, f'fold_{i}')

            partition_paths.append({
                'train': os.path.join(fold_dir, 'train.parquet'),
                'validate': os.path.join(fold_dir, 'val.parquet')
            })
    else:
        partition_paths = {
            'train': os.path.join(root_output_dir, 'X_train.parquet'),
            'validate': os.path.join(root_output_dir, 'val.parquet')
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
