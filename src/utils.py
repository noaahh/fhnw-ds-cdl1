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


def get_partition_paths(root_partition_dir, k_folds=None):
    paths = {}
    if k_folds is not None:
        folds = []
        for fold in range(k_folds):
            folds.append({
                'base_dir': os.path.join(root_partition_dir, f"fold_{fold}"),
                'train': os.path.join(root_partition_dir, f"fold_{fold}", 'train.parquet'),
                'validate': os.path.join(root_partition_dir, f"fold_{fold}", 'validate.parquet')
            })
        paths['folds'] = folds
    else:
        paths['train'] = os.path.join(root_partition_dir, 'train.parquet')
        paths['validate'] = os.path.join(root_partition_dir, 'validate.parquet')

    paths['test'] = os.path.join(root_partition_dir, 'test.parquet')
    paths['train_all'] = os.path.join(root_partition_dir, 'train_all.parquet')
    return paths


def get_partitioned_data(partition_paths):
    data_structure = {}
    if 'folds' in partition_paths:
        data_structure['folds'] = []
        for fold in partition_paths['folds']:
            loaded_fold = {
                'base_dir': fold['base_dir'],
                'train': pd.read_parquet(fold['train']) if os.path.exists(fold['train']) else None,
                'validate': pd.read_parquet(fold['validate']) if os.path.exists(fold['validate']) else None
            }
            data_structure['folds'].append(loaded_fold)
    else:
        for key in ['train', 'validate']:
            data_structure[key] = pd.read_parquet(partition_paths[key]) if os.path.exists(partition_paths[key]) else None

    for key in ['test', 'train_all']:
        data_structure[key] = pd.read_parquet(partition_paths[key]) if os.path.exists(partition_paths[key]) else None

    return data_structure


def validate_smoothing(value: str):
    if value.lower() not in ('butterworth', 'wavelet', 'true', 'false'):
        raise typer.BadParameter("Smoothing must be either 'butterworth', 'wavelet', 'true', or 'false'")
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False
    return value.lower()
