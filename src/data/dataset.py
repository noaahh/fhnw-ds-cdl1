import os

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

from src.data.label_mapping import LABEL_MAPPING, NUM_CLASSES, LABEL_COLUMN
from src.utils import get_partitioned_data, get_partition_paths

ONE_HOT_VECTORS = {
    label: torch.nn.functional.one_hot(
        torch.tensor(LABEL_MAPPING[label], dtype=torch.long),
        num_classes=NUM_CLASSES
    ) for label in LABEL_MAPPING
}


class SensorDatasetTorch(Dataset):
    def __init__(self, data_df, transform=None):
        self.transform = transform
        self.data_df = self._preprocess_data(data_df)

        self.segments = [
            group.drop(columns=['_time', 'segment_id', 'session_id'])
            for _, group in self.data_df.groupby('segment_id')
        ]

        self.labels = self.data_df.groupby('segment_id')['label'].first().values

        self.unique_columns = []
        self.sequence_columns = []
        self._analyze_columns(self.segments[0])

    def _analyze_columns(self, segment):
        for column in segment.columns:
            if column != LABEL_COLUMN:
                values = segment[column].values
                if len(np.unique(values)) == 1:
                    self.unique_columns.append(column)
                else:
                    self.sequence_columns.append(column)

    @staticmethod
    def _preprocess_data(data_df):
        if not pd.api.types.is_datetime64_any_dtype(data_df['_time']):
            data_df['_time'] = pd.to_datetime(data_df['_time'])
        data_df.sort_values(['segment_id', '_time'], inplace=True)
        return data_df

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        numpy_array = segment.drop(columns='label').values.astype(np.float32)
        data_tensor = torch.from_numpy(numpy_array)
        label_one_hot = ONE_HOT_VECTORS[self.labels[idx]]

        if self.transform:
            data_tensor = self.transform(data_tensor)

        return data_tensor, label_one_hot

    def _process_segment(self, segment):
        data_tensors = {col: torch.tensor([segment[col].iloc[0]], dtype=torch.float32)
                        for col in self.unique_columns}
        data_tensors.update({col: torch.tensor(segment[col].values, dtype=torch.float32)
                             for col in self.sequence_columns})
        return data_tensors


def create_data_loader(data_df, batch_size, shuffle=True, num_workers=0, pin_memory=False, persistent_workers=True):
    return DataLoader(SensorDatasetTorch(data_df),
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      persistent_workers=persistent_workers)


class SensorDataModule(LightningDataModule):
    def __init__(self, batch_size, partitioned_data_dir, k_folds=None,
                 num_workers=None, pin_memory=True, use_persistent_workers=True):
        super().__init__()
        self.current_fold = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.k_folds = k_folds
        self.partition_paths = get_partition_paths(partitioned_data_dir, k_folds)
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self.pin_memory = pin_memory
        self.use_persistent_workers = use_persistent_workers

    def setup(self, stage=None):
        self.train_data = None
        self.val_data = None
        self.test_data = None

        data_list = get_partitioned_data(self.partition_paths)
        if stage == 'refit':
            self.train_data = data_list['train_all']
            self.test_data = data_list['test']
            return

        if stage == 'test':
            self.test_data = data_list['test'] if self.k_folds is None else data_list[0]['test']
            return

        if self.current_fold is None and self.k_folds:
            raise ValueError("Current fold is not set for cross-validation. Please set the fold before setup.")

        partition = data_list if self.k_folds is None \
            else data_list["folds"][self.current_fold]

        self.train_data = partition.get('train')
        self.val_data = partition.get('validate')

    def set_current_fold(self, fold):
        if self.k_folds is None:
            raise ValueError("Cross-validation is not enabled. Set k_folds to enable it.")
        if fold < 0 or fold >= self.k_folds:
            raise ValueError(f"Invalid fold index {fold}. Must be in range [0, {self.k_folds}).")
        self.current_fold = fold

    def _create_dataloader(self, data, shuffle, persistent_workers=False):
        if data is None:
            raise ValueError("Data is not loaded. Call setup before creating a dataloader.")
        return DataLoader(SensorDatasetTorch(data),
                          batch_size=self.batch_size,
                          shuffle=shuffle,
                          num_workers=self.num_workers,
                          persistent_workers=persistent_workers,
                          pin_memory=self.pin_memory)

    def train_dataloader(self):
        return self._create_dataloader(self.train_data, shuffle=True)

    def val_dataloader(self):
        return self._create_dataloader(self.val_data, shuffle=False, persistent_workers=self.use_persistent_workers)

    def test_dataloader(self):
        return self._create_dataloader(self.test_data, shuffle=False, persistent_workers=self.use_persistent_workers)