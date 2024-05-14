import os

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

from src.data.dataset import ONE_HOT_VECTORS, NUM_CLASSES
from src.utils import get_partition_paths, get_partitioned_data


class SensorDatasetTorch(Dataset):
    LABEL_COLUMN = 'label'

    def __init__(self, data_df, transform=None):
        self.transform = transform
        self.data_df = self._preprocess_data(data_df)
        self.segments = [
            group.drop(columns=['_time', 'segment_id'])
            for _, group in self.data_df.groupby('segment_id')
        ]

        self.labels = self.data_df.groupby('segment_id')['label'].first().values

        self.unique_columns = []
        self.sequence_columns = []
        self._analyze_columns(self.segments[0])

    def _analyze_columns(self, segment):
        for column in segment.columns:
            if column != self.LABEL_COLUMN:
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


def create_data_loader(data_df, batch_size, shuffle=True, num_workers=-1, pin_memory=False, persistent_workers=True):
    return DataLoader(SensorDatasetTorch(data_df),
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      persistent_workers=persistent_workers)


class SensorDataModule(LightningDataModule):
    def __init__(self, batch_size, num_workers=None, pin_memory=True, partition_paths=None):
        super().__init__()
        self.data_dict = None

        self.partition_paths = get_partition_paths() if partition_paths is None else partition_paths
        self.batch_size = batch_size
        self.num_workers = os.cpu_count() if num_workers is None else num_workers
        self.pin_memory = pin_memory

    @property
    def num_classes(self):
        return NUM_CLASSES

    def setup(self, stage=None):
        partitioned_data_list = get_partitioned_data(self.partition_paths)
        if isinstance(partitioned_data_list, list):
            self.data_dict = {idx: {
                "train": create_data_loader(partition['train'], self.batch_size, num_workers=self.num_workers,
                                            pin_memory=self.pin_memory),
                "validate": create_data_loader(partition['validate'], self.batch_size, shuffle=False,
                                               num_workers=self.num_workers, pin_memory=self.pin_memory)
            } for idx, partition in enumerate(partitioned_data_list)}

        elif isinstance(partitioned_data_list, dict):
            self.data_dict = {0: partitioned_data_list}

    def train_dataloader(self):
        return create_data_loader(self.data_dict['train'],
                                  self.batch_size,
                                  num_workers=self.num_workers,
                                  pin_memory=self.pin_memory)

    def val_dataloader(self):
        return create_data_loader(self.data_dict['validate'],
                                  self.batch_size,
                                  num_workers=self.num_workers,
                                  pin_memory=self.pin_memory)

    def test_dataloader(self):
        raise NotImplementedError("Test data loader not implemented.")
