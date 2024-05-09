import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.partition_helper import get_partitioned_data

LABEL_MAPPING = {
    "walking": 0,
    "running": 1,
    "sitting": 2,
    "standing": 3,
    "climbing": 4
}

NUM_CLASSES = len(LABEL_MAPPING)
ONE_HOT_VECTORS = {
    label: torch.nn.functional.one_hot(
        torch.tensor(LABEL_MAPPING[label], dtype=torch.long),
        num_classes=NUM_CLASSES
    ) for label in LABEL_MAPPING
}


class SensorDatasetSKLearn:
    LABEL_COLUMN = 'label'

    def __init__(self, data_df):
        self.data_df = self._preprocess_data(data_df)
        self.labels = self.data_df.groupby('segment_id')[self.LABEL_COLUMN].first().values
        self.data_df = self.data_df.drop(columns=[self.LABEL_COLUMN])

    @staticmethod
    def _preprocess_data(data_df):
        if not pd.api.types.is_datetime64_any_dtype(data_df['_time']):
            data_df['_time'] = pd.to_datetime(data_df['_time'])
        data_df.sort_values(['segment_id', '_time'], inplace=True)
        data_df.drop(columns=['_time'], inplace=True)
        return data_df

    def get_data(self):
        nunique = self.data_df.groupby('segment_id').nunique()

        single_value_columns = nunique.columns[nunique.max() == 1].tolist()
        multi_value_columns = nunique.columns[nunique.max() > 1].tolist()

        single_value_df = self.data_df.groupby('segment_id')[single_value_columns].first().reset_index(drop=True)

        if multi_value_columns:
            self.data_df['time_idx'] = self.data_df.groupby('segment_id').cumcount()
            multi_value_df = (self.data_df.set_index(['segment_id', 'time_idx'])[multi_value_columns]
                              .unstack(fill_value=0))
            multi_value_df.columns = [f"{col[0]}_{col[1]}" for col in multi_value_df.columns]
            multi_value_df.reset_index(drop=True, inplace=True)
            processed_df = pd.concat([single_value_df, multi_value_df], axis=1)
        else:
            processed_df = single_value_df

        return processed_df, self.labels


class SensorDatasetTorch(Dataset):
    LABEL_COLUMN = 'label'

    def __init__(self, data_df, transform=None):
        self.transform = transform
        self.data_df = self._preprocess_data(data_df)
        self.segments = [
            group.drop(columns=['_time', 'id'])
            for _, group in self.data_df.groupby('id')
        ]

        self.labels = self.data_df.groupby('id')['label'].first().values

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
        data_df.sort_values(['id', '_time'], inplace=True)
        return data_df

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        data_tensors = self._process_segment(segment)
        label_one_hot = ONE_HOT_VECTORS[self.labels[idx]]

        if self.transform:
            data_tensors = {k: self.transform(v) for k, v in data_tensors.items()}

        return data_tensors, label_one_hot

    def _process_segment(self, segment):
        data_tensors = {col: torch.tensor([segment[col].iloc[0]], dtype=torch.float32)
                        for col in self.unique_columns}
        data_tensors.update({col: torch.tensor(segment[col].values, dtype=torch.float32)
                             for col in self.sequence_columns})
        return data_tensors


def create_data_loader(data_df, batch_size, shuffle=True, num_workers=0, pin_memory=False):
    return DataLoader(SensorDatasetTorch(data_df),
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory)


class SensorDataModule(pl.LightningDataModule):
    def __init__(self, partition_paths, batch_size, num_workers=None, pin_memory=True):
        super().__init__()
        self.data_dict = None
        self.partition_paths = partition_paths

        self.batch_size = batch_size
        self.num_workers = os.cpu_count() if num_workers is None else num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        self.data_dict = get_partitioned_data(self.partition_paths)

    def train_dataloader(self):
        return create_data_loader(self.data_dict['train'],
                                  self.batch_size,
                                  num_workers=self.num_workers,
                                  pin_memory=self.pin_memory)

    def val_dataloader(self):
        return create_data_loader(self.data_dict['val'],
                                  self.batch_size,
                                  num_workers=self.num_workers,
                                  pin_memory=self.pin_memory)

    def test_dataloader(self):
        raise NotImplementedError("Test data loader not implemented.")
