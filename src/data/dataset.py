import os

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset

from src.utils import get_partitioned_data, get_partition_paths

LABEL_MAPPING = {
    "walking": 0,
    "running": 1,
    "sitting": 2,
    "standing": 3,
    "climbing": 4
}

LABEL_COLUMN = 'label'

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
        data_df.drop(columns=['_time', 'session_id'], inplace=True)
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


def create_data_loader(data_df, batch_size, shuffle=True, num_workers=-1, pin_memory=False, persistent_workers=True):
    return DataLoader(SensorDatasetTorch(data_df),
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      persistent_workers=persistent_workers)


class SensorDataModule(LightningDataModule):
    def __init__(self, batch_size, partitioned_data_dir, k_folds, num_workers=os.cpu_count(), pin_memory=True):
        super().__init__()
        self.current_fold = None
        self.train_data = None
        self.val_data = None
        self.k_folds = None  # Currently not supported

        self.partition_paths = get_partition_paths(partitioned_data_dir)
        self.batch_size = batch_size
        self.num_workers = os.cpu_count() if num_workers is None else num_workers
        self.pin_memory = pin_memory

    @property
    def num_classes(self):
        return NUM_CLASSES

    def setup(self, stage=None):
        partitioned_data_list = get_partitioned_data(self.partition_paths)

        if self.k_folds:
            # Cross-validation setup
            if self.current_fold is None:
                raise ValueError("Current fold is not set. Please set the fold before calling setup.")

            if isinstance(partitioned_data_list, list):
                partition = partitioned_data_list[self.current_fold]
            else:
                raise ValueError("Partitioned data is not in expected format (list or dict).")

        else:
            # Single partition setup (no cross-validation)
            if isinstance(partitioned_data_list, dict):
                partition = partitioned_data_list
            else:
                raise ValueError("Partitioned data is not in expected format or contains multiple partitions.")

        self.train_data = partition['train']
        self.val_data = partition['validate']

    def set_current_fold(self, fold):
        if self.k_folds is None:
            raise ValueError("Cross-validation is not enabled. Set k_folds to enable cross-validation.")

        if fold < 0 or fold >= self.k_folds:
            raise ValueError(f"Invalid fold index {fold}. Must be in range [0, {self.k_folds}).")

        self.current_fold = fold

    def train_dataloader(self):
        if self.train_data is None:
            raise ValueError("Train data is not loaded. Call setup before accessing the dataloader.")

        return create_data_loader(self.train_data,
                                  self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  pin_memory=self.pin_memory)

    def val_dataloader(self):
        if self.val_data is None:
            raise ValueError("Validation data is not loaded. Call setup before accessing the dataloader.")

        return create_data_loader(self.val_data,
                                  self.batch_size,
                                  shuffle=False,
                                  num_workers=self.num_workers,
                                  pin_memory=self.pin_memory)

    def test_dataloader(self):
        raise NotImplementedError("Test data loader not implemented.")


### CANCER STARTS HERE

import math
from typing import Any, Tuple
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from pl_bolts.utils import _SKLEARN_AVAILABLE
from pl_bolts.utils.stability import under_review
from pl_bolts.utils.warnings import warn_missing_pkg

if _SKLEARN_AVAILABLE:
    from sklearn.utils import shuffle as sk_shuffle
else:  # pragma: no cover
    warn_missing_pkg("sklearn", pypi_name="scikit-learn")


@under_review()
class SklearnDataset(Dataset):
    """Mapping between numpy (or sklearn) datasets to PyTorch datasets.

    Args:
        X: Numpy ndarray
        y: Numpy ndarray
        x_transform: Any transform that works with Numpy arrays
        y_transform: Any transform that works with Numpy arrays

    Example:
        >>> from sklearn.datasets import load_diabetes
        >>> from pl_bolts.datamodules import SklearnDataset
        ...
        >>> X, y = load_diabetes(return_X_y=True)
        >>> dataset = SklearnDataset(X, y)
        >>> len(dataset)
        442
    """

    def __init__(
            self,
            X: np.ndarray,  # noqa: N803
            y: np.ndarray,
            x_transform: Any = None,
            y_transform: Any = None,
    ) -> None:
        super().__init__()
        self.data = X
        self.labels = y
        self.data_transform = x_transform
        self.labels_transform = y_transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        x = self.data[idx].astype(np.float32)
        y = self.labels[idx]

        # Do not convert integer to float for classification data
        if not ((y.dtype == np.int32) or (y.dtype == np.int64)):
            y = y.astype(np.float32)

        if self.data_transform:
            x = self.data_transform(x)

        if self.labels_transform:
            y = self.labels_transform(y)

        return x, y

class SklearnDataModule(LightningDataModule):
    name = "sklearn"

    def __init__(
            self,
            X,  # noqa: N803
            y,
            x_val=None,
            y_val=None,
            x_test=None,
            y_test=None,
            val_split=0.2,
            test_split=0.1,
            num_workers=0,
            random_state=1234,
            shuffle=True,
            batch_size: int = 16,
            pin_memory=True,
            drop_last=False,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        # shuffle x and y
        if shuffle and _SKLEARN_AVAILABLE:
            X, y = sk_shuffle(X, y, random_state=random_state)  # noqa: N806
        elif shuffle and not _SKLEARN_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                "You want to use shuffle function from `scikit-learn` which is not installed yet."
            )

        val_split = 0 if x_val is not None or y_val is not None else val_split
        test_split = 0 if x_test is not None or y_test is not None else test_split

        hold_out_split = val_split + test_split
        if hold_out_split > 0:
            val_split = val_split / hold_out_split
            hold_out_size = math.floor(len(X) * hold_out_split)
            x_holdout, y_holdout = X[:hold_out_size], y[:hold_out_size]
            test_i_start = int(val_split * hold_out_size)
            x_val_hold_out, y_val_holdout = x_holdout[:test_i_start], y_holdout[:test_i_start]
            x_test_hold_out, y_test_holdout = x_holdout[test_i_start:], y_holdout[test_i_start:]
            X, y = X[hold_out_size:], y[hold_out_size:]  # noqa: N806

        # if don't have x_val and y_val create split from X
        if x_val is None and y_val is None and val_split > 0:
            x_val, y_val = x_val_hold_out, y_val_holdout

        # if don't have x_test, y_test create split from X
        if x_test is None and y_test is None and test_split > 0:
            x_test, y_test = x_test_hold_out, y_test_holdout

        self._init_datasets(X, y, x_val, y_val, x_test, y_test)

    def _init_datasets(
            self,
            x: np.ndarray,
            y: np.ndarray,
            x_val: np.ndarray,
            y_val: np.ndarray,
            x_test: np.ndarray,
            y_test: np.ndarray,
    ) -> None:
        self.train_dataset = SklearnDataset(x, y)
        self.val_dataset = SklearnDataset(x_val, y_val)
        self.test_dataset = SklearnDataset(x_test, y_test)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

class SensorDataModuleBolt(LightningDataModule):
    def __init__(self, batch_size, partitioned_data_dir, k_folds=None, num_workers=os.cpu_count(), pin_memory=True):
        super().__init__()
        self.batch_size = batch_size
        self.partitioned_data_dir = partitioned_data_dir
        self.k_folds = k_folds
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.current_fold = None
        self.train_data = None
        self.val_data = None

    def setup(self, stage=None):
        partitioned_data_list = get_partitioned_data(get_partition_paths(self.partitioned_data_dir))

        if self.k_folds:
            # Cross-validation setup
            if self.current_fold is None:
                raise ValueError("Current fold is not set. Please set the fold before calling setup.")
            partition = partitioned_data_list[self.current_fold]
        else:
            # Single partition setup (no cross-validation)
            partition = partitioned_data_list

        train_df = partition['train']
        val_df = partition['validate']

        train_dataset = SensorDatasetSKLearn(train_df)
        val_dataset = SensorDatasetSKLearn(val_df)

        X_train, y_train = train_dataset.get_data()
        X_val, y_val = val_dataset.get_data()

        self.train_data_module = SklearnDataModule(X_train, y_train, batch_size=self.batch_size)
        self.val_data_module = SklearnDataModule(X_val, y_val, batch_size=self.batch_size)

    def train_dataloader(self):
        return self.train_data_module.train_dataloader()

    def val_dataloader(self):
        return self.val_data_module.val_dataloader()

    def test_dataloader(self):
        raise NotImplementedError("Test data loader not implemented.")

