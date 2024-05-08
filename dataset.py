import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LABEL_MAPPING = {"walking": 0, "running": 1, "sitting": 2, "standing": 3, "climbing": 4}
NUM_CLASSES = len(LABEL_MAPPING)


class SensorDataset(Dataset):
    def __init__(self, data_df, transform=None):
        self.transform = transform
        self.data_df = self._preprocess_data(data_df)
        self.segments = [group.drop(columns=['_time', 'id']) for _, group in self.data_df.groupby('id')]

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
        # data_tensor = torch.tensor(segment.drop(columns='label').values, dtype=torch.float32)
        numpy_array = segment.drop(columns='label').values.astype(np.float32)  # Convert DataFrame to NumPy array first
        data_tensor = torch.from_numpy(numpy_array)

        label = segment['label'].iloc[0]
        label_one_hot = torch.nn.functional.one_hot(torch.tensor(LABEL_MAPPING[label],
                                                                 dtype=torch.long),
                                                    num_classes=NUM_CLASSES)
        if self.transform:
            data_tensor = self.transform(data_tensor)

        return data_tensor, label_one_hot

    def get_feature_index(self, feature_name):
        return self.segments[0].drop(columns='label').columns.get_loc(feature_name)
