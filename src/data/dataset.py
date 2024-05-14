import pandas as pd
import torch

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
