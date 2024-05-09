import hashlib
import os
import zipfile
from pathlib import Path

import pandas as pd

from src.data.label_mapping import MeasurementGroup, extract_label_from_file_name
from src.helper import get_env_variable

FILE_SENSOR_COLUMN_MAPPINGS = {
    'accelerometer': ['time', 'z', 'y', 'x'],
    'gyroscope': ['time', 'z', 'y', 'x'],
    'gravity': ['time', 'z', 'y', 'x'],
    'orientation': ['time', 'yaw', 'qx', 'qz', 'roll', 'qw', 'qy', 'pitch']
}


class MeasurementFile:
    def __init__(self, zip_file_path):
        self.validate_file(zip_file_path)
        self.zip_file_path = zip_file_path

        self.data = self.init_data()
        self.file_hash = self.generate_file_hash()

    def __repr__(self):
        file_name = Path(self.zip_file_path).name
        return (f"MeasurementFile(label={self.get_label()}, path={file_name}, hash={self.file_hash}, "
                f"group={self.get_measurement_group()})")

    @staticmethod
    def validate_file(zip_file_path):
        if not os.path.exists(zip_file_path):
            raise ValueError(f"File {zip_file_path} does not exist")
        if not zipfile.is_zipfile(zip_file_path):
            raise ValueError(f"File {zip_file_path} is not a zip file")

    def get_label(self):
        return extract_label_from_file_name(self.zip_file_path)

    def get_measurement_group(self):
        path = Path(self.zip_file_path)

        try:
            parent_dir = path.parts[-2]
            if parent_dir.isdigit():
                return MeasurementGroup(int(parent_dir))
        except IndexError:
            pass

        return MeasurementGroup.NO_GROUP

    def get_metadata(self):
        if not self.data:
            raise ValueError("No data to get metadata from")

        metadata_df = self.data['metadata']
        return {'device_name': metadata_df['device name'].iloc[0],
                'platform': metadata_df['platform'].iloc[0],
                'app_version': str(metadata_df['appVersion'].iloc[0]),
                'recording_time': metadata_df['recording time'].iloc[0],
                'device_id': metadata_df['device id'].iloc[0]}

    def get_sensor_data(self):
        sensor_data_dict = {k: v for k, v in self.data.items() if k != 'metadata'}
        if not all(sensor in sensor_data_dict for sensor in FILE_SENSOR_COLUMN_MAPPINGS):
            raise ValueError("Missing sensor data")

        raw_data_length = {sensor: len(data) for sensor, data in sensor_data_dict.items() if
                           sensor in FILE_SENSOR_COLUMN_MAPPINGS}

        if len(set(raw_data_length.values())) != 1:
            raise ValueError(f"Data length mismatch: {raw_data_length}")

        merged_data = pd.DataFrame()
        for sensor_name, expected_cols in FILE_SENSOR_COLUMN_MAPPINGS.items():
            if sensor_name in sensor_data_dict:
                data = sensor_data_dict[sensor_name]
                if not all(col in data.columns for col in expected_cols):
                    raise ValueError(f"Columns mismatch for sensor {sensor_name}: {data.columns} vs {expected_cols}")
                data = data[expected_cols]
                data = data.set_index('time')
                data.columns = [f"{sensor_name}_{col}" for col in data.columns if col != 'time']
                merged_data = pd.concat([merged_data, data], axis=1) if not merged_data.empty else data

        merged_data.reset_index(inplace=True)

        if merged_data.empty:
            raise ValueError("No sensor data found")

        if list(raw_data_length.values())[0] != len(merged_data):
            raise ValueError("Data length mismatch: raw data vs merged data is not aligned on the time axis")

        return merged_data

    def init_data(self):
        data = {}
        with zipfile.ZipFile(self.zip_file_path, 'r') as z:
            csv_file_names = get_env_variable("CSV_FILE_NAMES").split(",")
            for csv_filename in csv_file_names:
                try:
                    with z.open(f"{csv_filename}.csv") as f:
                        data[csv_filename.lower()] = pd.read_csv(f)
                except KeyError:
                    pass

        return data

    def generate_file_hash(self):
        if not self.data:
            raise ValueError("No data to hash")

        metadata = self.get_metadata()

        unique_str = (f"{self.get_label()}"
                      f"{metadata['device_name']}"
                      f"{metadata['platform']}"
                      f"{metadata['app_version']}"
                      f"{metadata['recording_time']}"
                      f"{metadata['device_id']}"
                      f"{self.get_measurement_group()}")

        return hashlib.sha256(unique_str.encode()).hexdigest()
