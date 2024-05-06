import hashlib
import logging
import os
import zipfile

import pandas as pd

from src.data.label_mapping import MeasurementGroup, extract_label_from_file_name
from src.helper import get_env_variable

logger = logging.getLogger(__name__)


class MeasurementFile:
    def __init__(self, zip_file_path):
        self.validate_file(zip_file_path)
        self.zip_file_path = zip_file_path

        self.data = self.init_data()
        self.file_hash = self.generate_file_hash()

    def __repr__(self):
        return (f"MeasurementFile(label={self.get_label()}, path={self.zip_file_path}, hash={self.file_hash}, "
                f"measurement_group={self.get_measurement_group()}")

    @staticmethod
    def validate_file(zip_file_path):
        if not os.path.exists(zip_file_path):
            raise ValueError(f"File {zip_file_path} does not exist")
        if not zipfile.is_zipfile(zip_file_path):
            raise ValueError(f"File {zip_file_path} is not a zip file")

    def get_label(self):
        return extract_label_from_file_name(self.zip_file_path)

    def get_measurement_group(self):
        if not self.zip_file_path.split("/")[-2].isdigit():
            return None

        return MeasurementGroup(int(self.zip_file_path.split("/")[-2]))

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
        if not self.data:
            raise ValueError("No data to get sensor data from")

        return {k: v for k, v in self.data.items() if k != 'metadata'}

    def init_data(self):
        data = {}
        with zipfile.ZipFile(self.zip_file_path, 'r') as z:
            csv_file_names = get_env_variable("CSV_FILE_NAMES").split(",")
            for csv_filename in csv_file_names:
                try:
                    with z.open(f"{csv_filename}.csv") as f:
                        data[csv_filename.lower()] = pd.read_csv(f)
                except KeyError:
                    logger.warning(f"Warning: {csv_filename}.csv not found in {self.zip_file_path}")

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
