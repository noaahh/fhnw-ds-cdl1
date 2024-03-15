import glob
import hashlib
import logging
import os
import time
import zipfile

import pandas as pd
from dotenv import load_dotenv
from influxdb_client import Point, WritePrecision
from tqdm import tqdm

from src.db import InfluxDBWrapper, is_file_processed
from src.label_mapping import extract_label_from_file_name, MeasurementGroup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

PROCESSED_FILES_LOCAL = set()  # In-memory set to keep track of processed files

COLUMN_MAPPINGS = {'accelerometer': ['time', 'z', 'y', 'x'],
                   'gyroscope': ['time', 'z', 'y', 'x'],
                   'gravity': ['time', 'z', 'y', 'x'],
                   'accelerometeruncalibrated': ['time', 'z', 'y', 'x'],
                   'gyroscopeuncalibrated': ['time', 'z', 'y', 'x'],
                   'orientation': ['time', 'yaw', 'qx', 'qz', 'roll', 'qw', 'qy', 'pitch']}


def get_env_variable(variable_name):
    value = os.getenv(variable_name)
    if value is None:
        raise ValueError(f"Environment variable {variable_name} must be set")

    if value.isdigit():
        return int(value)

    return value


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


class BaseProcessor:
    def __init__(self, measurement_file):
        self.measurement_file = measurement_file

    def preprocess(self, data):
        data = data.drop(columns=['seconds_elapsed'])
        data['time'] = pd.to_datetime(data['time'], unit='ns')
        data = data.set_index('time', drop=True)

        return data

    def crop(self, data):
        total_duration = (data.index.max() - data.index.min()).total_seconds()

        start_seconds, end_seconds = get_env_variable("START_CROP_SECONDS"), get_env_variable("END_CROP_SECONDS")

        if total_duration < start_seconds + end_seconds:
            raise ValueError("Signal is too short to crop")

        start_time = data.index.min() + pd.Timedelta(seconds=start_seconds)
        end_time = data.index.max() - pd.Timedelta(seconds=end_seconds)

        return data.loc[start_time:end_time]

    def resample(self, data):
        rate_hz = get_env_variable("RESAMPLE_RATE_HZ")

        rate = f"{int(1E6 / rate_hz)}us"
        data = data.resample(rate, origin="start").mean()  # TODO: check if mean is the best way to interpolate the data

        if data.isnull().values.any() or data.isna().values.any():
            logger.warning(f"Warning: NaNs found in resampled data for {self.measurement_file}")
            data = data.bfill()  # Backfill NaNs

        return data

    def segment(self, data):
        segment_size_seconds = get_env_variable("SEGMENT_SIZE_SECONDS")
        overlap_seconds = get_env_variable("OVERLAP_SECONDS")

        segment_length = pd.Timedelta(seconds=segment_size_seconds)
        overlap_length = pd.Timedelta(seconds=overlap_seconds)
        segments = []

        start_time = data.index.min()
        end_time = data.index.max()

        while start_time + segment_length <= end_time:  # TODO: check ways to use overlap in a better way in case the we are running out of data at the end of the signal
            segment_end = start_time + segment_length
            segment = data.loc[start_time:segment_end]
            segments.append(segment)

            start_time = segment_end - overlap_length

        segment_lengths = set([len(segment) for segment in segments])
        if len(segment_lengths) != 1:
            raise ValueError("Segments are not of equal length")

        return segments

    def process(self, data):
        preprocessed_data = self.preprocess(data)
        cropped_data = self.crop(preprocessed_data)
        resampled_data = self.resample(cropped_data)
        segments = self.segment(resampled_data)

        if len(segments) == 0:
            raise ValueError("No segments found")

        return segments


def get_processor(sensor_name, measurement_file):
    return BaseProcessor(measurement_file)


def filter_measurement_files(measurement_files):
    return [mf for mf in measurement_files if
            mf.get_metadata()['device_name'] != 'iPhone X'
            and mf.get_label() is not None
            and mf.get_measurement_group() is not None
            and mf.generate_file_hash() not in PROCESSED_FILES_LOCAL
            and not is_file_processed(mf.generate_file_hash())]


def create_measurement_files(data_folder):
    zip_files = glob.glob(os.path.join(data_folder, "**", "*.zip"), recursive=True)
    return [MeasurementFile(zip_path) for zip_path in zip_files]


def process_sensor_data(sensor_name, sensor_data, measurement_file):
    processor = get_processor(sensor_name, measurement_file)
    segments = processor.process(sensor_data)

    for i, segment in enumerate(segments):
        segment_id = f"{sensor_name}_{i}"
        write_segment(measurement_file, segment, sensor_name, segment_id)


def process_measurement_file(measurement_file):
    csv_file_names = get_env_variable("CSV_FILE_NAMES").split(",")
    for csv_file_name in csv_file_names:
        sensor_name = csv_file_name.lower()
        sensor_data = measurement_file.get_sensor_data().get(sensor_name)
        if sensor_data is not None:
            process_sensor_data(sensor_name, sensor_data, measurement_file)


def mark_file_as_processed(zip_file_path):
    processed_file_path = f"{zip_file_path}.processed"
    os.rename(zip_file_path, processed_file_path)


def process_zip_files():
    data_folder = get_env_variable("DATA_FOLDER")
    measurement_files = create_measurement_files(data_folder)
    filtered_files = filter_measurement_files(measurement_files)

    for measurement_file in tqdm(filtered_files):
        process_measurement_file(measurement_file)
        mark_file_as_processed(measurement_file.zip_file_path)

        PROCESSED_FILES_LOCAL.add(measurement_file.generate_file_hash())
        logger.info(f"File {measurement_file} processed")

    logger.info("All files processed")


def write_segment(measurement_file, segment_df, sensor_name, segment_id):
    points = []
    with InfluxDBWrapper() as client:
        write_api = client.write_api

        for index, row in segment_df.iterrows():
            point = Point(sensor_name).time(index, WritePrecision.MS)

            for field in COLUMN_MAPPINGS.get(sensor_name, []):
                if field in segment_df.columns and field != 'time':
                    point = point.field(field, row.get(field, 0))

            point = point.tag("measurement_group", measurement_file.get_measurement_group().name)
            point = point.tag("label", measurement_file.get_label())
            point = point.tag("segment_id", segment_id)
            point = point.tag("file_hash", measurement_file.generate_file_hash())

            for k, v in measurement_file.get_metadata().items():
                point = point.tag(k, v)

            points.append(point)

        write_api.write(get_env_variable("INFLUXDB_INIT_BUCKET"),
                        get_env_variable("INFLUXDB_INIT_ORG"),
                        points,
                        write_precision=WritePrecision.MS)


if __name__ == "__main__":
    logger.info("--- DB CONNECTION ---")
    logger.info(f"INFLUXDB_URL: {get_env_variable('INFLUXDB_URL')}")
    logger.info(f"INFLUXDB_INIT_BUCKET: {get_env_variable('INFLUXDB_INIT_BUCKET')}")
    logger.info(f"INFLUXDB_INIT_ORG: {get_env_variable('INFLUXDB_INIT_ORG')}")
    logger.info("---------------------")

    interval = 60 * 1  # Interval in seconds
    next_run_time = time.time()

    try:
        while True:
            if time.time() >= next_run_time:
                logger.info("Processing zip files...")

                process_zip_files()

                next_run_time = time.time() + interval

                logger.info(f"Next run in {interval} seconds")

            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Exiting...")
