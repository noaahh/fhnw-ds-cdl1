import glob
import hashlib
import logging
import os
import zipfile

import pandas as pd
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

CONFIG = {"data_folder": "./data",
          "resample_rate_hz": 50,
          "start_crop_seconds": 5,
          "end_crop_seconds": 5,
          "segment_size_seconds": 5,
          "overlap_seconds": 2,
          "csv_file_names": ["Accelerometer",
                             "AccelerometerUncalibrated",
                             "Gravity",
                             "Gyroscope",
                             "GyroscopeUncalibrated",
                             "Metadata"],
          "column_mappings": {'accelerometer': ['time', 'z', 'y', 'x'],
                              'gyroscope': ['time', 'z', 'y', 'x'],
                              'gravity': ['time', 'z', 'y', 'x'],
                              'accelerometeruncalibrated': ['time', 'z', 'y', 'x'],
                              'gyroscopeuncalibrated': ['time', 'z', 'y', 'x'],
                              'orientation': ['time', 'yaw', 'qx', 'qz', 'roll', 'qw', 'qy', 'pitch']}}


class MeasurementFile:
    def __init__(self, zip_file_path):
        self.validate_file(zip_file_path)
        self.zip_file_path = zip_file_path

        self.data = self.init_data()
        self.file_hash = self.generate_file_hash()

    def __repr__(self):
        return f"MeasurementFile(label={self.get_label()}, path={self.zip_file_path}, hash={self.file_hash})"

    @staticmethod
    def validate_file(zip_file_path):
        if not os.path.exists(zip_file_path):
            raise ValueError(f"File {zip_file_path} does not exist")
        if not zipfile.is_zipfile(zip_file_path):
            raise ValueError(f"File {zip_file_path} is not a zip file")

    def get_label(self):
        return os.path.basename(self.zip_file_path).rsplit('.', 1)[0].split('-', 1)[0]

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
            for csv_filename in CONFIG['csv_file_names']:
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

        unique_str = f"{self.zip_file_path}{metadata['device_name']}{metadata['platform']}{metadata['app_version']}{metadata['recording_time']}{metadata['device_id']}"
        return hashlib.sha256(unique_str.encode()).hexdigest()


class BaseProcessor:
    def __init__(self, measurement_file):
        self.measurement_file = measurement_file

    @staticmethod
    def preprocess(data):
        data = data.drop(columns=['seconds_elapsed'])
        data['time'] = pd.to_datetime(data['time'], unit='ns')
        data = data.set_index('time', drop=True)

        return data

    @staticmethod
    def crop(data):
        total_duration = (data.index.max() - data.index.min()).total_seconds()
        start_seconds, end_seconds = CONFIG['start_crop_seconds'], CONFIG['end_crop_seconds']
        if total_duration < start_seconds + end_seconds:
            raise ValueError("Signal is too short to crop")

        start_time = data.index.min() + pd.Timedelta(seconds=start_seconds)
        end_time = data.index.max() - pd.Timedelta(seconds=end_seconds)

        return data.loc[start_time:end_time]

    @staticmethod
    def resample(data):
        rate_hz = CONFIG['resample_rate_hz']
        rate = f"{int(1E6 / rate_hz)}us"
        data = data.resample(rate, origin="start").mean()  # TODO: check if mean is the best way to interpolate the data

        if data.isnull().values.any() or data.isna().values.any():
            logger.warning(f"Warning: NaNs found in resampled data for {self.measurement_file}")
            data = data.fillna(method='bfill')  # Backfill NaNs

        return data

    @staticmethod
    def segment(data):
        segment_length = pd.Timedelta(seconds=CONFIG['segment_size_seconds'])
        overlap_length = pd.Timedelta(seconds=CONFIG['overlap_seconds'])
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


def resample_sensor_data(sensor_data, resample_rate):
    sensor_data['time'] = pd.to_datetime(sensor_data['time'])
    sensor_data = sensor_data.set_index('time')
    sensor_data = sensor_data.resample(resample_rate).mean()
    sensor_data = sensor_data.reset_index()
    return sensor_data


def segment_sensor_data(sensor_data, segment_size, overlap):
    segments = []

    for i in range(0, len(sensor_data), segment_size - overlap):
        segment = sensor_data[i:i + segment_size]
        if len(segment) == segment_size:
            segments.append(segment)

    return segments


def get_processor(sensor_name, measurement_file):
    return BaseProcessor(measurement_file)


def filter_measurement_files(measurement_files):
    return [mf for mf in measurement_files if
            mf.get_metadata()['device_name'] != 'iPhone X']  # TODO: find a better way to filter the broken phone


def create_measurement_files(data_folder):
    zip_files = glob.glob(os.path.join(data_folder, "*.zip"))
    return [MeasurementFile(zip_path) for zip_path in zip_files]


def process_sensor_data(sensor_name, sensor_data, measurement_file):
    processor = get_processor(sensor_name, measurement_file)
    segments = processor.process(sensor_data)

    for i, segment in enumerate(segments):
        segment_id = f"{sensor_name}_{i}"
        write_segment(measurement_file, segment, sensor_name, segment_id)


def process_measurement_file(measurement_file):
    if is_file_processed(measurement_file.generate_file_hash()):
        logger.info(f"File {measurement_file} already processed. Skipping...")
        return

    for csv_file_name in CONFIG["csv_file_names"]:
        sensor_name = csv_file_name.lower()
        sensor_data = measurement_file.get_sensor_data().get(sensor_name)
        if sensor_data is not None:
            process_sensor_data(sensor_name, sensor_data, measurement_file)


def process_zip_files():
    measurement_files = create_measurement_files(CONFIG['data_folder'])
    filtered_files = filter_measurement_files(measurement_files)

    for measurement_file in tqdm(filtered_files, desc="Processing measurement files"):
        process_measurement_file(measurement_file)

    logger.info("All files processed")


def write_segment(measurement_file, segment_df, sensor_name, segment_id):
    points = []
    with InfluxDBWrapper() as client:
        write_api = client.write_api

        for index, row in segment_df.iterrows():
            point = Point(sensor_name).time(index, WritePrecision.NS)

            for field in CONFIG['column_mappings'].get(sensor_name, []):
                if field in segment_df.columns and field != 'time':
                    point = point.field(field, row.get(field, 0))

            point = point.tag("label", measurement_file.get_label())
            point = point.tag("segment_id", segment_id)
            point = point.tag("file_hash", measurement_file.generate_file_hash())

            for k, v in measurement_file.get_metadata().items():
                point = point.tag(k, v)

            points.append(point)

        write_api.write(os.getenv("INFLUXDB_INIT_BUCKET"), os.getenv("INFLUXDB_INIT_ORG"), points,
                        write_precision=WritePrecision.NS)


def is_file_processed(file_hash):
    query = (f'from(bucket: "{os.getenv("INFLUXDB_INIT_BUCKET")}") |> range(start: -900d) |> filter(fn: (r) => '
             f'r.file_hash == "{file_hash}") |> limit(n: 1)')

    with InfluxDBWrapper() as influx:
        query_api = influx.query_api
        result = query_api.query(query=query, org=os.getenv("INFLUXDB_INIT_ORG"))

        for table in result:
            for _ in table.records:
                return True
        return False


class InfluxDBWrapper:
    def __init__(self):
        load_dotenv()
        self.url = os.getenv("INFLUXDB_URL", "http://localhost:8086")
        self.token = os.getenv("INFLUXDB_TOKEN")
        self.org = os.getenv("INFLUXDB_ORG")
        self.bucket = os.getenv("INFLUXDB_BUCKET")
        self.username = os.getenv("INFLUXDB_ADMIN_USERNAME")
        self.password = os.getenv("INFLUXDB_ADMIN_PASSWORD")
        self.client = None
        self.write_api = None
        self.query_api = None

    def __enter__(self):
        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org, username=self.username,
                                     password=self.password)

        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()


if __name__ == "__main__":
    process_zip_files()
