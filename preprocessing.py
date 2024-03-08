import glob
import hashlib
import os
import zipfile

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

load_dotenv()

data_folder = "./data"

MEASUREMENT_CSV_FILE_NAMES = ["Accelerometer",
                              "AccelerometerUncalibrated",
                              "Gravity",
                              "Gyroscope",
                              "GyroscopeUncalibrated",
                              "Metadata"]

COLUMN_MAPPINGS = {
    'accelerometer': ['time', 'z', 'y', 'x'],
    'gyroscope': ['time', 'z', 'y', 'x'],
    'gravity': ['time', 'z', 'y', 'x'],
    'accelerometeruncalibrated': ['time', 'z', 'y', 'x'],
    'gyroscopeuncalibrated': ['time', 'z', 'y', 'x'],
    'orientation': ['time', 'yaw', 'qx', 'qz', 'roll', 'qw', 'qy', 'pitch']
}


class MeasurementFile:
    def __init__(self, zip_file_path):
        self.file_hash = None
        self.data = None

        if not os.path.exists(zip_file_path):
            raise ValueError(f"File {zip_file_path} does not exist")

        if not zipfile.is_zipfile(zip_file_path):
            raise ValueError(f"File {zip_file_path} is not a zip file")

        self.zip_file_path = zip_file_path
        self.init_data()

    def get_label(self):
        return os.path.basename(self.zip_file_path).rsplit('.', 1)[0].split('-', 1)[0]

    def get_metadata(self):
        if not self.data:
            raise ValueError("No data to get metadata from")

        metadata_df = self.data['metadata']
        return {
            'device_name': metadata_df['device name'].iloc[0],
            'platform': metadata_df['platform'].iloc[0],
            'app_version': str(metadata_df['appVersion'].iloc[0]),
            'recording_time': metadata_df['recording time'].iloc[0],
            'device_id': metadata_df['device id'].iloc[0]
        }

    def get_sensor_data(self):
        if not self.data:
            raise ValueError("No data to get sensor data from")

        return {k: v for k, v in self.data.items() if k != 'metadata'}

    def init_data(self):
        self.data = {}
        with zipfile.ZipFile(self.zip_file_path, 'r') as z:
            for csv_filename in MEASUREMENT_CSV_FILE_NAMES:
                with z.open(f"{csv_filename}.csv") as f:
                    self.data[csv_filename.lower()] = pd.read_csv(f)

        self.file_hash = self.generate_file_hash()

    def generate_file_hash(self):
        if not self.data:
            raise ValueError("No data to hash")

        metadata = self.get_metadata()

        unique_str = f"{self.zip_file_path}{metadata['device_name']}{metadata['platform']}{metadata['app_version']}{metadata['recording_time']}{metadata['device_id']}"
        return hashlib.sha256(unique_str.encode()).hexdigest()


class BaseProcessor:
    def __init__(self, measurement_file, sensor_name):
        self.measurement_file = measurement_file

    def preprocess(self, data):
        if 'seconds_elapsed' in data.columns:
            data = data.drop(columns=['seconds_elapsed'])

        data['time'] = pd.to_datetime(data['time'], unit='ns')
        data = data.set_index('time', drop=True)

        return data

    def crop(self, data, start_seconds=5, end_seconds=5):
        total_duration = (data.index.max() - data.index.min()).total_seconds()
        if total_duration < start_seconds + end_seconds:
            raise ValueError("Signal is too short to crop")

        start_time = data.index.min() + pd.Timedelta(seconds=start_seconds)
        end_time = data.index.max() - pd.Timedelta(seconds=end_seconds)

        return data.loc[start_time:end_time]

    def resample(self, data, rate_hz=50, interpolation_method='linear'):
        rate = f"{int(1E6 / rate_hz)}us"
        data = (data
                .resample(rate, origin="start")
                .mean()) # TODO: check if mean is the best way to interpolate the data

        if data.isnull().values.any() or data.isna().values.any():
            print(f"NaNs present in resampled data for {self.measurement_file.get_label()}")
            data = data.fillna(method='bfill')  # Backfill NaNs

        return data

    def segment(self, data, segment_size_seconds, overlap_size_seconds):
        segment_length = pd.Timedelta(seconds=segment_size_seconds)
        overlap_length = pd.Timedelta(seconds=overlap_size_seconds)
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
            print(segment_lengths)
            raise ValueError("Segments are not of equal length")

        return segments

    def process(self, data, rate_hz=50, segment_size_seconds=5, overlap_seconds=1):
        preprocessed_data = self.preprocess(data)
        cropped_data = self.crop(preprocessed_data)
        resampled_data = self.resample(cropped_data, rate_hz)
        segments = self.segment(resampled_data, segment_size_seconds, overlap_seconds)

        return segments


def create_measurement_files(data_folder):
    zip_files = glob.glob(os.path.join(data_folder, "*.zip"))

    measurement_files = []
    for zip_path in zip_files:
        measurement_files.append(MeasurementFile(zip_path))

    return measurement_files


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
    return BaseProcessor(measurement_file, sensor_name)


def process_zip_files(data_folder):
    measurement_files = create_measurement_files(data_folder)

    for measurement_file in tqdm(measurement_files):

        if is_file_processed(measurement_file.generate_file_hash()):
            print(f"Skipping {measurement_file.get_label()} as it has already been processed")
            continue

        print("\n\n---")
        print(measurement_file.get_metadata())
        print("---")

        for sensor_name, sensor_data in measurement_file.get_sensor_data().items():
            processor = get_processor(sensor_name, measurement_file)

            segments = processor.process(sensor_data)
            for i, segment in tqdm(enumerate(segments), desc=f"Processing {sensor_name} segments", total=len(segments)):
                segment_id = f"{sensor_name}_{i}"
                write_segment(measurement_file, segment, sensor_name, segment_id)

    print("All files processed.")


def write_segment(measurement_file, segment_df, sensor_name, segment_id):
    points = []
    write_api = client.write_api(write_options=SYNCHRONOUS)

    for index, row in segment_df.iterrows():
        date_time = index.to_pydatetime()
        point = Point(sensor_name).time(date_time)

        for field in COLUMN_MAPPINGS.get(sensor_name, []):
            if field in segment_df.columns and field != 'time':
                point = point.field(field, row.get(field, 0))

        point = point.tag("segment_id", segment_id)
        point = point.tag("file_hash", measurement_file.generate_file_hash())
        points.append(point)

    write_api.write(os.getenv("INFLUXDB_INIT_BUCKET"), os.getenv("INFLUXDB_INIT_ORG"), points, write_precision=WritePrecision.NS)


def is_file_processed(file_hash):
    query = f'from(bucket: "{os.getenv("INFLUXDB_INIT_BUCKET")}") |> range(start: -30d) |> filter(fn: (r) => r.file_hash == "{file_hash}") |> limit(n: 1)'
    result = query_api.query(query=query, org=os.getenv("INFLUXDB_INIT_ORG"))
    for table in result:
        for _ in table.records:
            return True
    return False


if __name__ == "__main__":
    print(f"--- InfluxDB Connection Params ---")
    print(f"USERNAME: {os.getenv('INFLUXDB_ADMIN_USERNAME')}")
    print(f"Org: {os.getenv('INFLUXDB_INIT_ORG')}")
    print(f"Bucket: {os.getenv('INFLUXDB_INIT_BUCKET')}")
    print("---")

    client = InfluxDBClient(url="http://localhost:8086",
                            org=os.getenv("INFLUXDB_INIT_ORG"),
                            username=os.getenv("INFLUXDB_ADMIN_USERNAME"),
                            password=os.getenv("INFLUXDB_ADMIN_PASSWORD"),
                            token=os.getenv("INFLUXDB_TOKEN"))

    query_api = client.query_api()
    write_api = client.write_api(write_options=SYNCHRONOUS)

    print("Connected to InfluxDB")
    process_zip_files(data_folder)
