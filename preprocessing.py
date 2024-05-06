import glob
import logging
import os
import time

import pandas as pd
from dotenv import load_dotenv
from influxdb_client import Point, WritePrecision
from tqdm import tqdm

from src.data.db import InfluxDBWrapper
from src.data.measurement_file import MeasurementFile
from src.helper import get_env_variable
from src.processing.base_processor import BaseProcessor

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

PROCESSED_FILES_LOCAL = set()  # In-memory set to keep track of processed files
RUN_PERIODICALLY = False

COLUMN_MAPPINGS = {'accelerometer': ['time', 'z', 'y', 'x'],
                   'gyroscope': ['time', 'z', 'y', 'x'],
                   'gravity': ['time', 'z', 'y', 'x'],
                   'orientation': ['time', 'yaw', 'qx', 'qz', 'roll', 'qw', 'qy', 'pitch']}


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


def merge_sensor_data(sensor_data):
    merged_data = pd.DataFrame()  # Initialize an empty DataFrame for merging data

    raw_data_length = {sensor: len(data) for sensor, data in sensor_data.items() if sensor in COLUMN_MAPPINGS}
    assert len(set(raw_data_length.values())) == 1, f"Data length mismatch: {raw_data_length}"

    for sensor_name, expected_cols in COLUMN_MAPPINGS.items():
        if sensor_name not in sensor_data:
            logger.warning(f"Sensor {sensor_name} not found in data")
            continue

        data = sensor_data.get(sensor_name)
        if not all(col in data.columns for col in expected_cols):
            raise ValueError(f"Columns mismatch for sensor {sensor_name}: {data.columns} vs {expected_cols}")

        data = data[expected_cols]
        data = data.set_index('time')

        data = data[[col for col in expected_cols if col in data.columns and col != 'time']]
        data.columns = [f"{sensor_name}_{col}" for col in data.columns]

        if merged_data.empty:
            merged_data = data
        else:
            merged_data = pd.concat([merged_data, data], axis=1)

    merged_data = merged_data.reset_index()

    assert len(merged_data) == raw_data_length[
        'accelerometer'], f"Data length mismatch after merging: {len(merged_data)} vs {raw_data_length['accelerometer']}"
    return merged_data


def process_measurement_file(measurement_file):
    sensor_data = measurement_file.get_sensor_data()
    merged_data = merge_sensor_data(sensor_data)
    segments = BaseProcessor(measurement_file).process(merged_data)
    write_segments_to_db(measurement_file, segments)


def mark_file_as_processed(zip_file_path):
    processed_file_path = f"{zip_file_path}.processed"
    os.rename(zip_file_path, processed_file_path)


def process_zip_files():
    data_folder = get_env_variable("DATA_FOLDER")
    measurement_files = create_measurement_files(data_folder)
    filtered_files = filter_measurement_files(measurement_files)
    logger.debug(f"Found {len(filtered_files)} files to process.")

    for measurement_file in tqdm(filtered_files):
        process_measurement_file(measurement_file)
        mark_file_as_processed(measurement_file.zip_file_path)

        PROCESSED_FILES_LOCAL.add(measurement_file.generate_file_hash())
        logger.info(f"File {measurement_file} processed")

    logger.info("All files processed")


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


def write_segments_to_db(measurement_file, segments_df):
    shared_segment_metadata = {
        'label': measurement_file.get_label(),
        'measurement_group': measurement_file.get_measurement_group().name,
        'file_hash': measurement_file.generate_file_hash(),
        **measurement_file.get_metadata()
    }

    all_points = []
    for i, segment_df in enumerate(segments_df):
        segment_metadata = {**shared_segment_metadata, 'segment_id': i}

        if not isinstance(segment_df.index, pd.DatetimeIndex):
            segment_df.index = pd.to_datetime(segment_df.index)

        for idx in segment_df.index:
            point = Point("segment").time(idx)
            for k, v in segment_metadata.items():
                point = point.tag(k, v)
            for field, value in segment_df.loc[idx, segment_df.columns != 'time'].items():
                point = point.field(field, value)
            all_points.append(point)

    with InfluxDBWrapper() as client:
        client.write_api.write(os.getenv("INFLUXDB_INIT_BUCKET"), os.getenv("INFLUXDB_INIT_ORG"),
                               all_points, write_precision=WritePrecision.NS,
                               batch_size=5000, protocol='line')


if __name__ == "__main__":
    logger.info("--- DB CONNECTION ---")
    logger.info(f"INFLUXDB_URL: {get_env_variable('INFLUXDB_URL')}")
    logger.info(f"INFLUXDB_INIT_BUCKET: {get_env_variable('INFLUXDB_INIT_BUCKET')}")
    logger.info(f"INFLUXDB_INIT_ORG: {get_env_variable('INFLUXDB_INIT_ORG')}")
    logger.info("---------------------")

    interval = 60 * 1  # Interval in seconds
    next_run_time = time.time()

    if RUN_PERIODICALLY:
        logger.info(f"Running every {interval} seconds")
        try:
            while RUN_PERIODICALLY:
                if time.time() >= next_run_time:
                    logger.info("Processing zip files...")

                    process_zip_files()

                    next_run_time = time.time() + interval

                    logger.info(f"Next run in {interval} seconds")

                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Exiting...")

    else:
        process_zip_files()

    logger.info("Done!")
