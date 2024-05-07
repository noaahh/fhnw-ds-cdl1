import glob
import logging
import os
import time

import pandas as pd
import typer
from dotenv import load_dotenv
from influxdb_client import Point, WritePrecision
from joblib import Parallel, delayed
from tqdm import tqdm

from src.data.db import InfluxDBWrapper
from src.data.measurement_file import MeasurementFile
from src.helper import get_env_variable
from src.processing.base_processor import BaseProcessor

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

PROCESSED_FILES_LOCAL = set()  # In-memory set to keep track of processed files

COLUMN_MAPPINGS = {'accelerometer': ['time', 'z', 'y', 'x'],
                   'gyroscope': ['time', 'z', 'y', 'x'],
                   'gravity': ['time', 'z', 'y', 'x'],
                   'orientation': ['time', 'yaw', 'qx', 'qz', 'roll', 'qw', 'qy', 'pitch']}

VERBOSE = False


def filter_measurement_files(measurement_files):
    return [mf for mf in tqdm(measurement_files, desc="Filtering files", unit="file") if
            mf.get_metadata()['device_name'] != 'iPhone X'
            and mf.get_label() is not None
            and mf.get_measurement_group() is not None
            and mf.generate_file_hash() not in PROCESSED_FILES_LOCAL
            and not is_file_processed(mf.generate_file_hash())]


def create_measurement_files(data_folder, multi_threading, n_jobs):
    zip_files = glob.glob(os.path.join(data_folder, "**", "*.zip"), recursive=True)
    if multi_threading:
        with Parallel(n_jobs=n_jobs, prefer="threads") as parallel:
            measurement_files = parallel(
                delayed(lambda x: MeasurementFile(x))(zip_path) for zip_path in
                tqdm(zip_files, desc="Reading files", unit="file"))
    else:
        measurement_files = [MeasurementFile(zip_path) for zip_path in
                             tqdm(zip_files, desc="Reading files", unit="file")]
    return measurement_files


def merge_sensor_data(sensor_data):
    merged_data = pd.DataFrame()

    raw_data_length = {sensor: len(data) for sensor, data in sensor_data.items() if sensor in COLUMN_MAPPINGS}
    assert len(set(raw_data_length.values())) == 1, f"Data length mismatch: {raw_data_length}"

    for sensor_name, expected_cols in COLUMN_MAPPINGS.items():
        if sensor_name not in sensor_data:
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

    missing_sensors = set(COLUMN_MAPPINGS.keys()) - set(sensor_data.keys())
    if missing_sensors:
        logger.warning(f"File {measurement_file} is missing data for sensors: {missing_sensors}")
    
    segments, extracted_features = BaseProcessor(measurement_file).process(merged_data)
    
    write_segments_to_db(measurement_file, segments, extracted_features)


def mark_file_as_processed(zip_file_path):
    processed_file_path = f"{zip_file_path}.processed"
    os.rename(zip_file_path, processed_file_path)


def process_zip_files(data_folder, multi_threading, n_jobs):
    measurement_files = create_measurement_files(data_folder, multi_threading, n_jobs)
    filtered_files = filter_measurement_files(measurement_files)
    logger.debug(f"Found {len(filtered_files)} files to process.")

    if multi_threading:
        Parallel(n_jobs=n_jobs)(delayed(process_and_mark_file)(mf) for mf in
                                tqdm(filtered_files, desc="Processing files", total=len(filtered_files), unit="file"))
    else:
        for mf in tqdm(filtered_files):
            process_and_mark_file(mf)

    logger.info("All files processed")


def process_and_mark_file(measurement_file):
    try:
        process_measurement_file(measurement_file)
        logger.info(f"File {measurement_file} processed")
    except Exception as e:
        logger.error(f"Error processing file {measurement_file}:")
        logger.error(e)
        logger.error(f"Skipping file {measurement_file}")
    finally:
        # mark_file_as_processed(measurement_file.zip_file_path)
        PROCESSED_FILES_LOCAL.add(measurement_file.generate_file_hash())


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


def write_segments_to_db(measurement_file, segments_df, extracted_features):
    shared_segment_metadata = {
        'label': measurement_file.get_label(),
        'measurement_group': measurement_file.get_measurement_group().name,
        'file_hash': measurement_file.generate_file_hash(),
        **measurement_file.get_metadata()
    }
    
    all_points = []
    for i, segment_df in enumerate(segments_df):
        print(f'Extracted Features: {extracted_features[i]}')
        segment_metadata = {**shared_segment_metadata, 
                            'segment_id': i,
                            **extracted_features[i]}

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


def process_and_import(data_folder: str,
                       run_periodically: bool = typer.Option(False, help="Enable periodic running of the process"),
                       interval: int = typer.Option(60, help="Interval between runs in seconds"),
                       multi_threading: bool = typer.Option(True, help="Enable multi-threading"),
                       n_jobs: int = typer.Option(-1, help="Number of jobs to run in parallel"),
                       verbose: bool = typer.Option(False, help="Enable verbose logging")):
    logger.info("--- PARAMETERS ---")
    logger.info(f"INFLUXDB_URL: {get_env_variable('INFLUXDB_URL')}")
    logger.info(f"INFLUXDB_INIT_BUCKET: {get_env_variable('INFLUXDB_INIT_BUCKET')}")
    logger.info(f"INFLUXDB_INIT_ORG: {get_env_variable('INFLUXDB_INIT_ORG')}")
    logger.info(f"Data folder: {data_folder}")
    logger.info(f"Run periodically: {run_periodically}")
    logger.info(f"Interval: {interval}")
    logger.info(f"Multi-threading: {multi_threading}")
    logger.info(f"Number of jobs: {n_jobs}")
    logger.info(f"Verbose: {verbose}")
    logger.info("-------------------------\n")

    global VERBOSE
    VERBOSE = verbose
    logger.disabled = not verbose

    next_run_time = time.time()

    if run_periodically:
        logger.info(f"Running every {interval} seconds")
        try:
            while True:
                if time.time() >= next_run_time:
                    logger.info("Periodic run started")

                    start_time = time.time()
                    process_zip_files(data_folder, multi_threading, n_jobs)
                    logger.info(f"Processing took {time.time() - start_time:.2f} seconds")

                    next_run_time = time.time() + interval
                    logger.info(f"Next run in {interval} seconds")

                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Exiting...")
    else:
        start_time = time.time()
        process_zip_files(data_folder, multi_threading, n_jobs)
        logger.info(f"Processing took {time.time() - start_time:.2f} seconds")

    logger.info("Done!")


if __name__ == "__main__":
    typer.run(process_and_import)
