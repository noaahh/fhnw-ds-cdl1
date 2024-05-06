import glob
import logging
import os
import time

from dotenv import load_dotenv
from influxdb_client import Point, WritePrecision
from tqdm import tqdm

from src.data.db import InfluxDBWrapper
from src.data.measurement_file import MeasurementFile
from src.helper import get_env_variable
from src.processing.base_processor import BaseProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

PROCESSED_FILES_LOCAL = set()  # In-memory set to keep track of processed files
RUN_PERIODICALLY = False

COLUMN_MAPPINGS = {'accelerometer': ['time', 'z', 'y', 'x'],
                   'gyroscope': ['time', 'z', 'y', 'x'],
                   'gravity': ['time', 'z', 'y', 'x'],
                   'accelerometeruncalibrated': ['time', 'z', 'y', 'x'],
                   'gyroscopeuncalibrated': ['time', 'z', 'y', 'x'],
                   'orientation': ['time', 'yaw', 'qx', 'qz', 'roll', 'qw', 'qy', 'pitch']}


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
