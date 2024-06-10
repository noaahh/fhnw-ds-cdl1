#!/usr/bin/env python3
import glob
import logging
import os
import time

import pandas as pd
import rootutils
import typer
from dotenv import load_dotenv
from influxdb_client import Point, WritePrecision
from joblib import Parallel, delayed
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.db import InfluxDBWrapper
from src.data.measurement_file import MeasurementFile
from src.utils import get_env_variable

app = typer.Typer()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

imported_local_files = set()

VERBOSE = False


def setup_logging(verbose: bool):
    """
    Set up logging configuration based on the verbosity level.
    :param verbose: If true, set logging to DEBUG, otherwise set to INFO.
    :return: Configured logger object.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    logger.disabled = not verbose


def filter_measurement_files(measurement_files):
    """Filters out measurement files based on specific criteria."""
    return [mf for mf in tqdm(measurement_files, desc="Filtering files", unit="file") if
            mf.get_label() is not None and
            mf.get_measurement_group() is not None and
            mf.generate_file_hash() not in imported_local_files and
            not is_file_imported(mf.generate_file_hash())]


def create_measurement_files(raw_data_folder, multi_threading, n_jobs):
    """Creates measurement files from zip files in the given directory."""
    zip_files = glob.glob(os.path.join(raw_data_folder, "**", "*.zip"), recursive=True)
    if multi_threading:
        with Parallel(n_jobs=n_jobs, prefer="threads") as parallel:
            return parallel(delayed(lambda x: MeasurementFile(x))(zip_path) for zip_path in
                            tqdm(zip_files, desc="Reading files", unit="file"))
    else:
        return [MeasurementFile(zip_path) for zip_path in tqdm(zip_files, desc="Reading files", unit="file")]


def import_all_zip_files(raw_data_folder, multi_threading, n_jobs):
    """Imports all zip files in the specified data folder."""
    measurement_files = create_measurement_files(raw_data_folder, multi_threading, n_jobs)
    filtered_files = filter_measurement_files(measurement_files)
    logger.debug(f"Found {len(filtered_files)} files to import.")
    if multi_threading:
        Parallel(n_jobs=n_jobs)(delayed(import_file)(mf) for mf in
                                tqdm(filtered_files, desc="Importing files", total=len(filtered_files), unit="file"))
    else:
        for mf in tqdm(filtered_files):
            import_file(mf)
    logger.info("All files imported")


def import_file(measurement_file):
    """Imports a single measurement file."""
    try:
        sensor_data = measurement_file.get_sensor_data()
    except ValueError as e:
        logger.error(f"Error reading file {measurement_file}: {e}")
        logger.error("Skipping file")
        return

    try:
        write_data_to_db(measurement_file, sensor_data)
    except Exception as e:
        logger.error(f"Error writing data to InfluxDB for file {measurement_file}: {e}")
        logger.error("Skipping file")
        return

    imported_local_files.add(measurement_file.generate_file_hash())
    logger.info(f"File {measurement_file} imported")


def is_file_imported(file_hash):
    """Checks if a file has been imported by querying InfluxDB."""
    query = (f'from(bucket: "{os.getenv("INFLUXDB_INIT_BUCKET")}") '
             f'|> range(start: 0, stop: now()) '
             f'|> filter(fn: (r) => '
             f'r.file_hash == "{file_hash}") |> limit(n: 1)')
    with InfluxDBWrapper() as influx:
        result = influx.query_api.query(query=query, org=os.getenv("INFLUXDB_INIT_ORG"))
        return any(True for _ in result)


def write_data_to_db(measurement_file, merged_data):
    """
    Writes merged sensor data to InfluxDB for each measurement file.
    :param measurement_file: An instance of MeasurementFile containing file metadata.
    :param merged_data: A pandas DataFrame of merged sensor data.
    """
    shared_metadata = {
        'label': measurement_file.get_label(),
        'measurement_group': measurement_file.get_measurement_group().name,
        'file_hash': measurement_file.generate_file_hash(),
        **measurement_file.get_metadata()
    }

    merged_data.index = pd.to_datetime(merged_data['time'])
    all_points = []
    for idx in merged_data.index:
        point = Point("measurement").time(idx)
        for k, v in shared_metadata.items():
            point = point.tag(k, v)
        for field, value in merged_data.loc[idx, merged_data.columns != 'time'].items():
            point = point.field(field, value)
        all_points.append(point)

    with InfluxDBWrapper() as client:
        client.write_api.write(os.getenv("INFLUXDB_INIT_BUCKET"),
                               os.getenv("INFLUXDB_INIT_ORG"),
                               all_points,
                               batch_size=5000,
                               protocol='line',
                               write_precision=WritePrecision.NS)


@app.command()
def import_data(run_periodically: bool = typer.Option(False, help="Enable periodic running of the import"),
                interval: int = typer.Option(60, help="Interval between runs in seconds"),
                multi_threading: bool = typer.Option(True, help="Enable multi-threading"),
                n_jobs: int = typer.Option(-1, help="Number of jobs to run in parallel"),
                verbose: bool = typer.Option(False, help="Enable verbose logging")):
    """Imports data from zip files in the specified folder to InfluxDB."""
    global VERBOSE
    VERBOSE = verbose
    setup_logging(verbose)

    raw_data_dir = os.path.join(get_env_variable("PROJECT_ROOT"), "data", "raw")

    logger.info("--- PARAMETERS ---")
    logger.info(f"INFLUXDB_URL: {get_env_variable('INFLUXDB_URL')}")
    logger.info(f"INFLUXDB_INIT_BUCKET: {get_env_variable('INFLUXDB_INIT_BUCKET')}")
    logger.info(f"INFLUXDB_INIT_ORG: {get_env_variable('INFLUXDB_INIT_ORG')}")
    logger.info(f"Raw data folder: {raw_data_dir}")
    logger.info(f"Run periodically: {run_periodically}")
    logger.info(f"Interval: {interval}")
    logger.info(f"Multi-threading: {multi_threading}")
    logger.info(f"Number of jobs: {n_jobs}")
    logger.info("-------------------------\n")

    if run_periodically:
        next_run_time = time.time()
        logger.info(f"Running every {interval} seconds")
        try:
            while True:
                if time.time() >= next_run_time:
                    logger.info("Periodic run started")
                    start_time = time.time()
                    import_all_zip_files(raw_data_dir, multi_threading, n_jobs)
                    logger.info(f"Import took {time.time() - start_time:.2f} seconds")
                    next_run_time = time.time() + interval
                    logger.info(f"Next run in {interval} seconds")
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Exiting...")
    else:
        start_time = time.time()
        import_all_zip_files(raw_data_dir, multi_threading, n_jobs)
        logger.info(f"Import took {time.time() - start_time:.2f} seconds")

    logger.info("Done!")


def import_file_data(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not os.path.isfile(file_path):
        raise ValueError("Only single file processing is supported at this time.")

    measurement_file = MeasurementFile(file_path)
    sensor_data = measurement_file.get_sensor_data()

    return sensor_data


def main():
    app()


if __name__ == "__main__":
    main()
