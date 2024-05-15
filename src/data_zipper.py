import os
import zipfile
from typing import Dict, Any

import pandas as pd
import typer
from tqdm import tqdm

app = typer.Typer()

COLUMN_MAPPINGS: Dict[str, Dict[str, Any]] = {
    'accelerometer': {
        'columns': ['timestamp', 'Accelerometer_z', 'Accelerometer_y', 'Accelerometer_x'],
        'rename': {'timestamp': 'time', 'Accelerometer_z': 'z', 'Accelerometer_y': 'y', 'Accelerometer_x': 'x'}
    },
    'gyroscope': {
        'columns': ['timestamp', 'Gyroscope_z', 'Gyroscope_y', 'Gyroscope_x'],
        'rename': {'timestamp': 'time', 'Gyroscope_z': 'z', 'Gyroscope_y': 'y', 'Gyroscope_x': 'x'}
    },
    'gravity': {
        'columns': ['timestamp', 'Gravity_z', 'Gravity_y', 'Gravity_x'],
        'rename': {'timestamp': 'time', 'Gravity_z': 'z', 'Gravity_y': 'y', 'Gravity_x': 'x'}
    },
    'orientation': {
        'columns': ['timestamp', 'Orientation_z', 'Orientation_y', 'Orientation_x'],
        'rename': {'timestamp': 'time', 'Orientation_z': 'yaw', 'Orientation_y': 'qx', 'Orientation_x': 'qz'}
    }
}


@app.command()
def process_data(csv_file_path: str, output_dir: str = 'output_zip_files'):
    data = pd.read_csv(csv_file_path)
    os.makedirs(output_dir, exist_ok=True)

    grouped = data.groupby('filename')
    typer.echo(f"Processing {len(grouped)} files...")

    for filename, group in tqdm(grouped, desc='Processing files', total=len(grouped)):
        process_group(group, output_dir)

    typer.echo("Data processing completed.")


def process_group(group: pd.DataFrame, output_dir: str):
    utc_timestamp = pd.to_datetime(group['timestamp'].iloc[0]).strftime('%Y-%m-%d_%H-%M-%S')
    activity = group['activity'].iloc[0].lower()
    zip_filename = f"{output_dir}/{activity}-{utc_timestamp}.zip"

    with zipfile.ZipFile(zip_filename, 'w') as z:
        for sensor, details in COLUMN_MAPPINGS.items():
            process_sensor_data(group, sensor, details, z)

        process_metadata(group, z, utc_timestamp)


def process_sensor_data(group: pd.DataFrame, sensor: str, details: Dict[str, Any], z: zipfile.ZipFile):
    sensor_df = group[details['columns']].rename(columns=details['rename'])
    required_columns = list(details['rename'].values())
    for col in required_columns:
        if col not in sensor_df.columns:
            sensor_df[col] = pd.NA

    sensor_csv_path = f'{sensor.capitalize()}.csv'
    sensor_df.to_csv(sensor_csv_path, index=False)
    z.write(sensor_csv_path, arcname=os.path.basename(sensor_csv_path))
    os.remove(sensor_csv_path)


def process_metadata(group: pd.DataFrame, z: zipfile.ZipFile, utc_timestamp: str):
    metadata_csv_path = 'Metadata.csv'
    metadata = {
        'device name': f'{group["person"].iloc[0]}_device',
        'platform': f'{group["person"].iloc[0]}_platform',
        'appVersion': "None",
        'recording time': utc_timestamp,
        'device id': group['person'].iloc[0]
    }
    metadata_df = pd.DataFrame(metadata, index=[0])
    metadata_df.to_csv(metadata_csv_path, index=False)
    z.write(metadata_csv_path, arcname=os.path.basename(metadata_csv_path))
    os.remove(metadata_csv_path)


if __name__ == "__main__":
    app()
