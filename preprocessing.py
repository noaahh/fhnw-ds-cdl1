import pandas as pd
import zipfile
import glob
import os
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from dotenv import load_dotenv

load_dotenv()

data_folder = "./data"
csv_names = ["Accelerometer", 
             "AccelerometerUncalibrated", 
             "Gravity", 
             "Gyroscope", 
             "GyroscopeUncalibrated"
             ]

def read_csv_from_zip(zip_path, csv_filename):
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(csv_filename) as f:
            return pd.read_csv(f)

def process_zip_files(data_folder, csv_filenames):
    zip_files = glob.glob(os.path.join(data_folder, "*.zip"))
    
    for zip_path in zip_files:
        class_label, timestamp = os.path.basename(zip_path).rsplit('.', 1)[0].split('-', 1)
        
        for csv_filename in csv_filenames:
            write_csv_to_influxdb(zip_path, f'{csv_filename}.csv')
            try:
                write_csv_to_influxdb(zip_path, f'{csv_filename}.csv')
                print(f"Processed {csv_filename} from {zip_path}")
            except KeyError:
                print(f"{csv_filename} not found in {zip_path}")

def write_csv_to_influxdb(zip_path, csv_filename):
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(csv_filename) as f:
            df = pd.read_csv(f)

            df['time'] = pd.to_datetime(df['time'])
            
            points = []
            for index, row in df.iterrows():
                point = Point("your_measurement").time(row['time'], WritePrecision.NS)\
                    .field("x", row['x'])\
                    .field("y", row['y'])\
                    .field("z", row['z'])\
                    .tag("seconds_elapsed", str(row['seconds_elapsed']))
                points.append(point)
            write_api.write(os.getenv("INFLUXDB_INIT_BUCKET"), os.getenv("INFLUXDB_INIT_ORG"), points)
            print(f"Data from {csv_filename} in {zip_path} written to InfluxDB")



client = InfluxDBClient(url="http://localhost:8086", 
                        org=os.getenv("INFLUXDB_INIT_ORG"), 
                        username=os.getenv("INFLUXDB_ADMIN_USERNAME"), 
                        password=os.getenv("INFLUXDB_ADMIN_PASSWORD"), 
                        token=os.getenv("INFLUXDB_TOKEN"))

write_api = client.write_api(write_options=SYNCHRONOUS)

process_zip_files(data_folder, csv_names)