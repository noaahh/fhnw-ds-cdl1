import os

from dotenv import load_dotenv
from influxdb_client.client.influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

load_dotenv()


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
