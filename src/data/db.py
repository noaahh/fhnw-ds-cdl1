import os

from dotenv import load_dotenv
from influxdb_client.client.influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

load_dotenv()


class InfluxDBWrapper:
    def __init__(self):
        self.url = os.getenv("INFLUXDB_URL", "http://localhost:8086")
        self.token = os.getenv("INFLUXDB_TOKEN")
        self.bucket = os.getenv("INFLUXDB_BUCKET")
        self.org = os.getenv("INFLUXDB_INIT_ORG")
        self.username = os.getenv("INFLUXDB_ADMIN_USERNAME")
        self.password = os.getenv("INFLUXDB_ADMIN_PASSWORD")
        self.client = None
        self.write_api = None
        self.query_api = None

    def __enter__(self):
        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org, username=self.username,
                                     password=self.password, timeout=120_000, retries=3, debug=False, enable_gzip=True)

        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()
