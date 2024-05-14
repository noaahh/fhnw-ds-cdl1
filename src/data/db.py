import os

from dotenv import load_dotenv
from influxdb_client.client.influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

load_dotenv()


class InfluxDBWrapper:
    def __init__(self,
                 url=None,
                 token=None,
                 bucket=None,
                 org=None,
                 username=None,
                 password=None,
                 timeout=120_000,
                 retries=3,
                 debug=False,
                 enable_gzip=True):
        self.url = os.getenv("INFLUXDB_URL", "http://localhost:8086") if url is None else url
        self.token = os.getenv("INFLUXDB_TOKEN") if token is None else token
        self.bucket = os.getenv("INFLUXDB_BUCKET") if bucket is None else bucket
        self.org = os.getenv("INFLUXDB_INIT_ORG") if org is None else org
        self.username = os.getenv("INFLUXDB_ADMIN_USERNAME") if username is None else username
        self.password = os.getenv("INFLUXDB_ADMIN_PASSWORD") if password is None else password
        self.timeout = timeout
        self.retries = retries
        self.debug = debug
        self.enable_gzip = enable_gzip

        self.client = None
        self.write_api = None
        self.query_api = None

        if not self._check_connection():
            raise ConnectionError("Could not connect to InfluxDB. Please check your connection settings.")

    def _create_client(self):
        return InfluxDBClient(url=self.url,
                              token=self.token,
                              org=self.org,
                              username=self.username,
                              password=self.password,
                              timeout=self.timeout,
                              retries=self.retries,
                              debug=self.debug,
                              enable_gzip=self.enable_gzip)

    def __enter__(self):
        self.client = self._create_client()
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()
        self.client = None
        self.write_api = None
        self.query_api = None

    def _check_connection(self):
        with self._create_client() as client:
            return client.ping()
