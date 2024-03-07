from influxdb_client import InfluxDBClient, Point, WriteOptions
import dotenv

dotenv.load_dotenv()

client = InfluxDBClient(url="http://localhost:8086", token="admintoken", org="sensorbased")

write_api = client.write_api(write_options=WriteOptions(batch_size=500, flush_interval=10_000))