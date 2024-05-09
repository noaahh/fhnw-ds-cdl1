# cdl1: Sensor Based Activity Recognition

## Docker setup

To run the InfluxDB from the docker compose file, execute the following command:

```bash
docker-compose up -d influxdb
```

This will start the InfluxDB service in the background and configures it with the variables defined in the `.env` file.

## Logistics for data import

The raw data from the SensorLogger app should be placed and structured as follows to align with the default settings of
the provided scripts:

```
data
├── raw
│   ├── 2 (Measurement Group)
│   │   ├── walking_2021-01-01_12-00-00.zip
│   │   ├── walking_2021-01-01_12-10-00.zip
│   │   ├── ...
│   ├── ...
```

The `raw` directory contains subdirectories named after the measurement groups. Each measurement group directory
contains zip files with sensor data for different activities. If the subdirectories are not named after the measurement
groups, the group will be deemed as `NO GROUP`. The defined measurement groups are defined in
the `src/data/label_mapping.py` file.

The zip files should be named according to the activity with an underscore-separated timestamp. The mapping of labels is
also defined in the `src/data/label_mapping.py` file.

## Scripts

Each scripts available parameters can be displayed by running the script with the `--help` flag.

```bash
python script.py --help
```

#### import.py

The `import.py` script is designed for the initial data handling phase. It imports sensor data stored in the specified
directory and can be configured to run either once or periodically based on the settings provided. This script supports
multi-threading to speed up the data import process, especially useful when dealing with large datasets.

To run the script with the default options and import data from the "data" folder:

```bash
python import.py --raw-data-dir "data/raw"
```

**Parameters**:
You can customize the import process using several options:

- `--run_periodically`: Enable this flag to import data at regular intervals.
- `--interval`: Set the time interval (in seconds) between consecutive imports when running periodically.
- `--multi_threading`: Enable multi-threading for faster execution.
- `--n_jobs`: Specify the number of threads to use.
- `--verbose`: Enable detailed logging.

Run the script with the `--help` flag to see all available options.

```bash
python import.py --help
```

#### data_pipeline.py

The `data_pipeline.py` script is a comprehensive data preprocessing tool that prepares the imported sensor data for
further analysis and modeling. It performs a series of preprocessing steps such as cropping the data to remove unwanted
segments, resampling the data to a consistent rate, and dividing the data into segments with specified overlaps. It also
provides options for feature extraction like FFT (Fast Fourier Transform) and Pearson correlation calculations. Advanced
settings include options for data scaling, dimensionality reduction using PCA, and handling of output data in various
ways.

To execute the script using default settings defined by environment variables:

```bash
python data_pipeline.py --output-dir "data/partitions"
```

**Parameters**:
This script offers a wide range of configurable parameters, allowing fine-tuning of the preprocessing steps:

- `--crop_start_s` and `--crop_end_s`: Define how many seconds to crop at the beginning and end of each signal.
- `--resample_rate_hz`: Set the resampling rate in Hz.
- `--segment_size_s` and `--overlap_s`: Specify the size of each data segment and the overlap between segments.
- Feature calculations can be toggled with `--fft` for FFT features and `--pearson_corr` for Pearson correlation.

Run the script with the `--help` flag to see all available options.

```bash
python data_pipeline.py --help
```