# cdl1: Sensor Based Activity Recognition

## Prerequisites

Before you begin, ensure you have met the following requirements:

* Python 3.11
* Docker and Docker Compose
* Git

## Installation

Set up the Python environment with Conda:

```bash
make setup
```

## Project Structure

The repository is structured to support scalable data processing and machine learning model development, with a focus on
flexibility and configurability for various experiments:

- **`configs/`**: Contains Hydra configuration files that define different setups for experiments. These configurations
  determine how data pipelines are executed and how models are trained. Hydra allows for hierarchical configuration by
  composition and overrides, making it simple to manage complex configurations dynamically during runtime.

- **`data/`**: This directory houses the raw data, alongside caches and partitions of processed data. It ensures that
  data management is streamlined and separated from code logic:
    - `raw/`: Original, immutable data files.
    - `cache/`: Temporary storage for intermediate processing steps.
    - `partitions/`: Segmented data files after processing, ready for modeling.

- **`src/`**: Source code directory containing all scripts for data handling and model training:
    - `data_pipeline.py`: Manages the workflow from data ingestion to preprocessing and feature engineering. It is
      configured dynamically based on the experiment specifications in the Hydra configs.
    - `train.py`: Define the logic of training models using PyTorch

- **`Dockerfile` and `docker-compose.yml`**: These files are crucial for defining and managing the project's container
  setup, facilitating reproducible environments and easy deployment.

- **`Makefile`**: Provides a collection of scripts for routine tasks in the project's lifecycle, such as setting up
  environments, running experiments, and cleaning up resources. It simplifies the execution of complex workflows and
  ensures consistency across different development environments.

## Data

The data used in this project is collected using the SensorLogger app available on the Google Play Store and Apple App
Store.

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


## Experiment Configuration

The project uses Hydra to manage configurations for data processing and model training. The configurations are stored in
the `configs/` directory and can be composed and overridden to define different experiment setups.

## Usage

To import the latest data and set up necessary infrastructure services:

```bash
make import-latest
```

To run an experiment with a specific configuration on either the data pipeline or model training:

```bash
python src/data_pipeline.py experiment=<experiment_name> [overrides]
```

or 

```bash
python src/train.py experiment=<experiment_name> [overrides]
```


To clean up resources and remove temporary files:

```bash
make clean
```

To stop the InfluxDB service:

```bash
make stop-influxdb
```

To tear down the InfluxDB. Be aware that this will remove all data stored in the database:

```bash
make teardown-influxdb
```