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

- **[`configs/`](/configs/)**: Contains [Hydra](https://hydra.cc/) configuration files that define different setups for experiments. These configurations
  determine how data pipelines are executed and how models are trained. Hydra allows for hierarchical configuration by
  composition and overrides, making it simple to manage complex configurations dynamically during runtime.

- **[`data/`](/data/)**: This directory houses the raw data, alongside caches and partitions of processed data. It ensures that
  data management is streamlined and separated from code logic:
    - [`raw/`](/data/raw/): Original, immutable data files.
    - [`cache/`](/data/cache/): Temporary storage for intermediate processing steps.
    - [`partitions/`](/data/partitions/): Segmented data files after processing, ready for modeling.

- **[`notebooks/`](/notebooks/)**: This directory contains the Jupyter Notebooks that were used to explore data at hand and to experiment with approaches which were then implemented into the pipeline process.
  - [`partitioning.ipynb`](/notebooks/partitioning.ipynb): The Partitioning Notebook was used to explore and validate the partitioning of all sessions and to validate that sessions were unique to a partition.
  - [`preprocessing.ipynb`](/notebooks/preprocessing.ipynb): The Preprocessing Notebook holds explorations on the data at hand upon which the preprocessing pipeline was later on constructed and configured. The findings also contributed to the validation of smoothing and feature extracting functions.
  - [`session_truncation.ipynb`](/notebooks/session_truncation.ipynb): Contains exploration on what "good" session truncation parameters could be (e.g. how long a segment inside a session should be).

- **[`scripts/`](/scripts/)**: This directory mainly holds a simple script used to train all proposed models on a machine. It was used to train the models at hand on a GPU cluster so the models' training didn't have to be called manually each time.

- **[`src/`](/src/)**: Source code directory containing all scripts for data handling and model training:
    - [`data/`](/src/data/): Contains helpers, wrappers and data mapping structures related to the data handling.
      - [`dataset.py`](/src/data/dataset.py): Holds `SensorDatasetTorch` and `SensorDataModule`. The `SensorDataModule` wraps around the `LightningDataModule` which can then be used to hold together the entire dataset in a streamlined way. The `SensorDataModule` gives us a structure to access and handle our data; For example creating a `Dataset` module which can seamlessly be passed into any PyTorch model. In this Project's context, the `SensorDatasetTorch` class is derived from PyTorch's `Dataset` class, allowing us to control how special columns are handled (for example dropping target variables or other response variables).
      - [`db.py`](/src/data/db.py): This file holds an `InfluxDBWrapper` which just makes accessing the timeseries Influx database easy.
      - [`label_mapping.py`](/src/data/label_mapping.py): Steers the label mapping from the measurements filenames onto typed Enums.
      - [`measurement_file.py`](/src/data/measurement_file.py): The `MeasurementFile` class provides a structured approach to managing and processing measurement data stored in zip files. It validates the input, reads and processes sensor data, handles platform-specific adjustments, and generates unique identifiers for each file based on its metadata.
    - [`extraction/`](/src/extraction/): This subfolder holds all outsourced files that control how further markers from the sensor recordings are extracted.
      - [`fft.py`](/src/extraction/fft.py): The `fft.py` file holds all Fourier Transform related feature extraction functions. It, for example, holds functions that will extract the Dominant Frequency of a signal (sensor recording).
      - [`moving_average.py`](/src/extraction/moving_average.py): As the name already reveals, the `moving_average.py` file holds two functions that perform a smoothing task using a rolling average on a sliding window.
    - [`models/`](/src/models/): The models directory contains all explored model architectures that were trained in the context of this challenge.
      - [`cnn.py`](/src/models/cnn.py): The Convolutional Neural Network (CNN) model as explained in [Overview]().
      - [`deep_res_bidir_lstm.py`](/src/models/deep_res_bidir_lstm.py): The Deep Residual Bidirectional LSTM model as explained in [Overview]().
      - [`log_reg.py`](/src/models/log_reg.py): The Logistic Regression model as explained in [Overview]().
      - [`lstm.py`](/src/models/lstm.py): The Long Short-Term Memory (LSTM) model as explained in [Overview]().
      - [`x_lstm.py`](/src/models/x_lstm.py): The Extended Long Short-Term memory (xLSTM) model as explained in [Overview]().
      - [`transformer.py`](/src/models/transformer.py): The Transformer model as explained in [Overview]().
    - [`processing/`](/src/processing/): The `processing` directory contains files with functions used for processing the signals in a preprocessing step.
      - [`denoising.py`](/src/processing/denoising.py): The denoising file contains functions that introduce Butterworth Filtering (as a denoising measure and to eliminate possibily faulty peaks) and Wavelet Denoising, another denoising measure with generalizing effect. 
      - [`time_series.py`](/src/processing/time_series.py): Controls functions used for cropping the segments into 5s pieces and resampling all files into the same sampling rate domain.
    - [`dashboard.py`](/src/dashboard.py): The Streamlit Dashboard used as a frontend component to make use of the trained models.
    - [`data_pipeline.py`](/src/data_pipeline.py): Manages the workflow from data ingestion to preprocessing and feature engineering. It is
      configured dynamically based on the experiment specifications in the Hydra configs.
    - [`data_zipper.py`](/src/data_zipper.py): Processes sensor data from a CSV file and organizes it into zipped files, each containing sensor-specific data and metadata.
    - [`file_import.py`](/src/file_import.py): The `file_import.py` script imports sensor data from zip files into an InfluxDB database. It sets up logging, filters and processes measurement files, and supports multi-threaded execution to improve performance. The script includes functionality to periodically import data and logs important parameters and progress throughout the process.
    - [`model_pipeline.py`](/src/model_pipeline.py): The `model_pipeline.py` script configures logging, loads the specified models, and runs predictions on sensor data files using a selected model. It supports model loading from local checkpoints or Weights & Biases artifacts, processes data through a pipeline, and generates predictions with PyTorch Lightning. The script logs detailed information throughout its execution and returns the majority label from the predictions.
    - [`predict_api.py`](/src/predict_api.py): The `prediction_api.py` script sets up an API using FastAPI to handle file uploads for prediction.
    - [`train.py`](/src/train.py): Define the logic of training models using PyTorch.
    - [`utils.py`](/src/utils.py): This utils script contains functions to manage environment variables, generate paths for partitioned data, load partitioned data from parquet files, and validate smoothing parameters.

- **[`Dockerfile`](/dockerfiles/Dockerfile) and [`docker-compose.yml`](./docker-compose.yml)**: These files are crucial for defining and managing the project's container
  setup, facilitating reproducible environments and easy deployment.

- **[`Makefile`](./Makefile)**: Provides a collection of scripts for routine tasks in the project's lifecycle, such as setting up
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