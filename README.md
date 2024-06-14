# cdl1: Sensor Based Activity Recognition

**Authors**: Dominik Filliger, Nils Fahrni, Noah Leuenberger (2024)

This repository contains the implementation for the Sensor Based Activity Recognition Challenge @ University of Applied
Sciences Northwestern Switzerland (FHNW). The challenge explores the development of a pipeline for sensor-based activity
recognition using machine/deep learning models. The pipeline includes data processing, feature engineering, model
training, and prediction. The project aims to provide a scalable and configurable solution for activity recognition
tasks using time-series data from multiple sensors.

<!-- TOC -->
* [cdl1: Sensor Based Activity Recognition](#cdl1-sensor-based-activity-recognition)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Project Overview](#project-overview)
  * [Data Structure](#data-structure)
  * [Experiment Configuration with Hydra](#experiment-configuration-with-hydra)
    * [Introduction to Hydra](#introduction-to-hydra)
    * [Configuration Files Structure](#configuration-files-structure)
    * [Composition of Configurations](#composition-of-configurations)
    * [Overriding Configurations](#overriding-configurations)
    * [Example Scenario](#example-scenario)
    * [Defaults](#defaults)
      * [General Configuration](#general-configuration)
      * [Data Module](#data-module)
      * [Database Configuration](#database-configuration)
      * [Preprocessing](#preprocessing)
      * [Data Partitioning](#data-partitioning)
  * [Workflow and Execution](#workflow-and-execution)
  * [Dashboard](#dashboard)
  * [Model Analysis](#model-analysis)
  * [Notebooks](#notebooks)
  * [Codebase](#codebase)
<!-- TOC -->

## Prerequisites

Before you begin, ensure you have met the following requirements:

* Python 3.11
* Conda
* Docker and Docker Compose
* Git

## Installation

Copy the example environment file and adjust the settings as needed:

```bash
cp .env.example .env
```

Set the variables in the `.env` file to match your environment. Defaults are provided for the database connection and
other services. Replace the `PROJECT_ROOT` variable with the absolute path to the project root directory.

Note: The absolute path wont work in the Docker container, so you need to set the `PROJECT_ROOT` variable to be relative
to the Docker container file system.

Set up the Python environment with Conda:

```bash
make setup
```

Afterwards you should be good to go and use the Python environment.

**If you prefer to use a virtual environment, you can create one using the following commands:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project Overview

- **[`configs/`](configs/)**: Houses [Hydra](https://hydra.cc/) configurations for experiments, detailing data pipelines
  and training setups. More at [Experiment Configuration with Hydra](#experiment-configuration-with-hydra).

- **[`data/`](data/)**: Stores raw, cached, and processed data:
    - **[`raw/`](data/raw/)**: Original data files.
    - **[`cache/`](data/cache/)**: Temporarily stores data during processing.
    - **[`partitions/`](data/partitions/)**: Data segments post-processing.

- **[`notebooks/`](notebooks/)**: Contains Jupyter Notebooks for data exploration and pipeline trials:
    - **[`partitioning.ipynb`](notebooks/partitioning.ipynb)**: Validates data partitioning.
    - **[`preprocessing.ipynb`](notebooks/preprocessing.ipynb)**: Develops preprocessing strategies.
    - **[`session_truncation.ipynb`](notebooks/session_truncation.ipynb)**: Examines session truncation, a preprocessing
      strategy.

- **[`scripts/`](scripts/)**: Simple script for model training on GPU clusters.

- **[`src/`](src/)**: Source code for data handling and model training:
    - **Data Management**: Includes helpers and wrappers for data integration and management.
    - **Feature Extraction**: Functions for signal processing and feature extraction.
    - **Model Architectures**: Contains scripts for CNN, LSTM, Logistic Regression, Transformer, and xLSTM models.
    - **Processing Utilities**: Manages data preprocessing, feature engineering, and prediction scripts.

- **[`dockerfiles`](dockerfiles/)**: Contains Docker configurations for project setup and deployment:
    - **[`Dockerfile`](dockerfiles/Dockerfile)** and **[`docker-compose.yml`](./docker-compose.yml)**: Define the
      project's container environment.
    - **[`Dockerfile.dashboard`](dockerfiles/Dockerfile.dashboard)**: Configures the Streamlit Dashboard container.

- **[`Makefile`](Makefile)**: Scripts for routine tasks like setting up environments, running experiments, and cleaning
  resources.

## Data Structure

The data used in this project is collected using the [SensorLogger](https://github.com/tszheichoi/awesome-sensor-logger)
app available on the Google Play Store and Apple App
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
│   ...
```

The `raw` directory contains subdirectories named after the measurement groups. Each measurement group directory
contains zip files with sensor data for different activities. If the subdirectories are not named after the measurement
groups, the group will be deemed as `NO GROUP`. The defined measurement groups are defined in
the `src/data/label_mapping.py` file. Measurement groups allow for grouping measurements efforts together, for example
when multiple people are involved in the data collection.

The zip files should be named according to the activity with an underscore-separated timestamp. The mapping of labels is
also defined in the `src/data/label_mapping.py` file.

You can find all the collected and used data on [SWITCHdrive](https://drive.switch.ch/index.php/s/l0o3EDXyeaFB2Je).

## Experiment Configuration with Hydra

### Introduction to Hydra

Hydra is a configuration management tool developed by Facebook AI. It simplifies the process of configuring complex
applications by allowing developers to dynamically create hierarchical configurations. This is particularly useful in
machine learning and data science projects where multiple experiments often need varying settings.

### Configuration Files Structure

The project stores all configuration files within the `configs/` directory. These files are typically organized in a
structured manner, where each file or directory corresponds to a specific aspect of the project like data processing,
model architecture, training parameters, etc.

### Composition of Configurations

Hydra allows configurations to be "composed," meaning different configuration files can be merged together at runtime to
form a final configuration. This composition is based on a main configuration file (
named `train.yaml` / `pipeline.yaml`) and
includes others as needed. For example, you might have a base configuration for training a model, but you can override
certain parameters for specific experiments like changing the learning rate or batch size.

### Overriding Configurations

One of Hydra's powerful features is the ability to override configurations directly from the command line. This means
you can adjust the parameters of your experiments without altering the configuration files. For instance, if you have a
parameter for batch size set in your configuration file, you can override it when starting the experiment by using a
command like:

```bash
python src/train.py trainer=gpu
```

This will temporarily set the trainer to use the GPU, even if the default configuration specifies a different trainer (
e.g. CPU).

### Example Scenario

Suppose you want to train the CNN model with a learning rate of 0.001. You can achieve this by running the following
command:

```bash
python src/train.py experiment=cnn model.optimizer.lr=0.001
```

In this command, `experiment=cnn` specifies that the CNN model should be trained, and `model.optimizer.lr=0.001`

### Defaults

The default configuration files are defined in the `configs/**/defaults.yaml` files. These files set the baseline
parameters for the various aspects of the application. Below are the most important default parameters outlined to help
understand their purpose and how they can be customized:

#### General Configuration

- **n_jobs**: Number of jobs to run in parallel. `null` indicates that it defaults to the number of cores.
- **verbose**: Enables verbose output, set to `true` by default.
- **seed**: Seed for random number generation, set to `1337` for reproducibility.

#### Data Module

- **data**:
    - **batch_size**: Number of samples in each batch, defaults to `128`.
    - **num_workers**: Number of subprocesses to use for data loading. `null` uses the number of CPU cores.
    - **pin_memory**: Whether to use pinned (page-locked) memory. Set to `true`.
    - **use_persistent_workers**: Use workers which remain alive between data iterations. Set to `false`.

#### Database Configuration

- **database**:
    - **bucket**: Bucket name for the database, sourced from environment variable `INFLUXDB_INIT_BUCKET`.
    - **org**: Organization name for the database, sourced from environment variable `INFLUXDB_INIT_ORG`.
    - **url**: Database URL, sourced from environment variable `INFLUXDB_URL`.
    - **use_cache**: Enable caching of database queries, set to `true`.

#### Preprocessing

- **preprocessing**:
    - **crop**: Parameters defining how to crop the data of each session.
        - **start_seconds**: Start time in seconds for cropping, set to `5`.
        - **end_seconds**: End time in seconds for cropping, also set to `5`.
    - **resample_rate_hz**: Data resampling rate in Hz, set to `50`.
    - **segment_size_seconds**: Size of each data segment in seconds, set to `5`.
    - **overlap_seconds**: Overlap between segments in seconds, set to `2`.
    - **max_session_length_s**: Maximum session length in seconds, set to `180`.
    - **feature_extraction**: Controls feature extraction methods.
        - **use_fft**: Use Fast Fourier Transform, set to `false`.
        - **use_pearson_corr**: Use Pearson correlation, set to `false`.
    - **smoothing**: Smoothing method parameters.
        - **type**: Type of filter, set to `butterworth`.
        - **cutoff**: Filter cutoff frequency, set to `6`.
        - **order**: Filter order, set to `4`.
    - **scaling**: Data scaling type.
        - **type**: Type of scaling, `standard` indicates standard normalization.

#### Data Partitioning

- **partitioning**:
    - **validation_size**: Fraction of data set aside for validation, set to `0.2`.
    - **test_size**: Fraction of data used for testing, also set to `0.2`.
    - **k_folds**: Number of folds for K-fold cross-validation. `null` indicates not used. Is situational set to `5`.
    - **stratify**: Whether to stratify splits, set to `true`.
    - **split_by**: Attribute used to split data, set to `session_id`. Session refers to a single measurement file.

## Workflow and Execution

To import the latest data and set up necessary infrastructure services:

```bash
make import-latest
```

This will start the InfluxDB service and import the latest data using the `file_import.py` script. The script will read
the data from the `data/raw` directory and import it into the InfluxDB database avoiding duplicates in the process.

After the latest data has been imported, you can run the data pipeline to prepare the data for training.

The main idea is to have defined experiments in the `configs/experiments` directory. These experiments define the
parameters for the data pipeline and model training. To run the data pipeline of and experiment, execute the following
command:

```bash
python src/data_pipeline.py experiment=<experiment_name> [overrides]
```

Be aware that the `experiment` argument should be set to the name of the experiment configuration file in the `configs/`
directory. The `overrides` argument can be used to override specific parameters in the configuration file.

Also note that the `data_pipeline.py` script will maintain a cache file in the `data/cache` directory to avoid
reprocessing the data if the script is run multiple times with the same configuration. If you want to use the newly
imported data, you can delete the cache file or run the script with the `database.use_cache=False` override. As this
process can take a while, it is recommended to use the provided cache file
from [SWITCHdrive](https://drive.switch.ch/index.php/s/l0o3EDXyeaFB2Je) and set `database.use_cache=True` in the
experiment configuration.

Once the data pipeline has been executed successfully, you can train the model using the following command:

```bash
python src/train.py experiment=<experiment_name> [overrides]
```

The `experiment` argument should be set to the name of the experiment configuration file in the `configs/` directory,
and
the `overrides` argument can be used to override specific parameters in the configuration file.

## Dashboard

After training the model, you can run the Streamlit dashboard to interact with the trained models:

```bash
make run-dashboard
```

This will start the Streamlit server and open the dashboard in your default web browser. The dashboard allows you to
upload sensor data files and get predictions from the trained models based on a Weights & Biases artifact.

Here are some pretrained models that can be used with the dashboard for each corresponding model architecture which can
be copy pasted as Wandb artifact path in the dashboard:

```plaintext
- CNN:                                    lang-based-yappers/cdl1/model-0gfliwdd:best
- Deep Residual Bidirectional LSTM:       lang-based-yappers/cdl1/model-kfezxxez:best
- Logistic Regression:                    lang-based-yappers/cdl1/model-4x2zewmj:best
- LSTM:                                   lang-based-yappers/cdl1/model-octutiu4:best
- xLSTM:                                  lang-based-yappers/cdl1/model-jfnd0mql:best
- Transformer:                            lang-based-yappers/cdl1/model-df757nif:best
```

As the data is usually scaled and normalized before training, the dashboard will also scale and normalize the uploaded
file using a provided scaler artifact. The scaler artifact used for the latest data and pretrained runs can be found
on [SWITCHdrive](https://drive.switch.ch/index.php/s/l0o3EDXyeaFB2Je).

Here is a video showing how to use the dashboard:

![VIDEO](https://cdn.loom.com/sessions/thumbnails/1007378404564d7b949bde2d19b872a9-with-play.gif)(https://www.loom.com/share/1007378404564d7b949bde2d19b872a9?sid=edc77fed-bc8a-49a1-9ad3-2a553065a0e0)

Along with the dashboard, you can also run the prediction API to handle file uploads for prediction:

```bash
make run-api
```

This will start the FastAPI server and allow you to upload sensor data files for prediction. The API will return the
majority label predicted by the model.

## Model Analysis

The project explores various deep learning models for sensor-based activity recognition. The models are designed to
process time-series data from multiple sensors and predict the activity label associated with the data. Information
about the models and their architectures can be found in the [Model Analysis](ANALYSIS.md) document.

## Notebooks

The project includes Jupyter Notebooks for exploring the partitioning, preprocessing, and session truncation of the
sensor data. These notebooks provide insights into the data structure and processing steps, aiding in the development of
the data pipeline and feature engineering strategies.

All notebooks are located in the `notebooks/` directory with corresponding names and HTML exports in
the `notebooks/exports/`.

## Codebase

This section provides a detailed breakdown of the Python files within the project's `src/` directory, outlining their
specific roles in data handling, feature extraction, model training, and utility functions.

**Data Handling (`src/data/`)**

- **[`dataset.py`](src/data/dataset.py)**: Implements the `SensorDatasetTorch` and `SensorDataModule` classes to manage
  datasets in PyTorch. It allows for efficient data handling and integration with PyTorch models.
- **[`db.py`](src/data/db.py)**: Contains the `InfluxDBWrapper`, simplifying the process of connecting to and querying
  from an InfluxDB timeseries database.
- **[`label_mapping.py`](src/data/label_mapping.py)**: Manages the conversion of measurement filenames to typed Enums,
  ensuring consistent label handling across the dataset.
- **[`measurement_file.py`](src/data/measurement_file.py)**: Provides structured management and processing of
  measurement data, ensuring data integrity and facilitating the handling of platform-specific variations.

**Feature Extraction (`src/extraction/`)**

- **[`fft.py`](src/extraction/fft.py)**: Focuses on Fourier Transform techniques for signal analysis, extracting key
  frequency-based features like the Dominant Frequency from sensor data.
- **[`moving_average.py`](src/extraction/moving_average.py)**: Implements smoothing techniques using rolling averages,
  enhancing data quality by reducing noise and variations.

**Models (`src/models/`)**

- More detailed information about the models can be found in the [Model Analysis](ANALYSIS.md) document.
- **[`cnn.py`](src/models/cnn.py)**: Defines the Convolutional Neural Network architecture for processing spatially
  correlated data inputs.
- **[`deep_res_bidir_lstm.py`](src/models/deep_res_bidir_lstm.py)**: Implements a Deep Residual Bidirectional LSTM
  model, enhancing the capability to capture both past and future contexts in sequence data.
- **[`log_reg.py`](src/models/log_reg.py)**: Sets up a Logistic Regression model, providing a baseline for performance
  comparison.
- **[`lstm.py`](src/models/lstm.py)**: Configures a standard Long Short-Term Memory network suitable for sequence
  prediction tasks.
- **[`x_lstm.py`](src/models/x_lstm.py)**: Describes an extension of the LSTM model.
- **[`transformer.py`](src/models/transformer.py)**: Details the implementation of the Transformer model, leveraging
  self-attention mechanisms for data processing.

**Processing and Prediction (`src/processing/`)**

- **[`denoising.py`](src/processing/denoising.py)**: Contains methods for signal denoising, such as Butterworth and
  Wavelet Denoising, to improve data quality before feature extraction.
- **[`time_series.py`](src/processing/time_series.py)**: Manages time series data operations like segment cropping and
  resampling, standardizing data inputs for modeling.

**Utility and Management (`src/`)**

- **[`dashboard.py`](src/dashboard.py)**: Implements a Streamlit dashboard for interactive model evaluations and
  visualizations.
- **[`data_pipeline.py`](src/data_pipeline.py)**: Coordinates the flow from data ingestion through preprocessing to
  feature engineering, dynamically configured via Hydra.
- **[`data_zipper.py`](src/data_zipper.py)**: Organizes sensor data into compressed formats for efficient storage and
  retrieval.
- **[`file_import.py`](src/file_import.py)**: Automates the import of sensor data from zip files into the database,
  supporting concurrent processing to enhance efficiency.
- **[`model_pipeline.py`](src/model_pipeline.py)**: Handles the setup and execution of model training and prediction
  workflows, including data loading and logging.
- **[`predict_api.py`](src/predict_api.py)**: Establishes an API using FastAPI for uploading files and retrieving model
  predictions.
- **[`train.py`](src/train.py)**: Outlines the procedure for training models using configurations and data specified.
- **[`utils.py`](src/utils.py)**: Offers miscellaneous functions for environment setup, data path generation, and
  parameter validation.