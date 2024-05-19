CONDA_ENV_NAME=cdl1

IMPORT_DATA=src/import.py
DATA_PIPELINE=src/data_pipeline.py
TRAIN_MODEL_TORCH=src/train_torch.py
TRAIN_MODEL_SKLEARN=src/train_sklearn.py

EXPERIMENT_NAME ?= log_reg
FRAMEWORK ?= sklearn

setup:
	conda create --name $(CONDA_ENV_NAME) python=3.11
	conda activate $(CONDA_ENV_NAME)
	pip install -r requirements.txt

import-latest:
	docker-compose up -d influxdb
	python $(IMPORT_DATA) --verbose

run-experiment:
	python $(DATA_PIPELINE) experiment=$(EXPERIMENT_NAME)
ifeq ($(FRAMEWORK),torch)
	python $(TRAIN_MODEL_TORCH) experiment=$(EXPERIMENT_NAME)
else
	python $(TRAIN_MODEL_SKLEARN) experiment=$(EXPERIMENT_NAME)
endif

teardown-influxdb:
	docker-compose down influxdb --volumes --remove-orphans

clean:
	rm -rf outputs/*
	rm -rf $(CONDA_ENV_NAME)/*
	rm -rf data/cache/*

.PHONY: setup import-latest run-experiment teardown-influxdb clean
