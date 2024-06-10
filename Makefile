CONDA_ENV_NAME=cdl1

IMPORT_DATA=src/import.py
DATA_PIPELINE=src/data_pipeline.py

RUN_PIPELINE ?= 1
EXPERIMENT_NAME ?= log_reg

setup:
	conda create --name $(CONDA_ENV_NAME) python=3.11
	conda activate $(CONDA_ENV_NAME)
	pip install -r requirements.txt

import-latest:
	docker-compose up -d influxdb
	python $(IMPORT_DATA) --verbose

run-experiment:
ifeq ($(RUN_PIPELINE),1)
	python $(DATA_PIPELINE) experiment=$(EXPERIMENT_NAME)
endif
	python src/train.py experiment=$(EXPERIMENT_NAME)

teardown-influxdb:
	docker-compose down influxdb --volumes --remove-orphans

stop-influxdb:
	docker-compose stop influxdb

clean:
	rm -rf outputs/*
	rm -rf $(CONDA_ENV_NAME)/*
	rm -rf data/cache/*

run-dashboard:
	streamlit run src/dashboard.py                                                                                                                                      INT 8m 19s py cdl1 23:20:34

run-api:
	fastapi run src/predict_api.py


.PHONY: setup import-latest run-experiment teardown-influxdb stop-influxdb clean run-dashboard run-api
