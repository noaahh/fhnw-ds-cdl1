CONDA_ENV_NAME=cdl1
IMPORT_DATA=src/file_import.py

setup:
	conda create --name $(CONDA_ENV_NAME) python=3.11
	conda activate $(CONDA_ENV_NAME)
	pip install -r requirements.txt

import-latest:
	docker-compose up -d influxdb
	python $(IMPORT_DATA) --verbose

teardown-influxdb:
	docker-compose down influxdb --volumes --remove-orphans

stop-influxdb:
	docker-compose stop influxdb

clean:
	rm -rf outputs/*
	rm -rf $(CONDA_ENV_NAME)/*
	rm -rf data/cache/*

run-dashboard:
	streamlit run src/dashboard.py

run-api:
	fastapi run src/predict_api.py


.PHONY: setup import-latest teardown-influxdb stop-influxdb clean run-dashboard run-api
