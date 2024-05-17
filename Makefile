CONDA_ENV_NAME=cdl1

DATA_PIPELINE=src/data_pipeline.py
TRAIN_MODEL_TORCH=src/train_torch.py
TRAIN_MODEL_SKLEARN=src/train_sklearn.py

setup:
	conda create --name $(CONDA_ENV_NAME) python=3.11
	conda activate $(CONDA_ENV_NAME)
	pip install -r requirements.txt

pipeline:
	python $(DATA_PIPELINE)

train-torch:
	python $(TRAIN_MODEL_TORCH)

train-sklearn:
	python $(TRAIN_MODEL_SKLEARN)

clean:
	rm -rf outputs/*
	rm -rf lightning_logs/*
	rm -rf cdl1/*
	rm -rf wandb/*

.PHONY: setup activate data train-torch train-sklearn train-all clean
