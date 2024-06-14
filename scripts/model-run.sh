#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <base_path>"
  exit 1
fi

BASE_PATH="$1"
EXPERIMENTS=(bidir_lstm cnn log_reg lstm transformer x_lstm)

# Regular runs
python "$BASE_PATH/src/data_pipeline.py"
for EXP in "${EXPERIMENTS[@]}"; do
    python "$BASE_PATH/src/train.py" experiment=$EXP
    python "$BASE_PATH/src/train.py" experiment=$EXP refit_on_all_data=True
done

# Runs with k-folds
python "$BASE_PATH/src/data_pipeline.py" partitioning.k_folds=5
for EXP in "${EXPERIMENTS[@]}"; do
    python "$BASE_PATH/src/train.py" experiment=$EXP partitioning.k_folds=5
done