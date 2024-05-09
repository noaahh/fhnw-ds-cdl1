import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.data.dataset import SensorDatasetSKLearn
from src.data.partition_helper import get_partitioned_data, get_partition_paths


def train_and_evaluate_model(data_partitions, model: BaseEstimator, metrics: dict):
    evaluation_results = []
    for fold_idx, fold in tqdm(enumerate(data_partitions), desc="Processing folds", unit="fold"):
        train_data, validate_data = fold['train'], fold['validate']

        train_dataset = SensorDatasetSKLearn(train_data)
        validate_dataset = SensorDatasetSKLearn(validate_data)
        X_train, y_train = train_dataset.get_data()
        X_validate, y_validate = validate_dataset.get_data()

        model.fit(X_train, y_train)
        predictions = model.predict(X_validate)

        results = {name: metric(y_validate, predictions) for name, metric in metrics.items()}
        evaluation_results.append(results)

        fold_results = ", ".join([f"{name}: {score:.2f}" for name, score in results.items()])
        print(f"Fold {fold_idx + 1} {fold_results}")

    average_scores = {name: np.mean([result[name] for result in evaluation_results]) for name in metrics.keys()}
    avg_results = ", ".join([f"Average {name}: {score:.2f}" for name, score in average_scores.items()])
    print(avg_results)


data = get_partitioned_data(get_partition_paths("./data/partitions", k_folds=5))

model = LogisticRegression(random_state=1337, max_iter=1000)
metrics = {
    'accuracy': accuracy_score,
    'f1': lambda y, pred: f1_score(y, pred, average='macro')
}

train_and_evaluate_model(data, model, metrics)
