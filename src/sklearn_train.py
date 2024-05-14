import hydra
import numpy as np
import rootutils
from omegaconf import DictConfig, OmegaConf
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.dataset import SensorDatasetSKLearn
from src.utils import get_partition_paths, get_partitioned_data


def train_and_evaluate_model(train_dataset, val_dataset, model: BaseEstimator, metrics: dict):
    X_train, y_train = train_dataset.get_data()
    X_val, y_val = val_dataset.get_data()

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    return {name: metric(y_val, predictions) for name, metric in metrics.items()}


def cross_validate(data, model: BaseEstimator, metrics: dict, k_folds: int):
    evaluation_results = []

    for k in tqdm(range(k_folds), desc="Processing folds", unit="fold"):
        fold = data[k]
        train_data, validate_data = fold['train'], fold['validate']

        train_dataset = SensorDatasetSKLearn(train_data)
        validate_dataset = SensorDatasetSKLearn(validate_data)

        results = train_and_evaluate_model(train_dataset, validate_dataset, model, metrics)
        evaluation_results.append(results)

        fold_results = ", ".join([f"{name}: {score:.2f}" for name, score in results.items()])
        print(f"Fold {k + 1}: {fold_results}")

    average_scores = {name: np.mean([result[name] for result in evaluation_results]) for name in metrics.keys()}
    avg_results = ", ".join([f"Average {name}: {score:.2f}" for name, score in average_scores.items()])

    print(f"Cross-validation ({k_folds} folds) results:")
    print(avg_results)


@hydra.main(version_base="1.3", config_path="../configs", config_name="sklearn_train.yaml")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    if cfg.get("seed"):
        np.random.seed(cfg.seed)

    k_folds = cfg.data.get("k_folds", None)
    data = get_partitioned_data(get_partition_paths(k_folds=k_folds))

    model = hydra.utils.instantiate(cfg.model)

    metrics = {
        'accuracy': accuracy_score,
        'f1': lambda y, pred: f1_score(y, pred, average='macro')
    }

    if k_folds:
        cross_validate(data, model, metrics, k_folds)
    else:
        train_dataset = SensorDatasetSKLearn(data['train'])
        val_dataset = SensorDatasetSKLearn(data['validate'])

        results = train_and_evaluate_model(train_dataset, val_dataset, model, metrics)
        results_str = ", ".join([f"{name}: {score:.2f}" for name, score in results.items()])
        print(f"Results: {results_str}")

    if cfg.get("ckpt_path"):
        raise NotImplementedError("Checkpointing not implemented yet")


if __name__ == "__main__":
    main()
