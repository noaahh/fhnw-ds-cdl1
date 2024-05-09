from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.data.dataset import SensorDatasetSKLearn
from src.data.partition_helper import get_partitioned_data, get_partition_paths

partitions_paths = get_partition_paths("./data/splits", k_folds=5)
data = get_partitioned_data(partitions_paths)

evaluation_results = []

for fold_idx, fold in tqdm(enumerate(data), desc="Processing folds", unit="fold"):
    train_data, validate_data = fold['train'], fold['validate']
    train_dataset = SensorDatasetSKLearn(train_data)
    validate_dataset = SensorDatasetSKLearn(validate_data)

    X_train, y_train = train_dataset.get_data()
    X_validate, y_validate = validate_dataset.get_data()

    model = LogisticRegression(random_state=1337, max_iter=1000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_validate)
    accuracy = accuracy_score(y_validate, predictions)
    f1 = f1_score(y_validate, predictions, average='macro')

    evaluation_results.append({
        'accuracy': accuracy,
        'f1': f1
    })
    print(f"Fold {fold_idx + 1} accuracy: {accuracy:.2f}, f1: {f1:.2f}")

average_accuracy = sum([r['accuracy'] for r in evaluation_results]) / len(evaluation_results)
average_f1 = sum([r['f1'] for r in evaluation_results]) / len(evaluation_results)
print(f"Average accuracy: {average_accuracy:.2f}, average f1: {average_f1:.2f}")
