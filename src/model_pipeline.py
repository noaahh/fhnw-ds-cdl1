import logging
import tempfile
from pathlib import Path

import rootutils
import torch
import wandb
from hydra import compose, initialize
from lightning import LightningModule, Trainer

from src.models.x_lstm import XLSTM

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.cnn import Simple1DCNN
from src.models.lstm import SimpleLSTM
from src.models.transformer import TransformerClassifier
from src.data.label_mapping import LABEL_MAPPING
from src.models.deep_res_bidir_lstm import DeepResBidirLSTM
from src.data.dataset import SensorDataModule
from src.data_pipeline import pipeline
from src.models.log_reg import MulticlassLogisticRegression

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool):
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    logger.disabled = not verbose


LIGHTNING_MODULES = {
    "MulticlassLogisticRegression": MulticlassLogisticRegression,
    "DeepResBidirLSTM": DeepResBidirLSTM,
    "Transformer": TransformerClassifier,
    "SimpleLSTM": SimpleLSTM,
    "CNN": Simple1DCNN,
    "xLSTM": XLSTM
}


def load_model(model_name, local_checkpoint_path=None, wandb_artifact_path=None):
    model_class = LIGHTNING_MODULES.get(model_name)

    if model_class is None:
        raise ValueError("Model class not found.")

    if not issubclass(model_class, LightningModule):
        raise ValueError("Model class must be a subclass of LightningModule.")

    if local_checkpoint_path:
        checkpoint_path = local_checkpoint_path
    elif wandb_artifact_path:
        api = wandb.Api()
        artifact = api.artifact(wandb_artifact_path)
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = artifact.download(temp_dir)
            checkpoint_path = Path(artifact_dir) / "model.ckpt"
            model = model_class.load_from_checkpoint(checkpoint_path)
            return model, Path(temp_dir)
    else:
        raise ValueError("Either a local checkpoint path or wandb run and checkpoint path must be provided.")

    try:
        model = model_class.load_from_checkpoint(checkpoint_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

    return model, Path(checkpoint_path).parent


@torch.no_grad()
def predict_file(measurement_file_path: Path,
                 model_name: str,
                 batch_size: int,
                 verbose: bool,
                 local_checkpoint_path: str = None,
                 wandb_artifact_path: str = None):
    setup_logging(verbose)
    logger.info("Starting model pipeline execution...")
    logger.debug(f"File path: {measurement_file_path}")

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="pipeline.yaml",
                      overrides=[f"+measurement_file_path={measurement_file_path}"])
        segments_df = pipeline(cfg)

    data_loader = SensorDataModule.create_dataloader(segments_df,
                                                     batch_size=batch_size,
                                                     num_workers=0,
                                                     pin_memory=False,
                                                     persistent_workers=False,
                                                     shuffle=False)

    trainer = Trainer()
    model, ckpt_dir = load_model(model_name, local_checkpoint_path, wandb_artifact_path)
    predictions = trainer.predict(model, data_loader)
    predictions = torch.cat(predictions, dim=0).tolist()

    reverse_label_mapping = {v: k for k, v in LABEL_MAPPING.items()}
    predicted_labels = [reverse_label_mapping[pred] for pred in predictions]
    logger.debug(f"Predictions: {predicted_labels}")

    majority_label = max(set(predicted_labels), key=predicted_labels.count)
    logger.info(f"Majority label: {majority_label}")

    logger.info("Model pipeline execution completed.")

    return majority_label
