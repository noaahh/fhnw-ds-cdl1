import logging
from pathlib import Path

import rootutils
import torch
from hydra import compose, initialize
from lightning import LightningModule, Trainer

from src.data.label_mapping import LABEL_MAPPING
from src.models.deep_res_bidir_lstm import DeepResBidirLSTM

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.dataset import create_data_loader
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
    "DeepResBidirLSTM": DeepResBidirLSTM
}

CHECKPOINTS = {
    "MulticlassLogisticRegression": "/Users/noah/dev/bsc/cdl1/outputs/2024-06-10/16-06-02/epoch=17-val_loss=0.52-val_f1=0.84-val_acc=0.83.ckpt",
    "DeepResBidirLSTM": "/Users/noah/dev/bsc/cdl1/outputs/2024-06-06/11-22-52/fold_1/epoch=20-val_loss=0.44-val_f1=0.88-val_acc=0.87.ckpt"
}


def load_model(model_name):
    model_class = LIGHTNING_MODULES.get(model_name)

    if model_class is None:
        raise ValueError("Model class not found.")

    if not issubclass(model_class, LightningModule):
        raise ValueError("Model class must be a subclass of LightningModule.")

    return model_class.load_from_checkpoint(CHECKPOINTS.get(model_name))


@torch.no_grad()
def run_model_pipeline(measurement_file_path: Path,
                       model_name: str,
                       batch_size: int,
                       verbose: bool):
    setup_logging(verbose)
    logger.info("Starting model pipeline execution...")
    logger.debug(f"File path: {measurement_file_path}")

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="pipeline.yaml",
                      overrides=[f"+measurement_file_path={measurement_file_path}"])
        segments_df = pipeline(cfg)

    data_loader = create_data_loader(segments_df,
                                     batch_size=batch_size,
                                     pin_memory=False,
                                     persistent_workers=False,
                                     shuffle=False)

    model = load_model(model_name)
    print(model)
    trainer = Trainer()

    predictions = trainer.predict(model, data_loader)
    predictions = torch.cat(predictions, dim=0).tolist()

    reverse_label_mapping = {v: k for k, v in LABEL_MAPPING.items()}

    predicted_labels = [reverse_label_mapping[pred] for pred in predictions]
    logger.debug(f"Predictions: {predicted_labels}")

    majority_label = max(set(predicted_labels), key=predicted_labels.count)
    logger.info(f"Majority label: {majority_label}")

    logger.info("Model pipeline execution completed.")

    return majority_label
