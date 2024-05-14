import hydra
import lightning as L
import rootutils
from lightning import LightningModule, Trainer
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.sensor_datamodule import SensorDataModule
from src.utils import setup_logging


@hydra.main(version_base="1.3", config_path="../configs", config_name="lightning_train.yaml")
def train(cfg) -> None:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log = setup_logging()

    print(OmegaConf.to_yaml(cfg))

    dataset = SensorDataModule(batch_size=cfg.data.batch_size,
                               num_workers=cfg.data.get("num_workers", None),
                               pin_memory=cfg.data.get("pin_memory", True))
    dataset.setup()

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    for key, fold in dataset.data_dict.items():
        train_dataloader, val_dataloader = fold['train'], fold['validate']

        log.info(f"Instantiating model <{cfg.model._target_}> for fold {key}")
        model: LightningModule = hydra.utils.instantiate(cfg.model)

        trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    train()
