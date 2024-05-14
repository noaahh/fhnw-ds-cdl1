import hydra
import lightning as L
import rootutils
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.sensor_datamodule import SensorDataModule
from src.utils import get_env_variable, setup_logging

from omegaconf import OmegaConf

log = setup_logging()


def log_hyperparameters(object_dict):
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def create_logger(cfg, group_id):
    return WandbLogger(project=get_env_variable("WANDB_PROJECT"),
                       entity=get_env_variable("WANDB_ENTITY"),
                       group=group_id,
                       log_model=True)


def create_callbacks(cfg):
    return [ModelCheckpoint(monitor="val_acc",
                            mode="max",
                            dirpath=cfg.get("ckpt_path"),
                            verbose=True)]


def seed_everything(seed):
    if seed:
        L.seed_everything(seed, workers=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="lightning_train.yaml")
def train(cfg):
    seed_everything(cfg.get("seed"))

    print(OmegaConf.to_yaml(cfg))

    k_folds = int(cfg.data.get("k_folds")) if cfg.data.get("k_folds") else None
    if k_folds is not None:
        raise NotImplementedError("K-Fold training is not yet supported")

    datamodule = SensorDataModule(batch_size=cfg.data.batch_size,
                                  num_workers=cfg.data.get("num_workers", None),
                                  pin_memory=cfg.data.get("pin_memory", True))
    datamodule.setup(stage='fit')

    model = hydra.utils.instantiate(cfg.model)

    callbacks = create_callbacks(cfg)
    logger = create_logger(cfg, group_id=None)

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks,
        _convert_="partial"
    )

    log_hyperparameters({
        "cfg": cfg,
        "model": model,
        "trainer": trainer,
        "datamodule": datamodule
    })

    train_dataloader, val_dataloader = datamodule.train_dataloader(), datamodule.val_dataloader()
    trainer.fit(model, train_dataloader, val_dataloader)

    log.info("Training complete!")


if __name__ == "__main__":
    train()
