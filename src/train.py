import os

import hydra
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
import wandb
import gc
import torch

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import get_env_variable, setup_logging

log = setup_logging()


def get_output_dir_path(cfg, k_fold):
    return os.path.join(cfg.paths.output_dir, f"fold_{k_fold}") if k_fold is not None else cfg.paths.output_dir


def instantiate_datamodule(cfg, fold_index=None):
    if fold_index is not None:
        data_module = hydra.utils.instantiate(cfg.data,
                                              k_folds=cfg.partitioning.get("k_folds", 1),
                                              partitioned_data_dir=cfg.paths.partitioned_data_dir)
        data_module.set_current_fold(fold_index)
        return data_module
    else:
        return hydra.utils.instantiate(cfg.data, k_folds=None,
                                       partitioned_data_dir=cfg.paths.partitioned_data_dir)


def instantiate_trainer(cfg, logger, callbacks, k_fold=None):
    return hydra.utils.instantiate(cfg.trainer,
                                   logger=logger,
                                   callbacks=callbacks,
                                   default_root_dir=get_output_dir_path(cfg, k_fold),
                                   _convert_="partial")


def create_callbacks(cfg, k_fold=None):
    return [ModelCheckpoint(monitor="val_f1",
                            mode="max",
                            dirpath=get_output_dir_path(cfg, k_fold),
                            filename="{epoch}-{val_loss:.2f}-{val_f1:.2f}-{val_acc:.2f}",
                            save_last=True,
                            verbose=False)]


def create_logger(cfg, group_id, k_fold=None):
    return WandbLogger(project=get_env_variable("WANDB_PROJECT"),
                       entity=get_env_variable("WANDB_ENTITY"),
                       group=group_id,
                       save_dir=get_output_dir_path(cfg, k_fold),
                       log_model="all")


def seed_everything(seed):
    if seed:
        L.seed_everything(seed, workers=True)


def manage_k_folds(cfg):
    k_folds = cfg.partitioning.get("k_folds")
    if k_folds is None or k_folds == 1:
        return [None]
    return range(k_folds)


def log_hyperparameters(cfg, model, trainer):
    hparams = {}

    cfg = OmegaConf.to_container(cfg)

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
    hparams["partitioning"] = cfg["partitioning"]
    hparams["preprocessing"] = cfg["preprocessing"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def train(cfg):
    seed_everything(cfg.get("seed"))
    print(OmegaConf.to_yaml(cfg))

    uses_k_folds = cfg.partitioning.get("k_folds") is not None and cfg.partitioning.get("k_folds") > 1
    group_id = wandb.util.generate_id() if uses_k_folds else None
    if uses_k_folds:
        log.info(f"Training {cfg.partitioning.get('k_folds')} folds")
        log.info(f"CV Wandb Group ID: {group_id}")

        if cfg.ckpt_path is not None:
            raise ValueError("Checkpointing is not supported with k-folds")

    for fold_index in manage_k_folds(cfg):
        run_name = f"{group_id}_fold_{fold_index + 1}" if uses_k_folds else None
        with wandb.init(project=get_env_variable("WANDB_PROJECT"),
                        group=group_id,
                        name=run_name):
            k_fold = fold_index + 1 if uses_k_folds else None

            datamodule = instantiate_datamodule(cfg, fold_index)
            datamodule.setup(stage='fit')

            model = hydra.utils.instantiate(cfg.model)
            logger = create_logger(cfg, group_id, k_fold)
            callbacks = create_callbacks(cfg, k_fold)
            trainer = instantiate_trainer(cfg, logger, callbacks, k_fold)

            log_hyperparameters(cfg, model, trainer)

            train_dataloader, val_dataloader = datamodule.train_dataloader(), datamodule.val_dataloader()
            trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=cfg.get("ckpt_path"))

            if fold_index is not None:
                log.info(f"Training complete for {'fold ' + str(fold_index + 1)}")

            gc.collect()
            torch.cuda.empty_cache()

    log.info("Training complete")


if __name__ == "__main__":
    train()
