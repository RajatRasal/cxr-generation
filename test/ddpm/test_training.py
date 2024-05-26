import os
import shutil
from typing import List

import pytest
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from ddpm.training.train import DiffusionLightningModule


def _reset(folder: str):
    """
    Delete the contents of the directory if it exists.
    """
    if os.path.exists(folder):
        for root, dirs, files in os.walk(folder):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
    else:
        raise ValueError(f"{folder} does not exist")


def _callbacks(folder: str, filename: str) -> List[Callback]:
    return [ModelCheckpoint(dirpath=folder, filename=filename)]


def _logger(folder: str, exp_name: str, seed: int) -> TensorBoardLogger:
    return TensorBoardLogger(
        save_dir=folder,
        name=exp_name,
        version=seed,
        default_hp_metric=False,
    )


def _trainer(callbacks: List[Callback], logger: TensorBoardLogger) -> Trainer:
    return Trainer(
        accelerator="gpu",
        max_epochs=1000,
        check_val_every_n_epoch=100,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        logger=logger,
    )


@pytest.mark.dependency(name="training")
def test_training(seed, model_folder_and_name_and_exp):
    # Setup directories
    model_folder, model_name, exp_name = model_folder_and_name_and_exp
    _reset(model_folder)

    # Load model
    model = DiffusionLightningModule(
        dim=20,
        dim_mults=[1],
        channels=1,
        train_timesteps=1000,
        sample_timesteps=50,
        dataset_size=20000,
        beta_schedule="cosine",
        sanity_check=True,
        folder=model_folder,
        dataset_seed=seed,
    )

    # Define Trainer
    callbacks = _callbacks(model_folder, model_name)
    logger = _logger(model_folder, exp_name, seed)
    trainer = _trainer(callbacks, logger)
    
    # Train model
    trainer.fit(model)

    # TODO: Gather training loss and check and write assertion
    # TODO: Gather val and train nlls and check also and write assertion
    # TODO: Assert contents of tensorboard dirs, logging dirs, and model checkpoint


@pytest.mark.dependency(depends=["training"])
def test_testing(seed, model_folder_and_name_and_exp):
    # Setup directories
    model_folder, model_name, exp_name = model_folder_and_name_and_exp
    path = os.path.join(model_folder, f"{model_name}.ckpt")

    # Load model
    model = DiffusionLightningModule.load_from_checkpoint(path)

    # Define Trainer
    callbacks = _callbacks(model_folder, model_name)
    logger = _logger(model_folder, exp_name, seed)
    trainer = _trainer(callbacks, logger)

    # Test model
    trainer.test(model)
