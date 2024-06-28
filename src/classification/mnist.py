import argparse

import lightning as L
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner.tuning import Tuner
from torch import nn
from torchmetrics import functional as F_metrics

from datasets.mnist.datamodules import get_mnist_variant


SEED = 0

class MNISTClassifierLightningModule(L.LightningModule):

    NUM_CLASSES = 10

    def __init__(
        self,
        learning_rate: float = 1e-3,
        data_dir: str = "src/datasets/mnist/files/raw",
        seed: int = 0,
        batch_size: int = 128,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.NUM_CLASSES),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def predict_step(self, batch):
        return self.model(batch["image"]).softmax(dim=1)  # .argmax(dim=1)

    def _step(self, batch, prefix):
        images, labels = batch["image"], batch["label"].float()
        logits = self.model(images)
        loss = self.loss_fn(logits, labels)
        acc = F_metrics.accuracy(logits.argmax(dim=1), labels.argmax(dim=1), task="multiclass", num_classes=self.NUM_CLASSES)
        metrics = {f"{prefix}/loss": loss, f"{prefix}/accuracy": acc}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return metrics

    def training_step(self, batch):
        metrics = self._step(batch, "train")
        return metrics["train/loss"]

    def validation_step(self, batch):
        self._step(batch, "val")

    def test_step(self, batch):
        self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
    
    def setup(self, stage):
        self.dm = get_mnist_variant(
            "grey",
            self.hparams.seed,
            self.hparams.batch_size,
            num_workers=0,
        )
        self.dm.setup(stage)

    def train_dataloader(self):
        return self.dm.train_dataloader()
    
    def val_dataloader(self):
        return self.dm.val_dataloader()
    
    def test_dataloader(self):
        return self.dm.test_dataloader()
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--val-freq", type=int, default=20)
    parser.add_argument("--logdir", type=str, default="/vol/biomedic3/rrr2417/cxr-generation/experiments/experiments_mnist_classifier")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    seed_everything(SEED)

    tb_logger = TensorBoardLogger(save_dir=args.logdir)
    checkpoint_callback = ModelCheckpoint(save_last=True)
    trainer = Trainer(
        accelerator="gpu",
        devices=args.device_id,
        deterministic=True,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.val_freq,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=1,
    )

    # TODO: Make data_dir some kind of global variable passed in via poetry
    data_dir = "src/datasets/mnist/files/raw"
    model = MNISTClassifierLightningModule(
        data_dir=data_dir,
        seed=SEED,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )

    trainer.fit(model)
    trainer.test(model)
