import torch

from torch import nn
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging
from src.pipeline.data_module import DataModule
from src.models.base import MLP

if __name__ == "__main__":
    seed_everything(42, workers=True)

    dm = DataModule(
        # "trainset_20240731_n5000_0_to_10.h5",
        "n5000_0_to_10_snr100.h5",
        batch_size=256,
        num_workers=0,
        pin_memory=True,
        split_type="fixed",
        norm=True,
        B_max=10.0,
    )

    model = MLP(
        input_size=201,
        hidden_size=1024,
        output_size=1,
        depth=4,
        dropout=0.,
        norm=False,
        lr=1e-6,
        lr_schedule="RLROP",
        weight_decay=1e-5,
        activation="LeakyReLU",
        output_activation="Sigmoid",
        plot_interval=25,
        metric=nn.L1Loss,
        data_info=dm.header
    )

    logger = WandbLogger(
        entity="aproppe",
        project="NVMagnetometry",
        # mode="offline",
        mode="online",
        # log_model=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        # max_epochs=1000,
        max_steps=50000,
        logger=logger,
        # enable_checkpointing=False,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=[3],
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            StochasticWeightAveraging(swa_lrs=1e-8, swa_epoch_start=0.8)
        ],
        gradient_clip_val=1.0,
        deterministic=True,
    )

    trainer.fit(model, datamodule=dm)
