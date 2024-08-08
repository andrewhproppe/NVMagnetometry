import torch

from torch import nn
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging
from src.pipeline.data_module import DataModule
from src.models.base import ConvMLP

if __name__ == "__main__":
    seed_everything(42, workers=True)

    dm = DataModule(
        # "n5000_0_to_10_snr100.h5",
        # "n5000_0_to_10_snr20.h5",
        "n5000_0_to_10_snr100_long.h5",
        # "n5000_0_to_10_snr20_long.h5",
        batch_size=256,
        num_workers=0,
        pin_memory=True,
        split_type="fixed",
        norm=True,
        B_max=10.0,
    )

    model = ConvMLP(
        # input_size=201,
        input_size=2000,
        channels=128,
        depth=5,
        kernels=[5, 3, 3, 3, 3],
        downsample=16,
        MLP_hidden_size=128,
        MLP_output_size=1,
        MLP_depth=3,
        dropout=0.,
        norm=False,
        lr=1e-6,
        lr_schedule="RLROP",
        weight_decay=1e-5,
        activation="PReLU",
        plot_interval=1000,
        metric=nn.L1Loss,
        data_info=dm.header
    )

    logger = WandbLogger(
        entity="aproppe",
        project="NVMagnetometry",
        mode="offline",
        # mode="online",
        # log_model=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        max_epochs=200,
        # max_steps=50000,
        logger=logger,
        # enable_checkpointing=False,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=[0],
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            StochasticWeightAveraging(swa_lrs=1e-6, swa_epoch_start=0.8)
        ],
        gradient_clip_val=1.0,
        deterministic=True,
    )

    trainer.fit(model, datamodule=dm)
