import torch

from torch import nn
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging
from src.pipeline.data_module import DataModule
from src.models.base import AttnLSTM, VisionTransformer1D

if __name__ == "__main__":
    seed_everything(42, workers=True)

    concat_log = False

    dm = DataModule(
        # "trainset_20240731_n5000_0_to_10.h5",
        # "n5000_0_to_10_snr100.h5",
        "n5000_0_to_10_snr20.h5",
        batch_size=256,
        num_workers=0,
        pin_memory=True,
        split_type="fixed",
        norm=True,
        B_max=10.0,
        log_scale=True,
        concat_log=concat_log,
    )

    model = VisionTransformer1D(
        input_size=200 * (1 + concat_log),
        patch_size=20,
        num_layers=8,
        num_heads=8,
        output_dim=1,
        mlp_dim=512,
        hidden_dim=256,
        dropout=0.,
        lr=1e-3,
        lr_schedule="RLROP",
        weight_decay=1e-5,
        activation="GELU",
        plot_interval=25,
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
        max_epochs=550,
        max_steps=50000,
        logger=logger,
        # enable_checkpointing=False,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=[0],
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            StochasticWeightAveraging(swa_lrs=1e-8, swa_epoch_start=0.8)
        ],
        gradient_clip_val=1.0,
        deterministic=True,
    )

    trainer.fit(model, datamodule=dm)