import torch

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging
from src.pipeline.data_module import DataModule
from src.models.base import MLP

if __name__ == "__main__":
    seed_everything(42, workers=True)

    dm = DataModule(
        # "trainset_20240710_n2560.h5",
        "trainset_20240715_n2560_sparsity0.9.h5",
        batch_size=32,
        num_workers=0,
        pin_memory=True,
        split_type="random",
        norm=False,
    )

    model = MLP(
        input_size=201,
        hidden_size=128,
        output_size=1,
        depth=3,
        dropout=0.,
        norm=True,
        lr=1e-3,
        # lr_scheduler="RLROP",
        weight_decay=1e-5,
        activation="LeakyReLU",
        output_activation="ReLU",
        plot_interval=10,
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
        # max_epochs=1000,
        max_steps=50000,
        logger=logger,
        # enable_checkpointing=False,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
        ],
        gradient_clip_val=1.0,
        deterministic=True,
    )

    trainer.fit(model, datamodule=dm)
