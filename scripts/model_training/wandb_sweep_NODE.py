import wandb
import torch
import os

from torch import nn
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging
from src.pipeline.data_module import DataModule
from src.models.base import NODE_MLP

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

project_name = "NVMagnetometry_sweep"

sweep_config = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "decoder_depth": {"values": [2, 3, 4, 5, 6]},
        "hidden_size": {"values": [32, 64, 128, 256, 512, 1024]},
        # "dropout": {"values": [0.0, 0.1, 0.2]},
        "activation": {"values": ["ReLU", "SiLU", "PReLU", "LeakyReLU", "GELU"]},
        # "weight_decay": {"values": [1e-7, 1e-6, 1e-5, 1e-4]},
        "norm": {"values": [True, False]},
        "vf_depth": {"values": [2, 3, 4, 5]},
        "vf_hidden_size": {"values": [32, 64, 128, 256, 512, 1024]},
    },
}

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

def train():
    # Default hyperparameters
    config_defaults = dict(
        input_size=2000,
        hidden_size=128,
        output_size=1,
        decoder_depth=3,
        vf_depth=3,
        vf_hidden_size=128,
        dropout=0.,
        lr=1e-3,
        lr_schedule="RLROP",
        weight_decay=1e-5,
        activation="LeakyReLU",
        plot_interval=100,
        data_info=dm.header
    )

    # Initialize a new wandb run
    wandb.init(
        config=config_defaults,
        project=project_name,
        entity="aproppe",
        mode="online",
        # mode="offline",
    )

    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config

    model = NODE_MLP(**config)

    logger = WandbLogger(log_model="False", save_code="False")

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(
        max_epochs=350,
        # max_steps=10000,
        logger=logger,
        enable_checkpointing=False,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=[2],
        gradient_clip_val=1.0,
        callbacks=[
            lr_monitor,
            # StochasticWeightAveraging(swa_lrs=1e-8, swa_epoch_start=0.9)
        ],
        deterministic=True,
    )

    trainer.fit(model, dm)

    return trainer, logger

sweep_id = wandb.sweep(sweep_config, project=project_name)

wandb.agent(sweep_id, function=train, count=100)
