import wandb
import torch
import os

from torch import nn
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging, ModelCheckpoint
from src.pipeline.data_module import DataModule
from src.models.base import NODE_MLP
from src.models.utils import LossThresholdEarlyStopping

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

project_name = "NVMagnetometry_sweep"

sweep_config = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "decoder_depth": {"values": [2, 3, 4, 5]},
        "hidden_size": {"values": [64, 128, 256, 512]},
        "activation": {"values": ["SiLU", "PReLU", "LeakyReLU", "GELU"]},
        "vf_depth": {"values": [4, 5, 6]},
        "vf_hidden_size": {"values": [256, 512, 1024]},
        "lr": {"values": [1e-3, 5e-4, 1e-4]},
        "lr_patience": {"values": [50, 60, 70]},
    },
}

seed_everything(42, workers=True)

dm = DataModule(
    "n5000_0_to_10_snr100.h5",
    # "n5000_0_to_10_snr20.h5",
    # "n5000_0_to_10_snr100_long.h5",
    # "n5000_0_to_10_snr20_long.h5",
    batch_size=500,
    num_workers=0,
    pin_memory=True,
    split_type="fixed",
    norm=True,
    B_max=10.0,
)


def train():
    # Default hyperparameters
    config_defaults = dict(
        input_size=201,
        # input_size=2000,
        hidden_size=128,
        output_size=1,
        decoder_depth=3,
        vf_depth=3,
        vf_hidden_size=128,
        dropout=0.,
        lr=1e-3,
        lr_schedule="RLROP",
        lr_patience=60,
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

    logger = WandbLogger(
        mode='online',
        log_model="True",
        # save_code="True"
    )
    #
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=os.path.join(wandb.run.dir, "checkpoints"),  # Save checkpoints in the W&B run directory
    #     save_top_k=-1,  # Save all checkpoints (optional: adjust to save only the best N checkpoints)
    #     every_n_epochs=10,  # Save a checkpoint every 10 epochs
    #     filename="{epoch:02d}-{val_loss:.2f}",  # Filename format (optional)
    #     save_weights_only=False,  # Save the entire model, not just weights
    # )

    trainer = Trainer(
        max_epochs=3500,
        # max_steps=10000,
        logger=logger,
        # enable_checkpointing=False,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=[3],
        gradient_clip_val=1.0,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            LossThresholdEarlyStopping(monitor="val_loss", threshold=0.4, patience=20),
            # checkpoint_callback
            # StochasticWeightAveraging(swa_lrs=1e-8, swa_epoch_start=0.9)
        ],
        deterministic=True,
    )

    trainer.fit(model, dm)

    return trainer, logger

sweep_id = wandb.sweep(sweep_config, project=project_name)

wandb.agent(sweep_id, function=train, count=100)
