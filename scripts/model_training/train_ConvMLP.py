import torch

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging
from src.pipeline.data_module import DataModule
from src.models.base import ConvMLP

if __name__ == "__main__":
    seed_everything(42, workers=True)

    dm = DataModule(
        "trainset_20240715_n2560_sparsity0.9.h5",
        batch_size=32,
        num_workers=0,
        pin_memory=True,
        split_type="random",
        norm=False,
    )

    model = ConvMLP(
        input_size=702,
        channels=32,
        depth=4,
        kernels=[5, 3, 3, 3, 3],
        downsample=16,
        MLP_hidden_size=64,
        MLP_output_size=36,
        MLP_depth=3,
        dropout=0.,
        norm=True,
        lr=5e-4,
        # lr_scheduler="RLROP",
        weight_decay=1e-6,
        activation="LeakyReLU",
        plot_interval=10,
        data_info=dm.header
    )

    logger = WandbLogger(
        entity="aproppe",
        project="PsiPost2Beta",
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
        devices=[2],
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
        ],
        gradient_clip_val=1.0,
        deterministic=True,
    )

    trainer.fit(model, datamodule=dm)
