import torch

from torch import nn
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging
from src.pipeline.data_module import DataModule
from src.models.base import AttnGRU, ConvGRU, GRU_NODE_MLP, NODE_MLP
from src.utils import paths

if __name__ == "__main__":
    seed_everything(42, workers=True)

    concat_log = False
    # start_stop_idx = (700, 1500)
    # input_size = start_stop_idx[1] - start_stop_idx[0]

    dm = DataModule(
        # "n5000_0_to_10_snr100.h5",
        "n20000_0_to_10_snr100.h5",
        # "n5000_0_to_10_snr20.h5",
        # "n5000_0_to_10_snr100_long.h5",
        # "n5000_0_to_10_snr20_long.h5",
        batch_size=500,
        num_workers=0,
        pin_memory=True,
        split_type="fixed",
        norm=True,
        B_max=10.0,
        # log_scale=True,
        # concat_log=concat_log,
        # start_stop_idx=start_stop_idx
    )

    model = NODE_MLP(
        input_size=201,
        # input_size=2000,
        # input_size=input_size,
        hidden_size=64,
        vf_depth=5,
        vf_hidden_size=256,
        output_size=1,
        dropout=0.,
        lr=1e-3,
        lr_schedule="RLROP",
        lr_patience=50,
        weight_decay=1e-5,
        activation="SiLU",
        decoder_depth=2,
        plot_interval=500,
        metric=nn.L1Loss,
        data_info=dm.header
    )


    # checkpoint_path = paths.get("trained_models").joinpath("twilight-sweep-29.ckpt")
    # model = NODE_MLP.load_from_checkpoint(
    #     # checkpoint_path=paths.get("trained_models").joinpath("dulcet-sweep-4.ckpt"),
    #     checkpoint_path=checkpoint_path,
    #     # checkpoint_path=paths.get("trained_models").joinpath("serene-sweep-1.ckpt"),
    # )
    # model.epoch_plotted = False
    # raise RuntimeError
    # 1
    # optimizers, schedulers = model.configure_optimizers()
    #
    # new_learning_rate = 1e-8
    #
    # # Update the learning rate in the optimizer(s)
    # for optimizer in optimizers:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = new_learning_rate
    #
    # # Optionally, update the learning rate in the scheduler(s) if needed
    # for scheduler in schedulers:
    #     if isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
    #         scheduler.base_lrs = [new_learning_rate for _ in scheduler.base_lrs]


    logger = WandbLogger(
        entity="aproppe",
        project="NVMagnetometry",
        # mode="offline",
        mode="online",
        # log_model=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        max_epochs=5000,
        # max_steps=50000,
        logger=logger,
        # enable_checkpointing=False,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=[3],
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            # StochasticWeightAveraging(swa_lrs=1e-8, swa_epoch_start=0.8)
        ],
        gradient_clip_val=1.0,
        deterministic=True,
    )

    trainer.fit(
        model,
        datamodule=dm,
        # ckpt_path=checkpoint_pat
    )