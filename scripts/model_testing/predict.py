import torch

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging
from src.pipeline.data_module import DataModule
from src.models.base import ConvMLP, MLP, AttnLSTM
from src.utils import paths, get_system_and_backend
from matplotlib import pyplot as plt
get_system_and_backend()

if __name__ == "__main__":
    seed_everything(42, workers=True)

    B_max = 10

    dm = DataModule(
        # "trainset_20240731_n5000_0_to_10.h5",
        "n5000_0_to_10_snr100.h5",
        batch_size=256,
        num_workers=0,
        pin_memory=True,
        split_type="fixed",
        norm=True,
        B_max=B_max,
    )

    dm.setup()
    X, Y = next(iter(dm.val_dataloader()))

    # #
    # model = MLP.load_from_checkpoint(
    #     checkpoint_path=paths.get("trained_models").joinpath("usual-grass-15.ckpt"),
    #     # checkpoint_path=paths.get("trained_models").joinpath("eternal-snowball-18.ckpt"),
    # ).eval()

    model = AttnLSTM.load_from_checkpoint(
        checkpoint_path=paths.get("trained_models").joinpath("celestial-darkness-28.ckpt"),
    ).eval()

    X = X.to(model.device)
    Y = Y.to(model.device)

    with torch.no_grad():
        Y_hat = model(X)

    Y *= B_max
    Y_hat *= B_max

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(Y.cpu(), Y_hat.cpu(), "o", ms=2)
