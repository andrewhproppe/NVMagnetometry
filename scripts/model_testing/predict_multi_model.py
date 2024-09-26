import torch

from pytorch_lightning import seed_everything
from src.pipeline.data_module import DataModule
from src.models.base import NODE_MLP
from src.utils import paths, get_system_and_backend
from src.visualization.fig_utils import *
get_system_and_backend()
from matplotlib import pyplot as plt

if __name__ == "__main__":
    seed_everything(42, workers=True)

    B_max = 10

    dm = DataModule(
        # "n20000_0_to_10_snr20_long.h5",
        # "n20000_0_to_10_snr50_long.h5",
        # "n20000_0_to_10_snr100_long.h5",
        # "n20000_0_to_10_snr20.h5",
        # "n20000_0_to_10_snr50.h5",
        "n20000_0_to_10_snr100.h5",
        batch_size=500,
        num_workers=0,
        pin_memory=True,
        split_type="fixed",
        norm=True,
        B_max=10.0,
        # log_scale=True,
        # concat_log=concat_log,
    )
    dm.setup()
    X, Y = next(iter(dm.test_dataloader()))


    model = NODE_MLP.load_from_checkpoint(
        # checkpoint_path=paths.get("trained_models").joinpath("solar-elevator-119.ckpt"), # 20 dB, 4 us
        # checkpoint_path=paths.get("trained_models").joinpath("warm-cloud-120.ckpt"), # 50 dB, 4 us
        # checkpoint_path=paths.get("trained_models").joinpath("vibrant-blaze-122.ckpt"), # 100 dB, 4 us
        # checkpoint_path=paths.get("trained_models").joinpath("true-eon-113.ckpt"), # 20 dB, 0.2 us
        # checkpoint_path=paths.get("trained_models").joinpath("dandy-pine-115.ckpt"), # 50 dB, 0.2 us
        checkpoint_path=paths.get("trained_models").joinpath("zesty-breeze-111.ckpt"), # 100 dB, 0.2 us
    ).eval()

    X = X.to(model.device)
    Y = Y.to(model.device)

    with torch.no_grad():
        Y_hat = model(X)

    Y *= B_max
    Y_hat *= B_max

    set_font_size(8)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=150)
    ax.plot(Y.cpu(), Y_hat.cpu(), "o", ms=2)

    x = np.linspace(0, 1, 201) * B_max
    y = x
    ax.plot(x, y, "k--", lw=1.0)

    abs_loss = torch.abs(Y - Y_hat.squeeze(-1)).mean()

    ax.set_xlabel("True B")
    ax.set_ylabel("Predicted B")
    ax.text(0.1, 0.9, f"Mean Abs Loss: {abs_loss:.2e}", transform=ax.transAxes)

    dress_fig()