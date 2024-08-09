import random
from typing import Any, Dict, Type, Tuple, Optional, List

import numpy as np
import pytorch_lightning as pl
import torch
import torchcde
import wandb
import seaborn as sns
import time

from matplotlib import pyplot as plt
from torch import nn
from torch.optim import AdamW, Optimizer
from torchdyn.models import NeuralODE
from src.models.ode.ode_models import (
    Reshape,
    AugmentLatent,
    ProjectLatent,
    ResNet2D,
    ResNet1DT,
)
from src.models.utils import fourier_loss


class NODE1D(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        lr_schedule: str = None,
        weight_decay: float = 1e-3,
        metric=nn.MSELoss,
        mono_loss_weight: float = 0,
        fourier_weight: float = 1,
        plot_interval: int = 20,
    ) -> None:
        super().__init__()

        self.encoder = None
        self.decoder = None
        self.attention = None
        self.vf = None
        self.augment = None
        self.dropout = None
        self.project = None

        self.loss = metric()

    def initialize_lazy(self, input_shape):
        if self.has_lazy_modules:
            with torch.no_grad():
                dummy = torch.ones(2, input_shape)
                dummy_t = torch.ones(100)
                try:
                    _ = self(dummy, dummy_t)  # this initializes the shapes
                except:
                    pass
                # _ = self(dummy)  # this initializes the shapes

    def initialize_weights(self, μ=0, σ=0.1):
        for m in self.vector_field.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=μ, std=σ)
                nn.init.normal_(m.bias, mean=μ, std=σ)

    def forward(
        self, t_0: tuple, t_span: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor]:
        if t_span is not None and isinstance(t_span, torch.Tensor):
            self.t_span = t_span
        z, h = self.encoder(t_0)
        z = self.attention(z)
        z = self.augment(z)
        t, zt = self.ode(z)
        zt = zt[-1, :, :] # get last time step from solver
        zt = self.project(zt)
        # zt = torch.permute(zt, (1, 0, 2))
        # zt = self.dropout(zt)
        d = self.decoder(zt, h)
        return d

    @property
    def has_lazy_modules(self) -> bool:
        for module in self.modules():
            name = module.__class__.__name__
            if "lazy" in name.lower():
                return True
        return False

    @property
    def t_span(self) -> torch.Tensor:
        return self.ode.t_span

    @t_span.setter
    def t_span(self, time_array: torch.Tensor) -> None:
        self.ode.t_span = time_array

    def configure_optimizers(self) -> Optimizer:
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.lr_schedule == "Cyclic":
            num_cycles = 1
            max_steps = 15000
            step_size = max_steps // 2 // num_cycles

            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=1e-4,
                max_lr=1e-2,
                cycle_momentum=False,
                step_size_up=7500,
                step_size_down=7500,
                mode="triangular2"
                # optimizer, base_lr=1e-4, max_lr=1e-2, cycle_momentum=False, step_size_up=5000, step_size_down=5000, mode="triangular2"
                # optimizer, base_lr=1e-4, max_lr=1e-2, cycle_momentum=False, step_size_up=step_size, step_size_down=step_size, mode="triangular2"
            )
            scheduler._scale_fn_custom = scheduler._scale_fn_ref()
            scheduler._scale_fn_ref = None

            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "step",
                "frequency": 1,
            }

        elif self.hparams.lr_schedule == "Step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=680, gamma=0.2
            )

            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }

        elif self.hparams.lr_schedule == "RLROP":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=50,
                factor=0.5,
                verbose=False,
                min_lr=1e-6,
                # optimizer, patience=10, factor=0.5, verbose=True, # original params that worked okay
            )

            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }

        elif self.hparams.lr_schedule == None:
            lr_scheduler = None

        else:
            raise ValueError("Not a valid learning rate scheduler.")

        if lr_scheduler is None:
            return optimizer
        else:
            return [optimizer], [lr_scheduler]

    @property
    def nfe(self) -> int:
        return self.ode.vf.nfe

    @nfe.setter
    def nfe(self, value: int) -> None:
        self.ode.vf.nfe = value

    def step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        X, Y = batch

        t_0 = X
        t = torch.linspace(0, 1, 2)

        pred_Y = self(t_0, t)  # solve the initial value problem

        l1_loss = self.loss(pred_Y, Y)

        loss = l1_loss

        self.log("nfe", torch.tensor(self.nfe, dtype=torch.float32))
        self.nfe = 0  # reset the number of function evaluations

        log = {"loss": loss}

        return loss, log, t, pred_Y, Y, t_0

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, log, t, pred_Y, Y, X = self.step(batch, batch_idx)
        self.log("training_loss", loss, prog_bar=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, log, time, pred_Y, Y, X = self.step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)

        if (
            self.current_epoch > 0
            and self.current_epoch % self.hparams.plot_interval == 0
            and self.epoch_plotted == False
        ):
            self.epoch_plotted = True  # don't plot again in this epoch
            with torch.no_grad():
                fig = self.plot_training_results(pred_Y, Y, X)
                log.update({"g2": fig})
                self.logger.experiment.log(log)

        return loss

    def plot_training_results(self, pred_Y, Y, X):
        pred_Y = pred_Y[0, :, :].cpu()
        Y = Y[0, :, :].cpu()
        X = X[0, :, :-2].cpu()

        fig, ax = plt.subplots(nrows=1, ncols=3, dpi=150, figsize=(6, 2))

        for i, g2 in enumerate(X):
            ax[0].plot(g2, label=f"Input {i}")
        ax[1].imshow(Y)
        ax[2].imshow(pred_Y)

        plt.legend()
        plt.tight_layout()
        wandb.Image(plt)

        return fig

    def on_train_epoch_end(self) -> None:
        self.epoch_plotted = False



class LSTMNODE(NODE1D):
    def __init__(
        self,
        input_size: int = 116,
        enc_hidden_size: int = 256,
        enc_depth: int = 3,
        z_size: int = 44,
        vf_depth: int = 4,
        vf_hidden_size: int = 64,
        attn_depth: int = 2,
        attn_heads: int = 4,
        norm: bool = False,
        dec_hidden_size=None,
        dec_depth=None,
        encoder: Type[nn.Module] = None,
        vector_field: Type[nn.Module] = None,
        decoder: Type[nn.Module] = None,
        attention: Type[nn.Module] = None,
        augment: bool = False,
        augment_dim: int = 0,
        augment_size: int = 1,
        time_dim: int = 1,
        dropout: float = 0.0,
        lr: float = 1e-3,
        lr_schedule: str = None,
        weight_decay: float = 1e-5,
        metric=nn.MSELoss,
        mono_loss_weight: float = 0,
        fourier_weight: float = 1,
        plot_interval: int = 20,
        data_info: dict = None,
        **ode_kwargs,
    ) -> None:
        super().__init__(
            lr,
            lr_schedule,
            weight_decay,
            metric,
            mono_loss_weight,
            fourier_weight,
            plot_interval,
        )

        dec_depth = enc_depth if dec_depth is None else dec_depth
        dec_hidden_size = (
            enc_hidden_size if dec_hidden_size is None else dec_hidden_size
        )

        self.encoder = encoder(
            input_size=input_size + time_dim,
            hidden_size=enc_hidden_size,
            depth=enc_depth,
            output_size=z_size,
        )

        self.attention = (
            attention(
                output_size=z_size,
                depth=attn_depth,
                num_heads=attn_heads,
                norm=norm,
                activation=nn.ReLU,
            )
            if attention is not None
            else nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(z_size),
            )
        )

        self.augment = AugmentLatent(
            augment=augment,
            augment_dim=augment_dim,
            augment_size=augment_size,
            input_size=z_size,
        )

        self.vector_field = vector_field(
            input_size=z_size + int(augment) * augment_size,
            hidden_size=vf_hidden_size,
            output_size=z_size + int(augment) * augment_size,
            depth=vf_depth,
            norm=norm,
        )

        self.ode = NeuralODE(self.vector_field, **ode_kwargs)

        self.project = ProjectLatent(
            project=augment,
            project_dim=augment_dim,
            project_size=augment_size,
            output_size=z_size,
        )

        self.dropout = nn.Dropout(dropout)

        self.decoder = (
            decoder(
                input_size=z_size,
                hidden_size=dec_hidden_size,
                depth=dec_depth,
                output_size=1,
            )
            if decoder is not None
            else nn.Identity()
        )

        self.initialize_lazy(input_size + time_dim)
        self.initialize_weights()
        self.colors = sns.color_palette("icefire", 5)
        self.time_dim = time_dim
        self.save_hyperparameters()


class ConvNODE(NODE1D):
    def __init__(
        self,
        input_size: int = 128,
        enc_depth: int = 2,
        enc_channels: list = None,
        enc_kernels: list = None,
        z_size: int = 2**5,
        vf_depth: int = 4,
        vf_hidden_size: int = None,
        attn_depth: int = 2,
        attn_heads: int = 4,
        dec_depth: int = None,
        vector_field: Type[nn.Module] = None,
        attention: Type[nn.Module] = None,
        augment: bool = False,
        augment_dim: int = 0,
        augment_size: int = 1,
        time_dim: int = 1,
        nobs: int = 10,
        obs_type: str = "fixed",
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        metric=nn.MSELoss,
        slope_weight: float = 1e-2,
        fourier_weight: float = 1,
        plot_interval: int = 20,
        **ode_kwargs,
    ) -> None:
        super().__init__(
            lr, weight_decay, metric, slope_weight, fourier_weight, plot_interval
        )

        enc_channels = [1, 4, 8, 16, 32, 64] if enc_channels is None else enc_channels
        kernels = [3, 3, 3, 3, 3] if enc_kernels is None else enc_kernels
        dec_depth = enc_depth if dec_depth is None else dec_depth
        vf_hidden_size = z_size if vf_hidden_size is None else vf_hidden_size

        self.encoder = ResNet2D(
            depth=enc_depth,
            channels=enc_channels,
            kernels=kernels,
            strides=[2, 2, 2, 2, 2],
            residual=True,
        )

        self.attention = (
            attention(
                output_size=z_size,
                depth=attn_depth,
                num_heads=attn_heads,
            )
            if attention is not None
            else nn.Identity()
        )

        self.augment = AugmentLatent(
            augment=augment,
            augment_dim=augment_dim,
            augment_size=augment_size,
            input_size=z_size,
        )

        self.vector_field = vector_field(
            input_size=z_size + int(augment) * augment_size,
            hidden_size=vf_hidden_size,
            depth=vf_depth,
        )

        self.ode = NeuralODE(self.vector_field, **ode_kwargs)

        self.project = ProjectLatent(
            project=augment,
            project_dim=augment_dim,
            project_size=augment_size,
            output_size=z_size,
        )

        self.dropout = nn.Dropout(dropout)

        ratio = input_size // z_size
        dec_strides = [1, 1, 1, 1, 1]
        for i in range(int(np.log2(ratio))):
            dec_strides[i] = 2

        self.decoder = ResNet1DT(
            depth=dec_depth,
            channels=[100, 100, 100, 100, 100, 100, 100],
            kernels=[3, 3, 3, 3, 3],
            strides=dec_strides,
            sym_residual=False,
            fwd_residual=True,
        )

        self.initialize_lazy((nobs, input_size + time_dim))
        self.initialize_weights()
        self.loss = nn.MSELoss()
        self.colors = sns.color_palette("icefire", 5)
        self.time_dim = time_dim
        self.nobs = nobs
        self.obs_type = obs_type
        self.save_hyperparameters()


if __name__ == "__main__":
    from g2_pcfs.models.ode.ode_models import (
        AttentionBlock,
        LSTMEncoder,
        LSTMDecoder,
        MLPStack,
    )

    model = LSTMNODE(
        input_size=120,
        # input_size=140,
        enc_hidden_size=16,
        enc_depth=2,
        z_size=2**3,
        vf_depth=2,
        vf_hidden_size=16,
        attn_depth=1,
        attn_heads=2,
        norm=False,
        encoder=LSTMEncoder,
        vector_field=MLPStack,
        attention=AttentionBlock,
        decoder=LSTMDecoder,
        time_dim=0,
        augment=True,
        augment_size=1,
        atol=1e-4,
        rtol=1e-4,
        dropout=0.0,
        lr=5e-4,
        lr_schedule="RLROP",
        weight_decay=1e-5,
        plot_interval=3,
    )

    input = torch.rand(10, 120)
    t = torch.linspace(0, 1, 2)
    output = model(input, t)