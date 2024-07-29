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
from src.models.base import MonotonicLoss
from src.models.utils import fourier_loss


class g2ODE(pl.LightningModule):
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
        self.mono_loss = MonotonicLoss(dimension=2)

    def initialize_lazy(self, input_shape):
        if self.has_lazy_modules:
            with torch.no_grad():
                dummy = torch.ones(2, *input_shape)
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
        zt = self.project(zt)
        zt = torch.permute(zt, (1, 0, 2))
        # zt = torch.cat((_zt, t.unsqueeze(0).repeat(_zt.shape[0], 1).unsqueeze(-1)), dim=2)
        zt = self.dropout(zt)
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
        X = batch.get("input")
        Y = batch.get("target")
        t = batch.get("ode_time")

        if self.obs_type == "fixed":
            obs_inds = torch.tensor(
                [0, 1, 2, 3, 6, 8, 14, 18, 22, 30, 38, 46, 54, 62, 70, 80, 90]
            )
            obs_inds = obs_inds[: self.nobs]
        elif self.obs_type == "random":
            obs_inds = torch.randperm(X.shape[1])[: self.nobs].sort()[0]

        # self.t_span = t
        t_obs = t[obs_inds].unsqueeze(0).repeat(X.shape[0], 1).unsqueeze(-1)
        _t_0 = X[:, obs_inds, :]

        if self.time_dim == 1:
            t_0 = torch.cat((_t_0, t_obs), dim=2)
        else:
            t_0 = _t_0

        pred_Y = self(t_0, t)  # solve the initial value problem

        recon_loss = self.loss(pred_Y, Y)
        # slope_loss = torch.sum(torch.nn.functional.relu(torch.diff(pred_Y * -1)))
        mono_loss = self.mono_loss(pred_Y)
        fourier = fourier_loss(pred_Y, Y)

        recon_loss /= self.hparams.nobs
        mono_loss /= self.hparams.nobs
        fourier /= self.hparams.nobs
        loss = (
            recon_loss
            + self.hparams.fourier_weight * fourier
            + self.hparams.mono_loss_weight * mono_loss
        )

        self.log("nfe", torch.tensor(self.nfe, dtype=torch.float32))
        self.log("num_points", torch.tensor(len(obs_inds), dtype=torch.float32))
        self.nfe = 0  # reset the number of function evaluations

        log = {
            "loss": loss,
            "recon": recon_loss,
            "fourier": fourier,
            "slope": mono_loss,
        }

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


class g2LSTMODE(g2ODE):
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
        nobs: int = 10,
        obs_type: str = "fixed",
        dropout: float = 0.0,
        lr: float = 1e-3,
        lr_schedule: str = None,
        weight_decay: float = 1e-5,
        metric=nn.MSELoss,
        mono_loss_weight: float = 1,
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

        self.decoder = (
            decoder(
                input_size=z_size,
                hidden_size=dec_hidden_size,
                depth=dec_depth,
                output_size=input_size,
            )
            if decoder is not None
            else nn.Identity()
        )

        self.initialize_lazy((nobs, input_size + time_dim))
        self.initialize_weights()
        self.colors = sns.color_palette("icefire", 5)
        self.time_dim = time_dim
        self.nobs = nobs
        self.obs_type = obs_type
        self.save_hyperparameters()


class g2ConvODE(g2ODE):
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


class F(pl.LightningModule):
    def __init__(
        self,
        nlayers: int = 1,
        input_channels: int = 118,
        hidden_channels: int = 250,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        norm: Optional[Type[nn.Module]] = nn.LayerNorm,
    ):
        super(F, self).__init__()

        activation = nn.Identity if activation is None else activation
        norm = nn.Identity if norm is None else nn.LayerNorm

        layers = []
        for i in range(nlayers):
            layers.append(torch.nn.Linear(hidden_channels, hidden_channels))
            layers.append(activation())
            layers.append(norm(hidden_channels))

        layers.append(
            torch.nn.Linear(hidden_channels, hidden_channels * input_channels)
        )
        layers.append(nn.Tanh())

        self.layers = torch.nn.Sequential(*layers)
        self.save_hyperparameters()

    def forward(self, t, z):
        batch_dims = z.shape[:-1]
        return self.layers(z).view(
            *batch_dims, self.hparams.hidden_channels, self.hparams.input_channels
        )


class g2NCDE(pl.LightningModule):
    def __init__(
        self,
        ode_model: Type[nn.Module],
        model_kwargs: Dict[str, Any],
        nlayers: int = 1,
        input_channels=118,
        hidden_channels=250,
        input_shape: Optional[List[int]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        plot_interval: int = 20,
        slope_weight: float = 1e-2,
        **ode_kwargs,
    ) -> None:
        super().__init__()

        self.encoder = torch.nn.Linear(input_channels, hidden_channels)

        self.vector_field = F(
            nlayers=nlayers,
            input_channels=input_channels,
            hidden_channels=hidden_channels,
        )

        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(1, 32, kernel_size=(6, 9), stride=(4, 2), output_padding=(0, 0)),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(32, 32, kernel_size=(5, 9), stride=(4, 2), output_padding=(0, 0)),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(32, 32, kernel_size=(3, 9), stride=(4, 1), output_padding=(0, 0)),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(32, 1, kernel_size=(2, 8), stride=(1, 1), output_padding=(0, 0)),
        # )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, 116 * 100),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(116 * 100),
            Reshape(1, 100, 116),
        )

        # self.vector_field = ode_model(**model_kwargs)
        self.model_kwargs = model_kwargs
        self.initialize_weights()
        self.loss = nn.MSELoss()
        self.colors = sns.color_palette("icefire", 5)
        self.save_hyperparameters()

    def initialize_lazy(self, input_shape):
        if self.has_lazy_modules:
            with torch.no_grad():
                dummy = torch.ones(2, *input_shape)
                _ = self.vector_field(dummy)  # this initializes the shapes

    def initialize_weights(self, μ=0, σ=0.1):
        for m in self.vector_field.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=μ, std=σ)
                nn.init.normal_(m.bias, mean=μ, std=σ)

    # def forward(self, coeffs, t) -> torch.Tensor:
    #     """
    #     Forward pass of the model.
    #     Args:
    #         x_ (torch.Tensor): Array of shape (batch_size, 1+n_channels+1, n_points). The two additional channels are for time and mask for missing observation (nan) cumulative sum.
    #         t (torch.Tensor): Time array.
    #     """
    #     X = torchcde.CubicSpline(coeffs, t)
    #     X0 = X.evaluate(X.interval[0])
    #     z0 = self.encoder(X0)
    #     zt = torchcde.cdeint(X=X, func=self.vector_field, z0=z0, t=t)
    #     return self.decoder(zt)

    def forward(self, coeffs, t) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x_ (torch.Tensor): Array of shape (batch_size, 1+n_channels+1, n_points). The two additional channels are for time and mask for missing observation (nan) cumulative sum.
            t (torch.Tensor): Time array.
        """
        X = torchcde.CubicSpline(coeffs)
        X0 = X.evaluate(X.interval[0])
        z0 = self.encoder(X0)
        zt = torchcde.cdeint(
            X=X, func=self.vector_field, z0=z0, t=X.interval, adjoint=False
        )
        zt = zt[..., -1, :]  # get the terminal value of the CDE
        zt = zt.view(
            zt.shape[0], 1, 1, zt.shape[-1]
        )  # prepare for convolutional decoder
        return self.decoder(zt).squeeze(1)

    @property
    def has_lazy_modules(self) -> bool:
        for module in self.modules():
            name = module.__class__.__name__
            if "lazy" in name.lower():
                return True
        return False

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

        elif self.hparams.lr_schedule == "RLROP":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=50,
                factor=0.5,
                verbose=False,
                min_lr=1e-5,
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
        return self.model.vf.nfe

    @nfe.setter
    def nfe(self, value: int) -> None:
        self.model.vf.nfe = value

    def step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        tic = time.time()
        X = batch.get("input")
        Y = batch.get("target")

        t = batch.get("ode_time")
        coeffs = batch.get("spline_coeffs")
        pred_Y = self(coeffs, t)

        recon_loss = self.loss(pred_Y, Y)
        # slope_loss = torch.sum(torch.nn.functional.relu(torch.diff(pred_Y*-1)))

        # recon_loss /= nobs
        # slope_loss /= nobs
        # loss = recon_loss + self.hparams.slope_weight*slope_loss

        loss = recon_loss
        # self.log("nfe", self.nfe)
        # self.log("num_points", coeffs.shape[0])
        # self.nfe = 0 # reset the number of function evaluations

        log = {
            "loss": loss,
            # "recon": recon_loss,
            # "slope": slope_loss,
        }

        # print(f"batch {batch_idx} took {time.time()-tic:.2f}s")

        return loss, log, pred_Y, Y, X

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, log, pred_Y, Y, X = self.step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)

        if (
            self.current_epoch > 0
            and self.current_epoch % self.hparams.plot_interval == 0
        ):
            with torch.no_grad():
                fig = self.plot_training_results(pred_Y, Y, X)
                log.update({"g2": fig})
                # self.logger.experiment.log(log)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, log, pred_Y, Y, X = self.step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)

        return loss

    def plot_training_results(self, pred_Y, Y, X):
        pred_Y = pred_Y[0, :, :].cpu()
        Y = Y[0, :, :].cpu()
        X = X[0, :, :].cpu()

        fig, ax = plt.subplots(nrows=1, ncols=2, dpi=150, figsize=(5, 2.5))

        ax[0].imshow(Y)
        ax[1].imshow(pred_Y)

        # fig = plt.figure(dpi=150, figsize=(5, 2.5))
        # spacer = 0.2
        # idxs = torch.randperm(sliced_Y.shape[1])[:4].sort()[0]
        # for i, idx in enumerate(idxs):
        #     plt.scatter(x_axis, sliced_Y[:, idx]+i*spacer, marker='s', s=3, color=self.colors[i], label="Target %i" %i)
        #     plt.plot(x_axis, pred_y[:, idx]+i*spacer, lw=2, color=self.colors[i])
        # plt.xlabel('log(τ)')
        plt.legend()
        plt.tight_layout()
        wandb.Image(plt)

        return fig


if __name__ == "__main__":
    import wandb
    from pytorch_lightning.loggers import WandbLogger
    from src.pipeline.image_data import NCDEDataModule

    pl.seed_everything(42)

    nchannels = 5  # number of g2s to use for solving initial value problem
    dim = 116
    prior = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dm = ODEDataModule("pcfs_g2_2d_n128_wPrior.h5", batch_size=32, nchannels=nchannels, nsteps=(15, 25), add_noise=False)
    # dm = NCDEDataModule("pcfs_g2_2d_n1_wPrior_ncde.h5", batch_size=1, nchannels=nchannels, nsteps=(15, 25), add_noise=False)
    # dm = NCDEDataModule("ncde_g2_n20.h5", batch_size=10, nchannels=nchannels, nsteps=(15, 25), add_noise=False)
    dm = NCDEDataModule("ncde_g2_n1.h5", batch_size=1, add_noise=False)

    model_kwargs = {
        # "input_channels": 116,
        # "hidden_channels": 200,
        # "dim": dim,
        # "MLP_dim": 250,
        # "nchannels": nchannels,
        # "activation": nn.ReLU
    }

    ode = g2NCDE(
        ode_model=F,
        model_kwargs=model_kwargs,
        nlayers=5,
        hidden_channels=20,
        # sensitivity='adjoint',
        input_shape=[100, 116],
        atol=1e-4,
        rtol=1e-4,
        lr=1e-3,
        plot_interval=100,
    )

    # logger = WandbLogger(
    #     entity="aproppe",
    #     save_dir='/Users/andrewproppe/Desktop/g2-pcfs_backup/wandb_garbage',
    #     project='g2-pcfs-nODE',
    #     log_model=False,
    #     save_code=False,
    #     offline=False,
    # )

    trainer = pl.Trainer(
        max_epochs=100,
        # gpus=int(torch.cuda.is_available()),
        # logger=logger,
        # profiler="simple",
        enable_checkpointing=False,
    )

    trainer.fit(
        ode,
        datamodule=dm,
    )

    # Look at output
    batch = next(iter(dm.train_dataloader()))
    coeffs = batch.get("spline_coeffs")
    t = batch.get("ode_time")
    target = batch.get("target")
    X = torchcde.CubicSpline(coeffs, t)
    interp_g2s = X.evaluate(t)[0, :, 1:-1]
    with torch.no_grad():
        pred = ode(coeffs, batch.get("ode_time"))
    pred = pred.detach().numpy()

    fig, ax = plt.subplots(nrows=1, ncols=2, dpi=150, figsize=(5, 2.5))
    ax[0].imshow(target[0, :, :])
    ax[1].imshow(pred[0, :, :])
