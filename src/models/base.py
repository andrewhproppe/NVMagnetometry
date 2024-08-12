from abc import abstractmethod
from typing import Tuple, Union, Optional
from argparse import ArgumentParser
from copy import deepcopy
from itertools import chain
from src.models.submodels import *

import numpy as np
import torch
import random
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import wandb

import matplotlib
matplotlib.use("Agg")  # this forces a non-X server backend
from matplotlib import pyplot as plt

from src.models.utils import BetaRateScheduler, format_time_sequence, init_rnn, init_fc_layers, get_conv_output_size, get_conv_output_shape, get_conv1d_flat_shape
# from src.utils import get_system_and_backend
# get_system_and_backend()

"""
Notes on style

1. Use capitals to denote what are meant to be tensors, excluding batches
2. Use `black` for code formatting
3. Use NumPy style docstrings
"""

def common_parser():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--encoder_num_layers", type=int, default=3)
    parser.add_argument("--decoder_num_layers", type=int, default=1)
    parser.add_argument("--bidirectional", type=bool, default=False)
    parser.add_argument("--hidden_dim", type=int, default=8)
    parser.add_argument("--input_dropout", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    return parser


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


class BaseModel(pl.LightningModule):
    """
    Base autoencoder model. encoder and decoder are assigned in the child class. recoder is usually just nn.Identity(),
    but can be used to perform transformations on the latent space.
    """
    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        metric=nn.MSELoss,
        plot_interval: int = 1000,
        lr_schedule: str = None,
    ) -> None:
        super().__init__()
        self.encoder = None
        self.decoder = None
        self.recoder = nn.Identity()
        self.metric = metric()
        self.save_hyperparameters()

    def encode(self, X: torch.Tensor):
        """Encodes input into latent space"""
        return self.encoder(X)

    def recode(self, X: torch.Tensor):
        """Perform transformations on the latent space"""
        return self.recoder(X)

    def decode(self, Z: torch.Tensor):
        """Decodes latent space into output"""
        return self.decoder(Z)

    def forward(self, X: torch.Tensor):
        X = self.encode(X)
        X = self.recode(X)
        X = self.decode(X)
        return X

    def step(self, batch, batch_idx):
        X, Y = batch
        pred_Y = self(X)

        l1_loss = self.metric(pred_Y, Y.unsqueeze(-1))
        # mse_loss = F.mse_loss(pred_Y, Y.unsqueeze(-1))
        abs_loss = F.l1_loss(pred_Y, Y.unsqueeze(-1))

        loss = l1_loss #+ mse_loss

        log = ({
            "loss": loss,
            "abs_loss": abs_loss,
         })

        return loss, log, X, Y, pred_Y

    def training_step(self, batch, batch_idx):
        loss, log, X, Y, pred_Y = self.step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, log, X, Y, pred_Y = self.step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        if (
            self.current_epoch > 0
            and self.current_epoch % self.hparams.plot_interval == 0
            and self.epoch_plotted == False
        ):
            self.epoch_plotted = True  # don't plot again in this epoch
            fig = self.plot_training_results(X, Y, pred_Y)
            log.update({"plot": fig})
        self.logger.experiment.log(log)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

        if self.hparams.lr_schedule == 'Cyclic':
            num_cycles = 1
            max_steps = 15000
            step_size = max_steps // 2 // num_cycles

            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=1e-4, max_lr=1e-2, cycle_momentum=False, step_size_up=7500, step_size_down=7500, mode="triangular2"
                # optimizer, base_lr=1e-4, max_lr=1e-2, cycle_momentum=False, step_size_up=5000, step_size_down=5000, mode="triangular2"
                # optimizer, base_lr=1e-4, max_lr=1e-2, cycle_momentum=False, step_size_up=step_size, step_size_down=step_size, mode="triangular2"
            )
            scheduler._scale_fn_custom = scheduler._scale_fn_ref()
            scheduler._scale_fn_ref = None

            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "step",
                "frequency": 1
            }

        elif self.hparams.lr_schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=680, gamma=0.2
            )

        elif self.hparams.lr_schedule == 'RLROP':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=15, factor=0.5, verbose=False, min_lr=1e-10, eps=1e-12
                # optimizer, patience=10, factor=0.5, verbose=True, # original params that worked okay
            )

            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }


        elif self.hparams.lr_schedule == None:
            lr_scheduler = None

        else:
            raise ValueError("Not a valid learning rate scheduler.")

        if lr_scheduler is None:
            return optimizer
        else:
            return [optimizer], [lr_scheduler]

    def on_train_epoch_end(self) -> None:
        self.epoch_plotted = False

    # @plot_with_agg_backend
    def plot_training_results(self, X, Y, pred_Y):
        Y = Y.cpu()
        pred_Y = pred_Y.cpu()

        x = np.linspace(0, 1, 201)
        y = x

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot(Y.cpu(), pred_Y.cpu(), "o", ms=2)
        ax.plot(x, y, "k--", lw=1.0)

        plt.tight_layout()

        wandb.Image(plt)
        plt.close()

        return fig

    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("AutoEncoder")
        group.add_argument("--lr", type=float, default=1e-3)
        group.add_argument("--weight_decay", type=float, default=0.0)
        return parent_parser

    def _init_lazy(self, input_shape):
        # if self.has_lazy_modules:
        with torch.no_grad():
            dummy = torch.ones(*input_shape)
            _ = self(dummy)  # this initializes the shapes

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                # Use He initialization for convolutional layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

                # Zero-initialize the biases
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.Linear):
                # Use He initialization for linear layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

                # Zero-initialize the biases
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # Initialize BatchNorm with small positive values to prevent division by zero
                nn.init.normal_(module.weight, mean=1, std=0.02)
                nn.init.constant_(module.bias, 0)

            elif isinstance(module, (nn.LSTM, nn.GRU)):
                # Initialize the weights for LSTM and GRU layers
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        # Input-hidden weights: Xavier initialization
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        # Hidden-hidden weights: Orthogonal initialization
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        # Set forget gate bias to 1 for LSTM (if applicable)
                        nn.init.constant_(param.data, 0)
                        if isinstance(module, nn.LSTM):
                            # LSTM has 4 gates: the forget gate is the second gate (bias[hidden_size:2*hidden_size])
                            hidden_size = param.data.shape[0] // 4
                            param.data[hidden_size:2 * hidden_size] = 1.0

    def count_parameters(self):
        return sum(param.numel() for param in self.parameters())


class MLP(BaseModel):
    """
    Phase Retrieval U-Net (PRUNe). 3D ResNet encoder, 2D ResNet decoder
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        depth: int = 3,
        dropout: float = 0.0,
        activation="ReLU",
        output_activation="ReLU",
        norm=True,
        residual: bool = True,
        lr: float = 2e-4,
        lr_schedule: str = None,
        weight_decay: float = 1e-5,
        metric=nn.MSELoss,
        plot_interval: int = 10,
        data_info: dict = None,
    ) -> None:
        super().__init__(lr, weight_decay, metric, plot_interval, lr_schedule)

        try:
            activation = getattr(nn, activation)
        except:
            activation = activation

        try:
            output_activation = getattr(nn, output_activation)
        except:
            output_activation = output_activation

        self.encoder = MLPStack(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            depth=depth,
            dropout=dropout,
            activation=activation,
            output_activation=output_activation,
            norm=norm,
            residual=residual,
            lazy=False,
        )

        self.decoder = nn.Identity()

        self._init_weights()
        self.save_hyperparameters()


class ConvMLP(BaseModel):
    """
    Phase Retrieval U-Net (PRUNe). 3D ResNet encoder, 2D ResNet decoder
    """
    def __init__(
        self,
        input_size,
        channels: int = 16,
        depth: int = 3,
        kernels: Union[int, Tuple[int]] = 3,
        downsample: int = 8,
        MLP_hidden_size: int = 64,
        MLP_output_size: int = 36,
        MLP_depth: int = 3,
        dropout: float = 0.0,
        activation="ReLU",
        output_activation="ReLU",
        norm=True,
        residual: bool = True,
        lr: float = 2e-4,
        lr_schedule: str = None,
        weight_decay: float = 1e-5,
        metric=nn.MSELoss,
        plot_interval: int = 10,
        data_info: dict = None
    ) -> None:
        super().__init__(lr, weight_decay, metric, plot_interval, lr_schedule)

        try:
            activation = getattr(nn, activation)
        except:
            activation = activation

        try:
            output_activation = getattr(nn, output_activation)
        except:
            output_activation = output_activation

        channels = [1] + [channels] * depth if isinstance(channels, int) else channels

        strides = [2 if i < int(np.log2(downsample)) else 1 for i in range(depth)]

        self.encoder = AttnResNet1d(
            depth=depth,
            channels=channels,
            kernels=kernels,
            strides=strides,
            dropout=dropout,
            activation=activation,
            norm=norm,
            residual=residual,
        )

        self.recoder = nn.Flatten()

        self.decoder = MLPStack(
            input_size=None,
            hidden_size=MLP_hidden_size,
            output_size=MLP_output_size,
            depth=MLP_depth,
            dropout=dropout,
            activation=activation,
            output_activation=output_activation,
            norm=norm,
            residual=residual,
            lazy=True,
        )

        self._init_lazy((2, input_size))
        self._init_weights()
        self.save_hyperparameters()

    def forward(self, X: torch.Tensor):
        if X.ndim == 2:
            X = X.unsqueeze(1)
        X, _ = self.encode(X)
        X = self.recode(X)
        X = self.decode(X)
        return X


class AttnLSTM(BaseModel):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.0,
        activation: str = "ReLU",
        bidirectional: bool = False,
        attn_on: bool = False,
        attn_layers: int = 1,
        attn_heads: int = 4,
        attn_norm: bool = False,
        lr: float = 1e-3,
        lr_schedule: str = None,
        weight_decay: float = 0.0,
        metric=nn.L1Loss,
        plot_interval: int = 1000,
        data_info: dict = None
    ) -> None:
        super().__init__(lr, weight_decay, metric, plot_interval, lr_schedule)

        self.bidirectional = bidirectional

        self.attention = (
            AttentionBlock(
                output_size=input_size,
                depth=attn_layers,
                num_heads=attn_heads,
                norm=attn_norm,
            )
            if attn_on else nn.Identity()
        )

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Adjust the hidden size for bidirectional LSTM
        if bidirectional:
            hidden_size *= 2

        try:
            self.activation = getattr(nn, activation)()
        except AttributeError:
            raise ValueError(f"Activation function '{activation}' is not valid.")

        self.decoder = nn.Linear(hidden_size, output_size)

        self.save_hyperparameters()

    def forward(self, X: torch.Tensor):
        X = self.attention(X)
        X, _ = self.encoder(X)
        X = self.activation(X)  # Apply activation function to the last time step output
        X = self.decode(X)  # Decode to the output
        return X


class SelfAttention1D(nn.Module):
    def __init__(
        self,
        input_size: int,
        embedding_size: int,
        output_size: int,
        num_heads: int = 4,
        activation: str = "ReLU",
        norm: bool = False,
        dropout: float = 0.0
    ):
        super(SelfAttention1D, self).__init__()

        # Linear layers for query, key, and value projections
        self.query = nn.Linear(input_size, embedding_size)
        self.key = nn.Linear(input_size, embedding_size)
        self.value = nn.Linear(input_size, embedding_size)

        # Multi-head self-attention mechanism
        self.attention = nn.MultiheadAttention(embedding_size, num_heads, dropout=dropout, batch_first=True)

        # Optional normalization
        self.norm = nn.LayerNorm(embedding_size) if norm else nn.Identity()

        # Activation function
        try:
            self.activation = getattr(nn, activation)()
        except AttributeError:
            raise ValueError(f"Activation function '{activation}' is not valid.")

        # Output projection
        self.output = nn.Linear(embedding_size, output_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # Project the input to query, key, and value
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Apply self-attention
        attn_output, _ = self.attention(Q, K, V)

        # Apply normalization, if enabled
        attn_output = self.norm(attn_output)

        # Apply activation function
        attn_output = self.activation(attn_output)

        # Apply dropout
        attn_output = self.dropout(attn_output)

        # Project to output size
        output = self.output(attn_output)

        return output


class AttnGRU(BaseModel):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.0,
        activation: str = "ReLU",
        bidirectional: bool = False,
        attn_on: bool = False,
        attn_heads: int = 4,
        attn_downsample: int = 2,
        attn_norm: bool = False,
        decoder_depth: int = 3,
        lr: float = 1e-3,
        lr_schedule: str = None,
        weight_decay: float = 0.0,
        metric=nn.L1Loss,
        plot_interval: int = 1000,
        data_info: dict = None
    ) -> None:
        super().__init__(lr, weight_decay, metric, plot_interval, lr_schedule)

        self.bidirectional = bidirectional

        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Adjust the hidden size for bidirectional GRU
        if bidirectional:
            decoder_hidden_size = hidden_size * 2
        else:
            decoder_hidden_size = hidden_size

        self.attention = (
            SelfAttention1D(
                input_size=decoder_hidden_size,
                embedding_size=decoder_hidden_size//attn_downsample,
                output_size=decoder_hidden_size,
                num_heads=attn_heads,
                activation="ReLU",
                norm=attn_norm,
                dropout=0.0
            )
            if attn_on else nn.Identity()
        )

        try:
            self.activation = getattr(nn, activation)()
        except AttributeError:
            raise ValueError(f"Activation function '{activation}' is not valid.")

        # self.decoder = nn.Linear(decoder_hidden_size, output_size)

        self.decoder = MLPStack(
            input_size=decoder_hidden_size,
            hidden_size=decoder_hidden_size,
            output_size=output_size,
            depth=decoder_depth,
            norm=False,
            residual=True,
            lazy=False,
        )

        # self._init_lazy((2, input_size))
        # self._init_weights()
        self.save_hyperparameters()

    def forward(self, X: torch.Tensor):
        X, _ = self.encoder(X)
        X = self.activation(X)  # Apply activation function to the last time step output
        X = self.attention(X)
        X = self.decoder(X)  # Decode to the output
        return X


class GRU_NODE_MLP(BaseModel):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.0,
        activation: str = "ReLU",
        bidirectional: bool = False,
        decoder_depth: int = 3,
        lr: float = 1e-3,
        lr_schedule: str = None,
        weight_decay: float = 0.0,
        metric=nn.L1Loss,
        plot_interval: int = 1000,
        data_info: dict = None
    ) -> None:
        super().__init__(lr, weight_decay, metric, plot_interval, lr_schedule)

        self.bidirectional = bidirectional

        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Adjust the hidden size for bidirectional GRU
        if bidirectional:
            decoder_hidden_size = hidden_size * 2
        else:
            decoder_hidden_size = hidden_size

        try:
            self.activation = getattr(nn, activation)()
        except AttributeError:
            raise ValueError(f"Activation function '{activation}' is not valid.")
        #
        # self.recoder = NODE(
        #     input_size=decoder_hidden_size,
        # )

        self.recoder = nn.Identity()

        # self.decoder = nn.Linear(decoder_hidden_size, output_size)
        self.decoder = MLPStack(
            input_size=decoder_hidden_size,
            hidden_size=decoder_hidden_size,
            output_size=output_size,
            depth=decoder_depth,
            norm=False,
            residual=True,
            lazy=False,
        )

        # self._init_lazy((2, input_size))
        # self._init_weights()
        self.save_hyperparameters()

    def forward(self, X: torch.Tensor):
        X, _ = self.encoder(X)
        X = self.recoder(X)
        X = self.activation(X) # Apply activation function to the last time step output
        X = self.decoder(X)  # Decode to the output
        return X


class NODE_MLP(BaseModel):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        vf_depth: int = 3,
        vf_hidden_size: int = 256,
        decoder_depth: int = 3,
        dropout: float = 0.0,
        activation: str = "ReLU",
        norm: bool = False,
        lr: float = 1e-3,
        lr_schedule: str = None,
        weight_decay: float = 0.0,
        metric=nn.L1Loss,
        plot_interval: int = 1000,
        data_info: dict = None
    ) -> None:
        super().__init__(lr, weight_decay, metric, plot_interval, lr_schedule)

        try:
            _activation = getattr(nn, activation)
        except AttributeError:
            raise ValueError(f"Activation function '{activation}' is not valid.")

        self.encoder = nn.Linear(input_size, vf_hidden_size)

        self.recoder = NODE(
            input_size=vf_hidden_size,
            hidden_size=vf_hidden_size,
            depth=vf_depth,
            norm=norm,
            activation=_activation
        )

        self.decoder = MLPStack(
            input_size=vf_hidden_size,
            hidden_size=hidden_size,
            output_size=output_size,
            depth=decoder_depth,
            dropout=0.0,
            activation=_activation,
            norm=False,
            residual=True,
            lazy=False,
        )

        self.save_hyperparameters()

    def forward(self, X: torch.Tensor):
        X = self.encoder(X)
        X = self.recoder(X)
        X = self.decoder(X)
        return X


class ConvGRU(BaseModel):
    def __init__(
        self,
        input_size: int = 201,
        hidden_size: int = 128,
        num_layers: int = 3,
        output_size: int = 1,
        conv_channels: int = 32,
        conv_depth: int = 4,
        conv_kernels: list = [5, 3, 3, 3, 3],
        conv_downsample: int = 16,
        dropout: float = 0.0,
        activation: str = "ReLU",
        bidirectional: bool = False,
        lr: float = 1e-3,
        lr_schedule: str = None,
        weight_decay: float = 0.0,
        metric=nn.L1Loss,
        plot_interval: int = 1000,
        data_info: dict = None
    ) -> None:
        super().__init__(lr, weight_decay, metric, plot_interval, lr_schedule)

        self.bidirectional = bidirectional

        conv_channels = [1] + [conv_channels] * conv_depth if isinstance(conv_channels, int) else conv_channels
        conv_strides = [2 if i < int(np.log2(conv_downsample)) else 1 for i in range(conv_depth)]

        self.preprocess_conv = AttnResNet1d(
            depth=conv_depth,
            channels=conv_channels,
            kernels=conv_kernels,
            strides=conv_strides,
            dropout=0.0,
            norm=False,
            residual=True,
        )

        self.encoder = nn.GRU(
            input_size=conv_channels[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.flatten = nn.Flatten()

        # Adjust the hidden size for bidirectional GRU
        if bidirectional:
            hidden_size *= 2

        try:
            self.activation = getattr(nn, activation)()
        except AttributeError:
            raise ValueError(f"Activation function '{activation}' is not valid.")

        self.decoder = nn.LazyLinear(output_size)

        self.save_hyperparameters()

    def forward(self, X: torch.Tensor):
        X, _ = self.preprocess_conv(X.unsqueeze(1))
        X, _ = self.encoder(X.permute(0, 2, 1))
        # z, h = self.encoder(X.permute(0, 2, 1))
        X = self.flatten(X)
        X = self.activation(X)  # Apply activation function to the last time step output
        X = self.decoder(X)  # Decode to the output
        return X


class VisionTransformer1D(BaseModel):

    def __init__(
        self,
        input_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        output_dim: int,
        mlp_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        activation: str = "GeLU",
        lr: float = 1e-3,
        lr_schedule: str = None,
        weight_decay: float = 0.0,
        metric=nn.MSELoss,
        plot_interval: int = 1000,
        data_info: dict = None
    ) -> None:
        super().__init__(lr, weight_decay, metric, plot_interval, lr_schedule)

        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_patches = input_size // patch_size
        self.linear_proj = nn.Linear(patch_size, hidden_dim)

        try:
            self.activation = getattr(nn, activation)()
        except AttributeError:
            raise ValueError(f"Activation function '{activation}' is not valid.")

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                activation='gelu',
                norm_first=True,
            ),
            num_layers=num_layers
        )

        self.decoder = nn.Linear(hidden_dim, output_dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.dropout = nn.Dropout(dropout)

        self._init_weights()
        self.save_hyperparameters()


    def encode(self, X: torch.Tensor):
        X = X.view(X.size(0), self.num_patches, self.patch_size)
        X = self.linear_proj(X)

        batch_size = X.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        X = torch.cat((cls_tokens, X), dim=1)
        X += self.pos_embedding[:, :X.size(1), :]

        return self.dropout(X)

    def forward(self, X: torch.Tensor):
        X = self.encode(X)
        X = self.transformer(X)
        X = X[:, 0]
        X = self.decode(X)
        X = F.sigmoid(X)
        return X


### Old models ###
class LinearDiscriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(input_size // 2, input_size // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(input_size // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        validity = self.model(x)
        return validity


class ConvDiscriminator(pl.LightningModule):
    def __init__(self, input_size):
        super().__init__()

        conv = ConvSeq(
            input_size=input_size,
            output_channels=64,
            num_layers=5,
            kernel_one=9,
            kernel_two=5,
        )
        self.model = nn.Sequential(
            conv,
            nn.Flatten(),
            nn.Linear(conv.calculate_flat_size(), 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        validity = self.model(x.unsqueeze(1))
        return validity


class ConvSeq(pl.LightningModule):
    def __init__(
        self,
        input_size: int = 301,
        input_channels: int = 1,
        output_channels: int = 32,
        num_layers: int = 5,
        kernel_one: int = 9,
        kernel_two: int = 5,
        pool_size: int = 2,
        pool_all: bool = True,
    ):
        super().__init__()

        modules = list()
        conv_channels = np.interp(
            np.linspace(0.0, 1.0, num_layers + 1),
            [0, 1],
            [input_channels, output_channels],
        ).astype(int)
        for idx in range(num_layers):
            if idx == 0:
                kernel_size = kernel_one
            else:
                kernel_size = kernel_two

            modules.extend(
                [
                    nn.Conv1d(conv_channels[idx], conv_channels[idx + 1], kernel_size),
                    nn.PReLU(),
                    nn.BatchNorm1d(conv_channels[idx + 1]),
                ]
            )

            if pool_all or idx == num_layers - 1:
                modules.extend([nn.MaxPool1d(pool_size)])

        self.model = nn.Sequential(*modules)
        self.model.apply(init_rnn)
        self.save_hyperparameters()

    def calculate_flat_size(self):
        hparams = objectview(self.hparams)
        if hparams.pool_all:
            pool = 2
        else:
            pool = 1

        L = (hparams.input_size - (hparams.kernel_one - 1) - 1) + 1
        L = (L - 1 * (pool - 1) - 1) // pool + 1
        for n in range(1, hparams.num_layers + 1 - 1):
            if n == hparams.num_layers + 1 - 2:
                L = (L - (hparams.kernel_two - 1) - 1) + 1
                L = (L - 1 * (2 - 1) - 1) // 2 + 1
            else:
                L = (L - (hparams.kernel_two - 1) - 1) + 1
                L = (L - 1 * (pool - 1) - 1) // pool + 1

        flat_size = hparams.output_channels * L
        return flat_size

    def forward(self, x):
        output = self.model(x)
        return output


class BaseEncoder(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def encode(self, X: torch.Tensor):
        return self.forward(X)

    def loss(self, batch):
        (X, Y, params) = batch
        return 0.0, self.encode(X)


class BaseDecoder(pl.LightningModule):
    def __init__(self, hidden_dim: int, *args, **kwargs):
        super().__init__()
        self._model = None
        # the parameter regressor is shared regardless of
        # the decoder architecture
        self.param_regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.PReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.PReLU(),
            nn.Linear(hidden_dim // 4, 4),
        )

    @property
    def model(self):
        return self._model

    def decode(self, Z: torch.Tensor):
        return self.model(Z)

    @abstractmethod
    def loss(self, batch, z: torch.Tensor):
        raise NotImplementedError

    def predict_g2_params(self, Z: torch.Tensor):
        return self.param_regressor(Z)


class AutoEncoder(pl.LightningModule):
    def __init__(
        self, lr: float = 1e-3, weight_decay: float = 0.0, plot_percent: float = 0.05
    ) -> None:
        super().__init__()
        self.encoder = None
        self.decoder = None
        self.metric = nn.MSELoss()
        self.save_hyperparameters("lr", "weight_decay", "plot_percent")

    def encode(self, X: torch.Tensor):
        """
        Generate encodings given an input. Expects a tensor input shape
        of [N, T, 2] for N batch size, T timesteps; the second feature
        of the last dimesion corresponds to intensities.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor containing the timesteps and normalized
            counts

        Returns
        -------
        z
            Tensor of shape [N, Z] for N batch size and Z embedding
            dimension
        """
        return self.encoder(X[:, :, 1])

    def decode(self, Z: torch.Tensor):
        return self.decoder(Z)

    def forward(self, X: torch.Tensor):
        Z = self.encode(X)
        return self.decode(Z), Z

    def step(self, batch, batch_idx):
        (X, Y, g2_params) = batch
        # take only the intensities
        pred_Y, Z = self(X)
        recon = self.metric(Y[:, :, 1], pred_Y)
        loss = recon
        log = {"recon": recon}
        return loss, log

    def training_step(self, batch, batch_idx):
        loss, log = self.step(batch, batch_idx)
        self.log("training_loss", log)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log = self.step(batch, batch_idx)
        self.log("validiation_loss", log)

        if random.uniform(0, 1) < self.hparams.plot_percent:
            (X, Y, g2_params) = batch
            pred_Y, pred_params = self(X)
            # pred_Y, Z = self(X[:, :, 1])
            pred_Y = pred_Y[5].cpu()
            Y = Y.cpu()[5]
            X = X.cpu()[5]
            fig, ax = plt.subplots()
            ax.plot(X[:, 0], X[:, 1], label="Input")
            ax.plot(X[:, 0], pred_Y[:], label="Model")
            ax.plot(Y[:, 0], Y[:, 1], label="Target")
            ax.set_xscale("log")
            # ax.set_ylim([0.5, 1.1])
            ax.set_xlabel("time (ps)")
            ax.set_ylabel("g2")
            log.update({"spectrum": fig})

        self.logger.experiment.log(log)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("AutoEncoder")
        group.add_argument("--lr", type=float, default=1e-3)
        group.add_argument("--weight_decay", type=float, default=0.0)
        return parent_parser


class AdversarialAutoEncoder(AutoEncoder):
    def __init__(
        self,
        input_size: int,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        plot_percent: float = 1.0,
    ) -> None:
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            plot_percent=plot_percent,
        )

        # self.discriminator = LinearDiscriminator(input_size=input_size)
        self.discriminator = ConvDiscriminator(input_size=input_size)
        self.save_hyperparameters("lr", "weight_decay", "plot_percent")

    def _adversarial_loss(self, batch, batch_index):
        (X, Y, g2_params) = batch
        batch_size = X.size(0)

        valid = torch.ones(batch_size, 1, device=X.device, dtype=Y.dtype)
        real_loss = F.binary_cross_entropy(self.discriminator(Y[:, :, 1]), valid)

        pred_Y, z = self(X)

        fake = torch.zeros_like(valid)
        fake_loss = F.binary_cross_entropy(self.discriminator(pred_Y), fake)
        d_loss = (real_loss + fake_loss) / 2
        return d_loss

    def step(self, batch, batch_idx):
        (X, Y, g2_params) = batch
        pred_Y, pred_params = self(X[:, :, 1])
        recon = self.metric(Y[:, :, 1], pred_Y)
        # Discriminator
        d_loss = self._adversarial_loss(batch, batch_idx)

        loss = recon + d_loss
        log = {"recon": recon, "adversarial": d_loss}
        return loss, log


class ConvLSTMEncoder(BaseEncoder):
    def __init__(
        self,
        encoder_lstm_layers: int,
        encoder_hidden_dim: int,
        conv_num_layers: int,
        conv_output_channels: int,
        encoder_bidirectional: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.conv = ConvSeqProcessor(1, conv_output_channels, conv_num_layers)
        self.lstm = nn.LSTM(
            conv_output_channels,
            encoder_hidden_dim,
            num_layers=encoder_lstm_layers,
            batch_first=True,
            bidirectional=encoder_bidirectional,
        )
        true_hidden = encoder_hidden_dim
        if encoder_bidirectional:
            true_hidden *= 2
        self.true_hidden = true_hidden
        self.output = nn.Linear(true_hidden, encoder_hidden_dim)
        self._model = nn.ModuleList([self.conv, self.lstm, self.output])
        self.save_hyperparameters()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        conv_output = self.conv(X)
        _, (z, c) = self.lstm(conv_output)
        # swap back to [N x F]
        z = self.output(z.view(self.true_hidden, -1).permute(1, 0))
        return z

    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("ConvLSTMEncoder")
        group.add_argument("--encoder_lstm_layers", type=int, default=1)
        group.add_argument("--encoder_hidden_dim", type=int, default=32)
        group.add_argument("--conv_num_layers", type=int, default=5)
        group.add_argument("--conv_output_channels", type=int, default=256)
        group.add_argument("--encoder_bidirectional", type=bool, default=True)
        return parent_parser


class GRUAutoEncoder(pl.LightningModule):
    """
    Implements a simple autoencoder model, which
    takes noisy photon counts, and returns a de-noised version.

    The computational flow is summarized like this:

    x_t -> z_t -> y_t

    ...a vanilla autoencoder that happens to be recurrent.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        encoder_num_layers: int = 5,
        decoder_num_layers: int = 3,
        lr=1e-3,
        weight_decay=0.0,
        batch_size: int = 64,
        bidirectional: bool = False,
        input_dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.encoder = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=encoder_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            **kwargs,
        )
        self.input_dropout = nn.Dropout(input_dropout)
        if bidirectional:
            true_hidden = hidden_dim * 2
        else:
            true_hidden = hidden_dim
        self.hidden = nn.Sequential(
            nn.Linear(true_hidden, true_hidden),
            nn.PReLU(),
            nn.Linear(true_hidden, true_hidden),
            nn.PReLU(),
            nn.Linear(true_hidden, true_hidden),
            nn.PReLU(),
        )
        self.bias_step = nn.Sequential(
            nn.Linear(true_hidden, true_hidden),
            nn.PReLU(),
            nn.Linear(true_hidden, true_hidden),
            nn.PReLU(),
            nn.Linear(true_hidden, true_hidden),
            nn.PReLU(),
        )
        # the decoder uses the timestep at each point and
        # the encoder hidden state to reproduce the spectrum
        self.decoder = nn.GRU(
            1, true_hidden, num_layers=decoder_num_layers, batch_first=True, **kwargs
        )
        self.output = nn.GRU(true_hidden, 1, batch_first=True, **kwargs)
        # apply specific initialization to RNN layers
        self.encoder.apply(init_rnn)
        self.decoder.apply(init_rnn)
        self.output.apply(init_rnn)
        # initialize the FC layer
        nn.init.kaiming_uniform_(self.hidden[0].weight)
        nn.init.zeros_(self.hidden[0].bias)
        # define the loss function
        self.metric = nn.MSELoss()
        # stores the learning rate and whatnot to a struct
        self.save_hyperparameters()

    @property
    def num_directions(self):
        return 2 if self.hparams.bidirectional else 1

    @format_time_sequence
    def forward(self, X: torch.Tensor):
        N, T, _ = X.shape
        # z is the embedding for each element in the sequence
        # and h is the final state; i.e. z[-1] == h
        z, h = self.encoder(self.input_dropout(X))
        # transform the hidden state of the encoder into
        # the right shape
        h_t = self._get_hidden_state(h)
        decoder_state = self.hidden(h_t).repeat(self.hparams.decoder_num_layers, 1, 1)
        # take the timesteps only and have the decoder reproduce the
        # noise free spectrum
        output, _ = self.decoder(X[:, :, 0].unsqueeze(-1), decoder_state)
        output, _ = self.output(output)
        return F.relu(output), None

    def _get_hidden_state(self, h: torch.Tensor):
        """
        Utility function to extract the last hidden state of
        the recurrent embedding. In the single directional
        case, we simply return a Tensor of shape [N, H]
        with N batchsize and H hidden dimensionality.
        In the bidirectional case, we return [N, H * 2]
        by concatenating the directions.

        Parameters
        ----------
        h : torch.Tensor
            Hidden state output of an RNN

        Returns
        -------
        [type]
            [description]
        """
        hidden_dim, num_layers = (
            self.hparams.hidden_dim,
            self.hparams.encoder_num_layers,
        )
        # for a bidirectional model, we will concatenate the directions
        # into a single vector
        h = h.view(num_layers, self.num_directions, -1, hidden_dim)
        if self.num_directions == 2:
            h_state = torch.cat([h[-1][0], h[-1][1]], dim=-1).unsqueeze(0)
        else:
            h_state = h[-1]
        return h_state

    def step(self, batch, batch_idx):
        # unpack the noisy spectrum X, the ground truth Y
        (X, Y, g2_params) = batch
        pred_Y, pred_params = self(X)
        # squeeze gets rid of the erroenous dimension
        recon_error = self.metric(Y[:, :, 1], pred_Y)
        param_error = self.metric(g2_params, pred_params)
        loss = recon_error + param_error
        losses = {"joint": loss, "recon": recon_error, "param": param_error}
        return loss, losses

    def training_step(self, batch, batch_idx):
        loss, losses = self.step(batch, batch_idx)
        losses = {f"{key}": value for key, value in losses.items()}
        self.log("train_loss", losses)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        # log some spectra to show off
        (X, Y, g2_params) = batch
        pred_Y, pred_params = self(X)
        pred_Y = pred_Y[5].cpu()
        Y = Y.cpu()[5]
        X = X.cpu()[5]
        fig, ax = plt.subplots()
        ax.plot(X[:, 0], X[:, 1], label="Input")
        ax.plot(X[:, 0], pred_Y[:], label="Model")
        ax.plot(Y[:, 0], Y[:, 1], label="Target")
        ax.set_xscale("log")
        # ax.set_ylim([0.5, 1.1])
        ax.set_xlabel("time (ps)")
        ax.set_ylabel("g2")
        logs.update({"spectrum": fig})
        # this uses wandb to log a matplotlib figure, which
        # will convert it to a plotly chart
        self.logger.experiment.log(logs)
        # self.log("validation_loss", logs)
        # have to delete this to supress matplotlib from
        # overfilling a buffer
        del fig
        del ax
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[parent_parser, common_parser()], add_help=False
        )
        return parser


class GRUVAE(GRUAutoEncoder):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        encoder_num_layers: int = 3,
        decoder_num_layers: int = 1,
        lr=1e-3,
        weight_decay=0.0,
        batch_size: int = 64,
        bidirectional: bool = False,
        input_dropout: float = 0.3,
        beta: float = 4.0,
        **kwargs,
    ):
        super().__init__(
            input_dim,
            hidden_dim,
            encoder_num_layers,
            decoder_num_layers,
            lr,
            weight_decay,
            batch_size,
            bidirectional,
            input_dropout,
            **kwargs,
        )
        if bidirectional:
            true_hidden = hidden_dim * 2
        else:
            true_hidden = hidden_dim
        self.z_mu = nn.Linear(true_hidden, true_hidden)
        self.z_logvar = nn.Linear(true_hidden, true_hidden)
        # initialize weights
        for fc in [self.z_mu, self.z_logvar]:
            nn.init.zeros_(fc.bias)
            nn.init.kaiming_uniform_(fc.weight)
        # define the loss function
        self.metric = nn.MSELoss()
        # stores the learning rate and whatnot to a struct
        self.save_hyperparameters()

    def forward(self, X: torch.Tensor):
        z, h = self.encoder(self.input_dropout(X))
        h_t = self._get_hidden_state(h)
        decoder_state = self.hidden(h_t)
        p, q, vae_z = self.reparameterize(z, decoder_state)
        output, _ = self.decoder(X[:, :, 0].unsqueeze(-1), vae_z)
        output, _ = self.output(output)
        return F.relu(output), (p, q, vae_z)

    def reparameterize(self, z: torch.Tensor, h: torch.Tensor):
        mu, logvar = self.z_mu(h), self.z_logvar(h)
        p, q, vae_z = self.sample(mu, logvar)
        if self.num_directions == 2:
            vae_z = vae_z.repeat(self.hparams.decoder_num_layers, 1, 1)
        return p, q, vae_z

    def sample(self, mu: torch.Tensor, log_var: torch.Tensor):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def _vae_loss(self, Y, pred_Y, vae_params):
        p, q, vae_z = vae_params
        recon_loss = self.metric(pred_Y, Y)
        log_qz = q.log_prob(vae_z)
        log_pz = p.log_prob(vae_z)
        # calculate the prior regularization term
        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.hparams.beta
        loss = recon_loss + kl
        logs = {"recon_loss": recon_loss, "kl": kl, "loss": loss}
        return loss, logs

    def step(self, batch, batch_idx):
        (X, Y, g2_params) = batch
        pred_Y, vae_params = self(X)
        loss, logs = self._vae_loss(Y[:, :, 1], pred_Y, vae_params)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """
        Modified slightly from the base GRU case, as the
        forward method returns the VAE parameters which
        we don't explicitly care about here.
        """
        loss = self.step(batch, batch_idx)
        self.log("validation_loss", loss)
        # log some spectra to show off
        (X, Y, g2_params) = batch
        pred_Y, _, _ = self(X)
        pred_Y = pred_Y.cpu()[5]
        Y = Y.cpu()[5]
        X = X.cpu()[5]
        fig, ax = plt.subplots()
        ax.plot(X[:, 0], X[:, 1], label="Input")
        ax.plot(X[:, 0], pred_Y.squeeze(), label="Model")
        ax.plot(Y[:, 0], Y[:, 1], label="Target")
        ax.set_xscale("log")
        # ax.set_ylim([0.5, 1.1])
        ax.set_xlabel("time (ps)")
        ax.set_ylabel("g2")
        # this uses wandb to log a matplotlib figure, which
        # will convert it to a plotly chart
        self.logger.experiment.log({"spectrum": fig})
        # have to delete this to supress matplotlib from
        # overfilling a buffer
        del fig
        del ax
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[parent_parser, common_parser()], add_help=False
        )
        parser.add_argument("--beta", type=float, default=4.0)
        return parser


class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_layers: int,
        kernel_size: int = 5,
        kernel_size2: int = 3,
        pool_size: int = 2,
        activation: str = "SiLU",
        activation_args: str = "",
    ):
        super().__init__()
        act_func = eval(f"nn.{activation}({activation_args})")
        modules = []
        channels = np.interp(
            np.linspace(0.0, 1.0, num_layers), [0, 1], [input_channels, output_channels]
        ).astype(int)
        for idx in range(num_layers - 1):
            if idx < 2:
                kernel_size = kernel_size
            else:
                kernel_size = kernel_size2
            modules.extend(
                [
                    nn.Conv1d(channels[idx], channels[idx + 1], kernel_size),
                    # nn.SiLU(),
                    act_func,
                    nn.BatchNorm1d(channels[idx + 1]),
                    nn.MaxPool1d(pool_size),
                ]
            )
        modules.append(nn.Flatten())
        self.model = nn.Sequential(*modules)

    def forward(self, X: torch.Tensor):
        if X.ndim == 2:
            temp = X.unsqueeze(1)
        else:
            temp = X
        return self.model(temp)


class ConvSeqProcessor(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_layers: int,
        kernel_size: int = 3,
        pool_size: int = 2,
    ):
        super().__init__()
        modules = list()
        # calculate linear decrease from input to output in the
        # number of channels
        channels = np.interp(
            np.linspace(0.0, 1.0, num_layers), [0, 1], [input_channels, output_channels]
        ).astype(int)
        for idx in range(num_layers - 1):
            modules.extend(
                [
                    nn.Conv1d(channels[idx], channels[idx + 1], kernel_size),
                    nn.PReLU(),
                    nn.BatchNorm1d(channels[idx + 1]),
                    nn.MaxPool1d(pool_size),
                ]
            )
        self.model = nn.Sequential(*modules)
        self.model.apply(init_rnn)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size

    @staticmethod
    def calc_output_size(input_channels: int, kernel_size: int):
        # assumes a stride and dilation of 1, and a padding of zero
        return (input_channels - (kernel_size - 1) - 1) + 1

    def forward(self, X: torch.Tensor):
        """
        Returns an embedding that represents the entire
        sequence, with shape [N, L, C] for L features and
        C channels. The C/L dimensions are swapped in order
        to preserve the same number of features, as to
        feed into an RNN: L depends on the spectrum length,
        and so this may be a way to be invariant to that.

        Parameters
        ----------
        X : torch.Tensor
            [description]

        Returns
        -------
        [type]
            [description]
        """
        intensities = X[:, :, 1].unsqueeze(1)
        output = self.model(intensities)
        return output.permute(0, 2, 1)


class ConvLSTMAutoEncoder(GRUAutoEncoder):
    """
    TODO: need to hook up the convolution layer with
    the inputs and feed them into encoder RNN. This requires
    that the sequence length be known, but not so big an issue.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        encoder_num_layers: int = 5,
        decoder_num_layers: int = 3,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 64,
        bidirectional: bool = False,
        input_dropout: float = 0.0,
        conv_output_channels: int = 128,
        conv_num_layers: int = 5,
        kernel_size: int = 3,
        pool_size: int = 2,
        num_params: int = 15,
        **kwargs,
    ):
        super().__init__(
            input_dim,
            hidden_dim,
            encoder_num_layers,
            decoder_num_layers,
            lr,
            weight_decay,
            batch_size,
            bidirectional,
            input_dropout,
        )
        # preprocessor runs through the intensities
        self.preprocess_conv = ConvSeqProcessor(
            1, conv_output_channels, conv_num_layers, kernel_size, pool_size
        )
        self.encoder = nn.LSTM(
            conv_output_channels,
            hidden_dim,
            num_layers=encoder_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        if bidirectional:
            true_hidden = hidden_dim * 2
        else:
            true_hidden = hidden_dim
        self.hidden = nn.Sequential(
            nn.Linear(true_hidden, true_hidden),
            nn.PReLU(),
        )
        self.decoder = nn.LSTM(
            1 + true_hidden,
            true_hidden,
            num_layers=decoder_num_layers,
            batch_first=True,
            **kwargs,
        )
        self.output = nn.LSTM(true_hidden, 1, batch_first=True, **kwargs)

        self.param_regressor = nn.Sequential(
            nn.Linear(true_hidden, true_hidden // 2),
            nn.PReLU(),
            nn.Linear(true_hidden // 2, true_hidden // 4),
            nn.PReLU(),
            nn.Linear(true_hidden // 4, num_params),
        )

        # apply specific initialization to RNN layers
        self.encoder.apply(init_rnn)
        self.decoder.apply(init_rnn)
        self.output.apply(init_rnn)
        # apply another initialization scheme for fc layers
        self.param_regressor.apply(init_fc_layers)
        self.hidden.apply(init_fc_layers)
        # define the loss function
        self.metric = nn.MSELoss()
        # stores the learning rate and whatnot to a struct
        self.save_hyperparameters()

    def forward(self, X: torch.Tensor):
        N, T, D = X.shape
        # use convolution windows to preprocess the whole spectrum
        # and use an encoding RNN to summarize the channels
        conv_context = self.preprocess_conv(X)
        z, (h, c) = self.encoder(conv_context)
        # transform the hidden state of the encoder into
        # the right shape
        h_t = self._get_hidden_state(h)
        # hidden is shape N x D
        hidden = self.hidden(h_t).squeeze()
        params = self.param_regressor(hidden)
        # this applies additional transformations to the embedding
        # to try and bump up the bias in the first timesteps
        # this is then passed as the initial hidden state for every decoder layer
        hidden_bias = self.bias_step(h_t).repeat(self.hparams.decoder_num_layers, 1, 1)
        # tack the hidden state every time step
        repeated_hidden = hidden.unsqueeze(1).repeat(1, T, 1)
        decoder_input = torch.cat([X[:, :, 0].unsqueeze(-1), repeated_hidden], -1)
        # run through decoder, and then shape out
        decoded, _ = self.decoder(
            decoder_input, (hidden_bias, torch.ones_like(hidden_bias))
        )
        output, _ = self.output(decoded)
        return output.squeeze(-1), params

    def step(self, batch, batch_idx):
        # unpack the noisy spectrum X, the ground truth Y
        (X, Y, g2_params) = batch
        pred_Y, pred_params = self(X)
        # squeeze gets rid of the erroenous dimension
        recon_error = self.metric(Y[:, :, 1], pred_Y)
        param_error = self.metric(g2_params, pred_params)
        # loss = recon_error + param_error
        loss = recon_error
        losses = {
            "recon": recon_error,
            # "joint": loss,
            # "param": param_error
        }

        return loss, losses

    def training_step(self, batch, batch_idx):
        loss, losses = self.step(batch, batch_idx)
        losses = {f"{key}": value for key, value in losses.items()}
        self.log("train_loss", losses)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        # log some spectra to show off
        (X, Y, g2_params) = batch
        pred_Y, pred_params = self(X)
        pred_Y = pred_Y[5].cpu()
        Y = Y.cpu()[5]
        X = X.cpu()[5]
        fig, ax = plt.subplots()
        ax.plot(X[:, 0], X[:, 1], label="Input")
        ax.plot(X[:, 0], pred_Y[:], label="Model")
        ax.plot(Y[:, 0], Y[:, 1], label="Target")
        ax.set_xscale("log")
        # ax.set_ylim([0.5, 1.1])
        ax.set_xlabel("time (ps)")
        ax.set_ylabel("g2")
        logs.update({"spectrum": fig})
        # this uses wandb to log a matplotlib figure, which
        # will convert it to a plotly chart
        self.logger.experiment.log(logs)
        # self.log("validation_loss", logs)
        # have to delete this to supress matplotlib from
        # overfilling a buffer
        del fig
        del ax
        return loss

    @torch.no_grad()
    def predict(self, X: torch.Tensor):
        if X.ndim == 2:
            X.unsqueeze_(0)
        return self(X)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[parent_parser, common_parser()], add_help=False
        )
        parser.add_argument("--conv_output_channels", type=int, default=256)
        parser.add_argument("--conv_num_layers", type=int, default=5)
        parser.add_argument("--kernel_size", type=int, default=3)
        parser.add_argument("--pool_size", type=int, default=2)
        parser.add_argument("--num_params", type=int, default=6)
        return parser


class CLVAE(ConvLSTMAutoEncoder):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        encoder_num_layers: int = 5,
        decoder_num_layers: int = 3,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 64,
        bidirectional: bool = False,
        input_dropout: float = 0.0,
        conv_output_channels: int = 128,
        conv_num_layers: int = 5,
        kernel_size: int = 3,
        pool_size: int = 2,
        beta_scheduler_kwargs={},
        **kwargs,
    ):
        super().__init__(
            input_dim,
            hidden_dim,
            encoder_num_layers=encoder_num_layers,
            decoder_num_layers=decoder_num_layers,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            bidirectional=bidirectional,
            input_dropout=input_dropout,
            conv_output_channels=conv_output_channels,
            conv_num_layers=conv_num_layers,
            kernel_size=kernel_size,
            pool_size=pool_size,
            **kwargs,
        )
        if bidirectional:
            true_hidden = hidden_dim * 2
        else:
            true_hidden = hidden_dim
        self.z_mu = nn.Linear(true_hidden, true_hidden)
        self.z_logvar = nn.Linear(true_hidden, true_hidden)
        # initialize weights
        for fc in [self.z_mu, self.z_logvar]:
            nn.init.zeros_(fc.bias)
            nn.init.uniform_(fc.weight, a=1e-5, b=1e-3)
        self.beta_scheduler = BetaRateScheduler(**beta_scheduler_kwargs)
        self.beta_scheduler.reset()
        self.save_hyperparameters()

    def forward(self, X: torch.Tensor):
        """
        Run through the probabilistic model. Returns three output Tensors,
        corresponding to the predicted photon counts [N x T], the VAE
        related distributions and hidden state, and the g2 parameters [N x 4].

        Parameters
        ----------
        X : torch.Tensor
            [description]

        Returns
        -------
        [type]
            [description]
        """
        T = X.size(1)
        # use convolution windows to preprocess the whole spectrum
        # and use an encoding RNN to summarize the channels
        conv_context = self.preprocess_conv(X)
        z, (h, c) = self.encoder(conv_context)
        # transform the hidden state of the encoder into
        # the right shape
        h_t = self._get_hidden_state(h)
        temp = self.hidden(h_t).squeeze()
        # now parameterize Gaussians
        p, q, hidden = self.reparameterize(temp)
        params = self.param_regressor(hidden)
        # this applies additional transformations to the embedding
        # to try and bump up the bias in the first timesteps
        # this is then passed as the initial hidden state for every decoder layer
        hidden_bias = self.bias_step(hidden).repeat(
            self.hparams.decoder_num_layers, 1, 1
        )
        # tack the hidden state every time step
        repeated_hidden = hidden.unsqueeze(1).repeat(1, T, 1)
        decoder_input = torch.cat([X[:, :, 0].unsqueeze(-1), repeated_hidden], -1)
        decoded, _ = self.decoder(
            decoder_input, (hidden_bias, torch.ones_like(hidden_bias))
        )
        output, _ = self.output(decoded)
        return output.squeeze(-1), (p, q, hidden), params

    def reparameterize(self, h: torch.Tensor):
        mu, logvar = self.z_mu(h), self.z_logvar(h)
        # draw samples from parameterized Gaussians
        p, q, vae_z = self.sample(mu, logvar)
        # if self.num_directions == 2:
        #     vae_z = vae_z.repeat(self.hparams.decoder_num_layers, 1, 1)
        return p, q, vae_z

    def sample(
        self, mu: torch.Tensor, log_var: torch.Tensor, samples: Optional[int] = None
    ):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        if samples is not None:
            z = q.rsample(samples)
        else:
            z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        (X, Y, g2_params) = batch
        pred_Y, vae_params, pred_params = self(X)
        p, q, vae_z = vae_params
        recon_loss = self.metric(Y[:, :, 1], pred_Y)
        # g2 parameter regression and symmetry
        param_loss = self.metric(g2_params, pred_params)
        # if self.num_directions == 2:
        #     z = vae_z[0]
        # else:
        z = vae_z
        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)
        # calculate the prior regularization term
        kl = log_qz - log_pz
        kl = kl.mean()
        beta = next(self.beta_scheduler.beta())
        kl *= beta
        loss = recon_loss + kl + param_loss
        logs = {
            "recon": recon_loss,
            "kl": kl,
            "joint": loss,
            "param": param_loss,
            "beta": beta,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log("train_loss", logs)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """
        Modified slightly from the base GRU case, as the
        forward method returns the VAE parameters which
        we don't explicitly care about here.
        """
        loss, logs = self.step(batch, batch_idx)
        # for key, value in logs.items():
        #     self.log(f"val_{key}", value)
        # log some spectra to show off
        (X, Y, g2_params) = batch
        pred_Y, _, _ = self(X)
        pred_Y = pred_Y[5].cpu()
        Y = Y[5].cpu()
        X = X[5].cpu()
        fig, ax = plt.subplots()
        ax.plot(X[:, 0], X[:, 1], label="Input")
        ax.plot(X[:, 0], pred_Y.squeeze(), label="Model")
        ax.plot(Y[:, 0], Y[:, 1], label="Target")
        logs.update({"spectrum": fig})
        # this uses wandb to log a matplotlib figure, which
        # will convert it to a plotly chart
        self.logger.experiment.log(logs)
        # self.log("validation_loss", logs)
        # have to delete this to supress matplotlib from
        # overfilling a buffer
        del fig
        del ax
        return loss

    @torch.no_grad()
    def predict(self, X: Union[np.ndarray, torch.Tensor], n_samples: int = 200):
        """
        Perform variational inference on X. Takes a single spectrum, and draws
        `n_samples` from the posterior to obtain the noise-free spectrum and
        the uncertainty.

        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            NumPy array or torch tensor containing the spectra.
        n_samples : int, optional
            [description], by default 200

        Returns
        -------

            [description]
        """
        if type(X) is np.ndarray:
            # convert to single precision tensor
            X = torch.from_numpy(X).float()
        if X.ndim == 2:
            # just to get batch dimension
            X.unsqueeze_(0)
        X = X.repeat(n_samples, 1, 1)
        conv_context = self.preprocess_conv(X)
        z, (h, c) = self.encoder(conv_context)
        # run the final output of the encoder through FC layer
        h = self.hidden(self._get_hidden_state(h))
        # shape of h should be [1, 1, H]; draw `n_samples` from posterior
        p, q, decoder_state = self.reparameterize(h)
        decoded, _ = self.decoder(X, (decoder_state, torch.zeros_like(decoder_state)))
        output, _ = self.output(decoded)
        return output

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[parent_parser, common_parser()], add_help=False
        )
        parser.add_argument("--conv_output_channels", type=int, default=256)
        parser.add_argument("--conv_num_layers", type=int, default=5)
        parser.add_argument("--kernel_size", type=int, default=3)
        parser.add_argument("--pool_size", type=int, default=2)
        # parser.add_argument("--beta", type=float, default=4.0)
        return parser


class MLPBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        norm: bool = True,
        activation: str = "SiLU",
        activation_args: str = "",
    ):
        super().__init__()
        linear = nn.Linear(input_dim, output_dim)
        act_func = eval(f"nn.{activation}({activation_args})")
        drop_layer = nn.Dropout(dropout)
        if norm:
            norm = nn.BatchNorm1d(output_dim)
        else:
            norm = nn.Identity()
        self.layers = nn.Sequential(linear, norm, act_func, drop_layer)

    def forward(self, X: torch.Tensor):
        return self.layers(X)


class MLPAutoEncoder(AutoEncoder):
    def __init__(
        self,
        input_dim: int = 140,
        z_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.0,
        norm: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        activation: str = "SiLU",
        activation_args: str = "",
        plot_percent: float = 0.05,
    ):
        super().__init__(lr, weight_decay, plot_percent)
        dims = np.linspace(input_dim, z_dim, num_layers).astype(int)
        encoder = []
        # encoding direction
        for index in range(num_layers - 1):
            encoder.append(
                MLPBlock(
                    dims[index],
                    dims[index + 1],
                    dropout,
                    norm,
                    activation,
                    activation_args,
                )
            )
        self.encoder = nn.Sequential(*encoder)
        decoder = []
        # decoder direction
        for index in reversed(range(num_layers - 1)):
            if index != 0:
                decoder.append(
                    MLPBlock(
                        dims[index + 1],
                        dims[index],
                        dropout,
                        norm,
                        activation,
                        activation_args,
                    )
                )
            else:
                decoder.append(
                    MLPBlock(
                        dims[index + 1],
                        dims[index],
                        dropout,
                        norm=False,
                        activation="ReLU",
                    )
                )
        self.decoder = nn.Sequential(*decoder)
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[
                parent_parser,
            ],
            add_help=False,
        )
        parser = AutoEncoder.add_model_specific_args(parser)
        parser.add_argument("--input_dim", type=int, default=301)
        parser.add_argument("--z_dim", type=int, default=64)
        parser.add_argument("--num_layers", type=int, default=5)
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--norm", type=bool, default=True)
        parser.add_argument("--activation", type=str, default="SiLU")
        return parser


class MLPAutoEncoderAdversary(AdversarialAutoEncoder):
    def __init__(
        self,
        input_dim: int,
        z_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.0,
        norm: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        activation: str = "SiLU",
        activation_args: str = "",
        plot_percent: float = 0.05,
    ):
        super().__init__(
            input_size=input_dim,
            lr=lr,
            weight_decay=weight_decay,
            plot_percent=plot_percent,
        )
        dims = np.linspace(input_dim, z_dim, num_layers).astype(int)
        encoder = []
        # encoding direction
        for index in range(num_layers - 1):
            encoder.append(
                MLPBlock(
                    dims[index],
                    dims[index + 1],
                    dropout,
                    norm,
                    activation,
                    activation_args,
                )
            )
        self.encoder = nn.Sequential(*encoder)
        decoder = []
        # decoder direction
        for index in reversed(range(num_layers - 1)):
            if index != 0:
                decoder.append(
                    MLPBlock(
                        dims[index + 1],
                        dims[index],
                        dropout,
                        norm,
                        activation,
                        activation_args,
                    )
                )
            else:
                decoder.append(
                    MLPBlock(
                        dims[index + 1],
                        dims[index],
                        dropout,
                        norm=False,
                        activation="ReLU",
                    )
                )
        self.decoder = nn.Sequential(*decoder)
        self.save_hyperparameters()


class ConvMLPAutoEncoder(AutoEncoder):
    def __init__(
        self,
        input_dim: int = 140,
        encoder_layers: int = 6,
        decoder_layers: int = 3,
        num_channels: int = 64,
        kernel_size: int = 5,
        kernel_size2: int = 3,
        dropout: float = 0.0,
        norm: bool = True,
        lr: float = 2e-4,
        weight_decay: float = 0.0,
        activation: str = "SiLU",
        activation_args: str = "",
        plot_percent=0.0,
    ):
        super().__init__(lr, weight_decay, plot_percent)

        self.encoder = ConvEncoder(
            1, num_channels, encoder_layers, kernel_size, kernel_size2
        )
        expected_size = get_conv_output_size(self.encoder, torch.ones(1, input_dim))

        dims = np.linspace(input_dim, expected_size, decoder_layers).astype(int)
        decoder = []
        # decoder direction
        for index in reversed(range(decoder_layers - 1)):
            if index != 0:
                decoder.append(
                    MLPBlock(
                        dims[index + 1],
                        dims[index],
                        dropout,
                        norm,
                        activation,
                        activation_args,
                    )
                )
            else:
                decoder.append(
                    MLPBlock(
                        dims[index + 1],
                        dims[index],
                        dropout,
                        norm=False,
                        activation="ReLU",
                    )
                )
        self.decoder = nn.Sequential(*decoder)
        self.save_hyperparameters()


class Conv1DAutoEncoder(AutoEncoder):
    def __init__(
        self,
        input_dim: int = 140,
        num_layers: int = 4,
        num_channels: int = 256,
        z_dim: int = 32,
        kernel1: int = 11,
        kernel2: int = 3,
        dropout: float = 0.0,
        bottleneck: bool = True,
        norm: bool = True,
        lr: float = 5e-5,
        weight_decay: float = 0.0,
        activation: str = "SiLU",
        activation_args: str = "",
        plot_percent=0.05,
    ):
        super().__init__(lr, weight_decay, plot_percent)
        act_func = eval(f"nn.{activation}({activation_args})")

        if norm:
            norm = nn.BatchNorm1d
        else:
            norm = nn.Identity

        channels = np.interp(
            np.linspace(0.0, 1.0, num_layers), [0, 1], [1, num_channels]
        ).astype(int)

        encoder_modules = []
        for idx in range(num_layers - 1):
            if idx < num_layers - 2:
                kernel = kernel1
                stride = 2
                pad = 1
            else:
                kernel = kernel2
                stride = 1
                pad = 0
            encoder_modules.extend(
                [
                    nn.Conv1d(channels[idx], channels[idx + 1], kernel, stride, pad),
                    nn.Dropout(dropout),
                    norm(channels[idx + 1]),
                    act_func,
                ]
            )
        encoder_conv = nn.Sequential(*encoder_modules)
        encoder_modules.append(nn.Flatten())

        self.encoder = nn.Sequential(*encoder_modules)

        flat_shape = get_conv1d_flat_shape(
            self.encoder, torch.ones(input_dim).unsqueeze(0).unsqueeze(0)
        )  # for linear bottleneck
        conv_shape = get_conv_output_shape(
            encoder_conv, torch.ones(input_dim).unsqueeze(0).unsqueeze(0)
        )  # for Reshape layer

        if bottleneck:
            self.bottleneck = nn.Sequential(
                nn.Linear(flat_shape[0], z_dim),
                nn.Linear(z_dim, flat_shape[0]),
            )
        else:
            self.bottleneck = nn.Sequential()

        decoder_modules = []
        decoder_modules.append(
            Reshape(-1, conv_shape[1], conv_shape[2])
        )  # the reverse of nn.Flatten()
        for idx in reversed(range(num_layers - 1)):
            if idx < num_layers - 2:
                kernel = kernel1
                stride = 2
                pad = 1
            else:
                kernel = kernel2
                stride = 1
                pad = 0
            decoder_modules.extend(
                [
                    nn.ConvTranspose1d(
                        channels[idx + 1],
                        channels[idx],
                        kernel,
                        stride,
                        padding=pad,
                        output_padding=pad,
                    ),
                    act_func,
                ]
            )
        # decoder_modules.append(nn.Tanh())

        self.decoder = nn.Sequential(*decoder_modules)
        # self.apply(init_fc_layers)
        self.save_hyperparameters()

    def encode(self, X: torch.Tensor):
        X = X[:, :, 1].unsqueeze(1)
        return self.encoder(X)

    def decode(self, Z: torch.Tensor):
        return self.decoder(Z)

    def forward(self, X: torch.Tensor):
        Z = self.encode(X)
        B = self.bottleneck(Z)
        D = self.decode(B)
        D = D.squeeze(1)
        return D, B

    def step(self, batch, batch_idx):
        (X, Y, g2_params) = batch
        pred_Y, Z = self(X)
        recon = self.metric(Y[:, :, 1], pred_Y)
        loss = recon
        log = {"recon": recon}
        return loss, log

    def training_step(self, batch, batch_idx):
        loss, log = self.step(batch, batch_idx)
        self.log("training_loss", log)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log = self.step(batch, batch_idx)
        self.log("validiation_loss", log)

        if random.uniform(0, 1) < self.hparams.plot_percent:
            (X, Y, g2_params) = batch
            pred_Y, pred_params = self(X)
            pred_Y = pred_Y[5].cpu()
            pred_Y = pred_Y.squeeze(0)
            Y = Y.cpu()[5]
            X = X.cpu()[5]
            fig, ax = plt.subplots()
            ax.plot(X[:, 0], X[:, 1], label="Input")
            ax.plot(X[:, 0], pred_Y[:], label="Model")
            ax.plot(Y[:, 0], Y[:, 1], label="Target")
            ax.set_xscale("log")
            # ax.set_ylim([0.5, 1.1])
            ax.set_xlabel("time (ps)")
            ax.set_ylabel("g2")
            wandb.Image(plt)
            log.update({"spectrum": fig})

        self.logger.experiment.log(log)

        return loss


class ConvMLPAutoEncoderAdversary(AdversarialAutoEncoder):
    def __init__(
        self,
        input_dim: int,
        z_dim: int = 64,
        num_layers: int = 3,
        num_channels: int = 64,
        dropout: float = 0.0,
        norm: bool = True,
        lr: float = 2e-4,
        weight_decay: float = 0.0,
        activation: str = "SiLU",
        activation_args: str = "",
        plot_percent: float = 0.05,
    ):
        super().__init__(
            input_size=input_dim,
            lr=lr,
            weight_decay=weight_decay,
            plot_percent=plot_percent,
        )

        self.encoder = ConvEncoder(1, num_channels, num_layers)
        expected_size = get_conv_output_size(self.encoder, torch.ones(1, input_dim))
        dims = np.linspace(input_dim, expected_size, num_layers).astype(int)
        decoder = []
        # decoder direction
        for index in reversed(range(num_layers - 1)):
            if index != 0:
                decoder.append(
                    MLPBlock(
                        dims[index + 1],
                        dims[index],
                        dropout,
                        norm,
                        activation,
                        activation_args,
                    )
                )
            else:
                decoder.append(
                    MLPBlock(
                        dims[index + 1],
                        dims[index],
                        dropout,
                        norm=False,
                        activation="ReLU",
                    )
                )
        self.decoder = nn.Sequential(*decoder)
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[
                parent_parser,
            ],
            add_help=False,
        )
        parser = AutoEncoder.add_model_specific_args(parser)
        parser.add_argument("--input_dim", type=int, default=301)
        parser.add_argument("--num_channels", type=int, default=64)
        parser.add_argument("--num_layers", type=int, default=5)
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--norm", type=bool, default=True)
        parser.add_argument("--activation", type=str, default="SiLU")
        return parser


class EnsembleModel(pl.LightningModule):
    def __init__(
        self,
        base_model: nn.Module,
        num_models: int,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        plot_percent: float = 0.10,
        ens_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.models = nn.ModuleList([deepcopy(base_model) for _ in range(num_models)])
        self.metric = nn.GaussianNLLLoss()
        input_dim = base_model.hparams.input_dim
        # want the mean and variance out
        output_dim = input_dim * 2
        # this compacts the mean and variance calculation into a single
        # batched operation, and reshapes it later
        self.ensemble_output = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(ens_dropout),
                    nn.Linear(input_dim, output_dim),
                    nn.Softplus(),
                )
                for _ in range(num_models)
            ]
        )
        self.apply(init_fc_layers)
        self.save_hyperparameters()

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor]:
        # number of models in ensemble, batch size, number of timesteps
        N = X.size(0)
        M = self.hparams.num_models
        output_dim = X.size(1) * 2
        embeddings = []
        # preallocate an array to hold the results
        results = torch.zeros((M, N, output_dim), dtype=X.dtype, device=X.device)
        for index, model in enumerate(self.models):
            submodel_out, temp_z = model(X)
            temp_output = self.ensemble_output[index](submodel_out)
            results[index] = temp_output
            embeddings.append(temp_z)
        mean_y, var_y = results.split(X.size(1), dim=-1)
        ensemble_mean = mean_y.mean(dim=0)
        ensemble_var = (var_y + mean_y**2.0).mean(dim=0) - ensemble_mean**2.0
        return ensemble_mean, ensemble_var, results, embeddings

    def _submodel_reconstruction(self, batch):
        (X, Y, params) = batch
        submodel_losses = torch.zeros(X.size(1), dtype=X.dtype, device=X.device)
        for index, submodel in enumerate(self.models):
            recon, z = submodel(X)
            fake_var = torch.ones_like(recon, requires_grad=True)
            submodel_losses[index] = self.metric(recon, Y[:, :, 1], fake_var)
        return submodel_losses.sum()

    def step(self, batch, batch_idx: int, prefix: str):
        (X, Y, params) = batch
        mean_pred, var_pred, results, embeddings = self(X)
        nll = self.metric(mean_pred, Y[:, :, 1], var_pred)
        submodel_loss = self._submodel_reconstruction(batch)
        self.log(f"{prefix}", {"nll": nll, "submodel": submodel_loss})
        loss = nll + submodel_loss
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, "training")
        return loss

    def validation_step(self, batch, batch_idx):
        # loss, log = self.step(batch, batch_idx, "validation")
        loss = self.step(batch, batch_idx, "validation")
        if random.uniform(0, 1) < self.hparams.plot_percent:
            (X, Y, g2_params) = batch
            pred_Y, var_y, results, embeddings = self(X)
            pred_Y = pred_Y[5].cpu()
            var_y = var_y[5].cpu()
            Y = Y.cpu()[5]
            X = X.cpu()[5]
            fig, ax = plt.subplots()
            upper = pred_Y + var_y
            lower = pred_Y - var_y
            ax.plot(X[:, 0], X[:, 1], label="Input")
            ax.plot(X[:, 0], pred_Y[:], label="Model")
            ax.plot(Y[:, 0], Y[:, 1], label="Target")
            ax.plot(
                X[:, 0], upper, label="mean+var", color="#2b8cbe", alpha=0.6, ls="--"
            )
            ax.plot(
                X[:, 0], lower, label="mean-var", color="#2b8cbe", alpha=0.6, ls="--"
            )
            ax.set_xscale("log")
            # ax.set_ylim([0.5, 1.1])
            ax.set_xlabel("time (ps)")
            ax.set_ylabel("g2")
            # log.update({"spectrum": fig})
            if hasattr(self.logger, "experiment"):
                self.logger.experiment.log({"spectrum": fig})
            del fig
            del ax
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("Ensemble")
        group.add_argument("--ens_dropout", type=float, default=0.4)
        group.add_argument("--num_models", type=int, default=5)
        return parent_parser


class MLPEnsemble(EnsembleModel):
    def __init__(
        self,
        input_dim: int,
        num_models: int,
        num_layers: int = 3,
        z_dim: int = 64,
        dropout: float = 0.1,
        norm: bool = True,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        plot_percent: float = 0.05,
        activation: str = "SiLU",
        activation_args: str = "",
        ens_dropout: float = 0.4,
    ) -> None:
        base_model = MLPAutoEncoder(
            input_dim,
            z_dim,
            num_layers,
            dropout,
            norm,
            lr,
            weight_decay,
            activation,
            activation_args,
        )
        super().__init__(
            base_model,
            num_models,
            lr=lr,
            weight_decay=weight_decay,
            plot_percent=plot_percent,
            ens_dropout=ens_dropout,
        )
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[
                parent_parser,
            ],
            add_help=False,
        )
        parser = MLPAutoEncoder.add_model_specific_args(parser)
        parser.add_argument("--num_models", type=int, default=5)
        parser.add_argument("--ens_dropout", type=float, default=0.4)
        return parser


class ConvMLPEnsemble(EnsembleModel):
    def __init__(
        self,
        input_dim: int,
        num_models: int,
        num_channels: int = 64,
        num_layers: int = 7,
        dropout: float = 0.0,
        norm: bool = True,
        lr: float = 0.001,
        weight_decay: float = 0,
        plot_percent: float = 0.05,
        activation: str = "SiLU",
        activation_args: str = "",
    ) -> None:
        base_model = ConvMLPAutoEncoder(
            input_dim,
            num_layers=num_layers,
            num_channels=num_channels,
            dropout=dropout,
            norm=norm,
            activation=activation,
            activation_args=activation_args,
        )
        super().__init__(
            base_model,
            num_models,
            lr=lr,
            weight_decay=weight_decay,
            plot_percent=plot_percent,
        )
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[
                parent_parser,
            ],
            add_help=False,
        )
        parser = ConvMLPAutoEncoder.add_model_specific_args(parser)
        parser.add_argument("--num_models", type=int, default=5)
        return parser


class AdversarialEnsemble(EnsembleModel):
    def __init__(
        self,
        base_model: nn.Module,
        discriminator: nn.Module,
        num_models: int,
        lr: float = 0.001,
        weight_decay: float = 0,
        plot_percent: float = 0.05,
        ens_dropout: float = 0.0,
        adv_weight: float = 1.0,
        anneal_max: int = 50_000,
    ) -> None:
        super().__init__(
            base_model,
            num_models,
            lr=lr,
            weight_decay=weight_decay,
            plot_percent=plot_percent,
            ens_dropout=ens_dropout,
        )
        # this is a standin; replace with specific discriminator model
        self.discriminator = discriminator
        self.save_hyperparameters()

    def configure_optimizers(self):
        """
        Configure a pair of optimizers, one for the discriminator model,
        and another for the actual ensemble.
        """
        # combine the output and the submodel layers
        ens_parameters = chain(
            self.models.parameters(), self.ensemble_output.parameters()
        )
        ens_optimizer = torch.optim.Adam(
            ens_parameters, self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        discrim_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            1e-3,
            weight_decay=self.hparams.weight_decay,
        )
        annealer = torch.optim.lr_scheduler.CosineAnnealingLR(
            ens_optimizer, T_max=self.hparams.anneal_max
        )
        schedulers = {"scheduler": annealer, "name": "ensemble_annealing"}
        return [ens_optimizer, discrim_optimizer], schedulers

    def _adversarial_step(self, batch):
        (X, Y, params) = batch
        batch_size = Y.size(0)
        valid = torch.ones(batch_size, 1, device=Y.device, dtype=Y.dtype)
        real_loss = F.binary_cross_entropy(self.discriminator(Y[:, :, 1]), valid)

        fake_losses = torch.zeros(
            self.hparams.num_models, dtype=X.dtype, device=X.device
        )
        for index, submodel in enumerate(self.models):
            recon, z = submodel(X)
            fake = torch.zeros_like(valid)
            fake_losses[index] = F.binary_cross_entropy(self.discriminator(recon), fake)
        d_loss = real_loss + fake_losses.sum()
        return d_loss

    def _submodel_reconstruction(self, batch):
        """
        This is modified from the ensemble base method by including an adversarial
        component to the submodel reconstruction loss. The goal here is to have
        each submodel learn to produce real looking spectra by minimizing a
        reconstruction error in the form of a homoskedastic Gaussian NLL, and
        by fooling a discriminator.
        """
        (X, Y, params) = batch
        submodel_losses = torch.zeros(X.size(0), dtype=X.dtype, device=X.device)
        for index, submodel in enumerate(self.models):
            temp_recon, z = submodel(X)
            recon, var = self.ensemble_output[index](temp_recon).split(
                X.size(1), dim=-1
            )
            fake_var = torch.ones_like(recon, requires_grad=True)
            recon_loss = self.metric(recon, Y[:, :, 1], fake_var)
            # find out how the generators can improve wr.t fooling
            # the discriminator
            fake_labels = torch.ones_like(submodel_losses).unsqueeze(-1)
            discrim_loss = F.binary_cross_entropy(
                self.discriminator(recon), fake_labels
            )
            submodel_losses[index] = discrim_loss * self.hparams.adv_weight
        return submodel_losses.mean()

    def _ensemble_reconstruction(self, batch):
        """
        Compute the loss for the ensemble, as the sum of reconstruction
        and discriminator terms. The former constitutes the Gaussian NLL
        with ensemble variance, and the latter ensures that the ensemble
        averages also look reasonable.
        """
        (X, Y, params) = batch
        mean_pred, var_pred, results, embeddings = self(X)
        recon_loss = self.metric(mean_pred, Y[:, :, 1], var_pred)
        fake_labels = torch.ones(X.size(0), 1, dtype=X.dtype, device=X.device)
        discrim_loss = F.binary_cross_entropy(
            self.discriminator(mean_pred), fake_labels
        )
        return recon_loss + discrim_loss

    def _encoding_loss(self, batch):
        (X, Y, params) = batch
        encoding_loss = 0
        for submodel in self.models:
            noisy_embeddings = submodel.encode(X)
            true_embeddings = submodel.encode(Y)
            encoding_loss += F.mse_loss(noisy_embeddings, true_embeddings)
        return encoding_loss

    def step(self, batch, batch_idx: int, optimizer_idx: int, prefix: str):
        # for the first pass, we update the reconstruction
        if optimizer_idx == 0:
            submodel_loss = self._submodel_reconstruction(batch)
            ensemble_loss = self._ensemble_reconstruction(batch)
            encoding_loss = self._encoding_loss(batch)
            loss = submodel_loss + ensemble_loss + encoding_loss
            logs = {
                key: value
                for key, value in zip(
                    ["submodel", "ensemble", "encoding"],
                    [submodel_loss, ensemble_loss, encoding_loss],
                )
            }
            self.log(f"{prefix}", logs, on_epoch=True, on_step=False)
        # second pass is adversarial training
        else:
            loss = self._adversarial_step(batch)
            self.log(f"{prefix}_adver", loss, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        loss = self.step(batch, batch_idx, optimizer_idx, "training")
        return loss

    def validation_step(self, batch, batch_idx):
        # for validation we only look at the reconstruction
        loss = self.step(batch, batch_idx, 0, "validation")
        if random.uniform(0, 1) < self.hparams.plot_percent:
            (X, Y, g2_params) = batch
            pred_Y, var_y, results, embeddings = self(X)
            pred_Y = pred_Y[5].cpu()
            var_y = var_y[5].cpu()
            std_y = np.sqrt(var_y)
            Y = Y.cpu()[5]
            X = X.cpu()[5]
            fig, ax = plt.subplots()
            upper = pred_Y + 2 * std_y
            lower = pred_Y - 2 * std_y
            ax.plot(X[:, 0], X[:, 1], label="Input")
            ax.plot(X[:, 0], pred_Y[:], label="Model", color="r")
            ax.plot(Y[:, 0], Y[:, 1], label="Target", color="k")
            ax.plot(X[:, 0], upper, label="mean+var", color="k", alpha=0.5, ls="--")
            ax.plot(X[:, 0], lower, label="mean-var", color="k", alpha=0.5, ls="--")
            ax.set_xscale("log")
            # ax.set_ylim([0.5, 1.1])
            ax.set_xlabel("time (ps)")
            ax.set_ylabel("g2")
            if hasattr(self.logger, "experiment"):
                self.logger.experiment.log({"spectrum": fig})
            del fig
            del ax
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = EnsembleModel.add_model_specific_args(parent_parser)
        group = parser.add_argument_group("Adversarial")
        group.add_argument("--adv_weight", type=float, default=1.0)
        group.add_argument("--anneal_max", type=int, default=50_000)
        return parser


class AdvMLPEnsemble(AdversarialEnsemble):
    def __init__(
        self,
        input_dim: int,
        num_models: int,
        num_layers: int = 3,
        z_dim: int = 64,
        dropout: float = 0.0,
        norm: bool = True,
        lr: float = 0.001,
        weight_decay: float = 0,
        plot_percent: float = 0.10,
        activation: str = "SiLU",
        activation_args: str = "",
        ens_dropout: float = 0.4,
        adv_weight: float = 1.0,
        anneal_max: int = 50_000,
    ) -> None:
        base_model = MLPAutoEncoder(
            input_dim,
            z_dim,
            num_layers,
            dropout,
            norm,
            lr,
            weight_decay,
            activation,
            activation_args,
        )
        example_input_array = torch.zeros(64, input_dim, 2)
        conv_discrim = ConvEncoder(1, 128, 5)
        discrim_z_size = get_conv_output_size(conv_discrim, torch.zeros(32, input_dim))
        discriminator = nn.Sequential(
            conv_discrim, nn.Linear(discrim_z_size, 1), nn.Sigmoid()
        )
        super().__init__(
            base_model,
            discriminator=discriminator,
            num_models=num_models,
            lr=lr,
            weight_decay=weight_decay,
            plot_percent=plot_percent,
            ens_dropout=ens_dropout,
        )
        self.example_input_array = example_input_array
        self.apply(init_fc_layers)
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add adversrial ensemble specific arguments
        parser = AdversarialEnsemble.add_model_specific_args(parent_parser)
        # now add things like number of layers, z_dim, etc
        parser = MLPAutoEncoder.add_model_specific_args(parser)
        return parser


class AdvConvMLPEnsemble(AdversarialEnsemble):
    def __init__(
        self,
        input_dim: int = 140,
        encoder_layers: int = 6,
        decoder_layers: int = 3,
        num_channels: int = 64,
        kernel_size: int = 5,
        kernel_size2: int = 3,
        dropout: float = 0.0,
        norm: bool = True,
        lr: float = 2e-4,
        weight_decay: float = 0.0,
        activation: str = "SiLU",
        activation_args: str = "",
        plot_percent=0.0,
        ens_models: int = 2,
        ens_dropout: float = 0.4,
        adv_weight: float = 1.0,
        anneal_max: int = 50_000,
    ) -> None:
        base_model = ConvMLPAutoEncoder(
            input_dim,
            encoder_layers,
            decoder_layers,
            num_channels,
            kernel_size,
            kernel_size2,
            dropout,
            norm,
            lr,
            weight_decay,
            activation,
            activation_args,
        )

        example_input_array = torch.zeros(64, input_dim, 2)
        conv_discrim = ConvEncoder(1, 128, 5)
        discrim_z_size = get_conv_output_size(conv_discrim, torch.zeros(32, input_dim))
        discriminator = nn.Sequential(
            conv_discrim, nn.Linear(discrim_z_size, 1), nn.Sigmoid()
        )
        super().__init__(
            base_model,
            discriminator=discriminator,
            num_models=ens_models,
            lr=lr,
            weight_decay=weight_decay,
            plot_percent=plot_percent,
            ens_dropout=ens_dropout,
        )
        self.example_input_array = example_input_array
        self.apply(init_fc_layers)
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add adversrial ensemble specific arguments
        parser = AdversarialEnsemble.add_model_specific_args(parent_parser)
        # now add things like number of layers, z_dim, etc
        parser = MLPAutoEncoder.add_model_specific_args(parser)
        return parser


class AdvConv1DEnsemble(AdversarialEnsemble):
    def __init__(
        self,
        input_dim: int = 140,
        num_layers: int = 4,
        num_channels: int = 64,
        z_dim: int = 32,
        kernel1: int = 11,
        kernel2: int = 3,
        dropout: float = 0.1,
        bottleneck: bool = True,
        norm: bool = True,
        lr: float = 2e-4,
        weight_decay: float = 0.0,
        activation: str = "SiLU",
        activation_args: str = "",
        plot_percent=0.05,
        ens_models: int = 2,
        ens_dropout: float = 0.4,
        adv_weight: float = 1.0,
        anneal_max: int = 50_000,
    ) -> None:
        base_model = Conv1DAutoEncoder(
            input_dim,
            num_layers,
            num_channels,
            z_dim,
            kernel1,
            kernel2,
            dropout,
            bottleneck,
            norm,
            activation,
            activation_args,
        )

        example_input_array = torch.zeros(64, input_dim, 2)
        conv_discrim = ConvEncoder(1, 128, 5)
        discrim_z_size = get_conv_output_size(conv_discrim, torch.zeros(32, input_dim))
        discriminator = nn.Sequential(
            conv_discrim, nn.Linear(discrim_z_size, 1), nn.Sigmoid()
        )
        super().__init__(
            base_model,
            discriminator=discriminator,
            num_models=ens_models,
            lr=lr,
            weight_decay=weight_decay,
            plot_percent=plot_percent,
            ens_dropout=ens_dropout,
            adv_weight=adv_weight,
            anneal_max=anneal_max,
        )
        self.example_input_array = example_input_array
        self.apply(init_fc_layers)
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add adversrial ensemble specific arguments
        parser = AdversarialEnsemble.add_model_specific_args(parent_parser)
        # now add things like number of layers, z_dim, etc
        parser = MLPAutoEncoder.add_model_specific_args(parser)
        return parser


class AdvEnsemble(AdversarialEnsemble):
    def __init__(
        self,
        input_base_model,
        base_model_config,
        ens_lr: float = 5e-4,
        ens_weight_decay: float = 0.0,
        ens_plot_percent=0.10,
        ens_models: int = 2,
        ens_dropout: float = 0.4,
    ) -> None:
        base_model = input_base_model(**base_model_config)

        input_dim = base_model.hparams["input_dim"]
        example_input_array = torch.zeros(64, input_dim, 2)
        conv_discrim = ConvEncoder(1, 128, 5)
        discrim_z_size = get_conv_output_size(conv_discrim, torch.zeros(32, input_dim))
        discriminator = nn.Sequential(
            conv_discrim, nn.Linear(discrim_z_size, 1), nn.Sigmoid()
        )
        super().__init__(
            base_model,
            discriminator=discriminator,
            num_models=ens_models,
            lr=ens_lr,
            weight_decay=ens_weight_decay,
            plot_percent=ens_plot_percent,
            ens_dropout=ens_dropout,
        )
        self.example_input_array = example_input_array
        self.apply(init_fc_layers)
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add adversrial ensemble specific arguments
        parser = AdversarialEnsemble.add_model_specific_args(parent_parser)
        # now add things like number of layers, z_dim, etc
        parser = MLPAutoEncoder.add_model_specific_args(parser)
        return parser


class EnsembleSubmodels(pl.LightningModule):
    def __init__(
        self,
        base_model: nn.Module,
        num_models: int,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        plot_percent: float = 0.10,
        ens_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.models = nn.ModuleList(base_model)
        self.metric = nn.GaussianNLLLoss()
        input_dim = base_model[0].hparams.input_dim
        # want the mean and variance out
        output_dim = input_dim * 2

        # this compacts the mean and variance calculation into a single
        # batched operation, and reshapes it later
        self.ensemble_output = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(ens_dropout),
                    nn.Linear(input_dim, output_dim),
                    nn.Softplus()
                    # nn.ReLU()
                )
                for _ in range(num_models)
            ]
        )
        self.apply(init_fc_layers)
        self.save_hyperparameters()

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor]:
        # number of models in ensemble, batch size, number of timesteps
        N = X.size(0)
        M = self.hparams.num_models
        output_dim = X.size(1) * 2
        embeddings = []
        # preallocate an array to hold the results
        results = torch.zeros((M, N, output_dim), dtype=X.dtype, device=X.device)
        for index, model in enumerate(self.models):
            submodel_out, temp_z = model(X)
            temp_output = self.ensemble_output[index](submodel_out)
            results[index] = temp_output
            embeddings.append(temp_z)
        mean_y, var_y = results.split(X.size(1), dim=-1)
        ensemble_mean = mean_y.mean(dim=0)
        ensemble_var = (var_y + mean_y**2.0).mean(dim=0) - ensemble_mean**2.0
        return ensemble_mean, ensemble_var, results, embeddings

    def _submodel_reconstruction(self, batch):
        (X, Y, params) = batch
        submodel_losses = torch.zeros(X.size(1), dtype=X.dtype, device=X.device)
        for index, submodel in enumerate(self.models):
            recon, z = submodel(X)
            fake_var = torch.ones_like(recon, requires_grad=True)
            submodel_losses[index] = self.metric(recon, Y[:, :, 1], fake_var)
        return submodel_losses.sum()

    def step(self, batch, batch_idx: int, prefix: str):
        (X, Y, params) = batch
        mean_pred, var_pred, results, embeddings = self(X)
        nll = self.metric(mean_pred, Y[:, :, 1], var_pred)
        submodel_loss = self._submodel_reconstruction(batch)
        self.log(f"{prefix}", {"nll": nll, "submodel": submodel_loss})
        loss = nll + submodel_loss
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, "training")
        return loss

    def validation_step(self, batch, batch_idx):
        # loss, log = self.step(batch, batch_idx, "validation")
        loss = self.step(batch, batch_idx, "validation")
        if random.uniform(0, 1) < self.hparams.plot_percent:
            (X, Y, g2_params) = batch
            pred_Y, var_y, results, embeddings = self(X)
            pred_Y = pred_Y[5].cpu()
            var_y = var_y[5].cpu()
            std_y = np.sqrt(var_y)
            Y = Y.cpu()[5]
            X = X.cpu()[5]
            fig, ax = plt.subplots()
            upper = pred_Y + 2 * std_y
            lower = pred_Y - 2 * std_y
            ax.plot(X[:, 0], X[:, 1], label="Input")
            ax.plot(X[:, 0], pred_Y[:], label="Model")
            ax.plot(Y[:, 0], Y[:, 1], label="Target")
            ax.plot(
                X[:, 0], upper, label="mean+var", color="#2b8cbe", alpha=0.6, ls="--"
            )
            ax.plot(
                X[:, 0], lower, label="mean-var", color="#2b8cbe", alpha=0.6, ls="--"
            )
            ax.set_xscale("log")
            # ax.set_ylim([0.5, 1.1])
            ax.set_xlabel("time (ps)")
            ax.set_ylabel("g2")
            # log.update({"spectrum": fig})
            if hasattr(self.logger, "experiment"):
                self.logger.experiment.log({"spectrum": fig})
            del fig
            del ax
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("Ensemble")
        group.add_argument("--ens_dropout", type=float, default=0.4)
        group.add_argument("--num_models", type=int, default=5)
        return parent_parser


models = {
    "GRUAutoEncoder": GRUAutoEncoder,
    "GRUVAE": GRUVAE,
    "ConvLSTMAutoEncoder": ConvLSTMAutoEncoder,
    "CLVAE": CLVAE,
    "ConvLSTMEncoder": ConvLSTMEncoder,
    "MLPAutoEncoder": MLPAutoEncoder,
    "MLPAutoEncoderAdversary": MLPAutoEncoderAdversary,
    "ConvMLPAutoEncoder": ConvMLPAutoEncoder,
    "ConvMLPAutoEncoderAdversary": ConvMLPAutoEncoderAdversary,
    "EnsembleModel": EnsembleModel,
    "MLPEnsemble": MLPEnsemble,
    "ConvMLPEnsemble": ConvMLPEnsemble,
    "AdvMLPEnsemble": AdvMLPEnsemble,
    "AdvConvMLPEnsemble": AdvConvMLPEnsemble,
    "Conv1DAutoEncoder": Conv1DAutoEncoder,
    "AdvConv1DEnsemble": AdvConv1DEnsemble,
    "AdvEnsemble": AdvEnsemble,
}

if __name__ == "__main__":

    X = torch.randn(10, 201)

    # # Test MLP
    # model = MLP(
    #     input_size=702,
    #     hidden_size=64,
    #     output_size=36,
    #     depth=3,
    #     dropout=0.1,
    #     norm=True,
    #     lr=1e-4,
    #     weight_decay=1e-4,
    #     activation="ReLU",
    # )
    # pred = model(X)
    # print(pred.shape)

    # # Test ConvMLP
    # model = ConvMLP(
    #     input_size=702,
    #     channels=4,
    #     depth=3,
    #     kernels=[5, 3, 3, 3, 3],
    #     downsample=8,
    #     MLP_hidden_size=64,
    #     MLP_output_size=36,
    #     MLP_depth=3,
    #     dropout=0.1,
    #     norm=True,
    #     lr=1e-4,
    #     weight_decay=1e-4,
    #     activation="ReLU",
    # )

    # model = ConvGRU()

    model = GRU_NODE_MLP(
        # input_size=201 * (1 + concat_log),
        input_size=201,
        # input_size=input_size,
        hidden_size=128,
        num_layers=3,
        output_size=1,
        dropout=0.,
        lr=1e-3,
        lr_schedule="RLROP",
        weight_decay=1e-5,
        activation="LeakyReLU",
        bidirectional=True,
        decoder_depth=1,
        plot_interval=25,
        metric=nn.L1Loss,
    )

    pred = model(X)
    print(pred.shape)
    print(model.count_parameters())