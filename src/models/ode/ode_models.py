from typing import Optional, Type
from g2_pcfs.models.base import (
    init_fc_layers,
    get_conv_output_shape,
    get_conv_flat_shape,
)
import numpy as np
import torch
from torch import nn


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class MLPBlock(nn.Module):
    def __init__(
        self,
        out_dim: int,
        activation: Optional[Type[nn.Module]] = None,
        residual: bool = False,
    ) -> None:
        super().__init__()
        linear = nn.LazyLinear(out_dim)
        norm = nn.LazyBatchNorm1d()
        activation = nn.Identity() if activation is None else activation()
        self.model = nn.Sequential(linear, norm, activation)
        self.residual = residual

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        output = self.model(data)
        if self.residual:
            # output += data
            output = output + data
        return output


class MLPStack_old(nn.Module):
    def __init__(
        self,
        out_dim: int,
        depth: int,
        activation: Optional[Type[nn.Module]] = None,
        output_activation: Optional[Type[nn.Module]] = None,
        norm: Optional[Type[nn.Module]] = nn.LayerNorm,
        residual: bool = False,
    ) -> None:
        super().__init__()
        blocks = [MLPBlock(out_dim, activation, residual=False)]
        for _ in range(depth):
            blocks.append(MLPBlock(out_dim, activation, residual=True))
        blocks.append(MLPBlock(out_dim, output_activation, residual=False))
        self.model = nn.Sequential(*blocks)
        self.residual = residual
        self.norm = nn.LazyBatchNorm1d()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        output = self.model(data)
        # skip connection through the full model
        if self.residual:
            # output += data
            output = output + data
        # normalize the result to prevent exploding values
        output = self.norm(output)
        return output


class MLPBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: Optional[Type[nn.Module]] = None,
        norm: bool = True,
        residual: bool = False,
        lazy: bool = False,
    ) -> None:
        super().__init__()
        linear = nn.LazyLinear(output_size) if lazy else nn.Linear(input_size, output_size)
        norm_layer = nn.LazyBatchNorm1d(output_size) if lazy and norm else nn.BatchNorm1d(output_size) if norm else nn.Identity()
        activation_layer = nn.Identity() if activation is None else activation()
        self.model = nn.Sequential(linear, norm_layer, activation_layer)
        self.residual = residual

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        output = self.model(data)
        if self.residual:
            output = output + data
        return output


class MLPStack(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        depth: int,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        output_activation: Optional[Type[nn.Module]] = nn.ReLU,
        norm: bool = True,
        residual: bool = False,
        residual_full: bool = False,
        lazy: bool = False,
    ) -> None:
        super().__init__()
        blocks = [MLPBlock(input_size, hidden_size, activation, norm=norm, residual=residual, lazy=lazy)]
        for _ in range(depth - 1):
            blocks.append(MLPBlock(hidden_size, hidden_size, activation, norm=norm, residual=residual, lazy=False))
        blocks.append(MLPBlock(hidden_size, output_size, output_activation, norm=norm, residual=False, lazy=False))
        self.model = nn.Sequential(*blocks)
        self.residual_full = residual_full
        self.norm = nn.BatchNorm1d(output_size) if norm else nn.Identity()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        output = self.model(data)
        if self.residual_full:
            output = output + data
        output = self.norm(output)
        return output


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int = 20,
        output_size: int = None,
        hidden_size: int = 250,
        depth: int = 1,
        dropout: float = 0.0,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        norm: Optional[Type[nn.Module]] = nn.LayerNorm,
    ):
        super(MLP, self).__init__()

        output_size = input_size if output_size is None else output_size
        activation = nn.Identity if activation is None else activation
        norm = nn.Identity if norm is None else nn.LayerNorm
        dropout_layer = nn.Identity() if dropout == 0.0 else nn.Dropout(dropout)

        layers = []
        layers.append(torch.nn.Linear(input_size, hidden_size))
        layers.append(activation())
        layers.append(norm(hidden_size))

        for i in range(depth - 1):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(activation())
            layers.append(norm(hidden_size))

        layers.append(dropout_layer)  # ffff
        layers.append(torch.nn.Linear(hidden_size, output_size))
        layers.append(activation())
        layers.append(norm(output_size))  # ffff

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x, *args):
        return self.layers(x)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        output_size: int,
        depth: int,
        num_heads: int,
        norm: bool = True,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.projection_layer = nn.LazyLinear(output_size)
        # create some attention heads
        self.heads = nn.ModuleList(
            [
                MLPStack(
                    input_size=output_size,
                    hidden_size=output_size,
                    output_size=output_size,
                    depth=depth,
                    activation=activation,
                    norm=norm,
                    output_activation=activation,
                    residual=True,
                )
                for _ in range(num_heads)
            ]
        )
        self.attention = nn.Sequential(nn.LazyLinear(output_size), nn.Softmax())
        self.transform_layers = MLPStack(
            output_size, output_size, output_size, depth * 2, activation, norm=norm, residual=False, lazy=True
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # project so we can use residual connections
        if data.ndim >= 3:
            data = data.view(data.shape[0], -1)
        projected_values = self.projection_layer(data)
        # stack up the results of each head
        outputs = torch.stack([head(projected_values) for head in self.heads], dim=1)
        weights = self.attention(outputs)
        weighted_values = (weights * outputs).flatten(1)
        return self.transform_layers(weighted_values)


class Conv1DBlock(nn.Module):
    def __init__(
        self,
        out_dim: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: Optional[Type[nn.Module]] = None,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.LazyConv1d(out_dim, kernel, stride, padding)
        self.norm = nn.LazyBatchNorm1d()
        self.activation = nn.Identity() if activation is None else activation()
        self.model = nn.Sequential(self.conv, self.norm, self.activation)
        self.residual = residual

        if self.residual:
            self.downsample = nn.Sequential(
                nn.LazyConv1d(out_dim, 1, stride, bias=False),
                nn.LazyBatchNorm1d(),
            )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Y = self.model(X)
        if self.residual:
            Y += self.downsample(X)
        return Y


class Conv1DStack(nn.Module):
    def __init__(
        self,
        out_dim: int,
        depth: int = 3,
        input_channels: int = 1,
        output_channels: int = 64,
        kernel: int = 3,
        stride: int = 1,
        activation: Optional[Type[nn.Module]] = None,
        output_activation: Optional[Type[nn.Module]] = None,
        residual: bool = False,
    ) -> None:
        super().__init__()
        channels = np.interp(
            np.linspace(0.0, 1.0, depth + 1), [0, 1], [input_channels, output_channels]
        ).astype(int)

        blocks = [
            Conv1DBlock(
                channels[1], kernel, stride, activation=activation, residual=False
            )
        ]
        for idx in range(1, depth - 1):
            blocks.append(
                Conv1DBlock(
                    channels[idx + 1],
                    kernel,
                    stride,
                    activation=activation,
                    residual=residual,
                )
            )
        blocks.append(
            Conv1DBlock(
                channels[-1],
                kernel,
                stride,
                activation=output_activation,
                residual=False,
            )
        )

        # flattening and connect to linear layer to recover input size
        blocks.append(nn.Flatten())
        blocks.append(nn.LazyLinear(out_dim))

        self.model = nn.Sequential(*blocks)
        self.residual = residual
        self.norm = nn.LazyBatchNorm1d()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim == 2:
            X = X.unsqueeze(1)  # create channel index
        Y = self.model(X)
        # Y = Y.unsqueeze(1)
        # skip connection through the full model
        if self.residual:
            Y += X.mean(dim=1)
        # normalize the result to prevent exploding values
        Y = self.norm(Y).unsqueeze(1)

        return Y


class SmallMLP(nn.Module):
    def __init__(
        self,
        out_dim: int,
        nchannels: int,
        MLP_dim: int = 50,
        dropout_rate: float = 0.0,
        norm: Optional[Type[nn.Module]] = None,
        activation: Optional[Type[nn.Module]] = nn.Tanh,
        output_activation: Optional[Type[nn.Module]] = None,
        residual: bool = False,
        prior_const: torch.Tensor = None,
        flat_const: torch.Tensor = None,
    ) -> None:
        super().__init__()

        norm = nn.Identity() if norm is None else nn.LazyBatchNorm1d()
        activation = nn.Identity() if activation is None else activation()
        output_activation = (
            nn.Identity() if output_activation is None else output_activation()
        )

        self.out_dim = out_dim
        self.nchannels = nchannels
        self.MLP_dim = MLP_dim
        self.dropout_rate = dropout_rate
        self.residual = residual
        self.prior_const = prior_const
        self.flat_const = flat_const

        self.model = nn.Sequential(
            nn.LazyLinear(self.MLP_dim),
            nn.Dropout(self.dropout_rate),
            norm,
            activation,
            nn.LazyLinear(self.MLP_dim),
            nn.Dropout(self.dropout_rate),
            norm,
            activation,
            nn.LazyLinear(self.out_dim),
            output_activation,
        )

        # from Chen et al. Nature Communications volume 13, Article number: 1016 (2022)
        # self.model = nn.Sequential(
        #     nn.Linear(self.out_dim, self.MLP_dim),
        #     nn.Dropout(self.dropout_rate),
        #     nn.Tanh(),
        #     nn.Linear(self.MLP_dim, self.MLP_dim),
        #     nn.Dropout(self.dropout_rate),
        #     nn.Tanh(),
        #     nn.Linear(self.MLP_dim, self.out_dim),
        # )

    def forward(self, X: torch.Tensor, **args) -> torch.Tensor:
        Y = self.model(X)
        if self.prior_const is not None:
            Y = torch.cat(
                (Y, self.prior_const), dim=-1
            )  # need to pad with ones when input and output sizes are different, to work with NeuralODE functions
        if self.flat_const is not None:
            Y = torch.cat((Y, self.flat_const), dim=-1)  # same for flattened shapes
        if self.residual:
            Y += X
        return Y


class Conv2DAutoEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        nchannels: int,
        input_channels: int = 1,
        output_channels: int = 10,
        kernel_size: int = 3,
        norm: bool = None,
        bottleneck: bool = False,
        z_dim: int = 32,
        activation: Optional[Type[nn.Module]] = None,
        output_activation: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()

        norm = nn.Identity() if norm is None else nn.LazyBatchNorm2d()
        activation = nn.Identity() if activation is None else activation()
        output_activation = (
            nn.Identity() if output_activation is None else output_activation()
        )

        self.dim = dim
        self.nchannels = nchannels

        channels = np.interp(
            np.linspace(0.0, 1.0, 4), [0, 1], [input_channels, output_channels]
        ).astype(int)

        k1 = ((nchannels - nchannels // 2), kernel_size)
        k2 = (1, k1[-1])
        stride = (1, 2)
        padding = (0, 1)

        # TODO: try something like norm = nn.LazyBatchNorm2d().copy() or something
        self.encoder = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], k1, stride, padding),
            # nn.BatchNorm2d(channels[1]),
            nn.LazyBatchNorm2d(),
            activation,
            nn.Conv2d(channels[1], channels[2], k1, stride, padding),
            # nn.BatchNorm2d(channels[2]),
            nn.LazyBatchNorm2d(),
            activation,
            nn.Conv2d(channels[2], channels[3], k2),
            nn.LazyBatchNorm2d(),
            norm,
            activation,
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels[3], channels[2], k2),
            # nn.BatchNorm2d(channels[2]),
            # norm,
            activation,
            nn.ConvTranspose2d(
                channels[2], channels[1], k1, stride, padding, output_padding=padding
            ),
            # nn.BatchNorm2d(channels[1]),
            # norm,
            activation,
            nn.ConvTranspose2d(
                channels[1], channels[0], k1, stride, padding, output_padding=padding
            ),
            # nn.BatchNorm2d(channels[0]),
            # norm,
            activation,
            output_activation,
        )

        if bottleneck:
            dummy_input = torch.ones((1, 1, nchannels, dim))
            conv_shape = get_conv_output_shape(self.encoder, dummy_input)
            flat_shape = get_conv_flat_shape(self.encoder, dummy_input)

            self.linear_bottleneck = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_shape[0], z_dim),
                nn.Linear(z_dim, flat_shape[0]),
                Reshape(-1, conv_shape[1], conv_shape[2], conv_shape[3]),
            )
        else:
            self.linear_bottleneck = nn.Sequential()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim == 3:
            X = X.unsqueeze(1)  # create channel index
        z = self.encoder(X)
        Y = self.decoder(z)
        Y = Y.squeeze(1)  # delete channel index
        return Y


class ODEnet(nn.Module):
    def __init__(
        self,
        nchannels: int = 1,
        dim: int = 116,
        p_dim: int = 100,
        MLP_dim: int = 50,
        activation: Optional[Type[nn.Module]] = nn.Tanh,
    ) -> None:
        super().__init__()

        activation = nn.Identity() if activation is None else activation()

        self.nchannels = nchannels
        self.dim = dim
        self.p_dim = p_dim
        self.MLP_dim = MLP_dim

        self.model_X = nn.Sequential(
            nn.Linear(self.dim, self.MLP_dim),
            activation,
            nn.Linear(self.MLP_dim, self.MLP_dim),
            activation,
            nn.Linear(self.MLP_dim, self.dim),
        )

        self.model_P = nn.Sequential(
            nn.Linear(self.p_dim, self.MLP_dim),
            activation,
            nn.Linear(self.MLP_dim, self.MLP_dim),
            activation,
            nn.Linear(self.MLP_dim, self.dim),
        )

        self.model_tot = nn.Sequential(
            nn.Linear(self.dim, self.MLP_dim),
            activation,
            nn.Linear(self.MLP_dim, self.dim),
        )

        # self.model_x_and_h = nn.Sequential(
        #     nn.Linear(dim+p_dim, MLP_dim),
        #     activation,
        #     nn.Linear(MLP_dim, MLP_dim),
        #     activation,
        #     nn.Linear(MLP_dim, dim),
        # )

    def forward(self, t, X: torch.Tensor) -> torch.Tensor:
        if self.P is not None:
            P = self.P
            Y = self.model_X(X) + self.model_P(P)
        else:
            Y = self.model_X(X)
        dx = self.model_tot(Y)
        Y = dx
        return Y


class ResBlock2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=(3, 3),
        stride=1,
        downsample=None,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        dropout=0.0,
        residual: bool = True,
    ) -> None:
        super(ResBlock2d, self).__init__()

        self.residual = residual
        self.activation = nn.Identity() if activation is None else activation()
        padding = tuple(k // 2 for k in kernel)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            self.activation,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel,
                stride=1,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        self.downsample = downsample
        self.out_channels = out_channels

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]  # get only x, ignore residual that is fed back into forward pass
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        if self.residual:  # forward skip connection
            out += residual
        out = self.activation(out)
        return out, residual


class ResNet2D(nn.Module):
    def __init__(
        self,
        block: nn.Module = ResBlock2d,
        depth: int = 4,
        channels: list = [1, 64, 128, 256, 512],
        kernels: list = [3, 3, 3, 3, 3],
        strides: list = [1, 1, 1, 1, 1],
        dropout: float = 0.0,
        activation=nn.ReLU,
        norm=True,
        residual: bool = True,
    ) -> None:
        super(ResNet2D, self).__init__()
        self.depth = depth
        self.inplanes = channels[0]

        self.layers = nn.ModuleDict({})
        for i in range(0, self.depth):
            self.layers[str(i)] = self._make_layer(
                block,
                channels[i + 1],
                kernel=(kernels[i], kernels[i]),
                stride=(strides[i], strides[i]),
                dropout=dropout,
                activation=activation,
                norm=norm,
                residual=residual,
            )

    def _make_layer(
        self, block, planes, kernel, stride, dropout, activation, norm, residual
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes) if norm else nn.Identity(),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                kernel,
                stride,
                downsample,
                activation,
                dropout,
                residual,
            )
        )
        self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        residuals = []
        for i in range(0, self.depth):
            x, res = self.layers[str(i)](x)
            residuals.append(res)
        return x, residuals


class ResBlock1dT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=3,
        stride=1,
        upsample=None,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        dropout=0,
        residual: bool = True,
    ) -> None:
        super(ResBlock1dT, self).__init__()

        self.residual = residual
        self.activation = nn.Identity() if activation is None else activation()
        padding = kernel // 2

        self.convt1 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels,
                in_channels,
                kernel_size=kernel,
                stride=1,
                padding=padding,
                output_padding=0,
            ),
            nn.BatchNorm1d(in_channels),
            self.activation,
        )
        self.convt2 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                output_padding=stride - 1,
            ),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout),
        )
        self.upsample = upsample
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.convt1(x)
        out = self.convt2(out)
        if self.upsample:
            residual = self.upsample(x)
        if self.residual:
            out += residual
        out = self.activation(out)
        return out


class ResNet1DT(nn.Module):
    def __init__(
        self,
        block: nn.Module = ResBlock1dT,
        depth: int = 4,
        channels: list = [512, 256, 128, 64, 1],
        kernels: list = [3, 3, 3, 3, 3],
        strides: list = [1, 1, 1, 1, 1],
        dropout: float = 0.0,
        activation=nn.ReLU,
        norm=True,
        sym_residual: bool = True,
        fwd_residual: bool = True,
    ) -> None:
        super(ResNet1DT, self).__init__()
        self.depth = depth
        self.inplanes = channels[0]
        self.sym_residual = sym_residual  # for symmetric skip connections
        self.fwd_residual = fwd_residual  # for forward skip connections

        self.layers = nn.ModuleDict({})
        for i in range(0, self.depth):
            self.layers[str(i)] = self._make_layer(
                block,
                channels[i + 1],
                kernel=kernels[i],
                stride=strides[i],
                dropout=dropout,
                activation=activation,
                norm=norm,
                residual=fwd_residual,
            )

    def _make_layer(
        self, block, planes, kernel, stride, dropout, activation, norm, residual
    ):
        upsample = None
        if stride != 1 or self.inplanes != planes:
            upsample = nn.Sequential(
                nn.ConvTranspose1d(
                    self.inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    output_padding=stride - 1,
                ),
                nn.BatchNorm1d(planes) if norm else nn.Identity(),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                kernel,
                stride,
                upsample,
                activation,
                dropout,
                residual,
            )
        )
        self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x, residuals):
        for i in range(0, self.depth):
            if self.sym_residual:  # symmetric skip connection
                res = residuals[-1 - i]
                if res.ndim > x.ndim:  # for 3D to 2D
                    res = torch.mean(res, dim=2)
                x = x + res
            x = self.layers[str(i)](x)
        return x


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        depth,
        last_seq: bool = False,
    ):
        super(LSTMEncoder, self).__init__()
        self.last_seq = last_seq
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, depth, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.actv_layer = nn.PReLU()

    def forward(self, x):
        output, hidden = self.lstm(x)
        output = self.fc(output[:, -1] if self.last_seq else output)
        output = self.actv_layer(output)
        return output, hidden


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        depth,
    ):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, depth, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None):
        # Initialize hidden state and cell state
        if h0 is None:
            h0 = (
                torch.zeros(1, x.size(0), self.hidden_size),
                torch.zeros(1, x.size(0), self.hidden_size),
            )

        # Pass through LSTM
        lstm_out, _ = self.lstm(x, h0)

        # Pass LSTM output through the fully connected layer
        output = self.fc(lstm_out)
        return output

class AugmentLatent(nn.Module):
    def __init__(
        self,
        augment: bool = True,
        augment_dim: int = 0,
        augment_size: int = 1,
        input_size: int = 116,
    ):
        """
        Args:
            augment (bool): Whether to perform augmentation.
            augment_dim (int): The dimension to augment. Choose from 0 or 1.
            augment_size (int): The size of the augmentation.
        """
        super(AugmentLatent, self).__init__()
        self.augment = augment
        self.augment_dim = augment_dim
        self.augment_size = augment_size

    def forward(self, x):
        """
        Augment the input tensor with zeros.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The augmented tensor if self.augment is True, else the input tensor.
        """
        if self.augment:
            if self.augment_dim == 0:
                aug = torch.zeros(x.shape[0], self.augment_size).to(x.device)
            elif self.augment_dim == 1:
                x = x.unsqueeze(1)
                aug = torch.zeros(x.shape[0], self.augment_size, x.shape[-1]).to(
                    x.device
                )
            return torch.cat([x, aug], 1)
        else:
            return x


class ProjectLatent(nn.Module):
    def __init__(
        self,
        project: bool = True,
        project_dim: int = 0,
        project_size: int = 1,
        output_size: int = 116,
    ):
        """
        Args:
            project (bool): Whether to perform projection.
            project_dim (int): The dimension the augmentation to project out. Choose to match augment_dim from AugmentLatent.
            project_size (int): The size of the augmentation to project out. Choose to match augment_size from AugmentLatent.
            output_size (int): The size of the output. Should match the input size of AugmentLatent.
        """
        super(ProjectLatent, self).__init__()
        self.project = project
        self.project_dim = project_dim
        self.project_size = project_size

        if self.project:
            if self.project_dim == 0:
                self.project_layer = nn.Linear(output_size + project_size, output_size)
            elif self.project_dim == 1:
                self.project_layer = nn.Linear(
                    output_size * (1 + project_size), output_size
                )
        else:
            self.project_layer = nn.Identity()

    def forward(self, x):
        """
        Project out the augmentation.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The projected tensor if self.project is True, else the input tensor.
        """
        # return self.project_layer(x.view(x.shape[0], x.shape[1], -1))
        return self.project_layer(x)


if __name__ == "__main__":
    from g2_pcfs.pipeline.image_data import ODEDataModule
    from torchdyn.models import NeuralODE

    # dm = ODEDataModule("pcfs_g2_2d_n300.h5", batch_size=32, add_noise=False)
    dm = ODEDataModule("pcfs_g2_2d_n100.h5", batch_size=32, add_noise=False)

    input_size = 128
    enc_hidden_size = 20
    enc_depth = 2
    z_size = 2**5
    vf_depth = 2
    vf_hidden_size = z_size
    attn_depth = 2
    attn_heads = 4
    dec_hidden_size = 20
    dec_depth = 2
    time_dim = 0
    latent_time_dim = 0
    augment = False
    augment_size = 1

    nobs = 15
    X = torch.randn((32, nobs, input_size))
    obs_inds = [0, 1, 2, 3, 6, 8, 14, 18, 22, 30, 38, 46, 54, 62, 70, 80, 90]
    obs_inds = obs_inds[:nobs]
    t = torch.linspace(0, 1, 100)
    t_obs = t[obs_inds].unsqueeze(0).repeat(X.shape[0], 1).unsqueeze(-1)
    _t_0 = X
    t_0 = torch.cat((_t_0, t_obs), dim=2)

    encoder = ResNet2D(
        depth=3,
        channels=[1, 4, 8, 16],
        kernels=[3, 3, 3],
        strides=[2, 2, 2],
        residual=True,
    )

    # encoder = LSTMEncoder(
    #     input_size=input_size + time_dim,
    #     hidden_size=enc_hidden_size,
    #     depth=enc_depth,
    #     output_size=z_size,
    # )

    attention = AttentionBlock(
        output_size=z_size,
        depth=attn_depth,
        num_heads=attn_heads,
    )

    augment_layer = AugmentLatent(
        augment=augment,
        augment_dim=0,
        augment_size=augment_size,
        input_size=z_size,
    )

    vector_field = MLP(
        input_size=z_size + int(augment) * augment_size,
        hidden_size=vf_hidden_size,
        depth=vf_depth,
    )

    ode = NeuralODE(vector_field)

    project_layer = ProjectLatent(
        project=augment,
        project_dim=0,
        project_size=augment_size,
        output_size=z_size,
    )

    decoder = ResNet1DT(
        depth=3,
        channels=[100, 100, 100, 100],
        kernels=[3, 3, 3, 3],
        strides=[2, 2, 1, 1],
        sym_residual=False,
        fwd_residual=True,
    )

    # decoder = LSTMDecoder(
    #     input_size=z_size + latent_time_dim,
    #     hidden_size=dec_hidden_size,
    #     depth=dec_depth,
    #     output_size=input_size,
    # )

    # raise RuntimeError

    z, h = encoder(t_0)

    z = attention(z)

    z = augment_layer(z)

    # solve ode
    t_span, zt = ode(z, t)

    # # get final time step

    # zt = zt[-1, :, :].unsqueeze(0)
    zt = torch.permute(zt, (1, 0, 2))

    zt = project_layer(zt)

    # if latent_time_dim:
    #     zt = torch.cat((zt, t.unsqueeze(0).repeat(zt.shape[0], 1).unsqueeze(-1)), dim=2)

    d = decoder(zt, h)
