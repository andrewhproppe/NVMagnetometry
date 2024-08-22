from typing import Optional, Type
from src.models.utils import init_fc_layers, get_conv_output_shape, get_conv_flat_shape
from torchdyn.models import NeuralODE
import numpy as np
import torch
from torch import nn


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

### MLP ###
class MLPBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        dropout: float = 0.0,
        activation: Optional[Type[nn.Module]] = None,
        norm: bool = True,
        residual: bool = False,
        lazy: bool = False,
    ) -> None:
        super().__init__()
        linear = nn.LazyLinear(output_size) if lazy else nn.Linear(input_size, output_size)
        norm_layer = nn.LazyBatchNorm1d(output_size) if lazy and norm else nn.BatchNorm1d(output_size) if norm else nn.Identity()
        activation_layer = nn.Identity() if activation is None else activation()
        dropout_layer = nn.Dropout(dropout)
        self.model = nn.Sequential(linear, norm_layer, activation_layer, dropout_layer)
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
        dropout: float = 0.0,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        output_activation: Optional[Type[nn.Module]] = nn.ReLU,
        norm: bool = True,
        residual: bool = False,
        residual_full: bool = False,
        lazy: bool = False,
    ) -> None:
        super().__init__()
        blocks = [MLPBlock(input_size, hidden_size, dropout, activation, norm=norm, residual=False, lazy=lazy)]
        for _ in range(depth - 1):
            blocks.append(MLPBlock(hidden_size, hidden_size, dropout, activation, norm=norm, residual=residual, lazy=False))
        blocks.append(MLPBlock(hidden_size, output_size, dropout, output_activation, norm=norm, residual=False, lazy=False))
        self.model = nn.Sequential(*blocks)
        self.residual_full = residual_full
        self.norm = nn.BatchNorm1d(output_size) if norm else nn.Identity()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        output = self.model(data)
        if self.residual_full:
            output = output + data
        output = self.norm(output)
        return output


### Attention ###
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
            output_size, output_size, output_size, depth * 2, 0, activation, norm=norm, residual=False, lazy=True
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


### CNN / ResNet blocks ###
class ResBlock1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=3,
        stride=1,
        downsample=None,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        dropout=0.0,
        norm=True,
        residual: bool = True,
    ) -> None:
        super(ResBlock1d, self).__init__()

        self.residual = residual
        self.activation = nn.Identity() if activation is None else activation()
        padding = kernel // 2

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                bias=not norm
            ),
            nn.BatchNorm1d(out_channels) if norm else nn.Identity(),
            self.activation,
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel,
                stride=1,
                padding=padding,
                bias=not norm
            ),
            nn.BatchNorm1d(out_channels) if norm else nn.Identity(),
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


class AttnResBlock1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=3,
        stride=1,
        downsample=None,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        dropout=0.0,
        norm: bool = True,
        residual: bool = True,
        attn_on: bool = False,
        attn_depth: int = 1,
        attn_heads: int = 2,
    ) -> None:
        super().__init__()

        # Whether or not to activate ResNet block skip connections
        self.residual = residual
        self.residual_scale = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

        padding = kernel // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=not norm)
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(out_channels) if norm else nn.Identity()
        self.activation = nn.Identity() if activation is None else activation()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel, stride=1, padding=padding, bias=not norm)
        self.bn2 = nn.BatchNorm1d(out_channels) if norm else nn.Identity()

        # Add or skip attention layer based on the use_attention flag
        self.attention = AttentionBlock(out_channels, attn_depth, attn_heads, norm) if attn_on else nn.Identity()

        self.downsample = downsample
        self.out_channels = out_channels

    def forward(self, x):
        x = x[0] if isinstance(x, tuple) else x  # get only x, ignore residual that is fed back into forward pass
        residual = self.downsample(x) if self.downsample else x

        out = self.conv1(x)
        out = self.dropout(out)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)

        residual = self.attention(residual)  # Apply attention if available

        if self.residual:  # forward skip connection
            out += residual*self.residual_scale

        out = self.activation(out)

        return out, residual


class AttnResBlock1dT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=3,
        stride=1,
        upsample=None,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        dropout=0,
        norm=True,
        residual: bool = True,
        attn_on: bool = False,
        attn_depth: int = 1,
        attn_heads: int = 1,
    ) -> None:
        super().__init__()

        self.residual = residual
        self.residual_scale = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.activation = nn.Identity() if activation is None else activation()
        padding = kernel // 2

        self.convt1 = nn.ConvTranspose1d(
            in_channels, in_channels, kernel_size=kernel, stride=1, padding=padding, output_padding=0, bias=not norm
        )

        self.bn1 = nn.BatchNorm1d(in_channels) if norm else nn.Identity()
        self.activation = nn.Identity() if activation is None else activation()

        # out_channels may be wrong argument here
        self.attention = AttentionBlock(out_channels, attn_depth, attn_heads, norm) if attn_on else nn.Identity()

        self.convt2 = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, output_padding=stride - 1, bias=not norm
        )

        self.bn2 = nn.BatchNorm1d(out_channels) if norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        self.upsample = upsample
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.convt1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.attention(out)

        out = self.convt2(out)
        out = self.bn2(out)

        if self.upsample:
            residual = self.upsample(x)
        if self.residual:
            out += residual*self.residual_scale
        out = self.activation(out)
        return out


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
        norm=True,
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
                bias=not norm
            ),
            nn.BatchNorm2d(out_channels) if norm else nn.Identity(),
            self.activation,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel,
                stride=1,
                padding=padding,
                bias=not norm
            ),
            nn.BatchNorm2d(out_channels) if norm else nn.Identity(),
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


### ResNets ###
class AttnResNet1d(nn.Module):
    def __init__(
        self,
        block: nn.Module = AttnResBlock1d,
        depth: int = 4,
        channels: list = [1, 64, 128, 256, 512],
        kernels: list = [3, 3, 3, 3, 3],
        strides: list = [1, 1, 1, 1, 1],
        dropout: float = 0.0,
        activation=nn.ReLU,
        norm=True,
        residual: bool = False,
        attn_on: list = [0, 0, 0, 0, 0, 0, 0],  # List of 0s and 1s indicating whether attention is applied in each layer
        attn_depth: int = 1,
        attn_heads: int = 1,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.inplanes = channels[0]
        self.attn_on = attn_on

        self.layers = nn.ModuleDict({})
        for i in range(0, self.depth):
            attn_enabled = False if self.attn_on is None else bool(self.attn_on[i])

            self.layers[str(i)] = self._make_layer(
                block,
                channels[i + 1],
                kernel=kernels[i],
                stride=strides[i],
                dropout=dropout,
                activation=activation,
                norm=norm,
                residual=residual,
                attn_on=attn_enabled,
                attn_depth=attn_depth,
                attn_heads=attn_heads,
            )

    def _make_layer(
        self, block, planes, kernel, stride, dropout, activation, norm, residual, attn_on, attn_depth, attn_heads
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride, bias=not norm),
                nn.BatchNorm1d(planes) if norm else nn.Identity(),
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
                norm,
                residual,
                attn_on=attn_on,
                attn_depth=attn_depth,
                attn_heads=attn_heads,
            )
        )
        self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        residuals = []
        for i in range(0, self.depth):
            x, res = self.layers[str(i)](x)
            residuals.append(res)

        return x, residuals


class AttnResNet1dT(nn.Module):
    def __init__(
        self,
        block: nn.Module = AttnResBlock1dT,
        depth: int = 4,
        channels: list = [512, 256, 128, 64, 1],
        kernels: list = [3, 3, 3, 3, 3],
        strides: list = [1, 1, 1, 1, 1],
        dropout: float = 0.0,
        activation=nn.ReLU,
        norm=True,
        sym_residual: bool = True,
        fwd_residual: bool = True,
        attn_on: list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        attn_depth: int = 1,
        attn_heads: int = 1,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.inplanes = channels[0]
        self.sym_residual = sym_residual  # for symmetric skip connections
        self.fwd_residual = fwd_residual  # for forward (normal) skip connections
        self.attn_on = attn_on
        self.residual_scales = nn.ParameterList([nn.Parameter(torch.tensor([1.0]), requires_grad=True) for _ in range(depth)])

        self.layers = nn.ModuleDict({})
        # self.fusion_layers = nn.ModuleDict({})

        for i in range(0, self.depth):
            attn_enabled = False if self.attn_on is None else bool(self.attn_on[i])
            self.layers[str(i)] = self._make_layer(
                block,
                channels[i + 1], # // 2, # CCCCC
                kernel=kernels[i],
                stride=strides[i],
                dropout=dropout,
                activation=activation,
                norm=norm,
                residual=fwd_residual,
                attn_on=attn_enabled,
                attn_depth=attn_depth,
                attn_heads=attn_heads,
            )

            # self.fusion_layers[str(i)] = Conv2DFusion(channels[i])

    def _make_layer(
        self, block, planes, kernel, stride, dropout, activation, norm, residual, attn_on, attn_depth, attn_heads
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
                    bias=not norm
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
                norm,
                residual,
                attn_on=attn_on,
                attn_depth=attn_depth,
                attn_heads=attn_heads,
            )
        )
        self.inplanes = planes # * 2 # CCCCC

        return nn.Sequential(*layers)

    def forward(self, x, residuals):
        for i in range(0, self.depth):
            if self.sym_residual:  # symmetric skip connection
                res = residuals[-1 - i]
                if res.ndim > x.ndim:  # for 3D to 2D
                    res = torch.mean(res, dim=2)

                # Element-wise addition of residual
                x = x + res * self.residual_scales[i]

                # Concatenation and fusion of residual
                # x = torch.concat((x, res), dim=1)
                # x = self.fusion_layers[str(i)](x)

            x = self.layers[str(i)](x)
        return x


class UNet1d(nn.Module):
    def __init__(
        self,
        depth: int = 6,
        channels_in: int = 1,
        channels: list = [1, 64, 128, 256, 256, 256, 256],
        kernels: list = [5, 3, 3, 3, 3, 3, 3],
        downsample: int = 4,
        attn: list = [0, 0, 0, 0, 0, 0, 0, 0, 0],
        attn_heads: int = 1,
        attn_depth: int = 1,
        dropout: float = 0.0,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        norm=True,
        fwd_skip: bool = True,
        sym_skip: bool = True,
    ) -> None:
        super().__init__()

        encoder_channels = [channels_in] + [channels] * depth if isinstance(channels, int) else channels
        encoder_channels = encoder_channels[0: depth + 1]
        decoder_channels = list(reversed(encoder_channels))
        decoder_channels[-1] = 1

        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels

        # Automatically calculate the strides for each layer
        strides = [
            2 if i < int(np.log2(downsample)) else 1 for i in range(depth)
        ]

        self.encoder = AttnResNet1d(
            block=AttnResBlock1d,
            depth=depth,
            channels=self.encoder_channels,
            kernels=kernels,
            strides=strides,
            attn_on=attn,
            attn_depth=attn_depth,
            attn_heads=attn_heads,
            dropout=dropout,
            activation=activation,
            norm=norm,
            residual=fwd_skip,
        )

        self.decoder = AttnResNet1dT(
            block=AttnResBlock1dT,
            depth=depth,
            channels=self.decoder_channels,
            kernels=list(reversed(kernels)),
            strides=list(reversed(strides)),
            # attn_on=list(reversed(attn[0:depth])),
            attn_on=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            attn_depth=attn_depth,
            attn_heads=attn_heads,
            dropout=dropout,
            activation=activation,
            norm=norm,
            sym_residual=sym_skip,
            fwd_residual=fwd_skip,
        )

    def forward(self, X: torch.Tensor):
        X = X.unsqueeze(1) if X.ndim < 4 else X
        Z, res = self.encoder(X)
        D = self.decoder(Z, res).squeeze(1)
        del X, Z, res
        return D


### RNN ###
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


### NODE ###
class NODE(nn.Module):
    """
    Neural ordinary differential equation model with an MLP vector field.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        output_size: int = None,
        depth: int = 3,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        output_activation: Optional[Type[nn.Module]] = nn.ReLU,
        norm: bool = False,
        residual: bool = True,
        residual_full: bool = False,
        lazy: bool = False,
        take_last: bool = True,
        **ode_kwargs,
    ) -> None:
        super().__init__()

        self.take_last = take_last

        output_size = input_size if output_size is None else output_size

        self.vector_field = MLPStack(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            depth=depth,
            activation=activation,
            output_activation=output_activation,
            norm=norm,
            residual=residual,
            residual_full=residual_full,
            lazy=lazy,
        )

        self.ode = NeuralODE(self.vector_field, **ode_kwargs)

    def forward(self, t_0: torch.Tensor, t_span: Optional[torch.Tensor] = None) -> torch.Tensor:
        if t_span is not None and isinstance(t_span, torch.Tensor):
            self.t_span = t_span
        _, z = self.ode(t_0)
        z = z.permute(1, 0, 2)
        if self.take_last:
            z = z[:, -1, :]
        return z

    @property
    def t_span(self) -> torch.Tensor:
        return self.ode.t_span

    @t_span.setter
    def t_span(self, time_array: torch.Tensor) -> None:
        self.ode.t_span = time_array


class UNetNODE(nn.Module):
    """
    Neural ordinary differential equation model with a 1D UNet vector field.
    """
    def __init__(
        self,
        depth: int = 5,
        channels_in: int = 1,
        channels: int = 32,
        kernels: list = [5, 3, 3, 3, 3, 3],
        downsample: int = 4,
        dropout: float = 0.0,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        norm: bool = False,
        residual: bool = True,
        take_last: bool = True,
        **ode_kwargs,
    ) -> None:
        super().__init__()

        self.take_last = take_last

        self.vector_field = UNet1d(
            depth=depth,
            channels_in=channels_in,
            channels=channels,
            kernels=kernels,
            downsample=downsample,
            dropout=dropout,
            activation=activation,
            norm=norm,
            fwd_skip=residual,
            sym_skip=residual,
        )

        self.ode = NeuralODE(self.vector_field, **ode_kwargs)

    def forward(self, t_0: torch.Tensor, t_span: Optional[torch.Tensor] = None) -> torch.Tensor:
        if t_span is not None and isinstance(t_span, torch.Tensor):
            self.t_span = t_span
        _, z = self.ode(t_0)
        z = z.permute(1, 0, 2)
        if self.take_last:
            z = z[:, -1, :]
        return z

    @property
    def t_span(self) -> torch.Tensor:
        return self.ode.t_span

    @t_span.setter
    def t_span(self, time_array: torch.Tensor) -> None:
        self.ode.t_span = time_array


### MISC ###
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

    input_tensor = torch.randn(10, 702)

    # # Test ResNet
    # model = AttnResNet1d(
    #     depth=3,
    #     channels=[1, 64, 128, 256, 512],
    #     kernels=[3, 3, 3, 3, 3],
    #     strides=[2, 2, 2, 2, 2],
    #     dropout=0.0,
    #     activation=nn.ReLU,
    #     norm=True,
    #     residual=False,
    #     attn_on=[0, 0, 0, 0, 0, 0, 0],
    #     attn_depth=1,
    #     attn_heads=1,
    # )
    #
    # output, residuals = model(input_tensor.unsqueeze(1))
    #
    # model = MLPStack(
    #     input_size=702,
    #     hidden_size=256,
    #     output_size=36,
    #     depth=3,
    #     activation=nn.ReLU,
    #     output_activation=nn.ReLU,
    #     norm=True,
    #     residual=True,
    # )
    #
    # output = model(input_tensor)
    model = UNetNODE(
        channels=64,
    )

    #
    # model = UNet1d(
    #     depth=3,
    #     channels_in=1,
    #     channels=64,
    #     kernels=[5, 3, 3, 3, 3],
    #     downsample=4,
    #     attn=[0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     attn_heads=1,
    #     attn_depth=1,
    #     dropout=0.0,
    #     norm=True,
    #     fwd_skip=True,
    #     sym_skip=True,
    # )

    input_tensor = torch.randn(10, 2000, requires_grad=True)

    linear = nn.Linear(2000, 128)

    input_tensor = linear(input_tensor)

    z = model(input_tensor)