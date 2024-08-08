from typing import Optional, Type
from src.models.utils import init_fc_layers, get_conv_output_shape, get_conv_flat_shape
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


### CNN / ResNet ###
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

    model = MLPStack(
        input_size=702,
        hidden_size=256,
        output_size=36,
        depth=3,
        activation=nn.ReLU,
        output_activation=nn.ReLU,
        norm=True,
        residual=True,
    )

    output = model(input_tensor)