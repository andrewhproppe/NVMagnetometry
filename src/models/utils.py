from functools import wraps

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class BetaRateScheduler:
    def __init__(
        self,
        initial_beta: float = 0.0,
        end_beta: float = 4.0,
        cap_steps: int = 4000,
        hold_steps: int = 2000,
    ):
        self._initial_beta = initial_beta
        self._end_beta = end_beta
        self._cap_steps = cap_steps
        self._hold_steps = hold_steps
        self.reset()

    @property
    def current_step(self):
        return self._current_step

    @current_step.setter
    def current_step(self, value: int):
        assert value >= 0
        self._current_step = value

    def reset(self):
        self.current_step = 0

    def __iter__(self):
        return self.beta()

    def beta(self):
        """
        Returns a generator that yields the next value of beta
        according to the scheduler. In the current implementation,
        the scheduler corresponds to a linear ramp up to `cap_steps`
        and subsequently holds the value of `end_beta` for another
        `hold_steps`. Once this is done, the value of `beta` is
        set back to zero, and the cycle begins anew.

        Yields
        -------
        float
            Value of beta at the current global step
        """
        beta_values = np.concatenate(
            [
                np.linspace(self._initial_beta, self._end_beta, self._cap_steps),
                np.array([self._end_beta for _ in range(self._hold_steps)]),
            ]
        )
        while self.current_step < self._cap_steps + self._hold_steps:
            self.current_step = self.current_step + 1
            yield beta_values[self.current_step - 1]
        self.reset()


def format_time_sequence(method):
    """
    Define a decorator that modifies the behavior of the
    forward call in a PyTorch model. This basically checks
    to see if the dimensions of the input data are [batch, time, features].
    In the case of 2D data, we'll automatically run the method
    with a view of the tensor assuming each element is an element
    in the sequence.
    """

    @wraps(method)
    def wrapper(model, X: torch.Tensor):
        if X.ndim == 2:
            batch_size, seq_length = X.shape
            output = method(model, X.view(batch_size, seq_length, -1))
        else:
            output = method(model, X)
        return output

    return wrapper


def init_rnn(module):
    for name, parameter in module.named_parameters():
        # use orthogonal initialization for RNNs
        if "weight" in name:
            try:
                nn.init.orthogonal_(parameter)
            # doesn't work for batch norm layers but that's fine
            except ValueError:
                pass
        # set biases to zero
        if "bias" in name:
            nn.init.zeros_(parameter)


def init_fc_layers(module):
    for name, parameter in module.named_parameters():
        if "weight" in name:
            try:
                nn.init.kaiming_uniform_(parameter)
            except ValueError:
                pass

        if "bias" in name:
            nn.init.zeros_(parameter)


def get_conv_output_size(model, input_tensor: torch.Tensor):
    output = model(input_tensor)
    return output.size(-1)


def get_conv_output_shape(model, input_tensor: torch.Tensor):
    output = model(input_tensor)
    return output.shape


def get_conv_flat_shape(model, input_tensor: torch.Tensor):
    output = torch.flatten(model(input_tensor[0:1, :, :, :]))
    return output.shape


def get_conv1d_flat_shape(model, input_tensor: torch.Tensor):
    # output = torch.flatten(model(input_tensor[-1, :, :]))
    output = torch.flatten(model(input_tensor))
    return output.shape


def symmetry_loss(profile_output: torch.Tensor):
    """
    Computes a penalty for asymmetric profiles. Basically take
    the denoised profile, and fold half of it on itself and
    calculate the mean squared error. By minimizing this value
    we try to constrain its symmetry.
    Expected profile_output shape is [N, T, 2]

    Parameters
    ----------
    profile_output : torch.Tensor
        The output of the model, expected shape is [N, T, 2]
        for N batch size and T timesteps.

    Returns
    -------
    float
        MSE symmetry loss
    """
    half = profile_output.shape[-1]
    y_a = profile_output[:, :half]
    y_b = profile_output[:, -half:].flip(-1)
    return F.mse_loss(y_a, y_b)


def fourier_loss(input_image, target_image, index=0):
    """
    - (1 - image) convets g2 into interferogram, which is Fourier transformed along the δ axis
    - Adapted to just look at interferogram for τ = 0, where oscillations are the strongest
    """
    # fmt: off
    y = (1 - target_image)[:, :, index,]
    y_hat = (1 - input_image)[:, :, index]

    y_fft = torch.fft.fftshift(torch.fft.fft(y), dim=1)
    y_hat_fft = torch.fft.fftshift(torch.fft.fft(y_hat), dim=1)

    loss = F.mse_loss(y_fft.real, y_hat_fft.real)
    return loss