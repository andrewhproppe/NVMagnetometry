import torch
import numpy as np

from src.models.base import models
from src.pipeline.transforms import Normalize
from src.utils import paths
from src.visualization.AP_figs_funcs import find_nearest


def get_model(model_name: str = "CLpcfs", cuda: bool = False, train: bool = False):
    """
    Loads a specified model, including both the model definition and
    the committed model weights. The latter is stored in `/models/` as
    `.ckpt` files.

    Parameters
    ----------
    model_name : str, optional
        Name of the model, by default "CLpcfs". Valid models are
        defined in `g2_pcfs.models.base.models` as a dictionary.
    cuda : bool, optional
        If True, moves weights to a CUDA-enabled GPU, by default False
    train : bool, optional
        If True, keeps the model in training mode, by default False
        which puts sets `model.eval()`

    Returns
    -------
    PyTorch Lightning model
        Gives the pre-trained PyTorch lightning model

    Raises
    ------
    KeyError
        If the model name is not defined in `models`, then a `KeyError`
        is raised.
    """
    if model_name not in models.keys():
        raise KeyError(
            f"{model_name} is not a valid model; must be one of {models.keys()}"
        )
    ckpt_path = paths.get("models").joinpath(f"{model_name}.ckpt")
    # this loads the model definition and the weights
    model = models.get(model_name).load_from_checkpoint(ckpt_path)
    if not train:
        model.eval()
    if cuda:
        return model.cuda()
    return model


def get_checkpoint(
    model_name: str = "CLpcfs",
    ckpt_name: str = "CLpcfs.ckpt",
    cuda: bool = False,
    train: bool = False,
):
    """
    Loads a specified model, including both the model definition and
    the committed model weights. The latter is stored in `/models/` as
    `.ckpt` files.

    Parameters
    ----------
    model_name : str, optional
        Name of the model, by default "CLpcfs". Valid models are
        defined in `g2_pcfs.models.base.models` as a dictionary.
    cuda : bool, optional
        If True, moves weights to a CUDA-enabled GPU, by default False
    train : bool, optional
        If True, keeps the model in training mode, by default False
        which puts sets `model.eval()`

    Returns
    -------
    PyTorch Lightning model
        Gives the pre-trained PyTorch lightning model

    Raises
    ------
    KeyError
        If the model name is not defined in `models`, then a `KeyError`
        is raised.
    """
    if model_name not in models.keys():
        raise KeyError(
            f"{model_name} is not a valid model; must be one of {models.keys()}"
        )

    ckpt_path = paths.get("models").joinpath(f"{ckpt_name}")
    # this loads the model definition and the weights
    model = models.get(model_name).load_from_checkpoint(ckpt_path)
    if not train:
        model.eval()
    if cuda:
        return model.cuda()
    return model


def get_ensemble_checkpoint(
    model_name: str = "CLpcfs",
    ckpt_name: str = "CLpcfs.ckpt",
    cuda: bool = False,
    train: bool = False,
):
    if model_name not in models.keys():
        raise KeyError(
            f"{model_name} is not a valid model; must be one of {models.keys()}"
        )

    ckpt_path = paths.get("models").joinpath(f"{ckpt_name}")
    # this loads the model definition and the weights
    model = models.get(model_name).load_from_checkpoint(ckpt_path)
    if not train:
        model.eval()
    if cuda:
        return model.cuda()
    return model


def prepare_data(X: np.ndarray, time_steps: bool = True):
    """Takes an input NumPy array and preps it for processing.
    This function assumes that the shape of X is [2, T] for T
    timesteps, and that the first row (i.e. X[0]) corresponds
    to the timesteps, and the second row are the intensity
    counts.

    This function will check the data is in the correct shape,
    and then normalizes the intensity followed by converting
    it into a torch Tensor.

    Parameters
    ----------
    X : np.ndarray
        2D array corresponding to the spectra, with shape
        [2, T] for T timesteps. First row should be timesteps
        and second row should be intensity counts.

    time_steps: bool
        Whether or not to include the time steps when running
        input through model. Older models like ConvLSTMAutoEncoder
        and CLpcfs need the tensor shape to be [batch, input_size, 2],
        later models only need [batch, input_size]

    Returns
    -------
    torch.Tensor
        3D torch Tensor of shape [1, T, 2], with the intensities
        normalized.
    """
    assert X.ndim == 2
    assert X.shape[0] == 2
    norm = Normalize()
    # normalize the intensity between [0,1]
    X[1, :] = norm(X[1, :])
    X = torch.from_numpy(X.T).float()
    if not time_steps:
        X = X[:, 1]
    return X.unsqueeze(0)


def prepare_dot_g2s(dot, input_dim=140, min_τ=1e4, max_τ=1e11):
    """Takes in a dot object used for PCFS experiments with N g2 functions (one per stage position) and T time bins
    The neural networks models have a fixed input size of input_dim=140 time bins and were trained on g2 functions with photons correlated between
    τ = 1e4 - 1e11 ps. This function first takes dot.tau and takes it log before defining a truncating sparser array τ that matches input_dim using linspace

    g2s functions are then taken from the dot, spurious negative values deleted, and each g2 is normalized to 1. These maximum values are stored in an array,
    so that they can be used to restore the raw g2 and the model prediction to the original scale. The final shape of the g2s_norm array is [N, T, 2], where τ
    is stored along with the g2 and fed into the network (the time array doesn't actually play a role in the nn but was including in case time derivatives would
     ever become useful in the forward step of different networks)."""
    t_max_ind = find_nearest(max_τ, dot.tau)
    t_min_ind = find_nearest(min_τ, dot.tau)
    log_dot_τ = np.log(dot.tau)[
        0:t_max_ind
    ]  # need to take log of dot time so we can use linspace to interpolate later to match nn input. Truncate whatever the dots max time bound is to 1e11; this matches the nn training data
    τ = np.linspace(
        log_dot_τ[0], log_dot_τ[-1], input_dim
    )  # sets the number of time bins to 140; matches nn input dimension

    g2s = dot.g2
    g2s[g2s < 0] = 0
    g2s = g2s[:, t_min_ind:t_max_ind]
    g2_max_vals = np.zeros(g2s.shape[0])
    g2s_norm = np.zeros((g2s.shape[0], input_dim, 2))
    for i, g2 in enumerate(g2s):
        g2 = np.interp(τ, log_dot_τ, g2)
        g2_max_vals[i] = max(g2)
        g2s_norm[i, :, 0] = τ
        g2s_norm[i, :, 1] = g2 / max(g2)

    X = torch.Tensor(g2s_norm)
    return X, τ, g2_max_vals, g2s_norm


def multi_model_g2_prediction(model_name, ckpts, X):
    pred_arr = torch.zeros(len(ckpts), X.shape[0], X.shape[1])
    for i, ckpt in enumerate(ckpts):
        model = get_checkpoint(model_name, ckpt)
        model.eval()
        with torch.no_grad():
            temp_pred = model(X)
            pred_arr[i, :, :] = temp_pred[0]

    g2s_pred = pred_arr.detach().numpy()
    return g2s_pred
