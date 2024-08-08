from pathlib import Path
from typing import Callable, List, Union, Optional, Dict
from functools import lru_cache

import os
import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, random_split, DataLoader, Subset

from src.pipeline.transforms import (
    input_transform_pipeline,
    target_transform_pipeline,
)
from src.utils import paths, get_system_and_backend
get_system_and_backend()

def log_scale(x: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """
    Applies log scaling to a 1D tensor.

    Args:
        x (torch.Tensor): A 1D tensor with shape (sequence_length,).

    Returns:
        torch.Tensor: A log-scaled tensor with the same shape as the input.
    """
    # Ensure numerical stability
    x_shifted = x - x.min()
    x_shifted += eps

    # Log transformation
    x_logged = torch.log(x_shifted)

    # Subtract minimum and normalize
    x_logged -= x_logged.min()
    x_logged /= x_logged.max()

    return x_logged

def log_scale_batch(batch: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """
    Applies log scaling to a batch of 1D tensors.

    Args:
        batch (torch.Tensor): A batch of 1D tensors with shape (batch_size, sequence_length).

    Returns:
        torch.Tensor: A batch of log-scaled tensors with the same shape as the input.
    """
    # Ensure numerical stability
    batch_shifted = batch - batch.min(dim=1, keepdim=True)[0]
    batch_shifted += eps

    # Log transformation
    batch_logged = torch.log(batch_shifted)

    # Subtract minimum and normalize
    batch_logged -= batch_logged.min(dim=1, keepdim=True)[0]
    batch_logged /= batch_logged.max(dim=1, keepdim=True)[0]

    return batch_logged


class MagnetometryDataset(Dataset):
    """
    A PyTorch Dataset class to handle loading and transforming state vector data from an HDF5 file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the HDF5 file containing the dataset.
    seed : int, optional
        Seed for random number generation, by default 10236.
    **kwargs
        Additional keyword arguments to set as attributes.

    Attributes
    ----------
    kwargs : dict
        Additional keyword arguments provided during initialization.
    _filepath : Union[str, Path]
        Path to the HDF5 file containing the dataset.
    rng : numpy.random.Generator
        Random number generator initialized with the provided seed.
    norm : bool
        Normalization flag, set to True by default.
    input_transforms : callable
        Function pipeline for transforming input data.
    target_transforms : callable
        Function pipeline for transforming target data.

    Methods
    -------
    filepath() -> str
        Returns the file path of the dataset.
    __len__() -> int
        Returns the total number of psi_posts in the dataset.
    indices() -> np.ndarray
        Returns an array of indices for the dataset.
    data() -> h5py.File
        Returns the HDF5 file object for the dataset.
    psi_posts() -> np.ndarray
        Returns the psi_posts stored on disk.
    beta_weights() -> np.ndarray
        Returns the beta_weights stored on disk.
    __getitem__(index: int) -> Dict[str, torch.Tensor]
        Returns the transformed input and target data for a given index.
    """

    def __init__(self, filepath: Union[str, Path], seed: int = 10236, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self._filepath = filepath
        self.rng = np.random.default_rng(seed)
        self.norm = False
        self.B_max = 2
        self.log_scale = False
        self.concat_log = False

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.input_transforms = input_transform_pipeline(self.norm)
        self.target_transforms = target_transform_pipeline(norm=False)

    @property
    def filepath(self) -> str:
        """
        Returns the file path of the dataset.

        Returns
        -------
        str
            File path of the dataset.
        """
        return self._filepath

    @lru_cache()
    def __len__(self) -> int:
        """
        Returns the total number of psi_posts in the dataset.

        Returns
        -------
        int
            Number of psi_posts in the dataset.
        """
        input_shape = self.data["inputs"].shape
        return input_shape[0]

    @property
    @lru_cache()
    def indices(self) -> np.ndarray:
        """
        Returns an array of indices for the dataset.

        Returns
        -------
        np.ndarray
            Array of indices.
        """
        return np.arange(len(self))

    @property
    def data(self):
        """
        Returns the HDF5 file object for the dataset.

        Returns
        -------
        h5py.File
            HDF5 file object containing the dataset.
        """
        return h5py.File(self.filepath, "r")

    @property
    @lru_cache()
    def inputs(self) -> np.ndarray:
        """
        Returns the inputs (time traces) stored on disk. These are the 1D inputs to the model.

        Returns
        -------
        np.ndarray
            NumPy 1D array of magnetometry time trace.
        """
        return self.data["inputs"]

    @property
    @lru_cache()
    def labels(self) -> np.ndarray:
        """
        Returns the labels (B_values) stored on disk. These are the target outputs of the model.

        Returns
        -------
        np.ndarray
            NumPy 1D array of B values.
        """
        return self.data["labels"]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Returns the transformed input and target data for a given index.

        Parameters
        ----------
        index : int
            Index of the data point to retrieve.

        Returns
        -------
        Dict[str, torch.Tensor]
            Transformed input (psi_post) and target (beta_weight) tensors.
        """
        x = self.inputs[index]
        y = self.labels[index]

        x = self.input_transforms(x)
        y = self.target_transforms(y)

        if self.log_scale:
            x = log_scale(x)

        if self.concat_log:
            x = torch.cat((x, log_scale(x)))

        y = y / self.B_max

        return x, y


class DataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule class to handle data loading and preparation for training and validation.

    Parameters
    ----------
    h5_path : Union[None, str], optional
        Path to the HDF5 file containing the dataset, by default None.
    batch_size : int, optional
        Batch size for data loaders, by default 64.
    seed : int, optional
        Seed for random number generation, by default 120516.
    num_workers : int, optional
        Number of worker processes for data loading, by default 0.
    pin_memory : bool, optional
        Whether to pin memory in data loaders, by default False.
    persistent_workers : bool, optional
        Whether to keep data loader workers persistent, by default False.
    split_type : str, optional
        Type of dataset split ("fixed" or "random"), by default "fixed".
    val_size : float, optional
        Proportion of the dataset to use for validation, by default 0.1.
    **kwargs
        Additional keyword arguments to pass to the StateVectorDataset.

    Attributes
    ----------
    h5_path : Union[None, str]
        Path to the HDF5 file containing the dataset.
    batch_size : int
        Batch size for data loaders.
    seed : int
        Seed for random number generation.
    num_workers : int
        Number of worker processes for data loading.
    pin_memory : bool
        Whether to pin memory in data loaders.
    persistent_workers : bool
        Whether to keep data loader workers persistent.
    val_size : float
        Proportion of the dataset to use for validation.
    split_type : str
        Type of dataset split ("fixed" or "random").
    data_kwargs : dict
        Additional keyword arguments for the dataset.
    header : dict
        Header information including dataset path and batch size.

    Methods
    -------
    check_h5_path()
        Checks if the HDF5 file path exists.
    setup(stage: Union[str, None] = None)
        Sets up the dataset and splits it into training and validation sets.
    train_dataloader() -> DataLoader
        Returns the DataLoader for the training set.
    val_dataloader() -> DataLoader
        Returns the DataLoader for the validation set.
    """

    def __init__(
        self,
        h5_path: Union[None, str] = None,
        batch_size: int = 64,
        seed: int = 120516,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        split_type: str = "fixed",
        val_size: float = 0.1,
        env: str = 'local', # Add env parameter
        **kwargs
    ):
        super().__init__()

        # Define paths based on the environment
        if env == 'colab':
            # Assuming you've mounted Google Drive at /content/drive
            self.h5_path = h5_path
        else:
            base_path = Path(paths.get("datasets"))
            self.h5_path = base_path.joinpath(h5_path)

        # self.h5_path = paths.get("datasets").joinpath(h5_path)
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.val_size = val_size
        self.split_type = split_type
        self.data_kwargs = kwargs

        header = {
            "h5_path": h5_path,
            "batch_size": self.batch_size,
        }
        self.header = {**header, **self.data_kwargs}

        self.check_h5_path()

    def check_h5_path(self):
        """
        Checks if the HDF5 file path exists.

        Raises
        ------
        RuntimeError
            If the HDF5 file path does not exist.
        """
        if not os.path.exists(self.h5_path):
            raise RuntimeError(f"Unable to find h5 file path: {self.h5_path}.")

    def setup(self, stage: Union[str, None] = None):
        """
        Sets up the dataset and splits it into training and validation sets.

        Parameters
        ----------
        stage : Union[str, None], optional
            Stage of the setup process, by default None.
        """
        full_dataset = MagnetometryDataset(self.h5_path, self.seed, **self.data_kwargs)

        ntotal = int(len(full_dataset))
        ntrain = int(ntotal * (1 - self.val_size))
        nval = ntotal - ntrain

        if self.split_type == "fixed":
            self.train_set = Subset(full_dataset, range(0, ntrain))
            self.val_set = Subset(full_dataset, range(ntrain, ntotal))
        elif self.split_type == "random":
            self.train_set, self.val_set = random_split(
                full_dataset,
                [ntrain, nval],
            )

    def train_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the training set.

        Returns
        -------
        DataLoader
            DataLoader for the training set.
        """
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the validation set.

        Returns
        -------
        DataLoader
            DataLoader for the validation set.
        """
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from src.utils import get_system_and_backend
    from src.visualization.visualize import plot_adj_matrix
    from src.visualization.fig_utils import add_colorbar
    get_system_and_backend()

    dm = DataModule(
        "n5000_0_to_10_snr100.h5",
        batch_size=256,
        num_workers=4,
        pin_memory=True,
        split_type="random",
        norm=False,
        B_max=10.0,
        # log_scale=True,
        concat_log=True
    )

    dm.setup()

    X, Y = next(iter(dm.train_dataloader()))


    plt.figure()
    plt.plot(X[0, :])

    for i in range(0, 10):
        x  = X[i, :]
        x -= x.min()
        x /= x.max()
        # x = np.log(x)
        plt.plot(x, label = f'A {Y[i]}')

    plt.plot(x)
    #
    #
    # q = x.clone()
    # q -= q.min()
    # q += 1e-6
    # q = torch.log(q)
    # q -= q.min()
    # q /= q.max()
    #
    #
    # plt.plot(q)
    #
    # import torch
    #
    # Q = log_scale_batch(X)
    #
    # # plt.legend()
