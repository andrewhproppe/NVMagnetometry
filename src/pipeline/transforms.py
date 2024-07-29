import numpy as np
import torch
from torchvision.transforms import Compose
from typing import Tuple

class Normalize(object):
    def __call__(self, y: torch.Tensor):
        return y / torch.max(y)

class ArrayToTensor(object):
    def __call__(self, x):
        return torch.tensor(x, dtype=torch.float32)

class ConditionalTransform(object):
    def __init__(self, transform, condition: bool):
        self.transform = transform
        self.condition = condition

    def __call__(self, x):
        if self.condition:
            return self.transform(x)
        return x

def input_transform_pipeline(norm: bool = True):
    pipeline = Compose([
        ArrayToTensor(),
        ConditionalTransform(Normalize(), norm),
    ])
    return pipeline

def target_transform_pipeline(norm: bool = True):
    """
    Retrieves the training (Y) data transform pipeline.
    This normalizes the data and transforms NumPy arrays into torch tensors.

    Parameters
    ----------
    norm : bool
        Whether to apply normalization.

    Returns
    -------
    Compose
        A composed pipeline for training target data transformation.
    """
    pipeline = Compose([
        ArrayToTensor(),
        ConditionalTransform(Normalize(), norm),
    ])
    return pipeline

if __name__ == '__main__':
    # Example usage
    input_data = np.random.rand(702)
    target_data = np.random.rand(36)

    input_transform = input_transform_pipeline(norm=False)
    target_transform = target_transform_pipeline(norm=False)

    input_tensor = input_transform(input_data)
    target_tensor = target_transform(target_data)

    print(input_tensor)
    print(target_tensor)
