
import pytest

from g2_vae.pipeline.data import SpectraDataModule, H5Dataset
from g2_vae.utils import paths

@pytest.mark.skip(reason="Older version for windows")
def test_load_windows():
    BATCH_SIZE, WINDOW_SIZE = 16, 20
    module = SpectraDataModule(batch_size=BATCH_SIZE, window_size=WINDOW_SIZE)
    module.setup()
    train_loader = module.train_dataloader()
    (x, y, x_len, y_len) = next(iter(train_loader))
    assert x.shape == (BATCH_SIZE, x_len[0], WINDOW_SIZE)


def test_load_timesteps():
    BATCH_SIZE = 16
    module = SpectraDataModule(batch_size=BATCH_SIZE)
    module.setup()
    train_loader = module.train_dataloader()
    (x, y) = next(iter(train_loader))
    assert x.shape == (BATCH_SIZE, 1501, 2)
    assert y.shape == (BATCH_SIZE, 1501, 2)


def test_grab_data():
    h5_path = paths.get("raw").joinpath("devset.h5")
    data = H5Dataset(h5_path)
    (x, y) = data.__getitem__(0)
    assert x.shape == (1501, 2)