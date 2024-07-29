
import torch
import pytest
from g2_vae import GRUAutoEncoder, GRUVAE
from g2_vae.pipeline.data import SpectraDataModule


def test_autoencoder():
    model = GRUAutoEncoder(2, 16)
    # first iteration we test using the correct
    # sequence data shape for a recurrent model
    X = torch.rand(64, 100, 2)
    with torch.no_grad():
        o = model(X)
    assert o.shape == (64, 100, 1)
    # now do it again to test the decorator
    X = torch.rand(64, 128, 2)
    with torch.no_grad():
        o = model(X)
    assert o.shape == (64, 128, 1)


def test_bidirectional_autoencoder():
    model = GRUAutoEncoder(2, 8, bidirectional=True)
    X = torch.rand(64, 100, 2)
    with torch.no_grad():
        o = model(X)
    assert o.shape == (64, 100, 1)


@pytest.mark.skip(reason="Takes a long time and needs real data.")
def test_autoencoder_real_data():
    BATCH_SIZE, WINDOW_SIZE = 32, 20
    data_module = SpectraDataModule(batch_size=BATCH_SIZE, window_size=WINDOW_SIZE)
    data_module.setup()

    # load a dataset
    (x, y, x_len, y_len) = next(iter(data_module.train_dataloader()))
    model = GRUAutoEncoder(WINDOW_SIZE, 48, bidirectional=False, decoder_num_layers=3)
    with torch.no_grad():
        pred_y = model(x)
    assert pred_y.shape == y.shape
    
    # now try bidirectional model
    model = GRUAutoEncoder(WINDOW_SIZE, 48, bidirectional=True, decoder_num_layers=3)
    with torch.no_grad():
        pred_y = model(x)
    assert pred_y.shape == y.shape


def test_gruvae_forward():
    model = GRUVAE(2, 16, beta=4.)
    # first iteration we test using the correct
    # sequence data shape for a recurrent model
    X = torch.rand(64, 100, 2)
    with torch.no_grad():
        o, _ = model(X)
    assert o.shape == (64, 100, 1)
    # now try for bidirectional
    model = GRUVAE(2, 16, beta=4., bidirectional=True, decoder_num_layers=3)
    X = torch.rand(64, 100, 2)
    with torch.no_grad():
        o, _ = model(X)
    assert o.shape == (64, 100, 1)


def test_gruvae_loss():
    model = GRUVAE(2, 16, beta=4.)
    data = torch.rand(2, 64, 100, 2)
    batch = (data[0], data[1])
    # test the loss function
    loss = model.step(batch, 0)