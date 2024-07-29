
import pytest

import torch

from g2_vae.models.base import ConvSeqProcessor, ConvLSTMAutoEncoder, CLVAE, ConvLSTMEncoder

def test_conv_seq():
    BATCHSIZE, NUM_WINDOWS, OUTPUT_CHANNELS, NUM_LAYERS = 48, 1482, 32, 5
    model = ConvSeqProcessor(1, OUTPUT_CHANNELS, NUM_LAYERS)
    X = torch.rand(BATCHSIZE, NUM_WINDOWS, 2)
    with torch.no_grad():
        o = model(X)
    pred_N, pred_C = o.size(0), o.size(2)
    assert pred_N == BATCHSIZE
    assert pred_C == OUTPUT_CHANNELS


def test_convlstmencoder():
    model = ConvLSTMEncoder(1, 24, 5, 64)
    X = torch.rand(64, 1501, 2)
    z = model(X)


def test_convlstmautoencoder():
    BATCHSIZE, NUM_WINDOWS, OUTPUT_CHANNELS, NUM_LAYERS = 48, 1482, 32, 5
    X = torch.rand(BATCHSIZE, NUM_WINDOWS, 2)
    params = torch.rand(BATCHSIZE, 4)
    model = ConvLSTMAutoEncoder(
        2, 32, bidirectional=False, batch_size=BATCHSIZE,
        conv_output_channels=OUTPUT_CHANNELS,
        conv_num_layers=NUM_LAYERS, encoder_num_layers=1, decoder_num_layers=3
    )
    with torch.no_grad():
        o, pred_params = model(X)
        assert o.ndim == 2
        assert o.shape == (BATCHSIZE, NUM_WINDOWS)
        loss, logs = model.step((X, X, params), None)


def test_clvae():
    BATCHSIZE, NUM_WINDOWS, OUTPUT_CHANNELS, NUM_LAYERS = 48, 1482, 32, 5
    X = torch.rand(BATCHSIZE, NUM_WINDOWS, 2)
    params = torch.rand(BATCHSIZE, 4)
    model = CLVAE(2, 32, bidirectional=True, conv_output_channels=OUTPUT_CHANNELS,
        conv_num_layers=NUM_LAYERS, encoder_num_layers=1, decoder_num_layers=3
    )
    with torch.no_grad():
        o, vae_params, pred_params = model(X)
        assert o.ndim == 2
        assert o.shape == (BATCHSIZE, NUM_WINDOWS)
        # simulate a batch
        loss, logs = model.step((X, X, params), None)