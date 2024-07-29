
import torch

from g2_vae.models.base import DownsampleDecoder


def test_downsample_decoder():
    BATCH_SIZE, HIDDEN_DIM = 24, 32
    X = torch.rand(BATCH_SIZE, 1501, 2)
    z = torch.rand(1, BATCH_SIZE, HIDDEN_DIM)
    params = torch.rand(BATCH_SIZE, 4)
    
    model = DownsampleDecoder(HIDDEN_DIM)
    
    with torch.no_grad():
        pred_Y, pred_params = model(X, z)
        assert pred_Y.shape == (BATCH_SIZE, 1501)
        # now try the loss functions
        test_layer = model.layers[1]
        loss, pred_Y, _ = model._downsampled_layer_loss(test_layer, model.output, X, z, X)
        loss, logs = model.loss((X, X, params), z)