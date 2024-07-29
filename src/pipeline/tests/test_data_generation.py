
from ..make_dataset import generate_gridded_spectra


def test_grid():
    NUM_POINTS = 5
    y, params, x = generate_gridded_spectra(NUM_POINTS)
    assert y.shape == (NUM_POINTS, NUM_POINTS, NUM_POINTS, NUM_POINTS, 1501)
