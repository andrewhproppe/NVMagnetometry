import torch
from torchcde import hermite_cubic_coefficients_with_backward_differences, CubicSpline

def get_cubic_spline_coeffs(X, t, obs_inds):
    """
    Prepare data for use with Neural CDEs.
    g2s: tensor of observed g2s with shape (nobs, npoints)
    obs_inds: indices of observed g2s from time span with shape (nobs,)
    t: tensor of time points to infer with shape (ntimes,)
    """
    # Initialize an array of nans, in the dimensionality of a full PCFS experiment
    x = torch.zeros(len(t), X.shape[-1])
    x[x == 0] = float('nan')

    # Add observed datapoints
    x[obs_inds, :] = X

    # Get array that accumulates missing observations
    mask = (torch.isnan(x[:, 0])).cumsum(dim=-1)

    # Normalize between 0 and 1
    mask = mask / mask.max(dim=-1, keepdim=True)[0]

    # Concatenate time, data,and mask
    x_ = torch.cat([t[..., None], x, mask[..., None]], -1)

    # Get coeffs
    coeffs = hermite_cubic_coefficients_with_backward_differences(x_, t)

    # Get spline
    # X_ = CubicSpline(coeffs, t)

    return coeffs