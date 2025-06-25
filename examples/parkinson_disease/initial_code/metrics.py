import numpy as np
from scipy.special import softmax


def smape1p_ind(A, F):
    """Individual SMAPE+1 calculation."""
    val = 200 * np.abs(F - A) / (np.abs(A + 1) + np.abs(F + 1))
    return val


def smape1p(A, F):
    """SMAPE+1 metric calculation."""
    return smape1p_ind(A, F).mean()


def smape1p_opt(x):
    """Optimal SMAPE+1 calculation."""
    tgts = np.arange(0, 61)
    scores = [smape1p(x, val) for val in tgts]
    return tgts[np.argmin(scores)]


def single_smape1p(preds, tgt):
    """Single SMAPE+1 calculation for probability distributions."""
    x = np.tile(np.arange(preds.shape[1]), (preds.shape[0], 1))
    x = np.abs(x - tgt) / (2 + x + tgt)
    return (x * preds).sum(axis=1)


def opt_smape1p(preds):
    """Optimal SMAPE+1 for probability distributions."""
    x = np.hstack(
        [single_smape1p(preds, i).reshape(-1, 1) for i in range(preds.shape[1])]
    )
    return x.argmin(axis=1)


def max_dif(val, lst):
    """Calculate maximum difference."""
    lst0 = [x for x in lst if x < val]
    if len(lst0) == 0:
        return -1
    return val - max(lst0)


def count_prev_visits(val, lst):
    """Count previous visits."""
    lst0 = [x for x in lst if x < val]
    return len(lst0)
