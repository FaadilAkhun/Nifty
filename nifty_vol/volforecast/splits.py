"""Chronological split with purge gaps.

The target at row t depends on returns t+1..t+h, so a training row within h rows of the
next segment would have a label that overlaps the next segment. We drop h rows between
segments (the purge/embargo) so no training label ever sees validation/test returns.
"""
from __future__ import annotations

import numpy as np


def chronological_split(n: int, h: int, train_frac: float, val_frac: float):
    n_tr, n_va = int(n * train_frac), int(n * val_frac)
    tr = np.arange(0, n_tr)
    va = np.arange(n_tr + h, n_tr + h + n_va)
    te = np.arange(n_tr + h + n_va + h, n)
    assert len(te) > 0 and va[-1] < te[0] - h + 1
    return tr, va, te
