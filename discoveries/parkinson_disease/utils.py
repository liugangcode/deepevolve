import numpy as np


def repl(x1, x2, cond):
    """Replace values in x1 with x2 where condition is True."""
    res = x1.copy()
    res[cond] = x2[cond]
    return res


