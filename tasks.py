"""Canonical tasks for the paired reduced-order simulation protocol.

Only NARMA-10, parity-3 and the digital delay-line baseline belong to the
publication.  Retired exploratory tasks and application demos are intentionally
absent from this module.
"""

from numbers import Integral

import numpy as np

from physical_model import _require_real_array


def _require_integer(name, value, *, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not a boolean or fraction")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def narma10(length, seed=0):
    """Return a seeded NARMA-10 input/target sequence."""
    length = _require_integer("length", length, minimum=11)
    seed = _require_integer("seed", seed, minimum=0)
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, size=length)
    y = np.zeros(length)
    for t in range(10, length):
        y[t] = (
            0.3 * y[t - 1]
            + 0.05 * y[t - 1] * np.sum(y[t - 10 : t])
            + 1.5 * u[t - 10] * u[t - 1]
            + 0.1
        )
    return u, y


def parity_stream(length, seed=0, order=3):
    """Return a seeded binary drive and trailing-window parity target."""
    length = _require_integer("length", length, minimum=1)
    seed = _require_integer("seed", seed, minimum=0)
    order = _require_integer("order", order, minimum=1)
    if order > length:
        raise ValueError("order cannot exceed length")
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=length)
    target = np.zeros(length, dtype=int)
    for t in range(order - 1, length):
        target[t] = int(np.sum(bits[t - order + 1 : t + 1]) % 2)
    signal = 2.0 * bits - 1.0
    return signal, target


def delay_embed(signal, order):
    """Raw digital delay line ``[u(t), u(t-1), ...]``."""
    signal = _require_real_array("signal", signal, nonempty=False).ravel()
    order = _require_integer("order", order, minimum=1)
    n = len(signal)
    X = np.zeros((n, order))
    for delay in range(order):
        X[delay:, delay] = signal[: n - delay]
    return X
