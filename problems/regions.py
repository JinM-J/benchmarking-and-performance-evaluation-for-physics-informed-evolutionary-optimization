# problems/regions.py
"""Factories for geometric regions with contains/sample interfaces.

Construct coordinate faces and other common geometries without repeating
sampling code in every problem. Grid-based sampling is deterministic;
random-face sampling uses an explicit fixed seed."""
import numpy as np

from problems.base import Region


def face_region(bounds: np.ndarray, axis: int, value: float,
                sample_axis: int, eps: float = 1e-9) -> Region:
    """Construct the face axis=value, sampled uniformly along sample_axis.

    bounds has shape (dim,2). For example,
    face_region(b,axis=1,value=0.0,sample_axis=0) samples n x-coordinates
    on the t=0 initial-condition face through region.sample(n)."""
    bounds = np.asarray(bounds, dtype=float)
    lo = bounds[sample_axis, 0]
    hi = bounds[sample_axis, 1]

    def contains_fn(Q: np.ndarray) -> np.ndarray:
        Q = np.asarray(Q, dtype=float)
        return np.abs(Q[:, axis] - value) <= eps

    def sample_fn(n_points: int) -> np.ndarray:
        dim = bounds.shape[0]
        Q = np.zeros((n_points, dim), dtype=float)
        Q[:, axis] = value
        Q[:, sample_axis] = np.linspace(lo, hi, n_points)
        return Q

    return Region(contains_fn=contains_fn, sample_fn=sample_fn)


def circle_boundary(cx: float, cy: float, r: float,
                    eps: float = 1e-6, endpoint: bool = True) -> Region:
    """Circular hole-wall region with uniform angular sampling.

    endpoint=True includes coincident angles 0 and 2*pi, matching the
    reference collocation convention. Pass endpoint=False to omit the duplicate."""

    def contains_fn(Q: np.ndarray) -> np.ndarray:
        Q = np.asarray(Q, dtype=float)
        return np.abs(np.hypot(Q[:, 0] - cx, Q[:, 1] - cy) - r) <= eps

    def sample_fn(n_points: int) -> np.ndarray:
        theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=endpoint)
        return np.c_[cx + r * np.cos(theta), cy + r * np.sin(theta)]

    return Region(contains_fn=contains_fn, sample_fn=sample_fn)


def union_region(regions, weights=None, min_per: int = 50) -> Region:
    """Weighted union of regions, for example multiple hole walls in one BC.

    sample(n) allocates counts by weight with at least min_per per region,
    proportionally reducing counts if needed before applying the minimum.
    Sample each region independently and concatenate in order. Integer
    truncation can make the final count slightly smaller than n."""
    regs = list(regions)
    w = np.asarray(weights, dtype=float) if weights is not None else np.ones(len(regs))
    w = w / (w.sum() + 1e-12)

    def contains_fn(Q: np.ndarray) -> np.ndarray:
        Q = np.asarray(Q, dtype=float)
        m = np.zeros(Q.shape[0], dtype=bool)
        for r in regs:
            m |= r.contains(Q)
        return m

    def sample_fn(n_points: int) -> np.ndarray:
        counts = np.maximum(min_per, (w * int(n_points)).astype(int))
        if counts.sum() > n_points:
            scale = int(n_points) / float(counts.sum())
            counts = np.maximum(min_per, (counts * scale).astype(int))
        return np.concatenate(
            [r.sample(int(k)) for r, k in zip(regs, counts)], axis=0
        )

    return Region(contains_fn=contains_fn, sample_fn=sample_fn)


def random_face_region(bounds: np.ndarray, axis: int, value: float,
                       seed: int) -> Region:
    """Construct axis=value with free coordinates sampled by default_rng(seed).

    Parameterized PDE initial/boundary faces use random collocation.
    Equal seeds on paired faces give identical free-coordinate sequences,
    preserving pointwise boundary pairing."""
    bounds = np.asarray(bounds, dtype=float)
    free = [j for j in range(bounds.shape[0]) if j != axis]

    def contains_fn(Q: np.ndarray) -> np.ndarray:
        Q = np.asarray(Q, dtype=float)
        return np.abs(Q[:, axis] - value) <= 1e-9

    def sample_fn(n_points: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        Q = np.zeros((int(n_points), bounds.shape[0]), dtype=float)
        Q[:, axis] = value
        for j in free:   # Sample each coordinate axis in order to preserve the RNG stream.
            Q[:, j] = rng.uniform(bounds[j, 0], bounds[j, 1], size=int(n_points))
        return Q

    return Region(contains_fn=contains_fn, sample_fn=sample_fn)
