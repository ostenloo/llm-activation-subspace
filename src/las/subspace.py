"""PCA subspaces, the capture statistic A, and the estimand I (SPEC.md §2, §3, §6)."""

from __future__ import annotations

import dataclasses

import numpy as np

from .config import LOGIT_EPS


@dataclasses.dataclass(frozen=True)
class Preprocessor:
    """Fit-set preprocessing (§6).

    Centering is fit on C_fit and applied to C_fit before PCA. It is deliberately
    *not* applied to displacements: Dz = f(x^v) - f(x), so a common offset cancels
    identically. Scaling does not cancel, so under the standardize sensitivity the
    fit-set sigma must be applied to displacements too, or A would be computed in a
    different space from the one U was fit in.
    """

    mean: np.ndarray
    scale: np.ndarray | None  # None under the primary "center only" setting

    @classmethod
    def fit(cls, X: np.ndarray, mode: str = "center") -> "Preprocessor":
        if mode not in ("center", "standardize"):
            raise ValueError(f"unknown preprocessing mode {mode!r}")
        mean = X.mean(axis=0)
        scale = None
        if mode == "standardize":
            sd = X.std(axis=0)
            # A dead coordinate carries no variance; leave it alone rather than
            # amplifying float noise into a spurious direction.
            scale = np.where(sd > 0, sd, 1.0)
        return cls(mean=mean, scale=scale)

    def apply(self, X: np.ndarray) -> np.ndarray:
        out = X - self.mean
        return out if self.scale is None else out / self.scale

    def apply_displacement(self, dZ: np.ndarray) -> np.ndarray:
        return dZ if self.scale is None else dZ / self.scale


@dataclasses.dataclass(frozen=True)
class Subspace:
    """Leading PCs of a fit set, with the spectrum that produced them."""

    components: np.ndarray  # (n_components, D), orthonormal rows
    explained_variance_ratio: np.ndarray  # full available spectrum, descending
    mean: np.ndarray
    q: int
    v: float

    @property
    def rank(self) -> int:
        """Components the fit set can support at all (§7 rank rule)."""
        return int(self.explained_variance_ratio.size)

    def basis(self, start: int, stop: int) -> np.ndarray:
        """PCs in the half-open 0-indexed band [start, stop)."""
        return self.components[start:stop]


def pca_fit(X: np.ndarray, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full available PCA of X (n, D). Returns (components, evr, mean).

    X is assumed already preprocessed. With n << D the sample covariance has rank
    at most n-1, so only that many components exist; components beyond the
    numerical rank are dropped rather than returned as noise.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected 2-D activations, got shape {X.shape}")
    mean = X.mean(axis=0)
    Xc = X - mean
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = s ** 2
    keep = var > tol * var[0] if var[0] > 0 else np.zeros_like(var, dtype=bool)
    var, Vt = var[keep], Vt[keep]
    return Vt, var / var.sum(), mean


def q_at_variance(evr: np.ndarray, v: float) -> int:
    """Smallest component count whose cumulative explained variance reaches v (§2)."""
    if not 0.0 < v <= 1.0:
        raise ValueError(f"variance level must be in (0, 1], got {v}")
    return int(np.searchsorted(np.cumsum(evr), v) + 1)


def fit_subspace(X: np.ndarray, v: float) -> Subspace:
    """U_C(v) -- the leading PCs reaching cumulative variance v (§2)."""
    comps, evr, mean = pca_fit(X)
    q = q_at_variance(evr, v)
    return Subspace(components=comps[:q], explained_variance_ratio=evr, mean=mean, q=q, v=v)


def capture(U: np.ndarray, dZ: np.ndarray) -> float:
    """A = sum_i ||U^T Dz_i||^2 / sum_i ||Dz_i||^2 (§2).

    A ratio of sums, not a mean of ratios: it is dominated by the largest ||Dz_i||,
    which is why §7 makes the per-prompt distribution a mandatory diagnostic.
    """
    dZ = np.asarray(dZ, dtype=np.float64)
    if dZ.ndim != 2:
        raise ValueError(f"expected 2-D displacements, got shape {dZ.shape}")
    total = float(np.sum(dZ ** 2))
    if total == 0.0:
        raise ValueError("displacements are identically zero; A is undefined")
    return float(np.sum((dZ @ U.T) ** 2) / total)


def capture_per_prompt(U: np.ndarray, dZ: np.ndarray) -> np.ndarray:
    """a_i = ||U^T Dz_i||^2 / ||Dz_i||^2, the mandatory distribution diagnostic (§7)."""
    dZ = np.asarray(dZ, dtype=np.float64)
    denom = np.sum(dZ ** 2, axis=1)
    if np.any(denom == 0.0):
        raise ValueError("a displacement is identically zero; a_i is undefined for it")
    return np.sum((dZ @ U.T) ** 2, axis=1) / denom


def logit(a: float) -> tuple[float, bool]:
    """Logit with the §3 boundary rule. Returns (value, was_clipped)."""
    clipped = not (LOGIT_EPS <= a <= 1.0 - LOGIT_EPS)
    x = min(max(a, LOGIT_EPS), 1.0 - LOGIT_EPS)
    return float(np.log(x / (1.0 - x))), clipped


@dataclasses.dataclass(frozen=True)
class Interaction:
    """I(v, l) with the untransformed contrast alongside, per the §3 boundary rule."""

    I: float
    I_untransformed: float
    cells: dict[str, float]
    clipped: tuple[str, ...]

    @property
    def any_clipped(self) -> bool:
        return bool(self.clipped)


def interaction(a_ref_1: float, a_ref_2: float, a_arb_1: float, a_arb_2: float) -> Interaction:
    """I = [logit A_ref,1 - logit A_ref,2] - [logit A_arb,1 - logit A_arb,2] (§3).

    The raw-scale contrast is always carried alongside, so a clipped cell can be
    reported untransformed without a post-hoc choice about which scale to show.
    """
    cells = {"ref_1": a_ref_1, "ref_2": a_ref_2, "arb_1": a_arb_1, "arb_2": a_arb_2}
    lg, clipped = {}, []
    for name, a in cells.items():
        lg[name], was = logit(a)
        if was:
            clipped.append(name)
    return Interaction(
        I=(lg["ref_1"] - lg["ref_2"]) - (lg["arb_1"] - lg["arb_2"]),
        I_untransformed=(a_ref_1 - a_ref_2) - (a_arb_1 - a_arb_2),
        cells=cells,
        clipped=tuple(clipped),
    )
