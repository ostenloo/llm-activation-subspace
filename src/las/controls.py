"""Absolute floors and the rogue-dimension diagnostic (SPEC.md §7)."""

from __future__ import annotations

import dataclasses

import numpy as np

from .config import ROGUE_K


# --------------------------------------------------------------------------
# Absolute floors
# --------------------------------------------------------------------------

def spectrally_matched_basis(X_fit: np.ndarray, q: int, rng: np.random.Generator) -> np.ndarray:
    """q orthonormal directions drawn from N(0, Sigma_z), then orthonormalized (§7).

    Same anisotropy and same effective spectrum as the data, but not the top PCs.
    The naive q/D floor assumes isotropy, which is the wrong null for transformer
    activations, so it is deliberately not offered here.

    Drawn as Xc^T w / sqrt(n-1) with w ~ N(0, I_n): that has covariance exactly
    Sigma_z without ever forming the D x D matrix, and lands the draws in the same
    row space -- and therefore the same rank -- as the data.
    """
    Xc = np.asarray(X_fit, dtype=np.float64)
    Xc = Xc - Xc.mean(axis=0)
    n = Xc.shape[0]
    if q > n - 1:
        raise ValueError(
            f"cannot draw {q} spectrally-matched directions from a rank-{n - 1} fit set"
        )
    W = rng.standard_normal((n, q))
    R = (Xc.T @ W) / np.sqrt(n - 1)
    Qm, _ = np.linalg.qr(R)
    return Qm.T[:q]


@dataclasses.dataclass(frozen=True)
class RankBand:
    """The rank-band control, with the §7 rank rule applied."""

    start: int  # 0-indexed, inclusive
    stop: int   # 0-indexed, exclusive
    kind: str   # "rank_band" | "narrowed" | "inapplicable"
    q: int
    rank: int

    @property
    def applicable(self) -> bool:
        return self.kind != "inapplicable"

    @property
    def width(self) -> int:
        return self.stop - self.start

    @property
    def comparable(self) -> bool:
        """Whether the band is the same width as the leading set it is a foil for."""
        return self.applicable and self.width == self.q


def rank_band(q: int, rank: int) -> RankBand:
    """PCs q+1..2q where the rank allows it, else the §7 fallback.

    With N_fit points the centered sample covariance has rank at most N_fit - 1, so
    at high v the band q+1..2q can run past the end of the spectrum. Deterministically:

      rank >= 2q  -> the full band q+1..2q
      q < rank < 2q -> the surviving non-leading components q+1..rank, which are fewer
                       than q; reported as narrowed, since the width no longer matches
                       the leading set and the comparison is weakened
      rank <= q   -> no non-leading components exist at all; inapplicable

    Never silently omitted: an inapplicable band is reported as inapplicable.
    """
    if q <= 0:
        raise ValueError(f"q must be positive, got {q}")
    if rank >= 2 * q:
        return RankBand(start=q, stop=2 * q, kind="rank_band", q=q, rank=rank)
    if rank > q:
        return RankBand(start=q, stop=rank, kind="narrowed", q=q, rank=rank)
    return RankBand(start=q, stop=q, kind="inapplicable", q=q, rank=rank)


# --------------------------------------------------------------------------
# Rogue-dimension diagnostic (Timkey & van Schijndel 2021)
# --------------------------------------------------------------------------

def cosine_contributions(X: np.ndarray) -> np.ndarray:
    """CC_i: each coordinate's mean contribution to pairwise cosine similarity.

    Timkey & van Schijndel eqs. 3-4. Their estimate samples random token pairs;
    the closed form over all ordered pairs i != j is exact and cheaper:

        CC_i = [ (sum_a u_ai)^2 - sum_a u_ai^2 ] / (n(n-1))

    for row-normalized u. Sums to the anisotropy estimate A-hat(f_l).
    """
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("a zero-norm activation has no cosine contribution")
    U = X / norms
    n = U.shape[0]
    if n < 2:
        raise ValueError("cosine contributions need at least two activations")
    return (U.sum(axis=0) ** 2 - np.sum(U ** 2, axis=0)) / (n * (n - 1))


def participation_ratio(u: np.ndarray) -> float:
    """PR(u) = 1 / sum_i u_i^4 for unit-norm u (§7).

    1 => the direction is a single raw coordinate; D => fully delocalized. This is
    the quantity that bears on A, since a leading PC that is essentially one rogue
    coordinate makes A a measurement of that coordinate.
    """
    u = np.asarray(u, dtype=np.float64)
    nrm = np.linalg.norm(u)
    if nrm == 0:
        raise ValueError("participation ratio is undefined for a zero vector")
    u = u / nrm
    return float(1.0 / np.sum(u ** 4))


@dataclasses.dataclass(frozen=True)
class RogueReport:
    rogue_idx: np.ndarray           # top-k coordinates by |CC_i|, descending
    anisotropy: float               # A-hat(f_l) = sum_i CC_i
    cc_share: float                 # share of A-hat carried by the top k
    top_cc: np.ndarray
    participation_ratios: np.ndarray  # PR of the leading PCs
    var_share_top_k: float          # share of total variance in the top-k coordinates


def rogue_report(X: np.ndarray, components: np.ndarray, k: int = ROGUE_K) -> RogueReport:
    """The mandatory §7 diagnostic, run before the §8 gate.

    Adjudicates the two readings of I ~ 0 (§4): a substantive null, versus U_1 and
    U_2 both being carried by the same few high-variance coordinates.
    """
    cc = cosine_contributions(X)
    order = np.argsort(np.abs(cc))[::-1]
    idx = order[:k]
    anis = float(cc.sum())
    var = np.asarray(X, dtype=np.float64).var(axis=0)
    n_pcs = min(5, components.shape[0])
    return RogueReport(
        rogue_idx=idx,
        anisotropy=anis,
        cc_share=float(cc[idx].sum() / anis) if anis != 0 else float("nan"),
        top_cc=cc[idx],
        participation_ratios=np.array([participation_ratio(c) for c in components[:n_pcs]]),
        var_share_top_k=float(var[idx].sum() / var.sum()) if var.sum() > 0 else float("nan"),
    )


def ablate_coordinates(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Drop the named coordinates, for the rogue-ablated gate (§8)."""
    keep = np.ones(X.shape[1], dtype=bool)
    keep[np.asarray(idx, dtype=int)] = False
    return np.asarray(X, dtype=np.float64)[:, keep]
