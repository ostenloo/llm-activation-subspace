"""Split-half identifiability gate (SPEC.md §8)."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

from .config import GATE_THRESHOLD, ROGUE_K, VARIANCE_LEVELS
from .controls import ablate_coordinates, cosine_contributions
from .subspace import Preprocessor, fit_pca


@dataclasses.dataclass(frozen=True)
class Overlap:
    """One split-half overlap measurement."""

    overlap: float
    q1: int
    q2: int

    @property
    def q_star(self) -> int:
        return min(self.q1, self.q2)

    @property
    def q_gap(self) -> int:
        """|q1 - q2|. Reported always: a large gap is itself evidence of
        non-identification at this (v, l), independent of the overlap value."""
        return abs(self.q1 - self.q2)


def subspace_overlap(U1: np.ndarray, U2: np.ndarray) -> Overlap:
    """||U1^T U2||_F^2 / q*, with both bases truncated to q* = min(q1, q2) (§8).

    Under variance-matching the two halves reach v at different component counts, so
    the normalizing q must be named or the 0.70 threshold is undefined. Truncation is
    used rather than normalizing an unequal-width product: a larger U2 can contain a
    smaller U1 trivially, and that asymmetric quantity would reward instability.
    """
    q1, q2 = U1.shape[0], U2.shape[0]
    q = min(q1, q2)
    M = U1[:q] @ U2[:q].T
    return Overlap(overlap=float(np.sum(M ** 2) / q), q1=q1, q2=q2)


def _halves(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    perm = rng.permutation(n)
    return perm[: n // 2], perm[n // 2 : 2 * (n // 2)]


def split_half_overlaps(
    X: np.ndarray,
    vs: Sequence[float],
    rng: np.random.Generator,
    n_splits: int = 50,
    mode: str = "center",
) -> dict[float, tuple[float, list[Overlap]]]:
    """Median split-half overlap over n_splits disjoint random halvings, at every v.

    This is the ceiling for the between-corpus principal-angle diagnostic: between-
    corpus overlap is only interpretable relative to it.

    Each halving is decomposed once and every variance level is sliced off the same
    spectrum, so the levels also share their splits -- removing split noise between
    them, which a per-level refit would leave in.

    SPEC.md §8 does not fix n_splits or the summary across splits; both are recorded
    here as defaults and must be closed in writing before the gate is run for real
    (see the note in §8). The median is used rather than the mean because a single
    unlucky halving at small N can be arbitrarily bad.
    """
    X = np.asarray(X, dtype=np.float64)
    out: dict[float, list[Overlap]] = {v: [] for v in vs}
    for _ in range(n_splits):
        ia, ib = _halves(X.shape[0], rng)
        Xa, Xb = X[ia], X[ib]
        pa, pb = Preprocessor.fit(Xa, mode), Preprocessor.fit(Xb, mode)
        fa, fb = fit_pca(pa.apply(Xa)), fit_pca(pb.apply(Xb))
        for v in vs:
            out[v].append(subspace_overlap(fa.subspace(v).components,
                                           fb.subspace(v).components))
    return {v: (float(np.median([o.overlap for o in obs])), obs) for v, obs in out.items()}


def split_half_overlap(
    X: np.ndarray,
    v: float,
    rng: np.random.Generator,
    n_splits: int = 50,
    mode: str = "center",
) -> tuple[float, list[Overlap]]:
    """Single-level convenience wrapper over split_half_overlaps."""
    return split_half_overlaps(X, (v,), rng, n_splits=n_splits, mode=mode)[v]


@dataclasses.dataclass(frozen=True)
class GateResult:
    """Verdict for one (v, l) cell."""

    v: float
    layer: int | None
    raw: float
    ablated: float
    verdict: str  # "pass" | "rogue_carried" | "fail"
    q1_median: float
    q2_median: float
    q_gap_median: float
    rogue_idx: np.ndarray

    @property
    def proceeds(self) -> bool:
        return self.verdict == "pass"


def gate_cells(
    X: np.ndarray,
    vs: Sequence[float] = VARIANCE_LEVELS,
    rng: np.random.Generator | None = None,
    layer: int | None = None,
    k: int = ROGUE_K,
    n_splits: int = 50,
    mode: str = "center",
    threshold: float = GATE_THRESHOLD,
) -> dict[float, GateResult]:
    """Gate every (v, l) cell for one layer, on raw and rogue-ablated overlap (§8).

    A cell passing raw but failing ablated is rogue-carried: its apparent
    identifiability is supplied by a few high-variance coordinates that both halves
    recover trivially, while the rest of the subspace is noise. Such cells are
    excluded from the primary rather than counted as identified.

    The rogue set is a property of the corpus and layer, not of v, so it is computed
    once here and the ablated corpus is decomposed once per halving.
    """
    if rng is None:
        rng = np.random.default_rng()
    X = np.asarray(X, dtype=np.float64)
    raw = split_half_overlaps(X, vs, rng, n_splits=n_splits, mode=mode)

    cc = cosine_contributions(X)
    rogue_idx = np.argsort(np.abs(cc))[::-1][:k]
    ablated = split_half_overlaps(
        ablate_coordinates(X, rogue_idx), vs, rng, n_splits=n_splits, mode=mode
    )

    out = {}
    for v in vs:
        raw_med, obs = raw[v]
        abl_med, _ = ablated[v]
        if raw_med >= threshold and abl_med >= threshold:
            verdict = "pass"
        elif raw_med >= threshold:
            verdict = "rogue_carried"
        else:
            verdict = "fail"
        out[v] = GateResult(
            v=v, layer=layer, raw=raw_med, ablated=abl_med, verdict=verdict,
            q1_median=float(np.median([o.q1 for o in obs])),
            q2_median=float(np.median([o.q2 for o in obs])),
            q_gap_median=float(np.median([o.q_gap for o in obs])),
            rogue_idx=rogue_idx,
        )
    return out


def gate_cell(
    X: np.ndarray,
    v: float,
    rng: np.random.Generator,
    layer: int | None = None,
    k: int = ROGUE_K,
    n_splits: int = 50,
    mode: str = "center",
    threshold: float = GATE_THRESHOLD,
) -> GateResult:
    """Single-level convenience wrapper over gate_cells."""
    return gate_cells(X, (v,), rng, layer=layer, k=k, n_splits=n_splits,
                      mode=mode, threshold=threshold)[v]
