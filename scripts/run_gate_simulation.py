"""Reproduce the SPEC.md §8 feasibility table.

The table in §8 asserts split-half subspace overlap at N=225/450 under "fast decay"
and "slow decay", but the spec never defines those spectra or the number of random
halvings behind each number, so the original figures cannot be reproduced exactly.
This script fixes both choices explicitly and reports what they give, so the gate
implementation is validated against a stated generating process rather than against
an unrecorded one.

    ssh fedora 'cd ~/llm-activation-subspace && \
        /home/austin/unsupervised-concept-geometry/.venv/bin/python \
        scripts/run_gate_simulation.py'
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from las.config import D_MODEL, GATE_THRESHOLD, VARIANCE_LEVELS  # noqa: E402
from las.gate import split_half_overlap  # noqa: E402

SPEC_TABLE = {           # SPEC.md §8, for comparison
    ("fast", 225): (0.90, 0.83, 0.78),
    ("slow", 225): (0.35, 0.38, 0.40),
    ("slow", 450): (0.46, 0.52, 0.50),
}


def spectrum(kind: str, D: int) -> np.ndarray:
    """Eigenvalue profiles over all D dimensions. Neither is specified in §8.

    fast: exponential, exp(-i/2)  -- a handful of components carry most variance,
          and adjacent eigenvalues are well separated, so PCs are stable.
    slow: power law,   (i+1)^-0.5 -- variance spread thin across the spectrum with
          adjacent eigenvalues nearly tied; the regime §8 worries about.

    The spectrum is a property of the model and is therefore fixed in D, NOT scaled
    with n. Tying its support to the sample size makes the problem harder in exact
    proportion as N grows, which silently cancels the benefit of more data.
    """
    i = np.arange(D)
    if kind == "fast":
        return np.exp(-i / 2.0)
    if kind == "slow":
        return (i + 1.0) ** -0.5
    raise ValueError(kind)


def sample(kind: str, n: int, D: int, rng: np.random.Generator) -> np.ndarray:
    """n draws from N(0, diag(lambda)).

    Generated in the canonical basis with no random rotation: the overlap metric
    ||U1^T U2||_F^2 is invariant under a common rotation of the data, so rotating
    would cost a 4096x4096 QR and change nothing.
    """
    return rng.standard_normal((n, D)) * np.sqrt(spectrum(kind, D))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--dim", type=int, default=D_MODEL)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"D={args.dim}  splits={args.splits}  seed={args.seed}  "
          f"gate threshold={GATE_THRESHOLD}")
    print(f"\n{'spectrum':>9} {'N':>5} " +
          " ".join(f"{'v=' + format(v, '.2f'):>14}" for v in VARIANCE_LEVELS))
    print(f"{'':>9} {'':>5} " + " ".join(f"{'(this / spec)':>14}" for _ in VARIANCE_LEVELS))

    for kind, N in (("fast", 225), ("slow", 225), ("slow", 450)):
        rng = np.random.default_rng(args.seed)
        X = sample(kind, N, args.dim, rng)
        cells = []
        for j, v in enumerate(VARIANCE_LEVELS):
            med, _ = split_half_overlap(X, v, rng, n_splits=args.splits)
            ref = SPEC_TABLE[(kind, N)][j]
            flag = "" if abs(med - ref) <= 0.10 else " *"
            cells.append(f"{med:.2f} / {ref:.2f}{flag:>2}")
        print(f"{kind:>9} {N:>5} " + " ".join(f"{c:>14}" for c in cells))

    print("\n* = differs from the §8 table by more than 0.10. The spec does not record")
    print("  the spectra or split count behind that table, so a gap is a difference in")
    print("  generating process, not necessarily a defect in either.")
    print("\nGate reading: under slow decay the subspace is not identified at these N --")
    print("two samples from the SAME corpus give near-unrelated subspaces, so A_11 vs")
    print("A_12 would measure sampling noise (SPEC.md §8).")


if __name__ == "__main__":
    main()
