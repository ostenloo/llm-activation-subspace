"""Extract the two fit corpora and run the §8 feasibility gate.

Needs no arb family: the gate measures within-corpus split-half identifiability
of U_C, which depends only on C1_fit and C2_fit.

    python scripts/run_gate.py
"""
import argparse
import collections
import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from las.config import GATE_THRESHOLD, N_LAYERS, VARIANCE_LEVELS  # noqa: E402
from las.corpus import load_xstest  # noqa: E402
from las.gate import gate_cells  # noqa: E402

XSTEST = ("/home/austin/.cache/huggingface/hub/datasets--Paul--XSTest/blobs/"
          "ad38204d70b3964387f9690169d11afbc4756d01")
HARMLESS = pathlib.Path("/home/austin/unsupervised-concept-geometry/data/stageb")
ACTS = pathlib.Path("results/acts")
PRIMARY_V, PRIMARY_LAYER = 0.75, 12     # OPEN-5


def build_c2(tok, c1_prompts: list[str], seed: int = 0) -> list[str]:
    """C2 = refusal_direction harmless split, length-matched to C1_fit (OPEN-6)."""
    rows = []
    for part in ("train", "val", "test"):
        rows += json.loads((HARMLESS / f"harmless_{part}.json").read_text())
    pool = sorted({r["instruction"] for r in rows})
    n = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    by_len = collections.defaultdict(list)
    rng = np.random.default_rng(seed)
    for p in pool:
        by_len[n(p)].append(p)
    for v in by_len.values():
        rng.shuffle(v)
    out = []
    for target in sorted(n(p) for p in c1_prompts):
        # nearest available length, ties toward the shorter side
        cand = sorted((k for k, v in by_len.items() if v), key=lambda k: (abs(k - target), k))
        out.append(by_len[cand[0]].pop())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse-acts", action="store_true")
    args = ap.parse_args()
    ACTS.mkdir(parents=True, exist_ok=True)

    split = json.loads(pathlib.Path("results/c1_split.json").read_text())
    rows = {r["id"]: r for r in load_xstest(XSTEST)}
    c1 = [rows[i]["prompt"] for i in split["fit_ids"]]

    f1, f2 = ACTS / "c1_fit.npy", ACTS / "c2_fit.npy"
    if args.reuse_acts and f1.exists() and f2.exists():
        A1, A2 = np.load(f1), np.load(f2)
        c2 = json.loads((ACTS / "c2_fit_prompts.json").read_text())
        print(f"reusing cached activations {A1.shape} / {A2.shape}")
    else:
        from las import extract
        L = extract.load()
        c2 = build_c2(L.tok, c1, seed=args.seed)
        print(f"C1_fit {len(c1)} prompts | C2_fit {len(c2)} prompts")
        nt = lambda ps: [len(L.tok(p, add_special_tokens=False)["input_ids"]) for p in ps]
        print(f"  median tokens: C1 {int(np.median(nt(c1)))}  C2 {int(np.median(nt(c2)))}")
        A1, A2 = extract.capture(L, c1), extract.capture(L, c2)
        np.save(f1, A1); np.save(f2, A2)
        (ACTS / "c2_fit_prompts.json").write_text(json.dumps(c2, indent=1))
        print(f"saved {A1.shape} / {A2.shape}")

    rng = np.random.default_rng(args.seed)
    results, summary = {}, []
    for name, A in (("C1", A1), ("C2", A2)):
        for layer in range(N_LAYERS):
            cells = gate_cells(A[layer], VARIANCE_LEVELS, rng, layer=layer + 1,
                               n_splits=args.splits)
            for v, r in cells.items():
                results[(name, layer + 1, v)] = r
                summary.append((name, layer + 1, v, r.raw, r.ablated, r.verdict,
                                r.q1_median))

    print(f"\n{'':>4} " + "  ".join(f"{'v=' + format(v, '.2f'):>22}" for v in VARIANCE_LEVELS))
    for name in ("C1", "C2"):
        print(f"--- {name} " + "-" * 62)
        for layer in range(1, N_LAYERS + 1):
            cells = [results[(name, layer, v)] for v in VARIANCE_LEVELS]
            row = "  ".join(
                f"{c.raw:5.2f}/{c.ablated:5.2f} q{int(c.q1_median):<3} {c.verdict[:4]:<5}"
                for c in cells)
            print(f"{layer:>4} {row}")

    with open("results/gate.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["corpus", "layer", "v", "raw", "ablated", "verdict", "q_median"])
        w.writerows(summary)

    print(f"\nthreshold {GATE_THRESHOLD}; 'pass' needs raw AND ablated above it.")
    for name in ("C1", "C2"):
        v = collections.Counter(results[(name, l, vv)].verdict
                                for l in range(1, N_LAYERS + 1) for vv in VARIANCE_LEVELS)
        print(f"  {name}: " + ", ".join(f"{k} {n}" for k, n in sorted(v.items())))
    p1 = results[("C1", PRIMARY_LAYER, PRIMARY_V)]
    p2 = results[("C2", PRIMARY_LAYER, PRIMARY_V)]
    print(f"\nPRIMARY CELL (OPEN-5) v={PRIMARY_V}, layer={PRIMARY_LAYER}:")
    print(f"  C1 raw {p1.raw:.2f} ablated {p1.ablated:.2f} -> {p1.verdict}")
    print(f"  C2 raw {p2.raw:.2f} ablated {p2.ablated:.2f} -> {p2.verdict}")
    print("  wrote results/gate.csv")


if __name__ == "__main__":
    main()
