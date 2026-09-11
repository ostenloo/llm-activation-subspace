"""Gate the expanded corpora (SPEC.md §17, OPEN-3 C1-composition decision).

    C1 = 861 harmful + 861 harmless, balanced, N=1722
    C2 = 1722 harmless, disjoint from C1's harmless half

Computes no A value (SPEC.md §10). Reports the primary cell at v=0.50.
"""
import argparse
import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from las.config import GATE_THRESHOLD, N_LAYERS  # noqa: E402
from las.gate import gate_cells  # noqa: E402

SB = pathlib.Path("/home/austin/unsupervised-concept-geometry/data/stageb")
ACTS = pathlib.Path("results/acts")
VS = (0.50, 0.75)
PRIMARY_V, PRIMARY_LAYER = 0.50, 12


def load_split(name):
    rows = []
    for part in ("train", "val", "test"):
        rows += json.loads((SB / f"{name}_{part}.json").read_text())
    return sorted({r["instruction"] for r in rows})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ACTS.mkdir(parents=True, exist_ok=True)
    f1, f2 = ACTS / "c1_exp.npy", ACTS / "c2_exp.npy"

    if f1.exists() and f2.exists():
        A1, A2 = np.load(f1), np.load(f2)
        print(f"reusing {A1.shape} / {A2.shape}", flush=True)
    else:
        harmful, harmless = load_split("harmful"), load_split("harmless")
        rng = np.random.default_rng(args.seed)
        rng.shuffle(harmful); rng.shuffle(harmless)
        n_h = len(harmful)                       # 861; balance C1 around it
        c1 = harmful + harmless[:n_h]            # 1722, balanced
        c2 = harmless[n_h:n_h + 2 * n_h]         # 1722, disjoint from C1's half
        assert not (set(c1) & set(c2)), "C1 and C2 share a prompt"
        print(f"C1 {len(c1)} ({n_h} harmful / {n_h} harmless)  C2 {len(c2)} harmless",
              flush=True)
        from las import extract
        L = extract.load()
        A1 = extract.capture(L, c1, batch_size=32); np.save(f1, A1)
        A2 = extract.capture(L, c2, batch_size=32); np.save(f2, A2)
        (ACTS / "exp_prompts.json").write_text(json.dumps({"c1": c1, "c2": c2}))
        extract.release(L)
        print(f"saved {A1.shape} / {A2.shape}", flush=True)

    rng = np.random.default_rng(args.seed)
    res, out = {}, []
    for name, A in (("C1", A1), ("C2", A2)):
        for layer in range(N_LAYERS):
            for v, r in gate_cells(A[layer], VS, rng, layer=layer + 1,
                                   n_splits=args.splits).items():
                res[(name, layer + 1, v)] = r
                out.append((name, layer + 1, v, r.raw, r.ablated, r.verdict,
                            r.q1_median))
        print(f"  {name} done", flush=True)

    with open("results/gate_expanded.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["corpus", "layer", "v", "raw", "ablated", "verdict", "q_median"])
        w.writerows(out)

    for v in VS:
        both = [l for l in range(1, N_LAYERS + 1)
                if res[("C1", l, v)].verdict == "pass" and res[("C2", l, v)].verdict == "pass"]
        print(f"\nv={v:.2f}: {len(both)} layers pass for BOTH corpora")
        if both:
            runs = [[both[0]]]
            for l in both[1:]:
                if l == runs[-1][-1] + 1:
                    runs[-1].append(l)
                else:
                    runs.append([l])          # a NEW list; never mutate one already stored
            longest = max(runs, key=len)
            print(f"  layers: {both}")
            print(f"  longest contiguous run: {len(longest)} ({longest[0]}-{longest[-1]})"
                  f"  [OPEN-5 needs >= 5]")
    p1, p2 = res[("C1", PRIMARY_LAYER, PRIMARY_V)], res[("C2", PRIMARY_LAYER, PRIMARY_V)]
    print(f"\nPRIMARY CELL v={PRIMARY_V}, layer={PRIMARY_LAYER} (threshold {GATE_THRESHOLD}):")
    print(f"  C1 raw {p1.raw:.2f} ablated {p1.ablated:.2f} q~{int(p1.q1_median)} -> {p1.verdict}")
    print(f"  C2 raw {p2.raw:.2f} ablated {p2.ablated:.2f} q~{int(p2.q1_median)} -> {p2.verdict}")
    print(f"  VERDICT: {'PROCEED' if p1.verdict == p2.verdict == 'pass' else 'FAIL'}")


if __name__ == "__main__":
    main()
