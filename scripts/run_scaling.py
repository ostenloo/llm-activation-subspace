"""Does the general corpus become identifiable at v=0.50 if given more data?

The §8 gate left one cell alive for both corpora, and C2 is the binding
constraint. §8 pre-registers restricting to a passing level, so the live
question is whether C2's passing band widens with N, or whether it is stuck.

Computes no A value: this is identifiability only (SPEC.md §10).

    python scripts/run_scaling.py
"""
import argparse
import collections
import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from las.config import GATE_THRESHOLD, N_LAYERS  # noqa: E402
from las.gate import split_half_overlaps  # noqa: E402

HARMLESS = pathlib.Path("/home/austin/unsupervised-concept-geometry/data/stageb")
ACTS = pathlib.Path("results/acts")
NS = (278, 600, 1000, 1600)
VS = (0.50, 0.75)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ACTS.mkdir(parents=True, exist_ok=True)
    cache = ACTS / "c2_scaling.npy"

    if cache.exists():
        A = np.load(cache)
        print(f"reusing {A.shape}")
    else:
        from las import extract
        rows = []
        for part in ("train", "val", "test"):
            rows += json.loads((HARMLESS / f"harmless_{part}.json").read_text())
        pool = sorted({r["instruction"] for r in rows})
        rng = np.random.default_rng(args.seed)
        rng.shuffle(pool)
        prompts = pool[:max(NS)]
        print(f"extracting {len(prompts)} harmless prompts", flush=True)
        L = extract.load()
        A = extract.capture(L, prompts, batch_size=32)
        np.save(cache, A)
        (ACTS / "c2_scaling_prompts.json").write_text(json.dumps(prompts, indent=1))
        extract.release(L)
        print(f"saved {A.shape}", flush=True)

    rng = np.random.default_rng(args.seed)
    out = []
    print(f"\nC2 split-half overlap; threshold {GATE_THRESHOLD}", flush=True)
    for v in VS:
        print(f"\n  v={v:.2f}   layers passing (of {N_LAYERS}), and the band", flush=True)
        for N in NS:
            idx = rng.permutation(A.shape[1])[:N]
            per = {}
            for layer in range(N_LAYERS):
                med, obs = split_half_overlaps(A[layer][idx], (v,), rng,
                                               n_splits=args.splits)[v]
                per[layer + 1] = (med, int(np.median([o.q1 for o in obs])))
                out.append((v, N, layer + 1, med, per[layer + 1][1]))
            ok = [l for l, (m, _) in per.items() if m >= GATE_THRESHOLD]
            band = f"{min(ok)}-{max(ok)}" if ok else "none"
            q12 = per[12][1]
            print(f"    N={N:>5} half={N//2:>4}: {len(ok):>2} pass  band {band:>7}  "
                  f"| layer12 overlap {per[12][0]:.2f} q={q12}", flush=True)

    with open("results/scaling.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["v", "N", "layer", "overlap", "q_median"])
        w.writerows(out)
    print("\nwrote results/scaling.csv", flush=True)


if __name__ == "__main__":
    main()
