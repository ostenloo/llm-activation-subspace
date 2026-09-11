"""The main analysis: the four cells, I(v, l), controls, and the bootstrap.

SPEC.md §10 steps 4-5. Runs only at v=0.50 over the gate-passing layers, per
the §8 gate and the OPEN-3 closure.
"""
import argparse
import collections
import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from las.config import LOGIT_EPS  # noqa: E402
from las.controls import rank_band, spectrally_matched_basis  # noqa: E402
from las.subspace import (Preprocessor, capture, capture_per_prompt,  # noqa: E402
                          fit_pca, fit_pca_truncated, interaction)

ACTS = pathlib.Path("results/acts")
V, PRIMARY_LAYER, LAYERS = 0.50, 12, range(1, 22)   # gate-passing run 1-21


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = list(csv.DictReader(open("results/arb_worklist.csv")))
    base = [r["base_prompt"] for r in rows]
    ref = [r["ref_variant"] for r in rows]
    arb = [r["arb_variant"] for r in rows]

    ev = ACTS / "eval_families.npz"
    if ev.exists():
        d = np.load(ev, allow_pickle=True)
        Zb, Zr, Za, keep = d["Zb"], d["Zr"], d["Za"], d["keep"]
        print(f"reusing eval families; {keep.sum()} of {len(rows)} pairs kept", flush=True)
    else:
        from las import extract
        L = extract.load()
        print("behavioural flip check (OPEN-1)...", flush=True)
        g_b, g_r, g_a = (extract.generate(L, x) for x in (base, ref, arb))
        rb = np.array([extract.refuses(t) for t in g_b])
        rr = np.array([extract.refuses(t) for t in g_r])
        ra = np.array([extract.refuses(t) for t in g_a])
        keep = (~rb) & rr & (~ra)
        print(f"  base refused:  {rb.sum():>2}/{len(rows)}  (want 0)")
        print(f"  ref  refused:  {rr.sum():>2}/{len(rows)}  (want all)")
        print(f"  arb  refused:  {ra.sum():>2}/{len(rows)}  (want 0)")
        print(f"  pairs kept:    {keep.sum():>2}/{len(rows)}", flush=True)
        for i, r in enumerate(rows):
            if not keep[i]:
                why = ("base refused" if rb[i] else
                       "ref NOT refused" if not rr[i] else "arb refused")
                print(f"    dropped [{r['focus']}]: {why}")
        Zb, Zr, Za = (extract.capture(L, x) for x in (base, ref, arb))
        extract.release(L)
        np.savez(ev, Zb=Zb, Zr=Zr, Za=Za, keep=keep)

    k = np.asarray(keep, dtype=bool)
    A1, A2 = np.load(ACTS / "c1_exp.npy"), np.load(ACTS / "c2_exp.npy")
    rng = np.random.default_rng(a.seed)
    out, summary = [], {}

    for layer in LAYERS:
        dref = (Zr[layer - 1] - Zb[layer - 1])[k]
        darb = (Za[layer - 1] - Zb[layer - 1])[k]
        cells = {}
        for name, X in (("1", A1[layer - 1]), ("2", A2[layer - 1])):
            pre = Preprocessor.fit(X, "center")
            U = fit_pca(pre.apply(X)).subspace(V)
            cells[f"ref_{name}"] = capture(U.components, pre.apply_displacement(dref))
            cells[f"arb_{name}"] = capture(U.components, pre.apply_displacement(darb))
            cells[f"q_{name}"] = U.q
            if layer == PRIMARY_LAYER:
                R = spectrally_matched_basis(pre.apply(X), U.q, rng)
                bnd = rank_band(U.q, U.rank)
                summary[f"null_{name}"] = capture(R, pre.apply_displacement(dref))
                summary[f"band_{name}"] = (
                    capture(U.components[bnd.start:bnd.stop], pre.apply_displacement(dref))
                    if bnd.applicable else float("nan"))
                summary[f"band_kind_{name}"] = bnd.kind
                summary[f"perprompt_{name}"] = capture_per_prompt(
                    U.components, pre.apply_displacement(dref))
        I = interaction(cells["ref_1"], cells["ref_2"], cells["arb_1"], cells["arb_2"])
        out.append((layer, cells, I))
        print(f"  layer {layer:>2}  A_ref1 {cells['ref_1']:.3f} A_ref2 {cells['ref_2']:.3f} "
              f"A_arb1 {cells['arb_1']:.3f} A_arb2 {cells['arb_2']:.3f}  "
              f"I {I.I:+.4f}  q {cells['q_1']}/{cells['q_2']}", flush=True)

    # --- clustered bootstrap (§9, OPEN-4) ---
    print(f"\nbootstrap B={a.boot}, clustered on base prompt...", flush=True)
    n_eval = int(k.sum())
    boots = {l: [] for l in LAYERS}
    for b in range(a.boot):
        fi1 = rng.integers(0, A1.shape[1], A1.shape[1])
        fi2 = rng.integers(0, A2.shape[1], A2.shape[1])
        ei = rng.integers(0, n_eval, n_eval)          # clusters = base prompts
        for layer in LAYERS:
            dref = (Zr[layer - 1] - Zb[layer - 1])[k][ei]
            darb = (Za[layer - 1] - Zb[layer - 1])[k][ei]
            c = {}
            for name, X, fi in (("1", A1[layer - 1], fi1), ("2", A2[layer - 1], fi2)):
                Y = X[fi]
                pre = Preprocessor.fit(Y, "center")
                U = fit_pca_truncated(pre.apply(Y), k=80).subspace(V)
                c[f"ref_{name}"] = capture(U.components, pre.apply_displacement(dref))
                c[f"arb_{name}"] = capture(U.components, pre.apply_displacement(darb))
            boots[layer].append(
                interaction(c["ref_1"], c["ref_2"], c["arb_1"], c["arb_2"]).I)
        if (b + 1) % 100 == 0:
            print(f"    {b + 1}/{a.boot}", flush=True)

    print(f"\n{'layer':>6} {'I':>9} {'95% CI':>20} {'excl 0':>7}")
    excl = []
    for layer, cells, I in out:
        bs = np.array(boots[layer])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        e = (lo > 0) or (hi < 0)
        excl.append(layer if e else None)
        star = " *" if layer == PRIMARY_LAYER else ""
        print(f"{layer:>6} {I.I:>+9.4f}  [{lo:>+7.4f}, {hi:>+7.4f}]  {'yes' if e else 'no':>7}{star}")

    json.dump({"v": V, "primary_layer": PRIMARY_LAYER, "n_pairs": n_eval,
               "layers": [{"layer": l, "I": I.I, "cells": {k2: v2 for k2, v2 in c.items()},
                           "ci": list(np.percentile(boots[l], [2.5, 97.5]))}
                          for l, c, I in out],
               "controls": {k2: (v2.tolist() if isinstance(v2, np.ndarray) else v2)
                            for k2, v2 in summary.items()}},
              open("results/analysis.json", "w"), indent=1)
    print("\nwrote results/analysis.json")


if __name__ == "__main__":
    main()
