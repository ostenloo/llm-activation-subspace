"""Build the C1 refusal-balanced corpus split (SPEC.md §5).

Pair selection plus the asymmetric focus-clean allocation. Writes the split to JSON
so extraction reads a frozen artifact rather than re-deriving it.

This computes no activations and no A value: it is the §10 step-1 input that OPEN-1
and OPEN-3 need, not the start of extraction.

    ssh fedora 'cd ~/llm-activation-subspace && \
        /home/austin/unsupervised-concept-geometry/.venv/bin/python scripts/build_c1.py'
"""
import argparse
import collections
import dataclasses
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from las.config import D_MAX_RECOMMENDED, GATE_SIM_N  # noqa: E402
from las.corpus import build_split, load_xstest, select_pairs  # noqa: E402

XSTEST = ("/home/austin/.cache/huggingface/hub/datasets--Paul--XSTest/blobs/"
          "ad38204d70b3964387f9690169d11afbc4756d01")
MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=XSTEST)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--d-max", type=int, default=D_MAX_RECOMMENDED)
    ap.add_argument("--out", default="results/c1_split.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    rows = load_xstest(args.csv)
    by_id = {r["id"]: r for r in rows}
    ids = {r["id"]: tok(r["prompt"], add_special_tokens=False)["input_ids"] for r in rows}
    pairs = select_pairs(rows, ids)
    split = build_split(rows, pairs, d_max=args.d_max)

    labels = collections.Counter(by_id[i]["label"] for i in split.fit_ids)
    print(f"d_max            = {split.d_max}")
    print(f"base prompts     = {split.n_base_prompts}   (bootstrap clusters, SPEC §9)")
    print(f"fit set          = {split.n_fit}   ({labels['safe']} safe / {labels['unsafe']} unsafe)")
    print(f"evicted (leakage)= {len(split.evicted_for_leakage)}")
    print(f"gate scale N={GATE_SIM_N}: {'PASS' if split.n_fit >= GATE_SIM_N else 'FAIL'}"
          f"  (margin {split.n_fit - GATE_SIM_N:+d})")

    dist = collections.Counter(p.dist for p in split.pairs)
    print("\nselected pair distances: " +
          ", ".join(f"d={d}: {dist[d]}" for d in sorted(dist)))
    print("\ntypes represented: " +
          f"{len({p.type for p in split.pairs})} of 18")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"d_max": split.d_max,
               "pairs": [dataclasses.asdict(p) for p in split.pairs],
               "fit_ids": split.fit_ids,
               "evicted_for_leakage": split.evicted_for_leakage}, out.open("w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
