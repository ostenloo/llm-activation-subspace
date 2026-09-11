"""Audit XSTest matched-pair structure for OPEN-1 / OPEN-2 / OPEN-3 (SPEC.md §5, §8).

Reports, per token-Levenshtein edit-distance cap:
  - number of usable refusal-flip base prompts
  - the focus-clean fit pool remaining after the asymmetric split (SPEC.md §5)
  - the largest refusal-balanced N_fit that pool supports

Run on the GPU host, where the dataset and tokenizer are cached:
    ssh fedora '/home/austin/unsupervised-concept-geometry/.venv/bin/python -' \
        < scripts/xstest_pair_audit.py
"""
import collections
import csv
import itertools
import json
import os

from transformers import AutoTokenizer

CSV = os.environ.get(
    "XSTEST_CSV",
    "/home/austin/.cache/huggingface/hub/datasets--Paul--XSTest/blobs/"
    "ad38204d70b3964387f9690169d11afbc4756d01",
)
MODEL = os.environ.get("MODEL", "meta-llama/Llama-3.1-8B-Instruct")
GATE_N = 225  # SPEC.md §8 simulation scale


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = list(csv.DictReader(open(CSV)))
    by_id = {r["id"]: r for r in rows}
    all_ids = set(by_id)
    ids = {r["id"]: tok(r["prompt"], add_special_tokens=False)["input_ids"] for r in rows}

    by_focus = collections.defaultdict(lambda: {"safe": [], "unsafe": []})
    for r in rows:
        by_focus[r["focus"]][r["label"]].append(r)

    # For each focus term, the closest cross-label prompt pair.
    best = {}
    for f, g in by_focus.items():
        if not (g["safe"] and g["unsafe"]):
            continue
        d, s, u = min(
            ((lev(ids[s["id"]], ids[u["id"]]), s, u)
             for s, u in itertools.product(g["safe"], g["unsafe"])),
            key=lambda t: t[0],
        )
        best[f] = {"dist": d, "safe_id": s["id"], "unsafe_id": u["id"],
                   "safe": s["prompt"], "unsafe": u["prompt"], "type": s["type"]}

    print(f"prompts={len(rows)}  focus groups with both labels={len(best)}")
    hist = collections.Counter(v["dist"] for v in best.values())
    cum = 0
    print("\nmin token-Levenshtein distance per focus group:")
    for d in sorted(hist):
        cum += hist[d]
        print(f"  d={d:>2}: {hist[d]:>3}  (cumulative <=d: {cum:>3})")

    print(f"\n{'cap':>4} {'base':>5} {'clean fit pool':>15} {'balanced N_fit':>15} {'vs gate':>8}")
    for thr in range(1, 7):
        sel = {f: v for f, v in best.items() if v["dist"] <= thr}
        used = {i for v in sel.values() for i in (v["safe_id"], v["unsafe_id"])}
        eval_focus = {by_id[i]["focus"] for i in used}
        # Focus-clean: a fit prompt sharing a focus term with an eval pair leaks
        # near-duplicate content into U_C. See SPEC.md §5.
        clean = {i for i in (all_ids - used) if by_id[i]["focus"] not in eval_focus}
        lab = collections.Counter(by_id[i]["label"] for i in clean)
        bal = 2 * min(lab["safe"], lab["unsafe"])
        print(f"{thr:>4} {len(sel):>5} {len(clean):>15} {bal:>15} "
              f"{'PASS' if bal >= GATE_N else 'FAIL':>8}")

    out = os.environ.get("OUT", "/tmp/xstest_pairs.json")
    json.dump(best, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
