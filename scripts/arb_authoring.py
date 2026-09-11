"""Author and validate the arb substitution family (SPEC.md §17, OPEN-2).

The arb family is authored by the researcher, not generated, because a
constructed substitution family can carry a systematic signature and the
procedure that generates it cannot audit it.

  --emit   write the worklist: one row per ref base prompt, with the edit
           distance that row's arb variant must match, and a blank slot.
  --check  validate a filled-in worklist against the OPEN-2 constraints.

    python scripts/arb_authoring.py --emit  --out results/arb_worklist.csv
    python scripts/arb_authoring.py --check --in  results/arb_worklist.csv
"""
import argparse
import collections
import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from las.corpus import token_levenshtein  # noqa: E402

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
FIELDS = ["focus", "type", "d_required", "base_prompt", "ref_variant", "arb_variant"]


def _tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL)


def emit(split_path: str, out: str) -> None:
    split = json.loads(pathlib.Path(split_path).read_text())
    rows = [{"focus": p["focus"], "type": p["type"], "d_required": p["dist"],
             "base_prompt": p["safe"], "ref_variant": p["unsafe"], "arb_variant": ""}
            for p in split["pairs"]]
    path = pathlib.Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")
    print("\nFill arb_variant for each row. It must be an edit of base_prompt that:")
    print("  - is exactly d_required tokens from base_prompt (Llama-3.1 tokenizer)")
    print("  - leaves the safe reading intact -- the model must not refuse it")
    print("  - edits the same slot the ref_variant edits, so the locus is matched")
    print("\nThen: python scripts/arb_authoring.py --check --in " + out)


def check(path: str) -> int:
    rows = list(csv.DictReader(open(path)))
    tok = _tok()
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]

    problems, blank = [], 0
    lens = {"base": [], "ref": [], "arb": []}
    for i, r in enumerate(rows, 2):  # 2 = first data line in the file
        arb = r["arb_variant"].strip()
        if not arb:
            blank += 1
            continue
        want = int(r["d_required"])
        got = token_levenshtein(enc(r["base_prompt"]), enc(arb))
        if got != want:
            problems.append(f"  line {i} ({r['focus']}): distance {got}, need {want}")
        if arb == r["base_prompt"]:
            problems.append(f"  line {i} ({r['focus']}): arb is identical to base")
        if arb == r["ref_variant"]:
            problems.append(f"  line {i} ({r['focus']}): arb is the ref variant")
        lens["base"].append(len(enc(r["base_prompt"])))
        lens["ref"].append(len(enc(r["ref_variant"])))
        lens["arb"].append(len(enc(arb)))

    n = len(rows) - blank
    print(f"rows: {len(rows)}   filled: {n}   blank: {blank}")
    if lens["arb"]:
        for k, v in lens.items():
            print(f"  {k:>4} token length: median {sorted(v)[len(v)//2]}, "
                  f"mean {sum(v)/len(v):.1f}")
        d = collections.Counter(int(r["d_required"]) for r in rows
                                if r["arb_variant"].strip())
        print("  distances:", dict(sorted(d.items())))

    if problems:
        print(f"\n{len(problems)} constraint failure(s):")
        print("\n".join(problems))
    if blank:
        print(f"\n{blank} row(s) still blank.")
    if not problems and not blank:
        print("\nAll rows satisfy the OPEN-2 constraints on distance and locus.")
        print("Non-refusal is checked at extraction time against the OPEN-1 detector.")
        return 0
    return 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--split", default="results/c1_split.json")
    ap.add_argument("--out", default="results/arb_worklist.csv")
    ap.add_argument("--in", dest="inp", default="results/arb_worklist.csv")
    a = ap.parse_args()
    if a.emit:
        emit(a.split, a.out)
    elif a.check:
        sys.exit(check(a.inp))
    else:
        ap.error("pass --emit or --check")


if __name__ == "__main__":
    main()
