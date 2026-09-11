"""Author and validate the arb substitution family (SPEC.md §17, OPEN-2).

The arb family is authored by the researcher, not generated, because a
constructed substitution family can carry a systematic signature and the
procedure that generates it cannot audit it.

  --emit    write the worklist: one row per ref base prompt, with the edit
            distance that row's arb variant must match, and a blank slot.
  --check   validate a filled-in worklist; says how far off each row is.
  --tokens  show how a phrase tokenizes, for when a row will not land.

    python scripts/arb_authoring.py --emit  --out results/arb_worklist.csv
    python scripts/arb_authoring.py --check --in  results/arb_worklist.csv
    python scripts/arb_authoring.py --tokens "a party balloon"
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


def _diff(a_ids, b_ids, tok) -> str:
    """Token-level edit script, so a miss shows what is actually being counted."""
    import difflib
    a = [tok.decode([t]) for t in a_ids]
    b = [tok.decode([t]) for t in b_ids]
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b).get_opcodes():
        if tag == "replace":
            out.append(f"{a[i1:i2]} -> {b[j1:j2]}")
        elif tag == "delete":
            out.append(f"drop {a[i1:i2]}")
        elif tag == "insert":
            out.append(f"add {b[j1:j2]}")
    return "; ".join(out) or "(identical)"


def tokens(text: str) -> None:
    """How many tokens is this phrase, and where do the splits fall."""
    tok = _tok()
    ids = tok(text, add_special_tokens=False)["input_ids"]
    parts = [tok.decode([t]) for t in ids]
    print(f"{len(ids)} token(s): " + " | ".join(repr(p) for p in parts))


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

    problems, blank, ok = [], 0, 0
    lens = {"base": [], "ref": [], "arb": []}
    for i, r in enumerate(rows, 2):  # 2 = first data line in the file
        arb = r["arb_variant"].strip()
        if not arb:
            blank += 1
            continue
        want = int(r["d_required"])
        got = token_levenshtein(enc(r["base_prompt"]), enc(arb))
        label = r["focus"] or r["type"]
        if arb == r["base_prompt"]:
            problems.append(f"  line {i} [{label}] arb is identical to base_prompt")
        elif arb == r["ref_variant"]:
            problems.append(f"  line {i} [{label}] arb is the ref_variant")
        elif got != want:
            delta = want - got
            how = (f"add {delta} more changed token{'s' if delta > 1 else ''}"
                   if delta > 0 else
                   f"change {-delta} fewer token{'s' if -delta > 1 else ''}")
            problems.append(
                f"  line {i} [{label}] distance {got}, need {want} -> {how}\n"
                f"      base: {r['base_prompt']}\n"
                f"      arb : {arb}\n"
                f"      diff: {_diff(enc(r['base_prompt']), enc(arb), tok)}")
        else:
            ok += 1
        lens["base"].append(len(enc(r["base_prompt"])))
        lens["ref"].append(len(enc(r["ref_variant"])))
        lens["arb"].append(len(enc(arb)))

    n = len(rows) - blank
    print(f"rows: {len(rows)}   ok: {ok}   need work: {n - ok}   blank: {blank}")
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
    ap.add_argument("--tokens", metavar="TEXT")
    ap.add_argument("--split", default="results/c1_split.json")
    ap.add_argument("--out", default="results/arb_worklist.csv")
    ap.add_argument("--in", dest="inp", default="results/arb_worklist.csv")
    a = ap.parse_args()
    if a.tokens:
        tokens(a.tokens)
    elif a.emit:
        emit(a.split, a.out)
    elif a.check:
        sys.exit(check(a.inp))
    else:
        ap.error("pass --emit, --check, or --tokens")


if __name__ == "__main__":
    main()
