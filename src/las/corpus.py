"""C1 construction: XSTest pair selection and the asymmetric focus-clean split (SPEC.md §5)."""

from __future__ import annotations

import collections
import csv
import dataclasses
import itertools

from .config import D_MAX_RECOMMENDED


@dataclasses.dataclass(frozen=True)
class FlipPair:
    """One refusal-flip pair: a base prompt and its safe/unsafe counterpart."""

    focus: str
    type: str
    dist: int
    safe_id: str
    unsafe_id: str
    safe: str
    unsafe: str


@dataclasses.dataclass(frozen=True)
class Split:
    """The asymmetric focus-clean allocation (§5)."""

    pairs: list[FlipPair]          # eval side: the refusal-flip base pairs
    fit_ids: list[str]             # fit side: focus-clean, label-balanced
    evicted_for_leakage: list[str]
    d_max: int

    @property
    def n_base_prompts(self) -> int:
        """Bootstrap cluster count (§9) and the §7 distribution's sample size."""
        return len(self.pairs)

    @property
    def n_fit(self) -> int:
        return len(self.fit_ids)


def token_levenshtein(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def load_xstest(path: str) -> list[dict]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    missing = {"id", "prompt", "type", "label", "focus"} - set(rows[0])
    if missing:
        raise ValueError(f"XSTest CSV missing columns: {sorted(missing)}")
    return rows


def select_pairs(rows: list[dict], token_ids: dict[str, list[int]]) -> dict[str, FlipPair]:
    """Closest cross-label prompt pair within each focus group (§5, OPEN-1).

    The edit locus is the focus term: XSTest is natively matched, with 18 types
    forming 9 X / contrast_X pairs joined by the focus column.
    """
    by_focus = collections.defaultdict(lambda: {"safe": [], "unsafe": []})
    for r in rows:
        by_focus[r["focus"]][r["label"]].append(r)

    out = {}
    for focus, g in by_focus.items():
        if not (g["safe"] and g["unsafe"]):
            continue
        d, s, u = min(
            ((token_levenshtein(token_ids[s["id"]], token_ids[u["id"]]), s, u)
             for s, u in itertools.product(g["safe"], g["unsafe"])),
            key=lambda t: t[0],
        )
        out[focus] = FlipPair(focus=focus, type=s["type"], dist=d, safe_id=s["id"],
                              unsafe_id=u["id"], safe=s["prompt"], unsafe=u["prompt"])
    return out


def build_split(
    rows: list[dict],
    pairs: dict[str, FlipPair],
    d_max: int = D_MAX_RECOMMENDED,
    balance: bool = True,
) -> Split:
    """Allocate pairs to eval and the focus-clean remainder to fit (§5).

    Not 50/50: eval needs only enough prompts to supply base pairs, while fit needs
    enough for PCA to be identifiable at all (§8). Equal halves would starve fit.

    Disjointness of prompt identity is insufficient. XSTest prompts are keyed by a
    focus term, and a fit prompt sharing a focus term with an eval pair puts
    near-duplicate content into U_C -- a weaker form of exactly the leak the split
    exists to prevent. Every such prompt is evicted, which is costly but not optional.
    """
    by_id = {r["id"]: r for r in rows}
    selected = [p for p in pairs.values() if p.dist <= d_max]
    used = {i for p in selected for i in (p.safe_id, p.unsafe_id)}
    eval_focus = {by_id[i]["focus"] for i in used}

    leaked = sorted(i for i in set(by_id) - used if by_id[i]["focus"] in eval_focus)
    clean = sorted(set(by_id) - used - set(leaked))

    if balance:
        # C1 is refusal-balanced (§5); an imbalanced fit set would let label
        # composition, not corpus identity, drive the leading subspace.
        by_label = collections.defaultdict(list)
        for i in clean:
            by_label[by_id[i]["label"]].append(i)
        n = min(len(by_label["safe"]), len(by_label["unsafe"]))
        clean = sorted(by_label["safe"][:n] + by_label["unsafe"][:n])

    return Split(
        pairs=sorted(selected, key=lambda p: (p.dist, p.focus)),
        fit_ids=clean,
        evicted_for_leakage=leaked,
        d_max=d_max,
    )
