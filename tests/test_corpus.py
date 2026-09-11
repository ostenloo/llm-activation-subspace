import collections
import os

import pytest

from las.corpus import build_split, load_xstest, select_pairs, token_levenshtein

XSTEST = ("/home/austin/.cache/huggingface/hub/datasets--Paul--XSTest/blobs/"
          "ad38204d70b3964387f9690169d11afbc4756d01")


def _rows():
    """Focus groups at known distances: 'kill' and 'zebra' flip in one
    token, 'brew' in three (1 substitution + 2 insertions)."""
    return [
        {"id": "1", "prompt": "kill time",   "type": "homonyms", "label": "safe",   "focus": "kill"},
        {"id": "2", "prompt": "kill people", "type": "contrast", "label": "unsafe", "focus": "kill"},
        {"id": "3", "prompt": "brew tea",    "type": "homonyms", "label": "safe",   "focus": "brew"},
        {"id": "4", "prompt": "brew a bad thing", "type": "contrast", "label": "unsafe", "focus": "brew"},
        {"id": "5", "prompt": "kill a process", "type": "homonyms", "label": "safe", "focus": "kill"},
        {"id": "6", "prompt": "unrelated one", "type": "definitions", "label": "safe",   "focus": "zebra"},
        {"id": "7", "prompt": "unrelated two", "type": "contrast",    "label": "unsafe", "focus": "zebra"},
    ]


def _ids(rows):
    return {r["id"]: r["prompt"].split() for r in rows}


def test_token_levenshtein():
    assert token_levenshtein("a b c".split(), "a b c".split()) == 0
    assert token_levenshtein("a b c".split(), "a X c".split()) == 1
    assert token_levenshtein("a b".split(), "a b c d".split()) == 2


def test_select_pairs_takes_the_closest_cross_label_pair_per_focus():
    rows = _rows()
    pairs = select_pairs(rows, _ids(rows))
    assert pairs["kill"].dist == 1
    assert pairs["brew"].dist == 3
    # 'kill' has two safe candidates; the closer one wins.
    assert pairs["kill"].safe_id == "1"


def test_select_pairs_skips_focus_groups_missing_a_label():
    rows = _rows() + [{"id": "8", "prompt": "only safe", "type": "t",
                       "label": "safe", "focus": "lonely"}]
    assert "lonely" not in select_pairs(rows, _ids(rows))


def test_d_max_controls_which_pairs_are_selected():
    rows = _rows()
    pairs = select_pairs(rows, _ids(rows))
    assert {p.focus for p in build_split(rows, pairs, d_max=1, balance=False).pairs} == {"kill", "zebra"}
    assert {p.focus for p in build_split(rows, pairs, d_max=3, balance=False).pairs} == {"kill", "brew", "zebra"}


def test_split_evicts_focus_relatives_from_the_fit_set():
    """Prompt-identity disjointness is not enough: id 5 shares the 'kill' focus
    with an eval pair, so it carries near-duplicate content into U_C (§5)."""
    rows = _rows()
    split = build_split(rows, select_pairs(rows, _ids(rows)), d_max=1, balance=False)
    assert "5" in split.evicted_for_leakage
    assert "5" not in split.fit_ids


def test_fit_and_eval_are_disjoint():
    rows = _rows()
    split = build_split(rows, select_pairs(rows, _ids(rows)), d_max=3, balance=False)
    used = {i for p in split.pairs for i in (p.safe_id, p.unsafe_id)}
    assert used.isdisjoint(split.fit_ids)


def test_balance_makes_the_fit_set_refusal_balanced():
    rows = [{"id": str(i), "prompt": f"p{i}", "type": "t",
             "label": "safe" if i < 8 else "unsafe", "focus": f"f{i}"} for i in range(12)]
    split = build_split(rows, {}, d_max=3, balance=True)
    labels = collections.Counter(r["label"] for r in rows if r["id"] in split.fit_ids)
    assert labels["safe"] == labels["unsafe"] == 4


def test_n_base_prompts_is_the_bootstrap_cluster_count():
    rows = _rows()
    split = build_split(rows, select_pairs(rows, _ids(rows)), d_max=3, balance=False)
    assert split.n_base_prompts == len(split.pairs) == 3


@pytest.mark.skipif(not os.path.exists(XSTEST), reason="XSTest not cached here")
def test_real_xstest_shape_matches_the_spec():
    rows = load_xstest(XSTEST)
    assert len(rows) == 450
    labels = collections.Counter(r["label"] for r in rows)
    assert labels == {"safe": 250, "unsafe": 200}
    assert len({r["type"] for r in rows}) == 18


def test_empty_focus_rows_are_not_treated_as_one_group():
    """75 of XSTest's 450 rows leave focus empty. Grouping them together would
    select a 'matched' pair by minimum distance over unrelated prompts."""
    rows = _rows() + [
        {"id": "9",  "prompt": "alpha one", "type": "historical", "label": "safe",   "focus": ""},
        {"id": "10", "prompt": "beta two",  "type": "contrast",   "label": "unsafe", "focus": ""},
    ]
    assert "" not in select_pairs(rows, _ids(rows))


def test_empty_focus_rows_are_not_evicted_as_leakage():
    """An absent key cannot leak; treating '' as shared would evict every one."""
    rows = _rows() + [
        {"id": "9",  "prompt": "alpha one", "type": "historical", "label": "safe",   "focus": ""},
        {"id": "10", "prompt": "beta two",  "type": "contrast",   "label": "unsafe", "focus": ""},
    ]
    split = build_split(rows, select_pairs(rows, _ids(rows)), d_max=3, balance=False)
    assert "9" not in split.evicted_for_leakage
    assert "10" not in split.evicted_for_leakage
    assert {"9", "10"} <= set(split.fit_ids)
