import numpy as np
import pytest

from las.gate import gate_cell, split_half_overlap, subspace_overlap


def test_overlap_is_one_for_identical_bases():
    rng = np.random.default_rng(0)
    U = np.linalg.qr(rng.standard_normal((30, 6)))[0].T
    o = subspace_overlap(U, U)
    assert o.overlap == pytest.approx(1.0) and o.q_gap == 0


def test_overlap_is_zero_for_orthogonal_bases():
    Q = np.linalg.qr(np.random.default_rng(1).standard_normal((30, 12)))[0]
    assert subspace_overlap(Q[:, :6].T, Q[:, 6:].T).overlap == pytest.approx(0.0, abs=1e-12)


def test_overlap_truncates_to_q_star_and_reports_the_gap():
    """Under variance-matching the halves reach v at different q; the metric is
    undefined unless the normalizing q is named (§8)."""
    Q = np.linalg.qr(np.random.default_rng(2).standard_normal((30, 20)))[0]
    o = subspace_overlap(Q[:, :4].T, Q[:, :9].T)
    assert (o.q1, o.q2, o.q_star, o.q_gap) == (4, 9, 4, 5)
    assert o.overlap == pytest.approx(1.0)   # truncated U2 contains U1


def test_truncation_refuses_the_free_win_from_a_wider_basis():
    """Normalizing an unequal-width product by min(q1,q2) would score a nested
    pair 1.0 for the wrong reason. Truncation must not do that when the extra
    components are what carried the match."""
    Q = np.linalg.qr(np.random.default_rng(3).standard_normal((30, 20)))[0]
    U1 = Q[:, :3].T
    U2 = np.vstack([Q[:, 10:13].T, Q[:, :3].T])   # U1 present, but only past q*=3
    o = subspace_overlap(U1, U2)
    assert o.overlap == pytest.approx(0.0, abs=1e-12)


def test_split_half_overlap_is_high_for_an_identified_subspace():
    """Fast decay at n >> q: both halves recover the same leading directions."""
    rng = np.random.default_rng(4)
    B = np.linalg.qr(rng.standard_normal((80, 40)))[0]
    X = (rng.standard_normal((400, 40)) * np.exp(-np.arange(40) / 1.5)) @ B.T
    med, _ = split_half_overlap(X, 0.75, rng, n_splits=10)
    assert med > 0.9


def test_split_half_overlap_is_low_for_an_unidentified_subspace():
    """Slow decay at small n: two samples from the same corpus give near-unrelated
    subspaces, and A_11 vs A_12 would measure sampling noise (§8)."""
    rng = np.random.default_rng(5)
    X = rng.standard_normal((60, 400))
    med, _ = split_half_overlap(X, 0.75, rng, n_splits=10)
    assert med < 0.5


def test_gate_passes_a_genuinely_identified_cell():
    rng = np.random.default_rng(6)
    B = np.linalg.qr(rng.standard_normal((60, 30)))[0]
    X = (rng.standard_normal((400, 30)) * np.exp(-np.arange(30) / 1.2)) @ B.T
    assert gate_cell(X, 0.75, rng, n_splits=10).verdict == "pass"


def test_gate_flags_a_rogue_carried_cell():
    """The §8 defect this rule exists for: overlap supplied by a couple of
    high-variance coordinates that both halves recover trivially, with the rest
    of the subspace pure noise. Raw passes; ablated must not."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((60, 300))          # unidentifiable on its own
    X[:, 11] *= 400.0                           # two rogue coordinates
    X[:, 42] *= 300.0
    res = gate_cell(X, 0.50, rng, k=5, n_splits=10)
    assert res.verdict == "rogue_carried"
    assert res.raw >= 0.70 > res.ablated
    assert {11, 42} <= set(res.rogue_idx.tolist())


def test_gate_fails_an_unidentified_cell():
    rng = np.random.default_rng(8)
    assert gate_cell(rng.standard_normal((60, 400)), 0.75, rng, n_splits=10).verdict == "fail"


def test_gate_result_only_proceeds_on_pass():
    rng = np.random.default_rng(9)
    res = gate_cell(rng.standard_normal((60, 400)), 0.75, rng, n_splits=5)
    assert res.proceeds is (res.verdict == "pass")


def test_gate_cells_matches_single_level_calls():
    """Batching the levels must not change any verdict."""
    from las.gate import gate_cells
    rng = np.random.default_rng(20)
    B = np.linalg.qr(rng.standard_normal((60, 30)))[0]
    X = (rng.standard_normal((400, 30)) * np.exp(-np.arange(30) / 1.2)) @ B.T
    vs = (0.50, 0.75, 0.90)
    batched = gate_cells(X, vs, np.random.default_rng(21), n_splits=10)
    for v in vs:
        one = gate_cell(X, v, np.random.default_rng(21), n_splits=10)
        assert batched[v].verdict == one.verdict
        assert batched[v].raw == pytest.approx(one.raw, abs=0.05)


def test_split_half_overlaps_shares_splits_across_levels():
    """All levels are read off the same halvings, so between-level differences are
    spectrum, not split noise."""
    from las.gate import split_half_overlaps
    rng = np.random.default_rng(22)
    X = rng.standard_normal((80, 200))
    res = split_half_overlaps(X, (0.50, 0.75, 0.90), rng, n_splits=8)
    assert set(res) == {0.50, 0.75, 0.90}
    # Same halvings => same half sizes, and q is non-decreasing in v per split.
    for i in range(8):
        qs = [res[v][1][i].q1 for v in (0.50, 0.75, 0.90)]
        assert qs == sorted(qs)


def test_gate_cells_computes_the_rogue_set_once():
    """The rogue set is a property of the corpus and layer, not of v."""
    from las.gate import gate_cells
    rng = np.random.default_rng(23)
    X = rng.standard_normal((60, 200))
    X[:, 5] *= 300.0
    res = gate_cells(X, (0.50, 0.75), rng, n_splits=5)
    a, b = res[0.50].rogue_idx, res[0.75].rogue_idx
    assert np.array_equal(a, b) and 5 in a
