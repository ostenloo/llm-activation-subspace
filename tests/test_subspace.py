import numpy as np
import pytest

from las.config import LOGIT_EPS
from las.subspace import (
    Preprocessor, capture, capture_per_prompt, fit_subspace, interaction,
    logit, pca_fit, q_at_variance,
)


def planted(n=120, D=60, k=4, seed=0):
    """n points whose variance lives in a known k-dimensional subspace."""
    rng = np.random.default_rng(seed)
    B = np.linalg.qr(rng.standard_normal((D, k)))[0]
    return (rng.standard_normal((n, k)) * np.array([9.0, 6.0, 3.0, 1.0])) @ B.T, B


def test_pca_recovers_planted_subspace():
    X, B = planted()
    comps, evr, _ = pca_fit(X - X.mean(0))
    assert np.isclose(evr[:4].sum(), 1.0, atol=1e-8)   # rank-4 data
    # Leading 4 PCs span the planted subspace.
    assert np.isclose(np.sum((comps[:4] @ B) ** 2) / 4, 1.0, atol=1e-8)


def test_rank_is_capped_by_sample_size():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((20, 500))
    _, evr, _ = pca_fit(X - X.mean(0))
    assert evr.size <= 19      # n - 1 after centering


def test_q_at_variance_is_the_smallest_reaching_v():
    evr = np.array([0.5, 0.25, 0.15, 0.10])
    assert q_at_variance(evr, 0.50) == 1
    assert q_at_variance(evr, 0.75) == 2
    assert q_at_variance(evr, 0.76) == 3
    assert q_at_variance(evr, 1.00) == 4


def test_capture_is_one_when_displacements_lie_in_the_subspace():
    X, B = planted()
    U = fit_subspace(X - X.mean(0), 0.999)
    rng = np.random.default_rng(3)
    dZ = rng.standard_normal((30, 4)) @ B.T     # inside the planted span
    assert capture(U.components, dZ) == pytest.approx(1.0, abs=1e-8)


def test_capture_is_zero_for_orthogonal_displacements():
    X, B = planted(D=60, k=4)
    U = fit_subspace(X - X.mean(0), 0.999)
    # Build a direction orthogonal to the planted subspace.
    rng = np.random.default_rng(4)
    w = rng.standard_normal(60)
    w -= B @ (B.T @ w)
    dZ = np.outer(rng.standard_normal(10), w / np.linalg.norm(w))
    assert capture(U.components, dZ) == pytest.approx(0.0, abs=1e-8)


def test_capture_is_a_ratio_of_sums_not_a_mean_of_ratios():
    """A is dominated by the largest ||Dz_i||; that is why §7 mandates the
    per-prompt distribution alongside it."""
    U = np.eye(1, 3)                       # project onto e0
    dZ = np.array([[0.0, 1.0, 0.0],        # a_i = 0, small
                   [10.0, 0.0, 0.0]])      # a_i = 1, large
    assert capture(U, dZ) == pytest.approx(100 / 101)
    assert capture_per_prompt(U, dZ).mean() == pytest.approx(0.5)


def test_capture_is_permutation_invariant():
    """§7's discarded shuffled-pair null: A depends only on the multiset {Dz_i},
    so A_shuffled == A_paired identically. Guards against reintroducing it."""
    rng = np.random.default_rng(5)
    U = np.linalg.qr(rng.standard_normal((20, 5)))[0].T
    dZ = rng.standard_normal((40, 20))
    assert capture(U, dZ) == pytest.approx(capture(U, rng.permutation(dZ)), abs=1e-12)


def test_centering_cancels_in_displacements_but_scaling_does_not():
    rng = np.random.default_rng(6)
    X = rng.standard_normal((50, 8)) * np.arange(1, 9)
    dZ = rng.standard_normal((10, 8))
    assert np.allclose(Preprocessor.fit(X, "center").apply_displacement(dZ), dZ)
    scaled = Preprocessor.fit(X, "standardize").apply_displacement(dZ)
    assert not np.allclose(scaled, dZ)
    assert np.allclose(scaled, dZ / X.std(0))


def test_standardize_leaves_dead_coordinates_alone():
    X = np.ones((10, 3)) * [1.0, 2.0, 3.0]      # zero variance everywhere
    out = Preprocessor.fit(X, "standardize").apply(X)
    assert np.all(np.isfinite(out)) and np.allclose(out, 0.0)


def test_logit_boundary_rule():
    assert logit(0.5) == (0.0, False)
    v, clipped = logit(0.0)
    assert clipped and v == pytest.approx(np.log(LOGIT_EPS / (1 - LOGIT_EPS)))
    assert logit(1.0)[1] is True


def test_interaction_cancels_a_shared_corpus_advantage():
    """If corpus 1's advantage is the same on both rows, I = 0: that is the
    generic-concentration effect the arb row exists to remove (§3)."""
    out = interaction(a_ref_1=0.60, a_ref_2=0.40, a_arb_1=0.60, a_arb_2=0.40)
    assert out.I == pytest.approx(0.0, abs=1e-12)
    assert not out.any_clipped


def test_interaction_is_positive_when_ref_gains_more_than_arb():
    out = interaction(a_ref_1=0.70, a_ref_2=0.40, a_arb_1=0.60, a_arb_2=0.40)
    assert out.I > 0


def test_interaction_reports_clipped_cells_and_keeps_raw_contrast():
    out = interaction(a_ref_1=1.0, a_ref_2=0.4, a_arb_1=0.6, a_arb_2=0.4)
    assert out.clipped == ("ref_1",)
    assert out.I_untransformed == pytest.approx((1.0 - 0.4) - (0.6 - 0.4))


def test_logit_scale_can_disagree_with_raw_scale():
    """Why §3 fixes the link in advance: at unequal baselines the sign of the
    difference-in-differences is not link-invariant."""
    out = interaction(a_ref_1=0.98, a_ref_2=0.90, a_arb_1=0.50, a_arb_2=0.40)
    assert out.I_untransformed < 0 < out.I


def test_zero_displacement_is_rejected_not_silently_zero():
    with pytest.raises(ValueError):
        capture(np.eye(1, 3), np.zeros((4, 3)))
