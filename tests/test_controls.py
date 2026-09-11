import numpy as np
import pytest

from las.controls import (
    ablate_coordinates, cosine_contributions, participation_ratio,
    rank_band, rogue_report, spectrally_matched_basis,
)
from las.subspace import fit_subspace, pca_fit


def test_rank_band_full_width_when_rank_allows():
    b = rank_band(q=10, rank=100)
    assert (b.start, b.stop, b.kind) == (10, 20, "rank_band")
    assert b.comparable


def test_rank_band_narrows_when_spectrum_runs_out():
    """At v=0.90 under slow decay, q+1..2q can run past the available rank."""
    b = rank_band(q=80, rank=100)
    assert (b.start, b.stop, b.kind) == (80, 100, "narrowed")
    assert b.applicable and b.width == 20 and not b.comparable


def test_rank_band_inapplicable_when_no_trailing_components_exist():
    b = rank_band(q=100, rank=100)
    assert b.kind == "inapplicable" and not b.applicable and b.width == 0


def test_rank_band_never_overlaps_the_leading_set():
    """The control is a foil for the leading q; overlapping it would be circular."""
    for rank in range(2, 60):
        for q in range(1, rank + 1):
            b = rank_band(q, rank)
            if b.applicable:
                assert b.start >= q and b.stop <= rank and b.width > 0


def test_spectrally_matched_basis_is_orthonormal_and_in_the_data_span():
    rng = np.random.default_rng(0)
    B = np.linalg.qr(rng.standard_normal((40, 5)))[0]
    X = (rng.standard_normal((60, 5)) * [8, 4, 2, 1, 0.5]) @ B.T
    R = spectrally_matched_basis(X, q=3, rng=rng)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
    # Anisotropy is matched: the draws live in the data's own span, not all of R^D.
    assert np.sum((R @ B) ** 2) / 3 == pytest.approx(1.0, abs=1e-8)


def test_spectrally_matched_basis_refuses_to_exceed_the_rank():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        spectrally_matched_basis(rng.standard_normal((10, 50)), q=20, rng=rng)


def test_spectrally_matched_null_is_a_harder_floor_than_isotropic():
    """The naive q/D floor assumes isotropy, which is the wrong null: a
    spectrum-matched random subspace captures far more than q/D of a
    displacement drawn from the same anisotropic distribution (§7)."""
    rng = np.random.default_rng(2)
    D, q = 200, 5
    B = np.linalg.qr(rng.standard_normal((D, 20)))[0]
    spec = np.exp(-np.arange(20) / 2.0)
    X = (rng.standard_normal((300, 20)) * spec) @ B.T
    dZ = (rng.standard_normal((100, 20)) * spec) @ B.T
    R = spectrally_matched_basis(X, q, rng)
    a_null = np.sum((dZ @ R.T) ** 2) / np.sum(dZ ** 2)
    assert a_null > 10 * (q / D)


def test_cosine_contributions_sum_to_anisotropy():
    """CC_i sums to A-hat(f_l), the mean pairwise cosine similarity."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((40, 12)) + 5.0        # offset => anisotropic
    U = X / np.linalg.norm(X, axis=1, keepdims=True)
    n = U.shape[0]
    G = U @ U.T
    expected = (G.sum() - np.trace(G)) / (n * (n - 1))
    assert cosine_contributions(X).sum() == pytest.approx(expected, abs=1e-12)


def test_cosine_contributions_find_a_planted_rogue_dimension():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((80, 30))
    X[:, 7] += 50.0                                 # a rogue coordinate: far off-origin
    assert int(np.argmax(cosine_contributions(X))) == 7


def test_participation_ratio_bounds():
    D = 64
    assert participation_ratio(np.eye(1, D)[0]) == pytest.approx(1.0)
    assert participation_ratio(np.ones(D)) == pytest.approx(D)


def test_rogue_report_flags_a_coordinate_dominated_leading_pc():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((100, 40))
    X[:, 3] *= 60.0                                 # one huge-variance coordinate
    comps, _, _ = pca_fit(X - X.mean(0))
    rep = rogue_report(X, comps, k=5)
    assert 3 in rep.rogue_idx
    assert rep.participation_ratios[0] < 1.5        # PC1 is essentially one axis
    assert rep.var_share_top_k > 0.9


def test_rogue_report_on_delocalized_data_is_unalarming():
    rng = np.random.default_rng(6)
    X = rng.standard_normal((200, 40))
    comps, _, _ = pca_fit(X - X.mean(0))
    rep = rogue_report(X, comps, k=5)
    assert rep.participation_ratios[0] > 10
    assert rep.var_share_top_k < 0.35


def test_ablate_coordinates_drops_exactly_those_columns():
    X = np.arange(20, dtype=float).reshape(4, 5)
    out = ablate_coordinates(X, np.array([1, 3]))
    assert out.shape == (4, 3)
    assert np.allclose(out, X[:, [0, 2, 4]])


def test_rank_band_capture_is_nonzero_and_comes_from_the_full_decomposition():
    """Regression: the band was sliced off Subspace.components, which holds only
    the leading q rows, so [q:2q] was empty and every capture read exactly 0."""
    from las.subspace import capture, fit_pca
    rng = np.random.default_rng(40)
    X = rng.standard_normal((200, 300)) * np.exp(-np.arange(300) / 40)
    fit = fit_pca(X - X.mean(0))
    U = fit.subspace(0.50)
    b = rank_band(U.q, fit.rank)
    assert b.applicable
    band = fit.band(b.start, b.stop)
    assert band.shape[0] == b.width > 0
    dZ = rng.standard_normal((30, 300))
    assert capture(band, dZ) > 0.0
    # And the truncated subspace could not have supplied it.
    assert U.components[b.start:b.stop].shape[0] == 0


def test_band_rejects_a_range_past_the_spectrum():
    from las.subspace import fit_pca
    fit = fit_pca(np.random.default_rng(41).standard_normal((30, 100)))
    with pytest.raises(ValueError):
        fit.band(0, fit.rank + 5)
