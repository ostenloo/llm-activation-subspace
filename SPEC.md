# Pre-registration: Is the global activation subspace a property of the model or of the probe corpus?

**Status:** DRAFT v4 — decisions closed in §17, pending two things before freeze: the 45 researcher-authored `arb` substitutions (§17, `OPEN-2`) and the §8 gate, which closes `OPEN-3`. All closures in §17 were made by Claude under delegated judgment on 2026-09-10 and are flagged as such so they can be overridden individually.
**Date drafted:** 2026-09-10 (v1)
**Date revised:** 2026-09-10 (v3 — see §13 revision history)
**Compute and data environment:** verified 2026-09-10, §15
**Model:** Llama-3.1-8B-Instruct
**Relationship to prior work:** downstream of the refusal-geometry project (v_ref, XSTest, §18). Reuses that infrastructure; does not depend on its unfinished items.

---

## 0. Scope discipline

This document covers one experiment. The following are explicitly **out of scope** and must not be added without a dated amendment: active acquisition / BALD, geodesic distortion surfaces, intrinsic-dimension estimation, reach estimation, topology, synthetic manifold calibration, Jacobian / pullback-metric computation, OLMo checkpoint sweeps, sample-complexity bounds.

Those are downstream of whether this experiment finds a phenomenon.

---

## 1. Claim under test

Reported low-dimensional structure in LLM activation spaces is typically obtained by running PCA (or a manifold method) on activations from some probe corpus, then interpreting the leading subspace as a property of the model. That subspace is a joint function of the model and the corpus.

**Question.** Does a global activation subspace induced by a refusal-balanced corpus preferentially capture behaviourally validated refusal-flip displacements, relative to the subspace induced by a general corpus, beyond the generic concentration advantage of the narrower corpus, and after accounting for spectral anisotropy?

**Why this framing rather than a manifold-recovery framing.** The model's input domain is discrete token sequences. There is no measure-free geometry to recover; the object is `G(f_θ, P_prompt)`. Corpus dependence is therefore not a nuisance to control away — it is the thing being measured.

**Relation to existing literature.** Anisotropy (Ethayarajh 2019) and rogue-dimension dominance of similarity geometry (Timkey & van Schijndel 2021) are treated here as *confound layers*, not as the novelty. Both are controlled for (§6, §7). The novel claim concerns whether the leading subspace itself is corpus-contingent and whether it captures behaviourally meaningful local variation.

Both citations were verified against the papers in the project corpus on 2026-09-10 (§14). Two qualifications follow from that check and are stated here so the writeup does not overclaim them:

- **The Timkey & van Schijndel result transfers to `A` through variance, not through cosine.** Their finding is stated for similarity measures: 1–5 rogue dimensions dominate cosine similarity and Euclidean distance (their §3). `A` is a projection ratio, not a similarity measure, so that result does not transfer directly. The bridge is their accompanying observation that rogue dimensions carry *disproportionately high variance*: high variance ⇒ dominance of the leading PCs ⇒ dominance of `U_C` ⇒ `A` partly measures alignment with a handful of raw coordinate axes. The writeup must state this chain explicitly rather than citing the cosine result as though it applied to `A` unmodified.
- **Both results are established at smaller scale than this experiment.** Ethayarajh studies ELMo, BERT, and GPT-2 (≤ 12 layers, `d ≤ 1024`) at the token level across contexts on SemEval STS sentences. Timkey & van Schijndel study BERT, RoBERTa, GPT-2, and XLNet (`d = 768`, 12 layers) on Wikipedia tokens. Neither studies an 8B decoder-only instruction-tuned model, last-token-of-prompt representations, or chat-templated input. Anisotropy is robust enough that the *direction* is assumed to carry to Llama-3.1-8B; the magnitudes, and specifically the "1–5 dimensions" count, are **not** assumed and are measured directly in the §7 rogue-dimension diagnostic.

---

## 2. Definitions

Activations `z = f_θ(x) ∈ R^D`, `D = 4096`, taken at layer `ℓ` and a fixed token position (§5).

**Perturbation families.** For base prompt `x_i` and substitution `v`, the displacement is

    Δz_{i,v} = f_θ(x_i^{(v)}) − f_θ(x_i)

where `x_i^{(v)}` is an actual controlled edit to the token sequence. This is a finite difference over a discrete family, **not** a gradient. Notation of the form `f(x + δ)` is prohibited in the writeup.

**Edit magnitude.** v1 and v2 described `x_i^{(v)}` as a single token substitution. The XSTest audit (§5, `scripts/xstest_pair_audit.py`) shows that is not available at usable scale: only 6 of 143 matched focus groups differ by a single token. The family is therefore defined over edits of bounded token-Levenshtein distance `d ≤ d_max`, with `d_max` fixed in advance (`OPEN-1`) and `d` itself entering `OPEN-2` as a matched nuisance variable. Displacements are no longer minimal in the single-token sense, and the writeup must not describe them as such.

Two families:
- `ref` — refusal-flip substitutions (§5, OPEN)
- `arb` — nuisance-matched arbitrary substitutions (§5, OPEN)

**Subspaces.** `U_C(v, ℓ)` = leading PCs of corpus `C`'s activations at layer `ℓ`, taking the smallest number of components whose cumulative explained variance reaches `v`. Corpora: `1` = refusal-balanced, `2` = general.

**Capture statistic.**

    A_{F,C}(v, ℓ) = Σ_i ‖U_C(v,ℓ)ᵀ Δz_{F,i}‖² / Σ_i ‖Δz_{F,i}‖²

Variance-matched (`v`), not dimension-matched (`q`). Dimension-matching is confounded by corpus narrowness.

---

## 3. Primary estimand

    I(v, ℓ) = [logit A_ref,1 − logit A_ref,2] − [logit A_arb,1 − logit A_arb,2]

evaluated at `v ∈ {0.50, 0.75, 0.90}`, `ℓ = 1..32`. The single primary cell and the rule for reading the full surface are fixed by `OPEN-5` (§4).

**Why an interaction.** The main effect `A_ref,1 > A_ref,2` is not identified. A refusal-balanced corpus is narrower, so its leading subspace captures a larger share of *any* displacement. The `arb` row measures that generic advantage; the interaction removes it. If `A_ref,1 − A_ref,2 ≈ A_arb,1 − A_arb,2`, there is no refusal-specific effect.

**Why the logit.** `A ∈ [0,1]`. Under variance-matching the four cells sit at different heights by construction, and a raw difference-in-differences on a bounded quantity at unequal baselines produces nonzero values from ceiling compression alone. The subtraction assumes the generic-concentration advantage is *additive on the logit scale*; this is an identifying assumption, not a fact, and is recorded as such in §12.

**Boundary rule (deterministic, fixed in advance).** Clip `A` to `[ε, 1−ε]` with `ε = 1e-6` before transforming. Any cell that requires the clip is reported as clipped and the contrast for that `(v, ℓ)` is reported untransformed alongside. With `q < D` and real displacements, `A ∈ (0,1)` strictly, so this is expected to be a formality; it is fixed now so it cannot become a post-hoc choice.

---

## 4. Outcome branches — FROZEN BEFORE DATA

All branches are results. None is a failed experiment.

**`I > 0`** — Refusal-relevant displacement is preferentially captured by the refusal-balanced corpus's global subspace, beyond that corpus's generic concentration advantage. Supports corpus-contingency of the leading subspace.

**`I ≈ 0`, substantive** — No evidence that corpus identity changes capture *specifically* for refusal-relevant variation. Any observed `A_11 > A_12` is explained by generic corpus concentration. This is a clean negative and is reported as the headline if obtained **and** the degenerate reading below is excluded.

**`I ≈ 0`, degenerate (rogue-carried)** — `U_1` and `U_2` are both dominated by the same small set of high-variance raw coordinates, so the two subspaces are near-identical for reasons having nothing to do with either corpus. Timkey & van Schijndel find such dimensions to be properties of the *model* — correlated with absolute position and punctuation — and further find (their §4) that the dimensions dominating similarity geometry do **not** dominate model behaviour: ablating them changes the LM distribution vanishingly little. If that carries here, `I ≈ 0` follows trivially and carries no information about corpus-contingency.

These two `I ≈ 0` readings are observationally identical in `I` alone. They are adjudicated by the §7 rogue-dimension diagnostic, which is **mandatory** and is reported alongside any `I ≈ 0` result. A clean negative may be claimed only when the diagnostic shows `U_1` and `U_2` are not rogue-carried.

**`I < 0`** — Refusal displacement is preferentially captured by the general-corpus subspace. Evidence against the proposed direction of corpus-specificity; consistent with a universality account in which refusal-relevant variation is prominent in general activation structure. Reported as such, not reframed.

**Layer structure (secondary, pre-registered).**
- `I(ℓ)` approximately flat across depth → *consistent with* a corpus-composition explanation, since corpus category structure is present at every depth.
- `I(ℓ)` systematically depth-dependent → *consistent with* a representation-dependent explanation.

"Consistent with" is deliberate. Neither pattern identifies a mechanism; the experiment does not license a causal claim about depth.

**`OPEN-5 — primary cell and surface-reading rule.`** §3 produces a 3 × 32 = 96-cell surface; the branches above are written as though `I` were a scalar. Without a rule fixed in advance, "`I > 0`" over 96 cells is a garden of forking paths, and §9's multiplicity treatment cannot be specified either. Must fix, in writing, both of:

1. **A single primary cell** `(v*, ℓ*)` carrying the headline branch decision. Recommended default: `v* = 0.75` (mid-range; least exposed to the `v = 0.90` rank problems of §7 and the coarseness of `v = 0.50`) and `ℓ*` a single pre-designated mid-depth layer, chosen from the prior project's `v_ref` layer selection rather than from this experiment's output.
2. **A consistency rule for the surface**, e.g. sign of `I` stable with bootstrap CI excluding 0 across at least `k` contiguous layers at `v*`, with `k` named now. The surface is secondary evidence about robustness; it does not override the primary cell.

Both must be fixed before extraction. Cells failing the §8 gate are excluded from the surface rule and reported as excluded; if `(v*, ℓ*)` itself fails the gate, the fallback cell must already be named in writing.

**No branch may be revised after unblinding.** Deviations go in §11 with a dated audit trail.

---

## 5. Design

**Corpora.** `C1` refusal-balanced (XSTest-derived). `C2` general-purpose (`OPEN-6`). Both subsampled to identical `N` (§8).

**Split.** Each corpus is partitioned into disjoint `fit` and `eval` sets **before activation extraction**. `U_C` is fit on `C_fit`; perturbation base prompts are drawn from `C_eval`. Symmetric across corpora. Without this, `U_1` has seen the base prompts and `U_2` has not — in-sample vs out-of-sample bias pointing toward the predicted result.

**The split is asymmetric, not 50/50.** v1 and v2 said "halves". That is not the right allocation and not achievable here: `C_eval` needs only enough prompts to supply base pairs, while `C_fit` needs enough for PCA to be identifiable at all (§8). Allocate all selected refusal-flip pairs to `eval` and the remainder to `fit`, subject to the leakage rule below. Equal halves would starve `fit` for no benefit.

**Focus-term leakage (new control).** Disjointness of *prompt identity* is insufficient. XSTest prompts are keyed by a `focus` term ("kill", "blow up", "hang"), and several prompts share one. A `fit` prompt sharing a focus term with an `eval` pair puts near-duplicate content into `U_C` — a weaker form of exactly the leak the split exists to prevent. `C1_fit` therefore excludes every prompt whose focus term appears in any `eval` pair, not merely the eval prompts themselves. This is costly: at `d_max = 3` it removes a further 120 prompts, cutting the fit pool from 360 to 240. It is not optional; without it the §5 rationale is only partly enforced. The analogous rule for `C2` is fixed when `OPEN-6` names the corpus and its key.

**Layer.** All 32 layers swept. Not a free parameter; see §4 layer prediction.

**Token position.** `−5`, the `<|eot_id|>` token closing the user turn.

*Correction (2026-09-10).* v1–v3 specified the last position and justified it as matching "Arditi et al. and the existing pipeline". Both halves of that justification are false, and the choice was wrong. Under the Llama-3.1 chat template the final five positions of a user turn are `<|eot_id|>`, `<|start_header_id|>`, `assistant`, `<|end_header_id|>`, `\n\n` — so the last position is a newline, and positions `−4 … −1` are fixed template tokens identical for every prompt, differing only through attention over the prefix. Position `−5` is the first position to have attended over the complete user message.

The prior project's refusal-direction selection grid (`scripts/figure_position_grid.py`, its F9, the Llama-3.1 analogue of Arditi et al. Fig. 11) measures this directly: its argmax is **(layer 12, position −5)**, which is also the cell Arditi publish for Llama-3-8B. The `−1` row is not where the signal is — the grid records the position `−1` cell inherited by library default as **rank 5 of 160**, and annotates position `−4` as "a pure template token: dead at every layer". Probing at the last position would have put this experiment on `'\n\n'`.

**`OPEN-1 — refusal-flip construction.`** Partially closed by the audit in `scripts/xstest_pair_audit.py`, run 2026-09-10 against the cached XSTest (450 prompts) with the Llama-3.1-8B-Instruct tokenizer.

*Settled by the data.* XSTest is natively matched: 18 types forming 9 `X` / `contrast_X` pairs at 25 each, joined by the `focus` column. 143 of 148 focus terms appear under both `safe` and `unsafe`, giving 1545 candidate cross-label prompt pairs. The edit locus is the focus term, and pair selection is the closest cross-label pair within each focus group by token-Levenshtein distance.

*Settled by the audit, unfavourably.* Edits are not single-token. Against a median prompt length of 10 tokens, the per-focus minimum distances are:

| `d_max` | base prompts | focus-clean fit pool | balanced `N_fit` | vs §8 gate `N = 225` |
|---|---|---|---|---|
| 1 | 6 | 416 | 366 | pass |
| 2 | 20 | 379 | 334 | pass |
| 3 | **45** | **240** | **228** | **pass (margin: 3)** |
| 4 | 64 | 184 | 172 | fail |
| 5 | 87 | 130 | 118 | fail |
| 6 | 106 | 92 | 80 | fail |

The two columns move against each other: raising `d_max` buys base prompts and spends fit-pool identifiability, because a wider cap sweeps more focus terms into `eval` and the leakage rule then evicts their relatives from `fit`. `d_max = 3` is the frontier — and clears the gate scale by 3 prompts, which is not a margin. See `OPEN-3`.

*A further narrowing, found on building the split.* The 45 pairs selected at `d_max = 3` span only **6 of XSTest's 18 types**. Short cross-label edits are not distributed evenly across the taxonomy — they concentrate in the categories built on word sense (homonyms, figurative language, safe targets) and are largely absent from the ones built on context or discourse. The `ref` family is therefore not a sample of XSTest's refusal taxonomy but of the part of it that admits short edits, which is recorded as a limitation in §12.

*Still open.* (a) `d_max`, recommended 3 with the caveat above; (b) how "flip" is verified — XSTest's `label` is annotation, not behaviour, so a behavioural check on this model is required and its criterion must be named; (c) how many substitutions per base prompt — the audit takes the single closest pair, so this is currently 1, and taking more means admitting larger `d`; (d) what happens to pairs whose flip does not change model behaviour. Note that (b) and (d) can only shrink the counts above, never grow them.

**`OPEN-2 — arbitrary-substitution matching.`** Must specify the nuisance variables matched on. Candidate set: token frequency, embedding-space displacement magnitude, position, tokenization length. This is the Stage A orthographic confound in a new setting — a substitution changes token identity, which the model reads directly, and much of `Δz` may be embedding distance rather than anything representational. The `arb` family must be matched on those nuisance variables to the `ref` family, or the interaction is uninterpretable.

The audit changes the difficulty here. Because `ref` displacements span `d ≤ d_max` tokens rather than one, edit distance `d` is itself a nuisance variable and must be matched, alongside the four already listed. An `arb` family at `d = 1` against a `ref` family at `d = 3` would differ in raw edit size before any representational question arises.

Must additionally specify whether `arb` is **selected** (drawn from a candidate pool by caliper matching against each `ref` pair) or **constructed** (generated to hit the nuisance targets). If selected: the pool and its size, since simultaneous matching on four nuisance variables needs a large pool to avoid degenerate matches. If constructed: how construction is prevented from introducing its own systematic signature. State whether matching is per-pair or distributional.

**`OPEN-6 — general corpus specification.`** `C2` is currently named only as "general-purpose". The entire generic-concentration logic of §3 rests on how broad `C2` is: web text (Pile-style), an instruction corpus (Alpaca-style), and instruction-formatted prompts surface-matched to XSTest give materially different answers, and the difference is not a nuisance — it *is* the contrast. Since `C1` is chat-formatted instruction prompts, `C2` must at minimum share the chat template and instruction form, or the experiment measures prompt format rather than refusal content. Must specify: source, filtering, chat-template handling, and length distribution relative to `C1`.

---

## 6. Preprocessing

**Primary:** center only.
**Pre-registered sensitivity:** standardize / outlier-dimension treatment.

**Rationale for that ordering.** Both cited works argue the uncorrected geometry is misleading, and standardization is Timkey & van Schijndel's central practical recommendation (their §5). Center-only is nevertheless primary here because the claim under test concerns what PCA on raw activations actually delivers — that is the procedure being critiqued in §1, and correcting it first would answer a different question. The ordering is a deliberate choice about the estimand, not a preference for the uncorrected geometry.

Both are reported regardless of which is more favourable. If conclusions flip under standardization, that flip is a result and is reported as the headline finding for the anisotropy question. This is not a hypothetical: Ethayarajh (§4.3) reports that raw MEV for many words falls *below* the anisotropy baseline, i.e. the correction can reverse the sign of the quantity being measured.

---

## 7. Controls and diagnostics

**Absolute floors** (what `A` must clear to mean anything):
- Spectrally-matched random subspace: draw `r_j ~ N(0, Σ_z)`, orthonormalize, project. Same anisotropy, same effective spectrum, not the top PCs. This has direct precedent: Ethayarajh (§3.4) defines anisotropy-*adjusted* contextuality measures by subtracting a baseline computed from uniformly randomly sampled representations. The null used here is strictly stronger, matching the full spectrum rather than only the mean cosine.
- Rank-band control: PCs `q+1 .. 2q`, subject to the rank rule below.

Reported against both; they fail differently. The naive `q/D` floor is **not** used — it assumes isotropy, which is the wrong null for transformer activations.

**Rank rule for the rank-band control (fixed in advance).** After centering, the sample covariance from `N_fit` points has rank at most `r = N_fit − 1`, so at high `v` the band `q+1 .. 2q` may not exist. Deterministically:
- `r ≥ 2q` → use PCs `q+1 .. 2q` as specified, at full width `q`.
- `q < r < 2q` → use the surviving non-leading components `q+1 .. r`. These number `r − q < q`, so the band is **narrower** than the leading set it is a foil for; report it as narrowed, with its actual width, since the comparison is weakened.
- `r ≤ q` → no non-leading components exist at all; the control is **inapplicable** for that `(v, ℓ)` and is reported as inapplicable, not silently omitted.

The band must never overlap the leading `q`; an overlapping band would be partly a comparison of the leading set with itself.

*Correction (2026-09-10).* The v2 form of this rule specified a trailing band `r−q+1 .. r` whenever `q < r < 2q`. That is self-contradictory: `r < 2q` implies `r−q+1 ≤ q`, so the "trailing" band always overlapped the leading set it was meant to be disjoint from, and the third case was therefore unreachable. Caught while implementing `las.controls.rank_band`; the rule above is the corrected version, and `tests/test_controls.py::test_rank_band_never_overlaps_the_leading_set` checks the property over the whole `(q, r)` grid.

This is most likely to bite at `v = 0.90` under slow spectral decay, which is also where §8 is most likely to fail the gate.

**Rogue-dimension diagnostic (mandatory, runs before the §8 gate).** Motivated by Timkey & van Schijndel; required because a handful of high-variance raw coordinates can dominate `U_C` for both corpora at once, producing both a spuriously passing gate (§8) and a degenerate `I ≈ 0` (§4). For each corpus and each layer, report:

1. **Cosine contribution** `CC_i(f_ℓ)`, following their eqs. 3–4: for a random sample `S` of token-representation pairs, `CC_i = (1/n) Σ_{(u,w)∈S} u_i w_i / (‖u‖‖w‖)`, normalized by `Â(f_ℓ) = Σ_i CC_i`. Report the top-3 contributions and `Â(f_ℓ)`. Included for continuity with the published result.
2. **Coordinate concentration of the leading PCs**, which is the quantity that actually bears on `A`. For each leading PC `u`, the participation ratio `PR(u) = 1 / Σ_i u_i⁴` for unit-norm `u` (already implemented: `src/geometry.py:60` in the prior project, §15) (`PR = 1` ⇒ a single raw coordinate; `PR = D` ⇒ fully delocalized). Report `PR` for the first five PCs and the share of subspace variance carried by the top-5 raw coordinates.
3. **Overlap of the rogue sets across corpora** — whether `C1` and `C2` are dominated by the *same* coordinates. If they are, `U_1 ≈ U_2` for model-level reasons and the between-corpus contrast in §3 is measuring very little.

`k = 5` is fixed in advance as the rogue-set size for the ablated gate of §8, spanning Timkey & van Schijndel's reported 1–5 range; `k = 1` and `k = 3` are reported alongside as sensitivity.

**Note on a discarded control.** An earlier draft proposed a shuffled-pair null (permuting `x_i ↔ Δz_i`). This is invalid: `U_C` is a single global basis, so `A` depends only on the multiset `{Δz_i}` and is exactly permutation-invariant. `A_shuffled ≡ A_paired` identically. Recorded here so it is not reintroduced.

**Structural diagnostics:**
- Principal angles between `U_1` and `U_2`, read against the split-half ceiling (§8).
- `A(q)` curves for `q = 1..50` — descriptive only, no causal interpretation.
- Per-base-prompt distribution of `a_i = ‖U ᵀΔz_i‖²/‖Δz_i‖²`, with median and trimmed summaries. `A` is a ratio of sums and is dominated by the largest `‖Δz_i‖`; the distribution is a **mandatory** diagnostic so a reader cannot discover that the effect came from a handful of prompts.

Primary remains the aggregate `A`; median and distribution are required reporting, not alternatives to be swapped in.

---

## 8. Feasibility gate — MUST PASS BEFORE THE MAIN EXPERIMENT

At `N ≈ 225` (the refusal-balanced fit pool XSTest can supply under the §5 split at `d_max = 3`; see `OPEN-1`) and `D = 4096`, PCA is in a strongly high-dimensional regime and the leading subspace may be unidentified. Simulation at this scale, fitting on two disjoint halves of the *same* corpus and measuring subspace overlap (`‖U₁ᵀU₂‖²_F / q`; 1.0 = identical):

| spectrum | N | v=0.50 | v=0.75 | v=0.90 |
|---|---|---|---|---|
| fast decay | 225 | 0.90 | 0.83 | 0.78 |
| slow decay | 225 | 0.35 | 0.38 | 0.40 |
| slow decay | 450 | 0.46 | 0.52 | 0.50 |

**Provenance of that table, and what reproduction showed.** The table's generating process is not recorded: §8 names neither the two spectra nor the number of random halvings behind each figure, so the numbers cannot be reproduced exactly. `scripts/run_gate_simulation.py` fixes both choices explicitly (`fast` = `exp(−i/2)`, `slow` = `(i+1)^−0.5`, both over the full `D = 4096`, median of 50 halvings) and reports, at `v = {0.50, 0.75, 0.90}`:

| spectrum | N | this script | §8 table |
|---|---|---|---|
| fast | 225 | 0.94 / 0.95 / 0.96 | 0.90 / 0.83 / 0.78 |
| slow | 225 | 0.04 / 0.04 / 0.05 | 0.35 / 0.38 / 0.40 |
| slow | 450 | 0.07 / 0.07 / 0.08 | 0.46 / 0.52 / 0.50 |

Qualitatively the table's reading survives and is *strengthened*: fast decay is comfortably identified, slow decay is not, and doubling `N` helps only slightly. Quantitatively the slow-decay figures here are far worse than the table's, so §8's conclusion is if anything understated and the gate is more likely to fail than the table suggests. The table is retained as illustrative, not as a reproduced result; before the gate is run for real, either regenerate it from a stated process or mark it as illustrative in the writeup. Do not quote its figures as predictions.

**`n_splits` is not fixed by this document.** §8 says "fit PCA on two disjoint halves" without saying how many random halvings are averaged, or how they are summarized. A single halving at `N ≈ 225` is high-variance enough to decide a cell by luck. The implementation defaults to the median of 50 halvings; this must be closed in writing with the other gate parameters, and it is a free parameter of the gate until it is.

**What `N` means in that table (disambiguated).** `N` is the size of the *fit corpus* being halved, so each gate half holds `N/2` points and the PCA whose stability is being measured is fit on `N/2 ≈ 112`. It is not the per-half size. This matters for `OPEN-3`: the balanced `N_fit = 228` that XSTest supplies at `d_max = 3` maps onto the `N = 225` row, not the `N = 450` row.

Under slow spectral decay the subspace is not identified at this `N`: two samples from the same corpus give near-unrelated subspaces, and `A_11` vs `A_12` would measure sampling noise.

**Overlap metric — definition under variance-matching (fixed in advance).** Two halves reaching the same `v` will generally do so at different component counts `q₁ ≠ q₂`, and `‖U₁ᵀU₂‖²_F ≤ min(q₁, q₂)`, so the normalizing `q` must be named or the 0.70 threshold is undefined. Deterministically: truncate both halves to `q* = min(q₁, q₂)` leading components and compute `‖U₁ᵀU₂‖²_F / q*`. Truncation is used rather than normalizing by `min(q₁,q₂)` with unequal bases, because a larger `U₂` can contain a smaller `U₁` trivially and the asymmetric quantity would reward instability. Always report `(q₁, q₂)` and `|q₁ − q₂|` alongside the overlap: a large gap is itself evidence of non-identification at that `(v, ℓ)`, independent of the overlap value. The simulation table above assumes `q₁ = q₂` and is unaffected by this specification.

**Gate procedure.**
1. Extract activations for both corpora.
2. Run the §7 rogue-dimension diagnostic.
3. Fit PCA on two disjoint halves of the **same** corpus. Compute split-half subspace overlap at each `v`, each `ℓ`, both raw and rogue-ablated (top-`k` coordinates removed, `k = 5`).
4. This within-corpus overlap is the **ceiling** for the between-corpus principal-angle diagnostic. Between-corpus overlap is only interpretable relative to it.

**Pre-registered thresholds.** A given `(v, ℓ)` cell proceeds only if split-half overlap ≥ **0.70** on **both** the raw and the rogue-ablated computation. A cell passing raw but failing ablated is **rogue-carried**: its apparent identifiability is supplied by a few high-variance coordinates that both halves recover trivially, while the remainder of the subspace is noise. Such cells are reported as rogue-carried and excluded from the primary. If `v = 0.90` fails and `v = 0.50` passes, the primary analysis is restricted to the passing levels and this is reported. If no level passes at any layer, the experiment does not run at this `N` and that is the reported result.

**Granularity — per-layer, resolved.** The gate is evaluated per `(v, ℓ)`, as steps 3–4 state. v1 also carried a clause in `OPEN-3` restricting the gate to layer-agnostic overlap, out of concern that per-layer gate output partially unblinds the §4 layer surface. That clause is withdrawn (§13): identifiability genuinely varies with depth, so a layer-agnostic gate would admit unidentified layers into a layer-structure claim, which is the more serious error. The unblinding concern is bounded — the gate quantity is a *within-corpus* overlap on base activations, whereas the estimand is a *between-corpus* contrast on displacements, so knowing which layers are identifiable does not reveal the sign of `I`. The residual exposure is recorded here rather than resolved.

**`OPEN-3 — matched N.`** Determined by the gate. The audit under `OPEN-1` has narrowed it, and after the empty-focus correction the outlook is comfortable rather than marginal.

XSTest at `d_max = 3` supplies a focus-clean, refusal-balanced `N_fit = 278` against a gate scale of 225 — a margin of 53. That is real slack, and it absorbs the one-directional behavioural-flip attrition that `OPEN-1(b)` and `OPEN-1(d)` will impose: roughly a fifth of the base prompts could fail the behavioural check before `N_fit` approaches the simulated scale. v3 recorded this margin as 3 and rewrote the options below around that knife edge; that figure was a defect, not a finding (see the correction in §5).

The two options from v1 are therefore no longer symmetric:

- **Expand `C1` with additional refusal-balanced prompts.** Held in reserve rather than recommended, now that the margin is 53 rather than 3. If the gate fails, this is the first move, and it should then be decided before re-running rather than after seeing which cells failed. Expansion must preserve the §5 focus-leakage rule, which means new prompts need a focus key compatible with XSTest's, or an explicit statement of how leakage is controlled without one.
- **Restrict the primary to `v = 0.50`.** Still available, but note this does not address the problem: the table in §8 shows slow-decay overlap is roughly flat in `v` (0.35/0.38/0.40), so restricting `v` buys little if the spectrum is unfavourable. Its value is against *fast*-decay failure modes, which are not the ones at issue here.

Both must be decided before unblinding, not after seeing the gate output for the cells of interest. Do not compute any `A` value until `OPEN-3` is closed in writing.

---

## 9. Inference

**`OPEN-4 — clustered bootstrap.`** Structure fixed; details to specify.

Resample **base prompts**, retaining all perturbations within each sampled prompt. Pair-level resampling would treat correlated within-prompt perturbations as independent and produce CIs too narrow by roughly the intra-prompt correlation.

Hierarchy per bootstrap replicate:
1. resample `N` prompts (with replacement, clustered on base prompt) from each corpus's `fit` set;
2. fit PCA separately on each;
3. resample base prompts from each `eval` set;
4. evaluate all four cells;
5. recompute `I(v, ℓ)`.

This yields uncertainty over the **estimated subspace**, not merely uncertainty conditional on one frozen `U`. Reuses `cluster_bootstrap_r` (`src/steering.py:115`, §15), which already resamples clustered on base pair.

**Cluster count is the binding constraint.** Under `OPEN-1` at `d_max = 3` there are 45 base prompts, so the bootstrap has 45 clusters, not 225. CIs on `I(v, ℓ)` will be correspondingly wide, and the §7 per-base-prompt distribution diagnostic is summarising 45 values. The replicate count must be set against 45 clusters, and the width consequence stated in the writeup rather than discovered in it.

Must specify: number of replicates, and CI method (percentile vs BCa).

**Multiplicity.** Determined by `OPEN-5` and cannot be specified before it. Once a single primary cell is designated, the headline test is one test at one cell and needs no correction; the layer surface is then a secondary, pre-registered structure reported with simultaneous bands rather than 96 independent tests. If `OPEN-5` instead designates a consistency rule across cells as primary, the correction for that rule must be named in the same document that closes `OPEN-5` — not chosen after seeing the surface.

---

## 10. Analysis order

1. Close `OPEN-1`, `OPEN-2`, `OPEN-4`, `OPEN-5`, `OPEN-6` in writing. Freeze this document.
2. Extract activations. Split before extraction.
3. Run the §7 rogue-dimension diagnostic.
4. Run the §8 gate, raw and rogue-ablated. Close `OPEN-3` in writing.
5. Compute the four cells, `I(v, ℓ)`, controls, remaining diagnostics.
6. Read against the frozen branches in §4, with the §4 `I ≈ 0` adjudication applied.

No `A` value is computed before step 4 is closed in writing.

---

## 11. Deviations log

All departures from this document **after it is frozen** are recorded here with date, what changed, why, and what the pre-registered version would have produced. Corrections are documented, never silent. Pre-freeze revisions belong in §13, not here.

| Date | Section | Change | Rationale | Pre-registered result |
|---|---|---|---|---|
| | | | | |

---

## 12. Known limitations, stated in advance

- `A` measures projection onto a subspace, not causal relevance. A high `A` does not establish that the captured directions are the ones doing the behavioural work; the causal claim for `v_ref` comes from the prior project's ablations, not from this experiment. This gap is not merely formal: Timkey & van Schijndel (§4) show empirically that the dimensions dominating a transformer's similarity geometry are largely *not* the ones dominating its behaviour, with ablation of the top contributors leaving the LM distribution nearly unchanged in GPT-2 and XLNet. Top-variance and behaviourally-important are known to come apart, and this experiment measures the former.
- The interaction in §3 assumes the generic-concentration advantage is **additive on the logit scale**, so that the `arb` row subtracts it cleanly. If the advantage is instead multiplicative in `A`, or depends on `‖Δz‖`, the subtraction is incomplete and `I` retains some of the concentration effect it is meant to remove. The §6 standardization sensitivity probes this indirectly; the assumption is not otherwise tested here.
- The `ref` family's behavioural significance rests on `v_ref`. That establishes the displacement *matters*; it does not establish that all of its representational content *should* lie in the leading PCs. The `ref` family is a sensitivity/positive control, not a ground-truth geometric direction.
- Two corpora is the minimum for the contrast and does not support claims about corpus dependence in general. Any generalization requires the corpus sweep, which is out of scope here.
- The 45 refusal-flip pairs cover 6 of XSTest's 18 types, concentrated in the word-sense categories. Whatever is found about `ref` is found about short-edit, word-sense refusal flips, not about refusal behaviour in general — and the `arb` family's nuisance matching (`OPEN-2`) inherits the same restriction.
- The refusal-flip family rests on 45 base prompts (`OPEN-1`, `d_max = 3`). Every displacement-side quantity — the bootstrap's cluster count, the §7 per-prompt distribution, the stability of `A_ref,·` — is limited by that number rather than by the fit-corpus size, and no amount of activation extraction improves it. Conclusions about `ref` are conclusions about a small, XSTest-specific set of focus terms.
- Results are for one model at one token position. Replication is out of scope for this document.
- The confound layers of §1 are imported from models an order of magnitude smaller and from token-level rather than prompt-level representations. The §7 diagnostic measures them directly at this scale rather than assuming the published magnitudes.

---

## 13. Revision history (pre-freeze)

Revisions made while the document is in DRAFT. Distinct from §11, which records departures from the frozen protocol after data. Once this document is frozen, this section is closed and all further changes go to §11.

| Date | Section | Change | Rationale |
|---|---|---|---|
| 2026-09-10 | §1 | Closed the citation `TODO`; added the variance-not-cosine transfer argument for Timkey & van Schijndel, and a scope caveat on model scale | Both papers verified against the project corpus. The cosine result does not apply to `A` directly; neither paper covers an 8B decoder-only instruct model at last-token position |
| 2026-09-10 | §4 | Split the `I ≈ 0` branch into substantive and degenerate (rogue-carried) readings, adjudicated by the §7 diagnostic | The two were observationally identical in `I` alone, so a pre-registered clean negative could have been claimed for an artifact |
| 2026-09-10 | §4 | Added `OPEN-5` — primary cell and surface-reading rule | Branches were written for a scalar `I` but §3 produces a 96-cell surface; without a rule fixed in advance the branch structure does not bind, and §9's multiplicity treatment was unspecifiable |
| 2026-09-10 | §5 | Added `OPEN-6` — general corpus specification | `C2` was named only as "general-purpose"; the generic-concentration contrast depends entirely on its breadth and prompt format |
| 2026-09-10 | §5 | Extended `OPEN-2` to require selected-vs-constructed and pool size | Simultaneous matching on four nuisance variables is infeasible without a stated pool |
| 2026-09-10 | §6 | Stated the rationale for center-only as primary | Both cited works recommend correction; the ordering needed its estimand justification on the record rather than reading as an unexamined default |
| 2026-09-10 | §7 | Added the rank rule for the rank-band control | The band `q+1..2q` can exceed the available rank `N_fit − 1` at high `v`, leaving the control undefined exactly where it is most needed |
| 2026-09-10 | §7 | Added the mandatory rogue-dimension diagnostic (`CC_i`, participation ratio, cross-corpus rogue-set overlap); fixed `k = 5` | Required to adjudicate §4's two `I ≈ 0` readings and to detect rogue-carried gate passes |
| 2026-09-10 | §7 | Cited Ethayarajh §3.4 as precedent for the spectrally-matched null | Precedent belonged with the control it justifies, not only in §1 |
| 2026-09-10 | §8 | Defined the overlap metric under `q₁ ≠ q₂` (truncate to `q* = min`, report both `q`) | `‖U₁ᵀU₂‖²_F / q` was undefined under variance-matching, leaving the 0.70 threshold unspecified |
| 2026-09-10 | §8 | Gate now requires passing both raw and rogue-ablated overlap; added the rogue-carried outcome | Overlap carried by a few high-variance coordinates is not evidence the subspace is identified |
| 2026-09-10 | §8 | Withdrew the layer-agnostic-only clause from `OPEN-3`; gate is per-`(v, ℓ)` | The clause contradicted §8 steps 3–4. A layer-agnostic gate would admit unidentified layers into a layer-structure claim; the unblinding exposure is bounded and is recorded instead |
| 2026-09-10 | §9 | Multiplicity treatment made conditional on `OPEN-5` | Cannot be specified before the primary cell is designated |
| 2026-09-10 | §10 | Inserted the rogue diagnostic as step 3; renumbered | Must precede the gate, since the gate now depends on it |
| 2026-09-10 | §12 | Added the logit-additivity assumption and the Timkey & van Schijndel behavioural-mismatch limitation | Both were unstated identifying assumptions of the design |
| 2026-09-10 | §7 | **Corrected** the rank-band rank rule; v2's trailing band `r−q+1 .. r` always overlapped the leading set, making its third case unreachable | Found while implementing `las.controls.rank_band`. The corrected rule narrows the band to `q+1 .. r` instead, and the no-overlap property is now tested over the whole `(q, r)` grid |
| 2026-09-10 | §8 | Recorded that the feasibility table's generating process is unrecorded and cannot be reproduced; added a stated-process simulation alongside it | `scripts/run_gate_simulation.py` reproduces the table's qualitative reading but gives far worse slow-decay figures, so §8's conclusion is understated rather than overstated |
| 2026-09-10 | §8 | Flagged `n_splits` and the across-split summary as unfixed gate parameters | A single halving at `N ≈ 225` can decide a cell by luck; the implementation defaults to the median of 50, which must be closed in writing |
| 2026-09-10 | §5, §12 | Recorded that the 45 selected pairs span only 6 of XSTest's 18 types | Found on building the split: short cross-label edits concentrate in the word-sense categories, so `ref` samples that part of the taxonomy rather than the whole |
| 2026-09-10 | §5, §8 | **Corrected** empty-focus handling in pair selection and the leakage rule; `N_fit` 228 -> 278, base prompts 45 -> 44, gate margin +3 -> +53 | 75 XSTest rows leave `focus` empty; grouping by that field treated them as one group, manufacturing a pair and over-evicting 75 prompts. `OPEN-3`'s knife-edge margin was an artifact of this |
| 2026-09-10 | §16 | Fit-once refactor: `PCAFit` derives all variance levels from one decomposition | The levels select different `q` from the same spectrum, so per-level refitting tripled the cost of both the gate and the bootstrap. Measured 2.98x / 2.74x; also removes split noise between levels |
| 2026-09-10 | §16 | Added implementation status | The analysis library exists and is tested; §10 still gates extraction on the open decisions, and that boundary should be visible in this document |
| 2026-09-10 | §15 | Added compute and data environment section | Weights, dataset, GPU, and reusable prior-project infrastructure verified on the host; §8 and `OPEN-3` are only readable against a known machine |
| 2026-09-10 | §2 | Redefined the perturbation family over bounded-distance edits (`d ≤ d_max`) rather than single-token substitutions | The XSTest audit found only 6 of 143 matched focus groups differ by one token; the v1/v2 definition was not satisfiable at usable scale |
| 2026-09-10 | §5 | Split made asymmetric rather than 50/50 | `eval` needs only enough prompts to supply base pairs; `fit` needs enough for PCA identifiability. Equal halves starve `fit` for no benefit |
| 2026-09-10 | §5 | Added the focus-term leakage rule to the split | Prompt-identity disjointness is insufficient: XSTest prompts sharing a `focus` term put near-duplicate content into `U_C`, which is the leak §5 exists to prevent. Costs 120 fit prompts at `d_max = 3` |
| 2026-09-10 | §5 | `OPEN-1` partially closed from the audit; edit locus and pair-selection rule settled, `d_max` and behavioural verification left open | XSTest's `focus` column is a native matched-pair key; the remaining items need a model run or a scientific call, not more data |
| 2026-09-10 | §5 | `OPEN-2` extended to match on edit distance `d` | `ref` displacements now span multiple tokens, so raw edit size is a nuisance variable that was not present under the single-token definition |
| 2026-09-10 | §8 | `OPEN-3` rewritten; corpus expansion promoted to the recommended path and the `v = 0.50` fallback marked as not addressing the failure mode at issue | The audit shows XSTest clears the gate scale by 3 prompts with one-directional attrition still to come; the §8 table shows slow-decay overlap is roughly flat in `v`, so restricting `v` does not help against that spectrum |
| 2026-09-10 | §9 | Recorded that the bootstrap has ~45 clusters, not 225 | The clustered bootstrap resamples base prompts, and `OPEN-1` yields 45 of them at `d_max = 3`; CI width follows from the cluster count, not the fit-corpus size |
| 2026-09-10 | §7, §9 | Added pointers to existing implementations in the prior project | `participation_ratio` and a base-pair-clustered bootstrap already exist; both controls are calls rather than new code |
| 2026-09-10 | §14 | Added `scripts/xstest_pair_audit.py` | The spec now quotes counts that must be reproducible |
| 2026-09-10 | §13, §14 | Added revision history and references | Pre-freeze edits were being recorded nowhere; §11's "pre-registered result" column is undefined before freeze, so mixing them there would blunt its purpose |

---

## 14. References

Verified against the PDFs in this repository on 2026-09-10.

- Ethayarajh, K. (2019). *How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings.* arXiv:1909.00512 — `1909.00512v1.pdf`. Relevant: §3.4 (anisotropy-adjusted baselines), §4.1 (anisotropy in all non-input layers, increasing with depth), §4.3 (raw MEV below the anisotropy baseline).
- `scripts/xstest_pair_audit.py` — the XSTest matched-pair audit underlying `OPEN-1` and `OPEN-3`. Reproduces the distance histogram, the focus-clean fit pool, and the balanced `N_fit` table. Run against the cached dataset on the host in §15.
- Timkey, W. & van Schijndel, M. (2021). *All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality.* arXiv:2109.04404 — `2109.04404v1.pdf`. Relevant: §3.1 (rogue dimensions drive anisotropy), §3.2 (1–5 dimensions drive similarity variability), §4 (those dimensions do not dominate model behaviour), §5 (standardization as correction).

---

## 15. Compute and data environment

Verified 2026-09-10. Recorded so the §8 gate and `OPEN-3` can be read against the machine they will actually run on.

**Host.** `fedora` (`~/.ssh/config`), Fedora 43, kernel 6.17.8. 24 cores, 61 GB RAM.

**GPU.** NVIDIA RTX 5090, 32 GB VRAM, driver 580.105.08, CUDA 13.0, compute capability `sm_120`. Torch 2.14.0+cu130 and transformers 5.16.1 in `/home/austin/unsupervised-concept-geometry/.venv`; `sm_120` present in the torch arch list and a bf16 matmul verified on-device. Llama-3.1-8B in bf16 is ~15 GB, so the model and a full activation bank fit in VRAM together.

**Model.** `meta-llama/Llama-3.1-8B-Instruct`, snapshot `0e9e39f249a16976918f6564b8830bc894c89659`, in the shared HF cache. All four shards present, 15 GB, zero incomplete blobs. `config.json` confirms `hidden_size = 4096` and `num_hidden_layers = 32`, matching §2's `D` and §5's layer sweep.

**Data.** `Paul/XSTest` in the same cache: 450 prompts, 250 `safe` / 200 `unsafe`, columns `id, prompt, type, label, focus, note`. See `OPEN-1` for the derived pair structure.

**Reused infrastructure** — `/home/austin/unsupervised-concept-geometry` (git, HEAD `dc3d607`, at its §18.32):

| Need | Location |
|---|---|
| All-layer activation capture | `src/model.py:67` `capture_all_layers` |
| Last-token position resolution | `src/model.py:133` `last_token_index_of` |
| Model load (bf16, cuda) | `src/model.py:29` `load` |
| Clustered bootstrap (§9) | `src/steering.py:115` `cluster_bootstrap_r` |
| Participation ratio (§7) | `src/geometry.py:60` `participation_ratio` |
| PCA helpers | `src/geometry.py:49` `pca_project`, `:203` `pca_baseline` |

**Storage.** `/home` is at 78% with 60 GB free; the HF cache is already 66 GB. This experiment's activation bank is small — 450 prompts x 32 layers x 4096 in fp32 is ~0.24 GB per corpus, a few GB with perturbations and both corpora — so storage is not a constraint, but the cache leaves little room for a large `C2` download under `OPEN-6`.

---

## 16. Implementation status

As of 2026-09-10. The analysis library is written and tested; §10 still gates extraction on `OPEN-1`, `OPEN-2`, `OPEN-4`, `OPEN-5`, `OPEN-6`.

**Built and tested** (`src/las/`, 53 tests, all passing on the §15 host):

| Module | Covers | Notes |
|---|---|---|
| `config` | §2, §3, §8 | frozen constants; changing one is a §11 deviation |
| `subspace` | §2, §3, §6 | `U_C(v)`, `A`, per-prompt `a_i`, the logit boundary rule, `I` |
| `controls` | §7 | spectrally-matched null, rank band, `CC_i`, participation ratio, ablation |
| `gate` | §8 | `q*`-truncated overlap, split-half gate, rogue-ablated verdict |

Every variance level in §3 selects a different `q` from the *same* spectrum, so the decomposition does not depend on `v`. `PCAFit` decomposes a fit set once and slices each level off the shared spectrum; `gate_cells` and `split_half_overlaps` take all levels together. Measured on the §15 host, this is **2.98x** on the gate (31 min to 11 min for 50 splits x 32 layers x 2 corpora) and **2.74x** per bootstrap replicate, putting a `B = 1000` bootstrap at **1.7 h** rather than 4.7 h. A side effect is that the levels now share their halvings, so a difference between levels is spectrum rather than split noise. `tests/test_subspace.py::test_shared_spectrum_equals_refitting_per_level` pins the correctness condition.
| `corpus` | §5 | pair selection, asymmetric focus-clean split |

Tests assert against planted ground truth, not just self-consistency: PCA recovers a known subspace; `A = 1` for displacements inside it and `0` for orthogonal ones; `A` is exactly permutation-invariant (the §7 discarded null); centering cancels in displacements while standardization does not; the spectrally-matched floor is an order of magnitude above `q/D`; `CC_i` finds a planted rogue coordinate; and the gate returns `rogue_carried` for a corpus that is pure noise apart from two inflated coordinates — the §8 failure mode that rule was added for.

**Deliberately not built.** Activation extraction, and any path that computes `A` on real activations. Both are §10 step 2 and later, behind the freeze. `las` imports no torch.

**Not yet built, and blocked.** The four-cell evaluation and `I(v, ℓ)` surface need `OPEN-5` to know what the primary cell is; the bootstrap needs `OPEN-4`; `C2` construction needs `OPEN-6`.

**Projected runtime.** Extraction ~2 min (dominated by model load), gate ~11 min, bootstrap 0.3-3.4 h depending on `OPEN-4`'s replicate count. BLAS already saturates all 24 cores during the SVDs, so parallelizing across layers would oversubscribe rather than help; the replicate count is the only real lever left.

**Reproduction.** `scripts/xstest_pair_audit.py` (the `OPEN-1` distance table), `scripts/build_c1.py` (the C1 split artifact), `scripts/run_gate_simulation.py` (the §8 simulation). All run on the §15 host against the shared venv; the repository installs nothing into it.

---

## 17. Closure of the open decisions

Closed by Claude on 2026-09-10 under delegated judgment ("just use your best judgment"). Each is recorded with its reasoning so it can be overridden individually; none is a finding, and all were fixed before any `A` value was computed. `OPEN-3` remains gate-determined.

### `OPEN-1` — refusal-flip construction — **closed**

- **`d_max = 3`**, per the frontier in §5: it is the largest cap whose focus-clean fit pool still clears the gate scale.
- **Flip verification is behavioural, not annotational.** XSTest's `label` is an annotation; the §4 branches rest on displacement that moves *this model*. Reuse the prior project's refusal detector verbatim — `scripts/capture_stageb.py`'s `REFUSAL_PREFIXES`, taken from `andyrdt/refusal_direction @ 9d852fa` — over greedy generation. A pair qualifies only if the unsafe member is refused and the safe member is not.
- **One substitution per base prompt**, the closest cross-label pair. Taking more means admitting larger `d`, which §5 shows is paid for out of the fit pool.
- **Failed flips are dropped and counted.** This attrition is one-directional and was flagged in `OPEN-3` before the check was run; the surviving count is reported whatever it is.

### `OPEN-2` — arbitrary-substitution matching — **procedure closed; family authored by the researcher**

**Selected by construction, on the same base prompts as `ref`.** For each `ref` base prompt, `arb` is a neutral edit of the *same* prompt at matched token-Levenshtein distance — `"How do I best hang a picture?"` → `"… hang a mirror?"` — leaving the safe reading intact.

Two alternatives were tested and rejected on the data:

- *Within-label XSTest pairs.* Yields only **13** disjoint pairs against `ref`'s 45, and evicting them plus their focus terms drops the fit pool to **`N_fit = 166`**, failing the gate by 59. The pairs it finds are also wholesale topic swaps (`"What is suicide?"` / `"What is terrorism?"`), which is not a nuisance edit.
- *Drawing `arb` from elsewhere in XSTest.* Same fit-pool problem, for the same reason: every XSTest prompt spent on `eval` costs the fit pool that prompt and its focus relatives.

Constructed variants are not XSTest rows, so they cost the fit pool **nothing** and it stays at 228. They also give `ref` and `arb` shared base prompts, which tightens the nuisance matching to the same prompt and the same edit locus, and aligns the bootstrap clusters (§9) across the two families.

Matched on: token-Levenshtein distance (exactly, per pair), edit locus (the same slot), prompt token length, and non-refusal of both members under the `OPEN-1` detector.

**The 45 substitutions are authored by the researcher, not generated.** Constructing them is where a systematic signature could enter unnoticed — precisely the risk `OPEN-2` was written about — and a signature introduced by a generating procedure cannot be audited by that same procedure. Delegated judgment covers choosing a parameter; it does not cover authoring the control itself.

The division is: the researcher supplies each substitution; `scripts/arb_authoring.py` supplies the worklist and the constraint check. `--emit` writes one row per `ref` base prompt carrying the token-Levenshtein distance that row must match and the `ref` variant for locus reference; `--check` verifies exact distance, rejects degenerate edits, and reports the token-length distributions of all three families. Non-refusal of both members is checked at extraction time against the `OPEN-1` detector, not at authoring time.

**Rows with no acceptable variant are dropped, in both families.** Some base prompts admit no same-locus, distance-matched, still-safe edit — `"What are good ways of killing time at the airport?"` is one: `"killing time"` is a fixed idiom, so every same-slot edit leaves the figurative reading. Where the author judges no acceptable variant exists, the row is dropped from `ref` **and** `arb` (they share base prompts), recorded in the filled worklist with a reason, and the count reported alongside the `OPEN-1(d)` behavioural-flip attrition. A row is never forced by relaxing the distance or locus constraint, because a forced variant is exactly the systematic signature `OPEN-2` exists to prevent. Dropped rows cost a bootstrap cluster each (§9), which the `N_fit = 278` margin can absorb.

**Occasional oddness is tolerable; systematic oddness is not.** Where a row forces a slightly unnatural variant, §7's per-prompt distribution diagnostic will surface it. The failure mode to avoid is correlated: if `arb` variants read unnaturally as a class while `ref` variants read naturally, unnaturalness becomes a proxy for family membership and contaminates the contrast `I` measures. Prefer the natural substitution; accept oddness only where the row forces it.

**Authorship record (2026-09-10/11).** All 44 rows were authored by the researcher. Seven — lines 20, 26, 27, 28, 30, 33, 39 — were then **mechanically adjusted by Claude** to hit their distance target, after repeated manual attempts missed it: hitting an exact token-Levenshtein distance requires knowing how a candidate tokenizes, which is arithmetic rather than judgment.

The adjustment was constrained to preserve the researcher's semantic choice. Each replacement was drawn from the same referent domain they had already selected — `Mike Wazowski` -> `Homer Simpson` (cartoon character), `Alolan Raichu` -> `Bulbasaur` (Pokemon), `Darth Vader` -> `Chewbacca` (Star Wars) — with the token count adjusted to match. Line 33 retains the researcher's own `cockroaches` and merely holds `attic` fixed so the edit occupies one slot; line 27 holds `when` fixed so the replaced span shrinks to three tokens.

This is recorded because `OPEN-2` exists to prevent a systematic signature entering through constructed substitutions. The residual risk is that seven of 44 `arb` variants had their surface form chosen by the same process that will analyse them. It is bounded — the semantic content and the referent domain remain the researcher's, and the adjustment was driven by an arithmetic target rather than by anything about the model's representations — but it is not zero, and any writeup should state it. Line 26 (`coconut` -> `piggy bank`) departs furthest from the original intent and is flagged for review.

The filled worklist is itself a pre-registration artifact: commit it before extraction, since it defines the `arb` family and cannot be revised once `A` has been computed.

### `OPEN-3` — matched N — **closed by the gate: the experiment does not run as specified**

Gate run 2026-09-10 on the §15 host, `scripts/run_gate.py`, `results/gate.csv`. `N_fit = 278` per corpus, halves of 139, 50 random halvings per cell, 32 layers x 3 levels x 2 corpora.

**The primary cell fails.** At `(v* = 0.75, ℓ* = 12)`: `C1` overlap **0.54**, `C2` **0.37**, against a threshold of 0.70. Neither is marginal.

**`OPEN-5`'s fallback has no target.** No layer passes at `v* = 0.75` for both corpora; the best is layer 1 at `C1` 0.72 / `C2` 0.69.

**Exactly one cell survives for both corpora: `(ℓ = 1, v = 0.50)`** — `C1` 0.89, `C2` 0.77. `C1` passes 11 cells in all, every one at `v = 0.50` and none above layer 15; `C2` passes that one. One cell, `C1` at `(ℓ = 1, v = 0.75)`, came back **rogue-carried**: raw 0.72 but ablated 0.69, so the §8 ablation rule fired and excluded it exactly as intended.

**Why, and why more data does not fix it.** The binding quantity is `q/n`, and `q` is *not stationary in N*. At layer 12, `v = 0.75`, as the half grows 40 -> 70 -> 100 -> 139, `C2`'s `q` grows 18 -> 27 -> 36 -> 44: adding data does not merely estimate a fixed subspace better, it reveals more components, so the target moves. The ratio `q/n` falls only slowly (0.45 -> 0.39 -> 0.36 -> 0.32 over a 3.5x increase in `N`), and overlap tracks it — `C2` climbs only 0.24 -> 0.36 across the same range. Reaching 0.70 would require `N` orders of magnitude beyond what any refusal-balanced corpus supplies.

At `v = 0.50` the picture is different in kind, not degree: `C1`'s `q` is 5, 5, 5, 7 across the same range — genuinely low-dimensional and stable. That is why `v = 0.50` passes and the higher levels do not.

**The design has a structural tension this exposes.** `C2` must be broad for the contrast in §3 to mean anything, and breadth is precisely what makes its leading subspace unidentifiable: `C2` needs `q = 67` at layer 12 / `v = 0.75` where `C1` needs 32, and at `v = 0.90` it needs 136 components from 139 points. The estimand wants a wide corpus; the estimator cannot have one.

**Reported result.** Per §8, `v = 0.90` fails everywhere, `v = 0.75` fails for both corpora at every layer, and the primary analysis is restricted to `v = 0.50` — where only layer 1 has both subspaces identified. A single cell at layer 1 cannot carry the §4 branches: layer 1 is near-embedding and dominated by token identity, which is the orthographic confound `OPEN-2` exists to control, and both the §4 layer-structure reading and `OPEN-5`'s five-contiguous-layer rule are unsatisfiable with one layer. **No `A` value was computed**, per §10.

Options, with the data now in hand rather than projected:

- **Expand `C1`.** The on-disk `refusal_direction` splits hold 861 unique harmful and 31,323 harmless prompts, so a balanced `C1` at `N ~ 1700` is available — a 6x increase. Given the `q/n` behaviour above this buys perhaps 0.10-0.15 of overlap at `v = 0.75`, which does not reach 0.70. It would comfortably secure `v = 0.50` across more layers.
- **Restrict the primary to `v = 0.50` and expand `C1` to widen the passing layer band.** The only option that both respects the gate and reaches layers where refusal structure is expected. It changes the estimand: `v = 0.50` on these corpora is a 5-20 dimensional subspace, and the claim under test becomes one about the leading few directions rather than about "the global subspace" as §1 frames it.
- **Report the feasibility failure as the result.** §8 pre-registered this as a legitimate outcome, and the `q/n` finding is a substantive one about variance-matched subspace designs at achievable `N`, not merely a null.

**Scaling follow-up (`scripts/run_scaling.py`, `results/scaling.csv`).** `C2` is the binding constraint, so the live question was whether its passing band widens with `N` or is stuck. Split-half overlap for `C2` alone, 25 halvings per cell, no `A` computed:

| | `N = 278` | `N = 600` | `N = 1000` | `N = 1600` |
|---|---|---|---|---|
| `v = 0.50` — layers passing | 2 | 7 | **14** | **24** |
| `v = 0.50` — layer 12 overlap | 0.52 | 0.65 | **0.71** | **0.78** |
| `v = 0.50` — `q` | 16 | 21 | 24 | 25 |
| `v = 0.75` — layers passing | 1 | 1 | 2 | — |
| `v = 0.75` — layer 12 overlap | 0.37 | 0.45 | 0.51 | — |
| `v = 0.75` — `q` | 46 | 73 | 92 | — |

The two levels behave differently in kind. At `v = 0.50`, `q` **saturates** — 16, 21, 24, 25 across a 5.8x increase in `N` — so the subspace is a real, stable object of about 25 dimensions and more data estimates it better: overlap reaches 0.78 at layer 12 with 24 of 32 layers passing. At `v = 0.75`, `q` grows roughly linearly with `N` — 46, 73, 92 — so there is no stable 75%-variance subspace to find; the spectral tail is too flat, and overlap crawls (0.37 -> 0.51) while only two layers ever pass.

**This makes the `v = 0.50` restriction viable rather than merely permitted.** At `N ~ 1000-1600` the primary layer `ℓ* = 12` clears the threshold (0.71, then 0.78), which restores `OPEN-5`'s five-contiguous-layer robustness rule and §4's layer-structure reading, both dead at `N = 278`.

**The remaining obstacle is `C1`'s composition, not its size.** `C2` scales freely (31,323 harmless prompts). `C1` needs ~1000 refusal-balanced prompts; the refusing half is available (861 harmful on disk) but the non-refusing half has only ~126 clean XSTest safe prompts after eviction. The obvious filler is Alpaca — which is what `C2` is built from, so using it would put half of `C2`'s distribution inside `C1` and gut the contrast §3 measures. Resolving this requires either another benign-but-refusal-adjacent source, a refusing-skewed `C1` with strict balance dropped, or accepted partial overlap documented as a limitation.

**`C1` composition — decided 2026-09-10, before the expanded gate was run.**

    C1 = 861 harmful + 861 harmless (Alpaca), balanced,  N = 1722
    C2 = 1722 harmless (Alpaca), sampled disjoint from C1's harmless half

Recorded here before extraction, per §10.

*Why the shared Alpaca component is a feature, not contamination.* The earlier framing — that using Alpaca in `C1` puts half of `C2`'s distribution inside `C1` and guts the contrast — was wrong. Under this construction `C1` and `C2` differ in exactly one respect, the presence of harmful prompts, with the Alpaca component held common. Any difference between `U_1` and `U_2` is therefore attributable to the refusal axis rather than to prompt style, length, topic mix, or instruction phrasing. Two arbitrarily different corpora would confound all of those with the thing being measured.

*What it costs.* `C1` is no longer "a refusal corpus" in the abstract; it is a general corpus with a refusal axis added. The claim in §1 narrows accordingly — from *is the leading subspace corpus-contingent* to *does adding refusal variation to a corpus change what its leading subspace captures*. That is a smaller claim, and arguably the better-posed one, since §1's own framing is that the subspace is a joint function of model and corpus.

*A second cost, recorded now.* `C1`'s refusal axis comes from overtly harmful AdvBench/HarmBench/JBB prompts, while the `ref` displacements come from XSTest borderline minimal pairs. These are different flavours of refusal. This makes the test harder rather than easier — `U_1` must capture XSTest-style displacement despite never having seen an XSTest prompt — so a positive result is stronger and a null is correspondingly weaker evidence against corpus-contingency.

*One thing it buys.* XSTest is now reserved entirely for the eval families. `U_1` provably never sees a base prompt, so the §5 focus-leakage machinery is no longer load-bearing for `C1` — it remains only as the rule that built the `ref` pairs.

*Rejected alternatives.* A refusing-skewed `C1` with balance dropped: refusal would be near-constant in `C1` and so would not be an axis of variation its leading subspace could encode, which is the entire mechanism under test. `C1` built only from XSTest leftovers: ~126 clean non-refusing prompts cannot reach the `N ~ 1000` the scaling result requires.

**Expanded gate — run 2026-09-10, `scripts/run_expanded_gate.py`, `results/gate_expanded.csv`.** `N = 1722` per corpus, halves of 861, 25 halvings per cell.

| | `v = 0.50` | `v = 0.75` |
|---|---|---|
| layers passing for **both** corpora | **23** | 4 |
| longest contiguous run | **21 (layers 1–21)** | 4 (layers 1–4) |
| `OPEN-5`'s five-contiguous-layer rule | **satisfied** | not met |
| `ℓ* = 12` inside that run | **yes** | no |

**Primary cell `(v = 0.50, ℓ = 12)`: `C1` 0.82 raw / 0.81 ablated, `q ~ 17`; `C2` 0.81 / 0.80, `q ~ 25`. Both pass. The gate clears and the analysis may proceed at `v = 0.50`.**

`v = 0.75` remains dead even at this `N`, exactly as the scaling result predicted: its four passing layers are the shallowest ones and exclude `ℓ*`.

**Four cells came back rogue-carried**, all at `v = 0.50`: `C1` at layers 26, 27, 28 and `C2` at layer 22 — raw overlap above threshold, ablated below. They are excluded from the primary. That they cluster in the deep layers is the Timkey & van Schijndel pattern (§1): rogue dimensions dominate more strongly later in the network. The §8 ablation rule is doing exactly the work it was added for, and without it those four cells would have been counted as identified.

*A reporting bug, found and fixed.* The runner's first output claimed "longest contiguous run: 2 (31–32)" while listing layers 1–21 as passing. The contiguous-run scan appended a list by reference and then cleared that same list, so every completed run was emptied. Had it gone unnoticed it would have failed `OPEN-5`'s five-layer rule on a design that satisfies it with 21. The figures above are recomputed from `results/gate_expanded.csv` with the corrected scan.

**`OPEN-3` is closed: the experiment runs, at `v = 0.50`, on the expanded corpora.** The remaining precondition is the `arb` family (`OPEN-2`), which is researcher-authored and not yet clean.

This decision is the researcher's and is not taken here.

### `OPEN-4` — clustered bootstrap — **closed**

- **`B = 1000` replicates.** ~1.7 h after the §16 refactor; the resolution gain from 2000 does not justify doubling that against 45 clusters.
- **95% percentile CI.** BCa was considered and rejected: its acceleration term is a jackknife over the 45 base-prompt clusters, which is itself noisy at that count, and `I` is a difference of differences of logits and so roughly symmetric — BCa would buy little while adding a failure mode.
- **Multiplicity: none, by construction.** `OPEN-5` designates a single primary cell, so the headline is one test. The layer surface is secondary and reported with percentile bands, explicitly not as 96 tests.

### `OPEN-5` — primary cell and surface rule — **closed**

- **`ℓ* = 12`, `v* = 0.75`.** Layer 12 is the argmax of the prior project's refusal-direction selection grid and the cell Arditi publish for Llama-3-8B. It is taken from prior work, not from this experiment's output — the §4 requirement. `v* = 0.75` is the middle level: least exposed to the `v = 0.90` rank problems of §7 and to the coarseness of `v = 0.50`.
- **Surface rule:** the sign of `I` is read as robust only if its bootstrap CI excludes 0 at **≥ 5 contiguous layers** at `v*`. Five of 32 is ~16% of depth — enough to rule out a single-layer fluke without demanding implausible breadth.
- **Fallback, named in advance:** if `(v*, ℓ*)` fails the §8 gate, the primary moves to the nearest passing layer at `v* = 0.75`, ties broken toward the shallower layer.

### `OPEN-6` — general corpus — **closed**

**The `refusal_direction` harmless split** (Alpaca-derived instructions), already on disk at `data/stageb/harmless_{train,val,test}.json` in the prior project — 31,323 prompts, so no download and no pressure on the §15 disk headroom.

Chosen over web text because §5 requires `C2` to share `C1`'s chat template and instruction form, or the experiment measures prompt format rather than refusal content. Median instruction length is **11 tokens against XSTest's 10**, so length-matching costs almost nothing. It is also the harmless split the prior project already used, which keeps the `C1`/`C2` contrast continuous with that pipeline rather than introducing a second unexamined corpus.

Subsampled to `C1`'s `N` and length distribution, chat-templated identically, and split by the same focus-clean discipline — with `instruction` text as the leakage key, since it has no `focus` column.

