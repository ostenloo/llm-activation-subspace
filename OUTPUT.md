# Results — is the global activation subspace a property of the model or of the probe corpus?

**Status:** complete. All pre-registered analyses run; `SPEC.md` v4 §4 branches read below.
**Model:** Llama-3.1-8B-Instruct, snapshot `0e9e39f`, probe at position `−5`, float32.
**Date:** 2026-09-11.

---

## 1. Headline

**`I > 0` at the primary cell, robustly — and it does not clear the §7 absolute floor.**
Both halves are the result. The second is the more important one.

At the pre-registered primary cell (`v = 0.50`, layer 12, `n = 21` pairs):

    I = +1.525    95% CI [+0.912, +1.712]    (B = 1000, clustered on base prompt)

`OPEN-5`'s rule — sign of `I` stable with CI excluding zero across ≥5 contiguous layers
including layer 12 — is **satisfied with 15 contiguous layers (7–21)**. On the §4 branches this
is `I > 0`.

The four cells show exactly the shape the design was built to detect:

| | `C1` refusal-balanced | `C2` general |
|---|---|---|
| `A_ref` (refusal-flip) | **0.415** | 0.158 |
| `A_arb` (nuisance-matched) | 0.135 | 0.159 |

`C1`'s leading subspace captures refusal-flip displacement 2.6x better than `C2`'s, while
capturing the nuisance-matched control no better at all. The generic-concentration confound §3
was built to remove does not appear in the `arb` row.

**Then the §7 floor removes the interpretation.** Drawing `r_j ~ N(0, Σ_z)` — same anisotropy,
same spectrum, *not* the top PCs — and forming the same interaction from those random subspaces
gives

    I from spectrally-matched random subspaces = +1.698   [+1.223, +2.045] over 20 draws

which is **larger than the observed +1.525**. A random subspace with the corpus's spectrum
reproduces the entire effect. The four null cells show why:

| cell | observed | spectral null |
|---|---|---|
| `ref_1` | 0.415 | 0.298 |
| `ref_2` | 0.158 | 0.085 |
| `arb_1` | 0.135 | 0.071 |
| `arb_2` | 0.159 | 0.083 |

The `ref_1` / `arb_1` asymmetry that drives `I` is already present in the null. The refusal
displacement aligns with `C1`'s high-variance structure *in general*, not with its leading
eighteen directions in particular.

**That null is a fair control, not a degenerate one.** It overlaps the leading subspace at
0.32 (1.0 = identical; 0.01 = isotropic), so it shares the spectrum and only a third of the
directions. It is not simply the leading subspace under another name.

### What can and cannot be claimed

**Can:** corpus identity changes what an activation subspace captures, and the change is
specific to refusal-relevant displacement rather than generic. Corpus-contingency is real.

**Cannot:** that the *leading* subspace is where this lives. §1 frames the target as the
subspace obtained by running PCA and interpreting the leading directions as a model property.
This experiment finds the corpus-dependence operates through the corpus's **covariance structure
as a whole**, and the identity of the top directions adds nothing measurable on top of it.

The rank-band control agrees that the leading directions are not interchangeable with arbitrary
ones — PCs 19–36 capture only 0.085 against the leading eighteen's 0.415 — but "better than the
adjacent band" and "better than a spectrum-matched random draw" are different bars, and the
effect clears only the first.

This is what §7's insistence on a spectrally-matched floor rather than the naive `q/D` floor was
for. Under `q/D` (0.004) the result would have looked overwhelming.

---

## 2. The layer profile

| layer | `A_ref,1` | `A_ref,2` | `A_arb,1` | `A_arb,2` | `I` | `q₁/q₂` |
|---|---|---|---|---|---|---|
| 1 | 0.205 | 0.320 | 0.214 | 0.314 | −0.087 | 2/5 |
| 4 | 0.174 | 0.183 | 0.096 | 0.094 | −0.090 | 9/12 |
| 5 | 0.251 | 0.180 | 0.161 | 0.124 | +0.121 | 18/18 |
| 8 | 0.288 | 0.151 | 0.163 | 0.139 | +0.631 | 21/22 |
| 10 | 0.337 | 0.167 | 0.150 | 0.151 | +0.939 | 24/25 |
| 11 | 0.401 | 0.154 | 0.141 | 0.160 | +1.450 | 21/26 |
| **12** | **0.415** | **0.158** | **0.135** | **0.159** | **+1.525** | 18/27 |
| 13 | 0.378 | 0.160 | 0.138 | 0.154 | +1.292 | 18/25 |
| 15 | 0.374 | 0.201 | 0.111 | 0.129 | +1.041 | 19/28 |
| 18 | 0.346 | 0.213 | 0.096 | 0.123 | +0.958 | 23/32 |
| 21 | 0.335 | 0.218 | 0.088 | 0.119 | +0.923 | 22/32 |

`I` is negative in the first four layers, crosses zero at layer 5, rises to a peak at layer 12,
then declines slowly through layer 21.

**Layer 12 was fixed in advance.** It is the argmax of the prior project's refusal-direction
selection grid and the cell Arditi et al. publish for Llama-3-8B. It was taken from prior work,
never from this experiment's output — the §4 requirement. That the interaction peaks exactly
there is a prediction confirmed, not a maximum selected.

Under §4's secondary, pre-registered layer reading, `I(ℓ)` is **systematically depth-dependent
rather than flat**, which is *consistent with* a representation-dependent explanation rather
than a corpus-composition one. §4's wording is deliberate: neither pattern identifies a
mechanism, and this experiment does not license a causal claim about depth.

---

## 3. What had to happen first

**The gate failed at the pre-registered scale, and the design was rebuilt around it (§8, §17).**

At `N_fit = 278` (XSTest alone) the primary cell failed outright: `C1` 0.54, `C2` 0.37 against a
0.70 threshold, with no layer passing at `v = 0.75` for both corpora. Exactly one cell survived
anywhere — layer 1 at `v = 0.50`.

A scaling run isolated why. **`q` is not stationary in `N`.** At layer 12, `v = 0.75`, as the
fit half grows 40 → 70 → 100 → 139 → 800, `C2`'s `q` grows 18 → 27 → 36 → 44 → 113: adding data
does not merely estimate a fixed subspace better, it reveals more components, so the target
recedes. At `v = 0.50` the same sweep gives `q` = 16 → 21 → 24 → 25 — saturating. That is why
`v = 0.50` is estimable and `v = 0.75` is not, at any feasible `N`.

§8 pre-registers restricting the analysis to a passing level, so `v = 0.50` is the authorized
response rather than a post-hoc rescue.

**Corpora were expanded to `N = 1722` each (§17):**

    C1 = 861 harmful + 861 harmless (Alpaca), balanced
    C2 = 1722 harmless (Alpaca), disjoint sample

The shared Alpaca component is deliberate: `C1` and `C2` then differ in exactly one respect — the
presence of harmful prompts — so any gap between `U_1` and `U_2` is attributable to the refusal
axis rather than to style, length, or topic mix. It also retires the leakage problem: XSTest is
reserved wholly for the eval families, so `U_1` provably never sees a base prompt.

At `N = 1722` the gate passes: **23 layers pass for both corpora at `v = 0.50`, longest
contiguous run 21 (layers 1–21), containing layer 12.** Primary cell `C1` 0.82/0.81, `C2`
0.81/0.80.

Four cells returned **rogue-carried** (raw above threshold, ablated below) — `C1` layers 26–28,
`C2` layer 22, all in the deep layers, which is the Timkey & van Schijndel pattern. Without the
§8 ablation rule they would have counted as identified.

---

## 4. Attrition — the main limitation

The `OPEN-1` behavioural flip check, run against the prior project's `REFUSAL_PREFIXES`
(`andyrdt/refusal_direction @ 9d852fa`), kept **21 of 44 pairs**:

    base refused:  11/44   (want 0)   — XSTest over-refusal, working as designed
    ref  refused:  36/44   (want all) — 8 unsafe variants not refused
    arb  refused:  14/44   (want 0)   — authored arb variants that trigger refusal
    kept:          21/44

Every number above rests on 21 base prompts. This is the binding constraint on precision: §9's
bootstrap clusters on base prompt, so it has 21 clusters regardless of `N_fit = 1722`.

**The attrition is not random.** Dropping pairs whose base or `arb` member is refused selects
for prompts whose safe reading is robustly safe, which may bias the surviving `ref`
displacements toward cleaner flips. The direction of that bias on `I` is not established here.

---

## 5. Provenance and corrections

Defects found and fixed during execution, each recorded in `SPEC.md` §13 with its effect:

- **Token position was wrong.** v1–v3 probed the last position, justified as matching Arditi
  et al. and the existing pipeline; both claims were false. Under the Llama-3.1 template the last
  five user-turn positions are `<|eot_id|>`, `<|start_header_id|>`, `assistant`,
  `<|end_header_id|>`, `\n\n`, so `−1` is a newline. Corrected to `−5`.
- **Empty-focus grouping.** 75 of XSTest's 450 rows leave `focus` empty; grouping by that field
  treated them as one group, manufacturing a pair and over-evicting 75 prompts. Fixed: `N_fit`
  228 → 278, gate margin +3 → +53.
- **Self-contradictory rank-band rule** (§7) — the fallback band always overlapped the leading
  set it was meant to be disjoint from.
- **Aliasing bug in the contiguous-run scan** — reported "longest run 2" on a design with a
  21-layer run, which would have failed `OPEN-5` on a design that satisfies it.

`arb` authorship: all 44 rows authored by the researcher; 7 mechanically adjusted by Claude to
hit exact token-Levenshtein targets, semantic choice and referent domain preserved (§17).

---

## 6. Controls at layer 12 (§7)

| | `A_ref` | spectral null | rank band (PCs 19–36 / 28–54) |
|---|---|---|---|
| `C1` | 0.415 | 0.298 | 0.085 |
| `C2` | 0.158 | 0.085 | 0.073 |

**Per-base-prompt distribution** — mandatory reporting, because `A` is a ratio of sums and can
be dominated by the largest `‖Δz_i‖`:

| | aggregate | median | trimmed (10%) | range | top-3 share |
|---|---|---|---|---|---|
| `C1` | 0.415 | 0.383 | 0.372 | [0.139, 0.663] | 24.0% |
| `C2` | 0.158 | 0.145 | 0.160 | [0.067, 0.271] | 22.8% |

Aggregate, median, and trimmed mean agree closely for both corpora, and the top three prompts
carry 24% of the summed `a_i` against 14.3% under uniform. **The effect is broad-based across
the 21 pairs, not driven by a handful.** The `C1` / `C2` gap holds at the median (0.383 vs
0.145) as well as in the aggregate, so it is not an artifact of the ratio-of-sums.

---

## 7. Standing limitations

- **`n = 21` base prompts.** §9's bootstrap clusters on base prompt, so precision is bounded by
  21 regardless of `N_fit = 1722`. Attrition was selective (§4), not random.
- **`A` measures projection, not causal relevance** (§12). Nothing here establishes that the
  captured directions do the behavioural work.
- **`C1` is a general corpus with a refusal axis added**, not a refusal corpus (§17). The claim
  narrows accordingly.
- **`C1`'s refusal axis is overtly harmful prompts; the `ref` displacements are XSTest
  borderline pairs.** Different flavours, which makes a positive result stronger and a null
  weaker.
- **One model, one token position, two corpora.** §12 stands unchanged.
- **`v = 0.75` and `v = 0.90` were never estimable** at any feasible `N` (§3). The result speaks
  to a ~18–27 dimensional subspace, not to "the global subspace" as §1 frames it.
- Seven of 44 `arb` variants had their surface form adjusted by Claude to hit token targets
  (§17); semantic choice was preserved but the residual risk is nonzero.
