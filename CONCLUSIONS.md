# Research Conclusions

Consolidated findings of the minimum-data tool-wear prognosis study. Every number below is produced
by a deterministic script in `scripts/` and traced to a CSV in `results/` (see the reproduction table
in the README).

---

## 1. Principal findings

1. **Few-shot physics prognosis works at industrially useful accuracy.** From only the first four
   microscope measurements of a never-seen tool, a monotone power-law wear model with a fleet-learned
   exponent and a robust, extrapolation-weighted per-tool fit predicts the unseen remainder of the
   wear curve to **5.6 µm MAE** (2.8 % of the 200 µm wear criterion; pooled R² 0.70), against
   18.7 µm for the fleet-average baseline. The improvement is unanimous across all 18 tools
   (sign test p ≈ 8·10⁻⁶) and selection-robust (a nested double leave-one-tool-out re-run of the
   entire configuration search reproduces 5.60 µm with the same winner in every fold).

2. **Uncertainty can be guaranteed, not just estimated.** A horizon-binned (Mondrian) conformal band
   achieves its nominal 90 % coverage empirically (90.1 %) at 52.5 µm mean width — ±19 µm at the
   near horizon, 25.1 µm at 92 % for the precise operating point. Remaining-useful-life windows
   validated on 16 interval-censored threshold-crossing events contain the truth in 94 % of cases.

3. **End of life is a hazard, not a threshold.** Every tool in the campaign terminated in a sudden
   edge-chipping event; the wear level at chipping disperses from 127 to 291 µm. A discrete-time
   logistic hazard on the wear state (a degradation-threshold-shock formulation) turns those 18 real
   failures into a decision rule: a chipping-safe stop at **VB_safe ≈ 167 µm** for a one-cycle risk
   budget of 10 %.

4. **The vibration branch does not transfer across tools in an unreplicated design.** Given every
   fair chance (physics-anchored dimensionless indicators, break-in normalisation, PLS, six fusion
   strategies), the sensor branch never exceeds R² ≈ 0 out-of-sample. In a deliberately handicapped
   duel — physics forecasting blind from m early points versus the sensor branch *reading* the
   vibration at the very cut it must predict — physics wins on 16 of 18 tools (5.6 vs 35.2 µm,
   Wilcoxon p < 10⁻⁴). The live signal carries the cutting condition, not the tool-specific wear
   state. Residual value: radial transient indicators (spectral kurtosis) show exploratory promise
   as chipping-risk covariates (LRT p = 0.010; not robust to family-wise correction).

5. **The binding constraint is the experiment design, not the model.** One tool per condition
   confounds the condition effect with tool-to-tool variability. Every method family that promises
   to overcome data scarcity by modelling was tested and failed to beat the simple physics law
   (Section 2). The single highest-value next step is replication: at least two tools per cutting
   condition.

## 2. Negative-results catalogue

All methods below were evaluated under the identical leakage-safe leave-one-tool-out protocol with a
pre-stated adoption rule (adopt only if it beats the record at the same measurement budget with valid
interval coverage). None was adopted. Reference records: m = 3 → 11.0 µm, m = 4 → 5.6 µm.

| Method family | Representative result | Why it fails here |
|---|---|---|
| Hierarchical-Bayes / particle-filter posterior predictive | 15.3 / 12.0 µm | the fleet prior drags the new tool toward the population mean — the same mechanism that sinks the fleet-average baseline |
| Kalman filter as long-horizon forecaster | 12.2 µm, 621 µm bands | constant-velocity extrapolation is inferior to the power law; its domain is one-step monitoring (3.7 µm) |
| Inverse-Gaussian process with random effects | 12.1 / 8.7 µm | strongest stochastic-process competitor on record; still behind the extrapolation-weighted fit |
| Wiener process, hierarchical partial pooling, curve registration | worse than base | non-monotone / population-anchored |
| Canonical wear laws (Archard p = ½, extended Taylor, Usui-type) | 12.0 / 10.0 · 16–47 µm · marginal | fitting the exponent on the fleet is precisely what makes the law competitive |
| Condition-parameterized exponent p(condition) | loses at both budgets; p ~ condition R² ≈ noise | condition-to-shape map unidentifiable with one replicate per condition |
| Trajectory-similarity library | 22–39 µm | needs replicates to populate the library |
| Bayesian model averaging over law forms | ≈ base | fleet BIC places weight 0.997 on the power form |
| Data augmentation — physics Monte-Carlo (seeded wear equation), Mega-Trend-Diffusion, gamma process; swept over fleet sizes K ∈ [4, 17] | neutral-to-harmful at every K | synthetic tools carry no information beyond the real tools they are sampled from |
| Meta-learning / domain adaptation (CORAL, MMD) / GAN | non-starters | n = 18; domain coincides with the confounder; covariance estimates rank-deficient |
| Alternative conformal constructions (CQR, jackknife+, normalized scores × 3, severity bins, kinematic-kernel weighting, conformalized survival bounds) | under-cover or inflate | the error scale is governed by the forecast horizon; at a fixed 90 % guarantee the interval sits at its data-imposed floor |
| ElasticNet, SHAP-guided selection, few-shot PINN (legacy-stack re-audit) | R² −0.75 · R² −0.40 · 15.9/14.3 µm | confirm the adjudicated picture |

## 3. Inferential analysis of the design of experiments

Although the unreplicated design forecloses a *predictive* condition-to-wear map, the classical
machinery for unreplicated factorials (half-normal ordering; Lenth's pseudo-standard-error test;
ANOVA with the pooled three-factor interaction as error) answers the *inferential* question:

- The wear-**rate** scale is condition-silent even inferentially (strongest effect p ≈ 0.28).
- **Cooling** is the largest effect on the wear **levels** — ~41 % of the explained variation of the
  break-in level and ~33 % of the wear-at-chipping (Lenth p = 0.015 / 0.020; ANOVA p = 0.001 / 0.002)
  — a directional finding (not robust to a Holm correction over the full effect family).
- Interpretation: the condition acts on the wear *level*, which is exactly the per-tool parameter the
  few-shot fit personalizes from each tool's own early measurements — the reason the forecast
  succeeds without knowing the condition.

## 4. External validation (PHM2010)

Run unchanged on the public PHM2010 milling benchmark (three labelled cutters, subsampled to a sparse
inspection cadence), the method transfers out of the box at the conservative operating point (pooled
R² 0.69 from three inspections; time-only extrapolation collapses to R² −0.49) but does not beat the
fleet-average curve — and should not: PHM2010 is the opposite regime (one condition, true replicates,
gradual tertiary acceleration in which the final fifth of life wears 2.6–7.6× faster). A hybrid
"fleet shape + few-shot affine personalization" closes the gap there (15.2 vs 15.8 µm), confirming
the framework scales with replication. The external check validates the method's scope claims.

## 5. Limitations

- **Sample size:** 18 tools; the bootstrap 95 % CI of the mean improvement is wide ([0.1, 15.8] µm)
  even though the direction is unanimous.
- **Point RUL at short horizons** is bounded by the one-cut inspection grid: for 11 of the 16
  validation events the ±20 % accuracy cone is narrower than one inspection interval.
- **Wear regime:** conclusions hold for VB ≤ 300 µm and the monotone-decelerating regime; gradual
  tertiary acceleration lies outside the concave law's expressive range (kept as a declared annex).
- **Single laboratory, single tool/material system** — external validity beyond the PHM2010 scope
  check is untested.

## 6. Future work

1. **Replicated campaign** (≥ 2 tools per cutting condition) — makes the condition effect learnable,
   tightens the accuracy CI, gives the sensor branch a legitimate rematch, and unlocks the
   fleet-shape hybrid validated externally.
2. A few tools run through a **gradual, densely sampled wear-out** — point-validates the safe-stop
   RUL and the tertiary annex.
3. **Raw per-contact signal capture** — enables cyclostationary indicators for intermittent milling
   and the exploratory chipping-risk covariate's confirmatory test.
4. Multi-mechanism degradation labels (notch, crater, thermal corrosion) and image-based wear
   quantification as independent research lines on the existing specimens.
