# Patent Portfolio Scale, Relatedness, and Institutional Evaluation

Replication code for a study of Japanese pharmaceutical and life-science-related
patent applications (IPC classes A61, C07, C12; 1990–2021), examining whether
patent portfolio **scale** and **relatedness** are associated with (RQ1)
continued patenting within established technological fields, (RQ2) examiner
recognition of an applicant's own prior art, and (RQ3) the timing of grant
versus rejection-related examination records.

The manuscript, prepared for double-anonymous review, is in
[`manuscript/`](manuscript/).

## Data

This repository contains **code only**. The underlying data are drawn from the
**IIP Patent Database**, distributed by the Institute of Intellectual Property
(Tokyo) under a license that does not permit redistribution. Researchers with
independent access to the IIP Patent Database can reproduce the analysis by
placing the raw decade files (`ap_*.txt`, `applicant_*.txt`, `cc_*.txt`,
`hr_*.txt`) in the working directory and running the pipeline in
[`src/01_data_construction/`](src/01_data_construction/).

Derived, non-confidential outputs sufficient to inspect the estimation
samples' structure (e.g., variable definitions, sample sizes) are described
in the manuscript's Appendix A.

## Pipeline overview

Scripts are organized by stage. Each stage assumes the outputs of the
previous stage are available as `.pkl` files in the working directory
(paths are set via a `CACHE_DIR` variable at the top of each script — update
this to point at your local cache directory before running).

### `src/01_data_construction/`

Builds the analysis panel from raw IIP Patent Database decade files.

| Script | Purpose |
|---|---|
| `01_build_variables.py` | Constructs the application-level (`df_application`) and firm-field-year panel (`panel`) datasets: patent stock (decayed and raw), grant stock, relatedness, portfolio concentration/entropy, citation-context indicators, and event-history durations. |
| `02_apply_identity_filter.py` | Conservative applicant-identity screen: excludes ~2.7% of applications with string-similarity-flagged, manually-reviewed applicant-code conflicts. |
| `02b_apply_identity_filter_lenient.py` | Alternative, less conservative screen recovering 7 codes with documented corporate name lineage (used for the robustness re-estimation reported alongside each main table). |
| `diag_idname_consolidation_check.py`, `diag_idname_integrity_full.py` | Diagnostics supporting the identity-screening decision (string-similarity thresholds, manual review flags). |

### `src/02_main_models/`

Produces the paper's four main tables.

| Script | Paper table | Model |
|---|---|---|
| `03_model1a_established_field_filing.py` | Table 1 | RQ1: PPML, established-field filing intensity |
| `04_model1b_portfolio_diversification.py` | Table 2 | Portfolio-level HHI / entropy (raw and normalized) |
| `05_model2_examiner_recognition.py` | Table 3 | RQ2: two-part LPM, own-patent citation conditional on citation context |
| `06_model3_examination_timing.py` | Table 4 | RQ3: separate Cox event-history models (grant timing, rejection-record timing, rejection-to-grant transition) |
| `07_risk_set_correction_and_ipc_heterogeneity.py` | Table 1 columns/text | Established-field risk-set correction ($t>T^0_{ig}$), IPC-subclass heterogeneity, lenient-identity re-estimation |

### `src/03_robustness_checks/`

Robustness analyses reported in Section 4 and Appendix A.

| Script | Addresses |
|---|---|
| `stock_decomposition.py` | Granted vs. non-granted stock decomposition (Section 4.1) |
| `sole_applicant_robustness.py` | High-sole-applicant-share restriction for RQ1 and RQ2 |
| `divisional_family_approximation.py` | Bulk (same-day, same-firm-field) filing exclusion; extensive-margin (binary) specification |
| `leave_large_firms_out_relatedness.py` | Reconstructs the relatedness matrix excluding the top 1% of firms by filing volume (Appendix A, Table A2) |
| `conditional_citation_count_decomposition.py` | RQ2 self-/external-/total-citation count decomposition, conditional on a recorded rejection context (matches Table 3, Part 2's sample exactly) |
| `event_history_restructure_and_citation_counts.py` | Earlier version of the event-history and citation-count restructuring (superseded in reporting by the conditional version above, retained for provenance) |
| `ph_sensitivity_and_nonlinearity.py` | Piecewise Cox sensitivity to the grant-timing proportional-hazards violation; quadratic relatedness term (nonlinearity check) |

### `src/04_rq3_diagnostics/`

The extended diagnostic history behind RQ3's relatedness–grant-timing
finding (Appendix B). An initial full-sample estimate (0.020, n.s.) proved
unstable under 25–30% random firm subsampling (≈−0.40, p<0.001); the scripts
below trace the investigation that followed.

| Script | Finding |
|---|---|
| `joint_test_and_lenient_model3.py` | Stacked cause-specific joint test (grant vs. rejection-record hazards), application-clustered; lenient-identity re-estimation of Table 4 |
| `joint_test_firm_clustered.py` | Same joint test, firm-clustered — confirms identical results to three decimal places |
| `grant_timevarying_and_fieldonly_strata.py` | Field-only stratification (full sample, static): 0.034, n.s. A time-varying treatment of the rejection-related record *without* field stratification: −0.570, p<0.001 |
| `timevarying_field_stratified.py` | Time-varying treatment *with* field/regime/corporate stratification: 0.028, n.s. — resolves the divergence above as attributable to missing field controls, not the time-varying treatment itself |
| `subsample_influence_diagnostic.py` | Nine further diagnostics (top-1%-firm exclusion, five reseeded subsamples, three subsamples from a top-1%-excluded population, sequential exclusion of the ten largest firms) — none reproduce the original ≈−0.40 estimate |

**Summary of the RQ3 diagnostic conclusion:** three full-sample specifications
(varying stratification granularity and static-vs-time-varying treatment of
the rejection-related record) converge on a null relatedness–grant-timing
association; the original unstable subsample estimate is not reproduced
across a subsequent battery of independent checks and appears to be an
unrepresentative single realization rather than a demonstrated,
reproducible sensitivity.

## Requirements

See [`requirements.txt`](requirements.txt). Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

Fixed-effects Poisson (PPML) models use
[`pyfixest`](https://github.com/py-econometrics/pyfixest); survival models
use [`lifelines`](https://lifelines.readthedocs.io/).

## Citation

If you use this code, please cite the paper (citation details to be added
upon publication).

## License

Code in this repository is released under the MIT License (see
[`LICENSE`](LICENSE)). This license covers the code only; it does not extend
to the IIP Patent Database, which remains subject to its own licensing
terms and is not redistributed here.
