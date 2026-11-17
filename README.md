# GoCAP

Empirical **40m propagation research**, currently on the **`Survey`** branch.
GoCAP predicts **received signal strength for each requested UTC hour** between
two **four-character Maidenhead squares**, using an ML model trained on historical
positive reception reports and solar/geomagnetic context.

**Missing reception is unknown, never evidence that a band or path was closed.**
The current product predicts a typical SNR among reported receptions. It does not
provide a calibrated contact-success probability, and it does not select a
“strongest hour” as its objective.

## Run a prediction

No reception database, raw archive download, VOACAP installation or GPU is needed
for inference. The current model is
[`artifacts/gocap-40m-hourly.joblib`](artifacts/gocap-40m-hourly.joblib),
3,271,039 bytes (about 3.12 MiB). Use the hourly model rather than the earlier
ranking/tuning research artifacts.

The tested environment is **Python 3.15.2 on macOS arm64**, with scikit-learn 1.9.1.
The exact package versions are in
[`requirements-hourly-lock.txt`](requirements-hourly-lock.txt). Other platforms
need compatible wheels or a source build; they have not been qualified here.

```sh
git clone --branch Survey https://github.com/s53zo/GoCAP.git
cd GoCAP
python3.15 -m venv .venv
.venv/bin/python -m pip install -r requirements-hourly-lock.txt

# Optional integrity check (macOS; on Linux use sha256sum -c SHA256SUMS).
(cd artifacts && shasum -a 256 -c SHA256SUMS)

# JN76 -> FN31, beginning at midnight UTC, all 24 hours.
.venv/bin/python -m ml40.hourly_query \
  artifacts/gocap-40m-hourly.joblib \
  JN76 FN31 2026-11-17T00:00Z \
  --sfi 150 --kp 3 --ap 10 --power-watts 5 --hours 24
```

Append `--format json` for structured output, or choose an observation profile
with `--mode WSPR`, `FT8`, `FT4` or `CW`. The default is WSPR. Run
`.venv/bin/python -m ml40.hourly_query --help` for options. Run commands from the
repository root. Load only trusted joblib artifacts; the included model hashes
are in [`artifacts/SHA256SUMS`](artifacts/SHA256SUMS).

| Input | Meaning |
|---|---|
| `JN76 FN31` | Directed transmitter and receiver squares; square centres are used, not exact station coordinates. |
| Timestamp | Start hour in UTC; choose 1–24 consecutive hours. A naive timestamp is treated as UTC; an explicit `Z` is clearer. |
| `--sfi` | Previous completed-day F10.7 solar flux scenario, in solar flux units. |
| `--kp` | Most recent completed three-hour Kp scenario, from 0 to 9. |
| `--ap` | Previous completed-day planetary **Ap** scenario, from 0 to 400. This is the daily A-level input, distinct from three-hour lowercase ap. |
| `--power-watts` | Reported/scenario transmitter power in watts, used **only by the WSPR profile**. Other profiles have unknown historical power; this option does not scale their predictions. |

The output contains an SNR estimate in **dB referenced to 2500 Hz noise bandwidth**
for every requested hour, a nominal 80% received-population interval, and
**theoretical** WSPR/FT8/FT4/CW/SSB margins. There is no PHONE observation model:
PHONE margins use theoretical thresholds. Thresholds are in
[`ml40/inference.py`](ml40/inference.py); margins are not probabilities.
Actual antennas, antenna heights, receiver noise and losses were generally
unrecorded. A 100 W/dipole physical calibration remains **unverified**.

For Python use:

```python
from ml40.hourly_query import predict

result = predict(
    "artifacts/gocap-40m-hourly.joblib",
    "JN76", "FN31", "2026-11-17T00:00Z",
    sfi=150, kp=3, ap=10, mode="WSPR", power_watts=5, hours=24,
)
for row in result["hourly_predictions"]:
    print(row["utc_hour"], row["estimated_snr_db_2500hz"])
```

### Dates and solar history

You can request a historical or future UTC timestamp and supply hypothetical
solar values. This is a **conditional scenario**, not an automatic solar forecast.
Dates, locations and index combinations outside the training population may
extrapolate poorly; inspect the output range and magnetic-epoch flags. Marginal
range checks do not prove support for every joint scenario.

Without explicit history, queries assume steady preceding Kp over 72 hours and
steady preceding F10.7 over 27 days. For one-hour queries, `--history-file`
accepts JSON with:

- `ending_boundary_utc`: the last completed three-hour boundary for that query;
- `kp_3h`: 24 completed three-hour Kp values, oldest first, ending at that boundary;
- `sfi_previous27_days`: 27 preceding completed-day F10.7 values, oldest first.

The last Kp/F10.7 entries must agree with `--kp`/`--sfi`. `--ap` remains the
separately supplied previous-day Ap. History mode requires `--hours 1`; future
or inconsistent history boundaries are rejected. See the validation and exact
feature construction in [`ml40/v3_features.py`](ml40/v3_features.py).

## Experimental shared model with five path points

The [shared-model experiment](SHARED_MODEL_RESULTS.md) pools WSPR, FT8, FT4 and
CW training data into one regressor and adds explicit features at 0%, 25%, 50%,
75% and 100% along the great-circle path. Its query applies all theoretical mode
thresholds to one reference SNR, using the WSPR observation context at supplied
power. PHONE remains threshold-only; physical mode independence is unverified.

```sh
.venv/bin/python -m ml40.shared_query JN76 FN31 2026-11-17T00:00Z \
  --sfi 150 --kp 3 --ap 10 --power-watts 5 --hours 24 --format json
```

On **500 new paths / 4,820 observed hours**, case-average MAE is **5.267 dB** for
the shared candidate, **5.270 dB** for current GoCAP and **13.94 dB** for VOACAP.
The shared/current difference is within bootstrap uncertainty. The candidate
[`gocap-40m-shared-points.joblib`](artifacts/gocap-40m-shared-points.joblib) remains
**experimental**; the default hourly model above is retained. See the report for
the explicit-point ablation, mixed other-mode audit and reproducible evidence.

## March data expansion, 2010–2024

The separate [March expansion experiment](MARCH_MODEL_RESULTS.md) adds every
available March day in **each year 2010–2024**, at the existing 1-in-128 report
sampling rate. All **465 daily windows** contain usable data. Existing duplicate
observations and reserved validation/calibration dates are excluded from the
addition, leaving **1,821,048 new WSPR hourly records** and **2,950,381 total
multi-mode training records across 699 days**. Existing FT8/FT4/CW data is retained;
no full-month expansion of those sources is claimed. No 2025 receptions are fitted.

The March-expanded model is experimental. On six new March 2025 dates, hourly
MAE is **6.467 dB**, versus **6.494 dB** for its shared parent and **6.482 dB** for
current GoCAP. The broader known 500-path score gets slightly worse, so the
default model remains unchanged. Run the new model explicitly with:

```sh
.venv/bin/python -m ml40.hourly_query \
  artifacts/gocap-40m-shared-march.joblib \
  JN76 FN31 2026-11-17T00:00Z \
  --sfi 150 --kp 3 --ap 10 --power-watts 5 --hours 24
```

The March fitting path batches encoding into disk-backed matrices; fitting still
needs RAM for the learner workspace. Its measured peak process RSS was about
4.74 GiB on the 16 GB test machine. See the report for per-year coverage, the
March 2020 legacy-mode amendment, raw-data verification and reproduction stages.

## VOACAP comparison across 15 years

The [15-year VOACAP comparison](VOACAP_15YEAR_COMPARISON.md) uses **300 held-out
April path-days, 20 per year from 2010–2024**, with **3,081 real hourly medians**.
Those dates are absent from both original and March-expanded fitting data.
Case-average error is **6.07 dB for the March model versus 17.04 dB for primary
VOACAP**, with GoCAP lower in every annual average and 262/300 cases. VOACAP's
13.31 dB positive bias and 11.01–19.02 dB sensitivity range make the unknown
antenna/noise assumptions consequential. This is an April retrospective
held-out-day comparison, not forward-year forecasting or physical calibration.
The March/current GoCAP difference remains within uncertainty; the default is
unchanged. See the report for the per-year table and frozen case evidence.

## Data sources and attribution

| Source | Used for | Access and implementation |
|---|---|---|
| **WSPRNet volunteers / WSPR Live** | Positive WSPR reception SNR, endpoints, time, band and reported transmit power. Main historical 40m evidence and VOACAP comparison truth. | [WSPR Live](https://wspr.live/), read-only ClickHouse endpoint `https://db1.wspr.live/`; collectors: [`collect.py`](collect.py), [`ml40/collect.py`](ml40/collect.py), [`ml40/voacap_collect.py`](ml40/voacap_collect.py). |
| **PSK Reporter via HamSCI/HARC at CEDAR Madrigal** | Historical positive FT8 and FT4 receptions; source/mode context retained. | [CEDAR Madrigal](https://cedar.openmadrigal.org/), HARC instrument 8308; consent-based [`collect_harc.py`](collect_harc.py), [`ml40/harc.py`](ml40/harc.py). The live PSK Reporter probe is a schema/quality check, excluded from training. |
| **Reverse Beacon Network volunteers** | Positive CW reception evidence. Original daily files resolve mode and native SNR for HARC records. | Daily ZIPs at `https://data.reversebeacon.net/rbn_history/YYYYMMDD.zip`; collectors above. Unmatched mode records are excluded. |
| **GFZ geomagnetic indices** | Completed three-hour Kp/ap, preceding Ap and derived recent storm context. | [GFZ data](https://kp.gfz.de/en/data); [combined text series](https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt). Fetch with `collect.py --source gfz`. Cite Matzka et al. (2021), DOI [10.1029/2020SW002641](https://doi.org/10.1029/2020SW002641), data DOI [10.5880/Kp.0001](https://doi.org/10.5880/Kp.0001). |
| **DRAO / Natural Resources Canada, distributed by GFZ** | Previous completed-day F10.7 and preceding 27-day context. | Same GFZ combined series. GFZ sunspot columns are not inputs to the current GoCAP model. |
| **IAGA IGRF-14 / NOAA NCEI** | Degree-one centred-dipole magnetic-coordinate approximation for path features. | [Original coefficients](https://www.ngdc.noaa.gov/IAGA/vmod/coeffs/igrf14coeffs.txt); the small numerical subset, attribution and source checksum are in [`ml40/igrf_degree1.json`](ml40/igrf_degree1.json). This is not full magnetic-field ray tracing. |
| **WDC-SILSO, Royal Observatory of Belgium** | Final 13-month smoothed monthly Version 2 SSN for the **VOACAP benchmark**, not GoCAP inference. | [Series](https://www.sidc.be/SILSO/DATA/SN_ms_tot_V2.0.txt), [data policy](https://www.sidc.be/SILSO/datafiles), DOI [10.24414/qnza-ac80](https://doi.org/10.24414/qnza-ac80); CC BY-NC 4.0 with attribution. |
| **VOACAP native engine / dxtron public API pilot** | Physical-model comparator only; no VOACAP-generated labels are used to train GoCAP. | [Native port](https://github.com/jawatson/voacapl), pinned commit `c12a98b348c39f36f2659b8b3aa97b86284c8a1d`; [dxtron documentation](https://voacap.dxtron.com/api/docs/). Bulk comparisons run locally. |

WSPR Live states that service use is for noncommercial research/projects whose
results are freely accessible. This private source repository does not waive
that condition or establish permission for commercial use. Keep attribution and
arrange freely accessible research results when publishing this work; do not
assume that private GitHub storage changes upstream data conditions. No external
publication or blanket data redistribution permission is supplied by this repo.
Other providers' access and attribution requirements continue to apply; access
being public does not itself grant a new license. No blanket open-source license
has been selected for GoCAP in this private research branch.

Madrigal downloads require the researcher's authorized name, email and
affiliation. Put them in an **ignored local** `.cedar-identity.json` using keys
`user_fullname`, `user_email`, `user_affiliation`, and set permissions to 0600.
Submit only details whose storage is consented to. No identity file or credentials
are included in Git. Source queries, sampling rules, retrieval timestamps and
SHA-256 checksums are saved by the collectors; retain cached files to reproduce
an exact historical run because upstream data can change.

### What the current model learned from

The WSPR point model was fitted on **1,011,619 hourly records across 258 days**:

| Years | Actual WSPR training-date schedule |
|---|---|
| 2010, 2016, 2019, 2022 | Days 1, 8, 15 and 22 of every month. |
| 2013 | Same schedule except January 15/22 and all February dates are absent. |
| 2024 | Days 1 and 15 of every month. |

This is sampled data, **not a continuous 2010–2024 archive**. Latest point-fitting
date: **2024-12-15**. The 8th of each 2024 month is reserved for validation/model
selection, and the 22nd for interval calibration. FT8/FT4 fitting evidence covers
2024-01-15, 04-15, 07-15 and 10-15; CW fitting evidence covers the first three of
those dates. The combined fitting partition contains 1,129,333 hourly records.
[Exact dates and per-mode/split counts](results/ml40/training_dates.csv) are included.
No 2025 reception data was used for fitting.

Reception aggregates are medians by source, mode, UTC hour, transmitter,
receiver, square pair and reported power. Duplicates are removed and invalid
locations/frequencies screened. Native SNR bandwidth differences are normalized;
legacy WSPR decoder contexts remain explicit. HARC WSPR records without comparable
power/mode metadata are audit-only. Historical HARC timestamp inconsistencies
are checked against explicit UTC fields and original RBN records. Further
measurement caveats and sampling details are in [EXPERIMENT.md](EXPERIMENT.md),
[ML40_RESULTS.md](ML40_RESULTS.md) and the saved audits.

The selected WSPR/FT8/CW models are CPU histogram gradient-boosted regressors
trained for absolute SNR error. FT4 retains the reference model. Call sign IDs
are not learned input features in the deployed population prediction. Older
rankers and tuning candidates are retained as research history, not the main
prediction interface.

## Results and interpretation

- [Hourly model evaluation](HOURLY_RESULTS.md): 157,382 fresh WSPR hourly
  aggregates; hour/date-balanced MAE improved from 6.67 to 6.32 dB. Different
  sampling and weighting mean this score is not directly interchangeable with
  the path-balanced VOACAP score below.
- [1,000-path VOACAP comparison](VOACAP_1000_COMPARISON.md): 9,967 observed hourly
  medians from 31,840 reports; case-average MAE **5.01 dB GoCAP / 14.00 dB VOACAP**.
  GoCAP has lower error on **855/1,000** paths; no original 100-suite paths reused.
- [Original 100-path VOACAP comparison](VOACAP_COMPARISON.md): 968 observed hours;
  **4.90 / 14.37 dB** case-average MAE.
- Earlier work: [ML baseline](ML40_RESULTS.md), [500-case test](TEST500_RESULTS.md),
  [tuning](TUNING_RESULTS.md), [five-improvement experiment](V3_RESULTS.md).

The VOACAP tests use the sixth day of each 2025 month, 250 paths in each of four
distance strata for the expanded suite, fixed reported power and at least six
observed hours per case. Cases/configurations were frozen before scoring.
VOACAP is a monthly hourly climatological median, while truth is received signals
on specific days with unknown real antennas/noise. Twelve predeclared antenna,
noise and SSN settings are reported; none is selected after seeing errors.
Both models use final historical solar data, so these are retrospective
hindcasts, not archived operational forecast-vintage tests. The tests establish
neither contact probabilities nor accuracy in the unreported hours.

## What is in Git, and what stays local

| Versioned here | Excluded locally |
|---|---|
| Code, tests, exact dependency files and build script | `data/`: raw archives, Parquet feature samples, legacy pilot SQLite, training checkpoints and native source/build files |
| Compact model artifacts and checksums | `.venv*`, caches, runtime logs and local credentials |
| Reports, plots, frozen protocols, audits, main hourly/path CSVs and run checksums | Large intermediate rankings, per-observation prediction Parquets and full 12-scenario hourly CSVs |

At the survey snapshot, local data occupies about **61 GB**: roughly 58 GB of raw
archives and 2.5 GB of ML datasets/experiments, including saved native VOACAP runs.
Git is not a backup of those inputs. The 1,000-path suite's **24,000 primary
hourly comparisons** and all 1,000 path results are versioned under
[`results/ml40/voacap1000/`](results/ml40/voacap1000/); unreported actual values
remain blank. Excluded files can be regenerated with the corresponding scripts
or restored from a separate checksum-preserving archive backup.

## Tests and reproduction

```sh
.venv/bin/python -m unittest discover -v
```

The local full-data snapshot passes **48 tests**. A clean clone runs the synthetic
and inference-geometry checks and explicitly skips archive-backed checks whose
local data is absent. Restore the relevant archives to run those checks.
For the three GFZ-dependent checks only, a small index download is sufficient:

```sh
.venv/bin/python collect.py --source gfz
```

For a source-collection pilot, use `collect.py --source wspr` or `--source rbn`
with `--dates YYYY-MM-DD ...`; the original pilot pipeline is described in
[EXPERIMENT.md](EXPERIMENT.md). For the larger historical dataset, entry points
are `ml40.collect`, `ml40.harc`, `ml40.prepare --for-fit`, then `ml40.train fit`.
The baseline export also requires its separate evaluation and verification steps:
`ml40.train evaluate`, `ml40.verify`, then `ml40.export`. The original ranking
cohort is historical baseline research, not the hourly product objective.
See [ML40_PLAN.md](ML40_PLAN.md) for that version's frozen design.

HARC extraction uses a separate **Python 3.14** environment because h5py wheels
were unavailable for the original Python 3.15 setup:

```sh
python3.14 -m venv .venv-hdf
.venv-hdf/bin/python -m pip install -r requirements-hdf.txt
# Requires the authorized local Madrigal identity file.
.venv-hdf/bin/python -m ml40.harc
```

Daily HARC files can be several GB. The current collector caps each at 8 GB,
downloads bounded seasonal samples, filters 40m locally and records unavailable
files instead of fabricating data. Keep network collection sequential/cached;
more compute is not authorization to overload volunteer services.

Hourly-model experiments use `ml40.hourly_diagnose`, `ml40.hourly_collect`,
`ml40.hourly_harc` (HDF environment), `ml40.hourly_prepare_harc`,
`ml40.hourly_train`, `ml40.hourly_stress`, `ml40.hourly_evaluate`,
`ml40.hourly_verify` and `ml40.hourly_deploy_verify`, with prerequisites and
frozen artifact hashes. These are research pipeline stages, **not a single
fresh-clone retraining command**: baseline/internal model caches, strict-holdout
artifacts and exact feature samples must first be restored or regenerated.
`HOURLY_RESULTS.md` describes the final promoted model and original checks.

VOACAP is optional for inference. On macOS the native build requires Homebrew
`gcc`, `autoconf`, `automake`, plus `git`/`make`; run
`bash scripts/build_voacap.sh` after installing those tools. On other systems
install equivalent Fortran/build dependencies. The build script downloads the
pinned upstream source locally, retaining its license; no upstream engine source
or executable is redistributed in this repo. Each engine context uses its own
temporary run directory. Do not share that directory between workers.

For a **restored frozen benchmark**, select its size and follow its report:

```sh
export GOCAP_VOACAP_CASES=1000  # default is 100 when unset
.venv/bin/python -m ml40.voacap_collect
.venv/bin/python -m ml40.voacap_prepare
.venv/bin/python -m ml40.voacap_run
.venv/bin/python -m ml40.voacap_verify
.venv/bin/python -m ml40.voacap_verify_extra
.venv/bin/python -m ml40.voacap1000_report
unset GOCAP_VOACAP_CASES
```

Exact reproduction needs the frozen source caches (including SILSO), engine/API
pilot cache and matching native binary. Those inputs remain outside Git. A
fresh native build can have a different binary hash; upstream source revisions
can change data. The frozen protocol intentionally rejects mismatches. For a new
run, version the experiment paths/protocol and revalidate the native/API pilot
before scoring; do not overwrite the evidence of the old benchmark. The full
per-scenario CSV omitted from Git is generated by `ml40.voacap_run`.

## Expansion when better hardware is available

The priority is **more representative data and independent evaluation**, not
simply a larger model. Keep the deployed artifact and this `Survey` snapshot
unchanged; use a new branch plus versioned dataset, experiment directories,
protocol and artifact names. Current research stages contain fixed paths and
hashes, so update those explicitly rather than silently reusing their caches.

| Resource available | Useful next work | Required implementation changes/checks |
|---|---|---|
| More SSD capacity | Retain more years/months, full daily archives and raw checksummed backups. | Partition Parquet by source/band/year/month; retain source schema, query and provenance manifests; audit duplicates and location changes. Git still holds manifests/results, not bulk archives. |
| More RAM | Train on a much larger fraction of available positive observations. | Replace whole-dataset pandas concatenation with bounded Arrow/Parquet batches where necessary; measure peak RAM first. Current scikit-learn fitting still needs the selected training matrix in memory. |
| More CPU cores | Parallelize local feature preparation, candidate fits and independent VOACAP jobs. | Current code often limits native thread pools to 4; tune deliberately in a new experiment. Avoid worker × BLAS/OpenMP oversubscription. Each VOACAP process needs its own `Engine` context and temporary directory. Do not parallelize upstream requests indiscriminately. |
| A supported CUDA GPU | Test GPU tree regressors or a temporal/neural challenger after establishing a strong CPU baseline. | The deployed scikit-learn model does not automatically move to a GPU. A new backend, training/export path and comparison are required. For XGBoost, the documented starting point is `tree_method="hist", device="cuda"`; it is not a change to the deployed model. Check backend/version/platform support first. |
| More storage and compute together | Add genuinely new dates/years, rarer long paths, storms, polar paths and additional bands. | Freeze target, sampling and splits first; add band/frequency features or separate band-specific models and thresholds. Current inference is **40m only**. |

Implementation references: [scikit-learn histogram boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
and [XGBoost GPU support](https://xgboost.readthedocs.io/en/stable/gpu/index.html).
No hardware sizes or training-speed guarantees are inferred from this prototype.

For the next training experiment:

1. Choose new dates **before examining their reception values or errors**. The
   released 2025 tests, including both VOACAP suites, are now known benchmarks;
   reserve new untouched dates for final promotion. Expand temporal coverage
   across solar conditions before only increasing reports on existing days.
2. Collect/cache with source-aware limits. Keep historical reported power and
   decoder context. Preserve unknown antennas, power and noise; never make
   unreported slots negative labels. Add documented reference-station power,
   antenna/height/loss and receiver-noise measurements if physical calibration
   is the goal—hardware alone cannot supply those missing measurements.
3. Rebuild versioned features using only completed historical inputs. Separate
   point-fitting, model-selection and interval-calibration dates. Maintain
   held-out stations and geographic fields; add forecast-vintage solar inputs
   for an operational forecasting test instead of using final indices.
4. Train CPU/GPU challengers against the frozen hourly model. Tune only on
   validation data. Score **hourly SNR MAE, RMSE and bias**, 24 UTC-hour errors,
   interval coverage/width, and path/season/storm/source/mode subsets. Ranking
   alone cannot establish accurate hourly strength.
5. Freeze candidates/configurations, run the untouched final tests once, and
   independently reconstruct metrics from source evidence. Promote only a
   justified improvement without material per-mode/subgroup regressions.
   Export a compact model with dependency versions, checksum, training manifest
   and explicit limits; retain the previous artifact for rollback.

A scalable future architecture is checksummed raw object storage → partitioned
Parquet features → versioned training/evaluation jobs → compact inference model.
A classic relational database is optional; inference should continue to work
without querying a reception-history database.
