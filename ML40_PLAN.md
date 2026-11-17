# Next 40m ML experiment

Status: completed and independently verified. See [the results](ML40_RESULTS.md).
The first pilot remains intact in RESULTS.md and results/.

## Commands

```sh
.venv/bin/python -m ml40.collect
.venv-hdf/bin/python -m ml40.harc
.venv/bin/python -m ml40.ranking_cohort
.venv/bin/python -m ml40.prepare --for-fit
.venv/bin/python -m ml40.train fit
.venv/bin/python -m ml40.prepare
.venv/bin/python -m ml40.train evaluate
.venv/bin/python -m ml40.verify
.venv/bin/python -m ml40.export
.venv/bin/python -m ml40.report
```

Standalone inference uses the exported model, without the reception archives:

```sh
.venv/bin/python -m ml40.inference artifacts/gocap-40m.joblib JN76 FN31 2026-11-17T12:00Z --sfi 150 --kp 3 --ap 10
```

The `--for-fit` stage permits only reserved test HARC files to remain pending;
all training/validation/calibration inputs must be present. Fitting operates on
an independently hashed file containing no test rows. All final test data must
be present before evaluation. This allows data transfer and training to overlap
while preserving the evaluation boundary.

The deployment target is a compact trained function of four-character endpoints,
date/hour and solar scenario. Archives are training evidence; inference does not
require a spot-history database. Positive receptions only; missing reports stay
unknown. Absolute 100 W/dipole calibration remains a separate, unproven claim.

## Frozen sample and evaluation plan

The machine-readable plan is `results/ml40/plan.json`.

- 324 WSPR days: four monthly dates in 2010, 2013, 2016, 2019, 2022 and 2024,
  and three monthly dates in 2025. Sample each day by a 1/128 event-ID hash.
- 264 training days; twelve validation and twelve calibration days in 2024;
  36 final test days in 2025. Previously inspected 2025 day-15 dates are excluded.
- Eleven seasonal HARC days are selected for available PSK/RBN evidence. Each
  full archive is capped at 8 GB; missing or over-cap files must be recorded,
  never replaced based on observed model accuracy. Existing consent covers
  Madrigal identity submission; identity stays in an ignored local file.
- Training samples use station-pair/day balancing and retain source/mode.
- A separate outcome-blind ranking cohort on validation/calibration/test dates
  retains all reported receptions for sampled station-pair/day keys. It is never
  added to model training. See `results/ml40/ranking_plan.json`. Empty hours are
  still unknown. This supports both station-pair and square-pair hour ranking.
- Compare a boosted baseline, a station-adjusted boosted model and one small
  station-adjusted multilayer perceptron. Estimate station corrections using
  training data only, regularize them, and use zero correction for unseen keys.
- Choose model settings and neural epoch count on validation data. Use a separate
  calibration split for interval adjustment. Freeze settings before releasing
  final test scores.
- Evaluate later dates, callsigns excluded in both roles, and excluded geographic
  Maidenhead fields. Primary product check: ranking observed hours. Also retain
  SNR error, prediction-interval coverage/width, and per-mode/source diagnostics.

The implementation uses a 64/32-neuron scikit-learn MLP challenger and 180-step
histogram boosted trees. Station residuals come from date-held-out predictions
within training, followed by regularized two-way transmitter/receiver offsets.
Unsupported station keys receive zero offsets. The neural checkpoint is selected
by validation balanced MAE. Candidate selection uses validation station-pair hour
ranking when at least 20 cases exist, otherwise balanced MAE. Interval adjustment
uses per-source/mode absolute residuals on the independent calibration split;
future temporal coverage must be measured rather than assumed.

## Important interpretation limits

Reported power, antennas and local noise are not interchangeable. Station offsets
are approximate nuisance terms, not recovered physical antenna/noise measurements.
Unknown power is never replaced by a claimed 100 W observation. Mode thresholds
are applied as theoretical margins, not labels or artificial missing receptions.

Selection effects remain: detected signals differ from all transmitted signals.
Quantiles and calibrated intervals initially describe the received population.
Solar inputs are conditional scenarios; historical tests use final index series,
not operational forecast vintages.

Historical WSPR code 0 is retained for pre-2020 records with an explicit legacy
decoder context. Pre-2023 code 1 is separately marked because historical mode
metadata can mix protocols. These observations are not claimed to have a single
known hard decoding threshold. The modern ordinary-mode profile remains explicit.
