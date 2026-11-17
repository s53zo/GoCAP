# Tuning the 40 m March model

**The tuned model improved the existing 500-path March benchmark by 0.187 dB over its parent. The fresh March 2026 test showed no measurable gain.** Current GoCAP remains the default.

| Mean absolute hourly SNR error, dB | Original March | Tuned March | Current default |
|---|---:|---:|---:|
| Fresh March 2026, equal weight per UTC hour and date | 6.380 | 6.385 | 6.372 |
| Known March 2025, equal weight per path case | 5.063 | 4.876 | 5.007 |

Fresh paired error differences (tuned minus comparator) were +0.004 dB against the March parent (95% date-bootstrap interval −0.016 to +0.024), and +0.013 dB against the default (−0.006 to +0.032). The nominal 80% signal interval covered 76.9% of fresh receptions for the tuned model, 79.4% for the parent and 79.0% for the default. The tuned intervals were narrower. Previously computed VOACAP error on the March 2025 path benchmark was 13.491 dB.

## Training and selection

The experiment keeps the training data, five-point path geometry, solar inputs, mode and decoder context, and hourly SNR target from the [March-expanded model](MARCH_MODEL_RESULTS.md). Each candidate uses the same 2,950,381 training rows. The search compares the original configuration (240 boosting rounds, 31 leaves, learning rate 0.05, minimum leaf size 100) with 120, 240 and 400 rounds of a higher-capacity configuration (63 leaves, learning rate 0.08, minimum leaf size 80). All candidates use absolute-error loss and L2 regularization of 10. Changing several parameters together measures their combined effect.

Selection minimizes WSPR mean absolute error on 12 reserved 2024 validation dates, giving each UTC hour on each date equal weight. CW, FT8 and FT4 must each stay within 5% of their original validation error. The winner's 80% intervals use a separate calibration split. The validation sample covers several seasons. The fresh March test measures whether the chosen configuration transfers to March.

WSPR validation error fell from 6.383 dB for the parent to 6.297, 6.147 and 6.068 dB at 120, 240 and 400 rounds. The 400-round model qualified on all three other modes. On the previously scored 2025 other-mode audit its CW error improved by 0.040 dB over the parent, while FT4 and FT8 errors increased by 0.032 and 0.006 dB. PHONE margins remain theoretical.

## Evaluation

The primary test uses eight dates in March 2026 (2, 6, 10, 14, 18, 22, 26 and 30). A deterministic 1/128 sample of 40 m WSPR Live reception reports yields 135,019 hourly measurements from 138,166 positive reports across 25,615 directed four-character grid pairs. For each UTC date and hour we average absolute error over measured station-pair signals, then average the 192 hourly errors. Paired uncertainty intervals resample whole dates 10,000 times. Eight dates support a March-specific comparison with limited precision across conditions. The hindcast uses completed historical solar indices.

The secondary test replays the [March 2025 path comparison](MARCH2025_VOACAP_COMPARISON.md) using its frozen observations, path weights and VOACAP predictions. Its 500 paths each receive equal weight regardless of observed hours. Differences in cohort and weighting affect the *level* of error across the two tests. Both tests score positive observed signals only.

## Reproduction

Run from the repository root after building the [March training inputs and encoded matrix](MARCH_MODEL_RESULTS.md). The replay also needs the March 2025 case features and the archived other-mode audit.

```bash
.venv/bin/python -m ml40.march_tune
.venv/bin/python -m ml40.march_tune_test collect
.venv/bin/python -m ml40.march_tune_test score
```

Query the experimental model directly with the usual hourly interface.

```bash
.venv/bin/python -m ml40.hourly_query artifacts/gocap-40m-shared-march-tuned.joblib JN76 FN31 2026-03-02T12:00Z --sfi 150 --kp 3 --ap 10 --power-watts 5 --hours 24
```

The [validation scores](results/ml40/march_tune/selection.json), [WSPR source queries and hashes](results/ml40/march_tune/sources.json), [errors for each UTC hour and date](results/ml40/march_tune/fresh_hourly_errors.csv) and [evaluation summary](results/ml40/march_tune/summary.json) provide the inputs for checking the comparison.
