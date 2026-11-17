# Five experiments in hourly 40 m WSPR prediction

**Decision. Keep the current GoCAP default.** The full combination scored 6.007 dB mean absolute error against 5.967 dB for current GoCAP on 10 held-out 2026 dates. The difference was +0.040 dB (95% date-bootstrap interval −0.004 to +0.081). Training on consecutive-hour changes improved the daily curve relative to the same model trained on levels alone. Sampling complete station-pair days gave a more stable hourly reference. Neither result established an improvement in the primary hourly level score.

## Design

The base was the frozen current GoCAP model. The ten WSPR challengers learned corrections to its predictions. This makes each new source compete against information GoCAP already has. The corrections use a fixed nonlinear feature representation (192 random Fourier features and standardized path, power, time, solar and base-SNR inputs), ridge regression and, where indicated, physically motivated interaction terms. All settings are in [the model definitions](ml40/five_models.py). The original March and tuned March regressors were additional frozen comparators.

- **Fit:** 28,681 received station-pair hours on the 9th and 17th of every month of 2025. **Validation:** 14,763 hours on the 23rd of each month of 2025. Validation selected the arbitrary-date and nowcast candidates by hourly level MAE. **Test:** the 19th of January through October 2026. Ten future dates span several seasons. A single date per month limits the precision of seasonal or storm-specific conclusions.
- From WSPR Live we sampled whole directed station-pair days by a deterministic 1/64 hash. We kept all valid positive modern WSPR reports on the selected paths. For scoring, each date contributes 60 station-pair/grid/power cases in each of four distance ranges (100–1,000, 1,000–3,000, 3,000–6,000 and 6,000–15,000 km). Case selection was seeded and independent of SNR, predictions and errors. Reported power was 20–50 dBm. Paths with one reception were eligible. Station or square pairs may recur on different dates.
- The test contains **2,400 path-days, 11,710 hourly medians and 34,800 reports**. Of those hours, 7,356 contain multiple reports. Consecutive-hour change scores use 7,584 same-case pairs exactly one hour apart. A missing reception contributes to neither the level score nor a change label.
- **Hourly MAE** averages absolute errors within each UTC date-hour, then across 240 hours. **Path MAE** gives equal weight to each path-day. **Change MAE** measures error in the signed change in dB between two measured consecutive hours, averaged within each date and then over dates. Lower values are better. Paired 95% intervals resample the ten dates 10,000 times; the intervals are descriptive across multiple comparisons.

## Held-out measurements

| Model | Hourly MAE, dB | Path MAE, dB | Change MAE, dB |
|---|---:|---:|---:|
| Current GoCAP | 5.967 | 5.712 | 3.132 |
| Original March | 5.977 | 5.738 | 3.130 |
| Previously tuned March | **5.948** | 5.668 | **3.107** |
| Train on one random report per hour | 5.985 | 5.532 | 3.157 |
| Train on full hourly median | 5.990 | 5.521 | 3.158 |
| Median plus consecutive-hour objective | 5.997 | **5.508** | 3.138 |
| Median plus short-path F2 climatology | 5.987 | 5.522 | 3.155 |
| F2 climatology plus Dourbes sounding | 6.000 | 5.534 | 3.160 |
| Median plus prior-hour neighboring paths | 5.985 | 5.513 | 3.160 |
| Median plus relative VOACAP curve | 5.990 | 5.524 | 3.155 |
| Static combination (median, change, F2, VOACAP) | 6.000 | 5.531 | 3.128 |
| Static combination plus neighboring paths | 5.997 | 5.526 | 3.127 |
| Full combination including sounding | 6.007 | 5.534 | 3.130 |

Validation selected the one-report model for arbitrary dates and the neighboring-path model for nowcasts. Their hourly test differences against current GoCAP were +0.018 dB (95% interval −0.010 to +0.052) and +0.018 dB (−0.012 to +0.052). The tuned March comparator's −0.020 dB advantage on this sample had an interval from −0.093 to +0.057. All corrections' validation-derived nominal 80% intervals covered about 77% of held-out hours. These are received-population intervals.

## What each idea contributed

1. **Sampling.** For exactly the same test hours and frozen current GoCAP predictions, evaluating against one randomly selected report gave 6.189 dB average hourly error over 20 draws (6.154–6.216). Evaluating against each hour's median gave 5.967 dB. This 0.222 dB difference quantifies sensitivity to the reference measurement. Separately, changing the *training target* from one reception to the median gave +0.005 dB hourly error (−0.007 to +0.018). Use complete path-days to retain fading information and reproducible hourly targets.
2. **Hour-to-hour change.** The difference-aware fit adds a loss on adjacent prediction differences, normalized by pair count with weight two. It needs no previous observation to make a prediction. Against its otherwise identical median-only counterpart, change MAE improved by 0.020 dB (95% interval 0.014–0.025 improvement). Hourly level error increased by 0.007 dB (−0.015 to +0.027).
3. **Short-path ionosphere.** For paths up to 1,000 km, PyIRI calculates midpoint foF2 and F2 peak height at the hour centre with the previous day's F10.7. A spherical-Earth, single-hop secant approximation calculates the critical frequency required for 7.04 MHz. This is an approximate path feature. Adding the resulting frequency margin improved overall hourly error by 0.003 dB (−0.007 to +0.001 for the signed difference). We also tested Dourbes foF2 within 800 km of the midpoint, using the latest sounding strictly before the hour, at most 90 minutes old, with automatic confidence 60–100. For the 2,257 supported hours, adding the measured-minus-climatological frequency raised pooled error from 7.565 to 7.625 dB. One station tests regional transfer only.
4. **Live neighboring paths.** For each target hour, the model used measured minus base-predicted SNR from other paths during the immediately preceding completed hour. Both corresponding endpoints had to be within 1,200 km. The target station calls and exact directed square pair were excluded. We required three independent links, capped support at twelve, allowed each call once, distance-weighted the residuals and clipped outliers. There was support on 9,374 hours. Adding this input to the median model improved hourly error by 0.005 dB (signed interval −0.014 to +0.002). This replay uses observation timestamps; archive ingestion times and operational delays remain unverified.
5. **VOACAP diurnal shape.** The native 7.04 MHz engine produced a curve per square pair and month. We used the departure from the daily median and its previous-hour slope, with a fixed reference 5 W, 10 m dipoles, noise parameter 150 and SSN 100. Fixed SSN isolates the geometrical and seasonal shape and works for arbitrary dates. Changing the feature set to include this curve shifted hourly MAE by −0.0002 dB (−0.009 to +0.009). A solar-dependent or alternative-antenna curve could behave differently.

**Why some averages improved.** The selected nowcast reduced equal-path MAE by 0.198 dB against current GoCAP (95% interval 0.116–0.290 improvement). A descriptive split showed pooled MAE fell from 5.662 to 5.338 dB among cases with 1–2 measured hours, but rose from 6.013 to 6.171 dB among cases with seven or more. Often-reported paths contribute more observations to each hourly score. This split uses realized reporting density for analysis only. It cannot identify a band closure or select a model at forecast time.

## Query the saved models

The pilot supports WSPR and four-character grid squares. The operational default and other mode profiles retain their existing interfaces. The following runs the validation-selected arbitrary-date model with synthetic solar inputs and steady recent solar history.

```bash
.venv/bin/python -m ml40.five_query JN76 FN31 2026-03-19T00:00Z --sfi 150 --kp 3 --ap 10 --power 5 --hours 24
```

Choose an explicit model with `--model` (see `--help`). For a one-hour `--model nowcast`, supply `--hours 1 --donor-file prior.csv --tx-call CALL1 --rx-call CALL2`. Supply completed WSPR station-pair medians with columns `hour,tx_sign,rx_sign,tx_grid,rx_grid,snr2500,base`, where base is the matching frozen GoCAP prediction for the donor's own hour and power. The query filters the file to the preceding hour. A `--model combined_live` query can also read `--sonde-file dourbes.csv` containing UTC `time,confidence,fo` with Dourbes foF2 in MHz. Measured inputs are optional. The output includes their support where used. Solar and WSPR inputs in the study are retrospective archive data.

## Reproduction and evidence

Build the native engine using the existing [VOACAP instructions](VOACAP_COMPARISON.md). Use the repository environment and its historical GFZ data. These optional packages are needed only for preparing or querying F2 features; the installation below isolates them from the default environment.

```bash
.venv/bin/python -m pip install --target data/ml40/five/python --no-deps PyIRI==0.1.7 fortranformat==2.0.3
export PYTHONPATH="$PWD/data/ml40/five/python"
.venv/bin/python -m ml40.five_data
.venv/bin/python -m ml40.five_sonde
.venv/bin/python -m ml40.five_prepare
.venv/bin/python -m ml40.five_evaluate
.venv/bin/python -m ml40.five_verify
.venv/bin/python -m unittest discover -q
```

The [validation selection and hashes](results/ml40/five/selection.json), [per-date scores](results/ml40/five/daily_metrics.csv), [paired comparisons](results/ml40/five/summary.json), [subgroup scores](results/ml40/five/subgroups.csv) and [per-hour predictions](results/ml40/five/test_predictions.parquet) are saved alongside the [model bundle](artifacts/gocap-40m-five.joblib).

**Credits.** Reception reports were contributed by WSPRNet stations and sampled from [WSPR Live](https://wspr.live/). Solar and geomagnetic inputs follow the [project's source credits](README.md#data-sources-and-attribution). F2 climatology uses [PyIRI](https://doi.org/10.1029/2023SW003739). Dourbes DB049 ionosonde observations are credited to the Dourbes data provider and the Lowell GIRO Data Center, distributed under CC BY-NC-SA 4.0 via [GIRO FastChar](https://giro.uml.edu/didbase/scaled.php). Native VOACAP uses the [pinned engine and setup](VOACAP_COMPARISON.md). No prospective station availability, absolute antenna gain or receiver-noise calibration was assumed.
