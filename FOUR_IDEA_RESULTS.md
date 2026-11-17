# Four experiments in hourly 40 m WSPR prediction

**The strongest candidate adds an empirical path correction to the tuned March model.** On six fresh dates, the tuned March parent scored 5.905 dB mean absolute hourly error, the corrected scenario model 5.880 dB, and the corrected model with lagged D-RAP maps 5.880 dB. Relative to the parent, the map-enabled version improved by 0.025 dB (95% paired date-bootstrap interval 0.006–0.048 dB improvement). That is about 0.4%. The two corrected versions differed by just 0.00024 dB on these dates. Both accept four-character squares. The experiment is scoped to received WSPR signals on 40 m.

## Three tests

| Frozen model and test | Primary, 13 dates | Later confirmation, 6 dates | March transfer, 6 more dates |
|---|---:|---:|---:|
| Measured hourly medians | 162,040 | 83,558 | 83,917 |
| Existing default GoCAP, dB | 6.362 | 5.964 | 5.958 |
| Previously tuned March, dB | 6.286 | 5.915 | 5.905 |
| Four-idea scenario on default, dB | 6.317 | 5.943 | — |
| Four-idea nowcast on default, dB | 6.307 | 5.939 | 5.950 |
| Corrected March scenario, dB | — | — | 5.8800 |
| Corrected March nowcast, dB | — | — | 5.8798 |

Lower is better. Each observation is the median for a station pair, directed four-character path, reported power and UTC hour. Errors are averaged within each date-hour, then equally across date-hours. Unreported paths and hours remain unknown. The first cohort deliberately includes dates with noise coverage and solar activity. Comparing models within a column keeps the reception sample identical.

1. **Primary test.** Sample whole directed station-pair days with a 1/48 deterministic hash, preserving all valid modern WSPR decodes for each sampled pair. Include 7.0–7.3 MHz, 0–60 dBm reported power, 0–15,000 km, and deduplicate matching station/time/frequency reports. Fit the corrections on 140,962 hours across 17 dates in 2023 and 2025. Fit combination weights on 46,649 hours across five later 2025 dates. A hash of receiver call reserves approximately one fifth of receiver identities from both stages. Test all receivers on 13 subsequent dates in November 2025 through February 2026. Against the default, the combined nowcast improved by 0.054 dB (95% paired interval 0.042–0.067); it had lower error on 12 of 13 dates. Two pairs of adjacent dates were clustered together for uncertainty.
2. **Confirmation.** Freeze the same bundle and test the 27th of each month from March through August 2026. The combined nowcast improved over default on all six dates, averaging 0.025 dB improvement (95% date interval 0.011–0.040). The 40 m receiver-noise mirror had ended by these dates; the forecasts use their fallback. The previously tuned March comparator performed better than the default-based challengers in both cohorts.
3. **Transfer to March.** Fit the four-character geometry correction and D-RAP term to errors of the frozen, stronger March parent. Exclude every correction-training date that appeared in that parent's point training. This left 121,851 received hours before fitting the components. Select their strength on the original validation split, freeze the new bundle, and test the 13th of each month from March through August 2026. These six days had been excluded from both previous comparisons. This third test answers a follow-up suggested by the first two results; it is the independent evidence for the March transfer. The date-bootstrap intervals summarize only the sampled dates.

## Contributions from the four experiments

| Component on original default | Primary hourly MAE, dB | Interpretation |
|---|---:|---|
| Default | 6.3618 | Common reference |
| Centre-of-square geometry correction | 6.3155 | Most of the gain is available from existing input coordinates |
| Six-character training, averaged back to four | 6.3168 | Comparable to centre geometry. Confirmation slightly favored averaging, 5.9431 versus 5.9463 |
| Typical square and hour noise correction | 6.3615 | Very small aggregate effect |
| Previous-hour receiver noise | 6.3615 | Validation assigned the extra live term zero weight |
| Empirical power-step slope | 6.3618 | Validation assigned zero weight; applying the full correction gave 6.6801 |
| Lagged D-RAP absorption | 6.3514 | Larger gain in flare hours |
| Selected combined nowcast | 6.3074 | Geometry, a small typical-noise term, and D-RAP |

**Receiver noise.** WsprDaemon reports instrument-relative RMS levels. We compared each instrument's hourly median to its own preceding-week median. The experiment required valid non-overloaded measurements (unknown overload accepted), three readings per instrument-hour, and 24 historical hourly summaries per instrument. We used the previous completed hour and pooled available channels by median at the reporting site and square. A square/hour fallback was learned only from the correction-training receivers. Time-matched noise inside the observed hour was used separately to study receiver disagreement.

There were 6,697 matches with the same transmitter, timestamp, reported power and receiving square, different receiver calls, and at most 5 Hz frequency disagreement. Median absolute SNR disagreement was 6 dB; among 1,894 matches in the same reported six-character subsquare it was 2 dB. Those subgroups differ in receiver composition. For 767 matches with contemporaneous relative noise on both sides, the SNR difference and noise difference had correlation −0.213. The prior-hour correlation was −0.100 on 798 matches. Among 37,855 test observations from 829 receiver calls unseen by both the frozen base WSPR regressor and the new correction fit, combined MAE fell from 6.392 to 6.320 dB. Only 20 of those calls had usable lagged noise. These results provide limited evidence for forecasting an individual receiver's noise.

**Finer geometry.** The matched geometry regressions used the same 91,317 training rows with six-character locators at both ends. They fit the base prediction error with smooth functions of distance, midpoint illumination and bearing. One uses square centres, the other subsquare centres. At prediction time, the latter averages over 16 endpoint combinations drawn from training occupancy in the two squares, with quadrant samples for unseen squares. Neither operational candidate receives actual subsquares or calls. On 49,396 short-path hours with both precise locators, the default, centre correction and averaged correction scored 7.518, 7.501 and 7.504 dB respectively. The matched measurement result above does not establish that supplying finer locations improves a generic user's forecast.

**Power changes.** Match successive positive reports at the same stations and locators, within the same UTC hour, 2–12 minutes apart, allowing 3–20 dB reported power changes and up to 100 Hz frequency change. To obtain repeated observations we also collected all receivers for eight transmitters identified from the training period. The slope is the median across eligible transmitters' median SNR-change/power-change ratios. It fitted 0.357 dB/dB on 7,860 eligible training pairs. The correction anchors the default at 30 dBm and scales from there. Tested on 161 power-change pairs from 36 transmitters, the held-out median ratio was 0.30; mean absolute **change** error was 7.126 dB versus 6.225 for the existing default power response. The new response also lost on strong-signal and multi-receiver subsets. Conditioning on two successful decodes and using self-reported transmitter power limit physical interpretation.

**Flare absorption.** We used NOAA's actual archived global 1 dB highest-affected-frequency maps. Each hourly prediction uses the latest available map 5–15 minutes before that hour, sampled at both endpoints, the midpoint and the 25%/75% points. The normalized frequency raised to the 1.5 power provides a fitted loss proxy at 7.04 MHz. Mean and maximum are offered to a loss-only regression. This is an empirical path correction; it does not convert the map into calibrated end-to-end attenuation. Missing maps, missing cells, and the first hour of a sampled UTC day revert to the baseline absorption correction of zero. Each map already includes modeled solar illumination and polar effects.

The D-RAP term alone reduced MAE from 6.655 to 6.613 dB in 27,335 observations during hours labeled by NOAA with minor or stronger X-ray flux. Quiet-hour error improved from 6.301 to 6.298. Post-hour message labels define scoring subsets only. For 673 flare observations matched to a quiet observation at the same stations, grids, UTC hour and power within 35 days, observed SNR was a median 1 dB lower. Error predicting the paired difference improved from 5.511 to 5.444 dB with D-RAP. For 144 matched observations in the first two hours following flare hours, change error moved from 5.022 to 5.015 dB. Predictions were issued for the coming hour using prior map validity times. Actual publication latencies were unavailable.

## Query the experimental model

Use the March scenario correction for an arbitrary date. The example uses an illustrative 5 W transmitter and steady solar history.

```bash
.venv/bin/python -m ml40.four_query JN76 FN31 2026-05-13T00:00Z --sfi 150 --kp 3 --ap 10 --power 5 --hours 24 --model march_scenario
```

For a one-hour map-assisted estimate choose `--model march_nowcast --hours 1 --drap-file GLOBAL.txt` and supply a tabular NOAA global map valid 5–15 minutes before that hour. Without a map the query omits the absorption contribution. Each row shows the corrected SNR, the parent prediction and whether a usable path map was available. Beyond 15,000 km it falls back to the parent estimate. The existing default command and the other mode profiles retain their previous behavior.

## Reproduction

The existing Python environment, baseline artifacts, GFZ solar history, and March training-date archive are required. These commands rebuild the local archives, fit the models, score them, reconstruct the held-out medians, and check the comparisons. Confirmation data acquisition intentionally follows model freezing.

```bash
.venv/bin/python -m ml40.four_data
.venv/bin/python -m ml40.four_drap
.venv/bin/python -m ml40.four_prepare
.venv/bin/python -m ml40.four_steps
.venv/bin/python -m ml40.four_run
.venv/bin/python -m ml40.four_diagnostics
.venv/bin/python -m ml40.four_confirmation
.venv/bin/python -m ml40.four_transfer train
.venv/bin/python -m ml40.four_transfer test
.venv/bin/python -m ml40.four_verify
.venv/bin/python -m unittest discover -q
```

The [component scores and intervals](results/ml40/four/metrics.csv), [paired diagnostics](results/ml40/four/paired_diagnostics.json), [subgroup scores](results/ml40/four/subgroups.csv), [confirmation](results/ml40/four/confirmation.json), [March transfer](results/ml40/four/transfer_summary.json) and [verified daily counts](results/ml40/four/verified_dates.csv) accompany both model bundles. Source queries, map URLs and model hashes are in the same results directory.

WSPRNet contributors supplied the receptions through [WSPR Live](https://wspr.live/). Receiver-noise measurements come from [WsprDaemon](https://www.wsprdaemon.org/fields) via the WSPR Live mirror. NOAA SWPC produced the maps and NCEI maintains the [D-RAP archive](https://www.ncei.noaa.gov/products/space-weather/ionospheric-program/d-region-absorption-prediction). Solar and geomagnetic inputs retain the [project's existing source credits](README.md#data-sources-and-attribution).
