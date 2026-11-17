# 40m positive-reception feasibility experiment

Status: feasibility pilot completed. See [RESULTS.md](RESULTS.md) for measured
performance and calibration limits. This does not mark the full propagation
prediction product complete.

## Contract

- Four-character Maidenhead endpoint squares; retain original locations and direction.
- WSPR, RBN and PSK Reporter; successful receptions only. Absence stays unknown.
- Hourly SNR estimates conditional on successful reception, with source/mode retained.
- Normalize reference bandwidth to 2500 Hz (CW Skimmer 500 Hz reports: subtract 6.9897 dB).
- Preserve reported power; unknown power is not assumed to be 100 W.
- Reference scenario: 100 W, dipoles at 10 m; absolute calibration remains unverified.
- WSPR/FT8/FT4 nominal thresholds: -31/-21/-17.5 dB. CW/SSB require explicit quality profiles.
- SFI, Kp/ap history and daily Ap. Forecast evaluation must not use future index values.
- Evaluate later periods and stations excluded from training. Compare historical baseline and scikit-learn.

## Sources and access

- https://wspr.live/ documents bounded time/band queries and research-only access with freely accessible results; do not use this service for commercial work.
- https://www.wsprnet.org/archive/ is the primary monthly archive alternative.
- https://data.reversebeacon.net/rbn_history/ contains daily RBN archives.
- https://hamsci.org/data identifies historical mixed-network data on CEDAR Madrigal (instrument 8308). Source-specific coverage and location provenance need inspection.
- https://pskreporter.info/pskdev.html documents reception fields. Transmitter power is not a documented field.
- https://kp.gfz.de/en/about-kp defines Kp, ap and Ap.

## Initial checks

2026-11-16: workspace was empty. Created isolated Python environment.
Verified WSPR Live access with five actual 40m records from 2024-01-01 00:00–00:10 UTC. Records include callsigns, locators, SNR, reported power, frequency, software and mode code. This verifies access/schema only; five rows do not establish data quality or model performance.

## Completed pilot steps

1. Cache bounded, reproducibly sampled 40m observations across hours and seasons; preserve exact queries and checksums.
2. Verify mode-code mapping before treating WSPR Live rows as WSPR (database also carries FST4W).
3. Inspect RBN and historical PSK records; do not substitute country centres for four-character locations.
4. Audit duplicates, locator validity, power distributions, mode thresholds and station concentration.
5. Join timestamp-correct geomagnetic and solar data.
6. Train historical and regression baselines with temporal and station holdouts; publish observed error and coverage limitations.

All six steps have now been exercised on bounded samples. The SQLite database
contains 437,327 hourly reception aggregates (including 925 live schema-probe
aggregates excluded from training). WSPR uses 16 sampled days across 2023–2025;
historical PSK/RBN checks use January 15 in 2024 and 2025. Four measured modes
have temporal and unseen-callsign evaluations. PHONE remains a theoretical
threshold profile, with no direct PHONE validation.

Independent reconciliation and eight targeted invariant tests passed. The
remaining work is a broader modelling phase: more independent days, station
and receiver-noise calibration, detection-aware measurement treatment, and
validation beyond the observed-reception population.
