"""Evidence-backed readout for the all-year March data expansion."""
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .march_plan import ROOT,OUT,ARTIFACT,sha

def report():
    prep=json.loads((OUT/'preparation.json').read_text());run=json.loads((OUT/'training_run.json').read_text());s=json.loads((OUT/'summary.json').read_text())
    coverage=pd.read_csv(OUT/'coverage_by_year.csv');hourly=pd.read_csv(OUT/'fresh_march_hourly_errors.csv');mode=pd.read_csv(OUT/'known_mode_audit.csv')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    axes[0].bar(coverage.year,coverage.hourly_rows,color='#1769aa');axes[0].set_xticks(coverage.year,[str(y) for y in coverage.year],rotation=60);axes[0].set_ylabel('Sampled positive hourly aggregates');axes[0].set_title('March coverage in every year, 2010–2024')
    byhour=hourly.groupby('utc_hour')[['current_mae_db','shared_parent_mae_db','march_mae_db']].mean()
    for name,label,color,ls in [('current','Current GoCAP','#7b498b','--'),('shared_parent','Shared parent','#777777',':'),('march','March-expanded shared','#1769aa','-')]:
        axes[1].plot(byhour.index,byhour[name+'_mae_db'],label=label,color=color,ls=ls,lw=2)
    axes[1].set_xticks([0,6,12,18,23]);axes[1].set_xlabel('UTC hour');axes[1].set_ylabel('Hour/date mean absolute SNR error (dB)');axes[1].set_ylim(bottom=0);axes[1].legend(fontsize=9);axes[1].set_title('Six new March 2025 dates')
    fig.tight_layout();fig.savefig(OUT/'march_expansion.png',dpi=160);plt.close(fig)
    result=s['fresh_march_results'];parent=s['march_minus_shared_parent'];current=s['march_minus_current']
    verdict='The new March dates show lower hourly error than the shared parent.' if parent['date_bootstrap95_db'][1]<0 else 'The fresh March comparison does not establish lower error than the shared parent.'
    rows='\n'.join(f'| {name} | {result[key]["hour_date_mae_db"]:.3f} dB | {result[key]["mae_db"]:.3f} dB | {result[key]["rmse_db"]:.3f} dB | {result[key]["bias_db"]:.3f} dB |' for key,name in [('current','Current GoCAP'),('shared_parent','Shared + five points parent'),('march','March-expanded shared')])
    years='\n'.join(f'| {int(r.year)} | {int(r.requested_days)} | {int(r.available_days)} | {int(r.raw_reports):,} | {int(r.hourly_rows):,} |' for r in coverage.itertuples())
    other=[]
    for m in ['FT8','FT4','CW']:
        g=mode[mode['mode'].eq(m)].set_index('model');other.append(f'| {m} | {int(g.loc["march","rows"]):,} | {g.loc["current","hour_date_mae_db"]:.3f} | {g.loc["shared_parent","hour_date_mae_db"]:.3f} | {g.loc["march","hour_date_mae_db"]:.3f} |')
    other='\n'.join(other);known=s['known500_case_mae_db']
    validation_macro={name:float(np.mean([v['hour_date_mae_db'] for v in scores.values()])) for name,scores in run['validation'].items()}
    text=f'''# March expansion: every year from 2010 through 2024

**A separate model has been trained with March data from all 15 years.** The expansion adds **{prep['added_unique_march_rows']:,} unique hourly WSPR records** to the existing multi-mode training set, for **{prep['expanded_training_rows']:,} training records across {prep['expanded_training_days']} days**. {verdict} The existing deployed model is retained; this new artifact is experimental.

## Exactly what was added

All **465 calendar-day windows** were requested: March 1–31 of every year, 2010–2024 inclusive. **{prep['available_march_days']} days** contain usable positive reception aggregates. Collection uses the same deterministic **1-in-128 event-hash sample** as the earlier dataset. This is full calendar-month coverage, not a download of every raw reception in those months.

March2020 `code=0` archive reports initially failed the inherited cutoff. Before fitting, an [explicit amendment](results/ml40/march40/legacy_mode_amendment.json) retained them as `legacy_code0`, with decoder/mode uncertainty preserved. This does not certify a modern decoding threshold. [WSPR Live documents the historical/current code mappings](https://wspr.live/), and the [WSJT developers announced the first public FST4/FST4W release candidate in September2020](https://sourceforge.net/p/wsjt/mailman/message/37117193/), after the added March window. No prior artifacts or datasets were modified.

The augmentation is WSPR data. Existing FT8, FT4 and CW training records are retained; no additional full-month PSK/RBN archive or PHONE measurements are claimed. The final training counts are WSPR **{prep['training_by_mode']['WSPR']:,}**, FT8 **{prep['training_by_mode']['FT8']:,}**, FT4 **{prep['training_by_mode']['FT4']:,}**, CW **{prep['training_by_mode']['CW']:,}**.

Original observations are preserved exactly once. **{prep['duplicate_original_rows_removed']:,} overlapping March aggregates** were removed from the addition after checking their SNR/report counts against the original samples. Existing validation and calibration dates stay outside fitting, including **2024-03-08 and 2024-03-22**. Consequently those two days are collected but not used to fit the point model. **No 2025 reception data enters training.** Empty or unreported slots remain unknown, never closed-path labels.

| Year | Requested March days | Available sampled days | Raw sampled reports | Positive hourly aggregates |
|---|---:|---:|---:|---:|
{years}

The per-year aggregate counts include reserved March validation/calibration windows and duplicates before the fitting-only merge. [Every requested date and its role](results/ml40/march40/march_coverage.csv) and [the complete final training-date manifest](results/ml40/march40/training_dates.csv) distinguish those populations.

## Model and memory handling

The fixed shared-parent specification is unchanged: one pooled source/mode-aware histogram gradient-boosted regressor, absolute-error loss, 240 iterations, 31 leaves, minimum leaf size 100, L2=10 and learning rate 0.05. It retains all 45 explicit features at 0%, 25%, 50%, 75% and 100% of the path, plus the existing nine-point summaries and completed solar history. There was **no new hyperparameter search or test-driven retuning**; the experiment tests a data expansion. On the unchanged validation partition, equal-mode hour/date MAE is **{validation_macro["march"]:.3f} dB expanded**, **{validation_macro["shared_parent"]:.3f} dB parent**, and **{validation_macro["current"]:.3f} dB current**. The expanded model is not an overall validation winner.

On this 16 GB machine, features are encoded in batches into disk-backed arrays. Training medians are computed over the complete training matrix one feature at a time, not estimated from a small batch. The exported encoder retains only its fitted statistics and categories; inference needs no matrix files or reception archive. Peak process RSS recorded during training was approximately **{run['peak_process_rss_bytes_macos']/1024**3:.2f} GiB**. The compact training DataFrame occupies approximately **{prep['compact_dataframe_bytes']/1024**2:.0f} MiB**, excluding matrices and learner workspace.

New artifact: [`gocap-40m-shared-march.joblib`](artifacts/gocap-40m-shared-march.joblib), **{ARTIFACT.stat().st_size:,} bytes**; SHA256 `{sha(ARTIFACT)}`. More training data does not require a larger number of trees when the architecture is fixed.

## Fresh March evaluation

Six dates were reserved before collection/fitting: **2025-03-09, 12, 17, 20, 25 and 28**. They were not in the original or shared-model comparison suites. The artifact was frozen before scoring their **{s['fresh_march_rows']:,} observed hourly records**. The primary metric gives equal weight to each observed UTC hour/date cell; lower error is better. These are received-signal measurements, not a test of unknown-hour band-open probabilities.

| Model | Hour/date MAE (primary) | Pooled MAE | RMSE | Signed bias |
|---|---:|---:|---:|---:|
{rows}

March-expanded minus shared parent: **{parent['difference_db']:.3f} dB**, 95% date-cluster bootstrap **[{parent['date_bootstrap95_db'][0]:.3f}, {parent['date_bootstrap95_db'][1]:.3f}] dB**. March-expanded minus current GoCAP: **{current['difference_db']:.3f} dB**, interval **[{current['date_bootstrap95_db'][0]:.3f}, {current['date_bootstrap95_db'][1]:.3f}] dB**. Only six date clusters are available; these intervals do not establish performance for all years, seasons or stations.

Nominal 80% received-population intervals cover **{100*s['march_interval80_coverage']:.2f}%** of the new observations, with mean width **{s['march_interval80_mean_width_db']:.2f} dB**. Calibration uses the original pre-2025 calibration partition separately from fitting.

![March coverage and fresh hourly errors](results/ml40/march40/march_expansion.png)

## Known-benchmark checks

The previous 500-path benchmark was replayed for a broader seasonal check. It is **already published and no longer a blind test**; the observations, original predictions and VOACAP settings are unchanged. Equally weighted path MAE is **{known['march']:.3f} dB March-expanded**, **{known['shared_parent']:.3f} dB shared parent**, **{known['current']:.3f} dB current GoCAP**, and **{known['voacap']:.3f} dB VOACAP**. This reuses the existing VOACAP predictions; it is not a new native-engine experiment.

The existing 2025 FT8/FT4/CW audit is also a known benchmark. Below are hour/date-balanced MAEs, in dB:

| Mode | Observations | Current GoCAP | Shared parent | March-expanded |
|---|---:|---:|---:|---:|
{other}

These other-mode checks have only two or three dates. Added WSPR volume can change the balance of a pooled model, so a March-specific gain alone is insufficient to establish overall model improvement. The default model is not automatically replaced.

## Run the new model

Use the pinned environment described in [README](README.md):

```sh
.venv/bin/python -m ml40.hourly_query \\
  artifacts/gocap-40m-shared-march.joblib \\
  JN76 FN31 2026-11-17T00:00Z \\
  --sfi 150 --kp 3 --ap 10 --power-watts 5 --hours 24
```

Append `--format json` for structured output. In the default WSPR reference context, one SNR supplies all theoretical mode margins. The source/mode context, unknown antennas/noise and unknown non-WSPR powers still prevent a physically identified universal channel-gain model. PHONE remains threshold-only and physically uncalibrated. See the [shared-model report](SHARED_MODEL_RESULTS.md) for those limits.

## Evidence and reproduction

- [Frozen plan](results/ml40/march40/plan.json), [collection ledger](results/ml40/march40/collection.json), [coverage by year](results/ml40/march40/coverage_by_year.csv).
- [Preparation audit](results/ml40/march40/preparation.json), [training record](results/ml40/march40/training_run.json), [evaluation summary](results/ml40/march40/summary.json).
- [Fresh hourly errors](results/ml40/march40/fresh_march_hourly_errors.csv), [known 500-path errors](results/ml40/march40/known500_case_errors.csv), [other-mode audit](results/ml40/march40/known_mode_audit.csv), [independent verification](results/ml40/march40/verification.json).

Raw exact queries, source checksums, processed samples, fitting features and disk-backed matrices stay under ignored `data/ml40/march40/`. The source is WSPRNet contributors via WSPR Live, using bounded sequential cached requests. Existing GFZ/DRAO solar data, IAGA/NOAA magnetic coefficients and original PSK/RBN records retain the attribution and access conditions in [README](README.md). Historical decoder-context limitations remain explicit; pre-2020 code0 and pre-2023 code1 are not claimed to share one known decoding threshold.

```sh
.venv/bin/python -m ml40.march_collect
.venv/bin/python -m ml40.march_repair_legacy
.venv/bin/python -m ml40.march_prepare
.venv/bin/python -m ml40.march_train
.venv/bin/python -m ml40.march_evaluate
.venv/bin/python -m ml40.march_verify
.venv/bin/python -m ml40.march_report
.venv/bin/python -m unittest discover -v
```

These research stages require the exact original fitting/validation/calibration caches and previous benchmark features, which are excluded from Git. Frozen hashes reject changed inputs. Preserve old artifacts and use a new versioned experiment for further data changes. Independent verification reconstructs the daily medians directly from raw reports, checks duplicate retention and temporal separation, validates global encoder medians and sampled encoded rows, and independently rebuilds reported errors and bootstrap intervals.
'''
    (ROOT/'MARCH_MODEL_RESULTS.md').write_text(text);print('March model report saved.')
if __name__=='__main__':report()
