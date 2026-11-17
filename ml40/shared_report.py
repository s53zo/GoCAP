"""Report a negative/neutral result faithfully; retain the deployed baseline."""
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .shared_plan import ROOT,BASE,OUT,ARTIFACT,sha
from .shared_prepare import GROUPS

def report():
    s=json.loads((OUT/'summary.json').read_text());tr=json.loads((OUT/'training_run.json').read_text());c=pd.read_csv(OUT/'case_results.csv');h=pd.read_csv(OUT/'hourly_comparison.csv');audit=pd.read_csv(OUT/'known_mode_audit.csv');sensitivity=pd.read_csv(OUT/'sensitivity.csv')
    d=c.groupby('distance_group').agg(cases=('case_id','size'),baseline_mae_db=('baseline_mae_db','mean'),shared_mae_db=('shared_mae_db','mean'),voacap_mae_db=('voacap_mae_db','mean')).reindex(GROUPS);d.to_csv(OUT/'distance_results.csv')
    a=h.dropna(subset=['actual_snr_db','voacap_snr_db']).copy()
    for name in ['baseline','shared','voacap']:a[name+'_error']=abs(a[name+'_snr_db']-a.actual_snr_db)
    hours=a.groupby('utc_hour').agg(measured_paths=('case_id','nunique'),baseline_mae_db=('baseline_error','mean'),shared_mae_db=('shared_error','mean'),voacap_mae_db=('voacap_error','mean'));hours.to_csv(OUT/'utc_hour_results.csv')
    fig,axes=plt.subplots(1,2,figsize=(13,5));x=np.arange(4)
    for delta,name,label,color,hatch in [(-.24,'baseline','Current GoCAP','#7b498b',''),(0,'shared','Shared + five points','#1769aa','..'),(.24,'voacap','VOACAP','#c05a00','//')]:
        axes[0].bar(x+delta,d[name+'_mae_db'],width=.24,label=label,color=color,hatch=hatch)
        axes[1].plot(hours.index,hours[name+'_mae_db'],label=label,color=color,lw=2,ls='--' if name!='shared' else '-')
    axes[0].set_xticks(x,['100–1,000','1,000–3,000','3,000–6,000','6,000–15,000'],rotation=15);axes[0].set_xlabel('Path distance (km); 125 cases per group');axes[0].set_ylabel('Case-average hourly MAE (dB)');axes[0].set_ylim(bottom=0)
    axes[1].set_xlabel('UTC hour');axes[1].set_ylabel('Observed-hour MAE (dB)');axes[1].set_xticks([0,6,12,18,23]);axes[1].set_ylim(bottom=0)
    axes[0].legend(fontsize=9);axes[1].legend(fontsize=9);fig.suptitle('500 new 40m WSPR paths — current and shared models are effectively tied');fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9),sharex=True)
    for ax,g in zip(axes.flat,GROUPS):
        cid=c[c.distance_group.eq(g)].case_id.iloc[0];row=c[c.case_id.eq(cid)].iloc[0];z=h[h.case_id.eq(cid)].sort_values('utc_hour')
        for name,label,color,style in [('voacap','VOACAP','#c05a00','--'),('baseline','Current GoCAP','#7b498b','--'),('shared','Shared + five points','#1769aa','-')]:ax.plot(z.utc_hour,z[name+'_snr_db'],color=color,ls=style,label=label,lw=2)
        known=z[z.actual_snr_db.notna()];ax.scatter(known.utc_hour,known.actual_snr_db,c='black',label='Reported WSPR median',s=22,zorder=5)
        ax.set_title(f'{row.tx_grid} → {row.rx_grid} | {row.date_key} | {row.distance_km:,.0f} km | {row.reported_power_watts:.2g} W',fontsize=10)
        ax.set_xlabel('UTC hour');ax.set_ylabel('SNR (dB / 2500 Hz)');ax.set_xticks([0,6,12,18,23]);ax.grid(alpha=.2)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4,bbox_to_anchor=(.5,.965));fig.suptitle('Hourly predictions: first frozen ID per distance group',y=.998)
    fig.text(.5,.008,'Only black points are verified receptions. Missing hours remain unknown. Panels have independent vertical scales.\nCurrent/shared curves nearly overlap. Actual station hardware and noise are unknown.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.065,1,.92]);fig.savefig(OUT/'hourly_examples.png',dpi=160);plt.close(fig)
    ci=s['shared_minus_baseline_95pct_date_bootstrap_db'];ab=s['shared_minus_aggregate_95pct_date_bootstrap_db'];selection=tr['selected_candidate']
    metrics='\n'.join(f'| {label} | {s["baseline"][key]:.3f} dB | {s["shared"][key]:.3f} dB | {s["voacap"][key]:.3f} dB |' for label,key in [('Case-average hourly MAE','case_mae_db'),('Pooled hourly MAE','hour_mae_db'),('Pooled hourly RMSE','hour_rmse_db'),('Signed bias','bias_db')])
    validation=[]
    for name,r in tr['validation'].items():validation.append(f'| {name} | '+ ' | '.join(f'{r["by_mode"][m]["hour_date_mae_db"]:.3f}' for m in ['WSPR','FT8','FT4','CW'])+f' | {r["equal_mode_hour_date_mae_db"]:.3f} | {r["all_guards_pass"]} |')
    validation='\n'.join(validation)
    mode_rows=[]
    for mode in ['FT8','FT4','CW']:
        old=audit[(audit['model']=='baseline')&(audit['mode']==mode)].iloc[0];new=audit[(audit['model']=='shared')&(audit['mode']==mode)].iloc[0]
        mode_rows.append(f'| {mode} | {int(new.rows):,} | {int(new.days)} | {old.hour_date_mae_db:.3f} dB | {new.hour_date_mae_db:.3f} dB |')
    mode_rows='\n'.join(mode_rows)
    sensitivity_rows='\n'.join(f'| {r.scenario} | {r.case_mae_db:.3f} dB |' for r in sensitivity.itertuples())
    dist_rows='\n'.join(f'| {g} | {int(r.cases)} | {r.baseline_mae_db:.3f} | {r.shared_mae_db:.3f} | {r.voacap_mae_db:.3f} |' for g,r in zip(GROUPS,d.itertuples()))
    text=f'''# Shared multi-mode model with five explicit path points

**The new model works, but this experiment does not establish better accuracy than current GoCAP.** On **500 new real WSPR paths**, case-average hourly SNR error is **{s['shared']['case_mae_db']:.3f} dB shared / {s['baseline']['case_mae_db']:.3f} dB current / {s['voacap']['case_mae_db']:.2f} dB VOACAP**. The tiny shared/current difference is within sampling uncertainty. The current deployed model is retained; the new artifact is experimental.

Truth: **{s['observed_hours']:,} positive hourly medians from {s['positive_reports']:,} raw WSPR reports**, spanning the seventh day of each month in 2025. All 500 directed square pairs are distinct and excluded from the previous 100/1,000-path suites. Missing receptions remain unknown. Actual antennas/noise are unrecorded, so VOACAP comparison is under explicit hardware assumptions.

| Same paired observed hours | Current GoCAP | Shared + explicit points | VOACAP primary |
|---|---:|---:|---:|
{metrics}

Shared minus current case-average error: **{s['shared_minus_baseline_case_mae_db']:.4f} dB**, 95% paired date-cluster bootstrap **[{ci[0]:.4f}, {ci[1]:.4f}] dB**. It is effectively a tie: shared wins on **{s['shared_wins_baseline']}/500** paths, current on **{500-s['shared_wins_baseline']}**. Shared beats VOACAP on **{s['shared_wins_voacap']}/500** paths; that result does not establish universally superior propagation physics. Only twelve sampled dates are available; shared stations and detection selection limit generalization.

![Comparison by distance and UTC hour](results/ml40/shared40/comparison.png)

| Distance group km | Cases | Current MAE dB | Shared MAE dB | VOACAP MAE dB |
|---|---:|---:|---:|---:|
{dist_rows}

## What changed

One pooled histogram gradient-boosted regressor was trained on **{tr['training_rows']:,}** original pre-2025 hourly records: WSPR **{tr['training_by_mode']['WSPR']:,}**, FT8 **{tr['training_by_mode']['FT8']:,}**, FT4 **{tr['training_by_mode']['FT4']:,}**, CW **{tr['training_by_mode']['CW']:,}**. It replaces separate per-mode regressors within this candidate package, not the deployed artifact. Raw native SNR targets are standardized to 2500 Hz. Known WSPR power remains an input; unknown other-mode powers remain marked unknown. Encoder numerical imputation does not turn unknown power into a claimed physical measurement. No PHONE reception records were supplied.

The model retains source, mode and decoder context because successful-report populations differ. It is a shared *observation-context SNR model*, not a uniquely identified mode-independent channel-gain model. Pooling alone cannot disentangle unknown transmit power, antennas, noise and reception selection.

At **0%, 25%, 50%, 75% and 100%** along the shortest great-circle path, it adds **45 explicit features**: geographic latitude, cyclic longitude, solar-elevation sine, cyclic local solar time, degree-one geomagnetic latitude, daylight and twilight state. Existing nine-point aggregate geometry and completed solar-history features are retained. These positions describe path conditions, not known ionospheric reflection points. Antipodal interiors are flagged/undefined rather than choosing an arbitrary great circle.

The default new query uses the WSPR reference observation context at supplied power, produces **one reference SNR per hour**, and applies the existing theoretical WSPR/FT8/FT4/CW/PHONE thresholds to that same SNR. This is useful as a scenario interface; physical mode independence, PHONE accuracy and open-path probabilities remain unverified.

## Did the two ideas help?

Four candidates isolate pooling and explicit-point effects: aggregate versus explicit-point features, each with observation weighting or source/mode/hour balancing. All hyperparameters were frozen first: absolute-error loss, 240 iterations, 31 leaves, minimum leaf size 100, L2=10, learning rate 0.05. Selection uses only original2024day08 validation;2024day22 calibration is separate. Final tests never choose a candidate.

| Shared candidate | WSPR validation MAE | FT8 | FT4 | CW | Equal-mode mean | Regression guards pass |
|---|---:|---:|---:|---:|---:|---|
{validation}

Selected: **`{selection}`**, the lowest validation equal-mode error among eligible point-feature candidates. Guards allow at most5% WSPR and10% other-mode regression versus current GoCAP; passing them is not evidence of improvement. The original model's validation equal-mode mean is **{np.mean([r['hour_date_mae_db'] for r in tr['baseline_validation'].values()]):.3f} dB**, so the shared candidate is not better overall on that validation criterion either. The mode-balanced candidates failed the WSPR guard.

On the new500 paths, the matched pooled **aggregate-only** ablation scores **{s['pooled_aggregate_ablation']['case_mae_db']:.3f} dB**. Adding explicit points changes case MAE by **{s['shared_minus_aggregate_case_mae_db']:.4f} dB**, with95% date-bootstrap **[{ab[0]:.4f}, {ab[1]:.4f}] dB**. There is no demonstrated incremental gain from those explicit features in this experiment. Existing nine-point summaries may already capture much of the useful geometry, but that explanation is a hypothesis, not established causation.

## Other-mode audit

The frozen selected model was additionally checked against the **already published** 2025 FT8/FT4/CW hourly archive. These dates were excluded from fitting but were used in earlier research reports, so this is a known-benchmark audit, not a new blind all-mode test. No unknown power was invented to score VOACAP on these records. Metrics below are hour/date-balanced MAE, not the case-balanced WSPR metric above.

| Mode | Observed hours | Dates | Current GoCAP | Shared + points |
|---|---:|---:|---:|---:|
{mode_rows}

CW/FT4 changes are small and have only2–3 dates; FT8 gets slightly worse. These mixed results do not justify promoting the shared model. PHONE remains theoretical threshold conversion only. The nominal80% calibrated WSPR interval covers **{s['shared_interval80_coverage']*100:.2f}%** of the new observed hours with mean width **{s['shared_interval80_mean_width_db']:.2f} dB**; it is received-population uncertainty, not confidence that an unreported path is open.

## Hour-by-hour examples

![Hourly predictions and real receptions](results/ml40/shared40/hourly_examples.png)

First hash-sorted case per distance stratum, chosen without looking at performance. Current/shared curves almost overlap. Both predict all **12,000 case-hours**; only **{s['observed_hours']:,}** have measured truth, leaving **{12000-s['observed_hours']:,} unknown**. Very weak native VOACAP values in unreported hours are not scored as successful or failed predictions. Panels use independent vertical scales.

## VOACAP assumptions and sensitivity

The primary setting matches the prior comparisons: Method30/CCIR, short great-circle path at7.04MHz between four-character square centres; actual reported WSPR power; ideal half-wave horizontal dipoles10m high broadside to the path at both ends, ground dielectric4/conductivity0.001S/m, noise150, minimum takeoff3°, final SILSO13-month smoothed SSN Version2. Native SNR50 is dB-Hz; subtract33.9794dB for2500Hz. Average native H/H+1 SNR in dB for the observed hourly-bin centre. Left/right native-hour sensitivity scores are **{s['voacap_left']['case_mae_db']:.3f}/{s['voacap_right']['case_mae_db']:.3f} dB**.

All12 antenna/noise/SSN settings were predeclared; no favorable setting was selected after scoring. Actual station hardware is unknown. The SSN0.6 factor explores a legacy convention and is not an exact historic-scale reconstruction.

| VOACAP setting | Case-average MAE |
|---|---:|
{sensitivity_rows}

VOACAP predicts a monthly hourly median across days; the truth is detected signals on a particular day. GoCAP sees completed daily Kp/Ap/F10.7 history. Both comparisons use final retrospective indices, including centred13-month SSN for VOACAP, not archived forecast vintages. This and received-signal selection limit the inference. The local native engine is unchanged from the API-validated earlier benchmarks.

## Run the experimental model

Use the existing [`requirements-hourly-lock.txt`](requirements-hourly-lock.txt) environment. No archive lookup is needed for this query:

```sh
.venv/bin/python -m ml40.shared_query JN76 FN31 2026-11-17T00:00Z \\
  --sfi150 --kp3 --ap10 --power-watts5 --hours24
```

Append `--format json` for one SNR and all theoretical mode margins per hour. `--history-file` works for a single-hour query as with the existing interface. Artifact: [`gocap-40m-shared-points.joblib`](artifacts/gocap-40m-shared-points.joblib), **{ARTIFACT.stat().st_size:,} bytes** (about 1.0 MB, versus 3.27 MB for current GoCAP), SHA256 `{sha(ARTIFACT)}`. Current default `gocap-40m-hourly.joblib` is unchanged.

## Evidence and reproduction

- [Frozen experiment plan](results/ml40/shared40/plan.json), [training/selection record](results/ml40/shared40/training_run.json), [case freeze](results/ml40/shared40/case_freeze.json).
- [500 path results](results/ml40/shared40/case_results.csv), [12,000 hourly comparisons including theoretical margins](results/ml40/shared40/hourly_comparison.csv), [sensitivity](results/ml40/shared40/sensitivity.csv), [known other-mode audit](results/ml40/shared40/known_mode_audit.csv).
- [Independent verification](results/ml40/shared40/verification.json), [native run checksums](results/ml40/shared40/run_manifest.csv), [summary](results/ml40/shared40/summary.json).

Working data remains under ignored `data/ml40/shared40/`: raw exact queries/responses and their timestamps/checksums, coherent features, fitted challengers, all-scenario Parquet and6,000 native input/output decks. The fresh source sample uses identifier-hash128 at shorter paths and hash16 beyond a2,700km buffer; no truncation. Outcome-blind identifier hash/date round-robin selects125 cases per distance bin with at least6 observed hours,20–50dBm reported power and100–15,000km distance. All previous1,100 comparison square pairs are excluded.

Stages, with the exact prior fitting caches and native engine restored:

```sh
.venv/bin/python -m ml40.shared_collect
.venv/bin/python -m ml40.shared_prepare
.venv/bin/python -m ml40.shared_train
.venv/bin/python -m ml40.shared_voacap
.venv/bin/python -m ml40.shared_evaluate
.venv/bin/python -m ml40.shared_verify
.venv/bin/python -m ml40.shared_report
.venv/bin/python -m unittest discover -v
```

Frozen hashes intentionally reject changed data/code/model inputs. Do not refit using these released test results; start a separately versioned experiment with genuinely new final dates. Data sources, access conditions and hardware expansion instructions remain in [README](README.md). WSPR reports credit volunteer contributors/WSPR Live; solar indices credit GFZ and DRAO/NRCan; magnetic coefficients credit IAGA/NOAA; SSN credit WDC-SILSO, Royal Observatory of Belgium, DOI10.24414/qnza-ac80,CC BY-NC research use. [VOACAP SNR](https://www.voacap.com/2023/understanding/muf.html) and [smoothed SSN guidance](https://www.voacap.com/2023/understanding/choosingssn.html) describe its different climatological target.

**Decision: retain current GoCAP; share the new candidate as experimental with caveats.** Independent checks rebuild all4,820 real hourly medians from14,894 raw reports, parse all6,000 native outputs, verify2,500 point coordinates with a separate geodesic formula, reproduce primary errors/bootstraps and confirm that all mode margins use one reference SNR. Physical channel calibration and PHONE measurement validation remain unverified.
'''
    # Normalize compact numeric writing in prose and keep runnable shell flags correct.
    for old,new in {'--sfi150':'--sfi 150','--kp3':'--kp 3','--ap10':'--ap 10','--power-watts5':'--power-watts 5','--hours24':'--hours 24','31 leaves,min-leaf100,L2=10,learning-rate0.05':'31 leaves, minimum leaf100, L2=10, learning rate0.05','new500':'new 500','at most5%':'at most 5%','and10%':'and 10%','with95%':'with 95%','only2–3':'only 2–3','nominal80%':'nominal 80%','all12':'all 12','All12':'All 12','all4,820':'all 4,820','from14,894':'from 14,894','all6,000':'all 6,000','verify2,500':'verify 2,500','and6,000':'and 6,000','a2,700':'a 2,700','selects125':'selects 125','least6':'least 6','hours,20':'hours, 20','and100':'and 100','previous1,100':'previous 1,100','original2024day08':'original 2024 day08',';2024day22':'; 2024 day22','Method30':'Method 30','at7.04MHz':'at 7.04 MHz','dipoles10m':'dipoles 10 m','dielectric4':'dielectric 4','conductivity0.001S/m':'conductivity 0.001 S/m','noise150':'noise 150','takeoff3°':'takeoff 3°','SILSO13':'SILSO 13','Version2':'Version 2','subtract33.9794dB':'subtract 33.9794 dB','for2500Hz':'for 2500 Hz','SSN0.6':'SSN 0.6','centred13':'centred 13','DOI10':'DOI 10','ac80,CC':'ac80, CC'}.items():text=text.replace(old,new)
    (ROOT/'SHARED_MODEL_RESULTS.md').write_text(text)
    decision={'promoted':False,'decision':'Retain current hourly model; shared explicit-point candidate remains experimental.',
        'reason':'No clear500-case improvement versus current model or matched aggregate ablation; mixed known other-mode audit.',
        'baseline_sha256':s['baseline_model_sha256'],'candidate_sha256':s['candidate_model_sha256']}
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2))
    meta=json.loads(ARTIFACT.with_suffix('.json').read_text());meta.update({'promoted':False,'evaluation_summary':'No demonstrated improvement over current GoCAP; see SHARED_MODEL_RESULTS.md'})
    ARTIFACT.with_suffix('.json').write_text(json.dumps(meta,indent=2));print('Report saved. Experimental candidate only; current model retained.')
if __name__=='__main__':report()
