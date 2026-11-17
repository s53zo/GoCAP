"""Repository report for the frozen 1000-path extension of the VOACAP benchmark."""
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .voacap_protocol import BASE,OUT,ROOT,COUNT

GROUPS=['100to1000','1000to3000','3000to6000','6000to15000']
LABELS=['100–1,000','1,000–3,000','3,000–6,000','6,000–15,000']

def report():
    assert COUNT==1000,'Use GOCAP_VOACAP_CASES=1000'
    s=json.loads((OUT/'summary.json').read_text());c=pd.read_csv(OUT/'case_results.csv');h=pd.read_csv(OUT/'hourly_comparison.csv');sen=pd.read_csv(OUT/'sensitivity.csv')
    observed=h[h.actual_snr_db.notna()];paired=observed[observed.voacap_snr_db.notna()].copy()
    bydist=c.groupby('distance_group').agg(cases=('case_id','size'),gocap_mae_db=('gocap_mae_db','mean'),voacap_mae_db=('voacap_mae_db','mean'),gocap_wins=('gocap_minus_voacap_mae_db',lambda x:(x<0).sum())).reindex(GROUPS)
    bydist.to_csv(OUT/'distance_results.csv')
    for col in ['gocap_snr_db','voacap_snr_db']:paired[col+'_abs_error']=abs(paired[col]-paired.actual_snr_db)
    byhour=paired.groupby('utc_hour').agg(observed_paths=('case_id','nunique'),gocap_mae_db=('gocap_snr_db_abs_error','mean'),voacap_mae_db=('voacap_snr_db_abs_error','mean'))
    byhour.to_csv(OUT/'utc_hour_results.csv')
    bydate=c.groupby('date_key').agg(cases=('case_id','size'),gocap_mae_db=('gocap_mae_db','mean'),voacap_mae_db=('voacap_mae_db','mean'))
    bydate.to_csv(OUT/'date_results.csv')
    fig,axes=plt.subplots(1,2,figsize=(13,5));x=np.arange(4)
    for delta,col,label,color,hatch in [(-.18,'gocap_mae_db','GoCAP','#1769aa',''),(.18,'voacap_mae_db','VOACAP','#c05a00','//')]:
        axes[0].bar(x+delta,bydist[col],width=.36,label=label,color=color,hatch=hatch)
    axes[0].set_xticks(x,LABELS,rotation=15);axes[0].set_xlabel('Path distance (km); 250 paths per group');axes[0].set_ylabel('Case-average hourly MAE (dB)');axes[0].legend();axes[0].set_ylim(bottom=0)
    axes[0].set_title('Error by path distance')
    axes[1].plot(byhour.index,byhour.gocap_mae_db,color='#1769aa',label='GoCAP',lw=2)
    axes[1].plot(byhour.index,byhour.voacap_mae_db,color='#c05a00',label='VOACAP',ls='--',lw=2)
    axes[1].set_xlabel('UTC hour (HH:00–HH:59 bin)');axes[1].set_ylabel('Observed-hour MAE (dB)');axes[1].set_xticks([0,6,12,18,23]);axes[1].set_ylim(bottom=0);axes[1].legend();axes[1].set_title('Hourly error on paired reported receptions')
    fig.suptitle('1,000 real 40m WSPR paths — frozen primary VOACAP assumptions');fig.tight_layout();fig.savefig(OUT/'error_comparison.png',dpi=160);plt.close(fig)
    # First two frozen hash IDs per distance group: no performance-based selection.
    examples=[cid for g in GROUPS for cid in c[c.distance_group.eq(g)].case_id.head(2)]
    fig,axes=plt.subplots(4,2,figsize=(13,14),sharex=True,sharey=False)
    for ax,cid in zip(axes.flat,examples):
        d=h[h.case_id.eq(cid)].sort_values('utc_hour');m=c[c.case_id.eq(cid)].iloc[0]
        ax.plot(d.utc_hour,d.gocap_snr_db,label='GoCAP',color='#1769aa',lw=2)
        ax.plot(d.utc_hour,d.voacap_snr_db,label='VOACAP',color='#c05a00',ls='--',lw=2)
        known=d[d.actual_snr_db.notna()];ax.scatter(known.utc_hour,known.actual_snr_db,c='black',s=22,label='Reported WSPR median',zorder=4)
        ax.set_title(f'{m.tx_grid} → {m.rx_grid} | {m.date_key} | {m.reported_power_watts:.2g} W | {m.distance_km:,.0f} km',fontsize=10)
        ax.set_xticks([0,6,12,18,23]);ax.set_xlabel('UTC hour');ax.set_ylabel('SNR (dB / 2500 Hz)');ax.grid(alpha=.2)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.972),ncol=3)
    fig.suptitle('Hourly signal predictions: first two frozen IDs in each distance range',y=.997,fontsize=14)
    fig.text(.5,.008,'Black points: actual reported receptions only. Unreported hours remain unknown. Panels use independent vertical scales.\nVOACAP is a monthly median; station antennas and receiver noise are assumed, not measured.',ha='center')
    fig.tight_layout(rect=[0,.045,1,.945]);fig.savefig(OUT/'hourly_examples.png',dpi=160);plt.close(fig)
    n=s['observed_hours'];ng=s['gocap']['numeric_hours'];nv=s['voacap']['numeric_hours'];unknown=24000-n
    metricrows='\n'.join(f'| {label} | {s["gocap"][key]:.2f} dB | {s["voacap"][key]:.2f} dB |' for label,key in [('Case-average hourly MAE (primary)','case_mae_db'),('Pooled hourly MAE','hour_mae_db'),('Pooled hourly RMSE','hour_rmse_db'),('Signed bias: prediction − observation','bias_db')])
    distrows='\n'.join(f'| {label} | {int(r.cases)} | {r.gocap_mae_db:.2f} dB | {r.voacap_mae_db:.2f} dB | {int(r.gocap_wins)} |' for label,r in zip(LABELS,bydist.itertuples()))
    sensrows='\n'.join(f'| {r.scenario} | {r.case_mae_db:.2f} dB | {r.numeric_hours:,} |' for r in sen.itertuples())
    ci=s['date_cluster_bootstrap_95pct_db'];delta=s['gocap_minus_voacap_case_mae_db'];wins=s['case_wins_strict'];losses=s['case_losses_strict'];ties=s['case_ties_exact']
    total_reports=int(observed.positive_reports.sum())
    text=f'''# GoCAP versus VOACAP: 1,000 real 40m paths

**GoCAP case-average hourly SNR error is {s['gocap']['case_mae_db']:.2f} dB versus {s['voacap']['case_mae_db']:.2f} dB for VOACAP** under the frozen primary assumptions. GoCAP has lower error on **{wins:,}/1,000 paths**; VOACAP on **{losses:,}**; exact ties: **{ties}**. Truth consists of **{n:,} observed hourly signal medians**, reconstructed from **{total_reports:,} positive WSPR reports**, on twelve 2025 dates.

The 1,000 distinct directed four-character square pairs exclude all paths in the earlier 100-case suite. Actual station antennas and noise are unknown. This is a received-signal hindcast comparison with explicit VOACAP assumptions, not a controlled physical calibration or a worldwide accuracy guarantee.

| Same paired observed hours | GoCAP | VOACAP |
|---|---:|---:|
{metricrows}

The paired case-average difference (GoCAP − VOACAP) is **{delta:.2f} dB**, with a **95% date-cluster bootstrap interval [{ci[0]:.2f}, {ci[1]:.2f}] dB**. It uses 10,000 resamples of the twelve dates. More paths improve path coverage, but do not increase the number of independent dates. Station populations recur across paths. All results are for the same frozen model, with no fitting, offset correction or scenario selection using test outcomes.

## Hourly results

Both models produce curves for all **24,000 case-hours**. Error is measured only where reception was reported: **{n:,} hours**; **{unknown:,} hours remain unknown**, never zero or closed-path labels. Each path contributes its mean absolute error over its reported hours, then paths are weighted equally. Pooled hourly scores are secondary. Primary numeric coverage on paired truth is GoCAP **{ng:,}** and VOACAP **{nv:,}** hours; [all-observation coverage and metrics](results/ml40/voacap1000/summary.json) are retained separately.

![Error by path distance and UTC hour](results/ml40/voacap1000/error_comparison.png)

| Path distance, km | Paths | GoCAP MAE | VOACAP MAE | GoCAP wins |
|---|---:|---:|---:|---:|
{distrows}

The UTC-hour figure pools the actual paired receptions in each hour; its path population changes with reporting. It is not an accuracy claim for hours with no measurements. [Exact UTC-hour results](results/ml40/voacap1000/utc_hour_results.csv) and [per-date results](results/ml40/voacap1000/date_results.csv) accompany the figure.

![Hourly prediction examples](results/ml40/voacap1000/hourly_examples.png)

Examples are the first two hash-sorted frozen IDs in each distance range. Blue/orange curves cover all hours; black points are hourly median real receptions. They were not chosen by model performance. Panels have independent vertical scales to retain detail while showing the full native prediction range, including very weak unverified daytime predictions on long paths. [All 24,000 hourly rows](results/ml40/voacap1000/hourly_comparison.csv) include actual values where available and both predictions.

## VOACAP sensitivity

The primary setting remains `dipole10m_n150_s1.0`: native Method 30, CCIR coefficients, shortest great-circle path, 7.04 MHz, square-centre coordinates, reported transmit power, ideal half-wave horizontal dipoles 10 m high and broadside to the path at both ends, ground dielectric 4/conductivity 0.001 S/m, noise parameter 150, minimum takeoff 3°, final WDC-SILSO 13-month smoothed Version 2 SSN. Required reliability is 50%, WSPR nominal required SNR is −31 dB/2500 Hz; sporadic-E factor is zero.

VOACAP SNR50 in dB-Hz is converted to 2500 Hz by subtracting **33.9794 dB**. The requested-frequency output column is used, not MUF, REL or SNR90. Average the native H and H+1 SNR values in dB to approximate each [H,H+1) reception bin. Using the left or right native hour instead gives **{s['voacap_left']['case_mae_db']:.2f}/{s['voacap_right']['case_mae_db']:.2f} dB** case-average MAE. Printed native SNR is rounded to integer dB. The engine pilot matched the documented dxtron API at all 24 hours.

All twelve settings were declared before prediction. SSN scale 0.6 explores a legacy convention; it is not an exact historical-scale reconstruction. No favorable setting replaces the primary after scoring.

| Antenna / noise / SSN scale | VOACAP case-average MAE | Numeric observed hours |
|---|---:|---:|
{sensrows}

GoCAP with steady preceding solar history, corresponding to scalar-query assumptions, gives **{s['gocap_steady']['case_mae_db']:.2f} dB** case-average MAE. Primary GoCAP scoring uses actual completed historical solar context.

## What this means

This experiment measures hour-by-hour received WSPR SNR accuracy on frequently detected paths, under reported power and unknown hardware. Requiring at least six observed hours selects active, repeatedly detectable paths. Selection does not use SNR values or model errors, but observation availability is itself detection-selected. The balanced distance design is not traffic-weighted global performance.

VOACAP's target is the monthly hourly median across days; our observations are detected signals on a particular day. GoCAP was trained on a received-signal population and additionally sees completed daily solar history including Kp/Ap. VOACAP uses smoothed monthly SSN. Final solar data, including a centred 13-month SSN series that would not yet have been known on the query date, make this a retrospective hindcast. The result cannot establish band-open probabilities, undetected signal strength, calibrated 100 W/dipole performance, other-mode accuracy or forecast-vintage skill.

The earlier 100-case benchmark remains unchanged. This suite expands path coverage while using the same dates, source family and modeling assumptions; it is not an independent new-year validation, and its different path mix means changes in aggregate error are not model improvements.

## Reproduce and inspect

- [All 1,000 path results](results/ml40/voacap1000/case_results.csv)
- [Hourly predictions and real signals](results/ml40/voacap1000/hourly_comparison.csv)
- [Full 12-scenario hourly output](results/ml40/voacap1000/voacap_all_scenarios.csv)
- [Frozen protocol](results/ml40/voacap1000/protocol.json) and [case freeze](results/ml40/voacap1000/case_freeze.json)
- [Summary](results/ml40/voacap1000/summary.json), [independent verification](results/ml40/voacap1000/verification.json), [run checksums](results/ml40/voacap1000/run_manifest.csv)
- [Original 100-case report](VOACAP_COMPARISON.md)

Raw reports and exact source queries: `data/ml40/voacap1000/raw/`. Input/output decks: `data/ml40/voacap1000/runs/<case_id>/<scenario>/`. All endpoints are four-character Maidenhead square centres. Eligibility and quotas match the earlier suite: fixed station pair/day/reported power, at least six reported hours, power 20–50 dBm, 100–15,000 km, 250 unique directed paths per distance stratum. Identifier-hash/date round-robin selection uses seed `gocap-voacap1000-2026-v1`.

The original hash128 source sample has too few long paths for 250 per stratum. Source collection therefore retains hash128 at shorter distances and adds hash16 for source distances ≥2,700 km (a conservative buffer around the 3,000 km square-centre split). This rule was frozen before scoring. No date hit the 250,001-row cap. Original 100-suite directed paths are excluded before selection. Dates are the sixth day of each 2025 month. Model-fitting feature data ends in 2024; the model file is unchanged and checksummed.

```sh
export GOCAP_VOACAP_CASES=1000
.venv/bin/python -m ml40.voacap_collect
.venv/bin/python -m ml40.voacap_prepare
.venv/bin/python -m ml40.voacap_run
.venv/bin/python -m ml40.voacap_verify
.venv/bin/python -m ml40.voacap_verify_extra
.venv/bin/python -m ml40.voacap1000_report
.venv/bin/python -m unittest discover -v
```

Unsetting the environment variable selects the original 100-case suite. Cached source files and native binary must match frozen hashes; do not overwrite them with refreshed data. Native build instructions remain in [the original report](VOACAP_COMPARISON.md). A changed binary or source requires a separately versioned protocol.

**Validation: share with caveats.** All 12,000 native outputs and their power/solar/noise/antenna/coordinate input cards are independently checked; all real hourly medians are independently rebuilt from raw reports; hashes, path exclusions, distance quotas, direct model predictions and primary calculations are verified. All 38 tests pass. Physical hardware calibration, unknown-hour accuracy and operational forecast performance remain unverified.

Sources: [VOACAP SNR distribution and units](https://www.voacap.com/2023/understanding/muf.html), [VOACAP smoothed SSN guidance](https://www.voacap.com/2023/understanding/choosingssn.html), [native engine source](https://github.com/jawatson/voacapl) at commit `c12a98b348c39f36f2659b8b3aa97b86284c8a1d`, [dxtron API documentation](https://voacap.dxtron.com/api/docs/), [official SILSO data](https://www.sidc.be/SILSO/DATA/SN_ms_tot_V2.0.txt). Sunspot data credit: WDC-SILSO, Royal Observatory of Belgium, Brussels, DOI 10.24414/qnza-ac80; CC BY-NC research use. Real reception source: [WSPR.live](https://wspr.live/) with exact archived query/response metadata.
'''
    (ROOT/'VOACAP_1000_COMPARISON.md').write_text(text)
    print(f'Report written: {n:,} observed hours, {wins} GoCAP wins, {losses} VOACAP wins')
if __name__=='__main__':report()
