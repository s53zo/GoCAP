"""Unseen March2025 results, including the small expanded-model regression."""
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from .march_test_plan import ROOT,OUT,GROUPS

def report():
    s=json.loads((OUT/'summary.json').read_text());c=pd.read_csv(OUT/'case_results.csv');h=pd.read_csv(OUT/'hourly_comparison.csv');dist=pd.read_csv(OUT/'per_distance.csv').set_index('distance_group').reindex(GROUPS);days=pd.read_csv(OUT/'per_date.csv');sens=pd.read_csv(OUT/'sensitivity.csv')
    fig,axes=plt.subplots(1,2,figsize=(13,5));x=np.arange(4)
    for shift,key,label,color,hatch in [(-.24,'current','Current GoCAP','#7b498b',''),(0,'march','March-expanded','#1769aa','..'),(.24,'voacap','VOACAP primary','#c05a00','//')]:
        axes[0].bar(x+shift,dist[key+'_mae_db'],width=.24,label=label,color=color,hatch=hatch)
        axes[1].plot(pd.to_datetime(days.date_key),days[key+'_mae_db'],label=label,color=color,ls='-' if key=='march' else '--',lw=2)
    axes[0].set_xticks(x,['100–1,000','1,000–3,000','3,000–6,000','6,000–15,000'],rotation=15);axes[0].set_xlabel('Distance km; 125 paths per group');axes[0].set_ylabel('Case-average hourly MAE (dB)');axes[0].set_ylim(bottom=0);axes[0].legend(fontsize=9)
    axes[1].set_xticks([pd.Timestamp(2025,3,d) for d in [10,13,16,19,23,27,31]]);axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'));axes[1].tick_params(axis='x',rotation=30);axes[1].set_xlabel('Held-out date, March 2025');axes[1].set_ylabel('Daily case-average MAE (dB)');axes[1].set_ylim(bottom=0);axes[1].legend(fontsize=9)
    fig.suptitle('500 additional March 2025 paths: no retraining');fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9),sharex=True)
    for ax,group in zip(axes.flat,GROUPS):
        case=c[c.distance_group.eq(group)].sort_values('case_id').iloc[0];z=h[h.case_id.eq(case.case_id)].sort_values('utc_hour')
        for key,label,color,style in [('voacap','VOACAP','#c05a00','--'),('current','Current GoCAP','#7b498b','--'),('march','March-expanded','#1769aa','-')]:ax.plot(z.utc_hour,z[key+'_snr_db'],label=label,color=color,ls=style,lw=2)
        a=z[z.actual_snr_db.notna()];ax.scatter(a.utc_hour,a.actual_snr_db,c='black',s=22,label='Real hourly median',zorder=5)
        ax.set_title(f'{case.tx_grid} → {case.rx_grid} | {case.date_key} | {case.reported_power_watts:.2g} W | {case.distance_km:,.0f} km',fontsize=10)
        ax.set_ylabel('SNR (dB / 2500 Hz)');ax.set_xlabel('UTC hour');ax.set_xticks([0,6,12,18,23]);ax.grid(alpha=.2)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4,bbox_to_anchor=(.5,.965));fig.suptitle('First frozen hash ID per distance range',y=.998)
    fig.text(.5,.008,'Black points are reported receptions only. Missing hours remain unknown. Independent vertical scales.\nVOACAP uses assumed 10 m dipoles/noise 150; actual station antennas and noise are unrecorded.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.06,1,.92]);fig.savefig(OUT/'hourly_examples.png',dpi=160);plt.close(fig)
    table='\n'.join(f'| {label} | {s["models"][name]["case_mae_db"]:.3f} dB | {s["models"][name]["hour_mae_db"]:.3f} dB | {s["models"][name]["hour_rmse_db"]:.3f} dB | {s["models"][name]["bias_db"]:.3f} dB |' for name,label in [('current','Current default GoCAP'),('shared_parent','Shared parent'),('march','March-expanded GoCAP'),('voacap','VOACAP primary')])
    drows='\n'.join(f'| {label} |125 | {r.current_mae_db:.3f} | {r.march_mae_db:.3f} | {r.voacap_mae_db:.3f} |' for label,r in zip(['100–1,000','1,000–3,000','3,000–6,000','6,000–15,000'],dist.itertuples()))
    sensitivity='\n'.join(f'| {r.scenario} | {r.case_mae_db:.3f} dB |' for r in sens.itertuples())
    base=s['paired_differences']['current'];parent=s['paired_differences']['shared_parent'];voa=s['paired_differences']['voacap']
    date_text=', '.join(pd.to_datetime(days.date_key).dt.strftime('%d').tolist())
    text=f'''# Additional unseen March2025: GoCAP versus VOACAP

**All GoCAP variants have lower received-signal error than VOACAP under the declared settings. The March-expanded model is slightly worse than current GoCAP on this test.** On500 new path-days, case-average hourly MAE is **{s['models']['current']['case_mae_db']:.3f} dB current**, **{s['models']['march']['case_mae_db']:.3f} dB March-expanded**, and **{s['models']['voacap']['case_mae_db']:.2f} dB VOACAP primary**. No model was retrained or adjusted. The default remains unchanged.

Truth: **{s['observed_hours']:,} observed hourly medians from {s['positive_reports']:,} positive WSPR reports**, across15 previously unscored March2025 dates. These dates are absent from fitting and from the six March2025 dates evaluated during the expansion experiment. This tests March performance without scoring March2010–2024 fitting observations.

| Same paired observed hours | Case-average hourly MAE | Pooled hourly MAE | RMSE | Signed bias(prediction−observation) |
|---|---:|---:|---:|---:|
{table}

The March model wins **{s['march_wins_voacap']}/500** cases against primary VOACAP and **{s['march_wins_current']}/500** against current GoCAP. Its case-average difference versus current is **+{base['difference_db']:.3f} dB**, with95% date-cluster bootstrap **[{base['date_cluster95_db'][0]:.3f}, {base['date_cluster95_db'][1]:.3f}] dB**. The interval lies above zero: this is a small regression on the selected cohort, not a reason to promote the larger-data candidate. It does not establish a universal ranking across populations.

Against the shared parent the March change is **+{parent['difference_db']:.3f} dB**, interval **[{parent['date_cluster95_db'][0]:.3f}, {parent['date_cluster95_db'][1]:.3f}] dB**. Against VOACAP it is **{voa['difference_db']:.3f} dB**, interval **[{voa['date_cluster95_db'][0]:.3f}, {voa['date_cluster95_db'][1]:.3f}] dB**. Bootstrap uses10,000 resamples of15 dates. Routes/stations recur and selection depends on successful reporting, limiting wider inference.

![March comparison by distance and date](results/ml40/march2025voa/comparison.png)

| Distance km | Cases | Current MAE dB | March-expanded MAE dB | VOACAP MAE dB |
|---|---:|---:|---:|---:|
{drows}

## What was held out

Source dates are March2025 days **{date_text}**. None was used in the original, shared or March-expanded point fitting. The earlier March2025 experiment used09/12/17/20/25/28; older benchmark days were also excluded. All three model hashes were frozen before source collection. The selected500 case IDs were frozen before prediction, using only identifiers and availability/measurement-validity criteria.

Each case fixes transmitter/receiver callsigns, directed four-character squares, day, reported power and observation context. Require at least6 measured hours,20–50dBm reported power and100–15,000km distance. Identifier-hash/date round-robin selection yields125 cases per distance stratum and500 distinct directed square pairs. Routes may overlap earlier suites; the dates are new. No SNR magnitude or model error chooses cases.

The primary score averages each case's hourly absolute SNR error, then cases equally. This is not directly interchangeable with the earlier6-date March score of6.467dB, which used all sampled rows and equal UTC-hour/date cells. A lower numerical score here does not mean that a model changed or improved: the models are unchanged and the evaluation population/weighting differ.

## Hourly curves and unknown reports

![Predictions and real March receptions](results/ml40/march2025voa/hourly_examples.png)

Examples are the first frozen hash ID per distance group, not favorable cases chosen after scoring. Models predict all12,000 requested hours; only **{s['observed_hours']:,}** have observed truth. The remaining **{12000-s['observed_hours']:,} hours stay unknown**, never zero or closed-path labels. All case/hour predictions and actual measurements where available are in the accompanying CSV. Independent vertical scales preserve the full native range.

## VOACAP settings and sensitivity

Primary: `dipole10m_n150_s1.0`, matching earlier benchmarks—native Method30/CCIR, short great-circle path at7.04MHz between square centres, matched reported transmitter power, ideal half-wave horizontal dipoles10m high broadside to the path at both ends, ground dielectric4/conductivity0.001S/m, noise150, minimum takeoff3° and final SILSO13-month smoothed Version2 SSN. Real station antennas, height/losses and receiver noise remain unknown.

Native SNR50 is a monthly median across days in dB-Hz. Subtract33.9794dB to express2500Hz SNR, using the requested-frequency column rather than MUF/REL. Average native H/H+1 dB SNR for the received hourly-bin centre. Left/right native-hour sensitivities give **{s['voacap_left']['case_mae_db']:.3f}/{s['voacap_right']['case_mae_db']:.3f} dB** case-average error. All4,867 observed hours have numeric predictions under all twelve settings.

| Predeclared VOACAP setting | Case-average MAE |
|---|---:|
{sensitivity}

The scenario range is **{sens.case_mae_db.min():.2f}–{sens.case_mae_db.max():.2f} dB**. No favorable setting replaces the primary after scoring. SSN0.6 explores a legacy convention, not an exact historical-scale reconstruction. Final retrospective solar indices are used; this is not an archived forecast-vintage test.

This is a detected-signal accuracy comparison. VOACAP's monthly climatological target differs from a specific day's received-population median, and unknown hardware/noise prevents a controlled physical calibration. PHONE remains theoretical; no contact probabilities or accuracy in unreported hours are established. These limits apply even though the GoCAP errors are lower under every declared setting.

## Evidence and reproduce

- [Frozen plan](results/ml40/march2025voa/plan.json), [frozen cases](results/ml40/march2025voa/case_freeze.json), [summary](results/ml40/march2025voa/summary.json).
- [500 case results](results/ml40/march2025voa/case_results.csv), [all12,000 hourly comparisons](results/ml40/march2025voa/hourly_comparison.csv), [per-date](results/ml40/march2025voa/per_date.csv), [per-distance](results/ml40/march2025voa/per_distance.csv).
- [Independent verification](results/ml40/march2025voa/verification.json), [6,000 native run checksums](results/ml40/march2025voa/run_manifest.csv), [sensitivity](results/ml40/march2025voa/sensitivity.csv).

Exact time/band-bounded source queries, retrieval timestamps, raw responses, source features and native input/output decks remain under ignored `data/ml40/march2025voa/`. Sampling uses coherent pair/day hash128 plus hash16 for source distances≥2,700km, with a250,001-row cap that fails rather than truncate. Modern code1 WSPR observations are used. The unchanged source processor and fitting-date separation are independently checked.

```sh
.venv/bin/python -m ml40.march_test_collect
.venv/bin/python -m ml40.march_test_prepare
.venv/bin/python -m ml40.march_test_run
.venv/bin/python -m ml40.march_test_verify
.venv/bin/python -m ml40.march_test_report
.venv/bin/python -m unittest discover -v
```

Restore exact cached inputs and the native engine matching frozen hashes for reproduction. No model is fitted by these commands. Further tuning must reserve other untouched dates rather than optimize against this now-published test. Data/source access conditions and credits are in [README](README.md); the [March training report](MARCH_MODEL_RESULTS.md) and [15-year April comparison](VOACAP_15YEAR_COMPARISON.md) describe the related experiments. Primary references: [WSPR Live](https://wspr.live/), [VOACAP SNR interpretation](https://www.voacap.com/2023/understanding/muf.html), [SSN guidance](https://www.voacap.com/2023/understanding/choosingssn.html). SILSO credit: WDC-SILSO, Royal Observatory of Belgium, DOI10.24414/qnza-ac80; CC BY-NC research use.

**Decision: retain current default GoCAP.** Independent checks rebuild all4,867 hourly medians from14,415 raw reports, parse6,000 native runs, reproduce date/case/hour errors and bootstrap intervals, and confirm frozen model/source hashes. The March-expanded artifact stays experimental. No test-driven refitting occurred.
'''
    replacements={'On500':'On 500','across15':'across 15','March2025':'March 2025','March2010':'March 2010','bias(prediction':'bias (prediction','with95%':'with 95%','uses10,000':'uses 10,000','of15':'of 15','used09/12':'used 09/12','selected500':'selected 500','least6':'least 6','hours,20':'hours, 20','and100':'and 100','yields125':'yields 125','and500':'and 500','earlier6-date':'earlier 6-date','of6.467dB':'of 6.467 dB','all12,000':'all 12,000','Method30':'Method 30','at7.04MHz':'at 7.04 MHz','dipoles10m':'dipoles 10 m','dielectric4':'dielectric 4','conductivity0.001S/m':'conductivity 0.001 S/m','noise150':'noise 150','takeoff3°':'takeoff 3°','SILSO13':'SILSO 13','Version2':'Version 2','Subtract33.9794dB':'Subtract 33.9794 dB','express2500Hz':'express 2500 Hz','All4,867':'All 4,867','SSN0.6':'SSN 0.6','hash128':'hash 128','hash16':'hash 16','distances≥2,700km':'distances ≥2,700 km','a250,001':'a 250,001','code1':'code 1','DOI10':'DOI 10','all4,867':'all 4,867','from14,415':'from 14,415','parse6,000':'parse 6,000'}
    for old,new in replacements.items():text=text.replace(old,new)
    (ROOT/'MARCH2025_VOACAP_COMPARISON.md').write_text(text);print('Additional March2025 report saved; default model retained.')
if __name__=='__main__':report()
