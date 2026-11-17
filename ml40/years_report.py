"""Report15-year held-out comparison without implying calibrated physics superiority."""
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .years_plan import ROOT,OUT

def report():
    s=json.loads((OUT/'summary.json').read_text());cfg=json.loads((OUT/'plan.json').read_text());y=pd.read_csv(OUT/'per_year.csv');c=pd.read_csv(OUT/'case_results.csv');h=pd.read_csv(OUT/'hourly_comparison.csv');sens=pd.read_csv(OUT/'sensitivity.csv')
    fig,ax=plt.subplots(figsize=(11,5.5))
    for name,label,color,style in [('current','Current GoCAP','#7b498b','--'),('shared_parent','Shared parent','#777777',':'),('march','March-expanded GoCAP','#1769aa','-'),('voacap','VOACAP primary','#c05a00','--')]:
        ax.plot(y.year,y[name+'_mae_db'],label=label,color=color,ls=style,lw=2,marker='o' if name in ['march','voacap'] else None,ms=4)
    ax.set_xticks(y.year,[str(a) for a in y.year],rotation=60);ax.set_ylim(bottom=0);ax.set_ylabel('Case-average hourly SNR error (dB)');ax.set_title('Held-out April receptions, 2010–2024: 20 path-days per year');ax.legend(ncol=2,fontsize=9);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'per_year_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9),sharex=True)
    for ax,year in zip(axes.flat,[2010,2014,2020,2024]):
        case=c[c.year.eq(year)].sort_values('case_id').iloc[0];g=h[h.case_id.eq(case.case_id)].sort_values('utc_hour')
        for name,label,color,style in [('voacap','VOACAP','#c05a00','--'),('current','Current GoCAP','#7b498b','--'),('march','March-expanded GoCAP','#1769aa','-')]:ax.plot(g.utc_hour,g[name+'_snr_db'],label=label,color=color,ls=style,lw=2)
        actual=g[g.actual_snr_db.notna()];ax.scatter(actual.utc_hour,actual.actual_snr_db,c='black',label='Real hourly median',s=22,zorder=5)
        ax.set_title(f'{case.tx_grid} → {case.rx_grid} | {case.date_key} | {case.reported_power_watts:.2g} W | {case.distance_km:,.0f} km',fontsize=10)
        ax.set_xlabel('UTC hour');ax.set_ylabel('SNR (dB / 2500 Hz)');ax.set_xticks([0,6,12,18,23]);ax.grid(alpha=.2)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4,bbox_to_anchor=(.5,.965));fig.suptitle('First frozen case IDs in four preselected years',y=.998)
    fig.text(.5,.008,'Black points only: reported receptions. Missing hours remain unknown. Independent vertical scales.\nVOACAP uses assumed 10 m dipoles/noise 150; real station antennas and noise are unrecorded.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.06,1,.92]);fig.savefig(OUT/'hourly_examples.png',dpi=160);plt.close(fig)
    table='\n'.join(f'| {label} | {s["models"][key]["case_mae_db"]:.3f} dB | {s["models"][key]["hour_mae_db"]:.3f} dB | {s["models"][key]["hour_rmse_db"]:.3f} dB | {s["models"][key]["bias_db"]:.3f} dB |' for key,label in [('march','March-expanded GoCAP'),('shared_parent','Shared parent'),('current','Current default GoCAP'),('voacap','VOACAP primary')])
    annual='\n'.join(f'| {int(r.year)} | {r.march_mae_db:.2f} | {r.current_mae_db:.2f} | {r.voacap_mae_db:.2f} | {int(r.march_wins_voacap)}/20 | {r.median_distance_km:,.0f} | {r.smoothed_ssn:.1f} |' for r in y.itertuples())
    scenario='\n'.join(f'| {r.scenario} | {r.case_mae_db:.3f} dB |' for r in sens.itertuples())
    ci=s['paired_differences']['voacap'];base=s['paired_differences']['current'];parent=s['paired_differences']['shared_parent']
    bycontext=c.decoder_context.value_counts().to_dict();numeric=s['models']['voacap']['numeric_hours']
    text=f'''# GoCAP versus VOACAP across 2010–2024

**The March-expanded model has lower received-signal error than VOACAP in every sampled year.** Across **300 held-out path-days**, equally weighted by case and year, its hourly SNR MAE is **{s['models']['march']['case_mae_db']:.2f} dB**, versus **{s['models']['voacap']['case_mae_db']:.2f} dB** for the frozen primary VOACAP assumptions. GoCAP wins **{s['march_wins_voacap']}/300** cases. These are real reported receptions under unknown station hardware; this is not proof of universally better propagation physics.

Truth contains **{s['observed_hours']:,} hourly medians from {s['positive_reports']:,} positive WSPR archive reports**, on **60 held-out April dates across15 years**. All three GoCAP models and the native engine were frozen; no fitting or corrections were made using this benchmark.

| Same3,081 observed hours | Case-average hourly MAE | Pooled hourly MAE | RMSE | Signed bias(prediction−observation) |
|---|---:|---:|---:|---:|
{table}

The March-model minus VOACAP difference is **{ci['difference_db']:.3f} dB**, with95% year-cluster bootstrap interval **[{ci['year_cluster95_db'][0]:.3f}, {ci['year_cluster95_db'][1]:.3f}] dB**. There are15 year clusters and only20 cases per year; resampling describes this design, not worldwide prediction confidence. All300 cases and all **{numeric:,}** observed hours have numeric VOACAP predictions under every declared scenario.

**The larger March training set does not establish an improvement over existing GoCAP here.** March versus current default changes case MAE by **{base['difference_db']:.3f} dB**, interval **[{base['year_cluster95_db'][0]:.3f}, {base['year_cluster95_db'][1]:.3f}] dB**. Versus the shared parent the change is **{parent['difference_db']:.3f} dB**, interval **[{parent['year_cluster95_db'][0]:.3f}, {parent['year_cluster95_db'][1]:.3f}] dB**. Both intervals include zero. The default stays unchanged and the March artifact remains experimental.

![Year-by-year prediction error](results/ml40/years15/per_year_comparison.png)

| Year | March GoCAP MAE | Current GoCAP MAE | VOACAP MAE | GoCAP wins | Median path km | Smoothed SSN v2 |
|---|---:|---:|---:|---:|---:|---:|
{annual}

Every year has20 cases. Path/station populations, hardware, reporting and space weather differ, so these year differences must not be interpreted as a causal solar-cycle effect. No distance quota was imposed because the early network was sparse. Actual case distances span **{c.distance_km.min():.0f}–{c.distance_km.max():,.0f} km**; most median distances are around1,000km. This cohort is not directly comparable with the distance-balanced2025 suites.

## Why these are genuine held-out dates

The March-expanded model learned from every available March day in2010–2024. Scoring those same March observations would be a resubstitution test. Instead, source windows are **April6,13,20 and27 in every year**, all absent from the original and expanded point-fitting partitions. Before prediction, identifier hashes selected20 eligible directed square-pair path-days per year, round-robin across the four dates.

The cases are fixed station pair/day/four-character endpoint squares/reported power/decoder context. Eligibility requires at least6 observed hours,20–50dBm reported power and100–15,000km distance. Square pairs are unique within a year; repeats across years are permitted. There are **{c[['tx_grid','rx_grid']].drop_duplicates().shape[0]} distinct directed square pairs** overall. SNR values, model predictions and errors do not choose cases.

This is a **retrospective within-period held-out-day test**, not forward-year or operational forecasting. The models use training observations from later years when predicting an earlier held-out day, as well as final historical solar series. It tests April transfer across the expanded year span, not accuracy on March fitting records or every season.

## Hardware assumptions and the large VOACAP bias

Primary VOACAP remains `dipole10m_n150_s1.0`: Method30, CCIR, shortest great-circle path at7.04MHz between square centres, actual reported power, half-wave horizontal dipoles10m high broadside to the path at both ends, ground dielectric4/conductivity0.001S/m, noise parameter150, minimum takeoff3°, smoothed SILSO Version2 SSN. Reported power is matched; actual antennas, losses and receiver noise are not measured.

Under those assumptions VOACAP's mean prediction is **{s['models']['voacap']['bias_db']:.2f} dB higher** than the observed SNR, versus **{s['models']['march']['bias_db']:.2f} dB** for GoCAP. This large bias materially contributes to the result. Unknown hardware/noise, climatological versus daily targets and reception selection are unresolved explanations, not measured causes. A controlled reference-station calibration would be needed to attribute the discrepancy.

All twelve predeclared antenna/noise/SSN settings were scored without selecting a new primary afterward. VOACAP case MAE spans **{sens.case_mae_db.min():.2f}–{sens.case_mae_db.max():.2f} dB**. The range makes clear that the comparison depends on assumptions, even though GoCAP is lower across the declared settings.

| VOACAP setting | Case-average MAE |
|---|---:|
{scenario}

The0.6 SSN factor is a legacy-convention sensitivity, not an asserted exact historic-scale reconstruction. The model's SNR50 is monthly climatology across days; truth is detected receptions on particular days. GoCAP was trained on reported receptions and uses completed daily Kp/Ap/F10.7 history, while VOACAP uses smoothed monthly SSN. No controlled hardware match, band-open probability or PHONE calibration is claimed.

Native SNR in dB-Hz is converted to2500Hz by subtracting33.9794dB. The requested7.04MHz column is parsed, not MUF or REL. Average native H/H+1 dB SNR to approximate the reported hourly-bin centre. Using native left/right hours instead gives **{s['voacap_left']['case_mae_db']:.3f}/{s['voacap_right']['case_mae_db']:.3f} dB** case-average MAE. The engine and output parsing are unchanged from the API-validated earlier comparisons.

## Hourly examples and missing data

![Hourly curves against real receptions](results/ml40/years15/hourly_examples.png)

Examples are the first frozen hash IDs from preselected2010/2014/2020/2024, never selected by performance. All **7,200 case-hours** have requested predictions. Only **3,081** have measured truth; **4,119** stay unknown. A missing report is never zero or a closed-path label. Panels use independent vertical scales and all SNRs use2500Hz noise reference.

## Historical context and source sampling

Code0 legacy records account for **{bycontext.get('legacy_code0',0)} cases**, pre-2023 code1 for **{bycontext.get('pre2023_code1',0)}**, and modern ordinary code1 for **{bycontext.get('ordinary_code1',0)}**. Legacy decoder/mode metadata is uncertain; those populations do not imply one verified hard decoding threshold. Code0 is retained only before2020-05-01 in this April test. Code 1 before2023-01-16 can mix WSPR/FST4W metadata as documented by the source. Known other codes are excluded.

Coherent station-pair/day hash sampling uses divisor8 for2010–2015 and128 for2016–2024 to give the sparse early network enough eligible paths. Each request is time/band bounded and sequential; a50,001-row cap causes failure rather than truncation. Only identifier-based selection follows. The equal20/year design is deliberate, not a traffic-weighted score over every report ever received.

## Evidence and reproduction

- [All300 case results](results/ml40/years15/case_results.csv), [per-year results](results/ml40/years15/per_year.csv), [all7,200 hourly comparisons](results/ml40/years15/hourly_comparison.csv).
- [Frozen plan](results/ml40/years15/plan.json), [case freeze](results/ml40/years15/case_freeze.json), [summary](results/ml40/years15/summary.json), [sensitivity](results/ml40/years15/sensitivity.csv).
- [Independent verification](results/ml40/years15/verification.json), [all3,600 native run checksums](results/ml40/years15/run_manifest.csv).

Working inputs stay under ignored `data/ml40/years15/`: exact raw requests/responses and retrieval metadata, canonicalized source features, observed case features, full native scenario Parquet, and per-case native input/output decks. Parent/current/March model hashes and fitting data are unchanged.

```sh
.venv/bin/python -m ml40.years_collect
.venv/bin/python -m ml40.years_prepare
.venv/bin/python -m ml40.years_run
.venv/bin/python -m ml40.years_verify
.venv/bin/python -m ml40.years_report
.venv/bin/python -m unittest discover -v
```

Reproduction requires the cached fitting-date manifests, source inputs and native engine matching frozen hashes, as described in [README](README.md). This run does not train or promote a model. Data credits: WSPRNet volunteer contributors/WSPR Live; GFZ and DRAO/NRCan for solar/geomagnetic context; IAGA/NOAA for magnetic coefficients; WDC-SILSO, Royal Observatory of Belgium, DOI10.24414/qnza-ac80, CC BY-NC for SSN. Sources: [WSPR Live code mapping/access](https://wspr.live/), [VOACAP SNR distribution](https://www.voacap.com/2023/understanding/muf.html), [VOACAP SSN guidance](https://www.voacap.com/2023/understanding/choosingssn.html).

**Share with caveats:** all3,081 measured medians were independently rebuilt from11,050 raw reports; all3,600 native outputs/input power/solar/noise/antenna/coordinate cards were independently checked; source/model hashes, fitting-date exclusions, year/case/hour metrics and year-bootstrap intervals were verified. Actual hardware calibration, unknown-hour accuracy, every-season performance and forward forecasting remain unverified.
'''
    for old,new in {'across15':'across 15','Same3,081':'Same 3,081','bias(prediction':'bias (prediction','with95%':'with 95%','are15':'are 15','only20':'only 20','All300':'All 300','has20':'has 20','around1,000km':'around 1,000 km','distance-balanced2025':'distance-balanced 2025','in2010':'in 2010','April6,13,20 and27':'April 6, 13, 20 and 27','selected20':'selected 20','least6':'least 6','hours,20':'hours, 20','and100':'and 100','Method30':'Method 30','at7.04MHz':'at 7.04 MHz','dipoles10m':'dipoles 10 m','dielectric4':'dielectric 4','conductivity0.001S/m':'conductivity 0.001 S/m','parameter150':'parameter 150','takeoff3°':'takeoff 3°','Version2':'Version 2','The0.6':'The 0.6','to2500Hz':'to 2500 Hz','subtracting33.9794dB':'subtracting 33.9794 dB','requested7.04MHz':'requested 7.04 MHz','preselected2010':'preselected 2010','use2500Hz':'use 2500 Hz','Code0':'Code 0','code1':'code 1','before2020':'before 2020','before2023':'before 2023','divisor8':'divisor 8','for2010':'for 2010','and128':'and 128','for2016':'for 2016','a50,001':'a 50,001','equal20':'equal 20','All300':'All 300','all7,200':'all 7,200','all3,600':'all 3,600','all3,081':'all 3,081','from11,050':'from 11,050','DOI10':'DOI 10'}.items():text=text.replace(old,new)
    (ROOT/'VOACAP_15YEAR_COMPARISON.md').write_text(text);print('15-year report saved.')
if __name__=='__main__':report()
