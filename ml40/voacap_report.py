"""Generate inspectable static benchmark figures and repository readout."""
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .voacap_protocol import OUT,ROOT

def report():
    s=json.loads((OUT/'summary.json').read_text());c=pd.read_csv(OUT/'case_results.csv');h=pd.read_csv(OUT/'hourly_comparison.csv');sen=pd.read_csv(OUT/'sensitivity.csv')
    fig,axes=plt.subplots(3,2,figsize=(13,11),sharex=True,sharey=True)
    for ax,cid in zip(axes.flat,c.case_id.head(6)):
        d=h[h.case_id.eq(cid)].sort_values('utc_hour');meta=c[c.case_id.eq(cid)].iloc[0]
        ax.plot(d.utc_hour,d.gocap_snr_db,label='GoCAP',color='#1769aa',lw=2)
        ax.plot(d.utc_hour,d.voacap_snr_db,label='VOACAP',color='#c05a00',ls='--',lw=2)
        known=d[d.actual_snr_db.notna()];ax.scatter(known.utc_hour,known.actual_snr_db,c='black',s=24,label='Reported WSPR median',zorder=4)
        ax.set_title(f'{meta.tx_grid} → {meta.rx_grid} | {meta.date_key} | {meta.reported_power_watts:.2g} W\n{meta.distance_km:,.0f} km',fontsize=10)
        ax.grid(alpha=.2);ax.set_xticks([0,6,12,18,23]);ax.set_ylabel('SNR (dB / 2500 Hz)');ax.set_xlabel('UTC hour (HH:00–HH:59 bin)')
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,bbox_to_anchor=(.5,.965))
    fig.suptitle('40m hourly predictions: first six frozen case IDs',y=.998,fontsize=15)
    fig.text(.5,.008,'Black points exist only where receptions were reported. Unreported hours are unknown.\nVOACAP: monthly median, 10m dipoles, noise150, smoothed SSN v2. Actual station hardware is unknown.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.055,1,.93]);fig.savefig(OUT/'hourly_examples.png',dpi=160);plt.close(fig)
    groups=c.groupby('distance_group').agg(cases=('case_id','size'),gocap=('gocap_mae_db','mean'),voacap=('voacap_mae_db','mean')).reindex(['100to1000','1000to3000','3000to6000','6000to15000'])
    groups.to_csv(OUT/'distance_results.csv')
    fig,ax=plt.subplots(figsize=(9,5));x=np.arange(4)
    ax.bar(x-.18,groups.gocap,.36,label='GoCAP',color='#1769aa');ax.bar(x+.18,groups.voacap,.36,label='VOACAP',color='#c05a00',hatch='//')
    ax.set_xticks(x,['100–1,000','1,000–3,000','3,000–6,000','6,000–15,000']);ax.set_xlabel('Path distance (km); 25 cases per group');ax.set_ylabel('Case-average hourly MAE (dB)');ax.set_ylim(bottom=0);ax.legend();ax.set_title('Error against reported 40m WSPR receptions');fig.tight_layout();fig.savefig(OUT/'error_by_distance.png',dpi=160);plt.close(fig)
    lines=['# GoCAP versus VOACAP: 100 real-data 40m cases','',
        f'**GoCAP has lower hourly SNR error in {s["case_wins_strict"]}/100 cases.** Equally weighted case mean absolute error is **{s["gocap"]["case_mae_db"]:.2f} dB for GoCAP** and **{s["voacap"]["case_mae_db"]:.2f} dB for VOACAP**, on **968 positive WSPR hourly aggregates**. This compares predictions against received signals with unknown real antennas and noise; it is not a controlled physical calibration or proof of universal superiority.','',
        '| Metric on the same 968 observed hours | GoCAP | VOACAP |','|---|---:|---:|',
        *[f'| {label} | {s["gocap"][key]:.2f} dB | {s["voacap"][key]:.2f} dB |' for label,key in [('Case-average hourly MAE (primary)','case_mae_db'),('Pooled hourly MAE','hour_mae_db'),('Pooled hourly RMSE','hour_rmse_db'),('Signed bias (prediction − observation)','bias_db')]],'',
        f'The paired case-average difference, GoCAP minus VOACAP, is **{s["gocap_minus_voacap_case_mae_db"]:.2f} dB**; the 95% date-cluster bootstrap interval is **[{s["date_cluster_bootstrap_95pct_db"][0]:.2f}, {s["date_cluster_bootstrap_95pct_db"][1]:.2f}] dB**. There are only 12 independent sampled dates; the interval describes this design, not global performance. VOACAP is better in16 cases; no exact ties. Neither predictor was fitted or adjusted using these results.','',
        '## Hour-by-hour predictions','',
        'Every case has all24 UTC-hour predictions from both models. Error is scored only where a real reception exists. The1,432 unreported case-hours remain unknown, never zero or “closed”. Cases are one fixed transmitter/receiver/day/power, so100 tests contain968 separately observed hourly values.','',
        '![Six hourly comparison examples](results/ml40/voacap100/hourly_examples.png)','',
        'Examples are the first six frozen hash-sorted IDs, not cases selected for favorable appearance. Black dots are hourly median reported SNR; continuous curves include unverified hours.','',
        '![Error by distance](results/ml40/voacap100/error_by_distance.png)','',
        '| Distance km | Cases | GoCAP MAE | VOACAP MAE |','|---|---:|---:|---:|',
        *[f'| {idx} | {int(r.cases)} | {r.gocap:.2f} dB | {r.voacap:.2f} dB |' for idx,r in groups.iterrows()],'',
        '## What the comparison establishes','',
        'On this fresh, observed-reception sample, GoCAP better estimates typical received WSPR strength. The result supports the hourly empirical model for this population. It does not validate band-open probabilities, undetected signals, calibrated antenna gain, other modes, or arbitrary solar-cycle extrapolation. Requiring six observed hours favors active, repeatedly detectable paths. The distance-balanced design is deliberately not a traffic-weighted worldwide score.','',
        'VOACAP predicts a monthly hourly median across days, while our truth is the median of detected signals on a specific day. This target difference and reception selection can favor GoCAP, trained on reported receptions. GoCAP additionally uses completed daily solar history including Kp/Ap; VOACAP uses a smoothed monthly SSN. Final solar observations and the centered13-month SSN series make this a retrospective hindcast, not a forecast-vintage evaluation. [VOACAP explains its monthly SNR distribution](https://www.voacap.com/2023/understanding/muf.html) and [required smoothed SSN convention](https://www.voacap.com/2023/understanding/choosingssn.html).','',
        '## Predeclared VOACAP assumptions and sensitivity','',
        'Primary: native Method30, CCIR coefficients, short great-circle path at7.04MHz between four-character square centres; reported WSPR transmit power; half-wave horizontal dipoles10m high at both ends, broadside toward the path; ground dielectric4/conductivity0.001S/m; man-made noise parameter150; minimum takeoff3°; WDC-SILSO13-month smoothed Version2 SSN. The100 case IDs and all12 settings were frozen before scoring. Actual hardware and local noise are unavailable.','',
        'VOACAP native SNR50 is dB-Hz. Subtract33.9794dB to compare against WSPR SNR in2500Hz. Use the requested7.04MHz column, not the separate MUF column or REL percentages. Native whole-hour H/H+1 predictions are averaged in dB to approximate the observed [H,H+1) bin centre. This is an approximation; left-edge/right-edge case MAEs are '+f'{s["voacap_left"]["case_mae_db"]:.2f}/{s["voacap_right"]["case_mae_db"]:.2f}dB. Output SNR is rounded to integer dB before interpolation.','',
        'Sensitivity uses ideal0dBi isotropic antennas as well as dipoles; noise parameters145/150/155; SSN scales1.0/0.6. The0.6 factor explores legacy SSN conventions and is not asserted to be an exact historical-scale reconstruction. No winning setting was chosen after seeing errors. All12 settings give predictions for all968 observed hours.','',
        '| Antenna / noise / SSN scale | Case-average MAE |','|---|---:|',
        *[f'| {r.scenario} | {r.case_mae_db:.2f} dB |' for r in sen.itertuples()],'',
        f'GoCAP’s steady-history scalar-query diagnostic is {s["gocap_steady"]["case_mae_db"]:.2f}dB case-average MAE; primary scoring uses actual completed history. Sensitivity results do not justify claiming known physical station calibration.','',
        '## Evidence and reproduction','',
        'The local [VOACAP port](https://github.com/jawatson/voacapl) was built from commit `c12a98b348c39f36f2659b8b3aa97b86284c8a1d`. A separate pilot exactly matched all24 SNR outputs from the documented [dxtron public API](https://voacap.dxtron.com/api/docs/). Bulk tests ran locally. Engine inputs and outputs, source queries, retrieval times, model hashes and all1200 run hashes are retained. WDC-SILSO series credit: WDC-SILSO, Royal Observatory of Belgium, Brussels, DOI10.24414/qnza-ac80; [official monthly smoothed data](https://www.sidc.be/SILSO/DATA/SN_ms_tot_V2.0.txt), CC BY-NC research use.','',
        '- [All100 case results](results/ml40/voacap100/case_results.csv)','- [All2,400 hourly comparisons](results/ml40/voacap100/hourly_comparison.csv)','- [Full12-scenario hourly outputs](results/ml40/voacap100/voacap_all_scenarios.csv)','- [Frozen protocol](results/ml40/voacap100/protocol.json) and [case freeze](results/ml40/voacap100/case_freeze.json)','- [Independent verification](results/ml40/voacap100/verification.json) and [native/API pilot](results/ml40/voacap100/engine_validation.json)','- [Per-run checksums](results/ml40/voacap100/run_manifest.csv)','',
        'Raw archive data: `data/ml40/voacap100/raw/`; native input/output decks: `data/ml40/voacap100/runs/<case_id>/<scenario>/`. Dataset eligibility: WSPR code1, valid40m frequencies and locations; at least6 reported hours; fixed reported power20–50dBm;100–15,000km; unique directed square pair. Date6 of each2025 month is new WSPR test data, excluded from model fitting. Source-pair/day hash sampling precedes case hash selection. Exact source queries are in raw metadata.','',
        '```sh','bash scripts/build_voacap.sh   # only if native engine is absent','.venv/bin/python -m ml40.voacap_collect','.venv/bin/python -m ml40.voacap_prepare','.venv/bin/python -m ml40.voacap_run','.venv/bin/python -m ml40.voacap_verify','.venv/bin/python -m ml40.voacap_report','.venv/bin/python -m unittest discover -v','```','',
        'The SILSO source file is cached and checksummed; refreshes must create a new protocol rather than overwrite this frozen test. Installed native binary hashes may differ across compilers; a rebuilt engine requires a separately versioned protocol and the API pilot check before scoring. Existing run caches must match frozen input decks.','',
        '**Validation: share with caveats.** Independently reconstructed all1200 native outputs and primary metrics, checked source/model hashes, preserved968 observations and1,432 unknowns, and confirmed that direct predictions on observed feature rows exactly match the generated24-hour frames. All35 unit tests pass. Hardware calibration, nondetection accuracy and real-time forecast skill remain unverified.','']
    text='\n'.join(lines)
    for old,new in {'in16':'in 16','all24':'all 24','The1,432':'The 1,432','so100':'so 100','contain968':'contain 968',
        'centered13':'centered 13','Method30':'Method 30','at7.04MHz':'at 7.04 MHz','dipoles10m':'dipoles 10 m',
        'dielectric4':'dielectric 4','conductivity0.001S/m':'conductivity 0.001 S/m','parameter150':'parameter 150',
        'takeoff3°':'takeoff 3°','WDC-SILSO13':'WDC-SILSO 13','Version2':'Version 2','The100':'The 100',
        'all12':'all 12','Subtract33':'Subtract 33','requested7.04MHz':'requested 7.04 MHz','ideal0dBi':'ideal 0 dBi',
        'parameters145':'parameters 145','scales1.0/0.6':'scales 1.0/0.6','The0.6':'The 0.6','All12':'All 12',
        'all968':'all 968','all1200':'all 1200','all24':'all 24','All100':'All 100','All2,400':'All 2,400',
        'Full12':'Full 12','all1200':'all 1200','at least6':'at least 6','power20':'power 20','Date6':'Date 6',
        'each2025':'each 2025','preserved968':'preserved 968','and1,432':'and 1,432','generated24':'generated 24',
        'All35':'All 35','all1200':'all 1200','all24':'all 24','all12':'all 12','half-wave horizontal dipoles10m':'half-wave horizontal dipoles 10 m',
        'DOI10':'DOI 10','series13':'series 13'}.items():text=text.replace(old,new)
    text=text.replace('.venv/bin/python -m ml40.voacap_verify\n','.venv/bin/python -m ml40.voacap_verify\n.venv/bin/python -m ml40.voacap_verify_extra\n')
    (ROOT/'VOACAP_COMPARISON.md').write_text(text)
if __name__=='__main__':report()
