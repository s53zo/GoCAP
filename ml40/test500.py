"""Frozen, outcome-blind 500-case comparison of the portable model and receptions.

Run: python -m ml40.test500
A case is a fixed station pair in fixed four-character squares on one UTC day.
These are subsets of the previously released held-out dates, not new validation.
"""
import hashlib
import json
from collections import deque
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .plan import ROOT, DATA, RESULTS
from .inference import scenario_frame

OUT=RESULTS/'test500'
KEYS=['source','mode','date_key','tx_sign','rx_sign','tx_grid','rx_grid']
QUOTAS={'WSPR':200,'FT8':100,'FT4':100,'CW':100}
SEED='gocap-500-v1-seed40'

def digest(path):
    with open(path,'rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def case_id(key):
    return hashlib.sha256(json.dumps(list(key),separators=(',',':')).encode()).hexdigest()[:20]

def select_cases(d,quotas=QUOTAS):
    """Use only identifiers, geometry and observed-hour availability, never SNR."""
    candidates=d.groupby(KEYS,sort=True).agg(observed_hours=('hour','nunique'),
        distance_km=('distance_km','first')).reset_index()
    candidates=candidates[(candidates['mode']!='WSPR')|candidates.observed_hours.ge(6)].copy()
    candidates['distance_band']=pd.cut(candidates.distance_km,[-1,1000,3000,6000,20001],
        labels=['under1000','1000to3000','3000to6000','over6000']).astype(str)
    candidates['case_id']=[case_id(k) for k in candidates[KEYS].itertuples(index=False,name=None)]
    candidates['selection_hash']=candidates.case_id.map(lambda k:hashlib.sha256((SEED+k).encode()).hexdigest())
    selected=[]
    for mode,quota in quotas.items():
        c=candidates[candidates['mode'].eq(mode)].sort_values('selection_hash')
        queues=[deque(g.to_dict('records')) for _,g in c.groupby(['date_key','distance_band'],sort=True)]
        seen=set();n=0
        while n<quota:
            progressed=False
            for q in queues:
                while q:
                    row=q.popleft();path=(row['tx_grid'],row['rx_grid'])
                    if path in seen:continue
                    seen.add(path);selected.append(row);n+=1;progressed=True;break
                if n==quota:break
            if not progressed:raise ValueError(f'Insufficient unique paths for {mode}: {n}/{quota}')
    return pd.DataFrame(selected).sort_values(['mode','case_id']).reset_index(drop=True)

def rank_metrics(actual,prediction):
    if len(actual)<6 or len(np.unique(actual))<2 or len(np.unique(prediction))<2:
        return None,None
    # Average ranks handle ties, including plateaus in tree predictions.
    corr=float(pd.Series(actual).rank().corr(pd.Series(prediction).rank()))
    loss=float(np.max(actual)-actual[int(np.argmax(prediction))])
    return corr,loss

def load_evidence():
    main=pd.read_parquet(DATA/'features.parquet')
    other=main[main.split.eq('test')&main['mode'].isin(['FT8','FT4','CW'])].copy()
    wspr=pd.concat([pd.read_parquet(p) for p in sorted((DATA/'ranking'/'processed').glob('cohort-2025-*.parquet'))],ignore_index=True)
    wspr['date_key']=wspr.hour.dt.strftime('%Y-%m-%d')
    wspr['split']='test'
    wspr['power_known']=wspr.power_dbm.notna().astype(float)
    wspr['legacy_decoder']=wspr.decoder_context.ne('ordinary_code1').astype(float)
    d=pd.concat([other,wspr],ignore_index=True)
    assert d.split.eq('test').all() and d.hour.dt.year.eq(2025).all()
    assert not set(d.date_key)&set(main.loc[~main.split.eq('test'),'date_key'])
    assert d.tx_grid.str.fullmatch('[A-R]{2}[0-9]{2}').all()
    assert d.rx_grid.str.fullmatch('[A-R]{2}[0-9]{2}').all()
    return d

def summarize(cases,hours):
    out=[]
    for mode in ['ALL']+list(QUOTAS):
        c=cases if mode=='ALL' else cases[cases['mode'].eq(mode)]
        h=hours if mode=='ALL' else hours[hours['mode'].eq(mode)]
        valid=c.ranking_spearman.notna()
        err=h.prediction_db-h.actual_db
        out.append({'mode':mode,'cases':len(c),'observed_hours':len(h),'dates':c.date_key.nunique(),
            'case_balanced_mae_db':float(c.mae_db.mean()),'hour_weighted_mae_db':float(abs(err).mean()),
            'rmse_db':float(np.sqrt(np.mean(err**2))),'bias_db':float(err.mean()),
            'median_absolute_error_db':float(abs(err).median()),'p90_absolute_error_db':float(abs(err).quantile(.9)),
            'within_3db':float((abs(err)<=3).mean()),'within_6db':float((abs(err)<=6).mean()),
            'interval_coverage':float(h.covered.mean()),'mean_interval_width_db':float((h.upper_db-h.lower_db).mean()),
            'baseline_case_balanced_mae_db':float(c.baseline_mae_db.mean()),
            'scenario_api_case_balanced_mae_db':float(c.scenario_mae_db.mean()),
            'rankable_cases':int(valid.sum()),
            'median_spearman':float(c.loc[valid,'ranking_spearman'].median()) if valid.any() else None,
            'baseline_median_spearman':float(c.loc[valid,'baseline_spearman'].median()) if valid.any() else None,
            'mean_best_observed_hour_shortfall_db':float(c.loc[valid,'best_hour_shortfall_db'].mean()) if valid.any() else None})
    return out

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    artifact=ROOT/'artifacts/gocap-40m.joblib'
    metadata=json.loads((ROOT/'artifacts/gocap-40m.json').read_text())
    assert digest(artifact)==metadata['sha256']
    training=json.loads((RESULTS/'training_run.json').read_text())
    baseline_info=training['scenarios']['standard']['files']['boosted_baseline']
    baseline_path=DATA/baseline_info['path'];assert digest(baseline_path)==baseline_info['sha256']
    d=load_evidence();selected=select_cases(d)
    assert len(selected)==500 and selected.case_id.nunique()==500
    assert selected.groupby('mode').size().to_dict()==QUOTAS
    plan={'version':1,'seed':SEED,'quotas':QUOTAS,'model_sha256':digest(artifact),
        'baseline_sha256':digest(baseline_path),'main_evidence_sha256':digest(DATA/'features.parquet'),
        'ranking_evidence_sha256':{p.name:digest(p) for p in sorted((DATA/'ranking'/'processed').glob('cohort-2025-*.parquet'))},
        'case_ids':selected.case_id.tolist(),
        'selection':'Round-robin UTC date/distance strata; seeded hash within stratum; unique directed square pair within each mode; WSPR >=6 observed hours; other modes >=1. No SNR or prediction used.',
        'reuse':'Previously released 2025 test dates; not a new independent holdout or a model-selection run.'}
    path=OUT/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Frozen suite changed')
    path.write_text(json.dumps(plan,indent=2)) # Freeze membership before any prediction.
    evidence=d.merge(selected[KEYS+['case_id']],on=KEYS,how='inner',validate='many_to_one')
    evidence=evidence.sort_values(['case_id','hour','power_dbm']).reset_index(drop=True)
    assert evidence.case_id.nunique()==500 and evidence.snr2500.notna().all()
    evidence.to_parquet(OUT/'inputs.parquet',index=False)
    bundle=joblib.load(artifact);baseline=joblib.load(baseline_path)
    with threadpool_limits(limits=4):
        prediction=bundle['model'].predict(evidence,station_correction=False)
        base_prediction=baseline['model'].predict(evidence,station_correction=False)
    # Exercise the portable API scenario assumptions separately from full measured history.
    frames=[scenario_frame(r.tx_grid,r.rx_grid,r.hour,r.sfi_previous_day,r.kp_previous,
        r.Ap_previous_day,r.mode,10**((r.power_dbm-30)/10) if r.mode=='WSPR' else 100)
        for r in evidence.itertuples()]
    scenario=pd.concat(frames,ignore_index=True)
    with threadpool_limits(limits=4):scenario_prediction=bundle['model'].predict(scenario,station_correction=False)
    widths=np.array([bundle['calibration']['population_core'][r.source+'|'+r.mode]['half_width_db'] for r in evidence.itertuples()])
    # A common 1 W reference removes reported WSPR power changes when ranking hours.
    adjustment=np.where(evidence['mode'].eq('WSPR'),30-evidence.power_dbm.fillna(30),0)
    obs=evidence[KEYS+['case_id','hour','power_dbm','snr2500','reports','distance_km',
        'kp_previous','ap_previous','ap_trailing24','sfi_previous_day','Ap_previous_day']].copy()
    obs['actual_db']=evidence.snr2500+adjustment
    obs['prediction_db']=prediction+adjustment;obs['baseline_db']=base_prediction+adjustment
    obs['scenario_db']=scenario_prediction+adjustment
    obs['lower_db']=obs.prediction_db-widths;obs['upper_db']=obs.prediction_db+widths
    obs.to_csv(OUT/'observations.csv',index=False)
    hours=obs.groupby(['case_id','hour'],sort=True).agg(actual_db=('actual_db','median'),
        prediction_db=('prediction_db','median'),baseline_db=('baseline_db','median'),
        scenario_db=('scenario_db','median'),lower_db=('lower_db','median'),upper_db=('upper_db','median'),
        reports=('reports','sum')).reset_index()
    hours=hours.merge(selected[['case_id','mode']],on='case_id',validate='many_to_one')
    hours['covered']=hours.actual_db.between(hours.lower_db,hours.upper_db)
    hours['error_db']=hours.prediction_db-hours.actual_db
    hours.to_csv(OUT/'hourly_comparisons.csv',index=False)
    rows=[]
    for case in selected.to_dict('records'):
        h=hours[hours.case_id.eq(case['case_id'])].sort_values('hour')
        a=h.actual_db.to_numpy();p=h.prediction_db.to_numpy();b=h.baseline_db.to_numpy()
        corr,loss=rank_metrics(a,p);bcorr,bloss=rank_metrics(a,b)
        case.update({'observed_hours':len(h),'mae_db':float(np.mean(abs(p-a))),
            'rmse_db':float(np.sqrt(np.mean((p-a)**2))),'bias_db':float(np.mean(p-a)),
            'baseline_mae_db':float(np.mean(abs(b-a))),
            'scenario_mae_db':float(np.mean(abs(h.scenario_db.to_numpy()-a))),
            'interval_coverage':float(h.covered.mean()),'ranking_spearman':corr,
            'baseline_spearman':bcorr,'best_hour_shortfall_db':loss,
            'baseline_best_hour_shortfall_db':bloss})
        rows.append(case)
    cases=pd.DataFrame(rows);cases.to_csv(OUT/'cases.csv',index=False)
    summary=summarize(cases,hours)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    verify_outputs(evidence,cases,hours,summary,artifact)
    write_readout(cases,hours,summary)
    print(json.dumps(summary,indent=2))

def verify_outputs(evidence,cases,hours,summary,artifact):
    """Independent metric arithmetic and input membership checks; no new fitting."""
    from .verify import manual_spearman
    assert len(cases)==500 and cases.case_id.nunique()==500
    assert not cases.duplicated(['mode','tx_grid','rx_grid']).any()
    assert len(hours)==int(cases.observed_hours.sum())
    assert not hours.duplicated(['case_id','hour']).any()
    assert set(evidence.case_id)==set(cases.case_id)
    assert set(evidence.date_key)==set(cases.date_key)
    for row in cases.itertuples():
        h=hours[hours.case_id.eq(row.case_id)].sort_values('hour')
        errors=[float(p)-float(a) for p,a in zip(h.prediction_db,h.actual_db)]
        assert np.isclose(sum(map(abs,errors))/len(errors),row.mae_db)
        if pd.notna(row.ranking_spearman):
            assert np.isclose(manual_spearman(h.actual_db,h.prediction_db),row.ranking_spearman)
            i=int(np.argmax(h.prediction_db.to_numpy()))
            assert np.isclose(max(h.actual_db)-h.actual_db.iloc[i],row.best_hour_shortfall_db)
    for s in summary:
        c=cases if s['mode']=='ALL' else cases[cases['mode'].eq(s['mode'])]
        h=hours if s['mode']=='ALL' else hours[hours['mode'].eq(s['mode'])]
        assert np.isclose(sum(c.mae_db)/len(c),s['case_balanced_mae_db'])
        covered=sum(lo<=a<=hi for lo,a,hi in zip(h.lower_db,h.actual_db,h.upper_db))
        assert np.isclose(covered/len(h),s['interval_coverage'])
    # Recompute persisted predictions from the selected artifact, not earlier result tables.
    with threadpool_limits(limits=4):p=joblib.load(artifact)['model'].predict(evidence,station_correction=False)
    saved=pd.read_csv(OUT/'observations.csv')
    adjustment=np.where(evidence['mode'].eq('WSPR'),30-evidence.power_dbm.fillna(30),0)
    np.testing.assert_allclose(saved.prediction_db.to_numpy(),p+adjustment,atol=1e-10)
    np.testing.assert_allclose(saved.actual_db.to_numpy(),evidence.snr2500+adjustment,atol=1e-10)
    verification={'status':'verified','cases':500,'hourly_comparisons':len(hours),
        'raw_hourly_aggregates':len(evidence),'model_sha256':digest(artifact),
        'checks':['frozen outcome-blind membership','unique directed square paths within each mode',
            'test-only calendar dates','independently recomputed errors and interval coverage',
            'independently recomputed tied-rank correlations and hour-selection loss',
            'portable artifact predictions rerun and reconciled to saved observations'],
        'output_sha256':{name:digest(OUT/name) for name in ['plan.json','cases.csv','observations.csv','hourly_comparisons.csv','summary.json','inputs.parquet']}}
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2))

def write_readout(cases,hours,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    all_result=summary[0]
    lines=['# 500 propagation comparisons — 40m','',
        f'Completed **500 distinct path/day cases** covering **{len(hours):,} observed hours** on '
        f'**{cases.date_key.nunique()} UTC dates in 2025**. Each mode has unique directed four-character square pairs. '
        'The suite reruns the exported model against recorded positive receptions. '
        'These are outcome-blind subsets of the previously published test dates, not fresh independent validation.','',
        f'Each case receives equal weight: overall MAE is **{all_result["case_balanced_mae_db"]:.2f} dB** '
        f'(boosted baseline: **{all_result["baseline_case_balanced_mae_db"]:.2f} dB**). '
        'The adjusted model does not improve overall SNR accuracy on this suite. '
        'This deliberately balanced mode mix is not a population-wide accuracy estimate.','',
        '| Mode | Cases | Observed hours | Dates | Case-balanced MAE | Baseline MAE | Within 6 dB (hours) | 80% interval coverage (hours) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summary[1:]:
        lines.append(f'| {s["mode"]} | {s["cases"]} | {s["observed_hours"]:,} | {s["dates"]} | {s["case_balanced_mae_db"]:.2f} dB | {s["baseline_case_balanced_mae_db"]:.2f} dB | {s["within_6db"]:.1%} | {s["interval_coverage"]:.1%} |')
    w=next(s for s in summary if s['mode']=='WSPR')
    lines+=['',f'WSPR hourly ranking across {w["rankable_cases"]} eligible cases has median Spearman '
        f'correlation **{w["median_spearman"]:.3f}**, versus **{w["baseline_median_spearman"]:.3f}** for the baseline. '
        f'The model-selected hour is on average **{w["mean_best_observed_hour_shortfall_db"]:.2f} dB** below '
        'the strongest observed hour. Only reported hours compete; nothing is inferred about missing hours.','',
        '![Observed versus predicted SNR](results/ml40/test500/comparison.png)','',
        '## Definitions and limits','',
        '- One case is a fixed transmitting/receiving station pair, in fixed squares, on one UTC day. '
        'The 500 cases do not represent 500 independent days or controlled experiments.',
        '- Selection cycles through available date/distance strata with a seeded identifier hash; '
        'it never uses SNR or prediction error. WSPR requires six observed hours; other modes require one. '
        'This favors active observed paths, and is conditional on reception.',
        '- Actual SNR is the source hourly aggregate in dB relative to 2500 Hz noise bandwidth. '
        'WSPR actuals and predictions are both shifted using reported power to a common **1 W reference** '
        'for hourly comparisons. This does not remove unknown antennas or noise. Multiple power records '
        'within an hour are collapsed by median. Other modes retain their unknown power.',
        '- MAE averages absolute errors within each case, then averages cases. Hour-weighted MAE, '
        'RMSE, bias and the 90th-percentile absolute error are in summary.json. '
        'Interval coverage counts observed hours; intervals describe reported receptions.',
        '- Primary predictions use actual historical preceding Kp/ap/Ap/SFI features. '
        'A separate portable-scenario test uses hourly Kp held steady for the preceding day, plus '
        'the recorded preceding-day SFI and Ap. These scenario assumptions can change predictions.',
        '- The FT8/FT4 panel covers three dates and CW covers two. Their sparse observations do not '
        'support the same hourly-ranking claims as WSPR. No measured PHONE observations are available; '
        'PHONE margins remain theoretical.',
        '- A good fit to received signals does not establish a calibrated 100 W/dipole reference, '
        'the unseen weak-signal distribution, or a probability of communication.','',
        '## Portable scenario-input check','',
        '| Mode | Full historical inputs MAE | Steady-Kp scenario MAE |','|---|---:|---:|']
    for s in summary[1:]:lines.append(f'| {s["mode"]} | {s["case_balanced_mae_db"]:.2f} dB | {s["scenario_api_case_balanced_mae_db"]:.2f} dB |')
    lines+=['','## Reproduce and inspect','', 'Run `.venv/bin/python -m ml40.test500`. '
        'Membership, source checksums and model checksum are frozen before prediction. '
        'The script fails if the frozen membership or artifact changes; it does not retrain.', '',
        '- [All 500 cases and per-case results](results/ml40/test500/cases.csv)',
        '- [Every observed hour and prediction](results/ml40/test500/hourly_comparisons.csv)',
        '- [Source aggregates, power and solar inputs](results/ml40/test500/observations.csv)',
        '- [Exact model input snapshot](results/ml40/test500/inputs.parquet)',
        '- [Summary metrics](results/ml40/test500/summary.json)',
        '- [Frozen selection and source checksums](results/ml40/test500/plan.json)',
        '- [Independent arithmetic and prediction verification](results/ml40/test500/verification.json)',
        '- [Larger experiment and archive provenance](ML40_RESULTS.md)','']
    (ROOT/'TEST500_RESULTS.md').write_text('\n'.join(lines))
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    colors={'WSPR':'#00796B','FT8':'#315BA8','FT4':'#AD5900','CW':'#8B398A'}
    for mode,color in colors.items():
        h=hours[hours['mode'].eq(mode)]
        axes[0].scatter(h.actual_db,h.prediction_db,s=10,alpha=.35,label=mode,color=color)
    low=min(hours.actual_db.min(),hours.prediction_db.min())-2
    high=max(hours.actual_db.max(),hours.prediction_db.max())+2
    axes[0].plot([low,high],[low,high],color='#333333',linestyle='--',label='Perfect agreement')
    axes[0].set(xlim=(low,high),ylim=(low,high),xlabel='Observed SNR (dB / 2500 Hz)',ylabel='Predicted SNR (dB / 2500 Hz)',title=f'{len(hours):,} positive-reception hours')
    axes[0].legend(fontsize=8,frameon=False)
    for mode,color in colors.items():
        v=np.sort(cases.loc[cases['mode'].eq(mode),'mae_db'].to_numpy())
        axes[1].plot(v,np.arange(1,len(v)+1)/len(v),label=mode,color=color)
    axes[1].set(xlabel='Per-case mean absolute error (dB)',ylabel='Fraction of cases at or below error',ylim=(0,1),title='500 cases; equal weight per case')
    axes[1].legend(frameon=False)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)

if __name__=='__main__':run()
