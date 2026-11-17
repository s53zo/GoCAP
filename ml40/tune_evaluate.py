"""Release fresh test scores only after tuning freeze, with unchanged reference."""
import json
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from .plan import ROOT,DATA,RESULTS
from .tune import OUT,CACHE,RankPanel,utc,save_json
from .tune_collect import DATES
from .models import sample_weights
from .test500 import digest


def metric(d,bundle):
    p=bundle['model'].predict(d,station_correction=False)
    err=p-d.snr2500.to_numpy();w=sample_weights(d)
    half=np.array([bundle['calibration']['population_core'][r.source+'|'+r.mode]['half_width_db'] for r in d.itertuples()])
    return {'rows':len(d),'days':d.date_key.nunique(),'balanced_mae_db':float(np.average(abs(err),weights=w)),
        'mae_db':float(abs(err).mean()),'rmse_db':float(np.sqrt(np.mean(err**2))),
        'bias_db':float(err.mean()),'interval_coverage':float((abs(err)<=half).mean()),
        'mean_interval_width_db':float(2*half.mean())},p,half


def bootstrap_rank(reference,tuned,replicates=5000):
    paired=reference.rename(columns={'spearman':'reference'}).merge(
        tuned[['case_index','date','spearman']].rename(columns={'spearman':'tuned'}),on=['case_index','date'],validate='one_to_one')
    paired=paired.dropna(subset=['reference','tuned'])
    dates=sorted(set(paired.date));groups=[g for _,g in paired.groupby('date',sort=True)]
    rng=np.random.default_rng(40);deltas=[]
    for _ in range(replicates):
        indexes=rng.integers(0,len(groups),len(groups))
        a=np.concatenate([groups[i].reference.to_numpy() for i in indexes])
        b=np.concatenate([groups[i].tuned.to_numpy() for i in indexes])
        deltas.append(float(np.median(b)-np.median(a)))
    return {'statistic':'median case Spearman(tuned) minus median case Spearman(reference)',
        'paired_cases':len(paired),'independent_date_clusters':len(dates),'replicates':replicates,
        'difference':float(paired.tuned.median()-paired.reference.median()),
        '95_percent_date_bootstrap_interval':[float(x) for x in np.quantile(deltas,[.025,.975])]},paired


def regression500(reference,tuned):
    base=RESULTS/'test500';d=pd.read_parquet(base/'inputs.parquet')
    old=pd.read_csv(base/'cases.csv');hours=pd.read_csv(base/'hourly_comparisons.csv')
    shift=np.where(d['mode'].eq('WSPR'),30-d.power_dbm.fillna(30),0)
    w=d[['case_id','hour','mode']].copy()
    w['actual']=d.snr2500.to_numpy()+shift
    w['reference']=reference['model'].predict(d,station_correction=False)+shift
    w['tuned']=tuned['model'].predict(d,station_correction=False)+shift
    h=w.groupby(['case_id','hour','mode'],sort=True)[['actual','reference','tuned']].median().reset_index()
    cases=h.groupby(['case_id','mode'],sort=True).apply(lambda g:pd.Series({
        'reference_mae_db':float(abs(g.reference-g.actual).mean()),
        'tuned_mae_db':float(abs(g.tuned-g.actual).mean())})).reset_index()
    assert len(cases)==500 and set(cases.case_id)==set(old.case_id)
    np.testing.assert_allclose(cases.set_index('case_id').reference_mae_db.sort_index(),
        old.set_index('case_id').mae_db.sort_index(),atol=1e-10)
    cases.to_csv(OUT/'regression500_cases.csv',index=False)
    return {mode:{'cases':len(g),'reference_mae_db':float(g.reference_mae_db.mean()),
        'tuned_mae_db':float(g.tuned_mae_db.mean())} for mode,g in cases.groupby('mode')}


def evaluate():
    run=json.loads((OUT/'run.json').read_text());assert run['phase'] in ['frozen_before_fresh_test','evaluated']
    assert digest(DATA/'fit_features.parquet')==run['fit_sha256']
    reference_path=ROOT/'artifacts/gocap-40m.joblib';assert digest(reference_path)==run['reference_sha256']
    if run['chosen'] is None:
        run['phase']='evaluated';run['evaluated_utc']=utc();run['promoted']=False
        run['decision']='No finalist passed validation selection; keep the reference. Fresh test not used.'
        save_json(OUT/'run.json',run)
        save_json(OUT/'verification.json',{'status':'verified','selected_model':None,
            'fresh_test_scored':False,'reference_sha256_unchanged':digest(reference_path),
            'checks':['fitting and reference checksums','all full-data finalists failed the prespecified selection rule'],
            'output_sha256':{name:digest(OUT/name) for name in ['plan.json','screen_results.json','run.json']}})
        write_report(run);print(run['decision']);return
    tuned_path=ROOT/'artifacts/gocap-40m-tuned.joblib';assert digest(tuned_path)==run['tuned_sha256']
    # Enforce all prespecified fresh dates; no evaluation on a convenient partial transfer.
    for date in DATES:
        for kind in ['snr','ranking']:
            assert (CACHE/'fresh'/'processed'/f'{kind}-{date}.parquet').exists()
    d=pd.concat([pd.read_parquet(CACHE/'fresh'/'processed'/f'snr-{date}.parquet') for date in DATES],ignore_index=True)
    rd=pd.concat([pd.read_parquet(CACHE/'fresh'/'processed'/f'ranking-{date}.parquet') for date in DATES],ignore_index=True)
    assert set(d.date_key)==set(rd.date_key)==set(DATES)
    fitdays=set(pd.read_parquet(DATA/'fit_features.parquet',columns=['date_key']).date_key)
    olddays=set(pd.read_parquet(DATA/'features.parquet',columns=['date_key']).date_key)
    assert not set(DATES)&(fitdays|olddays)
    panel=RankPanel(rd);reference=joblib.load(reference_path);tuned=joblib.load(tuned_path)
    output=d[['source','mode','date_key','hour','tx_sign','rx_sign','tx_grid','rx_grid','snr2500','power_dbm']].copy()
    result={};ranks={}
    ranking_checks=0
    with threadpool_limits(limits=4):
        for name,bundle in [('reference',reference),('tuned',tuned)]:
            metrics,p,half=metric(d,bundle)
            rp=bundle['model'].predict(rd,station_correction=False)
            r=panel.score(rp).reset_index(names='case_index')
            # Independent implementation groups source rows and uses scipy-backed
            # Spearman rather than the vectorized RankPanel calculation.
            from .train import ranking
            check=pd.DataFrame(ranking(rd,rp,'population_core'))
            check=check[check.unit.eq('station_pair')]
            assert len(check)==len(r)
            np.testing.assert_allclose(np.sort(check.spearman),np.sort(r.spearman),atol=1e-12)
            np.testing.assert_allclose(np.sort(check.best_observed_hour_shortfall_db),
                np.sort(r.best_hour_shortfall_db),atol=1e-12)
            ranking_checks+=len(check)
            result[name]=metrics|{'ranking_cases':len(r),'finite_ranking_cases':int(r.spearman.notna().sum()),
                'ranking_median_spearman':float(r.spearman.median()),
                'mean_best_hour_shortfall_db':float(r.best_hour_shortfall_db.mean())}
            output[name+'_prediction']=p;output[name+'_half_width']=half;ranks[name]=r
        interval,paired=bootstrap_rank(ranks['reference'],ranks['tuned'])
        regression=regression500(reference,tuned)
    identities=panel.h.groupby('case',sort=True)[['source','mode','tx_sign','rx_sign','tx_grid','rx_grid']].first()
    identities=identities.reindex(panel.actual.index).reset_index(drop=True).reset_index(names='case_index')
    paired=paired.merge(identities,on='case_index',validate='one_to_one')
    output.to_parquet(OUT/'fresh_predictions.parquet',index=False)
    paired.to_csv(OUT/'fresh_ranking_pairs.csv',index=False)
    for name,r in ranks.items():r.to_csv(OUT/f'fresh_ranking_{name}.csv',index=False)
    # Independent arithmetic: saved observations, not the metric helper.
    for name in ['reference','tuned']:
        err=[abs(p-a) for p,a in zip(output[name+'_prediction'],output.snr2500)]
        assert np.isclose(sum(err)/len(err),result[name]['mae_db'])
        coverage=sum(e<=h for e,h in zip(err,output[name+'_half_width']))/len(err)
        assert np.isclose(coverage,result[name]['interval_coverage'])
    promoted=bool(interval['95_percent_date_bootstrap_interval'][0]>0 and
        result['tuned']['mae_db']<=result['reference']['mae_db']*1.02)
    run.update({'phase':'evaluated','evaluated_utc':utc(),'fresh_test':result,'rank_improvement':interval,
        'regression500':regression,'promoted':promoted,
        'decision':'Tuned model passes prespecified fresh WSPR promotion gates.' if promoted else
            'Tuned model does not pass every fresh WSPR promotion gate; retain the reference.',
        'fresh_input_sha256':{p.name:digest(p) for p in sorted((CACHE/'fresh'/'processed').glob('*.parquet'))}})
    run['implementation_sha256']={name:digest(ROOT/'ml40'/name) for name in
        ['tune.py','tune_collect.py','tune_evaluate.py','models.py','inference.py']}
    save_json(OUT/'run.json',run)
    metadata={'model':'tuned_station_adjusted_boosted','band_m':40,'params':run['chosen']['params'],
        'sha256':digest(tuned_path),'bytes':tuned_path.stat().st_size,'recommended_over_reference':promoted,
        'frozen_before_test_utc':run['frozen_utc'],'fresh_evaluation_utc':run['evaluated_utc'],
        'limitations':['positive receptions only','100 W/dipole calibration unverified',
            'fresh final evidence is WSPR only','FT8/FT4/CW comparison reuses published test cases']}
    save_json(ROOT/'artifacts/gocap-40m-tuned.json',metadata)
    verification={'status':'verified','validation_only_selection':True,'fresh_dates_disjoint_from_original_dates':True,
        'frozen_utc_before_evaluation':run['frozen_utc']<run['evaluated_utc'],
        'positive_reception_rows':len(d),'fresh_days':len(DATES),'ranking_cases':len(paired),
        'ranking_scores_independently_recomputed':ranking_checks,
        'checks':['original and tuned artifact checksums','fitting file checksum',
            'all 12 planned fresh dates present','test dates excluded from every original split',
            '500 reference-case errors reconciled to frozen suite',
            'fresh MAE and interval coverage independently recomputed',
            'vectorized ranking reconciled to separate source-grouping implementation'],
        'output_sha256':{name:digest(OUT/name) for name in ['plan.json','run.json','fresh_predictions.parquet','fresh_ranking_pairs.csv','regression500_cases.csv']}}
    save_json(OUT/'verification.json',verification)
    write_report(run)
    print(json.dumps({'decision':run['decision'],'fresh_test':result,'rank_improvement':interval,'regression500':regression},indent=2))


def write_report(run):
    chosen=run['chosen'];lines=['# Six-parameter tuning — 40m','',
        run['decision'],'',
        'The bounded search screened **36 randomized configurations plus the current configuration** '
        f'on **{run["screen_rows"]:,} training rows**, then retrained three finalists on '
        f'**{run["training_rows"]:,} rows**. All six parameters varied. '
        'Selection used the existing 2024 validation dates; calibration remained separate. '
        'The final WSPR test uses twelve previously unused dates, one in each month of 2025. '
        'No absent reports become negative examples.','',
        '## Validation results','',
        '| Configuration | Median hourly ranking correlation | Balanced SNR MAE | Meets validation guards |',
        '|---|---:|---:|---|',
        f'| Current exported model | {run["reference_validation"]["rank_median"]:.3f} | {run["reference_validation"]["balanced_mae_db"]:.2f} dB | Reference |']
    for x in run['finalists']:
        lines.append(f'| Trial {x["trial"]} | {x["score"]["rank_median"]:.3f} | {x["score"]["balanced_mae_db"]:.2f} dB | {x["eligible"]} |')
    if chosen:
        from .tune import REFERENCE
        lines+=['','## Selected parameters','', '| Parameter | Previous | Tuned |','|---|---:|---:|']
        for k,v in chosen['params'].items():lines.append(f'| `{k}` | {REFERENCE[k]} | {v} |')
        lines+=['','## Untouched WSPR final comparison','',
            '| Model | Reception aggregates | Dates | SNR MAE | Median hourly correlation | Loss from best observed hour | 80% interval coverage |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for name,s in run['fresh_test'].items():
            lines.append(f'| {name} | {s["rows"]:,} | {s["days"]} | {s["mae_db"]:.2f} dB | {s["ranking_median_spearman"]:.3f} | {s["mean_best_hour_shortfall_db"]:.2f} dB | {s["interval_coverage"]:.1%} |')
        r=run['rank_improvement'];lo,hi=r['95_percent_date_bootstrap_interval']
        lines+=['',f'The paired ranking comparison has **{r["paired_cases"]:,} fixed-path/day cases**. '
            f'The median-correlation difference is **{r["difference"]:+.3f}**, with a 95% paired date-bootstrap '
            f'interval of **[{lo:+.3f}, {hi:+.3f}]**, resampling all cases within each of twelve dates '
            'together (5,000 replicates). Twelve dates still limit confidence; this is not a guarantee '
            'for other dates, solar conditions or modes.','',
            'There are 900 eligible observed-hour curves. One challenger prediction curve is constant, '
            'so its rank correlation is undefined; the paired bootstrap uses the remaining 899 curves. '
            'The best-hour loss metric retains all 900 curves and selects the earliest observed hour '
            'when predictions tie.','',
            '![Validation search and fresh WSPR comparison](results/ml40/tuning/comparison.png)','',
            '## Published 500-case regression comparison','',
            'These previously examined cases are regression evidence, not new final validation. '
            'Each case has equal weight within its mode.','',
            '| Mode | Cases | Previous MAE | Tuned MAE |','|---|---:|---:|---:|']
        for mode,s in run['regression500'].items():
            lines.append(f'| {mode} | {s["cases"]} | {s["reference_mae_db"]:.2f} dB | {s["tuned_mae_db"]:.2f} dB |')
        lines+=['','## Strict validation stress checks','',
            'For each check, both models are refitted after excluding the designated callsigns or '
            'geographic fields from training. Scores are on validation rows containing excluded endpoints. '
            'All station-offset residuals and preprocessing are derived from the retained training rows.','',
            '| Exclusion | Ranking cases | Previous-parameter MAE | Tuned MAE | Previous ranking | Tuned ranking |',
            '|---|---:|---:|---:|---:|---:|']
        for name,s in run['strict_validation'].items():
            a,b=s['reference'],s['tuned']
            lines.append(f'| {name} | {a["rank_cases"]} | {a["balanced_mae_db"]:.2f} dB | {b["balanced_mae_db"]:.2f} dB | {a["rank_median"]:.3f} | {b["rank_median"]:.3f} |')
        lines+=['','The geographic ranking check also regressed, but has only 19 cases. '
            'It is a warning about transfer to excluded fields rather than a precise estimate '
            'of worldwide geographic performance.','']
        lines+=['','## Use the challenger','',
            'The tuned artifact is separate so that the original 500-case suite remains reproducible. '
            'Check the recommendation in its metadata before choosing it as the application default.','',
            '```sh',
            '.venv/bin/python -m ml40.inference artifacts/gocap-40m-tuned.joblib JN76 FN31 2026-11-17T18:00Z --sfi 150 --kp 3 --ap 10',
            '```','']
    lines+=['','## Evidence and reproduction','',
        '```sh','.venv/bin/python -m ml40.tune_collect','.venv/bin/python -m ml40.tune fit',
        '.venv/bin/python -m ml40.tune evaluate','```','',
        '- [Frozen search, parameter ranges and acceptance criteria](results/tuning/plan.json)',
        '- [Every screen trial](results/tuning/screen_results.json)',
        '- [Full run, validation and final results](results/tuning/run.json)',
        '- [Independent output verification](results/tuning/verification.json)',
        '- [Paired fresh path/day ranking results](results/tuning/fresh_ranking_pairs.csv)',
        '- [All 500 regression-case comparisons](results/tuning/regression500_cases.csv)',
        '- [Original experiment](ML40_RESULTS.md)','',
        'Paths above are under `results/ml40/tuning/` in this repository. '
        'Model predictions remain conditional on reported receptions, with unknown antennas and noise. '
        'Absolute 100 W/dipole and contact-probability calibration remain unproven.','']
    text='\n'.join(lines).replace('(results/tuning/','(results/ml40/tuning/')
    (ROOT/'TUNING_RESULTS.md').write_text(text)
    if chosen:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        screen=json.loads((OUT/'screen_results.json').read_text())
        fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
        axes[0].scatter([x['score']['balanced_mae_db'] for x in screen],
            [x['score']['rank_median'] for x in screen],s=25,color='#667085',label='Screen (sample training)')
        axes[0].scatter([x['score']['balanced_mae_db'] for x in run['finalists']],
            [x['score']['rank_median'] for x in run['finalists']],s=65,marker='x',color='#00796B',label='Full-data finalists')
        axes[0].set(xlabel='Validation balanced SNR MAE (dB)',ylabel='Median hourly rank correlation',title='Six-parameter search')
        axes[0].legend(fontsize=7,frameon=False)
        fresh=run['fresh_test'];colors=['#667085','#00796B']
        for ax,key,title,label in [(axes[1],'mae_db','Untouched dates: SNR error','MAE (dB; lower is better)'),
                                  (axes[2],'ranking_median_spearman','Untouched dates: hour ranking','Correlation (higher is better)')]:
            values=[fresh[k][key] for k in ['reference','tuned']]
            bars=ax.bar(['Reference','Tuned'],values,color=colors)
            ax.bar_label(bars,fmt='%.3f',padding=3)
            ax.set(title=title,ylabel=label,ylim=(0,max(values)*1.2))
        for ax in axes:ax.spines[['top','right']].set_visible(False)
        fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)

if __name__=='__main__':evaluate()
