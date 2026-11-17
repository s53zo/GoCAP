"""Release newly reserved v3 test data after component selection and stress freeze."""
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .plan import ROOT,DATA
from .v3_plan import BASE,OUT,TEST_DATES
from .v3_features import enrich
from .v3_models import profile_predict
from .v3_train import usable,ranking_score,error_summary,utc,save
from .tune import RankPanel
from .tune_evaluate import bootstrap_rank
from .test500 import digest


def interval_score(y,lo,hi):
    return hi-lo+10*np.maximum(lo-y,0)+10*np.maximum(y-hi,0)


def summary(y,p,lo=None,hi=None):
    e=p-y
    out={'rows':len(y),'mae_db':float(abs(e).mean()),'rmse_db':float(np.sqrt(np.mean(e**2))),
        'bias_db':float(e.mean())}
    if lo is not None:
        out.update({'coverage':float(((y>=lo)&(y<=hi)).mean()),'mean_width_db':float((hi-lo).mean()),
            'mean_interval_score':float(interval_score(y,lo,hi).mean())})
    return out


def bootstrap_error(d,reference,candidate,replicates=5000):
    work=pd.DataFrame({'date':d.date_key,'delta':abs(candidate-d.snr2500.to_numpy())-abs(reference-d.snr2500.to_numpy())})
    g=work.groupby('date').delta.agg(['sum','size']);rng=np.random.default_rng(40)
    index=rng.integers(0,len(g),size=(replicates,len(g)))
    sums=g['sum'].to_numpy()[index].sum(axis=1);n=g['size'].to_numpy()[index].sum(axis=1)
    return {'difference_db':float(work.delta.mean()),'95_percent_date_bootstrap_interval':list(map(float,np.quantile(sums/n,[.025,.975]))),
        'date_clusters':len(g),'replicates':replicates,'statistic':'mean absolute error(candidate) minus mean absolute error(reference)'}


def reference_intervals(d,bundle,p):
    half=np.array([bundle['calibration']['population_core'][r.source+'|'+r.mode]['half_width_db'] for r in d.itertuples()])
    return p-half,p+half


def evaluate():
    run=json.loads((OUT/'run.json').read_text());assert run['phase'] in ['frozen_before_fresh_test','evaluated']
    assert (OUT/'stress_training.json').exists(),'Finish strict fitting before releasing test data'
    assert digest(ROOT/'artifacts/gocap-40m-v3.joblib')==run['artifact_sha256']
    assert digest(ROOT/'artifacts/gocap-40m.joblib')==run['reference_sha256']
    assert digest(DATA/'fit_features.parquet')==run['fit_sha256']
    for kind in ['test-snr','test-coherent']:
        assert all((BASE/'processed'/f'{kind}-{date}.meta.json').exists() for date in TEST_DATES)
    snr=pd.concat([pd.read_parquet(BASE/'processed'/f'test-snr-{date}.parquet') for date in TEST_DATES],ignore_index=True)
    coherent=pd.concat([pd.read_parquet(BASE/'processed'/f'test-coherent-{date}.parquet') for date in TEST_DATES],ignore_index=True)
    original_dates=set(pd.read_parquet(DATA/'features.parquet',columns=['date_key']).date_key)
    from .tune_collect import DATES as prior_tuning_dates
    assert not set(TEST_DATES)&(original_dates|set(prior_tuning_dates))
    assert set(snr.date_key)==set(coherent.date_key)==set(TEST_DATES)
    missing_power=int(snr.power_dbm.isna().sum())
    snr=enrich(usable(snr));coherent=enrich(usable(coherent))
    snr.to_parquet(BASE/'fresh_snr_features.parquet',index=False);coherent.to_parquet(BASE/'fresh_rank_features.parquet',index=False)
    deployment=joblib.load(ROOT/'artifacts/gocap-40m-v3.joblib')
    reference=joblib.load(ROOT/'artifacts/gocap-40m.joblib')
    panel=RankPanel(coherent);y=snr.snr2500.to_numpy()
    with threadpool_limits(limits=4):
        p0=reference['model'].predict(snr,station_correction=False)
        p=profile_predict(snr,deployment['models'],deployment['profile_selection'])
        lo,hi=deployment['intervals'].predict(snr,deployment['models'])
        lo0,hi0=reference_intervals(snr,reference,p0)
        reference_rank=ranking_score(panel,reference['model'].predict(coherent,station_correction=False)).reset_index(names='case_index')
        ranker_rank=ranking_score(panel,deployment['ranker'].predict(coherent),True).reset_index(names='case_index')
        point_rank=ranking_score(panel,profile_predict(coherent,deployment['models'],deployment['profile_selection'])).reset_index(names='case_index')
        rank_delta,paired=bootstrap_rank(reference_rank,ranker_rank)
        point_delta=bootstrap_error(snr,p0,p)
        tests={'reference':summary(y,p0,lo0,hi0),'v3':summary(y,p,lo,hi)}
        for name,r in [('reference',reference_rank),('ranker',ranker_rank),('v3_point',point_rank)]:
            tests[name] = tests.get(name,{})|{'ranking_eligible_cases':len(r),'ranking_finite_cases':int(r.spearman.notna().sum()),
                'median_spearman':float(r.spearman.median()),'mean_best_hour_shortfall_db':float(r.best_hour_shortfall_db.mean())}
        # Diagnostic ablations are scored only after the product selection is frozen.
        ablations={}
        for name in ['scattered_wspr_base','coherent_wspr_base','coherent_wspr_rich','pooled_rich','pooled_source_heads']:
            model=joblib.load(BASE/f'{name}.joblib')['model'];rp=model.predict(snr)
            ranks=ranking_score(panel,model.predict(coherent))
            ablations[name]=summary(y,rp)|{'median_spearman':float(ranks.spearman.median()),
                'best_hour_shortfall_db':float(ranks.best_hour_shortfall_db.mean())}
        rank_ablations={}
        for name in ['ranker_base_240_3','ranker_rich_240_3']:
            diagnostic=joblib.load(BASE/f'{name}.joblib')
            rr=ranking_score(panel,diagnostic.predict(coherent),True)
            rank_ablations[name]={'median_spearman':float(rr.spearman.median()),
                'best_hour_shortfall_db':float(rr.best_hour_shortfall_db.mean()),'cases':len(rr)}
        old=pd.read_parquet(DATA/'features.parquet')
        old=old[old.split.eq('test')&~old['mode'].eq('WSPR')].reset_index(drop=True)
        old=enrich(usable(old));oldp=profile_predict(old,deployment['models'],deployment['profile_selection'])
        oldref=reference['model'].predict(old,station_correction=False)
        oldlo,oldhi=deployment['intervals'].predict(old,deployment['models'])
        mode_regression={mode:{'dates':g.date_key.nunique(),
            'reference':summary(g.snr2500.to_numpy(),oldref[g.index]),
            'v3':summary(g.snr2500.to_numpy(),oldp[g.index],oldlo[g.index],oldhi[g.index])}
            for mode,g in old.groupby('mode')}
        stress={}
        for name,column in [('station_holdout','station_held'),('geographic_holdout','geographic_held')]:
            bundle=joblib.load(BASE/f'stress-{name}.joblib')
            st=snr[snr[column]].reset_index(drop=True);sr=coherent[coherent[column]].reset_index(drop=True);sp=RankPanel(sr)
            # Actual retained support must exclude every designated endpoint.
            if name=='station_holdout':
                assert (~st.tx_sign.isin(bundle['stations'])|~st.rx_sign.isin(bundle['stations'])).all()
            else:assert (~st.tx_grid.str[:2].isin(bundle['fields'])|~st.rx_grid.str[:2].isin(bundle['fields'])).all()
            ar=ranking_score(sp,bundle['reference'].predict(sr,station_correction=False))
            br=ranking_score(sp,bundle['ranker'].predict(sr),True)
            stress[name]={'rows':len(st),'eligible_rank_cases':len(ar),
                'reference_snr':summary(st.snr2500.to_numpy(),bundle['reference'].predict(st,station_correction=False)),
                'v3_snr':summary(st.snr2500.to_numpy(),bundle['model'].predict(st,station_correction=False)),
                'reference_rank':float(ar.spearman.median()),'v3_rank':float(br.spearman.median())}
    # No new model or component is selected using the final scores.
    mode_guard=all(x['v3']['mae_db']<=x['reference']['mae_db']*1.05 for x in mode_regression.values())
    recommendations={'point_snr':bool(point_delta['95_percent_date_bootstrap_interval'][1]<0 and mode_guard),
        'hourly_ranking':bool(rank_delta['95_percent_date_bootstrap_interval'][0]>0 and
            tests['ranker']['mean_best_hour_shortfall_db']<=tests['reference']['mean_best_hour_shortfall_db']*1.05),
        'conditional_intervals':bool(.75<=tests['v3']['coverage']<=.85 and
            tests['v3']['mean_interval_score']<tests['reference']['mean_interval_score'])}
    output=snr[['source','mode','date_key','hour','tx_sign','rx_sign','tx_grid','rx_grid','power_dbm','snr2500',
        'distance_km','path_daylight_fraction','kp_previous']].copy()
    for name,value in [('reference_prediction',p0),('v3_prediction',p),('reference_lower',lo0),('reference_upper',hi0),('v3_lower',lo),('v3_upper',hi)]:output[name]=value
    output.to_parquet(OUT/'fresh_predictions.parquet',index=False)
    identities=panel.h.groupby('case')[['source','mode','tx_sign','rx_sign','tx_grid','rx_grid']].first().reindex(panel.actual.index).reset_index(drop=True).reset_index(names='case_index')
    paired.merge(identities,on='case_index',validate='one_to_one').to_csv(OUT/'paired_hour_rankings.csv',index=False)
    for name,r in [('reference',reference_rank),('ranker',ranker_rank),('point',point_rank)]:r.to_csv(OUT/f'ranking_{name}.csv',index=False)
    coverage=[]
    segments={'distance':pd.cut(snr.distance_km,[-1,1000,3000,6000,20001]).astype(str),
        'illumination':np.where(snr.path_daylight_fraction.eq(0),'all_dark',np.where(snr.path_daylight_fraction.eq(1),'all_lit','mixed')),
        'geomagnetic':np.where(snr.kp_previous.ge(4),'Kp>=4','Kp<4')}
    for dimension,values in segments.items():
        for label in sorted(set(values)):
            use=np.asarray(values)==label
            if use.sum()<100:continue
            coverage.append({'dimension':dimension,'segment':label,'rows':int(use.sum()),
                'reference_coverage':float(((y[use]>=lo0[use])&(y[use]<=hi0[use])).mean()),
                'v3_coverage':float(((y[use]>=lo[use])&(y[use]<=hi[use])).mean()),
                'reference_width_db':float((hi0[use]-lo0[use]).mean()),'v3_width_db':float((hi[use]-lo[use]).mean())})
    pd.DataFrame(coverage).to_csv(OUT/'interval_segments.csv',index=False)
    run.update({'phase':'evaluated','evaluated_utc':utc(),'fresh_test':tests,'fresh_snr_days':snr.date_key.nunique(),
        'fresh_excluded_unknown_power_rows':missing_power,'snr_difference':point_delta,'ranking_difference':rank_delta,
        'fresh_ablations':ablations,'fresh_rank_feature_ablations':rank_ablations,
        'published_mode_regressions':mode_regression,'strict_fresh_stress':stress,
        'recommend_components_over_reference':recommendations,
        'fresh_input_sha256':{p.name:digest(p) for p in sorted((BASE/'processed').glob('test-*.parquet'))},
        'evaluation_code_sha256':digest(ROOT/'ml40/v3_evaluate.py')})
    save('run.json',run)
    meta={'version':3,'sha256':run['artifact_sha256'],'bytes':(ROOT/'artifacts/gocap-40m-v3.joblib').stat().st_size,
        'profile_selection':deployment['profile_selection'],'recommend_components_over_reference':recommendations,
        'frozen_utc':run['frozen_utc'],'fresh_test_dates':TEST_DATES,'physical_calibration_verified':False,
        'scope':'Positive-reception SNR, relative observed-hour ranking and conditional intervals; fresh final evidence is WSPR only.'}
    (ROOT/'artifacts/gocap-40m-v3.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({'fresh_test':tests,'recommendations':recommendations,'snr_difference':point_delta,'rank_difference':rank_delta,'strict':stress},indent=2))
if __name__=='__main__':evaluate()
