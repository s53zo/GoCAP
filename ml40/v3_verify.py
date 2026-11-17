"""Independent source, calibration, error and tied-rank reconciliation for v3."""
import json,math
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .plan import ROOT,DATA
from .v3_plan import BASE,OUT,TRAIN_DATES,TEST_DATES
from .v3_features import RICH_NUMERIC,HISTORY
from .v3_models import PAIR,profile_predict
from .v3_train import save
from .test500 import digest
from .verify import manual_spearman


def verify():
    r=json.loads((OUT/'run.json').read_text());assert r['phase']=='evaluated'
    assert r['frozen_utc']<r['evaluated_utc']
    strict=json.loads((OUT/'stress_training.json').read_text());assert strict['frozen_utc']<r['evaluated_utc']
    assert digest(OUT/'plan.json')==r['definition_sha256']
    assert digest(DATA/'fit_features.parquet')==r['fit_sha256']
    assert digest(ROOT/'artifacts/gocap-40m.joblib')==r['reference_sha256']
    assert digest(ROOT/'artifacts/gocap-40m-v3.joblib')==r['artifact_sha256']
    for name,h in r['code_sha256'].items():assert digest(ROOT/'ml40'/name)==h
    assert digest(ROOT/'ml40/v3_evaluate.py')==r['evaluation_code_sha256']
    source_files=0
    for meta in (BASE/'raw').glob('*.meta.json'):
        info=json.loads(meta.read_text());path=meta.with_name(meta.name.replace('.meta.json','.jsonl'))
        assert digest(path)==info['sha256']
        assert info['rows']<50001 and 'LIMIT 50001' in info['query']
        assert 'band=7' in info['query'] and 'cityHash64(' in info['query']
        assert not any(word in info['query'].lower() for word in ['snr>','snr <','order by snr'])
        source_files+=1
    assert source_files==120
    train=pd.read_parquet(BASE/'coherent_train_features.parquet')
    other=pd.read_parquet(BASE/'other_train_features.parquet')
    validation=pd.read_parquet(BASE/'validation_features.parquet')
    calibration=pd.read_parquet(BASE/'calibration_features.parquet')
    assert train.split.eq('train').all() and other.split.eq('train').all()
    assert not (set(train.date_key)|set(other.date_key))&(set(validation.date_key)|set(calibration.date_key)|set(TEST_DATES))
    assert not set(validation.date_key)&set(calibration.date_key)
    assert set(train.date_key)<=set(TRAIN_DATES)
    for name,h in r['training_features_sha256'].items():assert digest(BASE/name)==h
    rt=pd.read_parquet(BASE/'rank_training.parquet')
    assert not rt.duplicated(PAIR+['hour']).any()
    assert rt.groupby('qid').hour.nunique().ge(4).all()
    assert rt['mode'].eq('WSPR').all() and rt.power_dbm.notna().all()
    assert rt.groupby('qid').relevance.min().eq(0).all()
    actual=train.assign(normalized=train.snr2500-(train.power_dbm-30)).groupby(PAIR+['hour']).normalized.median()
    keys=pd.MultiIndex.from_frame(rt[PAIR+['hour']])
    np.testing.assert_allclose(actual.reindex(keys).to_numpy(),rt.label_snr.to_numpy())
    wanted=rt.groupby('qid').label_snr.rank(method='dense').astype(int)-1
    np.testing.assert_array_equal(wanted.to_numpy(),rt.relevance.to_numpy())
    deployment=joblib.load(ROOT/'artifacts/gocap-40m-v3.joblib')
    assert deployment['ranker'].model.get_params()['objective']=='rank:pairwise'
    forbidden={'snr2500','label_snr','normalized_snr','relevance','qid','tx_sign','rx_sign','reports'}
    assert not set(deployment['ranker'].encoder.numeric)&forbidden
    for name,info in strict['artifacts'].items():
        path=BASE/f'stress-{name}.joblib';assert digest(path)==info['sha256']
        b=joblib.load(path)
        from .plan import station_held,geographic_held
        if name=='station_holdout':assert not any(station_held(s) for s in b['stations'])
        else:assert not any(geographic_held(f+'00') for f in b['fields'])
    snr=pd.read_parquet(BASE/'fresh_snr_features.parquet')
    rankdata=pd.read_parquet(BASE/'fresh_rank_features.parquet')
    saved=pd.read_parquet(OUT/'fresh_predictions.parquet')
    assert set(snr.date_key)==set(rankdata.date_key)==set(TEST_DATES)
    np.testing.assert_array_equal(snr.snr2500,saved.snr2500)
    assert len(saved)==r['fresh_test']['v3']['rows']
    with threadpool_limits(limits=4):
        p=profile_predict(snr,deployment['models'],deployment['profile_selection'])
        lo,hi=deployment['intervals'].predict(snr,deployment['models'])
        np.testing.assert_allclose(saved.v3_prediction,p,atol=1e-10)
        np.testing.assert_allclose(saved.v3_lower,lo,atol=1e-10);np.testing.assert_allclose(saved.v3_upper,hi,atol=1e-10)
        rawlo,rawhi=deployment['intervals'].raw(calibration,deployment['models'])
        errors=np.maximum(rawlo-calibration.snr2500.to_numpy(),calibration.snr2500.to_numpy()-rawhi)
        groups=(calibration.source+'|'+calibration['mode']).to_numpy();calibration_checks=0
        for key,info in deployment['intervals'].corrections.items():
            e=np.sort(errors[groups==key]);index=min(len(e),math.ceil((len(e)+1)*.8))-1
            assert np.isclose(max(0,e[index]),info['adjustment_db'])
            assert len(e)==info['rows'];calibration_checks+=1
        rank_scores=deployment['ranker'].predict(rankdata)
        reference=joblib.load(ROOT/'artifacts/gocap-40m.joblib')
        ref_scores=reference['model'].predict(rankdata,station_correction=False)
    for label,pcol,lcol,ucol in [('reference','reference_prediction','reference_lower','reference_upper'),
                                  ('v3','v3_prediction','v3_lower','v3_upper')]:
        y=saved.snr2500.to_numpy();pred=saved[pcol].to_numpy();low=saved[lcol].to_numpy();high=saved[ucol].to_numpy()
        error=[float(b)-float(a) for a,b in zip(y,pred)]
        assert np.isclose(sum(map(abs,error))/len(error),r['fresh_test'][label]['mae_db'])
        coverage=sum(a<=b<=c for a,b,c in zip(low,y,high))/len(y)
        assert np.isclose(coverage,r['fresh_test'][label]['coverage'])
        interval=[c-a+10*max(a-b,0)+10*max(b-c,0) for a,b,c in zip(low,y,high)]
        assert np.isclose(sum(interval)/len(interval),r['fresh_test'][label]['mean_interval_score'])
    work=rankdata[PAIR+['hour','power_dbm','snr2500']].copy()
    work['actual']=work.snr2500-(work.power_dbm-30)
    work['reference']=ref_scores-(work.power_dbm-30);work['ranker']=rank_scores
    paired=pd.read_csv(OUT/'paired_hour_rankings.csv')
    lookup={tuple(getattr(row,k) for k in PAIR):row for row in paired.rename(columns={'date':'date_key'}).itertuples()}
    ranking_checks=0;all_ranker=[];all_ref=[];losses=[];ref_losses=[]
    for key,g in work.groupby(PAIR,sort=True):
        h=g.groupby('hour')[['actual','reference','ranker']].median()
        if len(h)<6 or h.actual.nunique()<3:continue
        a=h.actual.to_numpy();b=h.ranker.to_numpy();c=h.reference.to_numpy()
        br=manual_spearman(a,b);cr=manual_spearman(a,c)
        all_ranker.append(br);all_ref.append(cr)
        losses.append(float(max(a)-a[np.argmax(b)]));ref_losses.append(float(max(a)-a[np.argmax(c)]))
        if key in lookup:
            row=lookup[key];assert np.isclose(br,row.tuned);assert np.isclose(cr,row.reference)
            ranking_checks+=1
    assert ranking_checks==r['ranking_difference']['paired_cases']
    assert np.isclose(np.nanmedian(all_ranker),r['fresh_test']['ranker']['median_spearman'])
    assert np.isclose(np.nanmedian(all_ref),r['fresh_test']['reference']['median_spearman'])
    assert np.isclose(np.mean(losses),r['fresh_test']['ranker']['mean_best_hour_shortfall_db'])
    assert np.isclose(np.mean(ref_losses),r['fresh_test']['reference']['mean_best_hour_shortfall_db'])
    from .tune_evaluate import bootstrap_rank
    a=pd.read_csv(OUT/'ranking_reference.csv');b=pd.read_csv(OUT/'ranking_ranker.csv')
    bootstrap,_=bootstrap_rank(a,b)
    np.testing.assert_allclose(bootstrap['95_percent_date_bootstrap_interval'],r['ranking_difference']['95_percent_date_bootstrap_interval'])
    recommendations=r['recommend_components_over_reference']
    assert recommendations['hourly_ranking']==bool(bootstrap['95_percent_date_bootstrap_interval'][0]>0 and np.mean(losses)<=np.mean(ref_losses)*1.05)
    audit=json.loads((OUT/'physical_calibration_audit.json').read_text());assert not audit['physical_100W_dipole_calibration_verified']
    result={'status':'verified','positive_only':True,'raw_queries_checked':source_files,
        'rank_training_hours':len(rt),'rank_training_groups':int(rt.qid.nunique()),
        'calibration_quantiles_recomputed':calibration_checks,'paired_ranking_cases_recomputed':ranking_checks,
        'fresh_snr_rows':len(saved),'fresh_dates':len(TEST_DATES),
        'checks':['frozen source, code and model checksums','untouched test dates and disjoint fitting splits',
            'positive source observations reconciled to every rank-training label','no target or station identifiers in rank features',
            'strict station/field exclusions in both roles','independent conformal-calibration order statistics',
            'point predictions, interval bounds, MAE, coverage and interval scores',
            'source-hour medians, manual tied-rank correlation and best-hour loss',
            'paired date-bootstrap reproducibility','physical calibration remains unverified'],
        'output_sha256':{name:digest(OUT/name) for name in ['plan.json','run.json','fresh_predictions.parquet','paired_hour_rankings.csv','interval_segments.csv']}}
    save('verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':verify()
