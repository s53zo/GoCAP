"""Fit on train/validation, calibrate separately, then explicitly release test scores."""
import argparse
import hashlib
import json
import math
import datetime as dt
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .plan import DATA,RESULTS,manifest
from .models import train_candidates,sample_weights,make_tree
from .prepare import annotate

ARTIFACTS=DATA/'models'
ARTIFACTS.mkdir(exist_ok=True)

def quantile_finite(values,coverage=.8):
    a=np.asarray(values,dtype=float)
    if not len(a):raise ValueError('No calibration observations')
    k=min(len(a),math.ceil((len(a)+1)*coverage))
    return float(np.partition(a,k-1)[k-1])

def scenario_rows(d,scenario,part):
    out=d[d.split.eq(part)].copy()
    if scenario=='station_holdout' and part!='test':out=out[~out.station_held]
    if scenario=='geographic_holdout' and part!='test':out=out[~out.geographic_held]
    return out

def validation_summary(d,models,ranking_validation=None):
    w=sample_weights(d);y=d.snr2500.to_numpy();rows=[]
    for name,model in models.items():
        for view,correction in [('population_core',False),('station_aware',True)]:
            p=model.predict(d,station_correction=correction)
            row={'model':name,'view':view,'balanced_mae_db':float(np.average(abs(y-p),weights=w))}
            if ranking_validation is not None and len(ranking_validation):
                rp=model.predict(ranking_validation,station_correction=correction)
                cases=pd.DataFrame(ranking(ranking_validation,rp,view))
                if len(cases):
                    cases=cases[cases.unit.eq('station_pair')]
                    row['ranking_cases']=len(cases)
                    row['ranking_median_spearman']=float(cases.spearman.median())
            rows.append(row)
    return rows

def calibrate(d,model):
    result={}
    for view,correction in [('population_core',False),('station_aware',True)]:
        p=model.predict(d,station_correction=correction)
        work=d[['source','mode']].copy()
        work['error']=abs(d.snr2500.to_numpy()-p)
        result[view]={}
        for (source,mode),g in work.groupby(['source','mode']):
            if len(g)<100:continue
            result[view][source+'|'+mode]={'half_width_db':quantile_finite(g.error),
                'calibration_rows':len(g),'coverage_target':.8,
                'method':'per-source/mode absolute-residual split calibration; temporal exchangeability not guaranteed'}
    return result

def fit():
    manifest()
    audit=json.loads((RESULTS/'data_audit.json').read_text())
    if audit['status'] not in ['complete','ready_for_fitting']:raise ValueError('Complete training/validation/calibration collection before fitting')
    path=DATA/'fit_features.parquet'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=audit['fit_features_sha256']:raise ValueError('Prepared fitting data hash changed')
    d=pd.read_parquet(path)
    run={'phase':'fitting','fit_features_sha256':digest,'scenarios':{},'positive_only':True}
    (RESULTS/'training_run.json').write_text(json.dumps(run,indent=2))
    for scenario in ['standard','station_holdout','geographic_holdout']:
        tr=scenario_rows(d,scenario,'train');val=scenario_rows(d,scenario,'validation')
        cal=scenario_rows(d,scenario,'calibration')
        files=[p for p in (DATA/'ranking'/'processed').glob('*.parquet') if '2024-' in p.name and p.stem.endswith('-08')]
        rankval=annotate(pd.concat([pd.read_parquet(p) for p in files],ignore_index=True)) if files else None
        if rankval is not None:
            if scenario=='station_holdout':rankval=rankval[~rankval.station_held]
            if scenario=='geographic_holdout':rankval=rankval[~rankval.geographic_held]
        print('Fitting',scenario,'train',len(tr),'validation',len(val),'calibration',len(cal),flush=True)
        if len(tr)<1000 or len(val)<100 or len(cal)<100:raise ValueError('Insufficient split support')
        support_hash=hashlib.sha256(('\n'.join(tr.row_key)+'|validation|'+ '\n'.join(val.row_key)).encode()).hexdigest()
        code_hash=hashlib.sha256((__import__('pathlib').Path(__file__).parent/'models.py').read_bytes()).hexdigest()
        cache=ARTIFACTS/f'{scenario}-candidates.joblib'
        with threadpool_limits(limits=4):
            cached=joblib.load(cache) if cache.exists() else None
            if cached and cached['support_hash']==support_hash and cached['code_hash']==code_hash:
                models,details=cached['models'],cached['details']
                print('Reused checked candidate checkpoint',scenario,flush=True)
            else:
                models,details=train_candidates(tr,val)
                joblib.dump({'support_hash':support_hash,'code_hash':code_hash,'models':models,'details':details},cache)
            validation=validation_summary(val,models,rankval)
            calibrated={name:calibrate(cal,model) for name,model in models.items()}
        files={}
        for name,model in models.items():
            domain={}
            for (source,mode),group in tr.groupby(['source','mode']):
                domain[source+'|'+mode]={c:[float(group[c].min()),float(group[c].max())]
                    for c in ['sfi_previous_day','kp_previous','Ap_previous_day','distance_km','power_dbm']
                    if group[c].notna().any()}
            bundle={'model':model,'calibration':calibrated[name],
                'training_domain':domain,
                'target':'SNR2500 among reported receptions; station-zero reference is not a calibrated dipole',
                'scenario':scenario,'source_features_sha256':digest}
            output=ARTIFACTS/f'{scenario}-{name}.joblib'
            joblib.dump(bundle,output)
            files[name]={'path':str(output.relative_to(DATA)),
                'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'bytes':output.stat().st_size}
        # Select before looking at any final test scores; validation MAE is the
        # checkpoint criterion, ranking remains an independent product metric.
        candidates=[r for r in validation if r['view']=='population_core']
        if all(r.get('ranking_cases',0)>=20 and np.isfinite(r.get('ranking_median_spearman',np.nan)) for r in candidates):
            chosen=max(candidates,key=lambda r:(r['ranking_median_spearman'],-r['balanced_mae_db']))['model']
            criterion='validation station-pair hour ranking; balanced MAE tie-break'
        else:
            chosen=min(candidates,key=lambda r:r['balanced_mae_db'])['model']
            criterion='validation balanced MAE; fewer than 20 usable ranking cases'
        training_keys={'stations':sorted(set(tr.tx_sign)|set(tr.rx_sign)),
            'fields':sorted(set(tr.tx_grid.str[:2])|set(tr.rx_grid.str[:2]))}
        (ARTIFACTS/f'{scenario}-training-support.json').write_text(json.dumps(training_keys))
        run['scenarios'][scenario]={'train_rows':len(tr),'train_days':tr.date_key.nunique(),
            'validation_rows':len(val),'calibration_rows':len(cal),'files':files,
            'validation':validation,'chosen_by_validation':chosen,'details':details}
        run['scenarios'][scenario]['selection_criterion']=criterion
        (RESULTS/'training_run.json').write_text(json.dumps(run,indent=2,default=int))
    run['phase']='frozen_before_test'
    run['frozen_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
    (RESULTS/'training_run.json').write_text(json.dumps(run,indent=2,default=int))
    print('Models frozen. Test scores have not been generated.',flush=True)

def ranking(d,prediction,view,trace=False):
    work=d[['source','mode','date_key','hour','tx_sign','rx_sign','tx_grid','rx_grid','snr2500','power_dbm']].copy()
    work['prediction']=np.asarray(prediction,dtype=np.float64)
    known=work.power_dbm.notna()
    work.loc[known,'snr2500']+=30-work.loc[known,'power_dbm']
    work.loc[known,'prediction']+=30-work.loc[known,'power_dbm']
    rows=[]
    for unit,endpoint_keys in [('station_pair',['tx_sign','rx_sign','tx_grid','rx_grid']),('square_pair',['tx_grid','rx_grid'])]:
        keys=['source','mode','date_key']+endpoint_keys
        # Most scattered hash samples have only one observed hour per pair/day.
        # This exact eligibility prefilter avoids hundreds of thousands of tiny
        # aggregations without changing any retained ranking case or score.
        enough=work.groupby(keys,sort=False).hour.transform('nunique').ge(6)
        eligible=work[enough]
        for group,g in eligible.groupby(keys,sort=False):
            a=g.groupby('hour')[['snr2500','prediction']].median()
            if len(a)<6 or a.snr2500.nunique()<3:continue
            corr=a.snr2500.corr(a.prediction,method='spearman') if a.prediction.nunique()>1 else np.nan
            shortfall=float(a.snr2500.max()-a.loc[a.prediction.idxmax(),'snr2500'])
            row={'source':group[0],'mode':group[1],'date':group[2],'unit':unit,
                'endpoint1':group[3],'endpoint2':group[4],'observed_hours':len(a),
                'spearman':corr,'best_observed_hour_shortfall_db':shortfall,'view':view}
            if unit=='station_pair':row.update({'tx_grid':group[5],'rx_grid':group[6]})
            if trace:
                row['actual_snr_by_hour']=json.dumps({str(k):float(v) for k,v in a.snr2500.items()})
                row['predicted_snr_by_hour']=json.dumps({str(k):float(v) for k,v in a.prediction.items()})
            rows.append(row)
    return rows

def evaluate():
    run=json.loads((RESULTS/'training_run.json').read_text())
    if run['phase'] not in ['frozen_before_test','evaluated']:raise ValueError('Freeze all models before evaluation')
    if hashlib.sha256((DATA/'fit_features.parquet').read_bytes()).hexdigest()!=run['fit_features_sha256']:raise ValueError('Fitting data changed after freeze')
    if json.loads((RESULTS/'data_audit.json').read_text())['status']!='complete':raise ValueError('Complete final test collection before evaluating')
    d=pd.read_parquet(DATA/'features.parquet')
    test=d[d.split.eq('test')].copy()
    cohort_files=sorted((DATA/'ranking'/'processed').glob('cohort-2025-*.parquet'))
    if len(cohort_files)!=36:raise ValueError('Complete the 36-day reserved ranking cohort before test evaluation')
    cohort=annotate(pd.concat([pd.read_parquet(p) for p in cohort_files],ignore_index=True))
    metrics=[];prediction_parts=[];ranks=[]
    for scenario,settings in run['scenarios'].items():
        support=json.loads((ARTIFACTS/f'{scenario}-training-support.json').read_text())
        if scenario=='station_holdout':
            unseen_tx=~test.tx_sign.isin(support['stations']);unseen_rx=~test.rx_sign.isin(support['stations'])
            selections={'unseen_endpoint':test[unseen_tx|unseen_rx],
                'both_endpoints_unseen':test[unseen_tx&unseen_rx]}
        elif scenario=='geographic_holdout':
            selections={'held_geographic_field':test[test.geographic_held]}
        else:selections={'later_dates':test}
        for name,file in settings['files'].items():
            path=DATA/file['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==file['sha256']
            bundle=joblib.load(path);model=bundle['model']
            for selection,part in selections.items():
                for view,correction in [('population_core',False),('station_aware',True)]:
                    with threadpool_limits(limits=4):p=model.predict(part,station_correction=correction)
                    output=part[['row_key','source','mode','date_key','hour','tx_sign','rx_sign','snr2500']].copy()
                    output['prediction']=p;output['scenario']=scenario;output['selection']=selection
                    output['model']=name;output['view']=view
                    lookup={k:v['half_width_db'] for k,v in bundle['calibration'][view].items()}
                    widths=(part.source+'|'+part['mode']).map(lookup).to_numpy(dtype=float)
                    output['lower']=p-widths;output['upper']=p+widths
                    prediction_parts.append(output)
                    for (source,mode),idx in part.groupby(['source','mode']).groups.items():
                        loc=part.index.get_indexer(idx);g=part.loc[idx];err=p[loc]-g.snr2500.to_numpy()
                        w=sample_weights(g);half=widths[loc]
                        calibrated=np.isfinite(half)
                        metrics.append({'scenario':scenario,'selection':selection,'model':name,'view':view,
                            'source':source,'mode':mode,'rows':len(g),'days':g.date_key.nunique(),
                            'mae_db':float(abs(err).mean()),'rmse_db':float(np.sqrt(np.mean(err**2))),
                            'balanced_mae_db':float(np.average(abs(err),weights=w)),
                            'interval_coverage':float(np.mean(abs(err)[calibrated]<=half[calibrated])) if calibrated.any() else np.nan,
                            'calibrated_rows':int(calibrated.sum()),
                            'mean_interval_width_db':float(np.mean(2*half[calibrated])) if calibrated.any() else np.nan})
                    if scenario=='standard':
                        local=ranking(part,p,view)
                        for row in local:row.update({'model':name,'cohort':'main_hash_sample','scenario':scenario,'selection':selection})
                        ranks.extend(local)
            # Coherent pair/day samples are independent of the scattered main sample.
            # They supply the primary hour-ranking metric without adding training rows.
            if scenario=='standard':cohort_selections={'later_dates':cohort}
            elif scenario=='station_holdout':
                u1=~cohort.tx_sign.isin(support['stations']);u2=~cohort.rx_sign.isin(support['stations'])
                cohort_selections={'unseen_endpoint':cohort[u1|u2],
                    'both_endpoints_unseen':cohort[u1&u2]}
            else:cohort_selections={'held_geographic_field':cohort[cohort.geographic_held]}
            for cohort_selection,cd in cohort_selections.items():
                for view,correction in [('population_core',False),('station_aware',True)]:
                    with threadpool_limits(limits=4):cp=model.predict(cd,station_correction=correction)
                    local=ranking(cd,cp,view,trace=True)
                    for row in local:row.update({'model':name,'cohort':'pair_day_cohort','scenario':scenario,'selection':cohort_selection})
                    ranks.extend(local)
        print('Evaluated',scenario,flush=True)
    pd.DataFrame(metrics).to_csv(RESULTS/'test_metrics.csv',index=False)
    pd.concat(prediction_parts,ignore_index=True).to_parquet(DATA/'test_predictions.parquet',index=False)
    pd.DataFrame(ranks).to_csv(RESULTS/'test_hour_ranking.csv',index=False)
    run['phase']='evaluated';run['evaluated_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
    (RESULTS/'training_run.json').write_text(json.dumps(run,indent=2,default=int))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['fit','evaluate'])
    args=parser.parse_args()
    fit() if args.action=='fit' else evaluate()
