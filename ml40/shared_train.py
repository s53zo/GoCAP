"""Train pooled ablations; select on2024 validation, never on the new500 cases."""
import json,datetime as dt,gc
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from .shared_plan import ROOT,DATA,BASE,OUT,ARTIFACT,plan,sha
from .shared_models import SharedModel
from .hourly_models import metric,predict_profiles,HourlyIntervals

def evaluate(d,p):return {mode:metric(d[d['mode'].eq(mode)],np.asarray(p)[d['mode'].eq(mode).to_numpy()]) for mode in sorted(d['mode'].unique())}

def train():
    cfg=plan();fit=pd.read_parquet(DATA/'fit_features.parquet',columns=['hour','split','mode'])
    assert fit.hour.max()<pd.Timestamp('2025-01-01',tz='UTC')
    tr=pd.read_parquet(DATA/'hourly/original_train_features.parquet');vd=pd.read_parquet(DATA/'hourly/validation_features.parquet');cd=pd.read_parquet(DATA/'hourly/calibration_features.parquet')
    assert len(tr)==int(fit.split.eq('train').sum()) and tr.split.eq('train').all()
    assert vd.split.eq('validation').all() and cd.split.eq('calibration').all()
    baseline=joblib.load(ROOT/'artifacts/gocap-40m-hourly.joblib');results={};models={}
    with threadpool_limits(limits=4):
        bp=predict_profiles(vd,baseline['models'],baseline['profile_selection']);base=evaluate(vd,bp)
        for spec in cfg['candidates']:
            path=BASE/(spec['id']+'.joblib')
            if path.exists():model=joblib.load(path)
            else:
                print('Training',spec['id'],len(tr),'records across',tr['mode'].value_counts().to_dict(),flush=True)
                model=SharedModel(spec['points'],spec['weighting']).fit(tr);joblib.dump(model,path)
            models[spec['id']]=model;p=model.predict(vd);scores=evaluate(vd,p)
            macro=float(np.mean([r['hour_date_mae_db'] for r in scores.values()]))
            guards={m:scores[m]['hour_date_mae_db']<=base[m]['hour_date_mae_db']*(1.05 if m=='WSPR' else 1.10) for m in scores}
            results[spec['id']]={'by_mode':scores,'equal_mode_hour_date_mae_db':macro,'guards':guards,'all_guards_pass':all(guards.values())}
            (OUT/'validation_progress.json').write_text(json.dumps({'baseline':base,'candidates':results},indent=2))
            print('Validation',spec['id'],'macro',macro,'guards',guards,flush=True);gc.collect()
        point_names=[x['id'] for x in cfg['candidates'] if x['points']]
        eligible=[n for n in point_names if results[n]['all_guards_pass']]
        chosen=min(eligible or point_names,key=lambda n:results[n]['equal_mode_hour_date_mae_db'])
        selection={m:'shared' for m in ['WSPR','FT8','FT4','CW']};one={'shared':models[chosen]}
        intervals=HourlyIntervals().fit(cd,one,selection)
        bundle={'version':'shared-five-points-1','models':one,'profile_selection':selection,'intervals':intervals,
            'training_domain':baseline['training_domain'],'selection_objective':'Equal-mode validation hourly SNR MAE, not ranking',
            'calibration_limit':'Shared observation-context model. Unknown antennas/noise and unknown non-WSPR transmit powers prevent physical calibration. PHONE is threshold-only.'}
        joblib.dump(bundle,ARTIFACT)
    meta={'sha256':sha(ARTIFACT),'bytes':ARTIFACT.stat().st_size,'objective':bundle['selection_objective'],
        'profile_selection':selection,'selected_candidate':chosen,'validation_guards_pass':bool(eligible),'experimental':True,'physical_calibration_verified':False}
    ARTIFACT.with_suffix('.json').write_text(json.dumps(meta,indent=2))
    run={'phase':'frozen_before_new_test','frozen_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'plan_sha256':sha(OUT/'plan.json'),'model_sha256':sha(ARTIFACT),'baseline_sha256':cfg['baseline_sha256'],'selected_candidate':chosen,
        'validation_guards_pass':bool(eligible),'validation':results,'baseline_validation':base,
        'training_rows':len(tr),'training_by_mode':tr['mode'].value_counts().to_dict(),'training_days':int(tr.date_key.nunique()),
        'latest_training_date':str(tr.hour.max()),'calibration_rows':len(cd),'intervals':intervals.half_width,
        'input_sha256':{name:sha(DATA/'hourly'/name) for name in ['original_train_features.parquet','validation_features.parquet','calibration_features.parquet']},
        'code_sha256':{name:sha(ROOT/'ml40'/name) for name in ['shared_models.py','shared_features.py','shared_train.py']}}
    (OUT/'training_run.json').write_text(json.dumps(run,indent=2));print('Frozen shared candidate',chosen,'guards passed:',bool(eligible),'SHA',run['model_sha256'],flush=True)
if __name__=='__main__':train()
