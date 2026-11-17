"""Train the fixed March-expanded shared model, preserving previous artifacts."""
import json,datetime as dt,resource
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from .march_plan import ROOT,DATA,BASE,OUT,ARTIFACT,plan,sha
from .march_models import MarchModel
from .shared_train import evaluate
from .hourly_models import HourlyIntervals,predict_profiles

def train():
    cfg=plan();prep=json.loads((OUT/'preparation.json').read_text());assert sha(BASE/'expanded_train_features.parquet')==prep['train_sha256']
    d=pd.read_parquet(BASE/'expanded_train_features.parquet');assert d.hour.dt.year.le(2024).all()
    print('Fitting fixed shared+five-points model:',len(d),'records;',d.memory_usage(deep=True).sum()/1e6,'MB compact frame',flush=True)
    with threadpool_limits(limits=4):model=MarchModel().fit_bounded(d,BASE/'matrix')
    validation=pd.read_parquet(DATA/'hourly/validation_features.parquet');calibration=pd.read_parquet(DATA/'hourly/calibration_features.parquet')
    parent=joblib.load(ROOT/'artifacts/gocap-40m-shared-points.joblib');baseline=joblib.load(ROOT/'artifacts/gocap-40m-hourly.joblib')
    one={'shared':model};selection={m:'shared' for m in ['WSPR','FT8','FT4','CW']}
    with threadpool_limits(limits=4):
        scores={'march':evaluate(validation,model.predict(validation)),
            'shared_parent':evaluate(validation,predict_profiles(validation,parent['models'],parent['profile_selection'])),
            'current':evaluate(validation,predict_profiles(validation,baseline['models'],baseline['profile_selection']))}
        intervals=HourlyIntervals().fit(calibration,one,selection)
    domain={}
    for (source,mode),g in d.groupby(['source','mode'],observed=True):
        cols=baseline['training_domain'].get(source+'|'+mode,{})
        domain[source+'|'+mode]={c:[float(g[c].min()),float(g[c].max())] for c in cols if c in g and g[c].notna().any()}
    bundle={'version':'shared-march2010to2024-1','models':one,'profile_selection':selection,'intervals':intervals,
        'training_domain':domain,'selection_objective':'Fixed parent specification; evaluate effect of expanded March data, no retuning.',
        'calibration_limit':'Received-signal population; unknown antenna/noise/power context and theoretical PHONE margins remain physically uncalibrated.'}
    joblib.dump(bundle,ARTIFACT)
    meta={'sha256':sha(ARTIFACT),'bytes':ARTIFACT.stat().st_size,'experimental':True,'promoted':False,'training_years':list(range(2010,2025)),
        'training_rows':len(d),'added_unique_march_rows':prep['added_unique_march_rows'],'profile_selection':selection,'physical_calibration_verified':False}
    ARTIFACT.with_suffix('.json').write_text(json.dumps(meta,indent=2))
    run={'phase':'frozen_before_fresh_test','frozen_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'plan_sha256':sha(OUT/'plan.json'),
        'model_sha256':sha(ARTIFACT),'parent_sha256':cfg['parent_sha256'],'baseline_sha256':cfg['baseline_sha256'],
        'training_data_sha256':prep['train_sha256'],'training_rows':len(d),'training_days':int(d.date_key.nunique()),'training_by_mode':prep['training_by_mode'],
        'validation':scores,'calibration_rows':len(calibration),'intervals':intervals.half_width,'peak_process_rss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'code_sha256':{name:sha(ROOT/'ml40'/name) for name in ['march_models.py','march_train.py','shared_models.py','shared_features.py']}}
    (OUT/'training_run.json').write_text(json.dumps(run,indent=2));print('Frozen March-expanded artifact',meta,flush=True)
if __name__=='__main__':train()
