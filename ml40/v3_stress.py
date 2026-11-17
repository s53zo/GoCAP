"""Refit selected components with designated stations/fields excluded in both roles."""
import json,time,datetime as dt
import joblib
import pandas as pd
from threadpoolctl import threadpool_limits
from .v3_plan import BASE,OUT
from .v3_models import SNRModel,HourRanker,rank_training
from .test500 import digest
from .plan import DATA


def fit():
    while not (OUT/'run.json').exists():time.sleep(20)
    run=json.loads((OUT/'run.json').read_text())
    coherent=pd.read_parquet(BASE/'coherent_train_features.parquet')
    other=pd.read_parquet(BASE/'other_train_features.parquet')
    selected=run['profile_selection']['WSPR'];ranker=run['selected_ranker'];audit={}
    for name,column in [('station_holdout','station_held'),('geographic_holdout','geographic_held')]:
        path=BASE/f'stress-{name}.joblib'
        if path.exists():continue
        ws=coherent[~coherent[column]].reset_index(drop=True)
        extra=other[~other[column]].reset_index(drop=True)
        train=pd.concat([ws,extra],ignore_index=True) if selected.startswith('pooled') else ws
        assert not train[column].any()
        reference=joblib.load(DATA/'models'/f'{name}-boosted_station.joblib')['model']
        with threadpool_limits(limits=4):
            if selected=='reference':point=reference
            else:point=SNRModel(rich=selected.endswith('rich') or selected=='pooled_source_heads',source_heads=selected=='pooled_source_heads').fit(train)
            rt=rank_training(ws)
            r=HourRanker().fit(rt,True,ranker['iterations'],ranker['depth'])
        # Store actual retained support to audit exclusions independently.
        info={'model':point,'ranker':r,'reference':reference,
            'stations':sorted(set(train.tx_sign)|set(train.rx_sign)),
            'fields':sorted(set(train.tx_grid.str[:2])|set(train.rx_grid.str[:2])),
            'train_rows':len(train),'ranking_groups':rt.qid.nunique(),'excluded_flag':column}
        joblib.dump(info,path);audit[name]={'artifact_sha256':digest(path),'train_rows':len(train),'ranking_groups':rt.qid.nunique()}
        print('Strict v3 components fitted',name,len(train),'rows',flush=True)
    (OUT/'stress_training.json').write_text(json.dumps({'phase':'frozen_before_test',
        'frozen_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'artifacts':{
        name:{'sha256':digest(BASE/f'stress-{name}.joblib')} for name in ['station_holdout','geographic_holdout']}},indent=2))
if __name__=='__main__':fit()
