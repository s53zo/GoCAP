"""Absolute SNR stress models; excluded endpoints never fit a candidate correction."""
import json,time,copy
import joblib,pandas as pd
from threadpoolctl import threadpool_limits
from .hourly_plan import BASE,OUT,DATA,ROOT,digest
from .hourly_train import utc
from .hourly_models import NativeModel,ResidualModel


def fit():
    while not (OUT/'run.json').exists():time.sleep(20)
    run=json.loads((OUT/'run.json').read_text());chosen=run['selection']['WSPR']
    tr=pd.read_parquet(BASE/'original_train_features.parquet');tr=tr[tr['mode'].eq('WSPR')].reset_index(drop=True)
    corrections=pd.read_parquet(BASE/'correction_train_features.parquet')
    for name,column in [('station_holdout','station_held'),('geographic_holdout','geographic_held')]:
        output=BASE/f'stress-{name}.joblib'
        if output.exists():continue
        train=tr[~tr[column]].reset_index(drop=True);cr=corrections[~corrections[column]].reset_index(drop=True)
        reference=joblib.load(DATA/'models'/f'{name}-boosted_station.joblib')['model']
        with threadpool_limits(limits=4):
            if chosen=='reference':model=reference
            elif chosen=='existing_boosted_baseline':model=joblib.load(DATA/'models'/f'{name}-boosted_baseline.joblib')['model']
            elif chosen.startswith('independent_residual'):
                model=ResidualModel(reference,.5 if chosen.endswith('half') else 1.).fit(cr)
            else:
                original=joblib.load(BASE/f'{chosen}.joblib')
                model=NativeModel(original.rich,original.weighting,original.small).fit(train)
        joblib.dump({'model':model,'reference':reference,'excluded_flag':column,
            'training_stations':sorted(set(train.tx_sign)|set(train.rx_sign)|set(cr.tx_sign)|set(cr.rx_sign)),
            'training_fields':sorted(set(train.tx_grid.str[:2])|set(train.rx_grid.str[:2])|set(cr.tx_grid.str[:2])|set(cr.rx_grid.str[:2])),
            'native_training_rows':len(train),'correction_training_rows':len(cr)},output)
        print('Hourly SNR exclusion fit complete',name,chosen,flush=True)
    (OUT/'stress_freeze.json').write_text(json.dumps({'frozen_utc':utc(),'selected_wspr_model':chosen,
        'artifacts':{name:digest(BASE/f'stress-{name}.joblib') for name in ['station_holdout','geographic_holdout']}},indent=2))
if __name__=='__main__':fit()
