"""Post-freeze diagnostic of explicit scalar-query solar-history assumptions."""
import json,joblib,pandas as pd
from threadpoolctl import threadpool_limits
from .hourly_plan import BASE,OUT,ROOT
from .hourly_models import metric,predict_profiles
from .v3_scenario_check import steady


def check():
    d=pd.read_parquet(BASE/'fresh_features.parquet')
    original=joblib.load(ROOT/'artifacts/gocap-40m.joblib')
    package=joblib.load(ROOT/'artifacts/gocap-40m-hourly.joblib');out={}
    with threadpool_limits(limits=4):
        for name,data in [('actual_completed_history',d),('steady_history_assumption',steady(d))]:
            p0=original['model'].predict(data,station_correction=False)
            p=predict_profiles(data,package['models'],package['profile_selection'])
            out[name]={mode:{'reference':metric(g,p0[g.index]),'recommended':metric(g,p[g.index])} for mode,g in data.groupby('mode')}
    (OUT/'scenario_history_diagnostic.json').write_text(json.dumps({'scope':'Post-freeze diagnostic, no model selection; current-hour Kp, previous-day SFI/Ap and reported power retained. It does not evaluate solar-forecast accuracy.',
        'results':out},indent=2))
    print('Steady-history WSPR hourly error:',out['steady_history_assumption']['WSPR']['reference']['hour_date_mae_db'],out['steady_history_assumption']['WSPR']['recommended']['hour_date_mae_db'])
if __name__=='__main__':check()
