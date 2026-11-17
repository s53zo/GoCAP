"""Diagnostic of steady-history query assumptions; no selection or refitting."""
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .v3_plan import BASE,OUT,ROOT
from .inference import KP_AP
from .tune import RankPanel
from .v3_train import ranking_score


def steady(d):
    d=d.copy();av=np.interp(d.kp_previous.to_numpy(),np.arange(28)/3,KP_AP)
    for col in ['ap_previous','ap_trailing24','ap_mean6','ap_mean72','ap_max24','ap_max72']:d[col]=av
    d['ap_trend6_vs_previous18']=0.
    d['hours_since_ap48_capped72']=np.where(av>=48,d.hour.dt.hour.to_numpy()%3,72.)
    d['sfi_mean27']=d.sfi_previous_day;d['sfi_difference27']=0.
    return d


def check():
    rd=pd.read_parquet(BASE/'fresh_rank_features.parquet');d=pd.read_parquet(BASE/'fresh_snr_features.parquet')
    original=joblib.load(ROOT/'artifacts/gocap-40m.joblib')
    bundle=joblib.load(ROOT/'artifacts/gocap-40m-recommended.joblib');panel=RankPanel(rd)
    result={}
    with threadpool_limits(limits=4):
        for label,historical in [('actual_completed_history',True),('steady_history_assumption',False)]:
            r=rd if historical else steady(rd);s=d if historical else steady(d)
            ranks=ranking_score(panel,bundle['ranker'].predict(r),True)
            ref=ranking_score(panel,original['model'].predict(r,station_correction=False))
            p=original['model'].predict(s,station_correction=False)
            result[label]={'ranking_cases':len(ranks),'reference_rank_correlation':float(ref.spearman.median()),
                'new_rank_correlation':float(ranks.spearman.median()),
                'reference_best_hour_loss_db':float(ref.best_hour_shortfall_db.mean()),
                'new_best_hour_loss_db':float(ranks.best_hour_shortfall_db.mean()),
                'reference_snr_mae_db':float(abs(s.snr2500.to_numpy()-p).mean())}
    (OUT/'scenario_history_diagnostic.json').write_text(json.dumps({'scope':'Post-freeze diagnostic only; current hourly Kp, previous-day SFI/Ap and reported power are retained; preceding Kp/SFI history either actual or assumed steady. No model selected using this diagnostic.',
        'results':result},indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':check()
