"""Paired receiver, power-step, and matched flare/control measurements."""
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiment import features
from .four_data import OUT,CACHE,DATES
from .four_run import ART,load,held,DEFAULT
from .four_prepare import noise,CASE
from .hourly_models import predict_profiles
from .prepare import annotate
from .v3_features import enrich


def receiver(pack):
    frames=[]
    for date in DATES['test']:
        m=pd.read_parquet(CACHE/f'matched-{date}.parquet')
        n=noise(date)
        # A next-hour issue carries the current observation-hour noise. Diagnostic only.
        n['hour']-=pd.Timedelta(hours=1)
        for side in ['left','right']:
            m=m.merge(n.rename(columns={'rx_sign':side,'noise_now':side+'_same'})[[side,'rx_grid','hour',side+'_same']],
                      on=[side,'rx_grid','hour'],how='left',validate='many_to_one')
        frames.append(m)
    m=pd.concat(frames,ignore_index=True)
    result={}
    for name,g in [('all',m),('same_six',m[m.same6]),('both_concurrent',m.dropna(subset=['left_same','right_same'])),
                   ('both_previous',m.dropna(subset=['left_noise','right_noise']))]:
        row=dict(pairs=len(g),median_gap=float(abs(g.gap).median()),mean_gap=float(abs(g.gap).mean()))
        if name.startswith('both'):
            col='same' if name=='both_concurrent' else 'noise'
            dif=g['left_'+col]-g['right_'+col]
            row.update(correlation=float(g.gap.corr(dif)),expected_sign_correlation='negative',
                       corrected_mean_gap=float(abs(g.gap-pack['live_coef']*dif).mean()))
        result[name]=row
    return result


def power(pack):
    bundle=joblib.load(DEFAULT)
    rows=[]
    for date in DATES['test']:
        s=pd.read_parquet(CACHE/f'allsteps-{date}.parquet')
        if s.empty:
            continue
        s=s.reset_index(drop=True)
        s['seq']=np.arange(len(s))
        other=s.assign(power_dbm=s.power_dbm-s.step)
        d=pd.concat([s.assign(after=True),other.assign(after=False)])
        d=d.assign(source='WSPRNET',mode='WSPR',decoder_context='ordinary_code1')
        x=enrich(annotate(features(d,solar_start='2009')))
        x['pred']=predict_profiles(x,bundle['models'],bundle['profile_selection'])
        levels=x.pivot(index='seq',columns='after',values='pred')
        s['base_change']=(levels[True]-levels[False]).to_numpy()
        s['step_change']=pack['slope']*s.step
        s['rx_count']=s.groupby(['time','tx_sign','step']).rx_sign.transform('nunique')
        rows.append(s)
    s=pd.concat(rows,ignore_index=True)
    output={}
    for name,g in [('all',s),('strong',s[s[['snr','previous_snr']].min(axis=1)>=-20]),
                   ('multireceiver',s[s.rx_count>=2]),('reserved_receiver',s[s.rx_sign.map(held)])]:
        output[name]=dict(pairs=len(g),transmitters=g.tx_sign.nunique(),
                          observed_ratio=float((g.delta/g.step).median()),base_change_mae=float(abs(g.base_change-g.delta).mean()),
                          step_model_change_mae=float(abs(g.step_change-g.delta).mean()))
    return output


def flare():
    d=pd.read_parquet(CACHE/'test_predictions.parquet')
    d['utc']=d.hour.dt.hour
    known=d[['hour','status']].drop_duplicates().set_index('hour').status
    d['recovery']=d.status.ne('flare')&((d.hour-pd.Timedelta(hours=1)).map(known).eq('flare')|
                                         (d.hour-pd.Timedelta(hours=2)).map(known).eq('flare'))
    keys=CASE+['utc']
    q=d[d.status.eq('quiet')&~d.recovery].copy()
    out={}
    for name,use in [('flare',d.status.eq('flare')),('recovery',d.recovery)]:
        e=d[use].copy().reset_index(drop=True).reset_index(names='event_id')
        matched=e.merge(q,on=keys,suffixes=('_e','_q'))
        matched['days']=(matched.hour_e-matched.hour_q).dt.days.abs()
        matched=matched[matched.days.between(1,35)].sort_values(['event_id','days','hour_q']).drop_duplicates('event_id')
        actual=matched.snr2500_e-matched.snr2500_q
        row=dict(pairs=len(matched),event_dates=matched.hour_e.dt.date.nunique(),observed_change_median=float(actual.median()))
        for model in ['default','flare','live_combined']:
            delta=matched[model+'_e']-matched[model+'_q']
            row[model+'_change_mae']=float(abs(delta-actual).mean())
        out[name]=row
    return out


if __name__=='__main__':
    with threadpool_limits(limits=3):
        pack=joblib.load(ART)
        result=dict(receiver_pairs=receiver(pack),power_steps=power(pack),weather_pairs=flare())
        (OUT/'paired_diagnostics.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
