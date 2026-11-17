"""Illustrate 24 hourly strength estimates against positive measured hours."""
import hashlib,json
import joblib,numpy as np,pandas as pd
from .hourly_plan import BASE,OUT,ROOT,digest
from .hourly_models import predict_profiles
from .v3_features import enrich
from .prepare import annotate
from experiment import features


def examples():
    d=pd.read_parquet(BASE/'fresh_features.parquet');d=d[d['mode'].eq('WSPR')]
    keys=['date_key','tx_sign','rx_sign','tx_grid','rx_grid','power_dbm']
    eligible=[]
    for key,g in d.groupby(keys,sort=True):
        if g.hour.nunique()<6:continue
        identity=hashlib.sha256(json.dumps(list(key),default=str).encode()).hexdigest()
        eligible.append((identity,key,g))
    eligible.sort(key=lambda x:x[0]);selected=[];quarters=set();paths=set()
    for record in eligible:
        key=record[1];quarter=pd.Timestamp(key[0]).quarter
        if quarter in quarters:continue
        selected.append(record);quarters.add(quarter);paths.add((key[3],key[4]))
    for record in eligible:
        if len(selected)>=6:break
        if (record[1][3],record[1][4]) in paths:continue
        selected.append(record);paths.add((record[1][3],record[1][4]))
    plan={'selection':'Outcome-blind identifier hash; at least six recorded hours at fixed reported power; cover available quarters then distinct square paths; no strength/error filter',
        'cases':[{'id':s[0],'date':s[1][0],'tx_sign':s[1][1],'rx_sign':s[1][2],'tx_grid':s[1][3],
                  'rx_grid':s[1][4],'power_dbm':s[1][5]} for s in selected]}
    path=OUT/'examples_plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    path.write_text(json.dumps(plan,indent=2))
    package=joblib.load(ROOT/'artifacts/gocap-40m-hourly.joblib');reference=joblib.load(ROOT/'artifacts/gocap-40m.joblib')
    all_rows=[]
    for identity,key,actual in selected:
        date,tx,rx,tg,rg,power=key
        frame=pd.DataFrame({'source':['WSPRNET']*24,'mode':['WSPR']*24,
            'hour':pd.date_range(date,periods=24,freq='h',tz='UTC'),'tx_sign':[tx]*24,'rx_sign':[rx]*24,
            'tx_grid':[tg]*24,'rx_grid':[rg]*24,'power_dbm':[power]*24})
        frame=features(frame,solar_start='2009');frame['decoder_context']='ordinary_code1';frame=enrich(annotate(frame))
        p=predict_profiles(frame,package['models'],package['profile_selection'])
        p0=reference['model'].predict(frame,station_correction=False);lo,hi=package['intervals'].predict(frame,package['models'])
        measured=actual.groupby('hour').snr2500.median()
        rows=frame[['hour','tx_grid','rx_grid','tx_sign','rx_sign','power_dbm']].copy()
        rows['case_id']=identity;rows['reference_prediction_db']=p0;rows['hourly_prediction_db']=p
        rows['lower_db']=lo;rows['upper_db']=hi;rows['observed_snr_db']=rows.hour.map(measured)
        assert len(rows)==24 and rows.observed_snr_db.notna().sum()==actual.hour.nunique()
        assert rows.hourly_prediction_db.notna().all()
        all_rows.append(rows)
    data=pd.concat(all_rows,ignore_index=True);data.to_csv(OUT/'24hour_examples.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,9),layout='constrained')
    for ax,(identity,key,_) in zip(axes.flat,selected):
        g=data[data.case_id.eq(identity)];hours=g.hour.dt.hour
        ax.fill_between(hours,g.lower_db,g.upper_db,color='#00796B',alpha=.12,label='80% received-population interval')
        ax.plot(hours,g.reference_prediction_db,color='#667085',linestyle='--',label='Established prediction')
        ax.plot(hours,g.hourly_prediction_db,color='#00796B',label='Hourly SNR model')
        observed=g.observed_snr_db.notna();ax.scatter(hours[observed],g.loc[observed,'observed_snr_db'],color='#111111',s=20,zorder=3,label='Recorded reception')
        watts=10**((key[5]-30)/10)
        ax.set(title=f'{key[3]} → {key[4]} | {key[0]} | reported {watts:g} W',xlabel='UTC hour',ylabel='SNR / 2500 Hz (dB)',xticks=[0,6,12,18,23])
        ax.spines[['top','right']].set_visible(False)
    axes.flat[0].legend(fontsize=7,frameon=False)
    fig.suptitle('Every hour predicted; only recorded receptions shown as observations',fontsize=13)
    fig.savefig(OUT/'24hour_examples.png',dpi=180);plt.close(fig)
    print('Saved',len(selected),'complete 24-hour examples; unreported actuals remain blank')
if __name__=='__main__':examples()
