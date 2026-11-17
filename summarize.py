"""Generate inspectable metrics and a standalone research figure from predictions."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collect import DATA
from experiment import OUT

def summarize():
    run=json.loads((OUT/'run.json').read_text())
    if run['status']!='complete':raise ValueError('Model run is not complete; refusing stale outputs')
    for key,path in [('hourly_sha256',DATA/'hourly.parquet'),('predictions_sha256',DATA/'predictions.parquet'),('metrics_sha256',OUT/'metrics.json')]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=run[key]:
            raise ValueError(f'Artifact does not match completed run: {path.name}')
    p=pd.read_parquet(DATA/'predictions.parquet')
    h=pd.read_parquet(DATA/'hourly.parquet',columns=['row_id','tx_grid','rx_grid','power_dbm','distance_km'])
    p=p.merge(h,on='row_id',validate='many_to_one')
    p['date']=p.hour.dt.strftime('%Y-%m-%d')
    # Unweighted pair-hour medians; assess ranking only where at least 6 hours were observed.
    records=[]
    for (source,mode,fold),frame in p.groupby(['source','mode','fold']):
        if fold!='later_period':continue
        for (date,tx,rx),g in frame.groupby(['date','tx_grid','rx_grid']):
            g=g.copy()
            power_adjust=g.power_dbm.notna().all()
            if power_adjust:
                # Compare hours at a common nominal 1W; antennas/noise remain uncalibrated.
                for col in ['snr2500','historical','geometry','geometry_solar']:
                    g[col]=g[col]+30-g.power_dbm
            a=g.groupby('hour')[['snr2500','historical','geometry','geometry_solar']].median()
            if len(a)<6 or a.snr2500.nunique()<3:continue
            for model in ['historical','geometry','geometry_solar']:
                corr=a.snr2500.corr(a[model],method='spearman') if a[model].nunique()>1 else np.nan
                # Actual SNR shortfall of the model's selected strongest observed hour.
                shortfall=a.snr2500.max()-a.loc[a[model].idxmax(),'snr2500']
                records.append({'source':source,'mode':mode,'date':date,'tx_grid':tx,'rx_grid':rx,
                    'observed_hours':len(a),'model':model,'power_normalized':power_adjust,
                    'spearman':corr,'best_hour_shortfall_db':shortfall})
    r=pd.DataFrame(records)
    r.to_csv(OUT/'hour_ranking.csv',index=False)
    if len(r):
        s=r.groupby(['source','mode','model']).agg(path_days=('date','size'),
            median_spearman=('spearman','median'),mean_best_hour_shortfall_db=('best_hour_shortfall_db','mean'))
        s.to_csv(OUT/'hour_ranking_summary.csv')
    rows=[]
    for key,g in p.groupby(['source','mode','fold','date']):
        for model in ['historical','geometry','geometry_solar']:
            rows.append(dict(zip(['source','mode','fold','date'],key))|{'model':model,'rows':len(g),
                'mae_db':float((g.snr2500-g[model]).abs().mean())})
    pd.DataFrame(rows).to_csv(OUT/'per_day_errors.csv',index=False)
    distance=[]
    p['distance_group']=pd.cut(p.distance_km,[-1,500,3000,21000],labels=['0–500 km','500–3000 km','3000+ km'])
    for (source,mode,fold,band),g in p.groupby(['source','mode','fold','distance_group'],observed=True):
        for model in ['historical','geometry','geometry_solar']:
            distance.append({'source':source,'mode':mode,'fold':fold,'distance_group':str(band),
                'model':model,'rows':len(g),'mae_db':float((g.snr2500-g[model]).abs().mean())})
    pd.DataFrame(distance).to_csv(OUT/'distance_errors.csv',index=False)
    m=pd.DataFrame(json.loads((OUT/'metrics.json').read_text()))
    m[m.status=='evaluated'].to_csv(OUT/'metrics.csv',index=False)
    view=m[(m.source=='WSPRNET') & m.model.isin(['historical','geometry','geometry_solar'])]
    piv=view.pivot(index='fold',columns='model',values='mae_db')
    folds=['later_period','later_unseen_endpoint','later_both_endpoints_unseen']
    labels=['Later sampled days','Later + unseen endpoint','Later + both endpoints unseen']
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
    colors=['#667085','#00796B','#B35600']
    x=np.arange(len(folds))
    for i,(model,label) in enumerate([('historical','Historical lookup'),('geometry','Geometry/time + reported power'),('geometry_solar','Geometry/time + power + solar')]):
        values=piv.loc[folds,model]
        bars=ax.bar(x+(i-1)*.25,values,width=.23,label=label,color=colors[i])
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
    ax.set(xticks=x,xticklabels=labels,ylabel='Mean absolute SNR error (dB; lower is better)',ylim=(0,9.6),
        title='40m WSPR pilot: prediction error on reported receptions')
    ax.legend(loc='upper left',fontsize=9,frameon=False)
    ax.spines[['top','right']].set_visible(False)
    fig.text(.01,-.035,'Train: sampled 2023–2024 days. Test: four sampled 2025 days. Positive receptions only; no contact-probability claim.',fontsize=9)
    fig.savefig(OUT/'wspr-errors.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print('Wrote ranking, per-day errors, metrics CSV and WSPR error figure.')

if __name__=='__main__':summarize()
