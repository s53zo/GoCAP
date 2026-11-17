"""Independent proof for the additional March2025 benchmark."""
import json,math,re
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
from .march_test_plan import ROOT,DATA,BASE,OUT,ARTIFACTS,DATES,sha
from .voacap_protocol import ENGINE

def verify():
    cfg=json.loads((OUT/'plan.json').read_text());f=json.loads((OUT/'case_freeze.json').read_text());s=json.loads((OUT/'summary.json').read_text())
    assert sha(OUT/'plan.json')==f['plan_sha256'] and sha(OUT/'case_freeze.json')==s['frozen_case_sha256']
    assert sha(ROOT/'ml40/years_collect.py')==json.loads((ROOT/'results/ml40/years15/case_freeze.json').read_text())['processor_sha256']
    assert pd.Timestamp(f['frozen_utc'])<pd.Timestamp(s['evaluated_utc'])
    for n,p in ARTIFACTS.items():assert sha(p)==cfg['models_sha256'][n]
    assert sha(ENGINE)==cfg['engine_sha256'] and sha(DATA/'voacap1000/silso_smoothed_v2.txt')==cfg['solar_sha256']
    assert sha(ROOT/'data/raw/gfz.txt')==cfg['gfz_sha256']
    for name,p in [('march',DATA/'march40/expanded_train_features.parquet'),('original',DATA/'hourly/original_train_features.parquet')]:
        assert sha(p)==cfg['fitting_inputs_sha256'][name]
        d=pd.read_parquet(p,columns=['hour']);assert d.hour.dt.year.le(2024).all()
    c=pd.read_csv(OUT/'cases_frozen.csv');h=pd.read_csv(OUT/'hourly_comparison.csv');obs=pd.read_parquet(BASE/'case_features.parquet')
    assert len(c)==500 and set(c.case_id)==set(f['case_ids']) and c.groupby('distance_group').size().eq(125).all()
    assert not c.duplicated(['tx_grid','rx_grid']).any() and c.observed_hours.ge(6).all()
    assert c.power_dbm.between(20,50).all() and c.distance_km.between(100,15000).all()
    assert set(c.date_key).issubset(DATES) and len(h)==12000 and not h.duplicated(['case_id','utc_hour']).any()
    truth=[];groupkeys=['hour','tx_sign','rx_sign','tx_grid','rx_grid','power_dbm']
    for date in DATES:
        raw=BASE/'raw'/f'wspr-{date}.jsonl';p=BASE/'processed'/f'wspr-{date}.parquet'
        assert sha(raw)==f['raw_sha256'][date]==json.loads(raw.with_suffix('.meta.json').read_text())['sha256']
        assert sha(p)==f['source_sha256'][date]
        if not c.date_key.eq(date).any():continue
        d=pd.read_json(raw,lines=True);d['hour']=pd.to_datetime(d.time,utc=True).dt.floor('h')
        d=d[d.code.eq(1)|(d.code.eq(0)&d.hour.lt(pd.Timestamp('2020-05-01',tz='UTC')))].copy()
        regex=r'[A-R]{2}[0-9]{2}(?:[A-X]{2}(?:[0-9]{2})?)?'
        for role in ['tx','rx']:d[role+'_grid']=d[role+'_loc'].str.upper().where(d[role+'_loc'].str.upper().str.fullmatch(regex,na=False)).str[:4]
        d=d[d.tx_grid.notna()&d.rx_grid.notna()&d.frequency.between(7000000,7300000)&d.snr.between(-60,100)&d.tx_sign.notna()&d.rx_sign.notna()&d.tx_sign.astype(str).str.len().ge(2)&d.rx_sign.astype(str).str.len().ge(2)]
        d=d.drop_duplicates(['time','tx_sign','rx_sign','frequency']).copy();d['power_dbm']=d.power.astype(float).where(d.power.between(0,60))
        d['context_rank']=np.where(d.code.eq(0),0,np.where(d.hour.lt(pd.Timestamp('2023-01-16',tz='UTC')),1,2))
        a=d.groupby(groupkeys,dropna=False,observed=True).agg(actual=('snr','median'),raw_reports=('snr','size'),rank=('context_rank','min')).reset_index()
        a['decoder_context']=a['rank'].map({0:'legacy_code0',1:'pre2023_code1',2:'ordinary_code1'});a['date_key']=date
        selected=c[c.date_key.eq(date)]
        joined=a.merge(selected[['case_id','date_key','tx_sign','rx_sign','tx_grid','rx_grid','power_dbm','decoder_context']],on=['date_key','tx_sign','rx_sign','tx_grid','rx_grid','power_dbm','decoder_context'],validate='many_to_one')
        truth.append(joined[['case_id','hour','actual','raw_reports']])
    t=pd.concat(truth,ignore_index=True);t['utc_hour']=t.hour.dt.hour;z=t.merge(h,on=['case_id','utc_hour'],validate='one_to_one')
    assert len(z)==len(obs)==s['observed_hours']==f['observed_hours']
    assert np.allclose(z.actual,z.actual_snr_db) and np.allclose(z.raw_reports,z.positive_reports) and z.raw_reports.sum()==s['positive_reports']
    native=pd.read_parquet(BASE/'native_predictions.parquet');groups={(cid,sc):g.sort_values('utc_hour').voacap_snr_db.to_numpy() for (cid,sc),g in native.groupby(['case_id','scenario'])}
    manifest=pd.read_csv(OUT/'run_manifest.csv').set_index(['case_id','scenario']);assert len(manifest)==6000
    def centre(g):return ((ord(g[1])-65)*10+int(g[3])+.5-90,(ord(g[0])-65)*20+int(g[2])*2+1-180)
    for case in c.itertuples():
        for sc in cfg['voacap_scenarios']:
            folder=BASE/'runs'/case.case_id/sc['id'];ip=folder/'input.dat';op=folder/'output.out';m=manifest.loc[(case.case_id,sc['id'])]
            assert sha(ip)==m.input_sha256 and sha(op)==m.output_sha256
            deck=ip.read_text().splitlines();tx=next(x for x in deck if x.startswith('ANTENNA'));sun=next(x for x in deck if x.startswith('SUNSPOT'));system=next(x for x in deck if x.startswith('SYSTEM'));month=next(x for x in deck if x.startswith('MONTH'))
            assert int(month[10:15])==case.year and float(month[15:])==3.
            assert abs(float(tx[-10:])*1000-case.reported_power_watts)<.000051 and abs(float(sun[10:])-case.ssn*sc['ssn_scale'])<=.050001
            assert float(system[15:20])==sc['noise_parameter'] and ('default/gc10m.voa' if sc['antenna']=='dipole10m' else 'default/isotrope') in tx
            circuit=next(x for x in deck if x.startswith('CIRCUIT'));coords=[float(v)*(-1 if side in 'SW' else 1) for v,side in re.findall(r'([0-9.]+)([NSEW])',circuit)]
            assert np.allclose(coords,[*centre(case.tx_grid),*centre(case.rx_grid)],atol=.001)
            values={};current=None
            for line in op.read_text().splitlines():
                label=line[66:].strip()
                if label=='FREQ':current=int(float(line[:6]))%24;assert float(line[11:16])==7.;values[current]={}
                elif current is not None and label in ('SNR','MODE'):values[current][label]=line[11:16].strip()
            snr={hour:float(v['SNR'])-10*math.log10(2500) if v.get('SNR') not in ('-',None,'') and v.get('MODE') not in ('-',None,'') else np.nan for hour,v in values.items()}
            expected=[(snr[hour]+snr[(hour+1)%24])/2 for hour in range(24)];assert np.allclose(expected,groups[(case.case_id,sc['id'])],equal_nan=True)
    primary=native[native.scenario.eq(cfg['voacap_primary'])][['case_id','utc_hour','voacap_snr_db']];native_match=h.merge(primary,on=['case_id','utc_hour'],suffixes=('','_raw'),validate='one_to_one')
    assert np.allclose(native_match.voacap_snr_db,native_match.voacap_snr_db_raw,equal_nan=True)
    paired=h.dropna(subset=['actual_snr_db','voacap_snr_db']);per=c[['case_id','date_key']].copy()
    for name in ['march','shared_parent','current','voacap']:
        e=paired[name+'_snr_db']-paired.actual_snr_db;case=abs(e).groupby(paired.case_id).mean();per[name]=per.case_id.map(case)
        for metric,value in [('case_mae_db',case.mean()),('hour_mae_db',abs(e).mean()),('hour_rmse_db',np.sqrt((e*e).mean())),('bias_db',e.mean())]:assert np.isclose(value,s['models'][name][metric])
    daily=pd.read_csv(OUT/'per_date.csv').set_index('date_key')
    for date,g in per.groupby('date_key'):
        assert len(g)==daily.loc[date,'cases']
        for name in ['march','shared_parent','current','voacap']:assert np.isclose(g[name].mean(),daily.loc[date,name+'_mae_db'])
    for other in ['voacap','current','shared_parent']:
        per['delta']=per.march-per[other];g=per.groupby('date_key').delta.agg(['sum','count']);ix=np.random.default_rng(40).integers(0,len(g),(10000,len(g)))
        estimates=g['sum'].to_numpy()[ix].sum(1)/g['count'].to_numpy()[ix].sum(1)
        assert np.isclose(per.delta.mean(),s['paired_differences'][other]['difference_db']) and np.allclose(np.quantile(estimates,[.025,.975]),s['paired_differences'][other]['date_cluster95_db'])
    sensitivity=pd.read_csv(OUT/'sensitivity.csv').set_index('scenario')
    for sc,g in native.groupby('scenario'):
        a=h[h.actual_snr_db.notna()][['case_id','utc_hour','actual_snr_db']].merge(g,on=['case_id','utc_hour'],validate='one_to_one').dropna(subset=['voacap_snr_db'])
        e=abs(a.voacap_snr_db-a.actual_snr_db);assert np.isclose(e.groupby(a.case_id).mean().mean(),sensitivity.loc[sc,'case_mae_db'])
    from .hourly_models import predict_profiles
    with threadpool_limits(limits=4):
        for name,path in ARTIFACTS.items():
            bundle=joblib.load(path);prediction=predict_profiles(obs,bundle['models'],bundle['profile_selection']);check=obs[['case_id','hour']].copy();check['utc_hour']=check.hour.dt.hour;check['prediction']=prediction
            check=check[['case_id','utc_hour','prediction']].merge(h,on=['case_id','utc_hour'],validate='one_to_one');assert np.allclose(check.prediction,check[name+'_snr_db'])
    result={'status':'passed','held_out_year':2025,'held_out_dates':len(DATES),'case_count':500,'raw_hourly_medians_rebuilt':len(t),'raw_positive_reports':int(t.raw_reports.sum()),
        'unknown_actual_case_hours_preserved':int(h.actual_snr_db.isna().sum()),'native_runs_independently_parsed':6000,'all_models_and_sources_match_frozen_hashes':True,
        'no_case_date_in_any_point_fitting_partition':True,'independent_date_case_hour_metrics_and_bootstrap_match':True,'direct_observation_features_match_all_hour_predictions':True}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':verify()
