"""Independent source/geometry/native-output/score verification of shared experiment."""
import json,math,re
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from .shared_plan import ROOT,DATA,RESULTS,BASE,OUT,ARTIFACT,sha
from .voacap_protocol import ENGINE

def verify():
    cfg=json.loads((OUT/'plan.json').read_text());freeze=json.loads((OUT/'case_freeze.json').read_text());tr=json.loads((OUT/'training_run.json').read_text());s=json.loads((OUT/'summary.json').read_text())
    assert sha(OUT/'plan.json')==freeze['plan_sha256']==tr['plan_sha256']
    assert sha(OUT/'case_freeze.json')==s['case_freeze_sha256']
    assert pd.Timestamp(freeze['frozen_utc'])<pd.Timestamp(s['evaluated_utc'])
    assert pd.Timestamp(tr['frozen_utc'])<pd.Timestamp(s['evaluated_utc'])
    assert sha(ARTIFACT)==tr['model_sha256']==s['candidate_model_sha256']
    assert sha(ROOT/'artifacts/gocap-40m-hourly.joblib')==cfg['baseline_sha256']==s['baseline_model_sha256']
    assert sha(ENGINE)==cfg['engine_sha256'] and sha(DATA/'fit_features.parquet')==cfg['fit_sha256']
    for name,digest in tr['code_sha256'].items():assert sha(ROOT/'ml40'/name)==digest
    for name,digest in tr['input_sha256'].items():assert sha(DATA/'hourly'/name)==digest
    for name,digest in freeze['source_sha256'].items():assert sha(BASE/'processed'/name)==digest
    for name,digest in freeze['raw_source_sha256'].items():assert sha(BASE/'raw'/name)==digest
    for n,digest in cfg['excluded_case_sha256'].items():assert sha(RESULTS/f'voacap{n}/cases_frozen.csv')==digest
    cases=pd.read_csv(OUT/'cases_frozen.csv');h=pd.read_csv(OUT/'hourly_comparison.csv');obs=pd.read_parquet(BASE/'case_features.parquet')
    assert len(cases)==cases.case_id.nunique()==500 and len(set(zip(cases.tx_grid,cases.rx_grid)))==500
    assert set(cases.case_id)==set(freeze['case_ids']) and len(h)==12000 and not h.duplicated(['case_id','utc_hour']).any()
    assert cases.groupby('distance_group').size().eq(125).all() and cases.observed_hours.ge(6).all()
    assert cases.power_dbm.between(20,50).all() and cases.distance_km.between(100,15000).all()
    prior=pd.concat([pd.read_csv(RESULTS/f'voacap{n}/cases_frozen.csv') for n in [100,1000]])
    assert not set(zip(cases.tx_grid,cases.rx_grid))&set(zip(prior.tx_grid,prior.rx_grid))
    assert set(cases.date_key).issubset(cfg['test_dates'])
    # Rebuild truth directly from raw provider records; no process_wspr/hourly helpers.
    parts=[]
    for date,selected in cases.groupby('date_key'):
        raw=pd.read_json(BASE/'raw'/f'wspr-{date}.jsonl',lines=True);raw=raw[raw.code.eq(1)&raw.frequency.between(7000000,7300000)&raw.snr.between(-60,100)].copy()
        raw=raw.drop_duplicates(['time','tx_sign','rx_sign','frequency']);raw['hour']=pd.to_datetime(raw.time,utc=True).dt.floor('h')
        raw['tx_grid']=raw.tx_loc.str.upper().str[:4];raw['rx_grid']=raw.rx_loc.str.upper().str[:4];raw=raw.rename(columns={'power':'power_dbm'})
        selected=raw.merge(selected[['case_id','tx_sign','rx_sign','tx_grid','rx_grid','power_dbm']],on=['tx_sign','rx_sign','tx_grid','rx_grid','power_dbm'],validate='many_to_one')
        parts.append(selected.groupby(['case_id','hour']).snr.agg(actual='median',reports='size').reset_index())
    truth=pd.concat(parts,ignore_index=True);truth['utc_hour']=truth.hour.dt.hour
    merged=truth.merge(h,on=['case_id','utc_hour'],validate='one_to_one')
    assert len(merged)==len(obs)==freeze['observed_hours']==s['observed_hours']
    assert np.allclose(merged.actual,merged.actual_snr_db) and np.allclose(merged.reports,merged.positive_reports)
    assert int(merged.reports.sum())==s['positive_reports']
    # Frozen training cache membership and temporal separation.
    fit=pd.read_parquet(DATA/'fit_features.parquet',columns=['row_key','split','hour'])
    train=pd.read_parquet(DATA/'hourly/original_train_features.parquet',columns=['row_key','hour','mode'])
    assert set(train.row_key)==set(fit[fit.split.eq('train')].row_key) and train.hour.max()<pd.Timestamp('2025-01-01',tz='UTC')
    assert set(train['mode'])=={'WSPR','FT8','FT4','CW'} and len(train)==tr['training_rows']
    # Check all five geographic locations with an independent forward-geodesic formula.
    from .shared_features import point_features
    sample=obs.groupby('case_id').head(1);points=point_features(sample)
    def centre(g):return ((ord(g[1])-65)*10+int(g[3])+.5-90,(ord(g[0])-65)*20+int(g[2])*2+1-180)
    for i,row in enumerate(sample.itertuples()):
        a,l=map(math.radians,centre(row.tx_grid));b,m=map(math.radians,centre(row.rx_grid));dl=m-l
        arc=math.acos(max(-1,min(1,math.sin(a)*math.sin(b)+math.cos(a)*math.cos(b)*math.cos(dl))))
        az=math.atan2(math.sin(dl)*math.cos(b),math.cos(a)*math.sin(b)-math.sin(a)*math.cos(b)*math.cos(dl))
        for percent in [0,25,50,75,100]:
            distance=arc*percent/100
            lat=math.asin(math.sin(a)*math.cos(distance)+math.cos(a)*math.sin(distance)*math.cos(az))
            lon=l+math.atan2(math.sin(az)*math.sin(distance)*math.cos(a),math.cos(distance)-math.sin(a)*math.sin(lat))
            assert abs(points.iloc[i][f'point{percent}_latitude']-math.degrees(lat))<.00002
            assert abs(points.iloc[i][f'point{percent}_lon_sin']-math.sin(lon))<1e-6
            assert abs(points.iloc[i][f'point{percent}_lon_cos']-math.cos(lon))<1e-6
    # Reparse all6000 raw native tables without the production parser.
    native=pd.read_parquet(BASE/'voacap_all_scenarios.parquet');groups={(c,sc):g.sort_values('utc_hour').voacap_snr_db.to_numpy() for (c,sc),g in native.groupby(['case_id','scenario'])}
    manifest=pd.read_csv(OUT/'run_manifest.csv').set_index(['case_id','scenario']);assert len(manifest)==6000
    for case in cases.itertuples():
        for scenario in cfg['voacap_scenarios']:
            folder=BASE/'runs'/case.case_id/scenario['id'];ip=folder/'input.dat';op=folder/'output.out';record=manifest.loc[(case.case_id,scenario['id'])]
            assert sha(ip)==record.input_sha256 and sha(op)==record.output_sha256
            cards=ip.read_text().splitlines();tx=next(x for x in cards if x.startswith('ANTENNA'));sun=next(x for x in cards if x.startswith('SUNSPOT'));sys=next(x for x in cards if x.startswith('SYSTEM'))
            assert abs(float(tx[-10:])*1000-case.reported_power_watts)<.000051
            assert abs(float(sun[10:])-case.ssn*scenario['ssn_scale'])<=.050001 and float(sys[15:20])==scenario['noise_parameter']
            assert ('default/gc10m.voa' if scenario['antenna']=='dipole10m' else 'default/isotrope') in tx
            circuit=next(x for x in cards if x.startswith('CIRCUIT'));geo=[float(v)*(-1 if q in 'SW' else 1) for v,q in re.findall(r'([0-9.]+)([NSEW])',circuit)]
            assert np.allclose(geo,[*centre(case.tx_grid),*centre(case.rx_grid)],atol=.001)
            values={};current=None
            for line in op.read_text().splitlines():
                label=line[66:].strip()
                if label=='FREQ':current=int(float(line[:6]))%24;assert float(line[11:16])==7.;values[current]={}
                elif current is not None and label in ('SNR','MODE'):values[current][label]=line[11:16].strip()
            snr={k:float(v['SNR'])-10*math.log10(2500) if v.get('MODE') not in ('-',None,'') and v.get('SNR') not in ('-',None,'') else np.nan for k,v in values.items()}
            expected=np.array([(snr[k]+snr[(k+1)%24])/2 for k in range(24)])
            assert np.allclose(expected,groups[(case.case_id,scenario['id'])],equal_nan=True)
    primary=native[native.scenario.eq(cfg['voacap_primary'])][['case_id','utc_hour','voacap_snr_db']]
    joined=h.merge(primary,on=['case_id','utc_hour'],validate='one_to_one',suffixes=('','_native'));assert np.allclose(joined.voacap_snr_db,joined.voacap_snr_db_native,equal_nan=True)
    sensitivity=pd.read_csv(OUT/'sensitivity.csv').set_index('scenario')
    for scenario,group in native.groupby('scenario'):
        measured=h[h.actual_snr_db.notna()][['case_id','utc_hour','actual_snr_db']].merge(group,on=['case_id','utc_hour'],validate='one_to_one').dropna(subset=['voacap_snr_db'])
        error=abs(measured.voacap_snr_db-measured.actual_snr_db)
        assert np.isclose(error.groupby(measured.case_id).mean().mean(),sensitivity.loc[scenario,'case_mae_db'])
        assert len(measured)==sensitivity.loc[scenario,'numeric_hours']
    measured=h[h.actual_snr_db.notna()]
    assert np.isclose(((measured.actual_snr_db>=measured.shared_lower80_db)&(measured.actual_snr_db<=measured.shared_upper80_db)).mean(),s['shared_interval80_coverage'])
    assert np.isclose((measured.shared_upper80_db-measured.shared_lower80_db).mean(),s['shared_interval80_mean_width_db'])
    # Reconstruct metrics from measured rows, independent of evaluation helpers.
    paired=h.dropna(subset=['actual_snr_db','voacap_snr_db']);per=cases[['case_id','date_key']].copy()
    for key,col in [('baseline','baseline_snr_db'),('shared','shared_snr_db'),('voacap','voacap_snr_db'),('pooled_aggregate_ablation','pooled_aggregate_snr_db')]:
        e=paired[col]-paired.actual_snr_db;macro=abs(e).groupby(paired.case_id).mean()
        for name,v in [('case_mae_db',macro.mean()),('hour_mae_db',abs(e).mean()),('hour_rmse_db',np.sqrt((e*e).mean())),('bias_db',e.mean())]:assert np.isclose(v,s[key][name])
        per[key]=per.case_id.map(macro)
    for comparator,key in [('baseline','shared_minus_baseline'),('voacap','shared_minus_voacap'),('pooled_aggregate_ablation','shared_minus_aggregate')]:
        per['difference']=per.shared-per[comparator];assert np.isclose(per.difference.mean(),s[key+'_case_mae_db'])
        g=per.groupby('date_key').difference.agg(['sum','count']);ix=np.random.default_rng(40).integers(0,len(g),(10000,len(g)))
        b=g['sum'].to_numpy()[ix].sum(1)/g['count'].to_numpy()[ix].sum(1)
        assert np.allclose(np.quantile(b,[.025,.975]),s[key+'_95pct_date_bootstrap_db'])
    from .hourly_models import predict_profiles
    candidate=joblib.load(ARTIFACT);baseline=joblib.load(ROOT/'artifacts/gocap-40m-hourly.joblib')
    assert list(candidate['models'])==['shared'] and set(candidate['profile_selection'].values())=={'shared'}
    assert not any('sign' in x for x in candidate['models']['shared'].encoder.numeric)
    with threadpool_limits(limits=4):
        for label,bundle in [('shared',candidate),('baseline',baseline)]:
            p=predict_profiles(obs,bundle['models'],bundle['profile_selection']);z=obs[['case_id','hour']].copy();z['direct']=p
            z['utc_hour']=z.hour.dt.hour
            check=z[['case_id','utc_hour','direct']].merge(h,on=['case_id','utc_hour'],validate='one_to_one')
            assert np.allclose(check.direct,check[label+'_snr_db'])
    # Recompute the supplementary audit's metrics from its saved measured rows.
    assert sha(DATA/'hourly/fresh_features.parquet')==s['mode_audit_input_sha256']
    mode_data=pd.read_parquet(BASE/'known_mode_audit_predictions.parquet')
    mode_table=pd.read_csv(OUT/'known_mode_audit.csv')
    assert set(mode_data['mode'])=={'CW','FT8','FT4'}
    original_modes=pd.read_parquet(DATA/'hourly/fresh_features.parquet')
    original_modes=original_modes[~original_modes['mode'].eq('WSPR')].reset_index(drop=True)
    pd.testing.assert_frame_equal(original_modes[['hour','mode','tx_grid','rx_grid','snr2500']],mode_data[['hour','mode','tx_grid','rx_grid','snr2500']])
    with threadpool_limits(limits=4):
        for label,bundle in [('shared',candidate),('baseline',baseline)]:
            direct=predict_profiles(original_modes,bundle['models'],bundle['profile_selection'])
            assert np.allclose(direct,mode_data[label+'_prediction'])
    for (model,mode),row in mode_table.set_index(['model','mode']).iterrows():
        subset=mode_data[mode_data['mode'].eq(mode)]
        error=subset[model+'_prediction']-subset.snr2500
        cells=pd.DataFrame({'date':subset.date_key,'hour':subset.hour.dt.hour,'error':abs(error)}).groupby(['date','hour']).error.mean()
        assert np.isclose(cells.mean(),row.hour_date_mae_db) and np.isclose(abs(error).mean(),row.mae_db)
        assert np.isclose(np.sqrt((error*error).mean()),row.rmse_db) and np.isclose(error.mean(),row.bias_db)
        assert len(subset)==row.rows
    from .shared_query import query
    from .inference import THRESHOLDS
    with threadpool_limits(limits=4):q=query('JN76','FN31','2026-11-17T00:00Z',150,3,10,hours=24)
    assert len(q['hourly_predictions'])==24
    for row in q['hourly_predictions']:
        for mode,threshold in THRESHOLDS.items():assert np.isclose(row['theoretical_mode_margins_db'][mode],row['estimated_snr_db_2500hz']-threshold)
    assert not q['physical_mode_independence_verified'] and not q['phone_measurement_validation']
    result={'status':'passed','cases':500,'actual_hourly_medians_rebuilt':len(truth),'positive_raw_reports':int(truth.reports.sum()),'missing_actual_hours_preserved':int(h.actual_snr_db.isna().sum()),
        'native_runs_independently_parsed':len(manifest),'path_point_coordinates_independently_verified':len(cases)*5,'one_pooled_model_all_four_training_modes':True,
        'baseline_unchanged':True,'frozen_hashes_match':True,'independent_metrics_and_bootstrap_match':True,'direct_observed_predictions_match_all_hour_frames':True,'all_mode_margins_use_one_reference_snr':True}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':verify()
