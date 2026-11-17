"""Independent raw-output and metric reconstruction for the frozen benchmark."""
import json,math
import numpy as np,pandas as pd
from .voacap_protocol import BASE,OUT,ROOT,ENGINE,sha,COUNT

def verify():
    cfg=json.loads((OUT/'protocol.json').read_text());freeze=json.loads((OUT/'case_freeze.json').read_text())
    assert sha(OUT/'protocol.json')==freeze['protocol_sha256']
    assert sha(ROOT/'artifacts/gocap-40m-hourly.joblib')==freeze['model_sha256']
    assert sha(ENGINE)==cfg['voacap_binary_sha256']
    assert sha(BASE/'silso_smoothed_v2.txt')==freeze['solar_sha256']
    for name,digest in freeze['source_sha256'].items():assert sha(BASE/'processed'/name)==digest
    c=pd.read_csv(OUT/'case_results.csv');h=pd.read_csv(OUT/'hourly_comparison.csv');native=pd.read_csv(OUT/'voacap_all_scenarios.csv')
    assert len(set(zip(c.tx_grid,c.rx_grid)))==COUNT
    assert c.observed_hours.ge(6).all() and c.power_dbm.between(20,50).all()
    assert c.distance_km.between(100,15000).all()
    assert c.groupby('distance_group').size().eq(COUNT//4).all()
    if COUNT==1000:
        from .voacap_protocol import RESULTS
        old=pd.read_csv(RESULTS/'voacap100/cases_frozen.csv')
        assert not set(zip(c.tx_grid,c.rx_grid)) & set(zip(old.tx_grid,old.rx_grid))
        assert sha(RESULTS/'voacap100/cases_frozen.csv')==cfg['excluded_prior_cases_sha256']
    assert len(c)==COUNT and set(c.case_id)==set(freeze['case_ids'])
    assert len(h)==COUNT*24 and not h.duplicated(['case_id','utc_hour']).any()
    assert len(native)==COUNT*24*12 and native.scenario.nunique()==12
    obs=pd.read_parquet(BASE/'case_features.parquet');obs['utc_hour']=obs.hour.dt.hour
    joined=h.merge(obs[['case_id','utc_hour','snr2500','reports']],on=['case_id','utc_hour'],how='left',validate='one_to_one')
    assert np.allclose(joined.actual_snr_db,joined.snr2500,equal_nan=True)
    assert np.allclose(joined.positive_reports,joined.reports,equal_nan=True)
    assert h.actual_snr_db.notna().sum()==len(obs)
    # Do not call the production parser: read fixed 5-character output slots directly.
    manifests=[]
    case_rows=c.set_index('case_id')
    native_groups={(cid,sc):g.sort_values('utc_hour').voacap_snr_db.to_numpy() for (cid,sc),g in native.groupby(['case_id','scenario'])}
    for cid in c.case_id:
        for scenario in cfg['scenarios']:
            folder=BASE/'runs'/cid/scenario['id'];text=(folder/'output.out').read_text();values={};current=None
            # Independently inspect executed input cards, including watt/kW conversion.
            cards=(folder/'input.dat').read_text().splitlines();case=case_rows.loc[cid]
            sun=next(line for line in cards if line.startswith('SUNSPOT'))
            assert abs(float(sun[10:])-case.ssn*scenario['ssn_scale'])<=.050001
            system=next(line for line in cards if line.startswith('SYSTEM'))
            assert float(system[15:20])==scenario['noise_parameter']
            txantenna=next(line for line in cards if line.startswith('ANTENNA'))
            assert abs(float(txantenna[-10:])*1000-case.reported_power_watts)<=.000051
            assert ('default/gc10m.voa' if scenario['antenna']=='dipole10m' else 'default/isotrope') in txantenna
            circuit=next(line for line in cards if line.startswith('CIRCUIT'))
            def square_centre(grid):
                return ((ord(grid[1])-65)*10+int(grid[3])+.5-90,
                        (ord(grid[0])-65)*20+int(grid[2])*2+1-180)
            import re
            coordinates=re.findall(r'([0-9.]+)([NSEW])',circuit)
            latlon=[float(v)*(-1 if side in 'SW' else 1) for v,side in coordinates]
            assert np.allclose(latlon,[*square_centre(case.tx_grid),*square_centre(case.rx_grid)],atol=.001)
            for line in text.splitlines():
                label=line[66:].strip()
                if label=='FREQ':
                    current=int(float(line[:6]))%24;assert float(line[11:16])==7.;values[current]={}
                elif current is not None and label in ('MODE','SNR'):values[current][label]=line[11:16].strip()
            points={k:(float(v['SNR'])-10*math.log10(2500) if v.get('MODE') not in ('-','',None) and v.get('SNR') not in ('-','',None) else np.nan) for k,v in values.items()}
            expected=np.array([(points[k]+points[(k+1)%24])/2 for k in range(24)])
            got=native_groups[(cid,scenario['id'])]
            assert np.allclose(got,expected,equal_nan=True,atol=1e-10)
            manifests.append({'case_id':cid,'scenario':scenario['id'],'input_sha256':sha(folder/'input.dat'),'output_sha256':sha(folder/'output.out')})
    pd.DataFrame(manifests).to_csv(OUT/'run_manifest.csv',index=False)
    paired=h.dropna(subset=['actual_snr_db','voacap_snr_db'])
    summary=json.loads((OUT/'summary.json').read_text())
    for label,col in [('gocap','gocap_snr_db'),('voacap','voacap_snr_db')]:
        errors=paired[col]-paired.actual_snr_db
        macro=abs(errors).groupby(paired.case_id).mean().mean()
        assert np.isclose(macro,summary[label]['case_mae_db'])
        assert np.isclose(abs(errors).mean(),summary[label]['hour_mae_db'])
        assert np.isclose(np.sqrt((errors**2).mean()),summary[label]['hour_rmse_db'])
        assert np.isclose(errors.mean(),summary[label]['bias_db'])
    # Direct predictions on original observed feature rows agree with all-hour generation.
    import joblib
    from threadpoolctl import threadpool_limits
    from .hourly_models import predict_profiles
    bundle=joblib.load(ROOT/'artifacts/gocap-40m-hourly.joblib')
    with threadpool_limits(limits=4):p=predict_profiles(obs,bundle['models'],bundle['profile_selection'])
    actual=obs[['case_id','utc_hour']].copy();actual['direct']=p
    check=actual.merge(h,on=['case_id','utc_hour'],validate='one_to_one')
    assert np.allclose(check.direct,check.gocap_snr_db,atol=1e-10)
    result={'status':'passed','frozen_cases':len(c),'native_runs_independently_parsed':len(manifests),'full_hourly_predictions':len(h),'actual_positive_hours':len(obs),
        'missing_reception_hours_preserved_unknown':int(h.actual_snr_db.isna().sum()),'all_input_source_model_hashes_match':True,'independent_metrics_match':True,'observed_features_predict_identically':True,
        'native_deck_power_sunspots_noise_antennas_and_coordinates_verified':len(manifests)}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':verify()
