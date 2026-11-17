"""Package only verified improvements; preserve the original and experimental models."""
import json
import joblib
import numpy as np
import pandas as pd
from .plan import ROOT
from .v3_plan import OUT
from .v3_reference import ReferenceIntervals
from .v3_features import scenario
from .inference import PROFILES
from .test500 import digest
from .v3_models import profile_predict


def export():
    run=json.loads((OUT/'run.json').read_text());verified=json.loads((OUT/'verification.json').read_text())
    assert verified['status']=='verified' and run['recommend_components_over_reference']['hourly_ranking']
    original=joblib.load(ROOT/'artifacts/gocap-40m.joblib')
    experiment=joblib.load(ROOT/'artifacts/gocap-40m-v3.joblib')
    selection={mode:'reference' for mode in ['WSPR','FT8','FT4','CW']}
    domain={mode:original['training_domain'][source+'|'+mode].copy() for mode,(source,_) in PROFILES.items()}
    # New ranker has additional support limits; expose them alongside the point-model limits.
    domain['WSPR'].update(experiment['training_domain']['WSPR'])
    bundle={'version':3,'models':{'reference':original['model']},'profile_selection':selection,
        'ranker':experiment['ranker'],'intervals':ReferenceIntervals(original['calibration']['population_core'],selection),
        'training_domain':domain,'calibration_limit':experiment['calibration_limit'],
        'interval_method':'Established source/mode absolute-residual 80% intervals for reported receptions; conditional-quantile challenger not promoted.',
        'component_policy':{'snr':'retained reference for all modes','intervals':'retained reference','ranking':'verified new WSPR pairwise ranker'}}
    path=ROOT/'artifacts/gocap-40m-recommended.joblib';joblib.dump(bundle,path)
    deployed=joblib.load(path)
    for mode in selection:
        frame,_=scenario('JN76','FN31','2026-11-17T18:00Z',150,3,10,mode)
        a=original['model'].predict(frame,station_correction=False)
        b=deployed['models']['reference'].predict(frame,station_correction=False)
        np.testing.assert_array_equal(a,b)
        lo,hi=deployed['intervals'].predict(frame,deployed['models'])
        half=original['calibration']['population_core'][frame.source.iloc[0]+'|'+mode]['half_width_db']
        np.testing.assert_allclose(lo,a-half);np.testing.assert_allclose(hi,a+half)
    meta={'sha256':digest(path),'bytes':path.stat().st_size,'recommended_default':True,
        'component_policy':bundle['component_policy'],
        'recommend_components_over_reference':run['recommend_components_over_reference'],
        'evaluation_reference_sha256':run['reference_sha256'],
        'experimental_artifact_sha256':run['artifact_sha256'],
        'physical_calibration_verified':False,'query_module':'ml40.v3_inference',
        'scope':'40m; four-character squares; scenario SNR; WSPR observed-hour ranking; missing reception stays unknown'}
    path.with_suffix('.json').write_text(json.dumps(meta,indent=2))
    verify_deployment()
    print('Recommended package exported:',path.name,path.stat().st_size,'bytes')


def verify_deployment():
    import os,tempfile
    from threadpoolctl import threadpool_limits
    from .v3_plan import BASE
    from .v3_inference import predict
    from .inference import predict as original_query
    verified=json.loads((OUT/'verification.json').read_text())
    for name,h in verified['output_sha256'].items():assert digest(OUT/name)==h
    artifact=ROOT/'artifacts/gocap-40m-recommended.joblib'
    package=joblib.load(artifact);experimental=joblib.load(ROOT/'artifacts/gocap-40m-v3.joblib')
    data=pd.read_parquet(BASE/'fresh_snr_features.parquet')
    saved=pd.read_parquet(OUT/'fresh_predictions.parquet')
    rankdata=pd.read_parquet(BASE/'fresh_rank_features.parquet')
    with threadpool_limits(limits=4):
        point=profile_predict(data,package['models'],package['profile_selection'])
        low,high=package['intervals'].predict(data,package['models'])
        np.testing.assert_array_equal(point,saved.reference_prediction)
        np.testing.assert_array_equal(low,saved.reference_lower)
        np.testing.assert_array_equal(high,saved.reference_upper)
        np.testing.assert_array_equal(package['ranker'].predict(rankdata),experimental['ranker'].predict(rankdata))
    previous=os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            for mode in PROFILES:
                result=predict(artifact,'JN76','FN31','2026-11-17T18:00Z',150,3,10,mode)
                prior=original_query(ROOT/'artifacts/gocap-40m.joblib','JN76','FN31','2026-11-17T18:00Z',150,3,10,mode)
                assert abs(result['predictions'][0]['snr2500_db']-prior['snr2500_db'])<1e-10
                json.dumps(result,allow_nan=False)
            day=predict(artifact,'JN76','FN31','2026-11-17T00:00Z',150,3,10,hours=24)
            assert len(day['predictions'])==24
            assert all(row['relative_hour_rank_score'] is not None for row in day['predictions'])
            json.dumps(day,allow_nan=False)
            assert not list(__import__('pathlib').Path(directory).iterdir())
    finally:os.chdir(previous)
    info={'status':'verified','recommended_artifact_sha256':digest(artifact),
        'fresh_rows_exactly_matching_established_snr_and_intervals':len(data),
        'fresh_rank_inputs_matching_verified_experimental_ranker':len(rankdata),
        'standalone_checks':'Four profiles and 24-hour WSPR query from empty working directory; strict JSON; no reception history loaded or directories created',
        'model_and_data_verification_sha256':digest(OUT/'verification.json'),
        'recommended_component_policy':package['component_policy']}
    (OUT/'deployment_verification.json').write_text(json.dumps(info,indent=2))
if __name__=='__main__':export()
