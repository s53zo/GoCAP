"""End-to-end checks of the all-hour query and recommended package."""
import json,os,tempfile,re
import numpy as np,pandas as pd
from .hourly_plan import ROOT,BASE,OUT,digest
from .hourly_query import predict,table


def verify():
    r=json.loads((OUT/'run.json').read_text());v=json.loads((OUT/'verification.json').read_text())
    assert v['status']=='verified'
    for name,sha in v['output_sha256'].items():assert digest(OUT/name)==sha,name
    artifact=ROOT/'artifacts/gocap-40m-hourly.joblib'
    assert digest(artifact)==r['recommended_artifact_sha256']
    previous=os.getcwd();profiles=[]
    try:
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            for mode in ['WSPR','FT8','FT4','CW']:
                result=predict(artifact,'JN76','FN31','2026-11-17T00:00Z',150,3,10,mode,hours=24)
                rows=result['hourly_predictions'];assert len(rows)==24
                assert len({row['utc_hour'] for row in rows})==24
                assert all(np.isfinite(row['estimated_snr_db_2500hz']) for row in rows)
                assert all(row['received_population_interval_80pct_db'][0]<=row['estimated_snr_db_2500hz']<=row['received_population_interval_80pct_db'][1] for row in rows)
                assert all(not any('rank' in key or 'strongest' in key for key in row) for row in rows)
                json.dumps(result,allow_nan=False)
                rendered=table(result);assert rendered.count('+00:00')==24
                if mode!='WSPR':assert not result['power_used_by_profile'] and 'power input unused' in rendered
                profiles.append(mode)
            assert not list(__import__('pathlib').Path(directory).iterdir())
    finally:os.chdir(previous)
    examples=pd.read_csv(OUT/'24hour_examples.csv')
    assert examples.groupby('case_id').size().eq(24).all()
    assert examples.observed_snr_db.isna().any()
    assert examples.hourly_prediction_db.notna().all()
    assert examples.groupby('case_id').observed_snr_db.count().ge(6).all()
    for link in re.findall(r'\]\(([^)]+)\)',(ROOT/'HOURLY_RESULTS.md').read_text()):
        if not link.startswith('https://'):assert (ROOT/link).exists(),link
    out={'status':'verified','artifact_sha256':digest(artifact),'profiles_queried':profiles,
        'hours_per_query':24,'query_from_empty_working_directory':True,
        'snr_and_intervals_finite':True,'no_ranking_or_strongest_hour_output':True,
        'unknown_power_explicit_for_other_modes':True,
        'example_cases':int(examples.case_id.nunique()),'example_predicted_hours':len(examples),
        'unreported_measured_hours_remain_unknown':True,'report_links_verified':True,
        'source_metric_verification_sha256':digest(OUT/'verification.json'),
        'query_code_sha256':digest(ROOT/'ml40/hourly_query.py')}
    (OUT/'deployment_verification.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
if __name__=='__main__':verify()
