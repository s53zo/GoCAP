"""Export the validation-selected population model without station lookup tables."""
import copy
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from .plan import ROOT,DATA,RESULTS
from .inference import scenario_frame

def export():
    run=json.loads((RESULTS/'training_run.json').read_text())
    assert run['phase']=='evaluated'
    assert json.loads((RESULTS/'verification.json').read_text())['status']=='verified'
    settings=run['scenarios']['standard'];name=settings['chosen_by_validation']
    source=DATA/settings['files'][name]['path']
    original=joblib.load(source)
    bundle=copy.copy(original);model=copy.copy(original['model']);model.effects=None
    bundle['model']=model
    bundle['calibration']={'population_core':original['calibration']['population_core']}
    bundle['deployment_policy']='Generic square-to-square population reference; nuisance station tables used only in training'
    output=ROOT/'artifacts';output.mkdir(exist_ok=True)
    artifact=output/'gocap-40m.joblib';joblib.dump(bundle,artifact)
    deployed=joblib.load(artifact)
    for mode in ['WSPR','FT8','FT4','CW']:
        d=scenario_frame('JN76','FN31','2024-06-01T12:00Z',150,3,10,mode)
        np.testing.assert_array_equal(original['model'].predict(d,station_correction=False),
            deployed['model'].predict(d,station_correction=False))
    meta={'model':name,'selected_by':settings['selection_criterion'],'band_m':40,
        'bytes':artifact.stat().st_size,'sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),
        'fit_features_sha256':run['fit_features_sha256'],'frozen_before_test_utc':run['frozen_utc'],
        'station_lookup_tables_in_artifact':False,'reception_history_required_for_inference':False,
        'target':bundle['target'],'limitations':['positive reported-reception population only',
            'absolute 100W/dipole calibration unverified','solar history supplied as a scenario assumption',
            'nominal intervals do not guarantee temporal or out-of-domain coverage']}
    (output/'gocap-40m.json').write_text(json.dumps(meta,indent=2))
    print('Exported',artifact.name,meta['bytes'],'bytes; generic predictions unchanged')

if __name__=='__main__':export()
