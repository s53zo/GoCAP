"""Documented positive reference-measurement intake; never invent physical calibration."""
import argparse,json,math
from pathlib import Path
import pandas as pd
from .inference import centre
from .v3_plan import OUT

REQUIRED=['id','utc','tx_sign','rx_sign','tx_grid','rx_grid','mode','measured_feedpoint_power_watts',
    'received_signal_dbm','noise_dbm_2500hz','tx_antenna','rx_antenna',
    'power_measurement_reference','receiver_calibration_reference','noise_measurement_method','observation_reference']

def validate(records):
    clean=[];errors=[];seen=set()
    for index,r in enumerate(records):
        try:
            missing=[k for k in REQUIRED if k not in r]
            if missing:raise ValueError('Missing fields: '+', '.join(missing))
            if r['id'] in seen:raise ValueError('Duplicate observation id')
            if r.get('received') is not True:raise ValueError('Only explicitly received positive measurements are eligible')
            centre(r['tx_grid']);centre(r['rx_grid'])
            t=pd.Timestamp(r['utc'])
            if t.tzinfo is None:raise ValueError('UTC timestamp requires a timezone')
            if r['mode'] not in ['WSPR','FT8','FT4','CW','PHONE_SSB']:raise ValueError('Unsupported mode')
            for k in ['measured_feedpoint_power_watts','received_signal_dbm','noise_dbm_2500hz']:
                if not math.isfinite(float(r[k])):raise ValueError('Non-finite measurement')
            if float(r['measured_feedpoint_power_watts'])<=0:raise ValueError('Feedpoint power must be positive')
            for role in ['tx','rx']:
                a=r[role+'_antenna']
                for k in ['type','height_m','documentation_reference']:
                    if k not in a:raise ValueError(role+' antenna needs '+k)
                if not a['type'] or not a['documentation_reference'] or not math.isfinite(float(a['height_m'])) or float(a['height_m'])<=0:
                    raise ValueError('Invalid documented antenna')
            for k in ['power_measurement_reference','receiver_calibration_reference','noise_measurement_method','observation_reference']:
                if not isinstance(r[k],str) or not r[k].strip():raise ValueError('Empty provenance: '+k)
            snr=float(r['received_signal_dbm'])-float(r['noise_dbm_2500hz'])
            clean.append(r|{'snr2500_db':snr,'utc':t.tz_convert('UTC').isoformat()});seen.add(r['id'])
        except (ValueError,TypeError,KeyError) as exc:errors.append({'record_index':index,'error':str(exc)})
    return clean,errors

def audit(path=None):
    records=json.loads(Path(path).read_text()) if path else []
    if not isinstance(records,list):raise ValueError('Measurement file must contain a list of records')
    clean,errors=validate(records)
    out={'status':'reference_measurements_available_for_further_evaluation' if clean else 'awaiting_actual_documented_measurements',
        'input_records':len(records),'valid_positive_measurements':len(clean),'invalid_records':errors,
        'physical_100W_dipole_calibration_verified':False,
        'reason':'Field validation verifies schema and traceability, not instrument truth, antenna gain, or transfer to other stations.',
        'valid_records':clean}
    (OUT/'physical_calibration_audit.json').write_text(json.dumps(out,indent=2,allow_nan=False))
    (OUT/'reference_measurement_schema.json').write_text(json.dumps({'format':'JSON list of actual received measurements',
        'required_fields':REQUIRED+['received=true'],'antenna_fields':['type','height_m','documentation_reference'],
        'units':{'received_signal_dbm':'calibrated receiver input dBm','noise_dbm_2500hz':'noise power integrated over 2500Hz in dBm',
            'measured_feedpoint_power_watts':'watts at antenna feedpoint'},
        'analysis_policy':'No automatic 100W/dipole correction. Fit and evaluate any proposed correction on separate station/date groups; retain positive-only selection limits.'},indent=2))
    print(json.dumps({k:v for k,v in out.items() if k!='valid_records'},indent=2))
    return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('measurements',nargs='?');a=p.parse_args();audit(a.measurements)
