"""Frozen, positive-only 100-case GoCAP versus native VOACAP comparison."""
import datetime as dt,hashlib,json,os
from pathlib import Path
from .plan import ROOT,DATA,RESULTS

COUNT=int(os.environ.get('GOCAP_VOACAP_CASES','100'))
if COUNT not in (100,1000):raise ValueError('Supported frozen suite sizes are100/1000')
SEED=f'gocap-voacap{COUNT}-2026-v1'
BASE=DATA/f'voacap{COUNT}';OUT=RESULTS/f'voacap{COUNT}';BASE.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
ENGINE_ROOT=ROOT/'data/voacap-study'
ENGINE=ENGINE_ROOT/'install/bin/voacapl'
ENGINE_DATA=ENGINE_ROOT/'install/share/voacapl/itshfbc'
DATES=[f'2025-{m:02d}-06' for m in range(1,13)]
KEYS=['date_key','tx_sign','rx_sign','tx_grid','rx_grid','power_dbm']


def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def protocol():
    scenarios=[{'id':f'{ant}_n{noise}_s{scale}','antenna':ant,'noise_parameter':noise,'ssn_scale':scale}
        for ant in ['dipole10m','isotropic'] for noise in [145,150,155] for scale in [1.,.6]]
    value={'version':1,'case_count':COUNT,'seed':SEED,'band_m':40,'frequency_mhz':7.04,
        'source':'WSPRNET positive receptions only','dates':DATES,'pair_day_hash_divisors':[128,256,512],
        'source_cap':50001,'minimum_observed_hours':6,'power_dbm_range':[20,50],
        'distance_km_range':[100,15000],'distance_quotas':{'100to1000':COUNT//4,'1000to3000':COUNT//4,'3000to6000':COUNT//4,'6000to15000':COUNT//4},
        'selection':'Identifier hash within UTC date/distance strata, round-robin dates; fixed stations, four-character squares and reported power; unique directed square pair across suite; no SNR/error selection.',
        'primary_scenario':'dipole10m_n150_s1.0','scenarios':scenarios,
        'antenna_assumptions':'Both ends: ideal IONCAP type23 half-wave horizontal dipoles at 10m over dielectric4/conductivity0.001 S/m, main beam pointed along shortest great-circle path. Isotropic0dBi sensitivity. Actual antennas unknown.',
        'voacap':{'method':30,'coefficients':'CCIR','path':'short','min_takeoff_degrees':3,'required_reliability_percent':50,
            'required_snr_db_hz':-31+10*__import__('math').log10(2500),'sporadic_E_factor':0,
            'multipath_tolerance_db':3,'delay_tolerance_ms':.1,
            'time_alignment':'Native TIME card uses whole UT hours. For an observed [H,H+1) aggregate, average the H and H+1 SNR dB values to approximate H+0.5. Preserve left/right edge sensitivities. UT24 wraps to0.',
            'solar':'WDC-SILSO final 13-month smoothed monthly total SSN Version2. Primary scale1; legacy-convention scale0.6 sensitivity, not a fitted conversion. VOACAP has no daily Kp/Ap input.',
            'target':'monthly hourly median SNR50 in dB-Hz; convert to2500Hz by subtracting 10log10(2500); never compare REL percentages with SNR.'},
        'gocap':'Frozen hourly artifact, WSPR profile at reported power, actual completed historical daily solar context; also score steady-history scalar-query diagnostic.',
        'scoring':'Each case contributes its mean absolute error across actual reported hours. Average cases equally; also report paired-hour MAE/RMSE/bias and date-cluster bootstrap. Missing actual hours remain unknown. Missing VOACAP outputs remain missing and numeric coverage is reported.',
        'limits':'Unknown station antennas/noise and positive detection selection prevent a controlled hardware-matched physics validation. Daily reported receptions differ from VOACAP monthly climatological population. Hindcast uses final solar series, not archived forecast vintages.',
        'model_sha256':sha(ROOT/'artifacts/gocap-40m-hourly.joblib'),'voacap_binary_sha256':sha(ENGINE)}
    if COUNT==1000:
        value.update({'pair_day_hash_divisors':[16,32,64], 'source_cap':250001,
            'sampling_extension':'Original hash128 short paths plus hash16 long-path stratum using source distance>=2700km (buffer before square-centre 3000km split); fallback32/64 only if capped. Same twelve2025day06 dates. Exclude all original100 directed square pairs before case selection.',
            'excluded_prior_cases_sha256':sha(RESULTS/'voacap100/cases_frozen.csv')})
    p=OUT/'protocol.json'
    if p.exists() and json.loads(p.read_text())!=value:raise ValueError('Frozen VOACAP protocol differs')
    p.write_text(json.dumps(value,indent=2));return value
if __name__=='__main__':protocol()
