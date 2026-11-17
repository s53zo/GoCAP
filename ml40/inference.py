"""Model-only scenario prediction: no reception-history query is performed."""
import argparse
import json
import math
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

KP_AP=np.array([0,2,3,4,5,6,7,9,12,15,18,22,27,32,39,48,56,67,80,94,111,132,154,179,207,236,300,400],dtype=float)
PROFILES={'WSPR':('WSPRNET','ordinary_code1'),
    'FT8':('HARC_PSK','archive_mode_known'),'FT4':('HARC_PSK','archive_mode_known'),
    'CW':('HARC_RBN','archive_mode_known')}
THRESHOLDS={'WSPR':-31,'FT8':-21,'FT4':-17.5,'CW_DX':-10,
    'CW_COMFORTABLE':-3,'PHONE_SSB_DX':4,'PHONE_SSB_COMFORTABLE':11}

def centre(g):
    import re
    if not re.fullmatch('[A-R]{2}[0-9]{2}',g):raise ValueError('Use a four-character Maidenhead square')
    return -90+(ord(g[1])-65)*10+int(g[3])+.5,-180+(ord(g[0])-65)*20+int(g[2])*2+1

def scenario_frame(tx,rx,utc,sfi,kp,ap,mode='WSPR',power_watts=100):
    tx=tx.upper();rx=rx.upper()
    if mode not in PROFILES:raise ValueError('Observation profile must be WSPR, FT8, FT4 or CW')
    if not (sfi>0 and 0<=kp<=9 and 0<=ap<=400 and power_watts>0):raise ValueError('Invalid solar or power scenario')
    t=pd.Timestamp(utc)
    t=t.tz_localize('UTC') if t.tz is None else t.tz_convert('UTC')
    t=t.floor('h');hour=t.hour+.5;doy=t.day_of_year
    ta,tl=centre(tx);ra,rl=centre(rx)
    a,b,la,lb=map(math.radians,[ta,ra,tl,rl])
    dot=np.clip(math.sin(a)*math.sin(b)+math.cos(a)*math.cos(b)*math.cos(lb-la),-1,1)
    va=np.array([math.cos(a)*math.cos(la),math.cos(a)*math.sin(la),math.sin(a)])
    vb=np.array([math.cos(b)*math.cos(lb),math.cos(b)*math.sin(lb),math.sin(b)])
    mid=va+vb
    midlat=math.degrees(math.atan2(mid[2],math.hypot(mid[0],mid[1])))
    midlon=math.degrees(math.atan2(mid[1],mid[0]))
    phase=2*math.pi*(doy-1)/365.2425
    dec=(.006918-.399912*math.cos(phase)+.070257*math.sin(phase)
        -.006758*math.cos(2*phase)+.000907*math.sin(2*phase)
        -.002697*math.cos(3*phase)+.00148*math.sin(3*phase))
    eqtime=229.18*(.000075+.001868*math.cos(phase)-.032077*math.sin(phase)
        -.014615*math.cos(2*phase)-.040849*math.sin(2*phase))
    source,decoder=PROFILES[mode]
    converted_ap=float(np.interp(kp,np.arange(28)/3,KP_AP))
    row={'source':source,'mode':mode,'decoder_context':decoder,'tx_grid':tx,'rx_grid':rx,
        'hour':t,'tx_sign':'REFERENCE','rx_sign':'REFERENCE','tx_key':'UNSEEN_REFERENCE_TX',
        'rx_key':'UNSEEN_REFERENCE_RX','tx_lat':ta,'rx_lat':ra,'mid_lat':midlat,
        'distance_km':6371.0088*math.acos(dot),'season_sin':math.sin(phase),'season_cos':math.cos(phase),
        'utc_sin':math.sin(2*math.pi*hour/24),'utc_cos':math.cos(2*math.pi*hour/24),
        'kp_previous':kp,'ap_previous':converted_ap,'ap_trailing24':converted_ap,
        'sfi_previous_day':sfi,'Ap_previous_day':ap,
        'power_dbm':10*math.log10(power_watts*1000) if mode=='WSPR' else np.nan,
        'power_known':float(mode=='WSPR'),'legacy_decoder':float(decoder!='ordinary_code1')}
    for role,lat,lon in [('tx',ta,tl),('rx',ra,rl),('mid',midlat,midlon)]:
        ha=math.radians(15*(hour+lon/15+eqtime/60-12))
        row[role+'_sun']=math.sin(math.radians(lat))*math.sin(dec)+math.cos(math.radians(lat))*math.cos(dec)*math.cos(ha)
        if role!='mid':
            row[role+'_lon_sin']=math.sin(math.radians(lon));row[role+'_lon_cos']=math.cos(math.radians(lon))
    return pd.DataFrame([row])

def predict(artifact,tx,rx,utc,sfi,kp,ap,mode='WSPR',power_watts=100):
    bundle=joblib.load(artifact)
    d=scenario_frame(tx,rx,utc,sfi,kp,ap,mode,power_watts)
    snr=float(bundle['model'].predict(d,station_correction=False)[0])
    cal=bundle['calibration']['population_core'].get(d.iloc[0].source+'|'+mode)
    half=cal['half_width_db'] if cal else None
    domain=bundle.get('training_domain',{}).get(d.iloc[0].source+'|'+mode,{})
    outside=[key for key,(low,high) in domain.items()
        if pd.notna(d.iloc[0][key]) and not low<=float(d.iloc[0][key])<=high]
    return {'origin':tx.upper(),'destination':rx.upper(),'utc_hour':str(d.hour.iloc[0]),
        'observation_profile':mode,'snr2500_db':snr,
        'historical_received_population_interval':None if half is None else [snr-half,snr+half],
        'nominal_coverage':.8 if half is not None else None,'theoretical_mode_margins_db':{k:snr-v for k,v in THRESHOLDS.items()},
        'outside_training_ranges':outside,'joint_scenario_support':'not established by marginal range checks',
        'power_input_watts':power_watts,'power_used_by_profile':mode=='WSPR',
        'solar_assumption':'Kp steady over the completed 24h; ap derived from Kp; Ap is independent previous-day context',
        'calibration_limit':'station-zero population reference; not calibrated for actual dipoles or local noise; margins are not contact probabilities'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('artifact');p.add_argument('tx');p.add_argument('rx');p.add_argument('utc')
    p.add_argument('--sfi',type=float,required=True);p.add_argument('--kp',type=float,required=True);p.add_argument('--ap',type=float,required=True)
    p.add_argument('--mode',choices=list(PROFILES),default='WSPR');p.add_argument('--power-watts',type=float,default=100)
    a=p.parse_args();print(json.dumps(predict(a.artifact,a.tx,a.rx,a.utc,a.sfi,a.kp,a.ap,a.mode,a.power_watts),indent=2))
