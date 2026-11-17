"""Auditable native VOACAP input decks and fixed-column Method30 output parsing."""
import math,re,shutil,subprocess,tempfile
from pathlib import Path
import numpy as np
from .voacap_protocol import ENGINE,ENGINE_DATA

BANDWIDTH_DB=10*math.log10(2500)
ANTENNA_TEXT='''HD/half-wave/10m: GoCAP comparison assumption, IONCAP type23
8 parameters
0.00 maximum additional gain dBi
23 antenna type
4 ground dielectric constant
0.001 ground conductivity S/m
7.04 operating frequency MHz (updated internally)
-0.50 half-wave total antenna length, negative means wavelengths
10.0 height in meters (positive physical length)
0.0 additional gain above dipole
'''


def bearing(a,b,c,d):
    a,b,c,d=map(math.radians,[a,b,c,d]);dl=d-b
    return math.degrees(math.atan2(math.sin(dl)*math.cos(c),math.cos(a)*math.sin(c)-math.sin(a)*math.cos(c)*math.cos(dl)))%360


def make_deck(txlat,txlon,rxlat,rxlon,year,month,ssn,power_watts,antenna='dipole10m',noise=150):
    if not 0< power_watts<=1000:raise ValueError('Transmit power must be positive watts')
    if not 0<=ssn<=299:raise ValueError('Invalid SSN')
    filename='default/gc10m.voa' if antenna=='dipole10m' else 'default/isotrope'
    az1=round(bearing(txlat,txlon,rxlat,rxlon),1)%360;az2=round(bearing(rxlat,rxlon,txlat,txlon),1)%360
    ns=lambda x:'N' if x>=0 else 'S';ew=lambda x:'E' if x>=0 else 'W'
    circuit=f'CIRCUIT   {abs(txlat):5.2f}{ns(txlat)}{abs(txlon):9.2f}{ew(txlon)}{abs(rxlat):9.2f}{ns(rxlat)}{abs(rxlon):9.2f}{ew(rxlon)}  S     0'
    lines=['COMMENT    Frozen GoCAP versus VOACAP 40m comparison','LINEMAX   9999','COEFFS    CCIR',
        f'TIME      {1:5d}{24:5d}{1:5d}{1:5d}',f'MONTH     {year:5d}{month:5.2f}',f'SUNSPOT   {ssn:5.1f}',
        'LABEL     '+f'{"TX square centre":<20}{"RX square centre":<20}',circuit,
        f'SYSTEM    {1.:5.0f}{noise:5.0f}{3.:5.2f}{50.:5.0f}{(-31+BANDWIDTH_DB):5.2f}{3.:5.2f}{.1:5.2f}',
        'FPROB      1.00 1.00 1.00 0.00',
        f'ANTENNA   {1:5d}{1:5d}{2:5d}{30:5d}{0.:10.3f}[{filename:<21}]{az1:5.1f}{power_watts/1000:10.7f}',
        f'ANTENNA   {2:5d}{2:5d}{2:5d}{30:5d}{0.:10.3f}[{filename:<21}]{az2:5.1f}{0.:10.7f}',
        'FREQUENCY '+f'{7.04:5.2f}'+f'{0.:5.2f}'*10,'METHOD       30    0','EXECUTE','QUIT']
    return '\n'.join(lines)+'\n'


def parse_output(text):
    """First numeric column is MUF, second is the requested frequency. Never swap."""
    result={};current=None
    for line in text.splitlines():
        label=line[66:].strip() if len(line)>=67 else ''
        if label=='FREQ':
            hour=float(line[1:6]);freq=float(line[11:16])
            if abs(freq-7.04)>.051:raise ValueError('Output frequency column is not the requested 40m channel')
            current=int(round(hour))%24
            if current in result:raise ValueError('Repeated native prediction hour')
            result[current]={'native_utc_hour':hour,'printed_frequency_mhz':freq}
        elif current is not None and label in ['SNR','SNR LW','SNR UP','S DBW','N DBW','MODE','TGAIN','RGAIN']:
            raw=line[11:16].strip()
            result[current][label]=raw if label=='MODE' else None if raw in ['-',''] else float(raw)
    if set(result)!=set(range(24)):raise ValueError(f'Expected 24 native UTC hours, got {len(result)}')
    for row in result.values():
        if row.get('MODE','-') in ['-','']:
            row['snr2500_db']=None;row['status']='no_mode_or_numeric_prediction'
        elif row.get('SNR') is None:
            row['snr2500_db']=None;row['status']='missing_snr'
        else:
            row['snr2500_db']=row['SNR']-BANDWIDTH_DB;row['status']='numeric'
    return result


def at_bin_centre(parsed,hour):
    a=parsed[hour%24];b=parsed[(hour+1)%24]
    if a['snr2500_db'] is None or b['snr2500_db'] is None:return None,None,None
    midpoint=(a['snr2500_db']+b['snr2500_db'])/2
    low=None;high=None
    if all(x.get('SNR LW') is not None and x.get('SNR UP') is not None for x in [a,b]):
        low=((a['SNR']-a['SNR LW'])+(b['SNR']-b['SNR LW']))/2-BANDWIDTH_DB
        high=((a['SNR']+a['SNR UP'])+(b['SNR']+b['SNR UP']))/2-BANDWIDTH_DB
    return midpoint,low,high

class Engine:
    def __enter__(self):
        self.temp=tempfile.TemporaryDirectory(prefix='gcv-',dir='/tmp');self.root=Path(self.temp.name)
        shutil.copytree(ENGINE_DATA,self.root,dirs_exist_ok=True)
        (self.root/'antennas/default/gc10m.voa').write_text(ANTENNA_TEXT)
        return self
    def __exit__(self,*args):self.temp.cleanup()
    def run(self,deck):
        p=self.root/'run/job.dat';p.write_text(deck)
        output=self.root/'run/job.out'
        if output.exists():output.unlink()
        r=subprocess.run([str(ENGINE),str(self.root),'job.dat','job.out'],capture_output=True,text=True,input='',timeout=45)
        if r.returncode or not output.exists() or 'PAUSE' in r.stderr:
            raise RuntimeError('VOACAP execution failed: '+r.stderr[-300:]+r.stdout[-300:])
        text=output.read_text(errors='replace');return text,parse_output(text)
