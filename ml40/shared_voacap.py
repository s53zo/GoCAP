"""Generate frozen native VOACAP curves without reading reception SNR values."""
import json
import pandas as pd
from .shared_plan import BASE,OUT,plan
from .voacap_protocol import sha
from .voacap_engine import Engine,make_deck,parse_output,at_bin_centre
from .inference import centre

def run():
    cfg=plan();cases=pd.read_csv(OUT/'cases_frozen.csv');assert len(cases)==500
    rows=[];manifest=[]
    with Engine() as engine:
        for i,c in enumerate(cases.itertuples(),1):
            txlat,txlon=centre(c.tx_grid);rxlat,rxlon=centre(c.rx_grid)
            for scenario in cfg['voacap_scenarios']:
                folder=BASE/'runs'/c.case_id/scenario['id'];folder.mkdir(exist_ok=True,parents=True)
                deck=make_deck(txlat,txlon,rxlat,rxlon,c.year,c.month,c.ssn*scenario['ssn_scale'],c.reported_power_watts,scenario['antenna'],scenario['noise_parameter'])
                ip=folder/'input.dat';op=folder/'output.out'
                if ip.exists():assert ip.read_text()==deck
                ip.write_text(deck)
                if op.exists():parsed=parse_output(op.read_text())
                else:
                    text,parsed=engine.run(deck);op.write_text(text)
                manifest.append({'case_id':c.case_id,'scenario':scenario['id'],'input_sha256':sha(ip),'output_sha256':sha(op)})
                for h in range(24):
                    mid,low,high=at_bin_centre(parsed,h)
                    rows.append({'case_id':c.case_id,'utc_hour':h,'scenario':scenario['id'],'voacap_snr_db':mid,
                        'voacap_left_db':parsed[h]['snr2500_db'],'voacap_right_db':parsed[(h+1)%24]['snr2500_db'],
                        'voacap_lower_decile_db':low,'voacap_upper_decile_db':high})
            if i%50==0:print('VOACAP curves generated',i,'/500; truth not scored.',flush=True)
    pd.DataFrame(rows).to_parquet(BASE/'voacap_all_scenarios.parquet',index=False)
    pd.DataFrame(manifest).to_csv(OUT/'run_manifest.csv',index=False)
if __name__=='__main__':run()
