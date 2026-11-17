"""Position-specific path context at0/25/50/75/100%; not reflection points."""
import numpy as np,pandas as pd
from .inference import centre
from .v3_features import MAG
PERCENT=[0,25,50,75,100]
KINDS=['latitude','lon_sin','lon_cos','sun_sin_elevation','local_sin','local_cos','magnetic_latitude','daylight','twilight']
POINTS=[f'point{p}_{k}' for p in PERCENT for k in KINDS]

def point_features(d):
    d=d.copy();grids=set(d.tx_grid)|set(d.rx_grid);coords={g:centre(g) for g in grids}
    ta,tl=np.deg2rad(np.array([coords[g] for g in d.tx_grid])).T
    ra,rl=np.deg2rad(np.array([coords[g] for g in d.rx_grid])).T
    a=np.column_stack([np.cos(ta)*np.cos(tl),np.cos(ta)*np.sin(tl),np.sin(ta)])
    b=np.column_stack([np.cos(ra)*np.cos(rl),np.cos(ra)*np.sin(rl),np.sin(ra)])
    omega=np.arccos(np.clip((a*b).sum(1),-1,1));t=np.array(PERCENT)/100
    sin=np.sin(omega);safe=np.where(abs(sin)<1e-8,1,sin)
    pts=(np.sin((1-t[None,:])*omega[:,None])[:,:,None]*a[:,None,:]+np.sin(t[None,:]*omega[:,None])[:,:,None]*b[:,None,:])/safe[:,None,None]
    pts[omega<1e-8]=a[omega<1e-8,None,:];pts[:,0]=a;pts[:,-1]=b
    pts/=np.maximum(np.linalg.norm(pts,axis=2)[:,:,None],1e-12)
    pts[omega>np.pi-1e-6,1:4]=np.nan
    lat=np.arctan2(pts[:,:,2],np.hypot(pts[:,:,0],pts[:,:,1]));lon=np.arctan2(pts[:,:,1],pts[:,:,0])
    doy=d.hour.dt.day_of_year.to_numpy();phase=2*np.pi*(doy-1)/365.2425
    dec=(.006918-.399912*np.cos(phase)+.070257*np.sin(phase)-.006758*np.cos(2*phase)+.000907*np.sin(2*phase)-.002697*np.cos(3*phase)+.00148*np.sin(3*phase))
    eq=229.18*(.000075+.001868*np.cos(phase)-.032077*np.sin(phase)-.014615*np.cos(2*phase)-.040849*np.sin(2*phase))
    local=d.hour.dt.hour.to_numpy()[:,None]+.5+np.rad2deg(lon)/15+eq[:,None]/60
    sun=np.sin(lat)*np.sin(dec[:,None])+np.cos(lat)*np.cos(dec[:,None])*np.cos(np.deg2rad(15*(local-12)))
    year=d.hour.dt.year.to_numpy()+(doy-1)/365.2425
    coeff=[np.interp(year,MAG['years'],MAG['coefficients'][c]) for c in ['g11','h11','g10']]
    axis=-np.column_stack(coeff);axis/=np.linalg.norm(axis,axis=1)[:,None]
    magnetic=np.rad2deg(np.arcsin(np.clip((pts*axis[:,None,:]).sum(2),-1,1)))
    values=[np.rad2deg(lat),np.sin(lon),np.cos(lon),sun,np.sin(2*np.pi*local/24),np.cos(2*np.pi*local/24),magnetic,
        np.where(np.isfinite(sun),sun>0,np.nan),np.where(np.isfinite(sun),abs(sun)<np.sin(np.deg2rad(6)),np.nan)]
    out={f'point{p}_{kind}':v[:,j].astype(np.float32) for j,p in enumerate(PERCENT) for kind,v in zip(KINDS,values)}
    return pd.concat([d.drop(columns=[c for c in POINTS if c in d]),pd.DataFrame(out,index=d.index)],axis=1)
