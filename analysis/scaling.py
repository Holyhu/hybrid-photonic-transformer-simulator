import numpy as np

def fit_power_law(N,loss):
    x=np.log(np.asarray(N,dtype=float)); y=np.log(np.asarray(loss,dtype=float))
    slope,intercept=np.polyfit(x,y,1); pred=slope*x+intercept
    ss_res=float(np.sum((y-pred)**2)); ss_tot=float(np.sum((y-y.mean())**2))
    return {'alpha':-float(slope),'intercept':float(intercept),'r2':1-ss_res/ss_tot if ss_tot>0 else float('nan')}
