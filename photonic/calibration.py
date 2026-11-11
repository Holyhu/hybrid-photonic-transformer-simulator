import numpy as np
from .core import LayerCalibration

class CalibrationCollector:
    def __init__(self, max_values_per_call=50000):
        self.max_values_per_call=max_values_per_call; self.values={}
    def add(self,name,x):
        a=x.detach().float().abs().reshape(-1).cpu().numpy()
        if len(a)>self.max_values_per_call:
            idx=np.linspace(0,len(a)-1,self.max_values_per_call,dtype=np.int64); a=a[idx]
        self.values.setdefault(name,[]).append(a)
    def finalize(self,percentile):
        out={}
        for name,chunks in self.values.items():
            a=np.concatenate(chunks); r=max(float(np.percentile(a,percentile)),1e-8)
            out[name]=LayerCalibration(r)
        return out
