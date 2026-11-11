import math

class PhotonicStats:
    def __init__(self): self.reset()
    def reset(self): self.records = {}
    def update(self, name, L, saturation, mse, signal_rms, error_rms):
        r = self.records.setdefault(name, {'calls':0,'L':int(L),'sat':0.0,'mse':0.0,'sig':0.0,'err':0.0})
        r['calls'] += 1; r['L'] = int(L)
        r['sat'] += float(saturation); r['mse'] += float(mse)
        r['sig'] += float(signal_rms); r['err'] += float(error_rms)
    def rows(self):
        out=[]
        for name,r in sorted(self.records.items()):
            c=max(r['calls'],1); sig=r['sig']/c; err=r['err']/c
            if sig>0 and err>0:
                sndr=20*math.log10(sig/err); enob=(sndr-1.76)/6.02
            else: sndr=enob=float('inf')
            out.append({'layer':name,'L':r['L'],'calls':r['calls'],
                        'mean_adc_saturation_fraction':r['sat']/c,
                        'mean_mse_vs_digital':r['mse']/c,
                        'mean_signal_rms':sig,'mean_error_rms':err,
                        'sndr_db':sndr,'enob':enob})
        return out

PHOTO_STATS = PhotonicStats()
