import torch

def ste_round(x):
    return x + (torch.round(x) - x).detach()

def symmetric_fake_quantize(x, abs_range, bits, use_ste=False):
    qmax = 2 ** (bits - 1) - 1
    r = max(float(abs_range), 1e-12)
    z = torch.clamp(x / r, -1.0, 1.0) * qmax
    zq = ste_round(z) if use_ste else torch.round(z)
    return zq / qmax * r

def adc_quantize_voltage(v, full_scale_V, bits, use_ste=False):
    qmax = 2 ** (bits - 1) - 1
    fs = max(float(full_scale_V), 1e-12)
    sat = (torch.abs(v) > fs).float().mean()
    z = torch.clamp(v / fs, -1.0, 1.0) * qmax
    zq = ste_round(z) if use_ste else torch.round(z)
    return zq / qmax * fs, sat
