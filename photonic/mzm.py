import math
import torch

def extinction_visibility(extinction_ratio_db):
    er = 10 ** (extinction_ratio_db / 10.0)
    return (er - 1.0) / (er + 1.0)

def signed_mzm_encode(z_norm, vpi_V, extinction_ratio_db, bias_error_rad=0.0,
                      phase_noise_std_rad=0.0, ideal=False):
    if ideal:
        return z_norm
    eps = 1e-7
    z = torch.clamp(z_norm, -1.0 + eps, 1.0 - eps)
    theta_cmd = torch.asin(z)
    drive_V = 2.0 * vpi_V / math.pi * theta_cmd
    theta = math.pi * drive_V / (2.0 * vpi_V) + bias_error_rad
    if phase_noise_std_rad > 0:
        theta = theta + torch.randn_like(theta) * phase_noise_std_rad
    return extinction_visibility(extinction_ratio_db) * torch.sin(theta)
