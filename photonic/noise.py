import math
import torch

Q_E = 1.602176634e-19
K_B = 1.380649e-23

def shot_charge_std_C(total_photocurrent_A, symbol_period_s, L):
    return math.sqrt(max(Q_E * total_photocurrent_A * symbol_period_s * L, 0.0))

def rin_charge_std_C(photocurrent_A, symbol_period_s, L, rin_db_per_Hz):
    rin = 10 ** (rin_db_per_Hz / 10.0)
    B = 1.0 / (2.0 * symbol_period_s)
    relative_rms = math.sqrt(max(rin * B, 0.0))
    return photocurrent_A * relative_rms * symbol_period_s * math.sqrt(L)

def thermal_charge_std_C(current_noise_A_per_sqrtHz, symbol_period_s, L):
    B = 1.0 / (2.0 * symbol_period_s)
    return current_noise_A_per_sqrtHz * math.sqrt(B) * symbol_period_s * math.sqrt(L)

def ktc_charge_std_C(temperature_K, capacitance_F):
    return math.sqrt(max(K_B * temperature_K * capacitance_F, 0.0))

def add_charge_noise(q, sigma_C):
    return q if sigma_C <= 0 else q + torch.randn_like(q) * sigma_C
