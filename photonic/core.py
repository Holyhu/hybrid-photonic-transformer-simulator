import math
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from .config import PhotonicConfig
from .quantization import symmetric_fake_quantize, adc_quantize_voltage
from .mzm import signed_mzm_encode
from .noise import (
    shot_charge_std_C,
    rin_charge_std_C,
    thermal_charge_std_C,
    ktc_charge_std_C,
    add_charge_noise,
)
from .stats import PHOTO_STATS


@dataclass
class LayerCalibration:
    input_abs_range: float


class PhotonicLinearEngine:
    """Linear-layer replacement for the realistic photonic simulation.

    The publication path uses mode='realistic'.  mode='ideal' is kept only for
    the internal unit test so that the optical MAC math can be checked against
    torch.nn.functional.linear.
    """

    def __init__(self, cfg: PhotonicConfig, mode='realistic', use_ste=False):
        if mode not in {'ideal', 'realistic'}:
            raise ValueError("mode must be ideal or realistic")
        if cfg.noise_scale < 0:
            raise ValueError("noise_scale must be >= 0")
        self.cfg = cfg
        self.mode = mode
        self.use_ste = use_ste

    def forward(self, x, weight, bias, layer_name, calibration):
        L = weight.shape[1]
        original_shape = x.shape
        x2 = x.reshape(-1, L)
        x_scale = max(float(calibration.input_abs_range), 1e-12)
        w_scale = max(float(weight.detach().abs().max().cpu()), 1e-12)

        if self.mode == 'ideal':
            xq, wq = x2, weight
        else:
            xq = symmetric_fake_quantize(x2, x_scale, self.cfg.dac_bits, self.use_ste)
            wq = symmetric_fake_quantize(weight, w_scale, self.cfg.dac_bits, self.use_ste)

        xn = torch.clamp(xq / x_scale, -1.0, 1.0)
        wn = torch.clamp(wq / w_scale, -1.0, 1.0)
        ideal = self.mode == 'ideal'

        # noise_scale multiplies the random phase-noise standard deviation.
        phase_noise_std = 0.0 if ideal else self.cfg.mzm_phase_noise_std_rad * self.cfg.noise_scale
        xopt = signed_mzm_encode(
            xn,
            self.cfg.mzm_vpi_V,
            self.cfg.mzm_extinction_ratio_db,
            0.0 if ideal else self.cfg.mzm_bias_error_x_rad,
            phase_noise_std,
            ideal,
        )
        wopt = signed_mzm_encode(
            wn,
            self.cfg.mzm_vpi_V,
            self.cfg.mzm_extinction_ratio_db,
            0.0 if ideal else self.cfg.mzm_bias_error_w_rad,
            phase_noise_std,
            ideal,
        )

        normalized_sum = xopt @ wopt.t()
        P = self.cfg.optical_power_after_loss_W
        R = self.cfg.responsivity_A_per_W
        Ts = self.cfg.symbol_period_s
        kC = R * P * Ts
        q_signal = normalized_sum * kC

        if self.mode == 'realistic':
            I_sum = R * P
            var = 0.0
            if self.cfg.enable_shot_noise:
                s = shot_charge_std_C(I_sum, Ts, L)
                var += s * s
            if self.cfg.enable_rin:
                s = rin_charge_std_C(I_sum, Ts, L, self.cfg.rin_db_per_Hz)
                var += s * s
            if self.cfg.enable_thermal_noise:
                s = thermal_charge_std_C(self.cfg.thermal_current_noise_A_per_sqrtHz, Ts, L)
                var += s * s
            if self.cfg.enable_ktc:
                s = ktc_charge_std_C(self.cfg.temperature_K, self.cfg.integrator_capacitance_F)
                var += s * s

            # noise_scale is explicitly a STANDARD-DEVIATION multiplier.
            sigma_C = math.sqrt(var) * self.cfg.noise_scale
            q_noisy = add_charge_noise(q_signal, sigma_C)
            v = q_noisy / self.cfg.integrator_capacitance_F
            v_adc, sat = adc_quantize_voltage(
                v,
                self.cfg.adc_full_scale_V,
                self.cfg.adc_bits,
                self.use_ste,
            )
            q_out = v_adc * self.cfg.integrator_capacitance_F
        else:
            q_out = q_signal
            sat = torch.tensor(0.0, device=x.device)

        y = q_out / kC * x_scale * w_scale
        if bias is not None:
            y = y + bias

        # Layer-output quality relative to the same digital linear layer.
        yd = F.linear(x2, weight, bias)
        err = y - yd
        mse = torch.mean(err.square())
        sig = torch.sqrt(torch.mean(yd.square()) + 1e-30)
        er = torch.sqrt(torch.mean(err.square()) + 1e-30)
        PHOTO_STATS.update(
            layer_name,
            L,
            float(sat.detach().cpu()),
            float(mse.detach().cpu()),
            float(sig.detach().cpu()),
            float(er.detach().cpu()),
        )
        return y.reshape(*original_shape[:-1], weight.shape[0])
