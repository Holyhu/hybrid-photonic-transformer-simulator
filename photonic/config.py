from dataclasses import dataclass


@dataclass
class PhotonicConfig:
    # Converter precision
    dac_bits: int = 8
    adc_bits: int = 10

    # Calibration / optical link
    input_percentile: float = 99.9
    laser_power_W: float = 0.01
    insertion_loss_db: float = 3.0
    responsivity_A_per_W: float = 0.8
    symbol_rate_Hz: float = 1e10
    integrator_capacitance_F: float = 5e-13
    adc_full_scale_V: float = 1.0

    # MZM
    mzm_vpi_V: float = 2.4
    mzm_extinction_ratio_db: float = 20.0
    mzm_bias_error_x_rad: float = 0.0
    mzm_bias_error_w_rad: float = 0.0
    mzm_phase_noise_std_rad: float = 0.001

    # Receiver / laser noise
    rin_db_per_Hz: float = -150.0
    thermal_current_noise_A_per_sqrtHz: float = 1e-12
    temperature_K: float = 300.0
    enable_shot_noise: bool = True
    enable_rin: bool = True
    enable_thermal_noise: bool = True
    enable_ktc: bool = True

    # Global RANDOM-noise sensitivity multiplier.
    # 0.0 = all random noise disabled (quantization and deterministic device
    #       nonidealities remain); 1.0 = baseline; 2.0 = 2x noise std.
    # It scales MZM phase-noise std and receiver charge-noise std.
    noise_scale: float = 1.0

    photonic_lm_head: bool = True

    @property
    def symbol_period_s(self):
        return 1.0 / self.symbol_rate_Hz

    @property
    def optical_power_after_loss_W(self):
        return self.laser_power_W * 10 ** (-self.insertion_loss_db / 10.0)
