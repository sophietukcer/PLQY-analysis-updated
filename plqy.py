from dataclasses import dataclass
import logging
from typing import Tuple
import numpy as np
from scipy.constants import Planck, speed_of_light
from lmfit.models import ConstantModel, VoigtModel

from utils import integrate_range

logger = logging.getLogger(__name__)


@dataclass
class PLQYResult:
    """Dataclass holding all computed physical parameters and output spectra."""

    plqy_percent: float
    absorptance_percent: float
    optical_density: float
    laser_power_val: float
    laser_power_unit: str
    laser_power_error: float
    peak_centre_nm: float
    fwhm_nm: float
    wavelengths: np.ndarray
    spec_empty: np.ndarray
    spec_in: np.ndarray
    spec_out: np.ndarray
    spec_proc: np.ndarray


def remove_stray_light(
    spec_in: np.ndarray,
    spec_out: np.ndarray,
    spec_empty: np.ndarray,
    wavelengths: np.ndarray,
    pl_range: Tuple[float, float],
    stray_range: Tuple[float, float] = (370.0, 390.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Correct for stray light background in the PL wavelength region.

    Parameters
    ----------
    spec_in : np.ndarray
        Sample in-beam spectrum.
    spec_out : np.ndarray
        Sample out-of-beam spectrum.
    spec_empty : np.ndarray
        Empty sphere spectrum.
    wavelengths : np.ndarray
        Wavelength array in nm.
    stray_range : Tuple[float, float]
        (min_wl, max_wl) below excitation where true signal should be zero.
    pl_range : Tuple[float, float]
        (min_wl, max_wl) region where photoluminescence occurs.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Stray-light corrected (in, out, empty) spectra.
    """
    logger.info("Applying stray light background correction in range %s nm.", stray_range)

    def _get_avg_signal(arr: np.ndarray) -> float:
        mask = (wavelengths >= stray_range[0]) & (wavelengths <= stray_range[1])
        return float(np.trapezoid(arr[mask], wavelengths[mask]))

    avg_empty = _get_avg_signal(spec_empty)
    avg_out = _get_avg_signal(spec_out)
    avg_in = _get_avg_signal(spec_in)

    if avg_empty == 0 or avg_out == 0 or avg_in == 0:
        logger.warning("Stray light baseline integration returned zero. Skipping correction.")
        return spec_in, spec_out, spec_empty

    def _correct_band(arr: np.ndarray, avg_arr: float) -> np.ndarray:
        # corrects for stray light in the PL range, meaning _empty should be zero #
        corr = arr.copy()
        pl_mask = (wavelengths > pl_range[0]) & (wavelengths < pl_range[1])
        empty_pl = spec_empty[pl_mask]
        arr_pl = arr[pl_mask]

        corrected_pl = (arr_pl * (avg_empty / avg_arr) - empty_pl) * (avg_arr / avg_empty)
        corr[pl_mask] = corrected_pl
        return corr

    in_corr = _correct_band(spec_in, avg_in)
    out_corr = _correct_band(spec_out, avg_out)
    empty_corr = _correct_band(spec_empty, avg_empty)

    return in_corr, out_corr, empty_corr


def calculate_laser_power(
    spec_empty: np.ndarray,
    wavelengths: np.ndarray,
    laser_range: Tuple[float, float],
) -> Tuple[float, str, float]:
    """ HUGE PROBLEMS HERE!!!!!!!!!!!!!!!!!!! Estimate excitation laser power and power error based on empty sphere counts.

    Parameters
    ----------
    spec_empty : np.ndarray
        Empty sphere spectrum in microWatts/nm.
    wavelengths : np.ndarray
        Wavelength array in nm.
    laser_range : Tuple[float, float]
        (min_wl, max_wl) bounds of the excitation line.

    Returns
    -------
    Tuple[float, str, float]
        (power_value, unit_str, power_error)
    """
    raw_power_uw = integrate_range(spec_empty, wavelengths, laser_range)

    # Determine calibration factor based on laser wavelength band - this needs to be double checked and updated, clearly we don't have a 532 nm or 660 nm laser #
    if 500.0 <= laser_range[0] <= 530.0:
        correction_factor = 1.17  # 532 nm setup
    else:
        correction_factor = 20.0  # 405 nm / 660 nm setup

    power_uw = raw_power_uw * correction_factor
    error_uw = power_uw * 0.05

    if power_uw > 100.0:
        val_mw = power_uw / 1000.0
        err_mw = error_uw / 1000.0
        return val_mw, "mW", err_mw
    
    return power_uw, "µW", error_uw


def fit_voigt_peak(
    wavelengths: np.ndarray,
    emission_signal: np.ndarray,
    fit_range: Tuple[float, float],
) -> Tuple[float, float, np.ndarray]:
    """Fit a Voigt profile to the PL emission line to extract peak centre and FWHM.

    Parameters
    ----------
    wavelengths : np.ndarray
        Wavelength array in nm.
    emission_signal : np.ndarray
        Sample minus empty emission intensity.
    fit_range : Tuple[float, float]
        (min_wl, max_wl) bounds for peak fitting.

    Returns
    -------
    Tuple[float, float, np.ndarray]
        (peak_centre_nm, fwhm_nm, fitted_curve_array)
    """
    mask = (wavelengths > fit_range[0]) & (wavelengths < fit_range[1])
    wl_fit = wavelengths[mask]
    sp_fit = emission_signal[mask]

    if len(wl_fit) == 0:
        logger.warning("No data found within fit_range %s nm. Returning zero peak properties.", fit_range)
        return 0.0, 0.0, np.array([])

    model = VoigtModel() + ConstantModel()
    params = model.make_params(
        amplitude=float(np.max(sp_fit)),
        centre=float(wl_fit[np.argmax(sp_fit)]),
        sigma=10.0,
        gamma=10.0,
        c=0.0,
    )

    try:
        fit_result = model.fit(sp_fit, params, x=wl_fit)
        centre = float(fit_result.params["centre"].value)
        fwhm = float(fit_result.params["fwhm"].value)
        best_fit = fit_result.best_fit
        logger.info("Voigt fit successful: Peak centre = %.2f nm, FWHM = %.2f nm", centre, fwhm)
        return centre, fwhm, best_fit
    except Exception as err:
        logger.error("Voigt fitting failed: %s", err)
        return 0.0, 0.0, sp_fit


def compute_plqy(
    wavelengths: np.ndarray,
    spec_in: np.ndarray,
    spec_out: np.ndarray,
    spec_empty: np.ndarray,
    laser_range: Tuple[float, float],
    pl_range: Tuple[float, float],
) -> Tuple[PLQYResult, float, float, np.ndarray]:
    """

    Parameters
    ----------
    wavelengths : np.ndarray
        Wavelengths in nm.
    spec_in : np.ndarray
        Calibrated in-beam intensity (µW/nm).
    spec_out : np.ndarray
        Calibrated out-of-beam intensity (µW/nm).
    spec_empty : np.ndarray
        Calibrated empty sphere intensity (µW/nm).
    laser_range : Tuple[float, float]
        (min_wl, max_wl) of excitation band.
    pl_range : Tuple[float, float]
        (min_wl, max_wl) of PL emission.
    

    Returns
    -------
    Tuple[PLQYResult, float, float, np.ndarray]
        (result_dataclass, peak_centre_nm, fwhm_nm, voigt_fit_curve)
    """
    # Conversion factor from W/nm spectrum to photon flux: E_photon = (h * c) / (lambda in meters)
    photon_energy_j = (Planck * speed_of_light) / (1e-9 * wavelengths)
    
    
    in_photons = spec_in / photon_energy_j
    out_photons = spec_out / photon_energy_j
    empty_photons = spec_empty / photon_energy_j

    
    in_laser_photons = integrate_range(in_photons, wavelengths, laser_range)
    out_laser_photons = integrate_range(out_photons, wavelengths, laser_range)
    empty_laser_photons = integrate_range(empty_photons, wavelengths, laser_range)


    absorptance = 1.0 - (in_laser_photons / out_laser_photons)

    
    in_pl = integrate_range(in_photons, wavelengths, pl_range)
    empty_pl = integrate_range(empty_photons, wavelengths, pl_range)
    out_pl = integrate_range(out_photons, wavelengths, pl_range)

    pl_signal = in_pl - empty_pl - (1.0 - absorptance) * (out_pl - empty_pl)
    
    
    qy = pl_signal / (empty_laser_photons * absorptance)
    plqy_pct = qy * 100.0
    abs_pct = absorptance * 100.0

    
    optical_density = -np.log10(1.0 - absorptance) if absorptance < 1.0 else np.nan

    
    power_val, power_unit, power_err = calculate_laser_power(spec_empty, wavelengths, laser_range)

    
    emission_signal = spec_in - spec_empty
    peak_centre, fwhm, voigt_fit = fit_voigt_peak(wavelengths, emission_signal, pl_range)

    result = PLQYResult(
        plqy_percent=plqy_pct,
        absorptance_percent=abs_pct,
        optical_density=optical_density,
        laser_power_val=power_val,
        laser_power_unit=power_unit,
        laser_power_error=power_err,
        peak_centre_nm=peak_centre,
        fwhm_nm=fwhm,
        wavelengths=wavelengths,
        spec_empty=spec_empty,
        spec_in=spec_in,
        spec_out=spec_out,
        spec_proc=spec_in - spec_out,
    )

    return result, peak_centre, fwhm, voigt_fit