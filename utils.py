from pathlib import Path
from typing import Tuple, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)


def find_data_start_row(file_path: Union[str, Path]) -> int:
    """Inspect a text file line-by-line to find the 0-indexed row where numerical data starts.

    Parameters
    ----------
    file_path : str or Path
        Path to the text file.

    Returns
    -------
    int
        The row index (0-based) where numerical data begins.

    Raises
    ------
    ValueError
        If no numerical data row is found in the file.
    """
    path = Path(file_path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for row_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # Handle tabs, commas, or spaces as column separators
            tokens = line.replace(",", " ").replace("\t", " ").split()

            # Expect at least 2 numeric columns (e.g., Wavelength, Intensity)
            if len(tokens) >= 2:
                try:
                    float(tokens[0])
                    float(tokens[1])
                    return row_idx ## index where numerical data begins
                except ValueError:
                    # Line contains non-numeric text (header/metadata), continue scanning
                    continue

    raise ValueError(f"No valid numerical data found in {path.name}")


def load_spectrum_file(file_path: Union[str, Path]) -> np.ndarray:
    """Load a 2D spectral text file, automatically detecting and skipping header lines.

    Parameters
    ----------
    file_path : str or Path
        Path to the spectrum text file.

    Returns
    -------
    np.ndarray
        2D array of spectral data where column 0 is wavelength and column 1 is intensity.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be parsed into numerical data.
    """
    path = Path(file_path)
    if not path.exists():
        logger.error("File not found: %s", path)
        raise FileNotFoundError(f"File not found: {path}")

    try:
        skip_rows = find_data_start_row(path)
        logger.debug("Detected numerical data starting at row %d in %s", skip_rows, path.name)

        data = np.loadtxt(path, skiprows=skip_rows)
        return data
    except Exception as err:
        logger.error("Failed to parse file %s: %s", path.name, err)
        raise ValueError(f"Could not parse spectral file {path.name}") from err


def scale_baseline_and_time(intensity: np.ndarray, wavelengths:np.ndarray, laser_range: Tuple[float, float], integration_time_ms: float) -> np.ndarray:
    """Subtract baseline noise floor and normalize by integration time. Baseline subtraction is used in addition to subtraction of a background file to correct for thermal drift over time.
def trim_spectrum(data: np.ndarray, config_name: str) -> np.ndarray:
    """Trim outer noisy pixels based on spectrometer hardware profile.

        Parameters
    ----------
    data : np.ndarray
        2D array of spectral data.
    config_name : str
        Spectrometer configuration name or identifier.

    Returns
    -------
    np.ndarray
        Trimmed array.
    """
    if "QE" in config_name:
        return data[4:-5, :]
    elif "Maya" in config_name:
        return data[5:-6, :]
    return data


def scale_baseline_and_time(intensity: np.ndarray, integration_time_ms: float) -> np.ndarray:
    """Subtract baseline noise floor and normalize by integration time.

    Parameters
    ----------
    intensity : np.ndarray
        1D array of intensity values.
    integration_time_ms : float
        Integration time in milliseconds.
    wavelengths : np.ndarray
            Corresponding wavelength array.
    laser_range : Tuple[float, float]
        (min_wl, max_wl) of the laser band in nm.

    Returns
    -------
    np.ndarray
        Baseline-corrected, normalized intensity values per millisecond.
    """
    if integration_time_ms <= 0:
        logger.warning("Integration time is <= 0 ms (%s). Setting to 1.0 to avoid division by zero.", integration_time_ms)
        integration_time_ms = 1.0

    # Baseline noise estimate from below laser peak (with 10 nm buffer)
    indices = np.where(wavelengths < laser_range[0] - 10)
    noise_floor = np.mean(intensity[indices])
    return (intensity - noise_floor) / integration_time_ms


def combine_short_long_spectra(
    short_intensity: np.ndarray,
    long_intensity: np.ndarray,
    wavelengths: np.ndarray,
    laser_range: Tuple[float, float],
) -> np.ndarray:
    """Splice short-exposure and long-exposure intensity arrays around the laser range.

    The short exposure is used inside the laser excitation band to avoid detector saturation,
    while the long exposure is used for high signal-to-noise in the PL region.

    Parameters
    ----------
    short_intensity : np.ndarray
        Intensity array from short exposure.
    long_intensity : np.ndarray
        Intensity array from long exposure.
    wavelengths : np.ndarray
        Corresponding wavelength array.
    laser_range : Tuple[float, float]
        (min_wl, max_wl) of the laser band in nm.

    Returns
    -------
    np.ndarray
        Spliced 1D intensity array.
    """
    low_idx = np.max(np.argwhere(wavelengths < laser_range[0]))
    high_idx = np.min(np.argwhere(wavelengths > laser_range[1]))

    spliced = np.append(long_intensity[: low_idx + 1], short_intensity[low_idx + 1 : high_idx + 1])
    spliced = np.append(spliced, long_intensity[high_idx + 1 :])
    return spliced


def integrate_range(y: np.ndarray, x: np.ndarray, range_bounds: Tuple[float, float]) -> float:
    """Trapezoidal integration of y(x) bounded within a specific x-range.

    Parameters
    ----------
    y : np.ndarray
        1D array of values to integrate.
    x : np.ndarray
        1D array of independent variable (e.g. wavelength).
    range_bounds : Tuple[float, float]
        (min_x, max_x) integration limits.

    Returns
    -------
    float
        Calculated integrated area.
    """
    mask = (x > range_bounds[0]) & (x < range_bounds[1])
    return float(np.trapezoid(y[mask], x=x[mask]))

def load_and_interpolate_calibration(
    cal_file_path: Union[str, Path],
    target_wavelengths: np.ndarray,
) -> np.ndarray:
    """Load calibration data and interpolate factors onto the target wavelength grid.

    Parameters
    ----------
    cal_file_path : str or Path
        Path to calibration text file.
    target_wavelengths : np.ndarray
        1D array of wavelengths from the sample measurement.

    Returns
    -------
    np.ndarray
        1D array of calibration factors matched element-by-element to target_wavelengths.
    """
    path = Path(cal_file_path)
    if not path.exists():
        logger.warning("Calibration file %s not found. Proceeding with uncalibrated counts (1.0).", path)
        return np.ones_like(target_wavelengths)

    skip_rows = find_data_start_row(path)
    data = np.loadtxt(path, skiprows=skip_rows)

    cal_wl = data[:, 0]
    cal_factors = data[:, 1]

    # Interpolate calibration curve onto exact sample wavelengths
    interpolated = np.interp(target_wavelengths, cal_wl, cal_factors)
    logger.info("Successfully loaded and interpolated calibration file: %s", path.name)
    return interpolated