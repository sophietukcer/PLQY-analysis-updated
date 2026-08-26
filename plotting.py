

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from plqy import PLQYResult


def generate_plqy_figure(
    result: PLQYResult,
    laser_range: tuple[float, float],
    pl_range: tuple[float, float],
    voigt_fit: np.ndarray,
    short_time_ms: float,
) -> plt.Figure:
    """Generate a 3-panel figure displaying excitation, PL spectra, and fitted peak parameters.

    Parameters
    ----------
    result : PLQYResult
        Dataclass containing computed results and spectra.
    laser_range : tuple[float, float]
        (min, max) laser integration range.
    pl_range : tuple[float, float]
        (min, max) PL region range.
    voigt_fit : np.ndarray
        Array of fitted Voigt intensities.
    short_time_ms : float
        Integration time in ms.

    Returns
    -------
    plt.Figure
        Configured Matplotlib figure.
    """
    fig = plt.figure(figsize=(11, 8))
    gs = gridspec.GridSpec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])

    wl = result.wavelengths

    # Panel 1 & 2: Log-scale excitation and emission spectra
    for ax in [ax1, ax2]:
        ax.semilogy(wl, result.spec_in, label="in")
        ax.semilogy(wl, result.spec_out, label="out")
        ax.semilogy(wl, result.spec_empty, label="empty")
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Intensity [µW/nm]")
        ax.legend()

    # Configure Laser Panel (ax1)
    ax1.set_xlim(laser_range[0] - 15, laser_range[1] + 15)
    ax1.set_ylim(bottom=1e-4)
    ax1.axvline(laser_range[0], linestyle="--", color="k")
    ax1.axvline(laser_range[1], linestyle="--", color="k")
    ax1.annotate(
        f"Laser power: {result.laser_power_val:.2f} ± {result.laser_power_error:.2f} {result.laser_power_unit}\n"
        f"Int. Time: {short_time_ms:.0f} ms",
        xy=(0.05, 0.82),
        xycoords="axes fraction",
    )

    # Configure PL Region Panel (ax2)
    ax2.set_xlim(pl_range[0] - 25, pl_range[1] + 25)
    ax2.axvline(pl_range[0], linestyle="--", color="k")
    ax2.axvline(pl_range[1], linestyle="--", color="k")

    # Panel 3: Net Sample Emission (in - empty) with Voigt fit
    net_in = result.spec_in - result.spec_empty
    net_out = result.spec_out - result.spec_empty

    ax3.plot(wl, net_in, label="in - empty")
    ax3.plot(wl, net_out, label="out - empty")

    # Plot Voigt fit if available
    fit_mask = (wl > pl_range[0]) & (wl < pl_range[1])
    if len(voigt_fit) == np.sum(fit_mask):
        ax3.plot(wl[fit_mask], voigt_fit, "k--", alpha=0.8, label="Voigt Fit")

    ax3.set_xlabel("Wavelength [nm]")
    ax3.set_ylabel("Intensity [µW/nm]")
    ax3.set_xlim(pl_range[0] - 25, pl_range[1] + 25)

    pl_mask = (wl > pl_range[0]) & (wl < pl_range[1])
    if np.any(pl_mask):
        max_val = np.max(net_in[pl_mask])
        ax3.set_ylim(0, 1.2 * max_val if max_val > 0 else 1.0)

    ax3.legend()
    ax3.annotate(
        f"PLQY = {result.plqy_percent:.2f} %\n"
        f"Absorptance = {result.absorptance_percent:.2f} %\n"
        f"OD = {result.optical_density:.2f}\n"
        f"Peak centre = {result.peak_centre_nm:.2f} nm\n"
        f"FWHM = {result.fwhm_nm:.2f} nm",
        xy=(0.05, 0.78),
        xycoords="axes fraction",
    )

    plt.tight_layout()
    return fig