

import logging
from pathlib import Path
import sys
from gooey import Gooey, GooeyParser
import matplotlib.pyplot as plt
import numpy as np

from plqy import compute_plqy, remove_stray_light
from plotting import generate_plqy_figure
from utils import (
    combine_short_long_spectra,
    load_and_interpolate_calibration,
    load_spectrum_file,
    scale_baseline_and_time,
)

# Configure logging to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("PLQY_App")


@Gooey(
    program_name="PLQY Calculator",
    advanced=True,
    default_size=(850, 700),
    show_success_modal=False,
    return_to_config=True,
    tabbed_groups=True,
)
def main():
    parser = GooeyParser(description="PLQY Calculator")

    # Required File Inputs Group
    req = parser.add_argument_group("1. Primary Input Files", gooey_options={"columns": 1})
    req.add_argument(
        "-sp",
        "--short_path",
        required=True,
        type=str,
        widget="FileChooser",
        help="Path to the '_short_in.txt' file (e.g. 'sample_short_in.txt')",
        gooey_options={"wildcard": "IN files (*_in.txt)|*_in.txt|All files (*.*)|*.*"},
    )
    req.add_argument(
        "-st",
        "--short_time",
        default=100,
        type=float,
        help="Short integration time in ms",
    )
    req.add_argument(
        "-cal",
        "--cal_path",
        type=str,
        widget="FileChooser",
        help="Path to calibration file",
        gooey_options={"wildcard": "Text files (*.txt)|*.txt|All files (*.*)|*.*"},
    )
  
    req.add_argument(
        "-c",
        "--common",
        action="store_true",
        default = True,
        help="Use common background ('bckg.txt') and empty ('empty.txt') files in directory.",
    )

    req.add_argument(
        "-sl",
        "--stray_light",
        action="store_true",
        default=True,
        help="Removes stray light background (Recommended).",
    )

    req.add_argument(
        "-lr",
        "--laser_range",
        nargs=2,
        default=[395, 415],
        type=float,
        help="Laser band (min max)",
    )
    req.add_argument(
        "-plr",
        "--pl_range",
        nargs=2,
        default=[560, 850],
        type=float,
        help="PL detection band (min max)",
    )

    req.add_argument(
        "-lp",
        "--long_path",
        type=str,
        default="",
        widget="FileChooser",
        help="Path to long exposure 'long_in.txt' file",
    )
    req.add_argument(
        "-lt",
        "--long_time",
        default=5000,
        type=float,
        help="Integration time for long measurement in ms",
    )

    args = parser.parse_args()

    short_in_path = Path(args.short_path).resolve()
    work_dir = short_in_path.parent
    short_name = short_in_path.name

    logger.info("Processing sample: %s", short_name)

    # File Name Derivations
    if args.common:
        bckg_path = work_dir / "bckg.txt"
        empty_path = work_dir / "empty.txt"
        out_path = work_dir / short_name.replace("in.txt", "out.txt")
    else:
        bckg_path = work_dir / short_name.replace("in.txt", "bckg.txt")
        empty_path = work_dir / short_name.replace("in.txt", "empty.txt")
        out_path = work_dir / short_name.replace("in.txt", "out.txt")

    # Load Short Exposure Arrays
    raw_in = load_spectrum_file(short_in_path)
    raw_bckg = load_spectrum_file(bckg_path)
    raw_empty = load_spectrum_file(empty_path)
    raw_out = load_spectrum_file(out_path)

    #raw_in = trim_spectrum(raw_in, args.cal_path or "")
    wavelengths = raw_in[:, 0]

    # Baseline & Integration Time Normalization
    short_in_proc = scale_baseline_and_time(raw_in[:, 1] - raw_bckg[:, 1], args.short_time)
    short_out_proc = scale_baseline_and_time(raw_out[:, 1] - raw_bckg[:, 1], args.short_time)
    short_empty_proc = scale_baseline_and_time(raw_empty[:, 1] - raw_bckg[:, 1], args.short_time)

    # Handle Optional Spliced Long Exposure
    if args.long_path and Path(args.long_path).exists():
        logger.info("Splicing long integration time spectrum...")
        long_in_path = Path(args.long_path).resolve()
        long_out_path = work_dir / long_in_path.name.replace("in.txt", "out.txt")
        long_bckg_path = work_dir / "long_bckg.txt"
        long_empty_path = work_dir / "long_empty.txt"
        

        raw_long_in = load_spectrum_file(long_in_path)
        raw_long_bckg = load_spectrum_file(long_bckg_path)
        raw_long_empty = load_spectrum_file(long_empty_path)
        raw_long_out = load_spectrum_file(long_out_path)

        long_in_proc = scale_baseline_and_time(raw_long_in[:, 1] - raw_long_bckg[:, 1], args.long_time)
        long_out_proc = scale_baseline_and_time(raw_long_out[:, 1] - raw_long_bckg[:, 1], args.long_time)
        long_empty_proc = scale_baseline_and_time(raw_long_empty[:, 1] - raw_long_bckg[:, 1], args.long_time)

        counts_in = combine_short_long_spectra(short_in_proc, long_in_proc, wavelengths, tuple(args.laser_range))
        counts_out = combine_short_long_spectra(short_out_proc, long_out_proc, wavelengths, tuple(args.laser_range))
        counts_empty = combine_short_long_spectra(short_empty_proc, long_empty_proc, wavelengths, tuple(args.laser_range))
    else:
        counts_in = short_in_proc
        counts_out = short_out_proc
        counts_empty = short_empty_proc

    # Interpolate & Apply Calibration Curve
    if args.cal_path:
        cal = load_and_interpolate_calibration(args.cal_path, wavelengths)
    else:
        logger.warning("No calibration file supplied. Using uncalibrated counts.")
        cal = np.ones_like(wavelengths)

    spec_in = counts_in * cal
    spec_out = counts_out * cal
    spec_empty = counts_empty * cal

    # Stray Light Background Correction
    if args.stray_light:
        spec_in, spec_out, spec_empty = remove_stray_light(
            spec_in,
            spec_out,
            spec_empty,
            wavelengths,
            stray_range=(370.0, 390.0),
            pl_range=tuple(args.pl_range),
        )

    # Compute Final PLQY
    result, centre, fwhm, voigt_fit = compute_plqy(
        wavelengths=wavelengths,
        spec_in=spec_in,
        spec_out=spec_out,
        spec_empty=spec_empty,
        laser_range=tuple(args.laser_range),
        pl_range=tuple(args.pl_range),
    )

    # Print Summary to Gooey Console
    print("\n" + "=" * 40)
    print(f"RESULTS ({short_name})")
    print("=" * 40)
    print(f"PLQY         : {result.plqy_percent:.2f} %")
    print(f"Absorptance  : {result.absorptance_percent:.2f} %")
    print(f"Optical Dens.: {result.optical_density:.2f}")
    print(f"Laser Power  : {result.laser_power_val:.2f} ± {result.laser_power_error:.2f} {result.laser_power_unit}")
    print(f"Peak centre  : {result.peak_centre_nm:.2f} nm")
    print(f"FWHM         : {result.fwhm_nm:.2f} nm")
    print("=" * 40 + "\n")

    # Save Output Plot / Text Data
    fig = generate_plqy_figure(
        result=result,
        laser_range=tuple(args.laser_range),
        pl_range=tuple(args.pl_range),
        voigt_fit=voigt_fit,
        short_time_ms=args.short_time,
    )

    pdf_out_path = work_dir / short_name.replace("in.txt", "fig.pdf")
    txt_out_path = work_dir / short_name.replace("in.txt", "spectra.txt")

    fig.savefig(pdf_out_path, format="pdf", bbox_inches="tight")
    logger.info("Saved plot to %s", pdf_out_path.name)

    spec_matrix = np.c_[wavelengths, spec_empty, spec_in, spec_out, spec_in - spec_out]
    np.savetxt(
        txt_out_path,
        spec_matrix,
        delimiter="\t",
        fmt="%.5e",
        header="Wavelength\tempty\tin\tout\tproc",
        comments="",
    )
    logger.info("Saved spectrum matrix to %s", txt_out_path.name)

    plt.show()


if __name__ == "__main__":
    main()
