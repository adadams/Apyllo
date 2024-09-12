import sys
from os.path import abspath

import numpy as np
from numpy.typing import NDArray

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/Apyllo"
)
sys.path.append(APOLLO_DIRECTORY)

from apollo.spectrum.band_bin_and_convolve import (  # noqa: E402
    get_wavelength_bins_from_wavelengths,
)


def spoof_data_file(
    starting_wavelength: float,
    ending_wavelength: float,
    resolution: float,
    output_savename: str,
) -> None:
    output_savename = output_savename if output_savename else "data_spoof.dat"

    number_of_wavelengths: int = int(
        resolution * np.log(ending_wavelength / starting_wavelength) + 1
    )

    wavelengths: NDArray[np.float_] = starting_wavelength * np.exp(
        np.arange(number_of_wavelengths) / resolution
    )

    wavelength_bin_starts, wavelength_bin_ends = get_wavelength_bins_from_wavelengths(
        wavelengths
    )

    dummy_column: NDArray[np.float_] = np.zeros_like(wavelength_bin_starts)

    concatenated_data: NDArray[np.float_] = np.c_[
        wavelength_bin_starts,
        wavelength_bin_ends,
        dummy_column,
        dummy_column,
        dummy_column,
        dummy_column,
    ]

    np.savetxt(output_savename, concatenated_data)


if __name__ == "__main__":
    spoof_data_file(
        starting_wavelength=0.61,
        ending_wavelength=5.30,
        resolution=500,
        output_savename="data_spoof_0.6_5.0_R500.dat",
    )
