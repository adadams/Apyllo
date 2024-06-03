from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

type Pathlike = str | Path


def load_APOLLO_opacity_file(filepath: Pathlike) -> NDArray[np.float_]:
    return np.loadtxt(filepath, skiprows=1)


def load_APOLLO_opacity_header(filepath: Pathlike) -> str:
    with open(filepath, "r") as file:
        header: str = file.readline().strip()

    return header


def combine_alkalis(
    Na_opacities: NDArray[np.float_],
    K_opacities: NDArray[np.float_],
    header_information: str,
    output_filename: Pathlike = None,
) -> NDArray[np.float_]:
    # solar abundances from the first column of Table 4 in Lodders (2019)
    Na_log_abundance: float = 6.33
    K_log_abundance: float = 5.12

    NaK_number_ratio: float = 10**Na_log_abundance / 10**K_log_abundance
    K_alkali_fraction: float = 1 / (1 + NaK_number_ratio)
    Na_alkali_fraction: float = NaK_number_ratio * K_alkali_fraction

    NaK_data: NDArray[np.float_] = (
        Na_alkali_fraction * Na_opacities + K_alkali_fraction * K_opacities
    )

    if output_filename:
        np.savetxt(
            output_filename,
            NaK_data,
            delimiter=" ",
            header=header_information,
            comments="",
        )

    return NaK_data


def main(
    Na_opacity_filepath: Pathlike,
    K_opacity_filepath: Pathlike,
    output_filename: Pathlike,
) -> NDArray[np.float_]:
    Na_opacities: NDArray[np.float_] = load_APOLLO_opacity_file(Na_opacity_filepath)
    K_opacities: NDArray[np.float_] = load_APOLLO_opacity_file(K_opacity_filepath)
    header_information: Sequence[str] = load_APOLLO_opacity_header(Na_opacity_filepath)

    return combine_alkalis(
        Na_opacities, K_opacities, header_information, output_filename
    )


if __name__ == "__main__":
    NA_OPACITY_FILEPATH = ""
    K_OPACITY_FILEPATH = ""
    ALKALI_OUTPUT_FILEPATH = ""

    main(NA_OPACITY_FILEPATH, K_OPACITY_FILEPATH, ALKALI_OUTPUT_FILEPATH)
