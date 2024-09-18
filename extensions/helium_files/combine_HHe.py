from pathlib import Path
from typing import Final

import msgspec
import numpy as np
from numpy.typing import NDArray

WORKING_DIRECTORY: Final[Path] = Path.cwd() / "extensions" / "helium_files"


class OpacityConfiguration(msgspec.Struct):
    opacity_catalog_id: str
    input_opacity_directory: str

    number_of_pressure_points: int = 18
    minimum_log10_pressure_in_bars: float = -6.0
    maximum_log10_pressure_in_bars: float = 2.5

    number_of_temperature_points: int = 36
    minimum_log10_temperature: float = 1.875061263  # i.e. 75 K
    maximum_log10_temperature: float = 3.599199194  # i.e. 4000 K

    minimum_wavelength_in_microns: float = 0.6
    maximum_wavelength_in_microns: float = 5.0
    mean_resolution: float = 10000


with open(WORKING_DIRECTORY / "configuration.toml", "rb") as opacity_configuration_file:
    config = msgspec.toml.decode(
        opacity_configuration_file.read(), type=OpacityConfiguration
    )

assert Path(
    config.input_opacity_directory
).exists(), "Opacity directory cannot be found."

H2_input_opacity_file: Path = (
    Path(config.input_opacity_directory) / f"h2only.{config.opacity_catalog_id}.dat"
)
He_input_opacity_file: Path = (
    Path(config.input_opacity_directory) / f"he.{config.opacity_catalog_id}.dat"
)

HHe_output_opacity_file: Path = (
    Path(config.input_opacity_directory) / f"h2he.{config.opacity_catalog_id}.dat"
)


def get_number_of_wavelengths(
    starting_wavelength: float, ending_wavelength: float, resolution: float
) -> int:
    return np.ceil(resolution * np.log(ending_wavelength / starting_wavelength)) + 1


He_molar_fraction: float = 0.14
H2_molar_fraction: float = 1 - He_molar_fraction


number_of_spectral_elements: int = get_number_of_wavelengths(
    config.minimum_wavelength_in_microns,
    config.maximum_wavelength_in_microns,
    config.mean_resolution,
)

HHe_header: str = (
    f"{config.number_of_pressure_points} "
    f"{config.minimum_log10_pressure_in_bars} "
    f"{config.maximum_log10_pressure_in_bars} "
    f"{config.number_of_temperature_points} "
    f"{config.minimum_log10_temperature} "
    f"{config.maximum_log10_temperature} "
    f"{number_of_spectral_elements} "
    f"{config.minimum_wavelength_in_microns} "
    f"{config.maximum_wavelength_in_microns} "
    f"{config.mean_resolution}"
)

H2_opacity: NDArray[np.float_] = np.loadtxt(H2_input_opacity_file, skiprows=1)
He_opacity: NDArray[np.float_] = np.loadtxt(He_input_opacity_file, skiprows=1)

assert np.shape(H2_opacity) == np.shape(
    He_opacity
), "Opacity files have different shapes."


def check_header_info_against_user_config(header_line: list[str]) -> str:
    check_statements = []

    if int(header_line[0]) != config.number_of_pressure_points:
        check_statements.append(
            f"Number of pressure points {int(header_line[0])} does not match configuration {config.number_of_pressure_points}.\n"
        )

    if float(header_line[1]) != config.minimum_log10_pressure_in_bars:
        check_statements.append(
            f"Minimum log10 pressure {float(header_line[1])} does not match configuration {config.minimum_log10_pressure_in_bars}.\n"
        )

    if float(header_line[2]) != config.maximum_log10_pressure_in_bars:
        check_statements.append(
            f"Maximum log10 pressure {float(header_line[2])} does not match configuration {config.maximum_log10_pressure_in_bars}.\n"
        )

    if int(header_line[3]) != config.number_of_temperature_points:
        check_statements.append(
            f"Number of temperature points {int(header_line[3])} does not match configuration {config.number_of_temperature_points}.\n"
        )

    if float(header_line[4]) != config.minimum_log10_temperature:
        check_statements.append(
            f"Minimum log10 temperature {float(header_line[4])} does not match configuration {config.minimum_log10_temperature}.\n"
        )

    if float(header_line[5]) != config.maximum_log10_temperature:
        check_statements.append(
            f"Maximum log10 temperature {float(header_line[5])} does not match configuration {config.maximum_log10_temperature}.\n"
        )

    if int(header_line[6]) != number_of_spectral_elements:
        check_statements.append(
            f"Number of spectral elements {int(header_line[6])} does not match configuration {number_of_spectral_elements}.\n"
        )

    if float(header_line[7]) != config.minimum_wavelength_in_microns:
        check_statements.append(
            f"Minimum wavelength {float(header_line[7])} does not match configuration {config.minimum_wavelength_in_microns}.\n"
        )

    if float(header_line[8]) != config.maximum_wavelength_in_microns:
        check_statements.append(
            f"Maximum wavelength {float(header_line[8])} does not match configuration {config.maximum_wavelength_in_microns}.\n"
        )

    if float(header_line[9]) != config.mean_resolution:
        check_statements.append(
            f"Mean resolution {float(header_line[9])} does not match configuration {config.mean_resolution}.\n"
        )

    if check_statements:
        return "".join(check_statements)
    else:
        return "All checks passed!"


with open(H2_input_opacity_file, "r") as f:
    H2_header = f.readline().split()

with open(He_input_opacity_file, "r") as f:
    He_header = f.readline().split()

print(f"Checking H2 input file:\n{check_header_info_against_user_config(H2_header)}")
print(f"Checking He input file:\n{check_header_info_against_user_config(He_header)}")

HHe_opacity: NDArray[np.float_] = (
    He_molar_fraction * He_opacity + H2_molar_fraction * H2_opacity
)

np.savetxt(
    HHe_output_opacity_file,
    HHe_opacity,
    fmt="%.6e",
    header=HHe_header,
    comments="",
)
