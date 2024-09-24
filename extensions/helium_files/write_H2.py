from pathlib import Path
from typing import Final, NamedTuple

import msgspec
import numpy as np
from numba import float64, guvectorize, int64, njit
from numpy.typing import NDArray

WORKING_DIRECTORY: Final[Path] = Path.cwd() / "extensions" / "helium_files"


class OpacityDirectory(msgspec.Struct):
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
        opacity_configuration_file.read(), type=OpacityDirectory
    )

assert Path(
    config.input_opacity_directory
).exists(), "Opacity directory cannot be found."

k: Final[float] = 1.38064852e-16  # boltzmann constant in CGS units
amagat: Final[float] = 2.6867805e19
# weird units for h2, he: Arthur has no idea what "amagat" stands for but it's some combination of physical constants


WORKING_DIRECTORY: Final[Path] = Path.cwd() / "extensions" / "helium_files"

input_hydrogen_spec_file: Final[Path] = WORKING_DIRECTORY / "h2spec.dat"


input_spec_file: NDArray[np.float64] = np.loadtxt(input_hydrogen_spec_file)
h2_grid: NDArray[np.float64] = 10 ** np.where(
    np.isnan(input_spec_file), -33.0, input_spec_file
)

number_of_input_grid_temperatures, number_of_input_grid_wavenumbers = np.shape(h2_grid)

INPUT_GRID_TEMPERATURE_MINIMUM = 75.0  # linear in temperature, pressure is irrelevant
INPUT_GRID_TEMPERATURE_MAXIMUM = (
    3500.0  # originally this said 4000, but grid spacing implies max is 3500
)

input_temperature_grid, input_temperature_grid_step = np.linspace(
    INPUT_GRID_TEMPERATURE_MINIMUM,
    INPUT_GRID_TEMPERATURE_MAXIMUM,
    number_of_input_grid_temperatures,
    retstep=True,
)

INPUT_GRID_WAVENUMBER_MINIMUM: Final[float] = 0.0
INPUT_GRID_WAVENUMBER_MAXIMUM = 19980.0  # cm^-1

input_wavenumber_grid_step: float = (
    INPUT_GRID_WAVENUMBER_MAXIMUM - INPUT_GRID_WAVENUMBER_MINIMUM
) / (number_of_input_grid_wavenumbers - 1)

OUTPUT_WAVELENGTH_MINIMUM: Final[float] = config.minimum_wavelength_in_microns  # um
OUTPUT_WAVELENGTH_MAXIMUM: Final[float] = config.maximum_wavelength_in_microns  # um
OUTPUT_RESOLUTION: Final[float] = config.mean_resolution


def get_number_of_wavelengths(
    starting_wavelength: float, ending_wavelength: float, resolution: float
) -> int:
    return int(
        np.ceil(resolution * np.log(ending_wavelength / starting_wavelength)) + 1
    )


def get_wavelengths_from_number_of_elements_and_resolution(
    starting_wavelength: float, number_of_elements: int, spectral_resolution: float
) -> NDArray[np.float64]:
    return starting_wavelength * np.exp(
        np.arange(number_of_elements) / spectral_resolution
    )


def wavelength_to_wavenumber(wavelength: float) -> float:
    return 1e4 / wavelength


output_wavenumbers: NDArray[np.float64] = wavelength_to_wavenumber(
    get_wavelengths_from_number_of_elements_and_resolution(
        starting_wavelength=OUTPUT_WAVELENGTH_MINIMUM,
        number_of_elements=get_number_of_wavelengths(
            OUTPUT_WAVELENGTH_MINIMUM, OUTPUT_WAVELENGTH_MAXIMUM, OUTPUT_RESOLUTION
        ),
        spectral_resolution=OUTPUT_RESOLUTION,
    )
)


@njit(float64[:, :](float64[:, :], int64[:], int64[:], float64[:], float64[:], float64))
def calculate_opacities(
    input_grid: NDArray[np.float64],
    temperature_indices: NDArray[np.int_],
    wavenumber_indices: NDArray[np.int_],
    temperature_steps: NDArray[np.float64],
    wavenumber_steps: NDArray[np.float64],
    amagat: float = amagat,
) -> NDArray[np.float64]:
    result_array = np.zeros((temperature_indices.shape[0], wavenumber_indices.shape[0]))

    for i in range(temperature_indices.shape[0]):
        for j in range(wavenumber_indices.shape[0]):
            ti = temperature_indices[i]
            wi = wavenumber_indices[j]

            temperature_step = temperature_steps[i]
            wavenumber_step = wavenumber_steps[j]

            low_end = input_grid[ti, wi] + temperature_step * (
                input_grid[ti + 1, wi] - input_grid[ti, wi]
            )
            high_end = input_grid[ti, wi + 1] + temperature_step * (
                input_grid[ti + 1, wi + 1] - input_grid[ti, wi + 1]
            )

            result_array[i, j] = (
                low_end + wavenumber_step * (high_end - low_end)
            ) / amagat**2

    return result_array


def calculate_evenly_spaced_logtemperatures(
    minimum_temperature: float = INPUT_GRID_TEMPERATURE_MINIMUM,
    maximum_temperature: float = INPUT_GRID_TEMPERATURE_MAXIMUM,
    number_of_temperatures: int = config.number_of_temperature_points,
) -> NDArray[np.float64]:
    delta_logT: float = (number_of_temperatures - 1) / np.log10(
        maximum_temperature / minimum_temperature
    )

    return np.log10(minimum_temperature) + (
        np.arange(number_of_temperatures) / delta_logT
    )


class TemperatureIndices(NamedTuple):
    index_of_TP_profile_in_grid: NDArray[np.int_]
    fractional_index_remainder: NDArray[np.float64]


def get_temperature_indices(TP_profile: NDArray[np.float64]) -> TemperatureIndices:
    fractional_index_remainder, index_of_TP_profile_in_grid = np.modf(
        (TP_profile - INPUT_GRID_TEMPERATURE_MINIMUM)
        / (INPUT_GRID_TEMPERATURE_MAXIMUM - INPUT_GRID_TEMPERATURE_MINIMUM)
        * (number_of_input_grid_temperatures - 1)
    )  # .clip(a_max=number_of_input_grid_temperatures - 2)

    index_of_TP_profile_in_grid = index_of_TP_profile_in_grid.clip(
        min=0, max=number_of_input_grid_temperatures - 2
    ).astype(int)

    fractional_index_remainder = np.where(
        index_of_TP_profile_in_grid < 0, 0, fractional_index_remainder
    )
    SMALL_NUDGE = 1e-6
    fractional_index_remainder = np.where(
        index_of_TP_profile_in_grid > number_of_input_grid_temperatures - 1,
        1 - SMALL_NUDGE,
        fractional_index_remainder,
    )

    return TemperatureIndices(
        index_of_TP_profile_in_grid=index_of_TP_profile_in_grid,
        fractional_index_remainder=fractional_index_remainder,
    )


output_temperature_profile: NDArray[np.float64] = (
    10 ** calculate_evenly_spaced_logtemperatures()
)

output_temperature_indices: TemperatureIndices = get_temperature_indices(
    output_temperature_profile
)


class WavenumberIndices(NamedTuple):
    index_of_wavenumber_in_grid: NDArray[np.int_]
    fractional_index_remainder: NDArray[np.float64]


def get_wavenumber_indices(wavenumbers: NDArray[np.float64]) -> WavenumberIndices:
    wavenumber_indices: NDArray[np.int_] = np.clip(
        (wavenumbers - INPUT_GRID_WAVENUMBER_MINIMUM) / input_wavenumber_grid_step,
        a_min=0,
        a_max=number_of_input_grid_wavenumbers - 2,
    ).astype(int)

    lower_wavenumber_bin_edge = (
        INPUT_GRID_WAVENUMBER_MINIMUM + wavenumber_indices * input_wavenumber_grid_step
    )
    upper_wavenumber_bin_edge = lower_wavenumber_bin_edge + input_wavenumber_grid_step

    delta_wavenumber_indices: NDArray[np.float64] = (
        wavenumbers - lower_wavenumber_bin_edge
    ) / (upper_wavenumber_bin_edge - lower_wavenumber_bin_edge)

    return WavenumberIndices(
        index_of_wavenumber_in_grid=wavenumber_indices,
        fractional_index_remainder=delta_wavenumber_indices,
    )


output_wavenumber_indices: WavenumberIndices = get_wavenumber_indices(
    output_wavenumbers
)

output_opacity_grid: NDArray[np.float64] = calculate_opacities(
    h2_grid,
    output_temperature_indices.index_of_TP_profile_in_grid,
    output_wavenumber_indices.index_of_wavenumber_in_grid,
    output_temperature_indices.fractional_index_remainder,
    output_wavenumber_indices.fractional_index_remainder,
    amagat,
)  # [temperature, wavenumber]


def calculate_log_pressure(
    minimum_log_pressure: float = config.minimum_log10_pressure_in_bars
    + 6.0,  # pressure in cgs units = 1e-6 bar = 0.1 Pa
    maximum_log_pressure: float = config.maximum_log10_pressure_in_bars
    + 6.0,  # pressure in cgs units = 1e-6 bar = 0.1 Pa
    number_of_pressure_layers: int = config.number_of_pressure_points,
) -> NDArray[np.float64]:
    return np.logspace(
        start=minimum_log_pressure,
        stop=maximum_log_pressure,
        num=number_of_pressure_layers,
        base=10,
    )


@njit
def calculate_number_of_molecules_per_wavelength(
    log_pressures: NDArray[np.float64],
    TP_profile: NDArray[np.float64],
) -> NDArray[np.float64]:
    # [pressure, temperature, wavenumber]
    return (
        np.expand_dims(log_pressures, axis=1) / np.expand_dims(TP_profile, axis=0) / k
    )


output_pressures: NDArray[np.float64] = calculate_log_pressure()

number_of_spectral_elements: int = get_number_of_wavelengths(
    config.minimum_wavelength_in_microns,
    config.maximum_wavelength_in_microns,
    config.mean_resolution,
)

output_molecular_density: NDArray[np.float64] = np.zeros(
    (
        number_of_spectral_elements,
        config.number_of_pressure_points,
        config.number_of_temperature_points,
    )
)


@guvectorize(
    [(float64[:], float64[:], float64[:], float64[:, :, :])],
    "(w),(p),(t)->(w,p,t)",
)
def calculate_number_density_of_molecules(
    delta_wavenumber_indices: NDArray[np.float64],
    pressures: NDArray[np.float64],
    temperatures: NDArray[np.float64],
    array_to_store_number_density: NDArray[np.float64],
) -> NDArray[np.float64]:
    for i in range(delta_wavenumber_indices.shape[0]):
        array_to_store_number_density[i] = calculate_number_of_molecules_per_wavelength(
            pressures, temperatures
        )


calculate_number_density_of_molecules(
    output_wavenumber_indices.fractional_index_remainder,
    output_pressures,
    output_temperature_profile,
    output_molecular_density,
)  # guvectorize places result in output_molecular_density


output_molecular_density = np.moveaxis(
    output_molecular_density, source=0, destination=-1
)  # swap axes to [pressure, temperature, wavenumber]


total_attenuation_coefficients: NDArray[np.float64] = (
    output_opacity_grid * output_molecular_density
)


H2_header: str = (
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

np.savetxt(
    Path(config.input_opacity_directory) / f"h2only.{config.opacity_catalog_id}.dat",
    total_attenuation_coefficients.reshape(
        (
            len(output_pressures) * len(output_temperature_profile),
            len(output_wavenumbers),
        )
    ),
    fmt="%.6e",
    header=H2_header,
    comments="",
)
