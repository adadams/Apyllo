from pathlib import Path
from typing import Final

import msgspec
import numpy as np
from numba import stencil
from numpy.typing import NDArray

INPUT_HELIUM_SPEC_FILE: Final[Path] = Path.cwd() / "extensions" / "hespec.dat"

k: Final[float] = 1.38064852e-16
amagat: Final[float] = 2.6867805e19  # weird units for h2,he

h2_grid: NDArray[np.float_] = 10 ** np.loadtxt(INPUT_HELIUM_SPEC_FILE)


def get_number_of_wavelengths(
    starting_wavelength: float, ending_wavelength: float, resolution: float
) -> int:
    return np.ceil(resolution * np.log(ending_wavelength / starting_wavelength)) + 1


def get_wavelengths_from_number_of_elements_and_resolution(
    starting_wavelength: float, number_of_elements: int, spectral_resolution: float
) -> NDArray[np.float_]:
    return starting_wavelength * np.exp(
        np.arange(number_of_elements) / spectral_resolution
    )


NUMTEMP, NUMWAVE = np.shape(h2_grid)

WAVENUMBER_MIN = 0.0
WAVENUMBER_MAX = 19980.0  # cm^-1
wavenumber_grid_step: float = (WAVENUMBER_MAX - WAVENUMBER_MIN) / (NUMWAVE - 1)

TMIN = 75.0  # linear in temperature, pressure is irrelevant
TMAX = 3500.0  # originally this said 4000, but grid spacing implies max is 3500
temperature_grid, temperature_grid_step = np.linspace(TMIN, TMAX, NUMTEMP, retstep=True)


@stencil
def interpolation_function(
    array: NDArray[np.float_], temperature_step: float, wavenumber_step: float
) -> float:
    low_end = array[0, 0] + temperature_step * (array[1, 0] - array[0, 0])
    high_end = array[0, 1] + temperature_step * (array[1, 1] - array[0, 1])

    return low_end + wavenumber_step * (high_end - low_end)


def calculate_opacities(
    input_grid: NDArray[np.float_],
    temperature_grid_step: float,
    wavenumber_grid_step: float,
) -> NDArray[np.float_]:
    return (
        interpolation_function(input_grid, temperature_grid_step, wavenumber_grid_step)
        / amagat**2
    )


def calculate_evenly_spaced_logtemperatures(
    minimum_temperature: float = TMIN,
    maximum_temperature: float = TMAX,
    number_of_temperatures: int = NUMTEMP,
) -> NDArray[np.float_]:
    delta_logT: float = (number_of_temperatures - 1) / np.log10(
        maximum_temperature / minimum_temperature
    )

    return np.log10(minimum_temperature) + (
        np.arange(number_of_temperatures) / delta_logT
    )


class TemperatureIndices(msgspec.Struct):
    index_of_TP_profile_in_grid: NDArray[np.int_]
    fractional_index_remainder: NDArray[np.float_]


def temperature_loop(
    TP_profile: NDArray[np.float_],
    TP_grid_profile: NDArray[np.float_],
) -> TemperatureIndices:
    index_of_TP_profile_in_grid, fractional_index_remainder = np.divmod(
        (TP_profile - TMIN) / (TMAX - TMIN) * (NUMTEMP - 1)
    )  # .clip(a_max=NUMTEMP - 2)

    index_of_TP_profile_in_grid = index_of_TP_profile_in_grid.clip(
        a_min=0, a_max=NUMTEMP - 2
    )

    fractional_index_remainder = np.where(
        index_of_TP_profile_in_grid < 0, 0, fractional_index_remainder
    )
    SMALL_NUDGE = 1e-6
    fractional_index_remainder = np.where(
        index_of_TP_profile_in_grid > NUMTEMP - 1,
        1 - SMALL_NUDGE,
        fractional_index_remainder,
    )

    return TemperatureIndices(
        index_of_TP_profile_in_grid=index_of_TP_profile_in_grid,
        fractional_index_remainder=fractional_index_remainder,
    )


class WavelengthIndices(msgspec.Struct):
    index_of_wavenumber_in_grid: NDArray[np.int_]
    fractional_index_remainder: NDArray[np.float_]


def wavelength_loop(wavenumber_grid: NDArray[np.float_]) -> WavelengthIndices: ...


def calculate_log_pressure(
    minimum_log_pressure: float = 0.0,  # pressure in cgs units = 1e-6 bar = 0.1 Pa
    maximum_log_pressure: float = 8.5,  # pressure in cgs units = 1e-6 bar = 0.1 Pa
    number_of_pressure_layers: int = 18,
) -> NDArray[np.float_]:
    return np.logspace(
        start=minimum_log_pressure,
        stop=maximum_log_pressure,
        num=number_of_pressure_layers,
        base=10,
    )


def calculate_number_of_molecules(
    log_pressures: NDArray[np.float_],
    TP_profile: NDArray[np.float_],
    boltzmann_constant: float = k,
) -> NDArray[np.float_]:
    # [pressure, temperature, wavenumber]
    return (
        log_pressures[:, np.newaxis, ...]
        / TP_profile[np.newaxis, :, ...]
        / boltzmann_constant
    )

    """
    for(int j=0; j<18; j++){
        double pressure = pow(10,0.5*j); // pressure in bar
        double nmol = pressure/k/tprof[i]; // compute number density
        double tempopac = opac * nmol;   // convert to cm^2 molecule^-1
    """

    """
    for(int i=0; i<36; i++){

      // compute interpolation indices
      double tl = tprof[i];
      double deltat = (tl-tmin)/(tmax-tmin)*(numtemp-1.);
      int jt = (int)deltat;
      if(jt>numtemp-2) jt = numtemp-2;
      double dti = (tl-logtemp[jt])/(logtemp[jt+1]-logtemp[jt]);
      if(jt<0){
        jt=0.;
        dti=0.;
      }
      if(jt>numtemp-1){
        jt=numtemp-2;
        dti=0.999999;
      }

      xsec[0] = h2grid[jt][jw];
      xsec[1] = h2grid[jt][jw+1];
      xsec[2] = h2grid[jt+1][jw];
      xsec[3] = h2grid[jt+1][jw+1];

      // interpolate
      opr1 = xsec[0] + dti*(xsec[2]-xsec[0]);
      opr2 = xsec[1] + dti*(xsec[3]-xsec[1]);
      opac = opr1 + dwi*(opr2-opr1);
      if(m%5000==0 && i%15==0) printf("%d %d %e %e %e\n",m,i,xsec[0],opr1,opac);

      // opac is given in cm^-1 amg^-2
      opac /= amagat*amagat;
      // convert to cm^5 molecule^-2

      // table is for opacity per unit number density in units of cm^-1 amg^-2
      // unit conversion
      for(int j=0; j<18; j++){
        double pressure = pow(10,0.5*j); // pressure in bar
        double nmol = pressure/k/tprof[i]; // compute number density
        double tempopac = opac * nmol;   // convert to cm^2 molecule^-1

        if(m==10000 && i==29) printf("%e %e %e %e\n",opac,pressure,nmol,tempopac);
        // This is the cross section, and a higher density results in a larger cross section.

        //double namg = (pressure/k/tprof[i])/(1013250./k/273.15); // compute number density in amg
        //double tempopac = opac*namg*namg; //convert to opacity per particle in units of cm^-1 amg^-2
        //tempopac /= pow(1013250./k/273.15,2); // convert to cm^2 per particle
        //if(m%5000==0 && i%15==0) printf("%d %d %d %e %e %e\n",m,i,j,pressure,namg,tempopac);
        //if(waven<10000) printf("%e %e\n",namg,opac);
        h2prof[j][i][m] = tempopac;
    """


def wavenumber_loop() -> ...: ...
