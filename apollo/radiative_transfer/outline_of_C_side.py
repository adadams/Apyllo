from collections.abc import Callable
from enum import Enum, EnumType, StrEnum, auto
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from apollo.radiative_transfer.RT_Toon1989 import RT_Toon1989


class ObservationMode(Enum):
    RESOLVED = 0
    ECLIPSE = 1
    TRANSIT = 2


class RadiativeTransferScheme(StrEnum):
    Toon_TwoStream = auto()
    Toon_OneStream = auto()
    Transmission = auto()


def Planet_init() -> ...: ...


def setParams() -> ...:
    # Loads in geometric parameters (pressure levels) and
    # uses them to calculate the altitude and T-P profile.

    # Loads in the fixed system parameters like stellar radius
    # and the fundamental parameters like Rp and gravity.

    # Checks which cloud model and loads those parameters.

    # NOTE: in Planet.cpp, is "rxsec" without an "s" ever used?
    ...


def setWave() -> ...:
    # Calls the opacity routines like getOpacProf and getTauProf
    # for both high- and low-resolution tables.
    # I'm actually not sure why this is called "setWave"...
    ...


def ProfParam() -> ...: ...


def getSpectrum(
    radiative_transfer_scheme: RadiativeTransferScheme,
) -> ...:
    return RT_functions[radiative_transfer_scheme]


def getClearSpectrum() -> ...: ...


class MaterialProperties(NamedTuple):
    optical_depth_per_layer: NDArray[np.float64]
    single_scattering_albedo: NDArray[np.float64]
    scattering_asymmetry: NDArray[np.float64]


class RTCoordinates(NamedTuple):
    wavelength_in_cm: NDArray[np.float64]
    temperature_in_K: NDArray[np.float64]


def getFlux(
    RT_coordinates: RTCoordinates, material_properties: MaterialProperties
) -> NDArray[np.float64]:
    return RT_Toon1989(**RT_coordinates._asdict(), **material_properties._asdict())


def getFluxOneStream() -> ...:
    # Calls getFluxes per integration angle (8 in total).
    ...


def getFluxes() -> ...:
    # Does the one-stream calculation at some fixed angle of emission.
    ...


def transFlux() -> ...: ...


def getTeff() -> ...:
    # Calls the relevant getFlux-like functions,
    # but for wide-wavelength, low-resolution tables.
    # Then backs out the effective temperature from the
    # flux and radius.
    ...


def getContribution() -> ...:
    # And associated variants.
    ...


RT_functions: dict[EnumType, Callable] = {
    RadiativeTransferScheme.Toon_TwoStream: getFlux,
    RadiativeTransferScheme.Toon_OneStream: getFluxOneStream,
    RadiativeTransferScheme.Transmission: transFlux,
}


def getP() -> ...: ...


def getT() -> ...:
    # This just linearly interpolates within a layer
    # using the altitude.
    ...


def getH() -> ...:
    # NOTE: hprof is calculated in getProfLayer.
    # This just interpolates within a pressure layer.
    ...


def getProfLayer() -> ...:
    # This uses the hydrostatic assumption to get altitude
    # from pressure and temperature.
    ...


def readopac() -> ...:
    # Sets a default value, does the Bound-Free and Free-Free
    # calculations, and then pulls in the gas opacity tables.
    ...


def getOpacProf() -> ...:
    # Calls the Rayleigh scattering routine for gases,
    # then interpolates in both log temperature and
    # log pressure.
    ...


def getTauProf() -> ...:
    # Calculates the tau, g, and w0 for whatever cloud model,
    # and folds it in with the gas scattering/extinction to
    # get the net scattering and absorption per layer.
    ...


def transTauProf() -> ...: ...


def HminFreeFree() -> ...:
    # Just a self-contained numerical calculation.
    ...


def HminBoundFree() -> ...:
    # Just a self-contained numerical calculation.
    ...
