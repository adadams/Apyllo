import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from os.path import abspath
from typing import Any, Final, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/APOLLO"
)
if APOLLO_DIRECTORY not in sys.path:
    sys.path.append(APOLLO_DIRECTORY)

import apollo.src.ApolloFunctions as af  # noqa: E402
from apollo.Apollo_ReadInputsfromFile import (  # noqa: E402
    DataParameters,
    ModelParameters,
    OpacityParameters,
    ParameterValue,
)
from apollo.planet import CPlanetBlueprint  # noqa: E402
from apollo.spectrum.read_spectral_data_into_xarray import (  # noqa: E402
    find_band_limits_from_wavelength_bins,  # noqa: E402
)
from apollo.src.wrapPlanet import PyPlanet  # noqa: E402
from custom_types import Pathlike  # noqa: E402
from user.TP_models import TP_models  # noqa: E402

PARSEC_IN_REARTH: Final[float] = 4.838e9


def set_default_output_filename(
    short: bool, name: str, modestr: str, pllen: int, nsteps: int
) -> str:
    if short:
        outfile = "/" + name + "."
    else:
        outfile = (
            "/"
            + name
            + "."
            + modestr
            + "."
            + str(pllen)
            + "params"
            + str(int(nsteps / 1000))
            + "k."
        )

    return outfile


def select_TP_model_function(gray: bool, atmtype: str) -> Callable:
    if gray:
        return TP_models["gray"]

    elif atmtype in TP_models:
        return TP_models[atmtype]

    else:
        return TP_models["verbatim"]


def get_number_of_walkers(
    sampler: str, ndim: int, nwalkers: int, override: bool
) -> int:
    if override:
        return ndim * 2 + 2

    elif sampler == "emcee":
        if nwalkers == 0:
            nwalkers = ndim * 8  # Default number of walkers
        if nwalkers < 2 * ndim:
            nwalkers = ndim * 2 + 2  # Minimum number of walkers
        if nwalkers % 2 == 1:
            nwalkers = nwalkers + 1  # Number of walkers must be even

        return nwalkers

    elif sampler == "dynesty":
        return 1

    else:
        raise ValueError("Invalid sampler type.")


override_parameters: dict[str, Any] = {
    "parallel": False,
    "printfull": True,
    "nsteps": 2,
    # "nwalkers": ndim * 2 + 2,
}


@dataclass
class RetrievalParameter(slots=True):
    value: ParameterValue
    guess: float
    mu: float
    sigma: float
    bounds: tuple[float]


def convert_area_ratio_to_radius(
    area_ratio: float | ArrayLike[np.float_], dist: float
) -> float:
    return 10**area_ratio * dist**2 * PARSEC_IN_REARTH**2


def convert_area_parameter_to_radius_parameter(
    area_parameter: RetrievalParameter, dist: float
) -> RetrievalParameter:
    radius = convert_area_ratio_to_radius(area_parameter.value, dist)
    guess = convert_area_ratio_to_radius(area_parameter.guess, dist)
    mu = convert_area_ratio_to_radius(area_parameter.mu, dist)
    sigma = (guess * (10**area_parameter.sigma - 1)) * mu
    bounds = convert_area_ratio_to_radius(area_parameter.bounds, dist)

    return RetrievalParameter(radius, guess, mu, sigma, bounds)


@dataclass
class DataContainer:
    # NOTE: this is a clone of an existing container for APOLLO-style data.
    wavelo: NDArray[np.float_]
    wavehi: NDArray[np.float_]
    wavemid: Optional[NDArray[np.float_]]
    flux: NDArray[np.float_]


@dataclass
class DataContainerwithErrors(DataContainer):
    errlo: NDArray[np.float_]
    errhi: NDArray[np.float_]


def read_in_observations(datain: Pathlike) -> DataContainerwithErrors:
    # Read in observations
    # Note: header contains info about star needed for JWST pipeline

    print("Reading in observations.")
    fobs = open(datain, "r")

    obslines = fobs.readlines()
    obslength = len(obslines)

    wavelo = np.zeros(obslength)
    wavehi = np.zeros(obslength)
    flux = np.zeros(obslength)
    errlo = np.zeros(obslength)
    errhi = np.zeros(obslength)

    for i in range(0, obslength):
        wavelo[i] = obslines[i].split()[0]
        wavehi[i] = obslines[i].split()[1]
        flux[i] = obslines[i].split()[5]
        errlo[i] = obslines[i].split()[3]
        errhi[i] = obslines[i].split()[4]

    wavelo = np.round(wavelo, 5)
    wavehi = np.round(wavehi, 5)
    wavemid = (wavehi + wavelo) / 2.0

    return DataContainerwithErrors(wavelo, wavehi, wavemid, flux, errlo, errhi)


@dataclass
class BandSpecs:
    bandindex: list[tuple]
    modindex: NDArray[np.int_]
    modwave: NDArray[np.float_]


def band_data(observations: DataContainer) -> dict[tuple, DataContainerwithErrors]:
    # Separate out individual bands
    band_indices, *banded_spectra = af.FindBands(
        observations.wavelo, observations.wavehi, observations.flux, observations.errhi
    )

    banded_data: dict[tuple, DataContainerwithErrors] = {
        band_index: DataContainerwithErrors(banded_spectrum)
        for band_index, banded_spectrum in zip(band_indices, banded_spectra)
    }
    return banded_data


def band_convolve_and_bin_observations(
    observations: DataContainer, data_parameters: DataParameters
) -> None:
    # Separate out individual bands
    bandlo, bandhi, bandflux, banderr = band_data(observations).values()
    # nband = len(bandhi)

    # Convolve the observations to account for effective resolving power or fit at lower resolving power
    convflux, converr = af.ConvBands(bandflux, banderr, data_parameters.dataconv)

    # Bin the observations to fit a lower sampling resolution
    binlo, binhi, binflux, binerr = af.BinBands(
        bandlo, bandhi, convflux, converr, data_parameters.databin
    )

    binmid = np.zeros(len(binlo))
    for i in range(0, len(binlo)):
        binmid[i] = (binlo[i] + binhi[i]) / 2.0

    return DataContainerwithErrors(binlo, binhi, binmid, binflux, binerr, binerr)


def calculate_total_flux_in_CGS(observations: DataContainer) -> float:
    binlen = len(observations.flux)
    binlo = observations.wavelo
    binhi = observations.wavehi
    binflux = observations.flux

    totalflux = 0
    for i in range(0, binlen):
        totalflux = totalflux + binflux[i] * (binhi[i] - binlo[i]) * 1.0e-4

    return totalflux


def get_wavelengths_from_data(data_parameters: DataParameters) -> tuple[float]:
    unbinned_data: DataContainerwithErrors = read_in_observations(
        data_parameters.datain
    )

    binned_data: DataContainerwithErrors = band_convolve_and_bin_observations(
        unbinned_data, data_parameters
    )

    return (binned_data.wavelo, binned_data.wavehi)


def calculate_wavelength_calibration_limits(
    deltaL_parameter: RetrievalParameter | None,
) -> tuple[float]:
    if deltaL_parameter is not None:
        return deltaL_parameter.bounds * 0.001
    else:
        return (0, 0)


def calculate_wavelength_limits(
    wavelo: ArrayLike[float],
    wavehi: ArrayLike[float],
    deltaL_parameter: RetrievalParameter | None = None,
) -> tuple[float]:
    minDL, maxDL = calculate_wavelength_calibration_limits(deltaL_parameter)

    # NOTE: this looks wrong, check?
    wavei = np.max(wavelo) + minDL
    wavef = np.min(wavehi) + maxDL
    return (wavei, wavef)


def select_default_opacity_tables(wavei: float, wavef: float) -> str:
    # wavei, wavef assumed to be in microns.
    if wavei < 5.0 and wavef < 5.0:
        if wavei < 0.6:
            wavei = 0.6
        return "nir"
    elif wavei < 5.0 and wavef > 5.0:
        if wavei < 0.6:
            wavei = 0.6
        if wavef > 30.0:
            wavef = 30.0
        return "wide"
    elif wavei > 5.0 and wavef > 5.0:
        if wavef > 30.0:
            wavef = 30.0
        return "mir"
    else:
        raise ValueError(
            "None of the default opacity tables completely cover the wavelengths of the data you provided."
        )
    # NOTE: wavei and wavef are modified, presumably to "chop" the input spectrum to match the range of the
    # default opacity tables, but they are local variables here. I need to fix this.


def get_unbanded_wavelengths_from_opacity_tables(
    opacdir: str,
    opacity_catalog_name: str,
    degrade: int | float,
    fiducial_species_name: str = "h2o",
) -> NDArray[np.float_]:
    # Compute hires spectrum wavelengths
    opacfile = f"{opacdir}/gases/{fiducial_species_name}.{opacity_catalog_name}.dat"
    with open(opacfile, "r") as fopac:
        opacshape = fopac.readline().split()
    nwave = (int)(opacshape[6])
    lmin = (float)(opacshape[7])
    resolv = (float)(opacshape[9])

    opaclen = (int)(
        np.floor(nwave / degrade)
    )  # THIS SEEMED TO NEED A +1 IN CERTAIN CASES.
    opacwave = np.zeros(opaclen)
    for i in range(0, opaclen):
        opacwave[i] = lmin * np.exp(i * degrade / resolv)
    return opacwave


def set_Teff_calculation_wavelengths_from_opacity_tables(
    opacdir: str, fiducial_species_name: str = "h2o"
) -> NDArray[np.float_]:
    return get_unbanded_wavelengths_from_opacity_tables(
        opacdir, "lores", 1.0, fiducial_species_name
    )


def pare_down_model_wavelengths(
    band_limits: tuple[tuple[float]],
    opacwave: NDArray[np.float_],
    minDL: float,
    maxDL: float,
) -> NDArray[np.float_]:
    j_starts, j_ends = af.SliceModel_bindex(band_limits, opacwave, minDL, maxDL)

    return af.SliceModel_modwave(opacwave, j_starts, j_ends)


def set_model_wavelengths_from_opacity_tables_and_data(
    data_parameters: DataParameters,
    opacity_parameters: OpacityParameters,
    minDL: float = 0.0,
    maxDL: float = 0.0,
) -> NDArray[np.float_]:
    unbinned_data: DataContainerwithErrors = read_in_observations(
        data_parameters.datain
    )
    data_band_limits: tuple[tuple[float]] = find_band_limits_from_wavelength_bins(
        unbinned_data.wavelo, unbinned_data.wavehi
    )

    opacdir: str = opacity_parameters.opacdir
    opacity_catalog_name: str = opacity_parameters.opacity_catalog_name
    degrade: int | float = opacity_parameters.degrade
    unbanded_opacity_wavelengths: NDArray[np.float_] = (
        get_unbanded_wavelengths_from_opacity_tables(
            opacdir=opacdir, opacity_catalog_name=opacity_catalog_name, degrade=degrade
        )
    )

    banded_opacity_wavelengths: NDArray[np.float_] = pare_down_model_wavelengths(
        band_limits=data_band_limits,
        opacwave=unbanded_opacity_wavelengths,
        minDL=minDL,
        maxDL=maxDL,
    )

    return banded_opacity_wavelengths


def get_index_for_polynomial_normalization(bandindex: list[tuple]) -> int | None:
    polyindex = None

    for i in range(1, len(bandindex)):
        if bandindex[i][0] < bandindex[i - 1][0]:
            polyindex = i

    return polyindex


def normalize_banded_data(
    bands_of_data: list[DataContainerwithErrors], band_specs: BandSpecs
) -> DataContainerwithErrors:
    # Handle bands and optional polynomial fitting
    polyindex = get_index_for_polynomial_normalization(band_specs.bandindex)
    if polyindex is None:
        raise ValueError("Could not calculate an index for polynomial fitting.")

    normlo = bands_of_data[0].wavelo
    normhi = bands_of_data[0].wavehi
    normflux = bands_of_data[0].flux
    normerr = bands_of_data[0].errhi
    for i in range(1, len(bands_of_data)):
        normlo = np.r_[normlo, bands_of_data[i].wavelo]
        normhi = np.r_[normhi, bands_of_data[i].wavehi]
        normflux = np.r_[normflux, bands_of_data[i].flux]
        normerr = np.r_[normerr, bands_of_data[i].errhi]
    normmid = (normlo + normhi) / 2.0

    slennorm = []
    elennorm = []
    for i in range(polyindex, len(band_specs.bandindex)):
        slennorm.append(band_specs.bandindex[i][0])
        elennorm.append(band_specs.bandindex[i][1])

    masternorm = af.NormSpec(normmid, normflux, slennorm, elennorm)
    fluxspecial = np.concatenate(
        (normerr[0 : slennorm[0]], normflux[slennorm[0] :]), axis=0
    )  # NOTE: this line needs clarification and checking.
    mastererr = af.NormSpec(normmid, fluxspecial, slennorm, elennorm)

    return DataContainerwithErrors(
        wavelo=normlo, wavehi=normhi, flux=masternorm, errlo=mastererr, errhi=mastererr
    )


@dataclass
class ModelSpectrumBinIndices:
    ibinlo: float
    ibinhi: float
    delibinlo: float
    delibinhi: float


def get_model_spectral_bin_indices(
    modwave: NDArray[np.float_], binlo, binhi
) -> ModelSpectrumBinIndices:
    bins = af.GetBins(modwave, binlo, binhi)
    ibinlo = bins[0]
    ibinhi = bins[1]
    # print(f"ibinlo: {ibinlo}, ibinhi: {ibinhi}")

    # Needed to calculate the spectrum with the wavelength offset later.
    delmodwave = modwave + 0.001
    delbins = af.GetBins(delmodwave, binlo, binhi)
    delibinlo = delbins[0] - ibinlo
    delibinhi = delbins[1] - ibinhi

    return ModelSpectrumBinIndices(ibinlo, ibinhi, delibinlo, delibinhi)


def get_indices_of_molecular_species(list_of_gases: list[str]) -> NDArray[np.int_]:
    return af.GetMollist(list_of_gases)


def make_pressure_grid(natm: int, minP: float, maxP: float) -> NDArray[np.float_]:
    return maxP + (minP - maxP) * np.arange(natm) / (natm - 1)


def set_TP_model_index(atmtype: str) -> int:
    if atmtype == "Layers" or atmtype in TP_models:
        return 0

    elif atmtype == "Parametric":
        return 1

    else:
        raise ValueError(f"Unrecognized atmosphere type: {atmtype}")


def select_free_parameters(
    retrieval_parameters: Sequence[RetrievalParameter],
) -> list[RetrievalParameter]:
    return [
        retrieval_parameter
        for retrieval_parameter in retrieval_parameters
        if retrieval_parameter.sigma > 0
    ]


def set_up_ensemble_mode(plparams, ensparams, pnames, bounds, sigma, outfile) -> None:
    esize = 1
    enstable = []
    n = 0
    for i in ensparams:
        epmin = bounds[i, 0]
        epmax = bounds[i, 1]
        estep = sigma[i]
        enum = int((epmax - epmin) / estep) + 1
        esize = esize * enum
        enstable.append([])
        for j in range(0, enum):
            epoint = epmin + j * estep
            enstable[n].append(epoint)
        n = n + 1

    if esize > 1000:
        prompt = input(
            "Warning: this ensemble is more than 1,000 entries and may take significant time. Continue? (y/n): "
        )
        while not (prompt == "y" or prompt == "Y"):
            if prompt == "n" or prompt == "N":
                sys.exit()
            else:
                prompt = input("Error. Please type 'y' or 'n': ")
    elif esize > 10000:
        print(
            "Error: this ensemble is more than 10,000 entries and may require several hours and several GB of memory. Please use a smaller ensemble."
        )
        print(
            "If you wish to continue with this ensemble, comment out this warning in the source code."
        )
        sys.exit()
    print("Ensemble size: {0:d}".format(esize))

    eplist = np.zeros((esize, len(plparams)))
    for i in range(0, esize):
        eplist[i] = plparams
        eindices = []
        n = i
        for j in range(0, len(enstable)):
            k = n % len(enstable[j])
            n = int(n / len(enstable[j]))
            eindices.append(k)

        for j in range(0, len(eindices)):
            eplist[i][ensparams[j]] = enstable[j][eindices[j]]

    foutnamee = "modelspectra" + outfile + "ensemble.dat"
    fout = open(foutnamee, "w")

    for i in range(0, len(ensparams)):
        fout.write("     {0:s}".format(pnames[ensparams[i]]))
        for j in range(0, esize):
            fout.write(" {0:f}".format(eplist[j][ensparams[i]]))
        fout.write("\n")


# NOTE: could set this up so instead of the data_parameters and such, one passes in
# a partialed version of the set_model_wavelengths... function? We would need to figure
# out what should be pre-partialed and what should be provided here.
# Or you just have the arguments be as close to the cclass inputs as possible, which means
# passing model_wavelengths as an argument and have that be separate.


# So in this idea, the current function is just an "adapter" that doesn't create any new
# structures, just convert them into the specific forms that the C++ side needs.
def set_up_PyPlanet(
    model_parameters: ModelParameters,
    model_wavelengths: Sequence[float],
    Teff_calculation_wavelengths: Sequence[float],
    atmtype: str,
    cloudmod: int,
    hazetype: int,
    list_of_gas_species: Sequence[str],
    opacity_parameters: OpacityParameters,
) -> PyPlanet:
    observation_mode_index: int = model_parameters.mode

    number_of_streams: int = model_parameters.streams

    # model_wavelengths: NDArray[np.float_] = (
    #    set_model_wavelengths_from_opacity_tables_and_data(
    #        data_parameters, opacity_parameters, mindL=minDL, maxdL=maxDL
    #    )
    # )

    # Teff_calculation_wavelengths = set_Teff_calculation_wavelengths_from_opacity_tables(
    #    opacity_parameters.opacdir
    # )

    cloud_model_index: int = cloudmod

    hazetype_index: int = hazetype

    gas_species_indices: list[int] = af.GetMollist(list_of_gas_species)

    TP_model_switch: int = set_TP_model_index(atmtype)

    gas_opacity_directory: Pathlike = opacity_parameters.opacdir
    opacity_catalog_name: str = opacity_parameters.hires
    Teff_opacity_catalog_name: str = opacity_parameters.lores

    return CPlanetBlueprint(
        observation_mode_index=observation_mode_index,
        cloud_model_index=cloud_model_index,
        hazetype_index=hazetype_index,
        number_of_streams=number_of_streams,
        TP_model_switch=TP_model_switch,
        model_wavelengths=model_wavelengths,
        Teff_calculation_wavelengths=Teff_calculation_wavelengths,
        gas_species_indices=gas_species_indices,
        gas_opacity_directory=gas_opacity_directory,
        opacity_catalog_name=opacity_catalog_name,
        Teff_opacity_catalog_name=Teff_opacity_catalog_name,
    )
