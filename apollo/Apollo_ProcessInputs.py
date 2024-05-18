import sys
from dataclasses import dataclass
from os.path import abspath

import numpy as np
from numpy.typing import NDArray

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/APOLLO"
)
if APOLLO_DIRECTORY not in sys.path:
    sys.path.append(APOLLO_DIRECTORY)

import apollo.src.ApolloFunctions as af  # noqa: E402
from apollo.Apollo_ReadInputsfromFile import DataParameters  # noqa: E402
from apollo.convenience_types import Pathlike  # noqa: E402


@dataclass
class DataContainer:
    # NOTE: this is a clone of an existing container for APOLLO-style data.
    wavelo: NDArray[np.float64]
    wavehi: NDArray[np.float64]
    wavemid: NDArray[np.float64]
    flux: NDArray[np.float64]
    errlo: NDArray[np.float64]
    errhi: NDArray[np.float64]


def read_in_observations(datain: Pathlike) -> DataContainer:
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

    return DataContainer(wavelo, wavehi, wavemid, flux, errlo, errhi)


@dataclass
class BandParameters:
    bandindex: NDArray[np.float64]
    bandlo: NDArray[np.float64]
    bandhi: NDArray[np.float64]
    bandflux: NDArray[np.float64]
    banderr: NDArray[np.float64]


def process_observations(
    observations: DataContainer, data_parameters: DataParameters
) -> None:
    # Separate out individual bands
    bandindex, bandlo, bandhi, bandflux, banderr = af.FindBands(
        observations.wavelo, observations.wavehi, observations.flux, observations.errhi
    )
    # nband = len(bandhi)

    # Convolve the observations to account for effective resolving power or fit at lower resolving power
    convflux, converr = af.ConvBands(bandflux, banderr, data_parameters.dataconv)

    # Bin the observations to fit a lower sampling resolution
    binlo, binhi, binflux, binerr = af.BinBands(
        bandlo, bandhi, convflux, converr, data_parameters.databin
    )
    binlen = len(binflux)
    binmid = np.zeros(len(binlo))
    for i in range(0, len(binlo)):
        binmid[i] = (binlo[i] + binhi[i]) / 2.0

    totalflux = 0
    for i in range(0, binlen):
        totalflux = totalflux + binflux[i] * (binhi[i] - binlo[i]) * 1.0e-4

    return None


def calculate_wavelength_calibration_limits(
    bounds: NDArray[np.float64], end: list[str], e1: int
) -> list[float]:
    if "deltaL" in end:
        pos = end.index("deltaL")
        minDL = bounds[e1 + pos, 0] * 0.001
        maxDL = bounds[e1 + pos, 1] * 0.001

        return minDL, maxDL

    else:
        return 0, 0


def select_default_opacity_tables(wavei: float, wavef: float) -> str:
    if wavei < 5.0 and wavef < 5.0:
        if wavei < 0.6:
            wavei = 0.6
        hires = "nir"
    elif wavei < 5.0 and wavef > 5.0:
        if wavei < 0.6:
            wavei = 0.6
        if wavef > 30.0:
            wavef = 30.0
        hires = "wide"
    elif wavei > 5.0 and wavef > 5.0:
        if wavef > 30.0:
            wavef = 30.0
        hires = "mir"
    else:
        raise ValueError(
            "None of the default opacity tables match the wavelengths of the data you provided."
        )

    return hires  # also need to return wavei and wavef, but they're not used anywhere after this???


def set_model_wavelength_range(
    opacdir: str,
    hires: str,
    lores: str,
    degrade: int | float,
) -> None:
    # Compute hires spectrum wavelengths
    opacfile = opacdir + "/gases/h2o." + hires + ".dat"
    fopac = open(opacfile, "r")
    opacshape = fopac.readline().split()
    fopac.close()
    nwave = (int)(opacshape[6])
    lmin = (float)(opacshape[7])
    resolv = (float)(opacshape[9])

    opaclen = (int)(
        np.floor(nwave / degrade)
    )  # THIS SEEMED TO NEED A +1 IN CERTAIN CASES.
    opacwave = np.zeros(opaclen)
    for i in range(0, opaclen):
        opacwave[i] = lmin * np.exp(i * degrade / resolv)
    """
    if wavehi[0]>opacwave[0] or wavelo[-1]<opacwave[-1]:
        trim = [i for i in range(0,len(wavehi)) if (wavehi[i]<opacwave[0] and wavelo[i]>opacwave[-1])]
        wavehi = wavehi[trim]
        wavelo = wavelo[trim]
        flux = flux[trim]
        errlo = errlo[trim]
        errhi = errhi[trim]
        obslength = len(trim)
    """
    # Compute lores spectrum wavelengths
    opacfile = opacdir + "/gases/h2o." + lores + ".dat"
    fopac = open(opacfile, "r")
    opacshape = fopac.readline().split()
    fopac.close()
    nwavelo = (int)(opacshape[6])
    lminlo = (float)(opacshape[7])
    resolvlo = (float)(opacshape[9])

    modwavelo = np.zeros(nwavelo)
    for i in range(0, nwavelo):
        modwavelo[i] = lminlo * np.exp(i / resolvlo)

    # Set up wavelength ranges
    """
    imin = np.where(opacwave<np.max(wavehi))[0]-1
    imax = np.where(opacwave<np.min(wavelo))[0]+2
    elif istart[0]<0: istart[0]=0
    if len(iend)==0: iend = [len(opacwave)-1]
    elif iend[-1]>=len(opacwave): iend[-1] = len(opacwave)-1

    # Truncated and band-limited spectrum that does not extend beyond the range of the observations
    modwave = np.array(opacwave)[(int)(imin[0]):(int)(imax[0])]
    lenmod = len(modwave)
    """

    return None


def handle_bands(
    band_parameters: BandParameters,
    opacwave,
    binflux,
    binerr,
    minDL,
    maxDL,
) -> None:
    # Handle bands and optional polynomial fitting
    bindex, modindex, modwave = af.SliceModel(
        band_parameters.bandlo, band_parameters.bandhi, opacwave, minDL, maxDL
    )

    # print(f"modwave is {modwave}")
    # if np.any(np.isnan(modwave)):
    #    print("There are at least some nans in the modwave.")

    polyindex = -1
    for i in range(1, len(bindex)):
        if bindex[i][0] < bindex[i - 1][0]:
            polyindex = i
    if polyindex == -1:
        polyfit = False

    if polyfit:
        normlo = band_parameters.bandlo[0]
        normhi = band_parameters.bandhi[0]
        normflux = band_parameters.bandflux[0]
        normerr = band_parameters.banderr[0]
        for i in range(1, len(band_parameters.bandlo)):
            normlo = np.r_[normlo, band_parameters.bandlo[i]]
            normhi = np.r_[normhi, band_parameters.bandhi[i]]
            normflux = np.r_[normflux, band_parameters.bandflux[i]]
            normerr = np.r_[normerr, band_parameters.banderr[i]]
        normmid = (normlo + normhi) / 2.0

        slennorm = []
        elennorm = []
        for i in range(polyindex, len(band_parameters.bandindex)):
            slennorm.append(band_parameters.bandindex[i][0])
            elennorm.append(band_parameters.bandindex[i][1])

        masternorm = af.NormSpec(normmid, normflux, slennorm, elennorm)
        fluxspecial = np.concatenate(
            (normerr[0 : slennorm[0]], normflux[slennorm[0] :]), axis=0
        )
        mastererr = af.NormSpec(normmid, fluxspecial, slennorm, elennorm)

    else:
        masternorm = binflux
        mastererr = binerr

    return masternorm, mastererr, ...
