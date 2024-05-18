import sys
from dataclasses import dataclass
from os.path import abspath
from pathlib import Path
from typing import IO, Any, Optional, TypedDict

import numpy as np
import tomllib
from numpy.typing import NDArray

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/APOLLO"
)
if APOLLO_DIRECTORY not in sys.path:
    sys.path.append(APOLLO_DIRECTORY)

from apollo.convenience_types import Pathlike  # noqa: E402
from apollo.useful_internal_functions import strtobool  # noqa: E402

default_settings_filepath: Path = Path("apollo") / "default_input_settings.toml"
with open(default_settings_filepath, "rb") as defaults_file:
    default_settings: dict[str, Any] = tomllib.load(defaults_file)

HAZELIST: list[str] = default_settings["hazelist"]


def get_number_of_parameters_from_input_file(input_file: Pathlike) -> int:
    try:
        fparams: IO = open(input_file, "r")
    except FileNotFoundError:
        print("\nError: input file not found.\n")
        sys.exit()

    lines1: list[str] = fparams.readlines()
    pllen: int = -1
    for i in range(0, len(lines1)):
        if len(lines1[i].split()) >= 6:
            pllen = pllen + 1

    fparams.close()

    return pllen


@dataclass
class DataParameters:
    datain: Pathlike
    dataconv: int | float
    databin: int | float
    degrade: int | float


@dataclass
class LocationParameters:
    dist: float
    RA: float
    dec: float


@dataclass
class ModelParameters:
    mode: int  # eventually enum?
    streams: int  # eventually enum?


@dataclass
class ModelRestrictionParameters:
    polyfit: bool
    gray: bool
    tgray: float


@dataclass
class OpacityParameters:
    opacdir: Pathlike
    hires: str  # eventually maybe enum?
    lores: str  # eventually maybe enum?


@dataclass
class OutputParameters:
    outmode: str  # eventually enum?
    exptime: float
    outdir: Pathlike
    short: bool
    printfull: bool


@dataclass
class PressureParameters:
    minP: float
    maxP: float
    vres: int
    P_profile: Optional[NDArray[np.float64]]


@dataclass
class SamplerParameters:
    sampler: str  # eventually enum?
    samples_file: Pathlike
    num_samples: int
    checkpoint_file: Pathlike
    prior_type: str  # eventually maybe enum?
    nwalkers: int
    nsteps: int
    minmass: float
    maxmass: float
    parallel: (
        bool  # This is not exclusively a sampler parameter, but fits best here for now.
    )


@dataclass
class TransitParameters:
    tstar: float
    rstar: float
    starspec: Pathlike
    sma: float


class SettingsBlueprint(TypedDict):
    name: str
    data_parameters: DataParameters
    location_parameters: LocationParameters
    model_parameters: ModelParameters
    model_restriction_parameters: ModelRestrictionParameters
    opacity_parameters: OpacityParameters
    output_parameters: OutputParameters
    pressure_parameters: PressureParameters
    sampler_parameters: SamplerParameters
    transit_parameters: TransitParameters


def read_in_settings_from_input_file(
    input_file: Pathlike = "examples/example.resolved.dat", override=True
) -> SettingsBlueprint:
    fparams = open(input_file, "r")

    # Read in settings

    nlines = 0
    while True:
        last_pos = fparams.tell()
        line = fparams.readline().split()
        if len(line) >= 6:  # ends the loop when the parameters start
            if line[0] == "Parameter":
                break
            else:
                fparams.seek(last_pos)
                break

        nlines = nlines + 1
        if nlines > 100:
            break  # prevents getting stuck in an infinite loop

        elif line[0] == "Object":
            if len(line) > 1:
                name = line[1]
        if line[0] == "Mode":
            if len(line) > 1:
                modestr = line[1]
            if modestr == "Resolved":
                mode = 0
            if modestr == "Eclipse":
                mode = 1
            if modestr == "Transit":
                mode = 2
        elif line[0] == "Parallel":
            if len(line) > 1:
                parallel = strtobool(line[1])
        elif line[0] == "Data":
            if len(line) > 1:
                datain = line[1]
            if len(line) > 2:
                if line[2] == "Polyfit":
                    polyfit = True
        elif line[0] == "Sampler":
            if len(line) > 1:
                sampler = line[1]
        elif line[0] == "Samples":
            if len(line) > 1:
                samples_file = line[1]
            if len(line) > 2:
                num_samples = (int)(line[2])
        elif line[0] == "Checkpoint":
            if len(line) > 1:
                checkpoint_file = line[1]
        elif line[0] == "Convolve":
            if len(line) > 1:
                dataconv = (int)(line[1])
        elif line[0] == "Binning":
            if len(line) > 1:
                databin = (int)(line[1])
        elif line[0] == "Degrade":
            if len(line) > 1:
                degrade = (int)(
                    line[1]
                )  # Only works with low-res data; mainly used to speed up execution for testing
        elif line[0] == "Prior":
            if len(line) > 1:
                prior_type = line[1]
        elif line[0] == "N_Walkers":
            if len(line) > 1:
                nwalkers = (int)(line[1])
            if override:
                nwalkers = 2
        elif line[0] == "N_Steps":
            if len(line) > 1:
                nsteps = (int)(line[1])
        elif line[0] == "Star":
            if len(line) > 1:
                tstar = (float)(line[1])
            if len(line) > 2:
                rstar = (float)(line[2])
            if len(line) > 3:
                sma = (float)(line[3])
        elif line[0] == "Star_Spec":
            if len(line) > 1:
                starspec = line[1]
        elif line[0] == "Location":
            if len(line) > 1:
                dist = (float)(line[1])
            if len(line) > 2:
                RA = (float)(line[2])
            if len(line) > 3:
                dec = (float)(line[3])
        elif line[0] == "Mass_Limits":
            if len(line) > 1:
                minmass = (float)(line[1])
            if len(line) > 2:
                maxmass = (float)(line[2])
        elif line[0] == "Tables":
            if len(line) > 1:
                hires = line[1]
            if len(line) > 2:
                lores = line[2]
        elif line[0] == "Pressure":
            if len(line) > 1:
                minP = (float)(line[1]) + 6.0  # Convert from bars to cgs
            if len(line) > 2:
                maxP = (float)(line[2]) + 6.0
            if maxP <= minP:
                maxP = minP + 0.01
            else:
                P_profile = None
        elif line[0] == "Gray":
            if len(line) > 1:
                gray = strtobool(line[1])
            if len(line) > 2:
                tgray = line[2]
        elif line[0] == "Vres":
            if len(line) > 1:
                vres = (int)(line[1])
        elif line[0] == "Streams":
            if len(line) > 1:
                streams = (int)(line[1])
        elif line[0] == "Output_Mode":
            if len(line) > 1:
                outmode = line[1]
            if len(line) > 2:
                exptime = (float)(line[2])
        elif line[0] == "Output":
            if len(line) > 1:
                outdir = line[1]
            if len(line) > 2:
                if line[2] == "Short":
                    short = True
                if line[2] == "Full":
                    printfull = True
            else:
                short = False
                printfull = False
            if len(line) > 3:
                if line[3] == "Short":
                    short = True
                if line[3] == "Full":
                    printfull = True
        elif line[0] == "Opacities":
            if len(line) > 1:
                opacdir = line[1]

    data_parameters: DataParameters = DataParameters(datain, dataconv, databin, degrade)
    location_parameters: LocationParameters = LocationParameters(dist, RA, dec)
    model_parameters: ModelParameters = ModelParameters(mode, streams)
    model_restriction_parameters: ModelRestrictionParameters = (
        ModelRestrictionParameters(polyfit, gray, tgray)
    )
    opacity_parameters: OpacityParameters = OpacityParameters(opacdir, hires, lores)
    output_parameters: OutputParameters = OutputParameters(
        outmode, exptime, outdir, short, printfull
    )
    pressure_parameters: PressureParameters = PressureParameters(
        minP, maxP, vres, P_profile
    )
    sampler_parameters: SamplerParameters = SamplerParameters(
        sampler,
        samples_file,
        num_samples,
        checkpoint_file,
        prior_type,
        nwalkers,
        nsteps,
        minmass,
        maxmass,
        parallel,
    )
    transit_parameters: TransitParameters = TransitParameters(
        tstar, rstar, starspec, sma
    )

    return SettingsBlueprint(
        name=name,
        data_parameters=data_parameters,
        location_parameters=location_parameters,
        model_parameters=model_parameters,
        model_restriction_parameters=model_restriction_parameters,
        opacity_parameters=opacity_parameters,
        output_parameters=output_parameters,
        pressure_parameters=pressure_parameters,
        sampler_parameters=sampler_parameters,
        transit_parameters=transit_parameters,
    )


@dataclass
class ReadinParameters:
    plparams: NDArray[np.float64]  # Parameter list, must be length pllen
    ensparams: list[
        int
    ]  # List of indices of parameters that will be varied in ensemble mode


@dataclass
class ParameterDistributionParameters:
    mu: NDArray[np.float64]  # Gaussian means, must be length pllen
    sigma: NDArray[np.float64]  # Standard errors, must be length pllen
    bounds: NDArray[np.float64]  # Bounds, must have dimensions (pllen, 2)
    guess: NDArray[np.float64]  # Used for initial conditions, must be length pllen


@dataclass
class FundamentalReadinParameters:
    b1: int  # Index of first fundamental ("basic") parameter in list of parameters
    bnum: int  # Number of fundamental parameters
    basic: list[str]  # List of fundamental parameter names
    ilogg: Optional[int]  # Index of gravity parameter
    norad: bool = False


@dataclass
class GasReadinParameters:
    g1: int  # Index of first gas parameter in list of parameters
    gnum: int  # Number of gas parameters
    gases: list[str]  # List of gas parameter names


@dataclass
class TPReadinParameters:
    atmtype: str = "Layers"  # Default layered atmosphere, could be enum?
    a1: int  # Index of first T-P parameter in list of parameters
    anum: int  # Number of T-P parameters
    igamma: Optional[int]  # Index of gamma parameter
    smooth: Optional[bool]  # Whether to smooth, determined by presence of gamma
    verbatim: bool


@dataclass
class CloudReadinParameters:
    cloudmod: int  # Cloud model index, should be enum or eventually strenum?
    hazestr: str  # Name of condensate species, should eventually be strenum?
    hazetype: int  # Index of hazestr in list of hazestr
    c1: int  # Index of first cloud parameter in list of parameters
    cnum: int  # Number of cloud parameters
    clouds: list[str]  # List of cloud parameter names


@dataclass
class CalibrationReadinParameters:
    e1: int  # Index of first calibration ("end") parameter in list of parameters
    enum: int  # Number of calibration parameters
    end: list[str]  # List of calibration parameter names


class ReadinParametersBlueprint(TypedDict):
    readin_parameters: ReadinParameters
    parameter_distribution_parameters: ParameterDistributionParameters
    fundamental_readin_parameters: FundamentalReadinParameters
    gas_readin_parameters: GasReadinParameters
    TP_readin_parameters: TPReadinParameters
    cloud_readin_parameters: CloudReadinParameters
    calibration_readin_parameters: CalibrationReadinParameters


def read_in_model_parameters(
    pllen: int,
    fparams: IO,
    minP: float,
    maxP: float,
    hazelist: list[str] = HAZELIST,
    norad=True,
) -> ReadinParametersBlueprint:
    # Read in model parameters

    print("Reading in parameters.")

    lines = fparams.readlines()

    plparams = np.zeros(pllen)  # Parameter list
    mu = np.zeros(pllen)  # Gaussian means
    sigma = np.zeros(pllen)  # Standard errors
    bounds = np.zeros((pllen, 2))  # Bounds
    guess = np.zeros(pllen)  # Used for initial conditions
    # prior_types = [prior_type] * pllen

    i = 0
    state = -1
    pnames = []
    pvars = []
    nvars = []
    basic = []
    gases = []
    atm = []
    clouds = []
    end = []
    atmtype = "Layers"  # Default layered atmosphere
    smooth = False  # Default no smoothing
    igamma = -1  # Index of gamma if included
    ilogg = -1

    b1 = -1
    bnum = 0
    g1 = -1
    gnum = 0
    a1 = -1
    anum = 0
    c1 = -1
    cnum = 0
    e1 = -1
    enum = 0

    ensparams = []

    for j in range(0, len(lines)):
        if str(lines[j]) == "Basic\n":
            state = 0
            b1 = i
        elif str(lines[j].split()[0]) == "Gases":
            state = 1
            g1 = i
            if len(lines[j].split()) > 1:
                gases.append(lines[j].split()[1])
            else:
                gases.append(
                    "h2he"
                )  # Filler gas is h2+he, may get more reliable results from h2 only
        elif str(lines[j].split()[0]) == "Atm":
            state = 2
            a1 = i
            atmtype = lines[j].split()[1]
            if len(lines[j].split()) > 2:
                if str(lines[j].split()[2]) == "Verbatim":
                    verbatim = True
        elif str(lines[j].split()[0]) == "Haze" or str(lines[j].split()[0]) == "Clouds":
            state = 3
            c1 = i
            cloudmod = int(lines[j].split()[1])
            if len(lines[j].split()) >= 3:
                hazetype = 0
                hazestr = str(lines[j].split()[2])
                if hazestr in hazelist:
                    hazetype = hazelist.index(hazestr)
        elif str(lines[j]) == "End\n":
            state = 4
            e1 = i

        elif len(lines[j].split()) < 6:
            print("Error: missing parameter values.")
            sys.exit()

        else:
            line = lines[j].split()
            pnames.append(line[0])
            if state == 0:
                basic.append(line[0])
                bnum = bnum + 1
            if state == 1:
                gases.append(line[0])
                gnum = gnum + 1
            if state == 2:
                atm.append(line[0])
                anum = anum + 1
            if state == 3:
                clouds.append(line[0])
                cnum = cnum + 1
            if state == 4:
                end.append(line[0])
                enum = enum + 1
            plparams[i] = (float)(line[1])
            guess[i] = (float)(line[1])
            mu[i] = (float)(line[2])
            sigma[i] = (float)(line[3])
            bounds[i, 0] = (float)(line[4])
            bounds[i, 1] = (float)(line[5])
            # if len(line) >= 9:
            #    prior_functions[i] = line[8]

            if sigma[i] > 0.0:
                pvars.append(plparams[i])
                nvars.append(i)

            # if guess[i]==0: guess[i] = 0.1*bounds[i,1]        # Prevents errors from occuring where parameters are zero.

            # Convert pressure units from bars to cgs
            if pnames[i] == "Cloud_Base" or pnames[i] == "P_cl":
                plparams[i] = plparams[i] + 6.0
                guess[i] = guess[i] + 6.0
                mu[i] = mu[i] + 6.0
                bounds[i, 0] = bounds[i, 0] + 6.0
                bounds[i, 1] = bounds[i, 1] + 6.0
                if bounds[i, 0] < minP:
                    bounds[i, 0] = minP
                if bounds[i, 1] > maxP:
                    bounds[i, 1] = maxP

            if line[0] == "gamma":
                smooth = True
                igamma = j
            # ada: We want to impose a normal prior on log g,
            # while keeping uniform priors on everything else.
            elif line[0] == "Log(g)":
                ilogg = i
            if len(line) > 6 and line[6] == "Ensemble":
                ensparams.append(i)
            i = i + 1

    readin_parameters = ReadinParameters(plparams, ensparams)
    parameter_distribution_parameters = ParameterDistributionParameters(
        mu, sigma, bounds, guess
    )
    fundamental_readin_parameters = FundamentalReadinParameters(
        b1, bnum, basic, ilogg, norad
    )
    gas_readin_parameters = GasReadinParameters(g1, gnum, gases)
    TP_readin_parameters = TPReadinParameters(
        atmtype, a1, anum, igamma, smooth, verbatim
    )
    cloud_readin_parameters = CloudReadinParameters(
        cloudmod, hazestr, hazetype, c1, cnum, clouds
    )
    calibration_readin_parameters = CalibrationReadinParameters(e1, enum, end)

    return ReadinParametersBlueprint(
        readin_parameters=readin_parameters,
        parameter_distribution_parameters=parameter_distribution_parameters,
        fundamental_readin_parameters=fundamental_readin_parameters,
        gas_readin_parameters=gas_readin_parameters,
        TP_readin_parameters=TP_readin_parameters,
        cloud_readin_parameters=cloud_readin_parameters,
        calibration_readin_parameters=calibration_readin_parameters,
    )
