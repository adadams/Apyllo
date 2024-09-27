import sys
from dataclasses import dataclass, field
from enum import Enum, StrEnum, auto
from pathlib import Path
from typing import IO, Any, Final, NamedTuple, Optional, TypedDict
from warnings import warn

import numpy as np
import tomllib
from numpy.typing import NDArray

APOLLO_DIRECTORY = Path.cwd().absolute()
if str(APOLLO_DIRECTORY) not in sys.path:
    sys.path.append(str(APOLLO_DIRECTORY))

import apollo.src.ApolloFunctions as af  # noqa: E402
from apollo.cloud_parameter_specs import (  # noqa: E402
    PowerLawOpacityClouds,
    SingleParticleSizeGaussianNumberDensityClouds,
    SingleParticleSizeUniformNumberDensityClouds,
)
from custom_types import Pathlike  # noqa: E402
from useful_internal_functions import strtobool  # noqa: E402

default_settings_filepath: Path = Path("apollo") / "default_input_settings.toml"
with open(default_settings_filepath, "rb") as defaults_file:
    default_settings: dict[str, Any] = tomllib.load(defaults_file)

HAZELIST: list[str] = default_settings["hazelist"]
HAZE_ENUM = Enum("HazeSpecies", default_settings["hazelist"], start=0)

# REARTH_IN_CM = R_earth.to(cm).value
REARTH_IN_CM: Final[float] = 6.371e8

# RJUPITER_IN_REARTH = (R_jup/R_earth).decompose().value
RJUPITER_IN_REARTH: Final[float] = 11.2

PARSEC_IN_REARTH: Final[float] = 4.838e9


class ObservationMode(Enum):
    RESOLVED = 0
    ECLIPSE = 1
    TRANSIT = 2


class RadiusInputType(StrEnum):
    Rad = auto()
    RtoD = auto()
    RtoD2U = auto()
    norad = auto()


@dataclass(slots=True)
class ParameterValue:
    name: str
    value: float


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
    degrade: int | float


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
    # Read in settings

    nlines = 0

    gray: bool = False
    polyfit: bool = False
    tgray: float = 1000.0

    outmode: str = ""
    exptime: float = 0.0
    printfull: bool = False

    vres: int = 71

    samples_file: str = ""

    num_samples: int = 0
    checkpoint_file: str = ""
    nwalkers: int = 0

    starspec: str = ""

    with open(input_file, "r") as fparams:
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

    data_parameters: DataParameters = DataParameters(datain, dataconv, databin)
    location_parameters: LocationParameters = LocationParameters(dist, RA, dec)
    model_parameters: ModelParameters = ModelParameters(mode, streams)
    model_restriction_parameters: ModelRestrictionParameters = (
        ModelRestrictionParameters(polyfit, gray, tgray)
    )
    opacity_parameters: OpacityParameters = OpacityParameters(
        opacdir, hires, lores, degrade
    )
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


def set_radius(size_parameter: ParameterValue, dist: float) -> ParameterValue:
    radius_case: RadiusInputType = RadiusInputType[size_parameter.name]

    # Radius handling
    if radius_case == RadiusInputType.Rad:
        radius = size_parameter.value * REARTH_IN_CM
    elif radius_case == RadiusInputType.RtoD:
        radius = 10**size_parameter.value * dist * PARSEC_IN_REARTH * REARTH_IN_CM
    elif radius_case == RadiusInputType.RtoD2U:
        radius = np.sqrt(size_parameter.value) * REARTH_IN_CM
    else:
        # norad True
        radius = RJUPITER_IN_REARTH * REARTH_IN_CM

    return ParameterValue(name="radius", value=radius)


def set_log_gravity(log_gravity: float = None, default_logg: float = 4.1) -> float:
    return log_gravity if log_gravity else default_logg


@dataclass
class FundamentalReadinParameters:
    b1: int  # Index of first fundamental ("basic") parameter in list of parameters
    bnum: int  # Number of fundamental parameters
    b2: int = field(init=False)
    basic: list[str]  # List of fundamental parameter names
    radius_case: RadiusInputType = field(init=False)
    radius_index: float = field(init=False)
    gravity_index: float  # ilogg

    def __post_init__(self):
        self.b2 = self.b1 + self.bnum

        radius_case_options: list[str] = [
            radius_case_enum.name for radius_case_enum in RadiusInputType
        ]
        for i, fundamental_parameter_name in enumerate(self.basic):
            if fundamental_parameter_name in radius_case_options:
                self.radius_case = RadiusInputType[fundamental_parameter_name]
                self.radius_index = self.b1 + i
                break  # there should only be one radius parameter
            else:
                self.radius_case = RadiusInputType.norad

    def bodge_radius_parameter(
        self,
        parameter_values: NDArray[np.float64],
        distance_to_system_in_parsecs: float,
    ) -> ParameterValue:
        size_value: float = parameter_values[self.radius_index]

        size_parameter: ParameterValue = ParameterValue(
            name=self.radius_case.name, value=size_value
        )

        return set_radius(size_parameter, distance_to_system_in_parsecs)

    def bodge_gravity_parameter(
        self, parameter_values: NDArray[np.float64]
    ) -> ParameterValue:
        gravity_value: float = parameter_values[self.gravity_index]

        return ParameterValue(name="Log(g)", value=set_log_gravity(gravity_value))


@dataclass
class MolecularParameters:
    species: list[str]
    weighted_molecular_weights: NDArray[np.float64]
    weighted_scattering_cross_sections: NDArray[np.float64]
    # mollist: NDArray[np.float64]


# NOTE: shuttle this to the GasReadin object.


@dataclass
class GasReadinParameters:
    g1: int  # Index of first gas parameter in list of parameters
    gnum: int  # Number of gas parameters
    g2: int = field(init=False)
    gases: list[str]  # List of gas parameter names
    ngas: int = field(init=False)

    def __post_init__(self):
        self.g2 = self.g1 + self.gnum
        self.ngas = self.gnum + 1

    def get_gas_nonfiller_log_abundances(
        self, parameter_values: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return parameter_values[self.g1 : self.g2]

    def bodge_gas_parameters(
        self, parameter_values: NDArray[np.float64]
    ) -> list[ParameterValue]:
        return [
            ParameterValue(gas_parameter_name, gas_parameter_value)
            for gas_parameter_name, gas_parameter_value in zip(
                self.gases, self.get_gas_nonfiller_log_abundances(parameter_values)
            )
        ]

    def get_molecular_weights_and_scattering_opacities(
        self, parameter_values: NDArray[np.float64]
    ) -> MolecularParameters:
        gas_parameters = self.get_gas_nonfiller_log_abundances(parameter_values)

        mmw, rxsec = af.GetScaOpac(self.gases, gas_parameters)
        return MolecularParameters(self.gases, mmw, rxsec)


@dataclass
class TPReadinParameters:
    atmtype: str  # = "Layers"  # Default layered atmosphere, could be enum?
    a1: int  # Index of first T-P parameter in list of parameters
    anum: int  # Number of T-P parameters
    a2: int = field(init=False)
    igamma: Optional[int]  # Index of gamma parameter
    smooth: Optional[bool]  # Whether to smooth, determined by presence of gamma
    verbatim: bool

    def __post_init__(self):
        self.a2 = self.a1 + self.anum

        if self.smooth:
            self.a2 = self.a2 - 1
            self.anum = self.anum - 1

    def bodge_TP_parameters(
        self, parameter_values: NDArray[np.float64]
    ) -> list[ParameterValue]:
        TP_parameter_values = parameter_values[self.a1 : self.a2]

        return [
            ParameterValue(TP_parameter_name, TP_parameter_value)
            for TP_parameter_name, TP_parameter_value in zip(
                [""] * len(TP_parameter_values), TP_parameter_values
            )
        ]


class CloudModel(Enum):
    no_clouds = 0
    opaque_deck = 1
    single_particle_size_uniform_number_density = 2
    single_particle_size_gaussian_number_density = 3
    power_law_opacity = 4


def get_cloud_deck_log_pressure(
    cloud_parameters: dict[str, ParameterValue],
    minimum_log_pressure_in_CGS: float = 0.0,
    maximum_log_pressure_in_CGS: float = 8.5,
) -> float:
    if "Cloud_Base" in cloud_parameters:
        cloud_deck_log_pressure: float = cloud_parameters["Cloud_Base"].value

    elif "P_cl" in cloud_parameters:
        cloud_deck_log_pressure: float = cloud_parameters["P_cl"].value

    else:
        warn(
            "Cloud deck pressure not found under either of expected names: 'Cloud_Base' or 'P_cl'."
            + f"Setting to {10**(maximum_log_pressure_in_CGS-6)} bar."
        )

        cloud_deck_log_pressure: float = maximum_log_pressure_in_CGS

    return (
        cloud_deck_log_pressure
        if cloud_deck_log_pressure >= minimum_log_pressure_in_CGS
        else minimum_log_pressure_in_CGS + 0.01
    )


def set_cloud_parameters(
    cloud_model_case: CloudModel, cloud_parameters: dict[str, ParameterValue]
) -> NamedTuple:
    cloud_pressure_parameter_names: tuple[str] = (
        "Haze_minP",
        "Haze_meanP",
        "Haze_maxP",
    )

    cloud_parameter_dict: dict[str, float] = {
        parameter_name: parameter.value + 6
        if parameter_name in cloud_pressure_parameter_names
        else parameter.value
        for parameter_name, parameter in cloud_parameters.items()
    }

    if cloud_model_case == CloudModel.single_particle_size_uniform_number_density:
        cloud_parameter_dict["Haze_maxP"] = cloud_parameter_dict[
            "Haze_minP"
        ] + cloud_parameter_dict.pop("Haze_thick")

        return SingleParticleSizeUniformNumberDensityClouds(
            cloud_number_density=cloud_parameter_dict["Haze_abund"],
            cloud_particle_size=cloud_parameter_dict["Haze_size"],
            cloud_minimum_log_pressure=cloud_parameter_dict["Haze_minP"],
            cloud_maximum_log_pressure=cloud_parameter_dict["Haze_maxP"],
        )

    elif cloud_model_case == CloudModel.single_particle_size_gaussian_number_density:
        return SingleParticleSizeGaussianNumberDensityClouds(
            cloud_peak_number_density=cloud_parameter_dict["Haze_Pabund"],
            cloud_particle_size=cloud_parameter_dict["Haze_size"],
            cloud_peak_log_pressure=cloud_parameter_dict["Haze_meanP"],
            cloud_log_pressure_scale=cloud_parameter_dict["Haze_scale"],
        )

    elif cloud_model_case == CloudModel.power_law_opacity:
        return PowerLawOpacityClouds(
            opacity_power_law_exponent=cloud_parameter_dict["Haze_alpha"],
            cloud_minimum_log_pressure=cloud_parameter_dict["Haze_minP"],
            cloud_log_pressure_thickness=cloud_parameter_dict["Haze_thick"],
            cloud_reference_log_optical_depth=cloud_parameter_dict["Haze_tau"],
            cloud_single_scattering_albedo=cloud_parameter_dict["Haze_w0"],
        )


@dataclass
class CloudReadinParameters:
    cloudmod: int  # Cloud model index, should be enum or eventually strenum?
    hazestr: str  # Name of condensate species, should eventually be strenum?
    hazetype: int  # Index of hazestr in list of hazestr
    c1: int  # Index of first cloud parameter in list of parameters
    cnum: int  # Number of cloud parameters
    c2: int = field(init=False)
    clouds: list[str]  # List of cloud parameter names
    ilen: int = field(init=False)  # ilen probably shouldn't be here but...

    def __post_init__(self):
        self.c2 = self.c1 + self.cnum
        self.ilen = 10 + self.cnum

    def make_cloud_parameter_tuple(
        self, parameter_values: NDArray[np.float64]
    ) -> NamedTuple:
        cloud_model_case: CloudModel = CloudModel(self.cloudmod)

        return set_cloud_parameters(
            cloud_model_case, self.bodge_cloud_parameters(parameter_values)
        )

    def bodge_cloud_parameters(
        self, parameter_values: NDArray[np.float64]
    ) -> dict[str, ParameterValue]:
        cloud_parameter_values = parameter_values[self.c1 : self.c2]

        return {
            cloud_parameter_name: ParameterValue(
                cloud_parameter_name, cloud_parameter_value
            )
            for cloud_parameter_name, cloud_parameter_value in zip(
                self.clouds, cloud_parameter_values
            )
        }

    def get_cloud_filling_fraction(
        self, parameter_values: NDArray[np.float64]
    ) -> float:
        cloud_parameters: dict[str, ParameterValue] = self.bodge_cloud_parameters(
            parameter_values
        )

        if "Cloud_Fraction" in cloud_parameters:
            cloud_filling_fraction: float = cloud_parameters["Cloud_Fraction"].value
        else:
            cloud_filling_fraction: float = 1.0

        return cloud_filling_fraction


# NOTE: How can this be combined with the ParameterValue class?
class FluxScaler(NamedTuple):
    band_lower_wavelength_boundary: float
    band_upper_wavelength_boundary: float
    scale_factor: NDArray[np.float64]


@dataclass
class CalibrationReadinParameters:
    e1: int  # Index of first calibration ("end") parameter in list of parameters
    enum: int  # Number of calibration parameters
    end: list[str]  # List of calibration parameter names

    def __post_init__(self):
        self.e2 = self.e1 + self.enum

    def bodge_calibration_parameters(
        self, parameter_values: NDArray[np.float64]
    ) -> dict[str, ParameterValue]:
        calibration_parameter_values = parameter_values[self.e1 : self.e2]

        return {
            calibration_parameter_name: ParameterValue(
                calibration_parameter_name, calibration_parameter_value
            )
            for calibration_parameter_name, calibration_parameter_value in zip(
                self.end, calibration_parameter_values
            )
        }

    def get_flux_scalers(
        self,
        parameter_values: NDArray[np.float64],
        name_to_wavelength_range_mapper: dict[str, tuple[float, float]],
    ) -> list[FluxScaler]:
        calibration_parameters: dict[str, ParameterValue] = (
            self.bodge_calibration_parameters(parameter_values)
        )

        scaler_parameters: dict[str, ParameterValue] = {
            scaler_parameter_name: scaler_parameter_value
            for scaler_parameter_name, scaler_parameter_value in calibration_parameters.items()
            if "scale" in scaler_parameter_name
        }

        return [
            FluxScaler(
                *name_to_wavelength_range_mapper[scaler_parameter_name],
                scaler_parameter_value.value,
            )
            for scaler_parameter_name, scaler_parameter_value in scaler_parameters.items()
        ]


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
    input_file: Pathlike,
    minP: float,
    maxP: float,
    hazelist: list[str] = HAZELIST,
    norad=True,
) -> ReadinParametersBlueprint:
    # Read in model parameters
    print("Reading in parameters.")

    with open(input_file, "r") as fparams:
        while True:
            line = fparams.readline()
            if "Parameter" in line:
                break

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

    verbatim: bool = False

    cloudmod: int = 0
    hazestr: str = ""
    hazetype: int = 0

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

    """
    if norad:
        radius_case: RadiusInputType = RadiusInputType.norad
    else:
        radius_case_options: list[str] = [
            radius_case.name for radius_case in RadiusInputType
        ]
        for fundamental_parameter_name in basic:
            if fundamental_parameter_name in radius_case_options:
                radius_case = RadiusInputType[fundamental_parameter_name]
                break
        else:
            raise ValueError("No matching radius case found.")
    """

    fundamental_readin_parameters = FundamentalReadinParameters(
        b1=b1, bnum=bnum, basic=basic, gravity_index=ilogg
    )
    gas_readin_parameters = GasReadinParameters(g1=g1, gnum=gnum, gases=gases)
    TP_readin_parameters = TPReadinParameters(
        atmtype=atmtype,
        a1=a1,
        anum=anum,
        igamma=igamma,
        smooth=smooth,
        verbatim=verbatim,
    )
    cloud_readin_parameters = CloudReadinParameters(
        cloudmod=cloudmod,
        hazestr=hazestr,
        hazetype=hazetype,
        c1=c1,
        cnum=cnum,
        clouds=clouds,
    )
    calibration_readin_parameters = CalibrationReadinParameters(
        e1=e1, enum=enum, end=end
    )

    return ReadinParametersBlueprint(
        readin_parameters=readin_parameters,
        parameter_distribution_parameters=parameter_distribution_parameters,
        fundamental_readin_parameters=fundamental_readin_parameters,
        gas_readin_parameters=gas_readin_parameters,
        TP_readin_parameters=TP_readin_parameters,
        cloud_readin_parameters=cloud_readin_parameters,
        calibration_readin_parameters=calibration_readin_parameters,
    )


class APOLLOFileReadinBlueprint(TypedDict):
    number_of_parameters: int
    settings: SettingsBlueprint
    parameters: ReadinParametersBlueprint


def read_inputs_from_file(input_file: Pathlike) -> APOLLOFileReadinBlueprint:
    number_of_parameters: int = get_number_of_parameters_from_input_file(input_file)

    settings: SettingsBlueprint = read_in_settings_from_input_file(input_file)

    parameters: ReadinParametersBlueprint = read_in_model_parameters(
        pllen=number_of_parameters, input_file=input_file, minP=-4, maxP=2.5
    )

    return APOLLOFileReadinBlueprint(
        number_of_parameters=number_of_parameters,
        settings=settings,
        parameters=parameters,
    )
