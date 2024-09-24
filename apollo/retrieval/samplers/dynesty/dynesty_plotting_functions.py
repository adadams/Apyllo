from dataclasses import asdict, dataclass
from pathlib import Path
from typing import (
    Any,
    Final,
    Iterable,
    Optional,
    Protocol,
    Sequence,
    TypedDict,
)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import CSS4_COLORS as cnames
from matplotlib.colors import Colormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.typing import ColorType
from numpy.typing import NDArray
from pandas.compat import pickle_compat
from yaml import safe_load

from apollo.generate_cornerplot import generate_cornerplot
from apollo.make_forward_model_from_file import evaluate_model_spectrum
from apollo.retrieval.plotting.contributions_plot import (
    draw_cloud_contribution,
    draw_gas_contribution,
)
from apollo.retrieval.plotting.multi_figure import (
    MultiFigure,
    MultiFigureBlueprint,
    MultiFigureBlueprintButt,
    create_Arthurs_multi_figure,
    setup_multi_figure,
)
from apollo.retrieval.plotting.residuals_plot import (
    calculate_residuals,
    generate_residual_plot_by_band,
)
from apollo.retrieval.plotting.spectral_plot import (
    generate_spectrum_plot_by_band,
    plot_alkali_lines_on_spectrum,
)
from apollo.retrieval.results.manipulate_results_datasets import (
    calculate_MLE,
    calculate_percentile,
)
from user_directories import USER_DIRECTORY

"""
from apollo.retrieval.samplers.dynesty.dynesty_interface_with_apollo import (
    create_MLE_output_dictionary,
    prep_inputs_for_model,
)
"""
from apollo.retrieval.results.IO import unpack_results_filepaths
from apollo.visualization_functions import (
    convert_to_greyscale,
    create_linear_colormap,
    create_monochromatic_linear_colormap,
)
from custom_types import Pathlike
from dataset.accessors import change_units
from dataset.IO import load_and_prep_dataset
from user.forward_models.inputs.parse_APOLLO_inputs import write_parsed_input_to_output
from user.plots.plots_config import DEFAULT_PLOT_FILETYPES

CMAP_KWARGS: Final = {
    "lightness_minimum": 0.15,
    "lightness_maximum": 0.85,
    "saturation_minimum": 0.2,
    "saturation_maximum": 0.8,
}

APOLLO_USER_PLOTS_DIRECTORY: Final = USER_DIRECTORY / "plots"

plt.style.use(APOLLO_USER_PLOTS_DIRECTORY / "arthur.mplstyle")

CMAP_H2O: Final[Colormap] = create_linear_colormap(
    ["#226666", "#2E4172"], **CMAP_KWARGS
)
CMAP_CO: Final[Colormap] = create_linear_colormap(["#882D60", "#AA3939"], **CMAP_KWARGS)
CMAP_CO2: Final[Colormap] = create_linear_colormap(
    ["#96A537", "#669933"], **CMAP_KWARGS
)
CMAP_CH4: Final[Colormap] = create_linear_colormap(
    ["#96A537", "#669933"], **CMAP_KWARGS
)

CMAP_CLOUDY: Final[Colormap] = create_linear_colormap(
    [cnames["lightcoral"], cnames["lightcoral"]], **CMAP_KWARGS
)
CMAP_CLEAR: Final[Colormap] = create_linear_colormap(
    [cnames["cornflowerblue"], cnames["cornflowerblue"]], **CMAP_KWARGS
)

CMAP_CLOUD: Final[Colormap] = plt.get_cmap("Greys")

PLOTTED_COMPONENTS: Final[list[str]] = ["h2o", "co", "co2", "ch4"]
PLOTTED_TITLES: Final[list[str]] = ["H$_2$O", "CO", "CO$_2$", "CH$_4$"]
CMAPS: Final[list[Colormap]] = [CMAP_H2O, CMAP_CO, CMAP_CO2, CMAP_CH4]

PADDING: float = 0.025

DEFAULT_SAVE_PLOT_KWARGS = dict(
    dpi=300,
    transparent=True,
    bbox_inches="tight",
)


def setup_contribution_plots(
    contributions: dict[str, NDArray[np.float64]], data_filepath: Pathlike
):
    FIDUCIAL_SPECIES = "h2o"
    MINIMUM_BAND_BREAK_IN_MICRONS = 0.05
    COMPOSITE_COMPONENTS = ["gas", "cloud", "total"]

    contributions_max = np.log10(
        np.nanmax(
            [
                np.nanmax(contribution)
                for (species, contribution) in contributions.items()
                if species not in COMPOSITE_COMPONENTS
            ]
        )
    )

    wavelengths_low, wavelengths_high, data, data_error = (
        np.genfromtxt(data_filepath).T
    )[:4]
    wavelengths = (wavelengths_low + wavelengths_high) / 2

    band_breaks = np.r_[
        0,
        np.nonzero(
            (
                contributions[FIDUCIAL_SPECIES].index.to_numpy()[1:]
                - contributions[FIDUCIAL_SPECIES].index.to_numpy()[:-1]
            )
            > MINIMUM_BAND_BREAK_IN_MICRONS
        )[0]
        + 1,
        len(contributions[FIDUCIAL_SPECIES].index),
    ]
    # number_of_bands = len(band_breaks) - 1

    band_lower_wavelength_limit = np.asarray(
        [
            contributions[FIDUCIAL_SPECIES].index[band_break]
            for band_break in band_breaks[:-1]
        ]
    )
    band_upper_wavelength_limit = np.asarray(
        [
            contributions[FIDUCIAL_SPECIES].index[band_break - 1]
            for band_break in band_breaks[1:]
        ]
    )
    wavelength_ranges = band_upper_wavelength_limit - band_lower_wavelength_limit

    return dict(
        contributions_max=contributions_max,
        band_breaks=band_breaks,
        wavelengths=wavelengths,
        data=data,
        errors=data_error,
        band_lower_wavelength_limit=band_lower_wavelength_limit,
        band_upper_wavelength_limit=band_upper_wavelength_limit,
        wavelength_ranges=wavelength_ranges,
    )


def make_spectrum_and_residual_axes(
    figure: plt.Figure, gridspec: GridSpec, band_index: int
) -> list[plt.Axes]:
    spectrum_ax = figure.add_subplot(gridspec[0, band_index])
    residual_ax = figure.add_subplot(gridspec[1, band_index], sharex=spectrum_ax)

    return spectrum_ax, residual_ax


def calculate_chi_squared(
    residuals: NDArray[np.float64], number_of_parameters: int
) -> float:
    reduced_chi_squared = np.sum(residuals**2) / (
        np.shape(residuals)[0] - number_of_parameters
    )

    return reduced_chi_squared


# You have structure by plot type (row), band (column), and then overplotting multiple
# runs/cases.


@dataclass
class SpectralPlotElements:
    datas: Sequence[float]
    models: Sequence[float]
    errors: Sequence[float]


def make_contribution_figure_per_species(
    contribution_ax: plt.Axes,
    contributions: dict[str, pd.DataFrame],
    band_breaks: Sequence[Sequence[float]],
    comp: str,
    contributions_max: float,
    outline_color: str,
    cont_cmap: str,
    i: int,
    j: int,
) -> list[plt.Figure, plt.Axes]:
    cf = contributions[comp].to_numpy()
    x, y = np.meshgrid(
        contributions[comp].index,
        contributions[comp].columns,
    )
    contribution_contour_resolution_reduction_factor = 8

    contribution_ax.contourf(
        x[
            :,
            band_breaks[i] : band_breaks[
                i + 1
            ] : contribution_contour_resolution_reduction_factor,
        ],
        y[
            :,
            band_breaks[i] : band_breaks[
                i + 1
            ] : contribution_contour_resolution_reduction_factor,
        ],
        np.log10(cf).T[
            :,
            band_breaks[i] : band_breaks[
                i + 1
            ] : contribution_contour_resolution_reduction_factor,
        ],
        cmap=cont_cmap,
        levels=contributions_max - np.array([4, 2, 0]),
        alpha=0.66,
        zorder=0,
    )
    contribution_ax.contour(
        x[
            :,
            band_breaks[i] : band_breaks[
                i + 1
            ] : contribution_contour_resolution_reduction_factor,
        ],
        y[
            :,
            band_breaks[i] : band_breaks[
                i + 1
            ] : contribution_contour_resolution_reduction_factor,
        ],
        np.log10(cf).T[
            :,
            band_breaks[i] : band_breaks[
                i + 1
            ] : contribution_contour_resolution_reduction_factor,
        ],
        colors=outline_color,
        levels=contributions_max - np.array([2]),
        # linewidths=3,
        alpha=1,
        zorder=1,
    )
    if j == 0:
        contribution_ax.invert_yaxis()
    if "cloud" in contributions:
        cloud_cf = contributions["cloud"]
        x, y = np.meshgrid(cloud_cf.index, cloud_cf.columns)
        contribution_ax.contourf(
            x[
                :,
                band_breaks[i] : band_breaks[
                    i + 1
                ] : contribution_contour_resolution_reduction_factor,
            ],
            y[
                :,
                band_breaks[i] : band_breaks[
                    i + 1
                ] : contribution_contour_resolution_reduction_factor,
            ],
            cloud_cf.to_numpy().T[
                :,
                band_breaks[i] : band_breaks[
                    i + 1
                ] : contribution_contour_resolution_reduction_factor,
            ],
            colors="k",
            # cmap=cmap_cloud,
            alpha=0.75,
            # levels=np.logspace(-1, 2, num=20),
            levels=[0.1, 0.75],
            zorder=2,
        )
        if i == 0:
            contribution_ax.contour(
                x[
                    :,
                    band_breaks[i] : band_breaks[
                        i + 1
                    ] : contribution_contour_resolution_reduction_factor,
                ],
                y[
                    :,
                    band_breaks[i] : band_breaks[
                        i + 1
                    ] : contribution_contour_resolution_reduction_factor,
                ],
                cloud_cf.to_numpy().T[
                    :,
                    band_breaks[i] : band_breaks[
                        i + 1
                    ] : contribution_contour_resolution_reduction_factor,
                ],
                colors="#DDDDDD",
                linestyles="solid",
                # linewidth=3,
                alpha=1,
                levels=[0.1],
                zorder=3,
            )


def plot_multi_figure_iteration(
    figure: plt.Figure,
    gridspec: GridSpec,
    contributions: dict[str, NDArray[np.float64]],
    list_of_band_boundaries: Sequence[Sequence[float]],
    band_breaks: Sequence[Sequence[float]],
    contributions_max: float,
    wavelengths: Sequence[float],
    datas: Sequence[float],
    models: Sequence[float],
    errors: Sequence[float],
    model_title: str,
    number_of_parameters: int,
    plot_color: str,
    iteration_index: int = 0,
    multi_figure_axes: Sequence[MultiFigure] | None = None,
) -> list[plt.Figure, tuple[plt.Axes]]:
    j = iteration_index

    if not multi_figure_axes:
        spectrum_axes = []
        residual_axes = []
        contribution_columns = []

    residuals = calculate_residuals(datas, models, errors)

    for i, band_boundaries in enumerate(list_of_band_boundaries):
        if j == 0:
            spectrum_ax, residual_ax = make_spectrum_and_residual_axes(
                figure, gridspec, band_index=i
            )
            spectrum_axes.append(spectrum_ax)
            residual_axes.append(residual_ax)
        else:
            spectrum_ax = spectrum_axes[i]
            residual_ax = residual_axes[i]

        band_condition = (wavelengths > band_boundaries[0]) & (
            wavelengths < band_boundaries[1]
        )
        wavelengths_in_band = wavelengths[band_condition]
        data = datas[band_condition]
        error = errors[band_condition]
        model = models[band_condition]
        residual = residuals[band_condition]

        spectrum_ax = generate_spectrum_plot_by_band(
            spectrum_ax,
            wavelengths_in_band,
            data,
            model,
            error,
            model_title,
            plot_color,
        )

        residual_ax = generate_residual_plot_by_band(
            residual_ax, wavelengths_in_band, residual, plot_color
        )

        if i == 0:
            spectrum_ax.set_ylabel("Flux (erg s$^{-1}$ cm$^{-3}$)", fontsize=36)
            residual_ax.set_ylabel(r"Residual/$\sigma$", fontsize=26)
        if (j == 1) and (i == len(list_of_band_boundaries) - 1):
            legend_handles, legend_labels = spectrum_ax.get_legend_handles_labels()
            legend_handles.append(Patch(facecolor="k", edgecolor="#DDDDDD"))
            legend_labels.append(r"Cloud $\tau > 0.1$")
            spectrum_ax.legend(
                handles=legend_handles, labels=legend_labels, fontsize=22, frameon=False
            )  # , loc="upper center")

        reduced_chi_squared = calculate_chi_squared(residual, number_of_parameters)
        print(
            f"Reduced chi squared for the band with boundaries {band_boundaries} is {reduced_chi_squared}."
        )

        spectrum_ax.tick_params(axis="x", labelsize=26)
        spectrum_ax.tick_params(axis="y", labelsize=26)
        spectrum_ax.yaxis.get_offset_text().set_size(26)

        residual_ax.tick_params(axis="x", labelsize=26)
        residual_ax.tick_params(axis="y", labelsize=26)

        # CONTRIBUTION PLOTS

        contribution_axes = []
        for n, (comp, cmap, title) in enumerate(
            zip(PLOTTED_COMPONENTS, CMAPS, PLOTTED_TITLES)
        ):
            if j == 0:
                contribution_ax = figure.add_subplot(
                    gridspec[n + 2, i], sharex=spectrum_axes[i]
                )
                contribution_axes.append(contribution_ax)
                if n == len(PLOTTED_COMPONENTS) - 1:
                    contribution_columns.append(contribution_axes)
                if i == 0:
                    contribution_ax.text(
                        0.025,
                        0.05,
                        title + " contribution",
                        horizontalalignment="left",
                        verticalalignment="bottom",
                        transform=contribution_ax.transAxes,
                        fontsize=32,
                    )
            else:
                contribution_ax = contribution_columns[i][n]

            if j == 0:
                cont_cmap = create_monochromatic_linear_colormap(plot_color)
                outline_color = plot_color
            else:
                cont_cmap = create_monochromatic_linear_colormap(plot_color)
                outline_color = plot_color

            cf = contributions[comp].to_numpy()
            x, y = np.meshgrid(
                contributions[comp].index,
                contributions[comp].columns,
            )
            contribution_contour_resolution_reduction_factor = 8

            contribution_ax.contourf(
                x[
                    :,
                    band_breaks[i] : band_breaks[
                        i + 1
                    ] : contribution_contour_resolution_reduction_factor,
                ],
                y[
                    :,
                    band_breaks[i] : band_breaks[
                        i + 1
                    ] : contribution_contour_resolution_reduction_factor,
                ],
                np.log10(cf).T[
                    :,
                    band_breaks[i] : band_breaks[
                        i + 1
                    ] : contribution_contour_resolution_reduction_factor,
                ],
                cmap=cont_cmap,
                levels=contributions_max - np.array([4, 2, 0]),
                alpha=0.66,
                zorder=0,
            )
            contribution_ax.contour(
                x[
                    :,
                    band_breaks[i] : band_breaks[
                        i + 1
                    ] : contribution_contour_resolution_reduction_factor,
                ],
                y[
                    :,
                    band_breaks[i] : band_breaks[
                        i + 1
                    ] : contribution_contour_resolution_reduction_factor,
                ],
                np.log10(cf).T[
                    :,
                    band_breaks[i] : band_breaks[
                        i + 1
                    ] : contribution_contour_resolution_reduction_factor,
                ],
                colors=outline_color,
                levels=contributions_max - np.array([2]),
                # linewidths=3,
                alpha=1,
                zorder=1,
            )
            if j == 0:
                contribution_ax.invert_yaxis()
            if "cloud" in contributions:
                cloud_cf = contributions["cloud"]
                x, y = np.meshgrid(cloud_cf.index, cloud_cf.columns)
                contribution_ax.contourf(
                    x[
                        :,
                        band_breaks[i] : band_breaks[
                            i + 1
                        ] : contribution_contour_resolution_reduction_factor,
                    ],
                    y[
                        :,
                        band_breaks[i] : band_breaks[
                            i + 1
                        ] : contribution_contour_resolution_reduction_factor,
                    ],
                    cloud_cf.to_numpy().T[
                        :,
                        band_breaks[i] : band_breaks[
                            i + 1
                        ] : contribution_contour_resolution_reduction_factor,
                    ],
                    colors="k",
                    # cmap=cmap_cloud,
                    alpha=0.75,
                    # levels=np.logspace(-1, 2, num=20),
                    levels=[0.1, 0.75],
                    zorder=2,
                )
                if i == 0:
                    contribution_ax.contour(
                        x[
                            :,
                            band_breaks[i] : band_breaks[
                                i + 1
                            ] : contribution_contour_resolution_reduction_factor,
                        ],
                        y[
                            :,
                            band_breaks[i] : band_breaks[
                                i + 1
                            ] : contribution_contour_resolution_reduction_factor,
                        ],
                        cloud_cf.to_numpy().T[
                            :,
                            band_breaks[i] : band_breaks[
                                i + 1
                            ] : contribution_contour_resolution_reduction_factor,
                        ],
                        colors="#DDDDDD",
                        linestyles="solid",
                        # linewidth=3,
                        alpha=1,
                        levels=[0.1],
                        zorder=3,
                    )
                contribution_ax.tick_params(axis="x", labelsize=26)
                contribution_ax.tick_params(axis="y", labelsize=26)
                contribution_ax.minorticks_on()
            # contribution_ax.contour(x[:, band_breaks[i]:band_breaks[i+1]:8], y[:, band_breaks[i]:band_breaks[i+1]:8],
            #                       np.log10(cf).T[:, band_breaks[i]:band_breaks[i+1]:8],
            #                       cmap=cmap,
            #                       levels=contributions_max-np.array([3, 2, 1, 0]),
            #                       alpha=1,
            #                       zorder=0)
    return figure, spectrum_axes, residual_axes, contribution_columns


def wrap_contributions_plot(figure: plt.Figure, gridspec: GridSpec):
    contributions_ax = figure.add_subplot(gridspec[2:, :])
    contributions_ax.spines["top"].set_color("none")
    contributions_ax.spines["bottom"].set_color("none")
    contributions_ax.spines["left"].set_color("none")
    contributions_ax.spines["right"].set_color("none")
    contributions_ax.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    contributions_ax.grid(False)
    contributions_ax.set_xlabel(
        r"$\lambda\left(\mu\mathrm{m}\right)$", fontsize=36, y=-0.075
    )
    contributions_ax.set_ylabel(
        r"$\log_{10}\left(P/\mathrm{bar}\right)$", fontsize=36, labelpad=20
    )

    return figure, contributions_ax


def make_multi_plots(
    results_directory: Pathlike,
    contribution_components: list[str],
    plotting_colors: dict[str, str],
    save_kwargs: dict[str, Any] = DEFAULT_SAVE_PLOT_KWARGS,
    plot_filetypes: Iterable[str] = DEFAULT_PLOT_FILETYPES,
):
    run_directories: dict[str, dict[str, Pathlike]] = unpack_results_filepaths(
        results_directory
    )

    j = 0
    for (run_name, run_filepath_dict), (run_name, plotting_color) in zip(
        run_directories.items(), plotting_colors.items()
    ):
        contribution_filepath = run_filepath_dict["contributions"]
        data_filepath: Path = run_filepath_dict["data"]

        with open(contribution_filepath, "rb") as pickle_file:
            contributions = pickle_compat.load(pickle_file)

        contribution_setup = setup_contribution_plots(contributions, data_filepath)

        number_of_bands: int = len(contribution_setup["wavelength_ranges"])
        multi_setup: MultiFigure = setup_multi_figure(
            create_Arthurs_multi_figure(
                number_of_contributions=len(contribution_components),
                number_of_bands=number_of_bands,
                wavelength_ranges=contribution_setup["wavelength_ranges"],
            )
        )

        band_boundaries = list(
            zip(
                contribution_setup["band_lower_wavelength_limit"],
                contribution_setup["band_upper_wavelength_limit"],
            )
        )

        MLE_spectrum_filepath = run_filepath_dict["MLE_model_spectrum"]
        _, _, model_spectrum, *_ = np.loadtxt(MLE_spectrum_filepath).T

        multi_figure_kwargs = MultiFigureBlueprintButt(
            contributions=contributions,
            list_of_band_boundaries=band_boundaries,
            band_breaks=contribution_setup["band_breaks"],
            contributions_max=contribution_setup["contributions_max"],
            wavelengths=contribution_setup["wavelengths"],
            datas=contribution_setup["data"],
            models=model_spectrum,
            errors=contribution_setup["errors"],
            model_title=run_name,
            number_of_parameters=23,
        )

        if j == 0:
            multi_figure, spectrum_axes, residual_axes, contribution_columns = (
                plot_multi_figure_iteration(
                    figure=multi_setup.figure,
                    gridspec=multi_setup.gridspec,
                    # axis_array=multi_setup.axis_array,
                    iteration_index=j,
                    plot_color=plotting_color,
                    **multi_figure_kwargs,
                )
            )

        else:
            multi_figure, spectrum_axes, residual_axes, contribution_columns = (
                plot_multi_figure_iteration(
                    figure=multi_setup.figure,
                    gridspec=multi_setup.gridspec,
                    # axis_array=multi_setup.axis_array,
                    iteration_index=j,
                    plot_color=plotting_color,
                    **multi_figure_kwargs,
                    spectrum_axes=spectrum_axes,
                    residual_axes=residual_axes,
                    contribution_columns=contribution_columns,
                )
            )

        wrap_contributions_plot(multi_figure, multi_setup.gridspec)

        for filetype in plot_filetypes:
            multi_figure.savefig(
                results_directory
                / run_name
                / (
                    run_name.replace("/", "_")
                    + f".fit-spectrum+contributions.{filetype}"
                ),
                **save_kwargs,
            )

    return None


def plot_spectra_across_runs() -> None:
    pass


class TPPlotBlueprint(TypedDict):
    log_pressures: Sequence[float]
    TP_profile_percentiles: Sequence[Sequence[float]]
    MLE_TP_profile: Sequence[float]
    plot_color: str
    MLE_plot_color: str
    plot_label: str
    object_label: str


class CloudLayerPlotBlueprint(TypedDict):
    minimum_log_pressure: float
    layer_thickness_in_log_pressure: float


def make_TP_profile_plot_by_run(
    figure: plt.Figure,
    axis: plt.Axes,
    log_pressures: Sequence[float],
    TP_profile_percentiles: Sequence[Sequence[float]],
    MLE_TP_profile: Sequence[float],
    plot_color: str,
    MLE_plot_color: str,
    plot_label: str,
    object_label: str,
    legend_dict: dict[str, Any] = None,
    cloud_layer_kwargs: CloudLayerPlotBlueprint = None,
) -> list[plt.Figure, plt.Axes]:
    T_minus_Nsigma, T_median, T_plus_Nsigma = TP_profile_percentiles
    # axis.plot(T_median, log_pressures, color=color, linewidth=1.5)
    axis.fill_betweenx(
        log_pressures,
        T_minus_Nsigma,
        T_plus_Nsigma,
        linewidth=0.5,
        color=plot_color,
        facecolor=plot_color,
        alpha=0.5,
        label=f"95\% confidence interval, {plot_label}",
    )
    # axis.plot(T_median, log_pressures, color=color, linewidth=1.5, label="Median profile")
    axis.plot(MLE_TP_profile, log_pressures, color=plot_color, linewidth=4)
    axis.plot(
        MLE_TP_profile,
        log_pressures,
        color=MLE_plot_color,
        linewidth=2,
        label="MLE profiles",
    )
    axis.set_ylim([np.min(log_pressures), np.max(log_pressures)])

    # reference_TP = true_values[temps]
    # log_pressures, reference_Tprofile = generate_profiles(reference_TP, Piette)
    # axis.plot(reference_Tprofile, log_pressures, color="#444444", linewidth=3)
    # axis.plot(reference_Tprofile, log_pressures, color=reference_color, linewidth=2, label="True profile")
    # axis.plot(sonora_T, sonora_P, linewidth=2, linestyle="dashed", color="sandybrown", label=r"SONORA, $\log g$=3.67, $T_\mathrm{eff}$=1584 K", zorder=-8)
    # axis.plot(sonora_T, sonora_P, linewidth=4, color="saddlebrown", label="", zorder=-9)

    figure.gca().invert_yaxis()
    # axis.set_xlim(axis.get_xlim()[0], 4000)

    if cloud_layer_kwargs is not None:
        axis.fill_between(
            np.linspace(axis.get_xlim()[0], 4000),
            cloud_layer_kwargs.minimum_log_pressure,
            cloud_layer_kwargs.minimum_log_pressure
            + cloud_layer_kwargs.layer_thickness_in_log_pressure,
            color="#444444",
            alpha=0.5,
            label="Retrieved cloud layer",
            zorder=-10,
        )

    axis.set_xlabel("Temperature (K)")
    axis.set_ylabel(r"log$_{10}$(Pressure/bar)")

    return figure, axis


def make_combined_TP_profile_plot(
    results_directory: Pathlike,
    combined_title: str,
    plotting_colors: dict[str, str],
    plot_kwargs=dict(figsize=(8, 6)),
    save_kwargs: dict[str, Any] = DEFAULT_SAVE_PLOT_KWARGS,
    plot_filetypes: Iterable[str] = DEFAULT_PLOT_FILETYPES,
) -> None:
    run_directories = unpack_results_filepaths(results_directory)

    TP_figure, TP_axis = plt.subplots(**plot_kwargs)

    for (run_name, run_filepath_dict), (run_name, plotting_color) in zip(
        run_directories.items(), plotting_colors.items()
    ):
        TP_profile_dataset = load_and_prep_dataset(run_filepath_dict["TP_dataset"])

        TP_1sigma_percentiles = calculate_percentile(
            TP_profile_dataset, percentiles=[16, 50, 84], axis=0
        ).temperatures.to_numpy()

        MLE_TP_profile_dataset = calculate_MLE(
            TP_profile_dataset
        ).temperatures.to_numpy()

        TP_plot_kwargs = TPPlotBlueprint(
            log_pressures=TP_profile_dataset.log_pressure,
            TP_profile_percentiles=np.asarray(TP_1sigma_percentiles),
            MLE_TP_profile=MLE_TP_profile_dataset,
            plot_color=plotting_color,
            MLE_plot_color=plotting_color,
            plot_label=run_name,
            object_label=run_name,
        )

        TP_figure, TP_axis = make_TP_profile_plot_by_run(
            TP_figure, TP_axis, **TP_plot_kwargs
        )

    original_handles, original_labels = TP_axis.get_legend_handles_labels()

    def replace_confidence_interval_legend_entries_with_general_entry(handles, labels):
        confidence_interval_entry_indices = [
            i for i, label in enumerate(labels) if "confidence" in label
        ]

        first_confidence_interval_handle = handles[confidence_interval_entry_indices[0]]
        first_confidence_interval_handle.set_facecolor(
            convert_to_greyscale(first_confidence_interval_handle.get_facecolor())
        )
        first_confidence_interval_handle.set_edgecolor(
            convert_to_greyscale(first_confidence_interval_handle.get_edgecolor())
        )

        non_confidence_handles = [
            handle
            for i, handle in enumerate(handles)
            if i not in confidence_interval_entry_indices
        ]
        non_confidence_labels = [
            label
            for i, label in enumerate(labels)
            if i not in confidence_interval_entry_indices
        ]

        def get_general_and_specfic_parts_of_legend_label(labels):
            general_labels = []
            specific_labels = []

            for label in labels:
                separator = ", "
                separator_index = label.index(separator)
                general_labels.append(label[:separator_index])
                specific_labels.append(label[separator_index + len(separator) :])

            return general_labels[0], specific_labels

        general_confidence_interval_label, specific_labels = (
            get_general_and_specfic_parts_of_legend_label(
                [
                    label
                    for i, label in enumerate(labels)
                    if i in confidence_interval_entry_indices
                ]
            )
        )

        return (
            non_confidence_handles + [first_confidence_interval_handle],
            [
                label + ", " + specific_label.replace("_", " ")
                for label, specific_label in zip(non_confidence_labels, specific_labels)
            ]
            + [general_confidence_interval_label],
        )

    handles, labels = replace_confidence_interval_legend_entries_with_general_entry(
        original_handles, original_labels
    )

    TP_axis.legend(
        handles,
        labels,
        fontsize=10,
        facecolor="#444444",
        framealpha=0.25,
        loc="upper right",
    )

    for filetype in plot_filetypes:
        TP_figure.savefig(
            results_directory / f"{combined_title}.TP_profiles.{filetype}",
            **save_kwargs,
        )

    return None


class CornerplotBlueprint(TypedDict):
    samples: NDArray[np.float64]
    weights: NDArray[np.float64]
    group_name: str
    parameter_names: Sequence[str]
    parameter_range: Sequence[Sequence[float]]
    confidence: float
    color: str
    MLE_values: Sequence[float]
    MLE_name: str
    MLE_color: str
    string_formats: str
    reference_values: Sequence[float] = None
    reference_name: str = None
    reference_markerstyle: str = None
    reference_color: str = None
    plot_generic_legend_labels: bool = True


def plot_cornerplot_by_group(
    group_name: str, cornerplot_kwargs: CornerplotBlueprint
) -> plt.Figure:
    print(f"Plotting corner plot for {group_name} parameters.")

    figure, _, _ = generate_cornerplot(**cornerplot_kwargs)

    return figure


def make_corner_plots(
    results_directory: Pathlike,
    plotting_colors: dict[str, str],
    shared_cornerplot_kwargs: dict[str, Any],
    plot_group_filepath: Pathlike = None,
    save_kwargs: dict[str, Any] = DEFAULT_SAVE_PLOT_KWARGS,
    plot_filetypes: dict[str, Any] = DEFAULT_PLOT_FILETYPES,
) -> None:
    run_directories = unpack_results_filepaths(results_directory)

    for (run_name, run_filepath_dict), (run_name, plotting_color) in zip(
        run_directories.items(), plotting_colors.items()
    ):
        samples_dataset = load_and_prep_dataset(run_filepath_dict["samples_dataset"])
        samples_dataset.update(change_units(samples_dataset.Rad, "Jupiter_radii"))

        MLE_values = calculate_MLE(samples_dataset)

        if plot_group_filepath is None:
            plot_group_filepath = Path(results_directory) / "cornerplot_groups.yaml"

        with open(plot_group_filepath, "r") as plot_group_file:
            cornerplot_groups = safe_load(plot_group_file)

        for group_name, parameters_specs in cornerplot_groups.items():
            parameter_names = list(parameters_specs.keys())

            print_names = [
                parameter_specs["print_name"]
                for parameter_specs in parameters_specs.values()
            ]

            print_formats = [
                parameter_specs["print_format"]
                for parameter_specs in parameters_specs.values()
            ]

            print(
                f"{run_name} - Plotting {group_name} corner plot, with parameter specs: {parameters_specs}."
            )
            group_dataset = samples_dataset.get(parameter_names).pint.dequantify()
            group_array = np.asarray(group_dataset.to_array().T)
            group_MLE_values = np.asarray(
                MLE_values.get(parameter_names).pint.dequantify().to_array()
            )

            cornerplot_kwargs = CornerplotBlueprint(
                samples=group_array,
                group_name=group_name,
                parameter_names=print_names,
                color=plotting_color,
                MLE_values=group_MLE_values,
                string_formats=print_formats,
                **shared_cornerplot_kwargs,
            )

            cornerplot_figure = plot_cornerplot_by_group(group_name, cornerplot_kwargs)

            for filetype in plot_filetypes:
                cornerplot_figure.savefig(
                    results_directory
                    / run_name
                    / f"{run_name}.{group_name}.{filetype}",
                    **save_kwargs,
                )

    return None


"""
def plot_MLE_spectrum_of_one_run_against_different_data(
    run_directory: Pathlike, other_data_filepath: Pathlike
) -> None:
    fitting_results_filepath = run_directory["fitting_results"]
    derived_fit_parameters_filepath = run_directory["derived_fit_parameters"]
    input_parameter_filepath = run_directory["input_parameters"]

    MLE_output_dict = create_MLE_output_dictionary(
        fitting_results_filepath,
        derived_fit_parameters_filepath,
        input_parameter_filepath,
    )
    MLE_output_dict["header"]["Data"] = (str(Path(other_data_filepath)),)

    output_MLE_filename = Path(
        str(input_parameter_filepath).replace("input", "retrieved_alt-data")
    )
    with open(output_MLE_filename, "w", newline="") as output_file:
        write_parsed_input_to_output(
            MLE_output_dict["header"], MLE_output_dict["parameters"], output_file
        )

    prepped_inputs_and_binned_wavelengths = prep_inputs_for_model(output_MLE_filename)
    prepped_inputs = prepped_inputs_and_binned_wavelengths["prepped_inputs"]
    binned_wavelengths = prepped_inputs_and_binned_wavelengths["binned_wavelengths"]

    observed_binned_model_spectrum = evaluate_model_spectrum(**prepped_inputs)

    different_data, different_errors = np.loadtxt(other_data_filepath).T[2:4]

    return {
        "wavelengths": binned_wavelengths,
        "data": different_data,
        "errors": different_errors,
        "spectrum": observed_binned_model_spectrum,
    }
"""
