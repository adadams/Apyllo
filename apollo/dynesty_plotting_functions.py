from matplotlib import pyplot as plt
from matplotlib.colors import CSS4_COLORS as cnames
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import numpy as np
from numpy.typing import ArrayLike
from pathlib import Path
from typing import Any, NamedTuple, Sequence

from apollo.general_protocols import Pathlike
from apollo.generate_cornerplot import generate_cornerplot
from apollo.visualization_functions import (
    create_linear_colormap,
    create_monochromatic_linear_colormap,
)

from user.plots.plots_config import PLOT_FILETYPES

CMAP_KWARGS = {
    "lightness_minimum": 0.15,
    "lightness_maximum": 0.85,
    "saturation_minimum": 0.2,
    "saturation_maximum": 0.8,
}

APOLLO_USER_PLOTS_DIRECTORY = (
    Path("~/Documents/Astronomy/2019/Retrieval/Code/APOLLO") / "user" / "plots"
)
plt.style.use(APOLLO_USER_PLOTS_DIRECTORY / "arthur.mplstyle")

CMAP_H2O = create_linear_colormap(["#226666", "#2E4172"], **CMAP_KWARGS)
CMAP_CO = create_linear_colormap(["#882D60", "#AA3939"], **CMAP_KWARGS)
CMAP_CO2 = create_linear_colormap(["#96A537", "#669933"], **CMAP_KWARGS)
CMAP_CH4 = create_linear_colormap(["#96A537", "#669933"], **CMAP_KWARGS)

CMAP_CLOUDY = create_linear_colormap(
    [cnames["lightcoral"], cnames["lightcoral"]], **CMAP_KWARGS
)
CMAP_CLEAR = create_linear_colormap(
    [cnames["cornflowerblue"], cnames["cornflowerblue"]], **CMAP_KWARGS
)

CMAP_CLOUD = plt.get_cmap("Greys")

PLOTTED_COMPONENTS = ["h2o", "co", "co2", "ch4"]
PLOTTED_TITLES = ["H$_2$O", "CO", "CO$_2$", "CH$_4$"]
CMAPS = [CMAP_H2O, CMAP_CO, CMAP_CO2, CMAP_CH4]

PADDING = 0.025
JWST_HARDCODED_BOUNDARIES = [[2.85, 4.01], [4.19, 5.30]]

DEFAULT_SAVE_PLOT_KWARGS = dict(
    dpi=300,
    transparent=True,
    bbox_inches="tight",
)


def setup_contribution_plots(
    contributions: dict[str, ArrayLike], data_filepath: Pathlike
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
    number_of_bands = len(band_breaks) - 1

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
    wavelength_ranges = np.array(
        [
            band_upper_wavelength_limit - band_lower_wavelength_limit
            for i in range(number_of_bands)
        ]
    )

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


def setup_multi_figure(
    number_of_contributions: int,
    number_of_bands: int,
    wavelength_ranges: Sequence[float],
) -> plt.figure, GridSpec:

    fig = plt.figure(figsize=(40, 30))
    gs = fig.add_gridspec(
        nrows=number_of_contributions + 2,
        ncols=number_of_bands,
        height_ratios=[4, 2] + ([3] * number_of_contributions),
        width_ratios=(1 + 2 * PADDING) * wavelength_ranges,
        wspace=0.1,
    )

    return dict(figure=fig, gridspec=gs)


def make_spectrum_and_residual_axes(
    figure: plt.figure, gridspec: GridSpec, band_index: int
) -> list[plt.axis]:
    spectrum_ax = figure.add_subplot(gridspec[0, band_index])
    residual_ax = figure.add_subplot(gridspec[1, band_index], sharex=spectrum_ax)

    return spectrum_ax, residual_ax


class MultiFigureBlueprint(NamedTuple):
    contributions: dict[str, ArrayLike]
    list_of_band_boundaries: Sequence[Sequence[float]]
    band_breaks: Sequence[Sequence[float]]
    contributions_max: float
    wavelengths: Sequence[float]
    datas: Sequence[float]
    models: Sequence[float]
    errors: Sequence[float]
    model_title: str
    number_of_parameters: int


def make_spectrum_figure() -> list[plt.figure, plt.axis]:
    pass


def calculate_and_print_chi_squared(
    residuals: ArrayLike, number_of_parameters: int
) -> float:
    reduced_chi_squared = np.sum(residuals**2) / (
        np.shape(residuals)[0] - number_of_parameters
    )

    print(f"Reduced chi squared is {reduced_chi_squared}")
    return reduced_chi_squared


def calculate_residuals(
    datas: ArrayLike,
    models: ArrayLike,
    errors: ArrayLike,
) -> ArrayLike:
    return (models - datas) / errors


def plot_on_residual_axis(
    residual_axis: plt.axis,
    wavelengths: ArrayLike,
    residuals: ArrayLike,
    plot_kwargs: dict[str, Any],
    axhline_kwargs: dict[str, Any] = None,
    yaxis_label_fontsize: int | float = None,
) -> plt.axis:
    wave_min = np.min(wavelengths)
    wave_max = np.max(wavelengths)
    xmin = wave_min - PADDING * np.abs(wave_max - wave_min)
    xmax = wave_max + PADDING * np.abs(wave_max - wave_min)
    residual_axis.set_xlim([xmin, xmax])

    residual_ymin = np.nanmin(residuals)
    residual_ymax = np.nanmax(residuals)
    ymin = residual_ymin - PADDING * np.abs(residual_ymax - residual_ymin)
    ymax = residual_ymax + PADDING * np.abs(residual_ymax - residual_ymin)
    residual_axis.set_ylim([ymin, ymax])

    # color=plot_color,
    # linewidth=3,
    # linestyle=linestyles[j],
    # alpha=1,
    # zorder=2 - j,
    residual_axis.plot(wavelengths, residuals, **plot_kwargs)

    if not axhline_kwargs:
        axhline_kwargs = dict(
            y=0, color="#444444", linewidth=2, linestyle="dashed", zorder=-10
        )
    residual_axis.axhline(**axhline_kwargs)

    residual_axis.minorticks_on()

    if yaxis_label_fontsize:
        residual_axis.set_ylabel(r"Residual/$\sigma$", fontsize=yaxis_label_fontsize)

    return residual_axis


def make_contribution_figure_per_species() -> list[plt.figure, plt.axis]:
    pass


def plot_multi_figure_iteration(
    figure: plt.figure,
    gridspec: GridSpec,
    contributions: dict[str, ArrayLike],
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
    spectrum_axes=None,
    residual_axes=None,
    contribution_columns=None,
) -> list[plt.figure, tuple[plt.axis]]:
    j = iteration_index

    if (
        (spectrum_axes is None)
        and (residual_axes is None)
        and (contribution_columns is None)
    ):
        spectrum_axes = []
        residual_axes = []
        contribution_columns = []

    residuals = (models - datas) / errors

    residual_ymin = np.nanmin(residuals)
    residual_ymax = np.nanmax(residuals)
    residual_ymin = residual_ymin - PADDING * np.abs(residual_ymax - residual_ymin)
    residual_ymax = residual_ymax + PADDING * np.abs(residual_ymax - residual_ymin)

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

        xmin = np.min(wavelengths_in_band)
        xmax = np.max(wavelengths_in_band)
        xmin = xmin - PADDING * np.abs(xmax - xmin)
        xmax = xmax + PADDING * np.abs(xmax - xmin)

        spectrum_ax.set_xlim([xmin, xmax])

        linestyles = ["solid", "solid", "solid", "solid"]
        spectrum_ax.errorbar(
            wavelengths_in_band,
            data,
            error,
            color="#444444",
            fmt="x",
            linewidth=0,
            elinewidth=2,
            alpha=1,
            zorder=-3,
        )
        spectrum_ax.plot(
            wavelengths_in_band,
            model,
            color=plot_color,
            linewidth=3,
            linestyle=linestyles[j],
            alpha=1,
            zorder=2 - j,
            label=model_title,
        )

        # if i==0 and j==0:
        # [spectrum_ax.axvline(line_position, linestyle="dashed", linewidth=1.5, zorder=-10, color="#888888")
        # for line_position in [1.139, 1.141, 1.169, 1.177, 1.244, 1.253, 1.268]]
        # y_text = spectrum_ax.get_ylim()[0] + 0.1*np.diff(spectrum_ax.get_ylim())
        # spectrum_ax.text((1.169+1.177)/2, y_text, "KI", fontsize=20, horizontalalignment="center")
        # spectrum_ax.text((1.244+1.253)/2, y_text, "KI", fontsize=20, horizontalalignment="center")
        # spectrum_ax.text(1.268, y_text, "NaI", fontsize=20, horizontalalignment="center")

        residual_ax.plot(
            wavelengths_in_band,
            residual,
            color=plot_color,
            linewidth=3,
            linestyle=linestyles[j],
            alpha=1,
            zorder=2 - j,
        )
        residual_ax.axhline(
            0, color="#444444", linewidth=2, linestyle="dashed", zorder=-10
        )
        residual_ax.set_ylim([residual_ymin, residual_ymax])
        residual_ax.minorticks_on()

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

        # residual = (model-data)/error
        reduced_chi_squared = np.sum(residual**2) / (
            np.shape(residual)[0] - number_of_parameters
        )
        print("Reduced chi squared is {}".format(reduced_chi_squared))

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
            contribution_ax.contourf(
                x[:, band_breaks[i] : band_breaks[i + 1] : 8],
                y[:, band_breaks[i] : band_breaks[i + 1] : 8],
                np.log10(cf).T[:, band_breaks[i] : band_breaks[i + 1] : 8],
                cmap=cont_cmap,
                levels=contributions_max - np.array([4, 2, 0]),
                alpha=0.66,
                zorder=0,
            )
            contribution_ax.contour(
                x[:, band_breaks[i] : band_breaks[i + 1] : 8],
                y[:, band_breaks[i] : band_breaks[i + 1] : 8],
                np.log10(cf).T[:, band_breaks[i] : band_breaks[i + 1] : 8],
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
                    x[:, band_breaks[i] : band_breaks[i + 1] : 8],
                    y[:, band_breaks[i] : band_breaks[i + 1] : 8],
                    cloud_cf.to_numpy().T[:, band_breaks[i] : band_breaks[i + 1] : 8],
                    colors="k",
                    # cmap=cmap_cloud,
                    alpha=0.75,
                    # levels=np.logspace(-1, 2, num=20),
                    levels=[0.1, 0.75],
                    zorder=2,
                )
                if i == 0:
                    contribution_ax.contour(
                        x[:, band_breaks[i] : band_breaks[i + 1] : 8],
                        y[:, band_breaks[i] : band_breaks[i + 1] : 8],
                        cloud_cf.to_numpy().T[
                            :, band_breaks[i] : band_breaks[i + 1] : 8
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


def wrap_contributions_plot(figure: plt.figure, gridspec: GridSpec):
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


def plot_multi_figure(
    figure: plt.figure,
    gridspec: GridSpec,
    list_of_multi_figure_kwargs: Sequence[MultiFigureBlueprint],
    object_label: str,
):
    for j, multi_figure_kwargs in enumerate(list_of_multi_figure_kwargs):
        plot_multi_figure_iteration(
            figure, gridspec, iteration_index=j, **multi_figure_kwargs
        )

    wrap_contributions_plot(figure, gridspec)

    for filetype in PLOT_FILETYPES:
        plt.savefig(
            object_label + ".fit-spectrum+contributions.{}".format(filetype),
            **DEFAULT_SAVE_PLOT_KWARGS,
        )

    return figure


class TPPlotBlueprint(NamedTuple):
    log_pressures: Sequence[float]
    TP_profile_percentiles: Sequence[Sequence[float]]
    MLE_TP_profile: Sequence[float]
    plot_color: str
    MLE_plot_color: str
    plot_label: str
    object_label: str


class CloudLayerPlotBlueprint(NamedTuple):
    minimum_log_pressure: float
    layer_thickness_in_log_pressure: float


def make_TP_profile_plot_by_run(
    figure: plt.figure,
    axis: plt.axis,
    log_pressures: Sequence[float],
    TP_profile_percentiles: Sequence[Sequence[float]],
    MLE_TP_profile: Sequence[float],
    plot_color: str,
    MLE_plot_color: str,
    plot_label: str,
    object_label: str,
    legend_dict: dict[str, Any] = None,
    cloud_layer_kwargs: CloudLayerPlotBlueprint = None,
) -> list[plt.figure, plt.axis]:
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


class CornerplotBlueprint(NamedTuple):
    samples: ArrayLike
    weights: ArrayLike
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
) -> plt.figure:
    print(f"Group: {group_name}")

    figure, _, _ = generate_cornerplot(**cornerplot_kwargs._asdict())

    return figure
