from matplotlib import pyplot as plt
from matplotlib.colors import Colormap, CSS4_COLORS as COLORS
import numpy as np
from pandas.compat import pickle_compat
from pathlib import Path
from typing import Any, Sequence, TypedDict
import yaml

##########################################
# BOILERPLATE CODE TO RESOLVE APOLLO PATH
from os.path import abspath
import sys

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/APOLLO"
)
sys.path.append(APOLLO_DIRECTORY)
##########################################

from user_directories import USER_DIRECTORIES

plt.style.use(USER_DIRECTORIES["plots"] / "arthur.mplstyle")

from apollo.dataset_functions import load_dataset_with_units, change_units
from apollo.dynesty_plotting_functions import (
    setup_contribution_plots,
    setup_multi_figure,
    MultiFigureBlueprint,
    plot_multi_figure_iteration,
    wrap_contributions_plot,
    CornerplotBlueprint,
    plot_cornerplot_by_group,
    TPPlotBlueprint,
    make_TP_profile_plot_by_run,
)
from apollo.retrieval.dynesty.parse_dynesty_outputs import unpack_results_filepaths
from apollo.retrieval.dynesty.build_and_manipulate_datasets import (
    calculate_MLE,
    calculate_percentile,
)
from apollo.general_protocols import Pathlike
from apollo.visualization_functions import (
    create_linear_colormap,
    create_monochromatic_linear_colormap,
    convert_to_greyscale,
)

from user.plots.plots_config import PLOT_FILETYPES


class ColormapBlueprint(TypedDict):
    lightness_minimum: float
    lightness_maximum: float
    saturation_minimum: float
    saturation_maximum: float


CMAP_KWARGS: ColormapBlueprint = {
    "lightness_minimum": 0.15,
    "lightness_maximum": 0.85,
    "saturation_minimum": 0.2,
    "saturation_maximum": 0.8,
}

CMAP_H2O: Colormap = create_linear_colormap(["#226666", "#2E4172"], **CMAP_KWARGS)
CMAP_CO: Colormap = create_linear_colormap(["#882D60", "#AA3939"], **CMAP_KWARGS)
CMAP_CO2: Colormap = create_linear_colormap(["#96A537", "#669933"], **CMAP_KWARGS)
CMAP_CH4: Colormap = create_linear_colormap(["#96A537", "#669933"], **CMAP_KWARGS)

CMAP_CLOUDY: Colormap = create_monochromatic_linear_colormap(
    COLORS["lightcoral"], **CMAP_KWARGS
)
CMAP_CLEAR: Colormap = create_monochromatic_linear_colormap(
    COLORS["cornflowerblue"], **CMAP_KWARGS
)
CMAP_CLOUD: Colormap = plt.get_cmap("Greys")

PLOTTED_COMPONENTS: list[str] = ["h2o", "co", "co2", "ch4"]
PLOTTED_TITLES: list[str] = ["H$_2$O", "CO", "CO$_2$", "CH$_4$"]
CMAPS: list[Colormap] = [CMAP_H2O, CMAP_CO, CMAP_CO2, CMAP_CH4]

PADDING: float = 0.025
HK_HARDCODED_BOUNDARIES: list[list[float]] = [[1.40, 1.90], [1.90, 2.50]]
JWST_HARDCODED_BOUNDARIES: list[list[float]] = [[2.85, 4.01], [4.19, 5.30]]

DEFAULT_SAVE_PLOT_KWARGS = dict(
    dpi=300,
    transparent=True,
    bbox_inches="tight",
)

SHARED_CORNERPLOT_KWARGS = dict(
    weights=None,
    parameter_range=None,
    confidence=0.95,
    MLE_name="MLE",
    MLE_color="gold",
)

PLOTTING_COLORS: dict[str, str] = {
    "Ross458c": "crimson",
    "HK+JWST_2M2236_logg-normal": "darkorange",
    "HK+JWST_2M2236_logg-free": "lightsalmon",
    "JWST-only_2M2236_logg-free": "mediumpurple",
    "JWST-only_2M2236_logg-normal": "indigo",
}
MLE_COLOR = "gold"

RESULTS_DIRECTORY = USER_DIRECTORIES["results"]
OBJECT_NAME: str = "2M2236"
SPECIFIC_RESULTS_DIRECTORY: Path = RESULTS_DIRECTORY / OBJECT_NAME


def make_multi_plots(results_directory):
    run_directories: dict[str, dict[str, Pathlike]] = unpack_results_filepaths(
        results_directory
    )

    j = 0
    for k, (run_name, run_filepath_dict) in enumerate(run_directories.items()):
        contribution_filepath = (
            SPECIFIC_RESULTS_DIRECTORY / run_name / run_filepath_dict["contributions"]
        )
        data_filepath = (
            SPECIFIC_RESULTS_DIRECTORY / run_name / run_filepath_dict["data"]
        )

        with open(contribution_filepath, "rb") as pickle_file:
            contributions = pickle_compat.load(pickle_file)

        contribution_setup = setup_contribution_plots(contributions, data_filepath)

        number_of_bands = len(contribution_setup["wavelength_ranges"])
        multi_setup: Any = setup_multi_figure(
            number_of_contributions=len(PLOTTED_COMPONENTS),
            number_of_bands=number_of_bands,
            wavelength_ranges=contribution_setup["wavelength_ranges"],
        )

        band_boundaries = list(
            zip(
                contribution_setup["band_lower_wavelength_limit"],
                contribution_setup["band_upper_wavelength_limit"],
            )
        )

        MLE_spectrum_filepath = (
            SPECIFIC_RESULTS_DIRECTORY
            / run_name
            / run_filepath_dict["MLE_model_spectrum"]
        )
        _, _, model_spectrum, _, _, _ = np.loadtxt(MLE_spectrum_filepath).T

        multi_figure_kwargs = MultiFigureBlueprint(
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
                    **multi_setup,
                    iteration_index=j,
                    plot_color=PLOTTING_COLORS[run_name],
                    **multi_figure_kwargs._asdict(),
                )
            )

        else:
            multi_figure, spectrum_axes, residual_axes, contribution_columns = (
                plot_multi_figure_iteration(
                    **multi_setup,
                    iteration_index=j,
                    plot_color=PLOTTING_COLORS[run_name],
                    **multi_figure_kwargs._asdict(),
                    spectrum_axes=spectrum_axes,
                    residual_axes=residual_axes,
                    contribution_columns=contribution_columns,
                )
            )

        wrap_contributions_plot(multi_figure, multi_setup["gridspec"])

        for filetype in PLOT_FILETYPES:
            multi_figure.savefig(
                SPECIFIC_RESULTS_DIRECTORY
                / run_name
                / (run_name + f".fit-spectrum+contributions.{filetype}"),
                **DEFAULT_SAVE_PLOT_KWARGS,
            )

    return None


def make_TP_profile_plots(results_directory) -> None:
    run_directories = unpack_results_filepaths(results_directory)

    TP_figure, TP_axis = plt.subplots(figsize=(8, 6))
    for k, (run_name, run_filepath_dict) in enumerate(run_directories.items()):
        TP_profile_dataset = load_dataset_with_units(
            results_directory / run_name / run_filepath_dict["TP_dataset"]
        )

        TP_1sigma_percentiles = calculate_percentile(
            TP_profile_dataset, percentiles=[16, 50, 84], axis=0
        ).temperatures.to_numpy()

        MLE_TP_profile_dataset = calculate_MLE(
            TP_profile_dataset
        ).temperatures.to_numpy()

        plotting_color = PLOTTING_COLORS[run_name]

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
            TP_figure, TP_axis, **TP_plot_kwargs._asdict()
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

    for filetype in PLOT_FILETYPES:
        TP_figure.savefig(
            SPECIFIC_RESULTS_DIRECTORY / f"{OBJECT_NAME}.TP_profiles.{filetype}",
            **DEFAULT_SAVE_PLOT_KWARGS,
        )

    return None


def make_corner_plots(
    results_directory: Pathlike, plot_group_filepath: Pathlike = None
) -> None:
    run_directories = unpack_results_filepaths(results_directory)

    for k, (run_name, run_filepath_dict) in enumerate(run_directories.items()):
        samples_dataset = load_dataset_with_units(
            results_directory / run_name / run_filepath_dict["samples_dataset"]
        )
        samples_dataset.update(change_units(samples_dataset.Rad, "Jupiter_radii"))

        MLE_values = calculate_MLE(samples_dataset)

        if plot_group_filepath is None:
            plot_group_filepath = Path(results_directory) / "cornerplot_groups.yaml"

        with open(plot_group_filepath, "r") as plot_group_file:
            cornerplot_groups = yaml.safe_load(plot_group_file)

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

            group_dataset = samples_dataset.get(parameter_names).pint.dequantify()
            group_array = np.asarray(group_dataset.to_array().T)
            group_MLE_values = np.asarray(
                MLE_values.get(parameter_names).pint.dequantify().to_array()
            )

            SHARED_CORNERPLOT_KWARGS = dict(
                weights=None,
                parameter_range=None,
                confidence=0.95,
                MLE_name="MLE",
                MLE_color="gold",
            )

            cornerplot_kwargs = CornerplotBlueprint(
                samples=group_array,
                group_name=group_name,
                parameter_names=print_names,
                color=PLOTTING_COLORS[run_name],
                MLE_values=group_MLE_values,
                string_formats=print_formats,
                **SHARED_CORNERPLOT_KWARGS,
            )

            cornerplot_figure = plot_cornerplot_by_group(group_name, cornerplot_kwargs)

            for filetype in PLOT_FILETYPES:
                cornerplot_figure.savefig(
                    SPECIFIC_RESULTS_DIRECTORY
                    / run_name
                    / f"{run_name}.{group_name}.{filetype}",
                    **DEFAULT_SAVE_PLOT_KWARGS,
                )

    return None


if __name__ == "__main__":
    make_multi_plots(SPECIFIC_RESULTS_DIRECTORY)
    make_TP_profile_plots(SPECIFIC_RESULTS_DIRECTORY)
    make_corner_plots(SPECIFIC_RESULTS_DIRECTORY)
