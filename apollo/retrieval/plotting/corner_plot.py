from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from yaml import safe_load

from apollo.dataset.accessors import change_units
from apollo.dataset.IO import load_dataset_with_units
from apollo.formats.custom_types import Pathlike
from apollo.generate_cornerplot import generate_cornerplot
from apollo.retrieval.results.IO import unpack_results_filepaths
from apollo.retrieval.results.manipulate_results_datasets import calculate_MLE
from user.plots.plots_config import DEFAULT_PLOT_FILETYPES, DEFAULT_SAVE_PLOT_KWARGS


class CornerplotBlueprint(TypedDict):
    samples: NDArray[np.float_]
    weights: NDArray[np.float_]
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
        samples_dataset = load_dataset_with_units(run_filepath_dict["samples_dataset"])
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
