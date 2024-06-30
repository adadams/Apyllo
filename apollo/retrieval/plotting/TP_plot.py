from collections.abc import Iterable, Sequence
from typing import Any, TypedDict

import numpy as np
from matplotlib import pyplot as plt

from apollo.dataset.IO import load_dataset_with_units
from apollo.formats.custom_types import Pathlike
from apollo.retrieval.results.IO import unpack_results_filepaths
from apollo.retrieval.results.manipulate_results_datasets import (
    calculate_MLE,
    calculate_percentile,
)
from apollo.visualization_functions import convert_to_greyscale
from user.plots.plots_config import DEFAULT_PLOT_FILETYPES, DEFAULT_SAVE_PLOT_KWARGS


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
        TP_profile_dataset = load_dataset_with_units(run_filepath_dict["TP_dataset"])

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
