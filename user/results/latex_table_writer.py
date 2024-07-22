from typing import Callable

from xarray import Dataset

from apollo.retrieval.results.manipulate_results_datasets import (
    calculate_MLE,
    calculate_percentile,
)
from custom_types import Pathlike
from dataset.accessors import load_dataset_with_units


def format_percentile_for_latex(
    lower_percentile: float,
    middle_percentile: float,
    upper_percentile: float,
    print_formatting: str = ".2f",
):
    lower_range: float = middle_percentile - lower_percentile
    upper_range: float = upper_percentile - middle_percentile

    formatted_lower_range: str = format(lower_range, print_formatting)
    formatted_middle_percentile: str = format(middle_percentile, print_formatting)
    formatted_upper_range: str = format(upper_range, print_formatting)

    formatted_percentile_range: str = (
        rf"${formatted_middle_percentile}^{{+{formatted_upper_range}}}_{{-{formatted_lower_range}}}$"
        if formatted_lower_range != formatted_upper_range
        else rf"${formatted_middle_percentile}\pm{formatted_lower_range}$"
    )
    return formatted_percentile_range


def format_number_for_latex(number: str, print_formatting: str = ".2f"):
    return rf"${format(number, print_formatting)}$"


def format_percentiles_and_MLEs_and_store_as_attributes(dataset: Dataset) -> Dataset:
    percentile_dataset: Dataset = calculate_percentile(dataset)
    MLE_dataset: Dataset = calculate_MLE(dataset)

    for (
        (variable_name, variable_percentiles),
        (variable_name, variable_MLE),
        (variable_name, variable_dataarray),
    ) in zip(percentile_dataset.items(), MLE_dataset.items(), dataset.items()):
        print_formatting = variable_dataarray.attrs["string_formatter"]

        percentile_range: str = format_percentile_for_latex(
            *variable_percentiles.values, print_formatting
        )
        MLE_value: str = format_number_for_latex(variable_MLE.values, print_formatting)

        cell_latex: str = f" {percentile_range} & {MLE_value} "

        dataset[variable_name] = variable_dataarray.assign_attrs(cell_latex=cell_latex)

    return dataset


def stitch_rows_across_runs(
    *datasets: list[Dataset], parameter_print_names: None
) -> None:
    rows = []
    for parameter_index, sequence_of_variable_names_and_values in enumerate(
        zip(*[dataset.items() for dataset in datasets])
    ):
        variable_names, variable_values = list(
            zip(*sequence_of_variable_names_and_values)
        )
        variable_name = (
            variable_names[0]
            if parameter_print_names is None
            else parameter_print_names[parameter_index]
        )

        variable_cell_latex = [
            variable_value.attrs["cell_latex"] for variable_value in variable_values
        ]
        variable_row_latex = "&".join(variable_cell_latex)

        rows.append(rf"{variable_name} &{variable_row_latex}\\")
        # rows.append(rf"\textbf{{{variable_name}}} &{variable_row_latex}\\")

    return rows


def load_results_dataset_and_store_formatted_table_text(
    samples_dataset_filepath: Pathlike,
    formatting_function: Callable[[Dataset], Dataset],
    radius_units: str = "Jupiter_radii",
) -> Dataset:
    samples_dataset: Dataset = load_dataset_with_units(samples_dataset_filepath)

    if "Rad" in samples_dataset.data_vars:
        samples_dataset["Rad"] = samples_dataset["Rad"].pint.to(radius_units)

    return formatting_function(samples_dataset)
