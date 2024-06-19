from typing import Sequence, TypedDict

from numpy import percentile
from pint import Unit
from xarray import Dataset, apply_ufunc

from apollo.dataset.dataset_builders import organize_parameter_data_in_xarray


class RunDatasetBlueprint(TypedDict):
    parameter_names: Sequence[str]
    parameter_values: Sequence[float]
    parameter_units: Sequence[Unit | str]
    parameter_default_string_formattings: Sequence[str]
    parameter_group_names: Sequence[str]
    log_likelihoods: Sequence[float]


def make_run_parameter_dataset(
    parameter_names: Sequence[str],
    # parameter_print_names: Sequence[str],
    parameter_values: Sequence[float],
    parameter_units: Sequence[Unit | str],
    parameter_default_string_formattings: Sequence[str],
    parameter_group_names: Sequence[str],
    log_likelihoods: Sequence[float],
) -> Dataset:
    return Dataset(
        {
            parameter_name: organize_parameter_data_in_xarray(
                name=parameter_name,
                print_name="",
                value=parameter_value,
                unit=parameter_unit,
                coords=dict(log_likelihood=log_likelihoods),
                string_formatter=string_formatter,
                base_group=parameter_group_name,
            )
            for parameter_name, parameter_value, parameter_unit, string_formatter, parameter_group_name in zip(
                parameter_names,
                parameter_values,
                parameter_units,
                parameter_default_string_formattings,
                parameter_group_names,
            )
        }
    )


def calculate_percentile(
    dataset: Dataset,
    percentile_dimension_name: str = "log_likelihood",
    percentiles: list[float] = [16, 50, 84],  # plus or minus 1 sigma
    **percentile_kwargs,
) -> Dataset:
    dimensions = list(dataset.dims.keys())
    preserved_dimensions = [
        dimension for dimension in dimensions if dimension != percentile_dimension_name
    ]

    return apply_ufunc(
        percentile,
        dataset,
        input_core_dims=[dimensions],
        output_core_dims=[["percentile"] + preserved_dimensions],
        kwargs=dict(q=percentiles, **percentile_kwargs),
        keep_attrs=True,
    ).assign_coords({"percentile": percentiles})


def calculate_MLE(dataset: Dataset) -> Dataset:
    return dataset.isel(dataset.log_likelihood.argmax(...))


def change_parameter_values_using_MLE_dataset(
    parameter_name: str,
    parameter_dict: dict[str, Sequence[float]],
    MLE_dataset: Dataset,
) -> dict[str, Sequence[float]]:
    MLE_parameter_dict = parameter_dict.copy()

    if parameter_name == "Rad":
        MLE_parameter_dict["MLE"] = float(
            MLE_dataset.get(parameter_name).pint.to("Earth_radii").to_numpy()
        )
    else:
        MLE_parameter_dict["MLE"] = float(MLE_dataset.get(parameter_name).to_numpy())

    return MLE_parameter_dict
