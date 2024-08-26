from collections.abc import Sequence

from numpy import percentile
from xarray import Dataset, apply_ufunc


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
