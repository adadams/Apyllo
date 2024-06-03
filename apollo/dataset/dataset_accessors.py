from typing import Any, Callable

from xarray import DataArray, Dataset


def change_units(dataarray: DataArray, new_units: str) -> Dataset:
    return {dataarray.name: dataarray.pint.to(new_units)}


def get_attribute_from_dataset(
    input_dataset: Dataset, attribute_name: str
) -> list[Any]:
    return [
        dataarray.attrs[attribute_name]
        for variable_name, dataarray in input_dataset.items()
    ]


def extract_dataset_subset_by_attribute(
    dataset: Dataset, attribute_condition: Callable[[Dataset, str], bool]
) -> Dataset:
    return dataset.get(
        [
            data_var
            for data_var in dataset.data_vars
            if attribute_condition(dataset, data_var)
        ]
    )


def extract_dataset_subset_by_parameter_group(
    dataset: Dataset, group_name: str, group_attribute_label: str = "base_group"
) -> Dataset:
    def belongs_to_a_group(dataset, variable):
        return dataset.get(variable).attrs[group_attribute_label] == group_name

    return extract_dataset_subset_by_attribute(dataset, belongs_to_a_group)


def extract_free_parameters_from_dataset(dataset: Dataset) -> Dataset:
    def is_a_free_parameter(dataset, variable):
        return dataset.get(variable).attrs["base_group"] != "Derived"

    return extract_dataset_subset_by_attribute(dataset, is_a_free_parameter)
