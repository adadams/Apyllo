# %%
from collections import ChainMap
import dynesty
import dynesty.plotting as dyplot
from functools import partial
from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from pathlib import Path
import pickle
from pint import Unit
from pprint import pprint
from typing import Sequence
from xarray import Dataset, apply_ufunc
import yaml

from general_protocols import organize_parameter_data_in_xarray, Pathlike
from make_forward_model_from_file import prep_inputs_for_model, evaluate_model_spectrum
import TP_functions
from user.models.inputs.parse_APOLLO_inputs import (
    parse_APOLLO_input_file,
    change_properties_of_parameters,
    write_parsed_input_to_output,
)


PLOTTED_FILETYPES = ["pdf", "png"]

PLOTTING_COLORS = ["lightcoral"]
MLE_COLOR = "gold"

USER_PATH = Path.cwd() / "user"
USER_DIRECTORY_PATH = USER_PATH / "directories.yaml"
with open(USER_DIRECTORY_PATH, "r") as directory_file:
    RESULTS_DIRECTORY = USER_PATH / yaml.safe_load(directory_file)["results"]


def load_results_directories(
    results_directory: Pathlike,
    directory_yaml_filename: Pathlike = "results_files.yaml",
):
    with open(results_directory / directory_yaml_filename, "r") as directory_file:
        loaded_file = yaml.safe_load_all(directory_file)
        directory_dict = ChainMap(*loaded_file)

    return directory_dict


def load_dynesty_results(filepath: Pathlike):
    with open(filepath, "rb") as results_file:
        dynesty_results = pickle.load(results_file)

    return dynesty_results


def load_derived_parameters(filepath: Pathlike):
    with open(filepath, "rb") as derived_parameters_file:
        derived_parameters = pickle.load(derived_parameters_file)

    return derived_parameters


def compile_dynesty_parameters(
    dynesty_results: dynesty.results.Results, derived_parameters: ArrayLike
):
    return np.c_[dynesty_results.samples, derived_parameters]


def make_filter_of_dynesty_samples_by_importance(
    dynesty_results: dynesty.results.Results, importance_weight_percentile=0.95
):
    importance_weights = dynesty_results.importance_weights()
    cumulative_importance_weights = np.cumsum(importance_weights[::-1])[::-1]

    is_important_enough = cumulative_importance_weights <= importance_weight_percentile
    return is_important_enough


def make_run_parameter_dataset(
    parameter_names,
    parameter_values,
    parameter_units,
    parameter_default_string_formattings,
    parameter_group_names,
    log_likelihoods,
):

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


def extract_dataset_subset_by_parameter_group(
    dataset: Dataset, group_name: str, attribute_label: str = "base_group"
):
    return dataset.get(
        [
            data_var
            for data_var in results_dataset.data_vars
            if results_dataset.get(data_var).attrs[attribute_label] == group_name
        ]
    )


def extract_free_parameters_from_dataset(
    dataset: Dataset, attribute_label: str = "base_group"
):
    return dataset.get(
        [
            data_var
            for data_var in results_dataset.data_vars
            if results_dataset.get(data_var).attrs[attribute_label] != "Derived"
        ]
    )


def guess_default_units_from_parameter_names(parameter_names: Sequence[str]):
    guessed_units = []

    for parameter_name in parameter_names:
        if "rad" in parameter_name.lower():
            guessed_unit = "Earth_radii"

        elif "mass" in parameter_name.lower():
            guessed_unit = "Jupiter_masses"

        elif ("T_" in parameter_name) or (parameter_name == "Teff"):
            guessed_unit = "kelvin"

        elif parameter_name == "deltaL":
            guessed_unit = "nanometers"

        else:
            guessed_unit = ""

        guessed_units.append(guessed_unit)

    return guessed_units


def guess_default_string_formats_from_parameter_names(parameter_names: Sequence[str]):
    guessed_float_precisions = []

    for parameter_name in parameter_names:
        if "mass" in parameter_name.lower():
            guessed_float_precision = ".0f"

        elif ("T_" in parameter_name) or (parameter_name == "Teff"):
            guessed_float_precision = ".0f"

        else:
            guessed_float_precision = ".2f"

        guessed_float_precisions.append(guessed_float_precision)

    return guessed_float_precisions


def get_parameter_properties_from_defaults(
    parsed_parameter_file: dict[str, dict[str, float]],
    derived_parameter_names: Sequence[str] = [
        "Mass",
        "C/O",
        "[Fe/H]",
        "Teff",
    ],
):
    free_parameter_names = parsed_parameter_file["parameter_names"]
    parameter_names = free_parameter_names + derived_parameter_names

    parameter_units = guess_default_units_from_parameter_names(parameter_names)

    free_parameter_group_slices = parsed_parameter_file["parameter_group_slices"]

    free_parameter_group_names = np.empty_like(free_parameter_names)
    for group_name, group_slice in free_parameter_group_slices.items():
        free_parameter_group_names[group_slice] = group_name
    derived_parameter_group_names = ["Derived"] * len(derived_parameter_names)
    parameter_group_names = (
        list(free_parameter_group_names) + derived_parameter_group_names
    )

    parameter_default_string_formattings = (
        guess_default_string_formats_from_parameter_names(parameter_names)
    )

    return dict(
        parameter_names=parameter_names,
        parameter_units=parameter_units,
        parameter_default_string_formattings=parameter_default_string_formattings,
        parameter_group_names=parameter_group_names,
    )


def calculate_percentile(
    dataset,
    percentile_dimension_name="log_likelihood",
    percentiles=[16, 50, 84],  # plus or minus 1 sigma
    **percentile_kwargs,
):
    return apply_ufunc(
        np.percentile,
        dataset,
        input_core_dims=[[percentile_dimension_name]],
        output_core_dims=[["percentile"]],
        kwargs=dict(q=percentiles, axis=-1, **percentile_kwargs),
        keep_attrs=True,
    ).assign_coords({"percentile": percentiles})


def calculate_MLE(dataset):
    return dataset.isel(dataset.log_likelihood.argmax(...))


def change_parameter_values_using_MLE_dataset(
    parameter_name, parameter_dict, MLE_dataset
):
    MLE_parameter_dict = parameter_dict.copy()
    if parameter_name == "Rad":
        MLE_parameter_dict["MLE"] = float(
            MLE_dataset.get(parameter_name).pint.to("Earth_radii").to_numpy()
        )
    else:
        MLE_parameter_dict["MLE"] = float(MLE_dataset.get(parameter_name).to_numpy())

    return MLE_parameter_dict


def make_MLE_parameter_file_from_input_parameter_file(
    run_filepath_dict: dict[str, Pathlike],
    results_directory: Path = RESULTS_DIRECTORY,
):
    dynesty_results = load_dynesty_results(run_filepath_dict["fitting_results"])
    derived_parameters = load_derived_parameters(
        run_filepath_dict["derived_fit_parameters"]
    )

    is_important_enough = make_filter_of_dynesty_samples_by_importance(dynesty_results)
    parameter_samples = compile_dynesty_parameters(dynesty_results, derived_parameters)[
        is_important_enough
    ]
    log_likelihoods = dynesty_results.logl[is_important_enough]

    input_parameter_file = results_directory / run_filepath_dict["input_parameters"]
    with open(input_parameter_file, newline="") as input_file:
        parsed_input_file = parse_APOLLO_input_file(input_file, delimiter=" ")

    input_parameter_dict = parsed_input_file["parameters"]

    parameter_properties = get_parameter_properties_from_defaults(input_parameter_dict)

    sample_dataset = make_run_parameter_dataset(
        **parameter_properties,
        parameter_values=parameter_samples.T,
        log_likelihoods=log_likelihoods,
    )
    MLE_parameters = calculate_MLE(sample_dataset)

    MLE_output_parameter_dict = change_properties_of_parameters(
        input_parameter_dict,
        change_parameter_values_using_MLE_dataset,
        MLE_parameters,
    )

    output_MLE_filename = run_filepath_dict["input_parameters"].replace(
        "input", "retrieved"
    )
    with open(output_MLE_filename, "w", newline="") as output_file:
        write_parsed_input_to_output(
            parsed_input_file["header"], MLE_output_parameter_dict, output_file
        )


def make_dataset_from_APOLLO_parameter_file(results_directory: Path, **parsing_kwargs):
    retrieved_parameter_file = (
        results_directory / run_name / run_filepath_dict["MLE_parameters"]
    )
    with open(retrieved_parameter_file, newline="") as retrieved_file:
        parsed_retrieved_file = parse_APOLLO_input_file(
            retrieved_file, **parsing_kwargs
        )

    parameter_properties = get_parameter_properties_from_defaults(parsed_retrieved_file)

    return make_run_parameter_dataset(
        **parameter_properties,
        parameter_values=parameter_samples.T,
        log_likelihoods=log_likelihoods,
    )


def get_TP_function_from_APOLLO_parameter_file(
    results_directory: Path, **parsing_kwargs
):
    retrieved_parameter_file = (
        results_directory / run_name / run_filepath_dict["MLE_parameters"]
    )
    with open(retrieved_parameter_file, newline="") as retrieved_file:
        parsed_retrieved_file = parse_APOLLO_input_file(
            retrieved_file, **parsing_kwargs
        )

    TP_function_name = parsed_retrieved_file["parameters"]["Atm"]["options"][0]
    return TP_function_name


def evaluate_TP_functions_from_dataset(
    TP_dataset: Dataset,
    log_pressures: Sequence[float],
    output_temperature_unit: Unit | str = "kelvin",
    loop_dimension: str = "log_likelihood",
):
    TP_variable_list = [TP_dataset.get(variable) for variable in TP_dataset]
    number_of_variables = len(TP_variable_list)

    print([*TP_variable_list])
    return (
        apply_ufunc(
            TP_function,
            *TP_variable_list,
            input_core_dims=[[loop_dimension]] * number_of_variables,
            output_core_dims=[["log_pressure", loop_dimension]],
            kwargs=dict(pressures=log_pressures),
            keep_attrs=True,
        )
        .transpose(loop_dimension, "log_pressure")
        .assign_coords(dict(log_pressure=log_pressures))
        .rename("T")
        .pint.quantify(output_temperature_unit)
    )


def evaluate_model_spectra_from_dataset(
    free_parameter_dataset: Dataset,
    parameter_filepath: Pathlike,
    output_flux_unit: Unit | str = "ergs / second / cm**3",
    loop_dimension: str = "log_likelihood",
):
    free_variable_list = [
        free_parameter_dataset.get(variable) for variable in free_parameter_dataset
    ]
    number_of_variables = len(free_variable_list)

    prepped_inputs, binned_wavelengths = prep_inputs_for_model(parameter_filepath)

    def evaluate_model_spectrum_with_parameters_only(*model_parameters):
        print(f"{model_parameters=}")
        evaluated_model = np.asarray(
            evaluate_model_spectrum(
                model_function=prepped_inputs["model_function"],
                observation=prepped_inputs["observation"],
                model_parameters=model_parameters,
            ),
            dtype=np.float64,
        )
        return evaluated_model

    vectorized_evaluate = np.vectorize(
        evaluate_model_spectrum_with_parameters_only,
        otypes=[np.ndarray],
    )
    return vectorized_evaluate(*free_variable_list)

    return (
        apply_ufunc(
            vectorized_evaluate,
            *free_variable_list,
            input_core_dims=[[loop_dimension]] * number_of_variables,
            output_core_dims=[[loop_dimension]],
            # output_core_dims=[["wavelength", loop_dimension]],
            keep_attrs=True,
        )
        # .transpose(loop_dimension, "wavelength")
        # .assign_coords(dict(wavelength=binned_wavelengths))
        .rename("flux").pint.quantify(output_flux_unit)
    )


def test_vectorized_evaluation(
    free_parameter_dataset: Dataset,
    parameter_filepath: Pathlike,
    max_spectra_count: int = 10,
):
    free_variable_list = [
        free_parameter_dataset.get(variable) for variable in free_parameter_dataset
    ]
    number_of_variables = len(free_variable_list)

    prepped_inputs, binned_wavelengths = prep_inputs_for_model(parameter_filepath)

    def evaluate_model_spectrum_with_parameters_only(*model_parameters):
        return evaluate_model_spectrum(
            model_function=prepped_inputs["model_function"],
            observation=prepped_inputs["observation"],
            model_parameters=model_parameters,
        )

    data_arrays = [
        free_parameter_dataset.get(variable).to_numpy()[:max_spectra_count]
        for variable in free_parameter_dataset
    ]

    vectorized_evaluate = np.vectorize(
        evaluate_model_spectrum_with_parameters_only,
        # cache=True,
        # otypes=[np.float64],
        otypes=[np.ndarray],
        # otypes=[np.ndarray] * max_spectra_count,
        # signature="(p)->(w)",
    )

    return vectorized_evaluate(*data_arrays)


if __name__ == "__main__":
    RESULTS_DIRECTORY_2M2236 = RESULTS_DIRECTORY / "2M2236"
    run_directories = load_results_directories(RESULTS_DIRECTORY_2M2236)

    for run_name, run_filepath_dict in run_directories.items():
        dynesty_results = load_dynesty_results(
            RESULTS_DIRECTORY_2M2236 / run_name / run_filepath_dict["fitting_results"]
        )
        derived_parameters = load_derived_parameters(
            RESULTS_DIRECTORY_2M2236
            / run_name
            / run_filepath_dict["derived_fit_parameters"]
        )

        is_important_enough = make_filter_of_dynesty_samples_by_importance(
            dynesty_results, importance_weight_percentile=0.95
        )
        parameter_samples = compile_dynesty_parameters(
            dynesty_results, derived_parameters
        )[is_important_enough]
        log_likelihoods = dynesty_results.logl[is_important_enough]

        parsing_kwargs_specific_to_2M2236_runs = (
            dict(delimiter=" ") if "HK" in run_name else dict(delimiter="\t")
        )
        results_dataset = make_dataset_from_APOLLO_parameter_file(
            RESULTS_DIRECTORY_2M2236, **parsing_kwargs_specific_to_2M2236_runs
        )
        MLE_parameters = calculate_MLE(results_dataset)

        TP_function_name = get_TP_function_from_APOLLO_parameter_file(
            RESULTS_DIRECTORY_2M2236, **parsing_kwargs_specific_to_2M2236_runs
        )
        TP_function = getattr(TP_functions, TP_function_name)
        TP_dataset = extract_dataset_subset_by_parameter_group(
            results_dataset, group_name="Atm"
        )
        # %%
        log_pressures = np.linspace(-4, 2.5, num=71)
        TP_profile_dataset = evaluate_TP_functions_from_dataset(
            TP_dataset, log_pressures
        )
        # %%
        TP_profile_dataset

        # %%
        free_parameter_dataset = extract_free_parameters_from_dataset(results_dataset)

        MLE_parameter_file = (
            RESULTS_DIRECTORY_2M2236 / run_name / run_filepath_dict["MLE_parameters"]
        )
        # %%
        free_parameter_dataset
        # %%
        # test_vectorized_evaluation(free_parameter_dataset, MLE_parameter_file)
        # %%
        with open(Path.cwd() / "last_500_modelspectra.pkl", "rb") as pickle_file:
            model_spectrum_dataset = pickle.load(pickle_file)
        # %%
        # model_spectrum_dataset = evaluate_model_spectra_from_dataset(
        #    free_parameter_dataset.tail(log_likelihood=500), MLE_parameter_file
        # )
        # %%
        model_spectrum_dataset.astype(np.float64)

        # %%
        print(f"{np.shape(model_spectrum_dataset)=}")
        print(model_spectrum_dataset[0])
        # %%
        # with open(Path.cwd() / "last_500_modelspectra.pkl", "wb") as pickle_file:
        #    pickle.dump(model_spectrum_dataset, pickle_file)
        # model_spectrum_dataset.to_netcdf(Path.cwd() / "last_500_modelspectra.nc")
# %%
if False:
    JHK_contributions = {}
    for directory, object_header in zip(directories, object_headers):
        # JHK_contribution_file = directory+"/"+object_header+"_contributions.p"
        JHK_contribution_file = DIRECTORY_RESULTS_2M2236 / CONTRIBUTIONS_FILE_2M2236
        # JHK_contribution_file = "mock-L/input/mock-L.2022-12-08.forward-model.PLO.JHK_contributions.p"
        with open(JHK_contribution_file, "rb") as pickle_file:
            JHK_contributions[directory] = pickle.load(pickle_file)

    JHK_contributions
