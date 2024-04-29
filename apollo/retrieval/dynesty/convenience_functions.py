from typing import Any, Sequence

from numpy import empty_like


def guess_default_units_from_parameter_names(
    parameter_names: Sequence[str],
) -> list[str]:
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
            guessed_unit = "dimensionless"

        guessed_units.append(guessed_unit)

    return guessed_units


def guess_default_string_formats_from_parameter_names(
    parameter_names: Sequence[str],
) -> list[str]:
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
    free_parameter_names: Sequence[str],
    free_parameter_group_slices: Sequence[slice],
    derived_parameter_names: Sequence[str] = [
        "Mass",
        "C/O",
        "[Fe/H]",
        "Teff",
    ],
) -> dict[str, Any]:
    parameter_names = free_parameter_names + derived_parameter_names

    parameter_units = guess_default_units_from_parameter_names(parameter_names)

    free_parameter_group_names = empty_like(free_parameter_names)
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
