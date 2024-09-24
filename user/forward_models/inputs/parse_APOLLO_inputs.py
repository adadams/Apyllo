from csv import reader, writer
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, TextIO


@dataclass(slots=True)
class APOLLOParameterEntry:
    MLE: float
    Mu: float
    Sigma: float
    Min: float
    Max: float
    Lower: float
    Higher: float


NUMBER_OF_PARAMETER_COLUMNS = len(APOLLOParameterEntry.__slots__)
print(f"{NUMBER_OF_PARAMETER_COLUMNS=}")


def is_parameter_line(line_name: str, line_entries: list[Any]):
    return len(line_entries) == NUMBER_OF_PARAMETER_COLUMNS


def is_parameter_header_line(line_name: str, line_entries: list[Any]):
    return line_name == "Parameter"


def parse_APOLLO_input_file(input_file: TextIO, delimiter="\t"):
    input_file_reader = reader(input_file, delimiter=delimiter)

    header_dict = {}
    for row in input_file_reader:
        line_name, *line_entries = [element.strip() for element in row if element]

        if is_parameter_header_line(line_name, line_entries):
            parameter_column_names = line_entries
            break
        else:
            header_dict[line_name] = (
                tuple(line_entries) if isinstance(line_entries, list) else line_entries
            )

    parameter_dict = {}
    parameter_names = []
    parameter_group_names = []
    parameter_group_slice_starts = []
    parameter_group_slice_ends = []
    for row in input_file_reader:
        line_name, *line_entries = [element.strip() for element in row if element]

        if not is_parameter_line(line_name, line_entries):
            # this line denotes the start of a new parameter group
            parameter_group_names.append(line_name)

            parameter_dict[line_name] = {}
            parameter_group_dict = parameter_dict[line_name]

            if line_entries:
                parameter_group_dict["options"] = tuple(line_entries)

            number_of_parameters_read_so_far = len(parameter_names)

            parameter_group_slice_starts.append(number_of_parameters_read_so_far)
            if len(parameter_group_slice_starts) > 1:
                # start of next slice will be end of previous slice
                parameter_group_slice_ends.append(number_of_parameters_read_so_far)
        else:
            # this line should represent a parameter within a group
            parameter_names.append(line_name)

            parameter_group_dict[line_name] = dict(
                zip(parameter_column_names, line_entries)
            )

    # the last group ends at the end of the parameters
    total_number_of_parameters = len(parameter_names)
    parameter_group_slice_ends.append(total_number_of_parameters)
    # parameter_group_slice_ends.append(None)

    parameter_group_slices = {
        group_name: slice(*indices)
        for group_name, indices in zip(
            parameter_group_names,
            zip(parameter_group_slice_starts, parameter_group_slice_ends),
        )
    }

    return dict(
        header=header_dict,
        parameters=parameter_dict,
        parameter_names=parameter_names,
        parameter_group_slices=parameter_group_slices,
    )


def write_parsed_input_to_output(
    header_dict: dict, parameter_dict: dict, output_file: TextIO
):
    output_file_writer = writer(output_file, delimiter="\t")

    header_lines = [
        [line_name, *line_entries] for line_name, line_entries in header_dict.items()
    ]
    output_file_writer.writerows(header_lines)

    first_group_dict = parameter_dict[list(parameter_dict.keys())[0]]
    first_parameter_dict = first_group_dict[list(first_group_dict.keys())[0]]
    parameter_column_names = list(first_parameter_dict.keys())

    output_file_writer.writerow(["Parameter"] + parameter_column_names)

    for group_name, group_dict in parameter_dict.items():
        group_header_elements = [group_name]
        if "options" in group_dict:
            group_header_elements += group_dict["options"]
            group_dict.pop("options")

        output_file_writer.writerow(group_header_elements)

        parameter_lines = [
            (
                [line_name, *line_entries.values()]
                if isinstance(line_entries, dict)
                else [line_name, *line_entries]
            )
            for line_name, line_entries in group_dict.items()
        ]

        output_file_writer.writerows(parameter_lines)


def change_properties_of_parameters(
    input_parameter_dict,
    property_changing_function: Callable,
    *func_args,
    **func_kwargs,
):
    parameter_object_dict = {}

    for group_name, group_dict in input_parameter_dict.items():
        parameter_object_dict[group_name] = {}
        group_object_dict = parameter_object_dict[group_name]

        # if "options" in group_dict:
        #    group_dict.pop("options")

        for parameter_name, parameter_dict in group_dict.items():
            group_object_dict[parameter_name] = (
                property_changing_function(
                    parameter_name, parameter_dict, *func_args, **func_kwargs
                )
                if parameter_name != "options"
                else group_dict[parameter_name]
            )

    return parameter_object_dict


def get_parameters_as_objects(input_parameter_dict):
    def make_APOLLO_input_parameter_objects(parameter_name, parameter_dict):
        return APOLLOParameterEntry(**parameter_dict)

    return change_properties_of_parameters(
        input_parameter_dict, make_APOLLO_input_parameter_objects
    )


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


if __name__ == "__main__":
    SAMPLE_INPUT_FILEPATH = (
        Path.cwd() / "models" / "inputs" / "2M2236.Piette.G395H.input.dat"
    )

    SAMPLE_OUTPUT_FILEPATH = (
        Path.cwd() / "models" / "inputs" / "2M2236.Piette.G395H.output.dat"
    )

    SAMPLE_TEST_FILEPATH = (
        Path.cwd() / "models" / "inputs" / "2M2236.Piette.G395H.output-dogfood.dat"
    )

    with open(SAMPLE_INPUT_FILEPATH, newline="") as input_file:
        parsed_input = parse_APOLLO_input_file(input_file, delimiter=" ")
        print(parsed_input)

    with open(SAMPLE_OUTPUT_FILEPATH, "w", newline="") as output_file:
        write_parsed_input_to_output(
            parsed_input["header"], parsed_input["parameters"], output_file
        )

    with open(SAMPLE_TEST_FILEPATH, "w", newline="") as output_dogfood_file:
        with open(SAMPLE_OUTPUT_FILEPATH, newline="") as input_file:
            parsed_input = parse_APOLLO_input_file(input_file)

        pprint(get_parameters_as_objects(parsed_input["parameters"]))

        write_parsed_input_to_output(
            parsed_input["header"], parsed_input["parameters"], output_dogfood_file
        )
