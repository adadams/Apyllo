from functools import partial
from pathlib import Path
import sys

sys.path.append("../APOLLO")
from useful_internal_functions import format_from_template

TP_TEMPLATE_FILE = Path.cwd() / "submodels" / "TP.yaml"

with open(TP_TEMPLATE_FILE, "r") as file:
    TP_template = file.read().rstrip()


def temperature_node_string_formatter(log_pressure_level: float | int):
    return {
        "[LOG_PRESSURE]": str(log_pressure_level).replace(".", "p"),
    }


use_temperature_node_template = partial(
    format_from_template,
    template=TP_template,
    formatter=temperature_node_string_formatter,
)

list_of_pressures = [-4, -3, -2, -1, 0, 0.5, 1, 1.5, 2, 2.5]

for item in map(use_temperature_node_template, list_of_pressures):
    print(item)
