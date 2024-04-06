from functools import partial
from pathlib import Path
import sys

sys.path.append("../APOLLO")
from useful_internal_functions import format_from_template

GASES_TEMPLATE_FILE = Path.cwd() / "submodels" / "gases.yaml"

with open(GASES_TEMPLATE_FILE, "r") as file:
    gas_template = file.read().rstrip()


def gas_string_formatter(input_species_name: str):
    return {
        "[GAS_NAME]": input_species_name.upper(),
        "[GAS_NAME_IN_LOWERCASE]": input_species_name.lower(),
    }


use_gas_template = partial(
    format_from_template, template=gas_template, formatter=gas_string_formatter
)

list_of_molecules = ["h2o", "co", "co2", "nh3", "h2s"]

for item in map(use_gas_template, list_of_molecules):
    print(item)
