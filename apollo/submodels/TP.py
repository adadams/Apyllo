import sys
from functools import partial
from os.path import abspath
from pathlib import Path
from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/APOLLO"
)
if APOLLO_DIRECTORY not in sys.path:
    sys.path.append(APOLLO_DIRECTORY)


from apollo.submodels.function_loader import FunctionLoader  # noqa: E402
from apollo.TP_functions import piette  # noqa: E402
from apollo.useful_internal_functions import format_yaml_from_template  # noqa: E402

TP_TEMPLATE_FILE = Path.cwd() / "apollo" / "submodels" / "TP.yaml"

with open(TP_TEMPLATE_FILE, "r") as file:
    TP_template = file.read().rstrip()


def temperature_node_string_formatter(log_pressure_level: float | int):
    return {
        "[LOG_PRESSURE]": str(log_pressure_level).replace(".", "p"),
    }


use_temperature_node_template = partial(
    format_yaml_from_template,
    template=TP_template,
    formatter=temperature_node_string_formatter,
)

list_of_pressures = [-4, -3, -2, -1, 0, 0.5, 1, 1.5, 2, 2.5]

# for item in map(use_temperature_node_template, list_of_pressures):
#    print(item)


# NOTE: Trying to use a dataclass to define this actually made it more complicated,
# so I went without.
class ThermalProfile(FunctionLoader):
    """
    Wraps a T-P function of some parameters and pressure.
    Returns a profile of T(P) when called.
    """

    def __repr__(self) -> str:
        return (
            f"Thermal profile function '{self.function.__name__}'"
            + f" with arguments: {self._loaded_arguments}"
        )

    def __call__(self, pressures: float | Sequence[float]) -> NDArray[np.float64]:
        return self._loaded_function(pressures)


PietteProfile = ThermalProfile(piette, *np.linspace(500, 2000, num=10))
print(PietteProfile)

pressures = np.linspace(-4, 2.5, num=71)
TP_profile = PietteProfile(pressures)

figure = plt.figure(figsize=(8, 6))
axis = figure.add_axes([0.1, 0.1, 0.8, 0.8])
axis.plot(TP_profile, pressures)
axis.invert_yaxis()
plt.show()
