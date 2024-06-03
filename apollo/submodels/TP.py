from functools import partial
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from apollo.submodels.function_model import FunctionModel
from apollo.TP_functions import piette

list_of_pressures = [-4, -3, -2, -1, 0, 0.5, 1, 1.5, 2, 2.5]


PietteProfile: FunctionModel = ThermalProfile(piette, *np.linspace(500, 2000, num=10))
print(PietteProfile)

pressures: NDArray[np.float_] = np.linspace(-4, 2.5, num=71)
TP_profile: NDArray[np.float_] = PietteProfile(pressures)

figure: Figure = plt.figure(figsize=(8, 6))
axis: Axes = figure.add_axes([0.1, 0.1, 0.8, 0.8])
axis.plot(TP_profile, pressures)
axis.invert_yaxis()
plt.show()
