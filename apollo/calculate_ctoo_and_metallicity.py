from enum import Enum
from typing import Final, Sequence

import numpy as np
from numpy.typing import NDArray

molecular_weights: dict[str, float] = {
    "h2o": 16.0,
    "ch4": 12.0,
    "co": 28.0,
    "co2": 44.0,
    "nh3": 14.0,
    "h2s": 32.0,
    "Burrows_alk": 24.0,
    "Lupu_alk": 24.0,
    "na": 23.0,
    "k": 39.0,
    "crh": 52.0,
    "feh": 56.0,
    "tio": 64.0,
    "vo": 67.0,
    "hcn": 26.0,
    "n2": 28.0,
    "ph3": 31.0,
}
MOLECULAR_WEIGHTS: Final[Enum] = Enum("MolecularWeights", molecular_weights)

carbon_atoms_per_molecule: dict[str, int] = {"ch4": 1, "co": 1, "co2": 1, "hcn": 1}
CARBON_COUNT: Final[Enum] = Enum("CarbonCount", carbon_atoms_per_molecule)


oxygen_atoms_per_molecule: dict[str, int] = {
    "h2o": 1,
    "co": 1,
    "co2": 2,
    "tio": 1,
    "vo": 1,
}
OXYGEN_COUNT: Final[Enum] = Enum("CarbonCount", oxygen_atoms_per_molecule)


def calculate_CtoO_and_metallicity(
    list_of_gases: Sequence[float], gas_logabundances: Sequence[float]
) -> NDArray[np.float64]:
    carbon = 0.0
    oxygen = 0.0
    metals = 0.0

    ccompounds, cmult = zip(*carbon_atoms_per_molecule.items())
    ocompounds, omult = zip(*oxygen_atoms_per_molecule.items())
    zcompounds, zmult = zip(*molecular_weights.items())

    for i in range(0, len(ccompounds)):
        if ccompounds[i] in list_of_gases:
            j = list_of_gases.index(ccompounds[i])
            carbon = carbon + cmult[i] * (
                10 ** gas_logabundances[j]
            )  # -1 because of hydrogen
    for i in np.arange(0, len(ocompounds)):
        if ocompounds[i] in list_of_gases:
            j = list_of_gases.index(ocompounds[i])
            oxygen = oxygen + omult[i] * (10 ** gas_logabundances[j])
    for i in np.arange(0, len(zcompounds)):
        if zcompounds[i] in list_of_gases:
            j = list_of_gases.index(zcompounds[i])
            metals = metals + zmult[i] * (10 ** gas_logabundances[j])

    ctoo = carbon / oxygen
    fetoh = np.log10(metals / 0.0196)

    return np.array([ctoo, fetoh])
