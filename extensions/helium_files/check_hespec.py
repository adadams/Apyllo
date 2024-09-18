from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

WORKING_DIRECTORY = Path.cwd() / "extensions" / "helium_files"

hespec = np.loadtxt(WORKING_DIRECTORY / "hespec.dat")

gas_opacity_directory = Path("/home") / "Research" / "Opacities_0v10" / "gases"
existing_he_nir = np.loadtxt(gas_opacity_directory / "he.nir.dat", skiprows=1).reshape(
    (18, 36, 21204)
)

first_row_existing = existing_he_nir[0, 0, :]

new_he_nir = np.loadtxt(
    Path("/media/gba8kj/ResearchStorage/Opacities_0v10/gases") / "he.nir.dat",
    skiprows=1,
).reshape((18, 36, 21204))

first_row_new = new_he_nir[0, 0, :]


plt.semilogy(first_row_existing, linewidth=5, c="k")
plt.semilogy(first_row_new)
plt.show()

existing_hhe_nir = np.loadtxt(
    gas_opacity_directory / "h2he.nir.dat", skiprows=1
).reshape((18, 36, 21204))

first_row_existing_hhe = existing_hhe_nir[0, 0, :]

new_hhe_nir = np.loadtxt(
    Path.cwd() / "extensions" / "helium_files" / "h2he.nir.dat", skiprows=1
).reshape((18, 36, 21204))

first_row_new_hhe = new_hhe_nir[0, 0, :]


plt.semilogy(first_row_existing_hhe, linewidth=5, c="k")
plt.semilogy(first_row_new_hhe)
plt.show()
