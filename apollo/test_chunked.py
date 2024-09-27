import sys
from pathlib import Path

import xarray as xr
from matplotlib import pyplot as plt

APOLLO_DIRECTORY = str(Path.cwd().absolute())
if APOLLO_DIRECTORY not in sys.path:
    sys.path.append(APOLLO_DIRECTORY)

from apollo.Apollo_chunked import (  # noqa: E402
    generate_emission_spectrum_from_APOLLO_file,
)
from apollo.Apollo_ProcessInputs import SpectrumWithWavelengths  # noqa: E402

TEST_2M2236_FILEPATH: Path = (
    Path.cwd() / "test_models" / "2M2236.potluck.test-model.desktop.dat"
)

observed_emission_flux: SpectrumWithWavelengths = (
    generate_emission_spectrum_from_APOLLO_file(TEST_2M2236_FILEPATH)
)

wavelength_coordinate: xr.Variable = xr.Variable(
    dims="wavelength",
    data=observed_emission_flux.wavelengths,
    attrs={"units": "microns"},
)

flux_variable: xr.Variable = xr.Variable(
    dims="wavelength",
    data=observed_emission_flux.flux,
    attrs={"units": "erg/s/cm^3", "long_name": r"Emission $F_\lambda$"},
)

observed_emission_flux_dataset: xr.Dataset = xr.Dataset(
    data_vars={"flux": flux_variable}, coords={"wavelength": wavelength_coordinate}
)

observed_emission_flux_dataset.to_netcdf("test_chunked.nc")

plt.plot(observed_emission_flux.wavelengths, observed_emission_flux.flux)
plt.savefig("test_chunked.pdf", bbox_inches="tight", dpi=150)
