import sys
from pathlib import Path

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

emission_flux_at_surface: SpectrumWithWavelengths = (
    generate_emission_spectrum_from_APOLLO_file(TEST_2M2236_FILEPATH)
)

print(f"{emission_flux_at_surface=}")

plt.plot(emission_flux_at_surface.wavelengths, emission_flux_at_surface.flux)
plt.savefig("test_chunked.pdf", bbox_inches="tight", dpi=150)
