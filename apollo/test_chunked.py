import sys
from os.path import abspath
from pathlib import Path

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/Apyllo"
)
if APOLLO_DIRECTORY not in sys.path:
    sys.path.append(APOLLO_DIRECTORY)
from apollo.Apollo_chunked import (  # noqa: E402
    generate_emission_spectrum_from_APOLLO_file,
)

TEST_2M2236_FILEPATH: Path = Path(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/Results/2M2236/spectral_range/2M2236.Piette.G395H.retrieved.ensemble.2024-01-22.dat"
)

emission_flux_at_surface = generate_emission_spectrum_from_APOLLO_file(
    TEST_2M2236_FILEPATH
)

# print(emission_flux_at_surface)
