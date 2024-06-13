import sys
from os.path import abspath

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/APOLLO"
)
if APOLLO_DIRECTORY not in sys.path:
    sys.path.append(APOLLO_DIRECTORY)

from apollo.submodels.TP import modified_piette  # noqa: E402

print(modified_piette)
print(modified_piette.load_parameters(T_m4=500, T_m3=1000, T_m2=1500))
