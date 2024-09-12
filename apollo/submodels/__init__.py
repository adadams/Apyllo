import sys
from os.path import abspath

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/Apyllo"
)
if APOLLO_DIRECTORY not in sys.path:
    sys.path.append(APOLLO_DIRECTORY)
