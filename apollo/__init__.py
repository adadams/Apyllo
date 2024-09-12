import sys
from os.path import abspath

"""
from beartype import BeartypeConf
from beartype.claw import beartype_all, beartype_this_package

beartype_this_package()  # <-- raise exceptions in your code
beartype_all(
    conf=BeartypeConf(violation_type=UserWarning)
)  # <-- omit warnings from other code
"""

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/Apyllo"
)
if APOLLO_DIRECTORY not in sys.path:
    sys.path.append(APOLLO_DIRECTORY)
