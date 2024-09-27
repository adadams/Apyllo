import sys
from pathlib import Path

"""
from beartype import BeartypeConf
from beartype.claw import beartype_all, beartype_this_package

beartype_this_package()  # <-- raise exceptions in your code
beartype_all(
    conf=BeartypeConf(violation_type=UserWarning)
)  # <-- omit warnings from other code
"""

APOLLO_DIRECTORY = Path.cwd().absolute()
if str(APOLLO_DIRECTORY) not in sys.path:
    sys.path.append(str(APOLLO_DIRECTORY))
