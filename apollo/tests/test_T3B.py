import sys
from os.path import abspath

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/Apyllo"
)
sys.path.append(APOLLO_DIRECTORY)

from custom_types import Pathlike  # noqa: E402


def parse_into_sections(filepath: Pathlike) -> None:
    with open(filepath, "r") as file:
        lines = [line.strip() for line in file.readlines()]

    print(lines)


parse_into_sections("user/forward_models/inputs/T3B_risotto_demo.txt")
