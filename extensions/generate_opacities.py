from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Final

import msgspec
from Lupu_opacity_reader import generate_opacity_files

OPACITY_DIRECTORY: Final[Path] = Path("/Volumes/ResearchStorage") / "Lupu_opacities"


class OpacityConfiguration(msgspec.Struct):
    save_name: str
    list_of_molecules: list[str]
    minimum_wavelength: float
    maximum_wavelength: float
    new_resolution: float
    combine_alkalis: bool = False


def read_configuration_file_into_kwargs() -> OpacityConfiguration:
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("configuration_file")

    args: Namespace = parser.parse_args()

    try:
        configuration_filepath: Path = Path(args.configuration_file)
    except FileNotFoundError:
        print("Could not find config file.")

    return msgspec.toml.decode(
        configuration_filepath.read_text(), type=OpacityConfiguration
    )


if __name__ == "__main__":
    opacity_configuration: OpacityConfiguration = read_configuration_file_into_kwargs()

    new_opacity_files: list[Path] = generate_opacity_files(
        original_opacity_directory=OPACITY_DIRECTORY,
        **msgspec.structs.asdict(opacity_configuration),
    )
