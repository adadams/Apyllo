import sys
from os.path import abspath
from pathlib import Path
from typing import Sequence

from xarray import Dataset

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/APOLLO"
)
sys.path.append(APOLLO_DIRECTORY)

from apollo.retrieval.dynesty.parse_dynesty_outputs import (  # noqa: E402
    unpack_results_filepaths,  # noqa: E402
)
from user.results.latex_table_writer import (  # noqa: E402
    format_percentiles_and_MLEs_and_store_as_attributes,
    load_results_dataset_and_store_formatted_table_text,
    stitch_rows_across_runs,
)
from user.results.process_dynesty_results import RESULTS_DIRECTORY  # noqa: E402


def print_tables_in_Per_style(header_name: str, rows: Sequence[str]) -> None:
    number_of_separators: int = rows[0].count("&")

    header_line: str = rf"{header_name} {"& "*number_of_separators} \\"

    print(header_line)
    print(r"\hline")

    for row in rows:
        print(row)

    print(r"\hline")

    return None


if __name__ == "__main__":
    OBJECT_NAME: str = "2M2236"

    SPECIFIC_RESULTS_DIRECTORY: Path = RESULTS_DIRECTORY / OBJECT_NAME

    results_filepath_dictionary: dict[str, dict[str, Path]] = unpack_results_filepaths(
        SPECIFIC_RESULTS_DIRECTORY
    )

    samples_datasets_with_table_entries: dict[str, Dataset] = {
        run_name: load_results_dataset_and_store_formatted_table_text(
            results_filepaths["samples_dataset"],
            format_percentiles_and_MLEs_and_store_as_attributes,
        )
        for run_name, results_filepaths in results_filepath_dictionary.items()
    }
    print(f"{list(samples_datasets_with_table_entries.keys())=}")

    fundamental_parameter_names: list[str] = ["Rad", "Log(g)", "Teff", "C/O", "[Fe/H]"]
    fundamental_parameter_titles: list[str] = [
        r"$R/R_{\rm Jup}$",
        r"$\log[g/({\rm cm s}^{-2})]$",
        r"$T_{\rm eff}$ (K)",
        r"C/O",
        r"Metallicity",
    ]

    fundamental_datasets: dict[str, Dataset] = {
        run_name: samples_dataset_with_table_entries.get(fundamental_parameter_names)
        for run_name, samples_dataset_with_table_entries in samples_datasets_with_table_entries.items()
    }

    fundamental_value_rows: list[str] = stitch_rows_across_runs(
        *fundamental_datasets.values(),
        parameter_print_names=fundamental_parameter_titles,
    )

    print_tables_in_Per_style("Fundamental", fundamental_value_rows)

    molecule_names: list[str] = ["h2o", "co", "co2", "ch4", "h2s", "Lupu_alk", "nh3"]
    molecule_titles: list[str] = [
        r"H$_2$O",
        r"CO",
        r"CO$_2$",
        r"CH$_4$",
        r"H$_2$S",
        r"Na$+$K",
        r"NH$_3$",
    ]

    molecular_datasets: dict[str, Dataset] = {
        run_name: samples_dataset_with_table_entries.get(molecule_names)
        for run_name, samples_dataset_with_table_entries in samples_datasets_with_table_entries.items()
    }

    molecular_rows: list[str] = stitch_rows_across_runs(
        *molecular_datasets.values(), parameter_print_names=molecule_titles
    )

    print_tables_in_Per_style(r"Gases (log$_{10}$ abundance)", molecular_rows)
