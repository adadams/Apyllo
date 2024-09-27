import time
from os.path import isfile
from pathlib import Path
from typing import Final, Union

import numpy as np
import picaso.opacity_factory as opa_fac
from combine_alkalis import load_and_combine_alkalis

ORIGINAL_LUPU_RESOLUTION: Final[float] = 1_000_000.0


def run_picaso_opacity_factory(
    original_opacity_directory,
    save_name,
    list_of_molecules,
    minimum_wavelength,
    maximum_wavelength,
    new_resolution,
    alkali_directory="individual_file",
    function=opa_fac.insert_molecular_1460,
):
    new_opacity_database_filepath = f"{original_opacity_directory}/lupu_{minimum_wavelength}_{maximum_wavelength}_R{new_resolution}_{save_name}.db"

    if not isfile(new_opacity_database_filepath):
        opa_fac.build_skeleton(new_opacity_database_filepath)

        for molecule in list_of_molecules:
            start_time = time.time()
            print("Inserting: " + molecule)
            opa_fac.insert_molecular_1460(
                molecule,
                minimum_wavelength,
                maximum_wavelength,
                original_opacity_directory,
                new_opacity_database_filepath,
                new_R=new_resolution,
                alkali_dir=alkali_directory,
            )
            print(
                molecule
                + " inserts finished in :"
                + str((time.time() - start_time) / 60.0)[0:3]
                + " minutes"
            )

    else:
        print("Opacity database already exists at that filepath")

    return new_opacity_database_filepath


def prepare_opacity_table(
    opacity_db,
    molecule,
    minimum_wavelength,
    maximum_wavelength,
    new_resolution,
    save_name,
):
    molecules, pt_pairs = opa_fac.molecular_avail(opacity_db)
    if molecule not in molecules:
        print("Molecule not in the available molecules.")
        return None

    pressure_levels = np.unique(np.array(pt_pairs)[:, 1])
    number_of_pressures = len(pressure_levels)
    pressure_min = np.log10(np.min(pressure_levels))
    pressure_max = np.log10(np.max(pressure_levels))

    temperature_levels = np.unique(np.array(pt_pairs)[:, 2])
    number_of_temperatures = len(temperature_levels)
    temperature_min = np.log10(np.min(temperature_levels))
    temperature_max = np.log10(np.max(temperature_levels))

    cur, conn = opa_fac.open_local(opacity_db)
    cur.execute(
        """SELECT molecule,ptid,pressure,temperature,opacity
               FROM molecular
               WHERE molecule = ?""",
        (molecule,),
    )
    data = cur.fetchall()

    actual_data = np.asarray([tup[-1] for tup in data])[::-1]

    transposed_data = np.einsum(
        "tpn->ptn",
        np.reshape(
            actual_data,
            (number_of_temperatures, number_of_pressures, np.shape(actual_data)[-1]),
        ),
    )[::-1, ::-1, ::-1]

    final_data = np.reshape(
        transposed_data,
        (number_of_pressures * number_of_temperatures, np.shape(transposed_data)[-1]),
    )
    number_of_spectral_elements = np.shape(final_data)[-1]

    actual_resolution_of_tables: float = ORIGINAL_LUPU_RESOLUTION / int(
        ORIGINAL_LUPU_RESOLUTION / new_resolution
    )

    header_info = [
        (number_of_pressures, "d"),
        (pressure_min, "1.1f"),
        (pressure_max, "1.1f"),
        (number_of_temperatures, "d"),
        (temperature_min, "1.6f"),
        (temperature_max, "1.6f"),
        (number_of_spectral_elements, "d"),
        (minimum_wavelength, "2.5f"),
        (maximum_wavelength, "2.5f"),
        (actual_resolution_of_tables, "6.2f"),
    ]
    header_string = " ".join(["{:{}}".format(*entry) for entry in header_info])

    np.savetxt(
        f"{molecule.lower()}.{save_name}.dat",
        final_data,
        delimiter=" ",
        header=header_string,
        comments="",
    )

    return final_data


def generate_opacity_files(
    original_opacity_directory: Union[str, Path],
    save_name: Union[str, Path],
    list_of_molecules: list[str],
    minimum_wavelength: float,
    maximum_wavelength: float,
    new_resolution: float,
    combine_alkalis: bool = False,
) -> list[Path]:
    new_opacity_database_filepath = run_picaso_opacity_factory(
        original_opacity_directory,
        save_name,
        list_of_molecules,
        minimum_wavelength,
        maximum_wavelength,
        new_resolution,
        alkali_directory="individual_file",
        function=opa_fac.insert_molecular_1460,
    )

    output_opacity_filepaths = []
    for molecule in list_of_molecules:
        if isfile(f"{molecule.lower()}.{save_name}.dat"):
            print(f"Opacity table for {molecule} already exists.")

        else:
            print(f"Generating table for {molecule}.")
            prepare_opacity_table(
                new_opacity_database_filepath,
                molecule,
                minimum_wavelength,
                maximum_wavelength,
                new_resolution,
                save_name,
            )

        output_opacity_filepaths.append(
            Path.cwd() / f"{molecule.lower()}.{save_name}.dat"
        )

    if combine_alkalis:
        alkali_output_filename = f"Lupu_alk.{save_name}.dat"

        load_and_combine_alkalis(
            Na_opacity_filepath=Path.cwd() / f"na.{save_name}.dat",
            K_opacity_filepath=Path.cwd() / f"k.{save_name}.dat",
            output_filename=alkali_output_filename,
        )

        output_opacity_filepaths.append(alkali_output_filename)

    return output_opacity_filepaths


# To do: store a csv file where when you make a new set of opacities,
# store the specs indexed by the save name.
