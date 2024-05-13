import sys
from os.path import abspath
from pathlib import Path

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/APOLLO"
)
if APOLLO_DIRECTORY not in sys.path:
    sys.path.append(APOLLO_DIRECTORY)

from apollo.data.read_data_into_xarray import (
    read_APOLLO_data_into_dataset,  # noqa: E402
)

DATA_DIRECTORY = Path.home() / "Documents/Astronomy/2019/Retrieval/Code/Data"
MOCK_L_DATA_DIRECTORY = DATA_DIRECTORY / "mock-L"
MOCK_L_TEST_DATA_FILE = (
    MOCK_L_DATA_DIRECTORY / "mock-L.2022-12-08.forward-model.PLO.JHK.noised.dat"
)
# %%
JHK_band_names = ["J", "H", "Ks"]
# %%
test_data = read_APOLLO_data_into_dataset(
    MOCK_L_TEST_DATA_FILE, band_names=JHK_band_names
).groupby("band")
# %%
test_data["J"]
# %%
test_data["H"]
# %%
test_data["Ks"]
# %%

JHK_data = read_APOLLO_data_into_dataset(
    MOCK_L_TEST_DATA_FILE, band_names=JHK_band_names
).groupby("band")
# %%
JHK_data
# %%

# print(f"mock-L: {JHK_data}")

_2M2236_DATA_DIRECTORY = DATA_DIRECTORY / "2M2236"
_2M2236_DATA_FILE = _2M2236_DATA_DIRECTORY / "2M2236_HK.dat"

HK_band_names = ["H", "Ks"]

HK_data = read_APOLLO_data_into_dataset(_2M2236_DATA_FILE, HK_band_names)
H_data = HK_data["H"]
K_data = HK_data["Ks"]
# print(f"2M2236 b: {HK_data}")

downsampled_H = H_data.down_resolve(convolve_factor=4, new_resolution=500)
# print(downsampled_H)

downsampled_K = K_data.down_resolve(convolve_factor=4, new_resolution=500)
# print(downsampled_K)

# %%
NIRSpec_band_names = ["NRS1", "NRS2"]

RYAN_2M2236_DATA_FILE = (
    _2M2236_DATA_DIRECTORY / "Ryan" / "2M2236b_NIRSpec_G395H_R500_APOLLO.dat"
)

Ryan_data = read_APOLLO_data(
    RYAN_2M2236_DATA_FILE,
    NIRSpec_band_names,
    post_polars_processing_functions=(reduce_float_precision_to_correct_band_finding,),
)
# print(f"2M2236 b from Ryan: {Ryan_data}")

merged_2M2236b_data_at_R500 = merge_bands_into_single_dataframe(
    [
        downsampled_H.dataframe,
        downsampled_K.dataframe,
    ]
    + [band.dataframe for band in Ryan_data.values()]
)

print(
    merge_bands_into_single_dataframe(
        [
            downsampled_H.dataframe,
            downsampled_K.dataframe,
        ]
        + [band.dataframe for band in Ryan_data.values()]
    )
)

output_filepath_as_csv = Path(DATA_DIRECTORY / "2M2236b_HK+G395H_R500.csv")

write_Polars_data_to_APOLLO_file(
    merged_2M2236b_data_at_R500,
    output_filepath_as_csv,
    write_csv_kwargs=dict(),
    pre_save_processing_functions=(reduce_float_precision_to_correct_band_finding,),
)

output_filepath_as_dat: Path = Path(DATA_DIRECTORY / "2M2236b_HK+G395H_R500.dat")

write_Polars_data_to_APOLLO_file(
    merged_2M2236b_data_at_R500,
    output_filepath_as_dat,
    pre_save_processing_functions=(reduce_float_precision_to_correct_band_finding,),
)
