import sys
from os.path import abspath
from pathlib import Path
from typing import Final

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/APOLLO"
)
if APOLLO_DIRECTORY not in sys.path:
    sys.path.append(APOLLO_DIRECTORY)

from apollo.spectrum.read_spectral_data_into_xarray import (  # noqa: E402
    read_APOLLO_data_into_dataset,
)
from apollo.spectrum.Spectrum_measured_using_xarray import DataSpectrum  # noqa: E402
from dataset.IO import prep_and_save_dataset  # noqa: E402

# %%
DATA_DIRECTORY: Final = Path.home() / "Documents/Astronomy/2019/Retrieval/Code/Data"
MOCK_L_DATA_DIRECTORY: Final = DATA_DIRECTORY / "mock-L"
MOCK_L_TEST_DATA_FILE: Final = (
    MOCK_L_DATA_DIRECTORY / "mock-L.2022-12-08.forward-model.PLO.JHK.noised.dat"
)

JHK_BAND_NAMES: Final = ["J", "H", "Ks"]

JHK_data = read_APOLLO_data_into_dataset(
    MOCK_L_TEST_DATA_FILE, band_names=JHK_BAND_NAMES
).groupby("band")

# %%
JHK_data["J"]

# %%
_2M2236_DATA_DIRECTORY: Final = DATA_DIRECTORY / "2M2236"
_2M2236_DATA_FILE: Final = _2M2236_DATA_DIRECTORY / "2M2236_HK.dat"

HK_BAND_NAMES: Final = ["H", "Ks"]

HK_data = read_APOLLO_data_into_dataset(
    _2M2236_DATA_FILE, band_names=HK_BAND_NAMES, data_name="2M2236 b"
)
# %%
HK_dataspectrum: DataSpectrum = DataSpectrum(HK_data)
print(HK_dataspectrum)
# %%
H_data = HK_dataspectrum.get_bands("H")
K_data = HK_dataspectrum.get_bands("Ks")
# %%
H_dataspectrum = DataSpectrum(H_data)
K_dataspectrum = DataSpectrum(K_data)
# %%
print(H_dataspectrum)
print(K_dataspectrum)
# %%
downsampled_H = H_dataspectrum.down_resolve(convolve_factor=4, new_resolution=500)
# print(downsampled_H)

downsampled_K = K_dataspectrum.down_resolve(convolve_factor=4, new_resolution=500)
# %%
_2M2236_DATA_FILE: Final = _2M2236_DATA_DIRECTORY / "2M2236b_HK+G395H_R500.dat"

BAND_NAMES: Final = ["H", "Ks", "G395H_NRS1", "G395H_NRS2"]

data = read_APOLLO_data_into_dataset(
    _2M2236_DATA_FILE, band_names=BAND_NAMES, data_name="2M2236 b"
)

print(data)

prep_and_save_dataset(data, path=_2M2236_DATA_DIRECTORY / "2M2236b_HK+G395H_R500.nc")
