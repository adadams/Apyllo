from typing import Final, Optional, Sequence, TypedDict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.typing import ColorType
from numpy.typing import NDArray

from apollo.spectrum.band_bin_and_convolve import find_band_slices_from_wavelength_bins
from apollo.spectrum.read_spectral_data_into_xarray import (
    read_APOLLO_data_into_dictionary,
)


class ContourBlueprint(TypedDict):
    X: NDArray[np.float_]
    Y: NDArray[np.float_]
    Z: NDArray[np.float_]


class ContourAestheticsBlueprint(TypedDict):
    cmap: Optional[str]
    colors: Optional[ColorType]
    linestyles: Optional[str]
    levels: NDArray[np.float_]
    alpha: float  # 0 to 1
    zorder: int


def calculate_maximum_contour_value(
    contributions: dict[str, pd.DataFrame],
) -> float:
    COMPOSITE_COMPONENTS: Final[list[str]] = ["gas", "cloud", "total"]

    return np.log10(
        np.nanmax(
            [
                np.nanmax(contribution)
                for (species, contribution) in contributions.items()
                if species not in COMPOSITE_COMPONENTS
            ]
        )
    )


def get_band_slices_from_data(data_filepath: str) -> tuple[slice]:
    data: dict[str, NDArray[np.float_]] = read_APOLLO_data_into_dictionary(
        data_filepath
    )

    wavelength_bin_starts: NDArray[np.float_] = data["wavelength_bin_starts"]
    wavelength_bin_ends: NDArray[np.float_] = data["wavelength_bin_ends"]

    return find_band_slices_from_wavelength_bins(
        wavelength_bin_starts, wavelength_bin_ends
    )


def make_contour_inputs_from_contributions(
    contribution: pd.DataFrame,
    band_slice: slice,
    contribution_contour_resolution_reduction_factor: int,
) -> tuple[ContourBlueprint]:
    slices: tuple[slice] = (
        slice(None),
        slice(
            band_slice.start,
            band_slice.stop,
            contribution_contour_resolution_reduction_factor,
        ),
    )

    contour_x, contour_y = np.meshgrid(contribution.index, contribution.columns)
    contour_x = contour_x[slices]
    contour_y = contour_y[slices]

    contour_values: NDArray[np.float_] = np.log10((contribution.to_numpy().T)[slices])

    return ContourBlueprint(X=contour_x, Y=contour_y, Z=contour_values)


def draw_gas_contribution(
    contribution_ax: plt.Axes,
    contribution: pd.DataFrame,
    band_breaks: Sequence[Sequence[float]],
    outline_color: str,
    contribution_cmap: str,
    contributions_max: float,
    contribution_contour_resolution_reduction_factor=8,
) -> plt.Axes:
    contour_geometry: ContourBlueprint = make_contour_inputs_from_contributions(
        contribution, band_breaks, contribution_contour_resolution_reduction_factor
    )

    interior_aesthetics: ContourAestheticsBlueprint = {
        "cmap": contribution_cmap,
        "levels": contributions_max - np.array([4, 2, 0]),
        "alpha": 0.66,
        "zorder": 0,
    }

    outline_aesthetics: ContourAestheticsBlueprint = {
        "color": outline_color,
        "levels": contributions_max - np.array([2]),
        "alpha": 1,
        "zorder": 1,
    }

    contribution_ax.contourf(**contour_geometry, **interior_aesthetics)
    contribution_ax.contour(**contour_geometry, **outline_aesthetics)

    return contribution_ax


def draw_cloud_contribution(
    contribution_ax: plt.Axes,
    contribution: pd.DataFrame,
    band_breaks: Sequence[Sequence[float]],
    draw_outline: bool,  # if i == 0
    contribution_contour_resolution_reduction_factor=8,
) -> plt.Axes:
    contour_geometry: ContourBlueprint = make_contour_inputs_from_contributions(
        contribution, band_breaks, contribution_contour_resolution_reduction_factor
    )

    cloud_interior_aesthetics: ContourAestheticsBlueprint = {
        "colors": "k",
        "levels": [0.1, 0.75],
        "alpha": 0.75,
        "zorder": 2,
    }

    cloud_outline_aesthetics: ContourAestheticsBlueprint = {
        "colors": "#DDDDDD",
        "linestyles": "solid",
        "alpha": 1,
        "levels": [0.1],
        "zorder": 3,
    }

    contribution_ax.contourf(**contour_geometry, **cloud_interior_aesthetics)
    if draw_outline:
        contribution_ax.contour(**contour_geometry, **cloud_outline_aesthetics)

    return contribution_ax
