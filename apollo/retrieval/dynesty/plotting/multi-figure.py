from dataclasses import dataclass
from typing import Protocol, Sequence, TypedDict

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray


class MultiFigureBlueprint(TypedDict):
    contributions: dict[str, NDArray[np.float_]]
    list_of_band_boundaries: Sequence[Sequence[float]]
    band_breaks: Sequence[Sequence[float]]
    contributions_max: float
    wavelengths: Sequence[float]
    datas: Sequence[float]
    models: Sequence[float]
    errors: Sequence[float]
    model_title: str
    number_of_parameters: int


class DrawsOnAxis(Protocol):
    def __call__(self, axis: plt.Axes, *args, **kwargs) -> plt.Axes: ...


@dataclass
class PlotArtists:
    draw_spectrum: DrawsOnAxis
    draw_residuals: DrawsOnAxis
    draw_contributions: Sequence[DrawsOnAxis]


class MultiFigure(TypedDict):
    figure: plt.Figure
    gridspec: GridSpec


def setup_contribution_plots(
    contributions: dict[str, NDArray[np.float_]], data_filepath: Pathlike
):
    FIDUCIAL_SPECIES = "h2o"
    MINIMUM_BAND_BREAK_IN_MICRONS = 0.05
    COMPOSITE_COMPONENTS = ["gas", "cloud", "total"]

    contributions_max = np.log10(
        np.nanmax(
            [
                np.nanmax(contribution)
                for (species, contribution) in contributions.items()
                if species not in COMPOSITE_COMPONENTS
            ]
        )
    )

    wavelengths_low, wavelengths_high, data, data_error = (
        np.genfromtxt(data_filepath).T
    )[:4]
    wavelengths = (wavelengths_low + wavelengths_high) / 2

    band_breaks = np.r_[
        0,
        np.nonzero(
            (
                contributions[FIDUCIAL_SPECIES].index.to_numpy()[1:]
                - contributions[FIDUCIAL_SPECIES].index.to_numpy()[:-1]
            )
            > MINIMUM_BAND_BREAK_IN_MICRONS
        )[0]
        + 1,
        len(contributions[FIDUCIAL_SPECIES].index),
    ]
    # number_of_bands = len(band_breaks) - 1

    band_lower_wavelength_limit = np.asarray(
        [
            contributions[FIDUCIAL_SPECIES].index[band_break]
            for band_break in band_breaks[:-1]
        ]
    )
    band_upper_wavelength_limit = np.asarray(
        [
            contributions[FIDUCIAL_SPECIES].index[band_break - 1]
            for band_break in band_breaks[1:]
        ]
    )
    wavelength_ranges = band_upper_wavelength_limit - band_lower_wavelength_limit

    return {
        "contributions_max": contributions_max,
        "band_breaks": band_breaks,
        "wavelengths": wavelengths,
        "data": data,
        "errors": data_error,
        "band_lower_wavelength_limit": band_lower_wavelength_limit,
        "band_upper_wavelength_limit": band_upper_wavelength_limit,
        "wavelength_ranges": wavelength_ranges,
    }


def setup_multi_figure(
    number_of_contributions: int,
    number_of_bands: int,
    wavelength_ranges: Sequence[float],
) -> MultiFigure:
    figure = plt.Figure(figsize=(40, 30))

    gridspec = figure.add_gridspec(
        nrows=number_of_contributions + 2,
        ncols=number_of_bands,
        height_ratios=[4, 2] + ([3] * number_of_contributions),
        width_ratios=wavelength_ranges,
    )

    return {"figure": figure, "gridspec": gridspec}
