import sys
from dataclasses import dataclass
from enum import Enum
from functools import partial
from os.path import abspath
from typing import Protocol, Sequence, TypedDict

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/APOLLO"
)
sys.path.append(APOLLO_DIRECTORY)


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


def plot_along_axis_dimension(
    axes: Sequence[plt.Axes],
    axis_plotters: Sequence[DrawsOnAxis],
) -> list[plt.Axes]:
    for axis, axis_plotter in zip(axes, axis_plotters):
        axis_plotter(axis)

    return axes


@dataclass
class MultiFigure:
    figure: plt.Figure
    gridspec: GridSpec
    axis_array: NDArray[np.object_]


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

    axis_array: NDArray[np.object_] = gridspec.subplots()

    return MultiFigure(figure=figure, gridspec=gridspec, axis_array=axis_array)


class RowOrColumn(Enum):
    ROW = 0
    COLUMN = 1


def draw_on_multi_figure(
    multi_figure: MultiFigure,
    plot_artists_by_type_by_band: Sequence[Sequence[PlotArtists]],
    artist_dimension: RowOrColumn = RowOrColumn.COLUMN,
) -> MultiFigure:
    # Set axis each plot artist should iterate over (row vs. column).
    # For each axis, iterate over each band, which is the dimension
    # you didn't pick for iteration.

    for plot_artists_by_type in plot_artists_by_type_by_band:
        np.apply_along_axis(
            partial(plot_along_axis_dimension, axis_plotters=plot_artists_by_type),
            axis=artist_dimension.value,
            arr=multi_figure.axis_array,
        )

    return multi_figure
