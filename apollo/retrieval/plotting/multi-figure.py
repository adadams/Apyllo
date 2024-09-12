import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from functools import partial
from os.path import abspath
from typing import Protocol, TypedDict

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/Apyllo"
)
sys.path.append(APOLLO_DIRECTORY)


class MultiFigureBlueprintButt(TypedDict):
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


class GridSpecBlueprint(TypedDict):
    nrows: int
    ncols: int
    height_ratios: Sequence[float]
    width_ratios: Sequence[float]


@dataclass
class MultiFigureBlueprint:
    figure_dimensions: tuple[int | float, int | float]
    number_of_rows: int
    number_of_columns: int
    height_ratios: Sequence[float]
    width_ratios: Sequence[float]

    @property
    def gridspec_kwargs(self) -> GridSpecBlueprint:
        return {
            "nrows": self.number_of_rows,
            "ncols": self.number_of_columns,
            "height_ratios": self.height_ratios,
            "width_ratios": self.width_ratios,
        }

    @property
    def figure_kwargs(self) -> dict[str, int | float]:
        return {"figsize": self.figure_dimensions}


def create_Arthurs_multi_figure(
    number_of_contributions: int,
    number_of_bands: int,
    wavelength_ranges: Sequence[float],
) -> MultiFigureBlueprint:
    return MultiFigureBlueprint(
        figure_dimensions=(40, 30),
        number_of_rows=number_of_contributions + 2,
        number_of_columns=number_of_bands,
        height_ratios=[4, 2] + ([3] * number_of_contributions),
        width_ratios=wavelength_ranges,
    )


def setup_multi_figure(
    multi_figure_blueprint: MultiFigureBlueprint,
) -> MultiFigure:
    figure: plt.Figure = plt.Figure(**multi_figure_blueprint.figure_kwargs)
    gridspec: GridSpec = figure.add_gridspec(**multi_figure_blueprint.gridspec_kwargs)
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
        axis_plotting_function: DrawsOnAxis = partial(
            plot_along_axis_dimension, axis_plotters=plot_artists_by_type
        )

        np.apply_along_axis(
            axis_plotting_function,
            axis=artist_dimension.value,
            arr=multi_figure.axis_array,
        )

    return multi_figure
