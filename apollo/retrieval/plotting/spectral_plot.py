from collections.abc import Iterable, Sequence
from typing import Any, Final

import numpy as np
from matplotlib import pyplot as plt

PADDING: Final[float] = 0.025


def generate_spectrum_plot_by_band(
    spectrum_axis: Iterable[plt.Axes],
    wavelengths: Sequence[float],
    data: Sequence[float],
    model: Sequence[float],
    error: Sequence[float],
    model_title: str,
    plot_color: str,
    errorbar_kwargs: dict[str, Any] = {
        "color": "#444444",
        "fmt": "x",
        "linewidth": 0,
        "elinewidth": 2,
        "alpha": 1,
        "zorder": -3,
    },
    spectrum_kwargs: dict[str, Any] = {
        "linewidth": 3,
        "linestyle": "solid",
        "alpha": 1,
        "zorder": 2,  # 2-j
    },
) -> plt.Axes:
    wavelength_min: float = np.min(wavelengths)
    wavelength_max: float = np.max(wavelengths)
    wavelength_range: float = wavelength_max - wavelength_min

    xmin: float = wavelength_min - PADDING * wavelength_range
    xmax: float = wavelength_max + PADDING * wavelength_range
    spectrum_axis.set_xlim([xmin, xmax])

    spectrum_axis.errorbar(wavelengths, data, error, **errorbar_kwargs)

    spectrum_axis.plot(
        wavelengths,
        model,
        color=plot_color,
        label=model_title,
        **spectrum_kwargs,
    )

    return spectrum_axis


def plot_alkali_lines_on_spectrum(spectrum_axis: plt.Axes) -> plt.Axes:
    ALKALI_LINE_POSITIONS: Final[tuple[float]] = tuple(
        1.139,
        1.141,
        1.169,
        1.177,
        1.244,
        1.253,
        1.268,
    )

    wavelength_min, wavelength_max = spectrum_axis.get_xlim()
    assert all(
        wavelength_min <= ALKALI_LINE_POSITIONS <= wavelength_max
    ), "At least one of the alkali lines may fall outside your plotted wavelength range."

    [
        spectrum_axis.axvline(
            line_position_in_microns,
            linestyle="dashed",
            linewidth=1.5,
            zorder=-10,
            color="#888888",
        )
        for line_position_in_microns in ALKALI_LINE_POSITIONS
    ]

    y_text = spectrum_axis.get_ylim()[0] + 0.1 * np.diff(spectrum_axis.get_ylim())
    spectrum_axis.text(
        (1.169 + 1.177) / 2, y_text, "KI", fontsize=20, horizontalalignment="center"
    )
    spectrum_axis.text(
        (1.244 + 1.253) / 2, y_text, "KI", fontsize=20, horizontalalignment="center"
    )
    spectrum_axis.text(1.268, y_text, "NaI", fontsize=20, horizontalalignment="center")

    return spectrum_axis
