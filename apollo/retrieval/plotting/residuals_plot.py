from typing import Any, Final

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

PADDING: Final[float] = 0.025


def calculate_residuals(
    datas: NDArray[np.float_],
    models: NDArray[np.float_],
    errors: NDArray[np.float_],
) -> NDArray[np.float_]:
    return (models - datas) / errors


def generate_residual_plot_by_band(
    residual_axis: plt.Axes,
    wavelengths: NDArray[np.float_],
    residuals: NDArray[np.float_],
    plot_color: str,
    plot_kwargs: dict[str, Any] = dict(
        linewidth=3, linestyle="solid", alpha=1, zorder=2
    ),
    axhline_kwargs: dict[str, Any] = dict(
        y=0, color="#444444", linewidth=2, linestyle="dashed", zorder=-10
    ),
    # yaxis_label_fontsize: int | float = 26,
) -> plt.Axes:
    wave_min = np.min(wavelengths)
    wave_max = np.max(wavelengths)
    xmin = wave_min - PADDING * np.abs(wave_max - wave_min)
    xmax = wave_max + PADDING * np.abs(wave_max - wave_min)
    residual_axis.set_xlim([xmin, xmax])

    residual_ymin = np.nanmin(residuals)
    residual_ymax = np.nanmax(residuals)
    ymin = residual_ymin - PADDING * np.abs(residual_ymax - residual_ymin)
    ymax = residual_ymax + PADDING * np.abs(residual_ymax - residual_ymin)
    residual_axis.set_ylim([ymin, ymax])

    residual_axis.plot(wavelengths, residuals, color=plot_color, **plot_kwargs)

    residual_axis.axhline(**axhline_kwargs)

    residual_axis.minorticks_on()

    # if yaxis_label_fontsize:
    #    residual_axis.set_ylabel(r"Residual/$\sigma$", fontsize=yaxis_label_fontsize)

    return residual_axis
