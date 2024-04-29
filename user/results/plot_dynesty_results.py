import sys
from os.path import abspath
from pathlib import Path

from matplotlib.pyplot import style

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/APOLLO"
)
sys.path.append(APOLLO_DIRECTORY)

from user.results.process_dynesty_results import RESULTS_DIRECTORY
from user_directories import USER_DIRECTORIES

style.use(USER_DIRECTORIES["plots"] / "arthur.mplstyle")

from apollo.dynesty_plotting_functions import (
    make_combined_TP_profile_plot,
    make_corner_plots,
    make_multi_plots,
)

if __name__ == "__main__":
    OBJECT_NAME: str = "2M2236"

    SPECIFIC_RESULTS_DIRECTORY: Path = RESULTS_DIRECTORY / OBJECT_NAME

    CONTRIBUTION_COMPONENTS: list[str] = ["h2o", "co", "co2", "ch4"]

    SHARED_CORNERPLOT_KWARGS = dict(
        weights=None,
        parameter_range=None,
        confidence=0.95,
        MLE_name="MLE",
        MLE_color="gold",
    )

    PLOTTING_COLORS: dict[str, str] = {
        # "Ross458c": "crimson",
        "HK+JWST_2M2236_logg-normal": "darkorange",
        "HK+JWST_2M2236_logg-free": "lightsalmon",
        "JWST-only_2M2236_logg-free": "mediumpurple",
        "JWST-only_2M2236_logg-normal": "indigo",
    }

    make_multi_plots(
        SPECIFIC_RESULTS_DIRECTORY, CONTRIBUTION_COMPONENTS, PLOTTING_COLORS
    )
    make_combined_TP_profile_plot(
        SPECIFIC_RESULTS_DIRECTORY, OBJECT_NAME, PLOTTING_COLORS
    )
    make_corner_plots(
        SPECIFIC_RESULTS_DIRECTORY, PLOTTING_COLORS, SHARED_CORNERPLOT_KWARGS
    )
