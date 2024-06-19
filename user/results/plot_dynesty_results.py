import pickle
import sys
from os.path import abspath
from pathlib import Path
from typing import Any, Final

from matplotlib.pyplot import style

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/APOLLO"
)
sys.path.append(APOLLO_DIRECTORY)


from apollo.convenience_types import Pathlike  # noqa: E402
from apollo.retrieval.dynesty.dynesty_plotting_functions import (  # noqa: E402
    make_combined_TP_profile_plot,
    make_corner_plots,
    make_multi_plots,
    plot_MLE_spectrum_of_one_run_against_different_data,
)
from apollo.retrieval.dynesty.parse_dynesty_outputs import (  # noqa: E402
    unpack_results_filepaths,
)
from user.results.process_dynesty_results import RESULTS_DIRECTORY  # noqa: E402
from user_directories import USER_DIRECTORIES  # noqa: E402

style.use(USER_DIRECTORIES["plots"] / "arthur.mplstyle")

if __name__ == "__main__":
    OBJECT_NAME: Final[str] = "2M2236"

    SPECIFIC_RESULTS_DIRECTORY: Final[Path] = RESULTS_DIRECTORY / OBJECT_NAME

    CONTRIBUTION_COMPONENTS: Final[list[str]] = ["h2o", "co", "co2", "ch4"]

    SHARED_CORNERPLOT_KWARGS: Final[dict[str, Any]] = {
        "weights": None,
        "parameter_range": None,
        "confidence": 0.95,
        "MLE_name": "MLE",
        "MLE_color": "gold",
    }

    PLOTTING_COLORS: Final[dict[str, str]] = {
        # "Ross458c": "crimson",
        "HK+JWST_2M2236_logg-free": "lightsalmon",
        "HK+JWST_2M2236_logg-free_cloudy": "orangered",
        "HK+JWST_2M2236_logg-normal": "darkorange",
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

    results_directory: dict[str, dict[str, Pathlike]] = unpack_results_filepaths(
        SPECIFIC_RESULTS_DIRECTORY
    )
    """
    run_name: str = ""
    comparison_run_name: str = "HK+JWST_2M2236_logg-free"
    plot_stuff: None = plot_MLE_spectrum_of_one_run_against_different_data(
        results_directory[run_name],
        results_directory[comparison_run_name]["data"],
    )

    with open(
        f"plot_stuff_{run_name}_against_{comparison_run_name}_data.pickle", "wb"
    ) as file:
        pickle.dump(plot_stuff, file)
    """
