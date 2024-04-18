from pathlib import Path
import yaml

# from user.results.plot_dynesty_results import RESULTS_DIRECTORY

USER_DIRECTORY = Path.cwd() / "user"
USER_DIRECTORY_FILEPATH = USER_DIRECTORY / "directories.yaml"
with open(USER_DIRECTORY_FILEPATH, "r") as directory_file:
    USER_DIRECTORIES = {
        directory_name: USER_DIRECTORY / directory_path
        for directory_name, directory_path in yaml.safe_load(directory_file).items()
    }

""" BOILERPLATE CODE TO RESOLVE APOLLO PATH
from os.path import abspath
import sys

APOLLO_DIRECTORY = abspath(
    "/Users/arthur/Documents/Astronomy/2019/Retrieval/Code/APOLLO"
)
sys.path.append(APOLLO_DIRECTORY)
"""
