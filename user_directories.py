from pathlib import Path
import yaml

USER_PATH = Path.cwd() / "user"
USER_DIRECTORY_PATH = USER_PATH / "directories.yaml"
with open(USER_DIRECTORY_PATH, "r") as directory_file:
    USER_DIRECTORIES = {
        directory_name: USER_PATH / directory_path
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
