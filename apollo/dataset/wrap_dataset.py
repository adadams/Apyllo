from abc import ABC
from typing import Any

from xarray import DataArray, Dataset


class UseXarray(ABC):
    # This is just saying, if you store things in an xarray dataset or dataarray,
    # then you can use them as attributes of the class you define.
    data: Dataset | DataArray

    def __getattr__(self, __name: str) -> Any:
        return self.data.__getattribute__(__name)
