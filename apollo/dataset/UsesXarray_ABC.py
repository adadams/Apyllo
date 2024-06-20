from abc import ABC
from typing import Self

from xarray import DataArray, Dataset


class UsesXarray(ABC):
    """
    This class uses an xarray dataset to hold attributes (data).
    It doesn't require every attribute to be in the dataset.
    However, please specify a constructor (from_elements) that
    takes in attributes and stores them in a dataset.
    """

    data: Dataset

    def __getattr__(self, __name: str) -> DataArray:
        return self.data.__getattribute__(__name)

    @classmethod
    def from_elements(cls, *args, **kwargs) -> Self:
        raise NotImplementedError
