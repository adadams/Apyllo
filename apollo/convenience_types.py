from pathlib import Path
from typing import IO, Protocol

from numpy.typing import ArrayLike

type Pathlike = Path | str
type Filelike = Pathlike | IO


class Measured(Protocol):
    """A thing that has errors."""

    lower_errors: ArrayLike
    upper_errors: ArrayLike

    @property
    def errors(self):
        return (self.lower_errors + self.upper_errors) / 2
