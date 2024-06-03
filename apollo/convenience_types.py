from pathlib import Path
from typing import IO, Protocol

import numpy as np
from numpy.typing import NDArray

type Pathlike = Path | str
type Filelike = Pathlike | IO


class Measured(Protocol):
    """A thing that has errors."""

    lower_errors: NDArray[np.float_]
    upper_errors: NDArray[np.float_]

    @property
    def errors(self):
        return (self.lower_errors + self.upper_errors) / 2
