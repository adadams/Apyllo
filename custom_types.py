from pathlib import Path
from typing import IO

type Pathlike = Path | str
type Filelike = Pathlike | IO
