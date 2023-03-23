

from dataclasses import dataclass
from datetime import date, datetime


@dataclass
class Metadata:
    NAME        : str
    RUN_DATE    : date
    RUN_DATETIME: datetime
