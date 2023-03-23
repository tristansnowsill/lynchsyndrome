from dataclasses import dataclass
from enum import Enum, auto

from .reporting import EventMetadata


class CauseOfDeath(Enum):
    COLORECTAL_CANCER  = auto()
    ENDOMETRIAL_CANCER = auto()
    OVARIAN_CANCER     = auto()
    OTHER_CAUSES       = auto()


@dataclass
class DeathMetadata(EventMetadata):
    cause: CauseOfDeath
