from dataclasses import dataclass
from enum import Enum, auto
from typing import Set

from .reporting import EventMetadata


class SurveillanceSite(Enum):
    COLORECTUM = auto()
    """Surveillance of the colorectum (large bowel including rectum)"""

    FEMALE_REPRODUCTIVE = auto()
    """Surveillance of the female reproductive organs"""

    ENDOMETRIUM = auto()
    """Surveillance of the endometrium (lining of the uterus)"""

    OVARIES = auto()
    """Surveillance of the ovaries"""


class SurveillanceTechnology(Enum):
    CA125 = auto()
    CIRCULATING_TUMOUR_DNA = auto()
    DIAGNOSTIC_COLONOSCOPY = auto()
    FLEXIBLE_SIGMOIDOSCOPY = auto()
    HYSTEROSCOPY = auto()
    THERAPEUTIC_COLONOSCOPY = auto()
    TRANSABDOMINAL_ULTRASOUND = auto()
    TRANSVAGINAL_ULTRASOUND = auto()
    UNDIRECTED_ENDOMETRIAL_BIOPSY = auto()
    UTERINE_CAVITY_WASHING = auto()


@dataclass
class SurveillanceMetadata(EventMetadata):
    site: SurveillanceSite
    technology: Set[SurveillanceTechnology]
