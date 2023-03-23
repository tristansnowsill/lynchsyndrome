from dataclasses import dataclass
from enum import Enum, auto

from .reporting import EventMetadata


class RouteToDiagnosis(Enum):

    SCREENING = auto()
    """Population screening, e.g., national bowel screening programme"""

    SURVEILLANCE = auto()
    """Surveillance in high-risk individual"""

    SYMPTOMATIC_PRESENTATION = auto()
    """Diagnosed following presentation with symptoms"""

    RISK_REDUCING_SURGERY = auto()
    """Diagnosed following pathological examination of specimens from risk reducing surgery"""


class CancerSite(Enum):
    COLORECTUM = auto()
    ENDOMETRIUM = auto()
    OVARIES = auto()


class CancerStage(Enum):

    STAGE_I   = 1
    STAGE_II  = 2
    STAGE_III = 3
    STAGE_IV  = 4

    # LOCAL = auto()
    # REGIONAL = auto()
    # METASTATIC = auto()

    # EARLY = auto()
    # LATE = auto()

    def __str__(self):
        return {
            CancerStage.STAGE_I  : 'I',
            CancerStage.STAGE_II : 'II',
            CancerStage.STAGE_III: 'III',
            CancerStage.STAGE_IV : 'IV'
        }[self]

    @classmethod
    def _missing_(cls, value):
        from lynchsyndrome.bowel import BowelState
        from lynchsyndrome.ovaries import OvarianCancerObservedState, OvarianCancerTrueState

        if isinstance(value, OvarianCancerObservedState):
            return {
                OvarianCancerObservedState.STAGE_I  : cls.STAGE_I,
                OvarianCancerObservedState.STAGE_II : cls.STAGE_II,
                OvarianCancerObservedState.STAGE_III: cls.STAGE_III,
                OvarianCancerObservedState.STAGE_IV : cls.STAGE_IV
            }[value]
        elif isinstance(value, OvarianCancerTrueState):
            return {
                OvarianCancerTrueState.STAGE_I  : cls.STAGE_I,
                OvarianCancerTrueState.STAGE_II : cls.STAGE_II,
                OvarianCancerTrueState.STAGE_III: cls.STAGE_III,
                OvarianCancerTrueState.STAGE_IV : cls.STAGE_IV
            }[value]
        elif isinstance(value, BowelState):
            return {
                BowelState.CLIN_STAGE_I     : cls.STAGE_I,
                BowelState.CLIN_STAGE_II    : cls.STAGE_II,
                BowelState.CLIN_STAGE_III   : cls.STAGE_III,
                BowelState.CLIN_STAGE_IV    : cls.STAGE_IV,
                BowelState.PRECLIN_STAGE_I  : cls.STAGE_I,
                BowelState.PRECLIN_STAGE_II : cls.STAGE_II,
                BowelState.PRECLIN_STAGE_III: cls.STAGE_III,
                BowelState.PRECLIN_STAGE_IV : cls.STAGE_IV
            }[value]
        else:
            return super()._missing_(value)


def is_local(stage: CancerStage):
    return (
        stage is CancerStage.STAGE_I
        or stage is CancerStage.STAGE_II
        # or stage is CancerStage.LOCAL
        # or stage is CancerStage.EARLY
    )

def is_early(stage: CancerStage):
    return is_local(stage)

def is_late(stage: CancerStage):
    return (
        stage is CancerStage.STAGE_III
        or stage is CancerStage.STAGE_IV
        # or stage is CancerStage.REGIONAL
        # or stage is CancerStage.METASTATIC
        # or stage is CancerStage.LATE
    )

def is_advanced(stage: CancerStage):
    return is_late(stage)


@dataclass
class DiagnosisMetadata(EventMetadata):
    route: RouteToDiagnosis
    site: CancerSite
    stage: CancerStage

