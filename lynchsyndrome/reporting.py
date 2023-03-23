from enum import Enum, auto
from functools import total_ordering


@total_ordering
class ReportingEvent(Enum):
    AEH_DIAGNOSIS          = auto()
    CANCER_DIAGNOSIS       = auto()
    CANCER_RECURRENCE      = auto()
    CANCER_SURVEILLANCE    = auto()
    DEATH                  = auto()
    ENTER_MODEL            = auto()
    REACH_TIME_HORIZON     = auto()
    RISK_REDUCING_SURGERY  = auto()
    START_AEH_MEDICAL      = auto()
    START_CHEMOPROPHYLAXIS = auto()
    STOP_AEH_MEDICAL       = auto()
    STOP_CHEMOPROPHYLAXIS  = auto()

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class EventMetadata:
    pass


# @total_ordering
# class ReportingEvent(Enum):
#     DEATH_FROM_OTHER_CAUSES = auto()
#     DEATH_FROM_SYMPTOMATIC_OVARIAN_CANCER = auto()
#     DEATH_FROM_SURVEILLANCE_DETECTED_OVARIAN_CANCER = auto()

#     OVARIAN_CANCER_SYMPTOMATIC_DIAGNOSIS = auto()
#     OVARIAN_CANCER_SURVEILLANCE_DIAGNOSIS = auto()

#     GYNAE_SURVEILLANCE_VISIT = auto()

#     SURVIVED_TO_TIME_HORIZON = auto()

#     def __lt__(self, other):
#         if self.__class__ is other.__class__:
#             return self.value < other.value
#         return NotImplemented


# def death_reporting_event(cause: CauseOfDeath) -> ReportingEvent:
#     return {
#         CauseOfDeath.OTHER_CAUSES: ReportingEvent.DEATH_FROM_OTHER_CAUSES,
#         CauseOfDeath.SURVEILLANCE_DETECTED_OVARIAN_CANCER: ReportingEvent.DEATH_FROM_SURVEILLANCE_DETECTED_OVARIAN_CANCER,
#         CauseOfDeath.SYMPTOMATIC_OVARIAN_CANCER: ReportingEvent.DEATH_FROM_SYMPTOMATIC_OVARIAN_CANCER
#     }[cause]


# def diagnosis_reporting_event(route: RouteToDiagnosis) -> ReportingEvent:
#     return {
#         RouteToDiagnosis.SURVEILLANCE: ReportingEvent.OVARIAN_CANCER_SURVEILLANCE_DIAGNOSIS,
#         RouteToDiagnosis.SYMPTOMATIC_PRESENTATION: ReportingEvent.OVARIAN_CANCER_SYMPTOMATIC_DIAGNOSIS
#     }[route]

