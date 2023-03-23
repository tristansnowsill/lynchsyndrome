

from dataclasses import dataclass
from .diagnosis import CancerSite, CancerStage
from .reporting import EventMetadata


@dataclass
class RecurrenceMetadata(EventMetadata):
    site: CancerSite
    stage_at_diagnosis: CancerStage
