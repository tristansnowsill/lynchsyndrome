from enum import auto, Enum


class ConstitutionalMMR(Enum):
    # No Lynch syndrome
    WILD_TYPE = auto()

    # "Classic" Lynch syndrome
    PATH_MLH1 = auto()
    PATH_MSH2 = auto()
    PATH_MSH6 = auto()
    PATH_PMS2 = auto()


class GeneticKnowledge(Enum):
    NOT_APPLICABLE = auto()
    NO_KNOWLEDGE = auto()
    SUSPECTED = auto()
    DIAGNOSED = auto()
