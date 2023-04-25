from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Mapping

import numpy
import simpy
from scipy.stats import lognorm

from lynchsyndrome.experiments.common.competing_options import Intervention
from lynchsyndrome.genetics import ConstitutionalMMR
from lynchsyndrome.reporting import EventMetadata
from lynchsyndrome.risk_reducing_surgery import RiskReducingSurgery, RiskReducingSurgeryMetadata
from lynchsyndrome.individual import Individual

def is_eligible(individual: Individual):
    return (
        (individual.has_ovaries or individual.has_uterus) and
        not (individual.ever_diagnosed_endometrial_cancer or individual.ever_diagnosed_ovarian_cancer)
    )

class ForceTwoStageRRGS(Intervention):

    def __init__(
        self,
        age_hbs: float,
        age_oophorectomy: float
    ):
        super().__init__()
        self.age_hbs = age_hbs
        self.age_oophorectomy = age_oophorectomy

    def run(
        self,
        env: simpy.Environment,
        rng: numpy.random.Generator,
        params: Mapping[str, Any],
        individual: Individual
    ):
        if individual.age < self.age_hbs:
            yield env.timeout(self.age_hbs - individual.age)

        if not (individual.alive and is_eligible(individual)):
            return

        # If they still have a uterus, do a hysterectomy and bilateral
        # salpingectomy
        if individual.has_uterus:
            env.process(individual.run_rrgs(RiskReducingSurgery.HYSTERECTOMY_AND_BILATERAL_SALPINGECTOMY))

        # Wait till age of bilateral oophorectomy
        yield env.timeout(self.age_oophorectomy - individual.current_age)

        if not (individual.alive and is_eligible(individual)):
            return

        # If they still have ovaries, do a bilateral oophorectomy
        if individual.has_ovaries:
            env.process(individual.run_rrgs(RiskReducingSurgery.BILATERAL_OOPHORECTOMY))


class ForceRiskReducingHBSO(Intervention):
    """Force Individual to have risk-reducing HBSO if they are eligible

    Unlike OfferRiskReducingHBSO, this Intervention does not give Individuals a
    choice whether to take up risk-reducing surgery.
    """

    def __init__(
        self,
        min_age: float
    ):
        super().__init__()
        self.min_age = min_age
    
    def run(
        self,
        env: simpy.Environment,
        rng: numpy.random.Generator,
        params: Mapping[str, Any],
        individual: Individual
    ):
        # Wait until they reach the minimum age to be offered RRGS
        if individual.age < self.min_age:
            yield env.timeout(self.min_age - individual.age)
        
        # Check eligibility
        if not individual.alive:
            logging.info("[%.2f, %s] ForceRiskReducingHBSO: cannot offer as individual already died", env.now, individual)
            return
        
        if not is_eligible(individual):
            return
        
        # Determine appropriate surgery
        surgery = self.determine_appropriate_surgery(individual)

        yield env.process(individual.run_rrgs(surgery))



    def determine_appropriate_surgery(self, individual):
        surgery = RiskReducingSurgery.HYSTERECTOMY_AND_BILATERAL_SALPINGO_OOPHORECTOMY
        
        if individual.has_ovaries and not individual.has_uterus:
            surgery = RiskReducingSurgery.BILATERAL_SALPINGO_OOPHORECTOMY
        
        if individual.has_uterus and not individual.has_ovaries:
            surgery = RiskReducingSurgery.HYSTERECTOMY
        
        return surgery

