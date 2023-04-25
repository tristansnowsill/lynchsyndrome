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

if TYPE_CHECKING:
    from lynchsyndrome.individual import Individual

class RiskReducingSurgery(Enum):
    BILATERAL_OOPHORECTOMY                           = auto()
    BILATERAL_SALPINGO_OOPHORECTOMY                  = auto()
    BILATERAL_SALPINGECTOMY                          = auto()
    HYSTERECTOMY                                     = auto()
    HYSTERECTOMY_AND_BILATERAL_SALPINGECTOMY         = auto()
    HYSTERECTOMY_AND_BILATERAL_SALPINGO_OOPHORECTOMY = auto()

    BSO  = BILATERAL_SALPINGO_OOPHORECTOMY
    HBS  = HYSTERECTOMY_AND_BILATERAL_SALPINGECTOMY
    HBSO = HYSTERECTOMY_AND_BILATERAL_SALPINGO_OOPHORECTOMY

    def adjust(self, has_ovaries: bool, has_uterus: bool):
        """Adjust the surgery (if needed) to account for organ state"""
        if not (has_ovaries or has_uterus):
            raise ValueError("RiskReducingSurgery.adjust() expects has_ovaries and/or has_uterus to be True")
        if not has_ovaries and self is RiskReducingSurgery.BSO:
            return None
        if not has_ovaries and self is RiskReducingSurgery.HBSO:
            return RiskReducingSurgery.HYSTERECTOMY
        if not has_uterus and self is RiskReducingSurgery.HYSTERECTOMY:
            return None
        if not has_uterus and self is RiskReducingSurgery.HBSO:
            return RiskReducingSurgery.BSO
        if not has_uterus and self is RiskReducingSurgery.HBS:
            return RiskReducingSurgery.BILATERAL_SALPINGECTOMY
        return self

    def affects_ovaries(self) -> bool:
        return self in [RiskReducingSurgery.BSO, RiskReducingSurgery.HBSO]

    def affects_uterus(self) -> bool:
        return self in [RiskReducingSurgery.HYSTERECTOMY, RiskReducingSurgery.HBS, RiskReducingSurgery.HBSO]


@dataclass
class RiskReducingSurgeryMetadata(EventMetadata):
    surgery: RiskReducingSurgery


class OfferRiskReducingHBSO(Intervention):

    def __init__(
        self,
        min_age: float,
        surveillance: bool
    ):
        super().__init__()
        self.min_age = min_age
        self.surveillance = surveillance
    
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
            logging.info("[%.2f, %s] OfferRiskReducingHBSO: cannot offer as individual already died", env.now, individual)
            return
        
        if not self.is_eligible(individual):
            return
        
        
        # Determine appropriate surgery
        surgery = self.determine_appropriate_surgery(individual)

        # Do they take up RRGS immediately?
        p_immediate = params['risk_reducing_surgery.uptake.immediately']
        if rng.random() < p_immediate:
            logging.info("[%.2f, %s] OfferRiskReducingHBSO: taking up RRGS immediately", env.now, individual)
            yield env.process(individual.run_rrgs(surgery))
            return
        
        # Behaviour will depend on genotype
        genotype = {
            ConstitutionalMMR.PATH_MLH1: "MLH1",
            ConstitutionalMMR.PATH_MSH2: "MSH2",
            ConstitutionalMMR.PATH_MSH6: "MSH6",
            ConstitutionalMMR.PATH_PMS2: "PMS2"
        }[individual.genotype]

        # Will they never take up RRGS?
        p_never = params[f'risk_reducing_surgery.uptake.never.{genotype}']
        if rng.random() < p_never/(1-p_immediate):
            logging.info("[%.2f, %s] OfferRiskReducingHBSO: will never take up RRGS", env.now, individual)
            return
        
        # They (may) take it up at some point
        mu = params[f'risk_reducing_surgery.uptake.mu.{genotype}']
        sigma = params[f'risk_reducing_surgery.uptake.sigma.{genotype}']
        # It must be no earlier than now
        # The way SciPy handles the lognormal distribution is a little strange
        # To achieve the conventional LogN(mu, sigma^2) distribution, it is
        # necessary to use s = sigma and scale = exp(mu)
        uptake = lognorm(s=sigma, scale=numpy.exp(mu))
        p_uptake = uptake.cdf(individual.current_age)
        u_uptake = rng.uniform(p_uptake, 1.0)
        x_uptake = uptake.ppf(u_uptake)
        t_uptake = x_uptake - individual.current_age

        logging.debug("individual.current_age=%.2f", individual.current_age)
        logging.debug("uptake.mean (unconditional)=%.2f", uptake.mean())
        logging.debug("p_uptake=%.3f", p_uptake)
        logging.debug("u_uptake=%.3f", u_uptake)
        logging.debug("x_uptake=%.2f", x_uptake)
        logging.debug("t_uptake=%.2f", t_uptake)
        
        if not self.surveillance:
            t_uptake *= params['risk_reducing_surgery.uptake.aft.no_surveillance']
            logging.debug("t_uptake (no surveillance)=%.2f", t_uptake)
        
        e_uptake = env.timeout(t_uptake)
        yield e_uptake | individual.died | individual.reach_time_horizon

        if e_uptake.processed and self.is_eligible(individual):
            logging.info("[%.2f, %s] OfferRiskReducingHBSO: taking up RRGS at age %.2f", env.now, individual, individual.current_age)
            yield env.process(individual.run_rrgs(surgery))



    def determine_appropriate_surgery(self, individual):
        surgery = RiskReducingSurgery.HYSTERECTOMY_AND_BILATERAL_SALPINGO_OOPHORECTOMY
        
        if individual.has_ovaries and not individual.has_uterus:
            surgery = RiskReducingSurgery.BILATERAL_SALPINGO_OOPHORECTOMY
        
        if individual.has_uterus and not individual.has_ovaries:
            surgery = RiskReducingSurgery.HYSTERECTOMY
        
        return surgery


    def is_eligible(self, individual: Individual):
        if individual.ever_diagnosed_endometrial_cancer:
            logging.info("[%.2f, %s] OfferRiskReducingHBSO: cannot offer as individual already diagnosed with endometrial cancer", individual.env.now, individual)
            return False
        
        if individual.ever_diagnosed_ovarian_cancer:
            logging.info("[%.2f, %s] OfferRiskReducingHBSO: cannot offer as individual already diagnosed with ovarian cancer", individual.env.now, individual)
            return False
        
        if not(individual.has_ovaries or individual.has_uterus):
            logging.info("[%.2f, %s] OfferRiskReducingHBSO: cannot offer as individual has neither uterus nor ovaries", individual.env.now, individual)
            return False
        
        return True