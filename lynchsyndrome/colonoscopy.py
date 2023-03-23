import logging
from typing import Any, Mapping

import numpy
import simpy

from lynchsyndrome.bowel import BowelState
from lynchsyndrome.diagnosis import CancerStage, RouteToDiagnosis
from lynchsyndrome.experiments.common.competing_options import Intervention, PathwayPoint
from lynchsyndrome.genetics import ConstitutionalMMR
from lynchsyndrome.individual import Individual
from lynchsyndrome.surveillance import SurveillanceMetadata, SurveillanceSite, SurveillanceTechnology
from lynchsyndrome.reporting import ReportingEvent


class SingleColonoscopy:

    def __init__(
        self,
        env: simpy.Environment,
        rng: numpy.random.Generator,
        params: Mapping[str, Any],
        individual: Individual
    ):
        self.env = env
        self.rng = rng
        self.params = params
        self.individual = individual
    
    def run(self):
        # Determine current bowel state
        state = self.individual.bowel.state
        if state in {BowelState.LOW_RISK_MSS_ADENOMA, BowelState.LOW_RISK_MSI_ADENOMA}:
            sensitivity = self.params['colonoscopy.sensitivity.lowrisk']
            if self.rng.random() < sensitivity:
                logging.info("[%.2f, %s] Low-risk adenoma detected and removed", self.env.now, self.individual)
                self.individual.bowel.state = BowelState.NORMAL
                yield self.env.process(self.individual.record_event(
                    ReportingEvent.CANCER_SURVEILLANCE,
                    SurveillanceMetadata(SurveillanceSite.COLORECTUM, { SurveillanceTechnology.THERAPEUTIC_COLONOSCOPY })
                ))
            else:
                logging.info("[%.2f, %s] Low-risk adenoma missed", self.env.now, self.individual)
                yield self.env.process(self.individual.record_event(
                    ReportingEvent.CANCER_SURVEILLANCE,
                    SurveillanceMetadata(SurveillanceSite.COLORECTUM, { SurveillanceTechnology.DIAGNOSTIC_COLONOSCOPY })
                ))
        elif state in {BowelState.HIGH_RISK_MSS_ADENOMA, BowelState.HIGH_RISK_MSI_ADENOMA}:
            sensitivity = self.params['colonoscopy.sensitivity.highrisk']
            if self.rng.random() < sensitivity:
                logging.info("[%.2f, %s] High-risk adenoma detected and removed", self.env.now, self.individual)
                self.individual.bowel.state = BowelState.NORMAL
                yield self.env.process(self.individual.record_event(
                    ReportingEvent.CANCER_SURVEILLANCE,
                    SurveillanceMetadata(SurveillanceSite.COLORECTUM, { SurveillanceTechnology.THERAPEUTIC_COLONOSCOPY })
                ))
            else:
                logging.info("[%.2f, %s] High-risk adenoma missed", self.env.now, self.individual)
                yield self.env.process(self.individual.record_event(
                    ReportingEvent.CANCER_SURVEILLANCE,
                    SurveillanceMetadata(SurveillanceSite.COLORECTUM, { SurveillanceTechnology.DIAGNOSTIC_COLONOSCOPY })
                ))
        elif state >= BowelState.PRECLIN_STAGE_I and state <= BowelState.PRECLIN_STAGE_IV:
            sensitivity = self.params['colonoscopy.sensitivity.crc']
            if self.rng.random() < sensitivity:
                logging.info("[%.2f, %s] Colorectal cancer identified by surveillance", self.env.now, self.individual)
                state = BowelState(state + BowelState.CLIN_STAGE_I - BowelState.PRECLIN_STAGE_I)
                self.individual.bowel.state = state
                yield self.env.process(self.individual.record_colorectal_cancer_diagnosis(
                    RouteToDiagnosis.SYMPTOMATIC_PRESENTATION,
                    CancerStage(state)
                ))
                p_pres = self.params['colorectal.presentation'][state - BowelState.CLIN_STAGE_I]
                p_prog = None
                if state is BowelState.CLIN_STAGE_IV:
                    p_prog = 0.0
                else:
                    if self.individual.genotype is ConstitutionalMMR.WILD_TYPE:
                        p_prog = self.params['colorectal.progression.genpop'][state - BowelState.CLIN_STAGE_I]
                    else:
                        p_prog = 1 - numpy.exp(-self.params['colorectal.progression.lynch'][state - BowelState.CLIN_STAGE_I])
                if p_prog + p_pres >= 1.0:
                    self.env.process(self.individual.bowel.run_crc_survival())
                else:
                    t_lead_time = -1 / numpy.log(1 - p_prog - p_pres)
                    e_lead_time = self.env.timeout(t_lead_time)
                    yield e_lead_time | self.individual.died | self.individual.reach_time_horizon
                    if e_lead_time.processed:
                        self.env.process(self.individual.bowel.run_crc_survival())
            else:
                logging.info("[%.2f, %s] Colorectal cancer missed", self.env.now, self.individual)
                yield self.env.process(self.individual.record_event(
                    ReportingEvent.CANCER_SURVEILLANCE,
                    SurveillanceMetadata(SurveillanceSite.COLORECTUM, {SurveillanceTechnology.DIAGNOSTIC_COLONOSCOPY})
                ))
        else:
            logging.debug("[%.2f, %s] Nothing to detect by colonoscopy", self.env.now, self.individual)
            yield self.env.process(self.individual.record_event(
                ReportingEvent.CANCER_SURVEILLANCE,
                SurveillanceMetadata(SurveillanceSite.COLORECTUM, {SurveillanceTechnology.DIAGNOSTIC_COLONOSCOPY})
            ))


class BiennialColonoscopy(Intervention):
    
    def __init__(
        self,
        start_age: float,
        stop_age: float
    ):
        self.start_age = start_age
        self.stop_age = stop_age
    
    def run(
        self,
        env: simpy.Environment,
        rng: numpy.random.Generator,
        params: Mapping[str, Any],
        individual: Individual
    ):
        logging.info("[%.2f, %s] BiennialColonoscopy.run", env.now, individual)
        if individual.current_age > self.stop_age:
            yield env.event().succeed()
        
        if individual.current_age < self.start_age:
            t_wait = self.start_age - individual.current_age
            e_wait = env.timeout(t_wait)
            yield e_wait | individual.died | individual.reach_time_horizon
            if not e_wait.processed:
                return
        
        while individual.current_age <= self.stop_age and individual.bowel.state < BowelState.CLIN_STAGE_I:
            # Do a single colonoscopy
            colonoscopy = SingleColonoscopy(env, rng, params, individual)
            env.process(colonoscopy.run())

            # Interval between colonoscopies
            t_interval = params['colonoscopy.interval.m'] * (rng.pareto(params['colonoscopy.interval.a']) + 1)
            e_interval = env.timeout(t_interval)

            yield e_interval | individual.died | individual.reach_time_horizon
            if not e_interval.processed:
                break
