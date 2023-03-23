import logging
from typing import Any, Mapping, Set

import numpy
import simpy

from lynchsyndrome.diagnosis import CancerStage, RouteToDiagnosis
from lynchsyndrome.endometrium import Endometrium
from lynchsyndrome.experiments.common.competing_options import Intervention, PathwayPoint
from lynchsyndrome.genetics import ConstitutionalMMR
from lynchsyndrome.individual import Individual
from lynchsyndrome.ovaries import OvarianCancerObservedState, OvarianCancerTrueState, OvarianState, OvarianSurgicalState
from lynchsyndrome.reporting import ReportingEvent
from lynchsyndrome.surveillance import SurveillanceMetadata, SurveillanceSite, SurveillanceTechnology


class SingleSurveillance:

    def __init__(
        self,
        env: simpy.Environment,
        rng: numpy.random.Generator,
        params: Mapping[str, Any],
        individual: Individual,
        site: SurveillanceSite,
        technologies: Set[SurveillanceTechnology]
    ):
        self.env = env
        self.rng = rng
        self.params = params
        self.individual = individual
        self.site = site
        self.technologies = technologies
    
    def run(self):
        # Record the surveillance
        yield self.env.process(self.individual.record_event(
            ReportingEvent.CANCER_SURVEILLANCE,
            SurveillanceMetadata(self.site, self.technologies)
        ))

        # Determine menopausal status (roughly)
        meno = 'postmenopausal' if self.individual.current_age >= 51.0 else 'premenopausal'

        oc_diagnosed = False
        ec_diagnosed = False
        aeh_diagnosed = False

        # Ovarian surveillance (if appropriate)
        if self.site in {SurveillanceSite.FEMALE_REPRODUCTIVE, SurveillanceSite.OVARIES}:
            ovarian_state = self.individual.ovaries.state
            if ovarian_state.ovarian_cancer_true_state is OvarianCancerTrueState.NONE:
                logging.debug("[%.2f, %s] No ovarian cancer to detect", self.env.now, self.individual)
            else:
                sensitivity = self.params[f'gynae_surveillance.sensitivity.oc.{meno}']
                if self.rng.random() < sensitivity:
                    logging.info("[%.2f, %s] Ovarian cancer detected by surveillance", self.env.now, self.individual)
                    oc_diagnosed = True
                else:
                    logging.info("[%.2f, %s] Ovarian cancer missed by surveillance", self.env.now, self.individual)

        # Endometrial surveillance (if appropriate)
        if self.site in {SurveillanceSite.FEMALE_REPRODUCTIVE, SurveillanceSite.ENDOMETRIUM}:
            aeh_sensitivity = self.params[f'gynae_surveillance.sensitivity.aeh.{meno}']
            ec_sensitivity  = self.params[f'gynae_surveillance.sensitivity.ec.{meno}']

            n_asymptomatic_aeh = self.individual.endometrium.lesions.loc[('asymptomatic', 0)]
            n_symptomatic_aeh  = self.individual.endometrium.lesions.loc[('symptomatic', 0)]
            n_asymptomatic_ec  = self.individual.endometrium.lesions.loc[('asymptomatic', slice(1, 4))].sum()
            n_symptomatic_ec   = self.individual.endometrium.lesions.loc[('symptomatic', slice(1, 4))].sum()

            n_detected_asymptomatic_aeh = self.rng.binomial(n=n_asymptomatic_aeh, p=aeh_sensitivity)
            n_detected_symptomatic_aeh  = self.rng.binomial(n=n_symptomatic_aeh, p=aeh_sensitivity)
            n_detected_asymptomatic_ec  = self.rng.binomial(n=n_asymptomatic_ec, p=ec_sensitivity)
            n_detected_symptomatic_ec   = self.rng.binomial(n=n_symptomatic_ec, p=ec_sensitivity)

            logging.info(
                "[%.2f, %s] Of %d AEH, %d were detected by surveillance",
                self.env.now, self.individual,
                (n_asymptomatic_aeh + n_symptomatic_aeh),
                (n_detected_asymptomatic_aeh + n_detected_symptomatic_aeh)
            )

            logging.info(
                "[%.2f, %s] Of %d endometrial cancers, %d were detected by surveillance",
                self.env.now, self.individual,
                (n_asymptomatic_ec + n_symptomatic_ec),
                (n_detected_asymptomatic_ec + n_detected_symptomatic_ec)
            )

            if n_detected_asymptomatic_aeh + n_detected_symptomatic_aeh > 0:
                aeh_diagnosed = True
            
            if n_detected_asymptomatic_ec + n_detected_symptomatic_ec > 0:
                ec_diagnosed = True

        if oc_diagnosed and ec_diagnosed:
            # Ovarian and endometrial cancer diagnosed
            self.env.process(self.individual.ovaries.run_oc_diagnosed(RouteToDiagnosis.SURVEILLANCE, True))
            self.env.process(self.individual.endometrium.run_ec_diagnosed(RouteToDiagnosis.SURVEILLANCE))
        elif oc_diagnosed:
            # Ovarian cancer diagnosed (no endometrial cancer)
            self.env.process(self.individual.ovaries.run_oc_diagnosed(RouteToDiagnosis.SURVEILLANCE, False))
        elif ec_diagnosed:
            # Endometrial cancer diagnosed (no ovarian cancer)
            self.env.process(self.individual.endometrium.run_ec_diagnosed(RouteToDiagnosis.SURVEILLANCE))
        elif aeh_diagnosed:
            self.env.process(self.individual.endometrium.run_aeh_diagnosed())
                    


class AnnualGynaecologicalSurveillance(Intervention):
    
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
        logging.info("[%.2f, %s] AnnualGynaecologicalSurveillance.run", env.now, individual)
        if individual.current_age > self.stop_age:
            yield env.event().succeed()
        
        t_wait = None

        if individual.current_age < self.start_age:
            # Not yet ready for surveillance
            t_wait = self.start_age - individual.current_age
        else:
            # Eligible for surveillance straight away, but wait a tiny bit
            # just in case they opt to have RRGS immediately
            t_wait = 1.0/365.0

        e_wait = env.timeout(t_wait)
        yield e_wait | individual.died | individual.reach_time_horizon
        if not e_wait.processed:
            return
        
        while self.is_eligible(individual):
            # Determine site and technologies
            surveil_ovaries = individual.has_ovaries
            surveil_endometrium = individual.has_uterus and not individual.endometrium.aeh_conservative_management
            site, technologies = {
                # Key is tuple of (individual.has_ovaries, individual.has_uterus)
                (True, True): (
                    SurveillanceSite.FEMALE_REPRODUCTIVE,
                    {
                        SurveillanceTechnology.CA125,
                        SurveillanceTechnology.HYSTEROSCOPY,
                        SurveillanceTechnology.TRANSVAGINAL_ULTRASOUND
                    }
                ),
                (True, False): (
                    SurveillanceSite.OVARIES,
                    {
                        SurveillanceTechnology.CA125,
                        SurveillanceTechnology.TRANSVAGINAL_ULTRASOUND
                    }
                ),
                (False, True): (
                    SurveillanceSite.ENDOMETRIUM,
                    {
                        SurveillanceTechnology.HYSTEROSCOPY,
                        SurveillanceTechnology.TRANSVAGINAL_ULTRASOUND
                    }
                ),
            }[(surveil_ovaries, surveil_endometrium)]
            # logging.info("[%.2f, %s] Gynaecological surveillance", env.now, individual)
            # logging.info("[%.2f, %s]   site=%s", env.now, individual, site)
            # logging.info("[%.2f, %s]   technologies=%s", env.now, individual, technologies)
            
            # Do a single gynae surveillance visit
            gynae_surveillance = SingleSurveillance(env, rng, params, individual, site, technologies)
            env.process(gynae_surveillance.run())

            # Interval between surveillance visits
            t_interval = params['gynae_surveillance.interval.m'] * (rng.pareto(params['gynae_surveillance.interval.a']) + 1)
            e_interval = env.timeout(t_interval)

            yield e_interval | individual.died | individual.reach_time_horizon
            if not e_interval.processed:
                break
    
    def is_eligible(self, individual: Individual) -> bool:
        if individual.current_age > self.stop_age:
            return False
        if not (individual.has_ovaries or individual.has_uterus):
            return False
        if individual.ever_diagnosed_endometrial_cancer or individual.ever_diagnosed_ovarian_cancer:
            return False
        if not individual.has_ovaries and individual.endometrium.aeh_conservative_management:
            return False
        return True
