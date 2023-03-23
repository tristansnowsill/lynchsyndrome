from __future__ import annotations
from dataclasses import dataclass
import logging
from copy import deepcopy
from enum import Enum, IntEnum, auto
from typing import TYPE_CHECKING, Any, List, Literal, Mapping, Optional

import numpy
import pandas
import simpy
from scipy.interpolate import BSpline

from lynchsyndrome.death import CauseOfDeath
from lynchsyndrome.diagnosis import CancerStage, RouteToDiagnosis
from lynchsyndrome.endometrial_baseline import BaselineLesionState, EndometrialBaselineCalculation
from lynchsyndrome.genetics import ConstitutionalMMR
from lynchsyndrome.inhomogeneous import InhomogeneousPoissonProcess
from lynchsyndrome.reporting import EventMetadata, ReportingEvent
from lynchsyndrome.risk_reducing_surgery import RiskReducingSurgery
from lynchsyndrome.surveillance import SurveillanceMetadata, SurveillanceSite, SurveillanceTechnology

if TYPE_CHECKING:
    from lynchsyndrome.individual import Individual

@dataclass
class AEHDiagnosedMetadata(EventMetadata):
    management: Literal['medical', 'surgical']

class Endometrium:

    incidence_spline_knots  = numpy.array([0.0, 0.0, 0.0, 0.0, 33.33, 66.67, 100.0, 100.0, 100.0, 100.0])
    incidence_spline_degree = 3

    baseline_state_cache = {
        ConstitutionalMMR.PATH_MLH1: None,
        ConstitutionalMMR.PATH_MSH2: None,
        ConstitutionalMMR.PATH_MSH6: None,
        ConstitutionalMMR.PATH_PMS2: None
    }
    cache_params = {
        ConstitutionalMMR.PATH_MLH1: None,
        ConstitutionalMMR.PATH_MSH2: None,
        ConstitutionalMMR.PATH_MSH6: None,
        ConstitutionalMMR.PATH_PMS2: None
    }
    
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
        self.params_hash = Endometrium._get_params_hash(params)
        self.individual = individual

        self.lesions: pandas.Series = pandas.Series(
            index=pandas.MultiIndex.from_product(
                (['asymptomatic', 'symptomatic'], numpy.arange(0, 5)),
                names=['symptomatic', 'stage']
            ),
            dtype="Int64",
            name='lesions'
        )
        self.ec_diagnosed = False
        self.aeh_conservative_management = False

        self.recalculate = env.event()
        self.stop = env.event()
        self.hysterectomy = env.event()
    
    def sample_baseline_state(self, init_rng: numpy.random.Generator):
        if not self.lesions.isnull().values.any():
            return
        
        gt = self.individual.genotype
        if self.baseline_state_cache[gt] is None or self.cache_params[gt] != self.params_hash:
            logging.debug("Endometrium.sample_baseline_state: table has not yet been cached")
            # We haven't yet done a calculation for this parameter set
            incidence_mapping = {
                ConstitutionalMMR.PATH_MLH1: self.params['endometrial.incidence.MLH1'],
                ConstitutionalMMR.PATH_MSH2: self.params['endometrial.incidence.MSH2'],
                ConstitutionalMMR.PATH_MSH6: self.params['endometrial.incidence.MSH6'],
                ConstitutionalMMR.PATH_PMS2: self.params['endometrial.incidence.PMS2'],
                ConstitutionalMMR.WILD_TYPE: self.params['endometrial.incidence.genpop']
            }
            alpha = incidence_mapping[self.individual.genotype]
            bs = BSpline(self.incidence_spline_knots, alpha, self.incidence_spline_degree)
            aeh_incidence = lambda x: numpy.exp(bs(x))

            self.baseline_state_cache[gt] = EndometrialBaselineCalculation(self.params, aeh_incidence).get_dataframe()
            self.cache_params[gt] = self.params_hash

        # Now we know there will be a baseline calculation ready for us to use
        table: pandas.DataFrame = self.baseline_state_cache[gt]
        
        # Grab the relevant probabilities based on age
        probs = table.loc[numpy.round(self.individual.age)]

        # Sample at random using the shared random number generator
        sampled = probs.sample(n=1, weights=probs, random_state=init_rng)
        lesions = sampled.index.to_list()[0]
        logging.debug("Endometrium.sample_baseline_state: sampled baseline lesions %s", lesions)

        # Populate self.lesions appropriately
        state_mapping = {
            BaselineLesionState.ASYMPTOMATIC_AEH      : ('asymptomatic', 0),
            BaselineLesionState.ASYMPTOMATIC_STAGE_I  : ('asymptomatic', 1),
            BaselineLesionState.ASYMPTOMATIC_STAGE_II : ('asymptomatic', 2),
            BaselineLesionState.ASYMPTOMATIC_STAGE_III: ('asymptomatic', 3),
            BaselineLesionState.ASYMPTOMATIC_STAGE_IV : ('asymptomatic', 4),
            BaselineLesionState.SYMPTOMATIC_AEH       : ('symptomatic', 0),
            BaselineLesionState.SYMPTOMATIC_STAGE_I   : ('symptomatic', 1),
            BaselineLesionState.SYMPTOMATIC_STAGE_II  : ('symptomatic', 2),
            BaselineLesionState.SYMPTOMATIC_STAGE_III : ('symptomatic', 3),
            BaselineLesionState.SYMPTOMATIC_STAGE_IV  : ('symptomatic', 4)
        }
        self.lesions.loc[:] = 0
        for l in lesions:
            if l is not BaselineLesionState.NONE:
                logging.debug("appending lesion %s", l)
                self.lesions.loc[state_mapping[l]] += 1
        
        logging.debug("self.lesions=%s", self.lesions)
        
    
    @property
    def has_occult_ec(self):
        return not self.ec_diagnosed and self.lesions.loc[(slice(None), slice(1, 4))].sum() > 0
    
    @property
    def occult_ec_stage(self):
        if not self.has_occult_ec:
            raise RuntimeError("occult_ec_stage called but there are no occult endometrial cancers")
        
        return next(i for i in list(range(1, 5))[::-1] if self.lesions.loc[(slice(None), slice(i, i))].sum() > 0)
    
    @property
    def has_diagnosed_ec(self):
        return self.ec_diagnosed
    
    @property
    def is_symptomatic(self):
        return self.lesions.loc[('symptomatic', slice(None))].sum() > 0

    def run(self):
        incidence_mapping = {
            ConstitutionalMMR.PATH_MLH1: self.params['endometrial.incidence.MLH1'],
            ConstitutionalMMR.PATH_MSH2: self.params['endometrial.incidence.MSH2'],
            ConstitutionalMMR.PATH_MSH6: self.params['endometrial.incidence.MSH6'],
            ConstitutionalMMR.PATH_PMS2: self.params['endometrial.incidence.PMS2'],
            ConstitutionalMMR.WILD_TYPE: self.params['endometrial.incidence.genpop']
        }
        alpha = incidence_mapping[self.individual.genotype]
        bs = BSpline(self.incidence_spline_knots, alpha, self.incidence_spline_degree)

        ipp_env = numpy.exp(numpy.sum(numpy.clip(alpha, a_min=0, a_max=None)))
        ipp_fn = lambda x: numpy.exp(bs(x))
        ipp = InhomogeneousPoissonProcess(
            ipp_fn,
            (self.individual.age, 100.0),
            [(self.individual.age, 100.0, ipp_env)], self.rng
        )
        ipp_sample = sorted(ipp.sample())

        logging.info("[%.2f, %s] Endometrium.run - sampled AEH incidence ages: %s", self.env.now, self.individual, ipp_sample)

        if len(ipp_sample) + self.lesions.sum() == 0:
            return

        e_aeh_incidence = [self.env.timeout(age - self.individual.age, value=('aeh_incidence', None)) for age in ipp_sample]

        while True:
            e_other = list()
            
            # Asymptomatic AEH events: symptomatic, regression, progression
            n_asymptomatic_aeh = self.lesions.loc[('asymptomatic', 0)]
            if n_asymptomatic_aeh > 0:
                e_other.append(
                    self.env.timeout(
                        self.rng.exponential(scale=1/(n_asymptomatic_aeh * self.params['endometrial.symptomatic'][0])),
                        value=('aeh_symptomatic', None)
                    )
                )
                e_other.append(
                    self.env.timeout(
                        self.rng.exponential(scale=1/(n_asymptomatic_aeh * self.params['endometrial.regression'])),
                        value=('aeh_regression', 'asymptomatic')
                    )
                )
                e_other.append(
                    self.env.timeout(
                        self.rng.exponential(scale=1/(n_asymptomatic_aeh * self.params['endometrial.progression'][0])),
                        value=('aeh_progression', 'asymptomatic')
                    )
                )

            # Symptomatic AEH events: regression, diagnosis, progression
            n_symptomatic_aeh = self.lesions.loc[('symptomatic', 0)]
            if n_symptomatic_aeh > 0:
                e_other.append(
                    self.env.timeout(
                        self.rng.exponential(scale=1/(n_symptomatic_aeh * self.params['endometrial.regression'])),
                        value=('aeh_regression', 'symptomatic')
                    )
                )
                if not self.aeh_conservative_management:
                    e_other.append(
                        self.env.timeout(
                            self.rng.exponential(scale=1/(n_symptomatic_aeh * self.params['endometrial.diagnosis'][0])),
                            value=('aeh_diagnosed', None)
                        )
                    )
                e_other.append(
                    self.env.timeout(
                        self.rng.exponential(scale=1/(n_symptomatic_aeh * self.params['endometrial.progression'][0])),
                        value=('aeh_progression', 'symptomatic')
                    )
                )

            # EC events: symptomatic/diagnosis, progression
            for stage in range(1, 5):
                n_asymptomatic = self.lesions.loc[('asymptomatic', stage)]
                n_symptomatic  = self.lesions.loc[('symptomatic', stage)]
                if n_asymptomatic > 0:
                    if stage < 4:
                        e_other.append(
                            self.env.timeout(
                                self.rng.exponential(scale=1/(n_asymptomatic * self.params['endometrial.progression'][stage])),
                                value=('ec_progression', ('asymptomatic', stage))
                            )
                        )
                    e_other.append(
                        self.env.timeout(
                            self.rng.exponential(scale=1/(n_asymptomatic * self.params['endometrial.symptomatic'][stage])),
                            value=('ec_symptomatic', stage)
                        )
                    )
                if n_symptomatic > 0:
                    if stage < 4:
                        e_other.append(
                            self.env.timeout(
                                self.rng.exponential(scale=1/(n_symptomatic * self.params['endometrial.progression'][stage])),
                                value=('ec_progression', ('symptomatic', stage))
                            )
                        )
                    e_other.append(
                        self.env.timeout(
                            self.rng.exponential(scale=1/(n_symptomatic * self.params['endometrial.diagnosis'][stage])),
                            value=('ec_diagnosed', stage)
                        )
                    )

            res = yield simpy.AnyOf(self.env, e_aeh_incidence + e_other + [self.stop, self.hysterectomy, self.recalculate])

            if self.stop.processed or self.hysterectomy.processed:
                return

            if self.recalculate.processed:
                ev, md = 'recalculate', None
                self.recalculate = self.env.event()
            else:
                ev, md = next(res.values())

            if ev == 'aeh_incidence':
                self.lesions.loc[('asymptomatic', 0)] += 1
                e_aeh_incidence = [e for e in e_aeh_incidence if not e.processed]
            elif ev == 'aeh_symptomatic':
                self.lesions.loc[('symptomatic', 0)] += 1
                self.lesions.loc[('asymptomatic', 0)] -= 1
            elif ev == 'aeh_regression':
                self.lesions.loc[(md, 0)] -= 1
            elif ev == 'aeh_progression':
                self.lesions.loc[(md, 0)] -= 1
                self.lesions.loc[(md, 1)] += 1
            elif ev == 'ec_progression':
                self.lesions.loc[(md[0], md[1])] -= 1
                self.lesions.loc[(md[0], md[1]+1)] += 1
            elif ev == 'ec_symptomatic':
                self.lesions.loc[('asymptomatic', md)] -= 1
                self.lesions.loc[('symptomatic', md)] += 1
            elif ev == 'aeh_diagnosed':
                self.env.process(self.run_aeh_diagnosed())
            elif ev == 'ec_diagnosed':
                self.env.process(self.run_ec_diagnosed(RouteToDiagnosis.SYMPTOMATIC_PRESENTATION))
                
            if len(e_aeh_incidence) + self.lesions.sum() == 0:
                return

    
    def run_aeh_diagnosed(self):
        logging.info("[%.2f, %s] Endometrium.run_aeh_diagnosed", self.env.now, self.individual)
        theta = self.params['endometrial.aeh_management']
        p_surgical = theta[0] + theta[1] / (1 + numpy.exp(-(self.individual.current_age - theta[2]) / theta[3]))
        if self.rng.random() < p_surgical:
            yield self.env.process(self.individual.record_event(ReportingEvent.AEH_DIAGNOSIS, AEHDiagnosedMetadata('surgical')))
            self.env.process(self.run_aeh_surgical_management())
        else:
            yield self.env.process(self.individual.record_event(ReportingEvent.AEH_DIAGNOSIS, AEHDiagnosedMetadata('medical')))
            self.env.process(self.run_aeh_medical_management())
        yield self.env.event().succeed()
    
    def run_aeh_surgical_management(self):
        logging.info("[%.2f, %s] Endometrium.run_aeh_surgical_management", self.env.now, self.individual)
        self.env.process(self.individual.run_rrgs())
        yield self.env.event().succeed()
    
    def run_aeh_medical_management(self):
        logging.info("[%.2f, %s] Endometrium.run_aeh_medical_management", self.env.now, self.individual)
        self.aeh_conservative_management = True
        yield self.env.process(self.individual.record_event(ReportingEvent.START_AEH_MEDICAL, EventMetadata()))

        # Check for any concurrent lesions which may respond to medical management
        n_symptomatic_aeh       = self.lesions.loc[('symptomatic', 0)]
        n_asymptomatic_aeh      = self.lesions.loc[('asymptomatic', 0)]
        n_symptomatic_early_ec  = self.lesions.loc[('symptomatic', 1)]
        n_asymptomatic_early_ec = self.lesions.loc[('asymptomatic', 1)]

        # Determine whether lesions will ultimately respond to medical management
        p_respond_aeh                   = self.params['endometrial.aeh_management.conservative_success']
        p_respond_early_ec              = self.params['endometrial.early_ec.response_rate']
        n_respond_symptomatic_aeh       = self.rng.binomial(n=n_symptomatic_aeh, p=p_respond_aeh)
        n_respond_asymptomatic_aeh      = self.rng.binomial(n=n_asymptomatic_aeh, p=p_respond_aeh)
        n_respond_symptomatic_early_ec  = self.rng.binomial(n=n_symptomatic_early_ec, p=p_respond_early_ec)
        n_respond_asymptomatic_early_ec = self.rng.binomial(n=n_asymptomatic_early_ec, p=p_respond_early_ec)

        # We remove the lesions that will respond straight away
        self.lesions.loc[('symptomatic', 0)]  -= n_respond_symptomatic_aeh
        self.lesions.loc[('asymptomatic', 0)] -= n_respond_asymptomatic_aeh
        self.lesions.loc[('symptomatic', 1)]  -= n_respond_symptomatic_early_ec
        self.lesions.loc[('asymptomatic', 1)] -= n_respond_asymptomatic_early_ec

        # We trigger self.recalculate to interrupt the event loop in run()
        if self.recalculate.triggered:
            logging.warning("Need to investigate")
        self.recalculate.succeed()

        # First examination after 3 months key for endometrial cancer...
        e_interval = self.env.timeout(0.25)
        yield e_interval | self.stop | self.hysterectomy
        if not e_interval.processed:
            yield self.env.process(self.individual.record_event(ReportingEvent.STOP_AEH_MEDICAL, EventMetadata()))
            self.aeh_conservative_management = False
            return
        
        yield self.env.process(self.individual.record_event(
            ReportingEvent.CANCER_SURVEILLANCE,
            SurveillanceMetadata(
                SurveillanceSite.ENDOMETRIUM,
                {SurveillanceTechnology.HYSTEROSCOPY,SurveillanceTechnology.TRANSVAGINAL_ULTRASOUND}
            )
        ))

        # Are there any endometrial cancers?
        if self.has_occult_ec:
            self.aeh_conservative_management = False
            self.env.process(self.run_ec_diagnosed(RouteToDiagnosis.SURVEILLANCE))
            return
        else:
            # Follow up for another 9 months before determining whether to use
            # surgical management
            for _ in range(3):
                e_interval = self.env.timeout(0.25)
                yield e_interval | self.stop | self.hysterectomy
                if not e_interval.processed:
                    yield self.env.process(self.individual.record_event(ReportingEvent.STOP_AEH_MEDICAL, EventMetadata()))
                    self.aeh_conservative_management = False
                    return
                yield self.env.process(self.individual.record_event(
                    ReportingEvent.CANCER_SURVEILLANCE,
                    SurveillanceMetadata(
                        SurveillanceSite.ENDOMETRIUM,
                        {SurveillanceTechnology.HYSTEROSCOPY,SurveillanceTechnology.TRANSVAGINAL_ULTRASOUND}
                    )
                ))
                if self.has_occult_ec:
                    self.aeh_conservative_management = False
                    self.env.process(self.run_ec_diagnosed(RouteToDiagnosis.SURVEILLANCE))
                    return
            
            # If there are still any AEH we do surgical management
            if self.lesions.sum() > 0:
                self.env.process(self.individual.endometrium.run_aeh_surgical_management())
                self.aeh_conservative_management = False
                return
            
    def run_ec_diagnosed(self, route: RouteToDiagnosis):
        logging.info("[%.2f, %s] Endometrium.run_ec_diagnosed", self.env.now, self.individual)

        if self.has_diagnosed_ec:
            logging.warning("%s, %s - oh", __file__, 301)
        if self.hysterectomy.triggered:
            logging.warning("[%.2f, %s] run_ec_diagnosed called but hysterectomy has already been triggered", self.env.now, self.individual)


        # Stop simulation of lesions
        self.stop.succeed()
        
        # Record event
        stage_at_diagnosis = CancerStage(self.occult_ec_stage)
        symptomatic = self.is_symptomatic

        self.ec_diagnosed = True
        yield self.env.process(self.individual.record_endometrial_cancer_diagnosis(
            route,
            stage_at_diagnosis
        ))

        # Hysterectomy unless was diagnosed during hysterectomy
        if route is not RouteToDiagnosis.RISK_REDUCING_SURGERY:
            if self.hysterectomy.triggered:
                logging.warning("[%.2f, %s] run_ec_diagnosed is about to signal_hysterectomy, but it has already been triggered", self.env.now, self.individual)
            self.signal_hysterectomy()
        if self.individual.has_ovaries:
            self.individual.ovaries.signal_oophorectomy()

        if self.individual.genotype is ConstitutionalMMR.WILD_TYPE:
            raise NotImplementedError()
        else:
            if route is not RouteToDiagnosis.SYMPTOMATIC_PRESENTATION:
                logging.info("[%.2f, %s] Not diagnosed following symptomatic presentation, sampling lead time...", self.env.now, self.individual)
                t_lead_time = self._lead_time(stage_at_diagnosis, symptomatic, True)
                e_lead_time = self.env.timeout(t_lead_time)
                yield e_lead_time | self.individual.died
                if not e_lead_time.processed:
                    return
            if stage_at_diagnosis in [CancerStage.STAGE_I, CancerStage.STAGE_II]:
                logging.info("[%.2f, %s] Diagnosed in early stage, sampling time to recurrence...", self.env.now, self.individual)
                t_recur = self.rng.exponential(scale=1/self.params['endometrial.recurrence.lynch.early'])
                e_recur = self.env.timeout(t_recur)
                yield e_recur | self.individual.died
                if e_recur.processed:
                    logging.info("[%.2f, %s] Cancer recurrence, sampling time to death...", self.env.now, self.individual)
                    yield self.env.process(self.individual.record_endometrial_cancer_recurrence(stage_at_diagnosis))
                    t_die = self.rng.exponential(scale=1/self.params['endometrial.postrecurrence.lynch'])
                    e_die = self.env.timeout(t_die)
                    yield e_die | self.individual.died
                    if e_die.processed:
                        logging.info("[%.2f, %s] Death from endometrial cancer...", self.env.now, self.individual)
                        yield self.env.process(self.individual.record_death(CauseOfDeath.ENDOMETRIAL_CANCER))
            else:
                logging.info("[%.2f, %s] Diagnosed in late stage, sampling time to recurrence...", self.env.now, self.individual)
                t_die = self.rng.exponential(scale=1/self.params['endometrial.survival.lynch.late'])
                e_die = self.env.timeout(t_die)
                yield e_die | self.individual.died
                if e_die.processed:
                    logging.info("[%.2f, %s] Death from endometrial cancer...", self.env.now, self.individual)
                    yield self.env.process(self.individual.record_death(CauseOfDeath.ENDOMETRIAL_CANCER))

    def run_hysterectomy(self):
        """Process for a hysterectomy
        
        Notes:
        - This is a generator, so must be called with env.process(...)
        - This process does not take any time, so it is safe to yield for synchronous execution
        - This process may register another process with the environment (run_ec_diagnosed)
        """
        if self.hysterectomy.triggered:
            logging.warning("[%.2f, %s] run_hysterectomy called, but hysterectomy event already triggered", self.env.now, self.individual)
        if not self.has_occult_ec:
            self.signal_hysterectomy()
        elif not self.has_diagnosed_ec:
            self.env.process(self.run_ec_diagnosed(RouteToDiagnosis.RISK_REDUCING_SURGERY))
        yield self.env.event().succeed()

    def _lead_time(self, stage: CancerStage, symptomatic: bool, lynch: bool):
        if lynch:
            if stage is CancerStage.STAGE_I:
                _lambda = self.params['endometrial.progression'][1:3]
                _delta  = self.params['endometrial.diagnosis'][1:3]
                if not symptomatic:
                    # Viable routes to presenting while still in Stage I/II are
                    #   a: symptomatic -> diagnosed
                    #   b: symptomatic -> progressed -> diagnosed
                    #   c: progressed -> symptomatic -> diagnosed
                    _xi = self.params['endometrial.symptomatic'][1:3]

                    p_a = _xi[0] / (_xi[0] + _lambda[0]) * _delta[0] / (_delta[0] + _lambda[0])
                    p_b = _xi[0] / (_xi[0] + _lambda[0]) * _lambda[0] / (_delta[0] + _lambda[0]) * _delta[1] / (_delta[1] + _lambda[1])
                    p_c = _lambda[0] / (_xi[0] + _lambda[0]) * _xi[1] / (_xi[1] + _lambda[1]) * _delta[1] / (_delta[1] + _lambda[1])
                    Et_a = 1 / (_xi[0] + _lambda[0]) + 1 / (_delta[0] + _lambda[0])
                    Et_b = 1 / (_xi[0] + _lambda[0]) + 1 / (_delta[0] + _lambda[0]) + 1 / (_delta[1] + _lambda[1])
                    Et_c = 1 / (_xi[0] + _lambda[0]) + 1 / (_xi[1] + _lambda[1]) + 1 / (_delta[1] + _lambda[1])

                    return (Et_a * p_a + Et_b * p_b + Et_c * p_c) / (p_a + p_b + p_c)
                else:
                    # Viable routes to presenting while still in Stage I/II are
                    #   a: diagnosed
                    #   b: progressed -> diagnosed
                    p_a = _delta[0] / (_delta[0] + _lambda[0])
                    p_b = _lambda[0] / (_delta[0] + _lambda[0]) * _delta[1] / (_delta[1] + _lambda[1])
                    Et_a = 1 / (_delta[0] + _lambda[0])
                    Et_b = 1 / (_delta[0] + _lambda[0]) + 1 / (_delta[1] + _lambda[1])

                    return (Et_a * p_a + Et_b * p_b) / (p_a + p_b)
            elif stage is CancerStage.STAGE_II:
                _lambda = self.params['endometrial.progression'][2]
                _delta  = self.params['endometrial.diagnosis'][2]
                if not symptomatic:
                    # Must become symptomatic then diagnosed without progressing
                    _xi = self.params['endometrial.symptomatic'][2]
                    return 1 / (_xi + _lambda) + 1 / (_delta + _lambda)
                else:
                    # Must be diagnosed without progressing
                    return 1 / (_lambda + _delta)
            elif stage is CancerStage.STAGE_III:
                _lambda = self.params['endometrial.progression'][3]
                _delta  = self.params['endometrial.diagnosis'][3:5]
                if not symptomatic:
                    # Viable routes to presenting
                    #   a: symptomatic -> diagnosed
                    #   b: symptomatic -> progressed -> diagnosed
                    #   c: progressed -> symptomatic -> diagnosed
                    _xi = self.params['endometrial.symptomatic'][3:5]

                    p_a = _xi[0] / (_xi[0] + _lambda) * _delta[0] / (_delta[0] + _lambda)
                    p_b = _xi[0] / (_xi[0] + _lambda) * _lambda / (_delta[0] + _lambda)
                    p_c = _lambda / (_xi[0] + _lambda)
                    Et_a = 1 / (_xi[0] + _lambda) + 1 / (_delta[0] + _lambda)
                    Et_b = 1 / (_xi[0] + _lambda) + 1 / (_delta[0] + _lambda) + 1 / _delta[1]
                    Et_c = 1 / (_xi[0] + _lambda) + 1 / _xi[1] + 1 / _delta[1]

                    return (Et_a * p_a + Et_b * p_b + Et_c * p_c) / (p_a + p_b + p_c)
                else:
                    # Viable routes to presenting while still in Stage III/IV are
                    #   a: diagnosed
                    #   b: progressed -> diagnosed
                    p_a = _delta[0] / (_delta[0] + _lambda)
                    p_b = _lambda / (_delta[0] + _lambda)
                    Et_a = 1 / (_delta[0] + _lambda)
                    Et_b = 1 / (_delta[0] + _lambda) + 1 / _delta[1]

                    return (Et_a * p_a + Et_b * p_b) / (p_a + p_b)
            elif stage is CancerStage.STAGE_IV:
                _delta = self.params['endometrial.diagnosis'][4]
                if not symptomatic:
                    _xi = self.params['endometrial.symptomatic'][4]
                    return 1 / _xi + 1 / _delta
                else:
                    return 1 / _delta
        else:
            raise NotImplementedError()
    
    @staticmethod
    def _get_params_hash(params: Mapping[str, Any]) -> int:
        return hash( (
            # _params_medical_management
            tuple(params['endometrial.aeh_management']),
            params['endometrial.aeh_management.conservative_success'],
            params['endometrial.early_ec.response_rate'],

            # _params_endometrial_calibration
            tuple(params['endometrial.incidence.MLH1']),
            tuple(params['endometrial.incidence.MSH2']),
            tuple(params['endometrial.incidence.MSH6']),
            tuple(params['endometrial.incidence.PMS2']),
            tuple(params['endometrial.incidence.genpop']),
            params['endometrial.regression'],
            tuple(params['endometrial.symptomatic']),
            tuple(params['endometrial.progression']),
            tuple(params['endometrial.diagnosis']),

            # _params_endometrial_survival
            params['endometrial.survival.lynch.late'],
            params['endometrial.recurrence.lynch.early'],
            params['endometrial.postrecurrence.lynch'],
            
            # _params_endometrial_sporadic_survival
            tuple(params['endometrial.recurrence.genpop']),
            params['endometrial.postrecurrence.genpop'],
            params['endometrial.frailty.genpop'],
        ) )

    def signal_mortality(self, cause: CauseOfDeath):
        if not self.stop.processed:
            self.stop.succeed(cause)
    
    def signal_reach_time_horizon(self):
        if not self.stop.processed:
            self.stop.succeed()
    
    def signal_hysterectomy(self):
        if self.hysterectomy.triggered:
            logging.warn("[%.2f, %s] Called signal_hysterectomy but event has already been triggered...", self.env.now, self.individual)
            print(self.individual.events())
        else:
            self.hysterectomy.succeed()
