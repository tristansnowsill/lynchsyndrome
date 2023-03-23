from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from enum import Enum, auto
from functools import partial
from typing import TYPE_CHECKING, Any, Mapping

import numpy
import simpy
from injector import inject
from scipy.integrate import solve_ivp
from scipy.interpolate import BSpline

from lynchsyndrome.death import CauseOfDeath
from lynchsyndrome.diagnosis import CancerSite, CancerStage, DiagnosisMetadata, RouteToDiagnosis
from lynchsyndrome.genetics import ConstitutionalMMR
from lynchsyndrome.inhomogeneous import InhomogeneousPoissonProcess

if TYPE_CHECKING:
    from lynchsyndrome.individual import Individual


class OvarianCancerTrueState(Enum):
    NONE      = auto()
    STAGE_I   = auto()
    STAGE_II  = auto()
    STAGE_III = auto()
    STAGE_IV  = auto()
    UNDEFINED = auto()


class MenopausalState(Enum):
    PREMENOPAUSAL  = auto()
    PERIMENOPAUSAL = auto()
    POSTMENOPAUSAL = auto()


class OvarianCancerObservedState(Enum):
    NONE      = auto()
    STAGE_I   = auto()
    STAGE_II  = auto()
    STAGE_III = auto()
    STAGE_IV  = auto()

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, OvarianCancerTrueState):
            return {
                OvarianCancerTrueState.NONE     : cls.NONE,
                OvarianCancerTrueState.STAGE_I  : cls.STAGE_I,
                OvarianCancerTrueState.STAGE_II : cls.STAGE_II,
                OvarianCancerTrueState.STAGE_III: cls.STAGE_III,
                OvarianCancerTrueState.STAGE_IV : cls.STAGE_IV
            }[value]
        else:
            return super()._missing_(value)

class OvarianSurgicalState(Enum):
    OVARIES_INTACT       = auto()
    ONE_OVARY_REMOVED    = auto()
    BOTH_OVARIES_REMOVED = auto()


@dataclass
class OvarianState:
    ovarian_cancer_true_state:     OvarianCancerTrueState
    ovarian_cancer_observed_state: OvarianCancerObservedState
    menopausal_state:              MenopausalState
    ovarian_surgical_state:        OvarianSurgicalState

healthy_postmenopausal = OvarianState(
    OvarianCancerTrueState.NONE,
    OvarianCancerObservedState.NONE,
    MenopausalState.POSTMENOPAUSAL,
    OvarianSurgicalState.OVARIES_INTACT
)

healthy_premenopausal = OvarianState(
    OvarianCancerTrueState.NONE,
    OvarianCancerObservedState.NONE,
    MenopausalState.PREMENOPAUSAL,
    OvarianSurgicalState.OVARIES_INTACT
)

# class Ovaries:
#     def __init__(
#         self,
#         individual: Individual
#     ):
#         """Create the ovaries for an individual.
        
#         The :py:class:`Ovaries` class handles a number of processes/events,
#         including the development of ovarian lesions (i.e. ovarian cancer),
#         menopause, and the removal of ovaries.
#         """
#         self.individual = individual

#     @property
#     def state(self) -> OvarianState:
#         """Get the summary :py:class:`OvarianState` for this
#         :py:class:`Ovaries` object."""
#         pass

#     @property
#     def utility_effect(self):
#         """Get the effect on utility of the :py:class:`Ovaries` state."""
#         pass


class Ovaries:
    
    incidence_spline_knots  = numpy.array([0.0, 0.0, 0.0, 0.0, 33.33, 66.67, 100.0, 100.0, 100.0, 100.0])
    incidence_spline_degree = 3
    csd_cache = dict()

    @inject
    def __init__(
        self,
        env: simpy.Environment,
        initial_state: OvarianState,
        individual: Individual,
        rng: numpy.random.Generator,
        params: Mapping[str, Any]
    ):
        self.env = env
        self.individual = individual
        self.state = replace(initial_state)
        self.rng = rng
        self.params = params
        self.diagnosed = env.event()
        self.ovaries_removed = env.event()
        self.stop_preclinical = env.event()
    
    def has_occult_oc(self):
        has_oc = self.state.ovarian_cancer_true_state in {OvarianCancerTrueState.STAGE_I, OvarianCancerTrueState.STAGE_II, OvarianCancerTrueState.STAGE_III, OvarianCancerTrueState.STAGE_IV}
        return has_oc and not self.diagnosed.processed
    
    @property
    def ever_diagnosed_ovarian_cancer(self) -> bool:
        return self.state.ovarian_cancer_observed_state is not OvarianCancerObservedState.NONE
    
    def signal_mortality(self, cause: CauseOfDeath):
        if not self.stop_preclinical.triggered:
            self.stop_preclinical.succeed(cause)
    
    def signal_reach_time_horizon(self):
        if not self.stop_preclinical.triggered:
            self.stop_preclinical.succeed()
    
    def signal_diagnosis(self):
        self.diagnosed.succeed()
        self.stop_preclinical.succeed()
    
    def signal_oophorectomy(self):
        self.state.ovarian_surgical_state = OvarianSurgicalState.BOTH_OVARIES_REMOVED
        self.ovaries_removed.succeed()
        if not self.stop_preclinical.triggered:
            self.stop_preclinical.succeed()
    
    def sample_baseline_state(self, init_rng: numpy.random.Generator):
        if self.state.ovarian_cancer_observed_state is not OvarianCancerObservedState.NONE:
            raise NotImplementedError('simulation for patients with known ovarian cancer not implemented')
        
        if self.state.ovarian_surgical_state is not OvarianSurgicalState.OVARIES_INTACT:
            raise NotImplementedError('simulation for patients without both ovaries intact not implemented')
        
        if self.state.ovarian_cancer_true_state is OvarianCancerTrueState.UNDEFINED:
            # We need to sample the true state from the conditional
            # distribution given the individual's age and that they have not
            # been diagnosed with ovarian cancer
            logging.debug('[%.2f, %s] ovarian cancer: sampling true state from conditional distribution', self.env.now, self.individual)
            state_probs = self._get_conditional_state_distribution()
            logging.debug('ovarian cancer: state_probs=%s', state_probs)
            state_index = init_rng.choice(5, p=state_probs)
            logging.debug('ovarian cancer: state_index=%i', state_index)
            self.state.ovarian_cancer_true_state = [ 
                OvarianCancerTrueState.NONE,
                OvarianCancerTrueState.STAGE_I,
                OvarianCancerTrueState.STAGE_II,
                OvarianCancerTrueState.STAGE_III,
                OvarianCancerTrueState.STAGE_IV
             ][state_index]
            logging.debug('ovarian cancer: state=%s', self.state.ovarian_cancer_true_state)
    
    def run(self):
        next_process = {
            OvarianCancerTrueState.NONE: self.run_unaffected,
            OvarianCancerTrueState.STAGE_I: self.run_stage_I_preclinical,
            OvarianCancerTrueState.STAGE_II: self.run_stage_II_preclinical,
            OvarianCancerTrueState.STAGE_III: self.run_stage_III_preclinical,
            OvarianCancerTrueState.STAGE_IV: self.run_stage_IV_preclinical
        }
        self.env.process(next_process[self.state.ovarian_cancer_true_state]())
        yield self.env.event().succeed()
    
    def run_unaffected(self):
        # SAMPLE TIME TO OVARIAN CANCER PRECLINICAL INCIDENCE
        # We use an inhomogeneous poisson process and take the minimum event time (if any)
        # We estimate the maximum value by adding all non-negative coefficients

        # Determine hazard function for ovarian cancer incidence
        incidence_mapping = {
            ConstitutionalMMR.PATH_MLH1: self.params['ovarian.incidence.MLH1'],
            ConstitutionalMMR.PATH_MSH2: self.params['ovarian.incidence.MSH2'],
            ConstitutionalMMR.PATH_MSH6: self.params['ovarian.incidence.MSH6'],
            ConstitutionalMMR.PATH_PMS2: self.params['ovarian.incidence.PMS2'],
            ConstitutionalMMR.WILD_TYPE: self.params['ovarian.incidence.genpop']
        }
        alpha = incidence_mapping[self.individual.genotype]
        bs = self.incidence_spline(alpha)

        ipp_env = numpy.exp(numpy.sum(numpy.clip(alpha, a_min=0, a_max=None)))
        ipp_fn = lambda x: numpy.exp(bs(x))
        ipp = InhomogeneousPoissonProcess(
            ipp_fn,
            (self.individual.age, 100.0),
            [(self.individual.age, 100.0, ipp_env)], self.rng
        )
        ipp_sample = ipp.sample(output_format='numpy')
        age_incidence = None
        if len(ipp_sample) >= 1:
            age_incidence = numpy.min(ipp_sample)
        if age_incidence is not None:
            logging.info('[%.2f, %s] ovarian cancer: age of ovarian cancer incidence=%f', self.env.now, self.individual, age_incidence)
            e1, e2, e3 = self.env.timeout(age_incidence - self.individual.age, value='preclinical_incidence'), self.ovaries_removed, self.stop_preclinical
            yield e1 | e2 | e3
            if e1.processed:
                self.state.ovarian_cancer_true_state = OvarianCancerTrueState.STAGE_I
                self.env.process(self.run_stage_I_preclinical())
                # self._preclinical_incidence(self.env)
        else:
            logging.info('[%.2f, %s] ovarian cancer: no cancer before age 100', self.env.now, self.individual)
            yield self.env.event().succeed()

    def run_stage_I_preclinical(self):
        logging.info('[%.2f, %s] ovarian cancer: preclinical stage I...', self.env.now, self.individual)
        progression = self.params['ovarian.progression.genpop'] if self.individual.genotype is ConstitutionalMMR.WILD_TYPE else self.params['ovarian.progression.lynch']
        t_prog = self.rng.exponential(scale=1/progression[0])
        t_pres = self.rng.exponential(scale=1/self.params['ovarian.presentation'][0])
        e_prog = self.env.timeout(t_prog, 'progression')
        e_pres = self.env.timeout(t_pres, 'presentation')

        val = yield e_prog | e_pres | self.diagnosed | self.stop_preclinical

        if e_prog.processed:
            self.state.ovarian_cancer_true_state = OvarianCancerTrueState.STAGE_II
            self.env.process(self.run_stage_II_preclinical())
        elif e_pres.processed:
            self.env.process(self.run_oc_diagnosed(RouteToDiagnosis.SYMPTOMATIC_PRESENTATION))


    def run_stage_II_preclinical(self):
        logging.info('[%.2f, %s] ovarian cancer: preclinical stage II...', self.env.now, self.individual)
        progression = self.params['ovarian.progression.genpop'] if self.individual.genotype is ConstitutionalMMR.WILD_TYPE else self.params['ovarian.progression.lynch']
        t_prog = self.rng.exponential(scale=1/progression[1])
        t_pres = self.rng.exponential(scale=1/self.params['ovarian.presentation'][1])
        e_prog = self.env.timeout(t_prog, 'progression')
        e_pres = self.env.timeout(t_pres, 'presentation')

        val = yield e_prog | e_pres | self.diagnosed | self.stop_preclinical

        if e_prog.processed:
            self.state.ovarian_cancer_true_state = OvarianCancerTrueState.STAGE_III
            self.env.process(self.run_stage_III_preclinical())
        elif e_pres.processed:
            self.env.process(self.run_oc_diagnosed(RouteToDiagnosis.SYMPTOMATIC_PRESENTATION))

    def run_stage_III_preclinical(self):
        logging.info('[%.2f, %s] ovarian cancer: preclinical stage III...', self.env.now, self.individual)
        progression = self.params['ovarian.progression.genpop'] if self.individual.genotype is ConstitutionalMMR.WILD_TYPE else self.params['ovarian.progression.lynch']
        t_prog = self.rng.exponential(scale=1/progression[2])
        t_pres = self.rng.exponential(scale=1/self.params['ovarian.presentation'][2])
        e_prog = self.env.timeout(t_prog, 'progression')
        e_pres = self.env.timeout(t_pres, 'presentation')

        val = yield e_prog | e_pres | self.diagnosed | self.stop_preclinical

        if e_prog.processed:
            self.state.ovarian_cancer_true_state = OvarianCancerTrueState.STAGE_IV
            self.env.process(self.run_stage_IV_preclinical())
        elif e_pres.processed:
            self.env.process(self.run_oc_diagnosed(RouteToDiagnosis.SYMPTOMATIC_PRESENTATION))

    def run_stage_IV_preclinical(self):
        logging.info('[%.2f, %s] ovarian cancer: preclinical stage IV...', self.env.now, self.individual)
        t_pres = self.rng.exponential(scale=1/self.params['ovarian.presentation'][3])
        e_pres = self.env.timeout(t_pres, 'presentation')

        val = yield e_pres | self.diagnosed | self.stop_preclinical

        if e_pres.processed:
            self.env.process(self.run_oc_diagnosed(RouteToDiagnosis.SYMPTOMATIC_PRESENTATION))
    
    def run_oc_diagnosed(self, route: RouteToDiagnosis, synchronous_ec: bool = False):
        self.state.ovarian_cancer_observed_state = OvarianCancerObservedState(self.state.ovarian_cancer_true_state)
        self.signal_diagnosis()
        
        if self.individual.has_uterus and not synchronous_ec:
            if self.individual.endometrium.hysterectomy.triggered:
                logging.warning("[%.2f, %s] clinical_presentation is about to call run_hysterectomy but hysterectomy event has already been triggered", self.env.now, self.individual)
            self.env.process(self.individual.endometrium.run_hysterectomy())

        self.state.ovarian_surgical_state = OvarianSurgicalState.BOTH_OVARIES_REMOVED
        logging.info('[%.2f, %s] ovarian cancer: diagnosed (genotype=%s) (stage=%s)', self.env.now, self.individual, self.individual.genotype, self.state.ovarian_cancer_observed_state)
        yield self.env.process(self.individual.record_ovarian_cancer_diagnosis(
            route,
            CancerStage(self.state.ovarian_cancer_true_state)
        ))

        # Include some lead time if appropriate
        if route is not RouteToDiagnosis.SYMPTOMATIC_PRESENTATION:
            t_lead_time = self._lead_time(self.state, self.individual.genotype is not ConstitutionalMMR.WILD_TYPE)
            e_lead_time = self.env.timeout(t_lead_time)

            yield e_lead_time | self.individual.died | self.individual.reach_time_horizon

            if not e_lead_time.processed:
                return

        # Run survival
        if self.individual.genotype is ConstitutionalMMR.WILD_TYPE:
            self.env.process(self.run_ovarian_survival_sporadic())
        else:
            self.env.process(self.run_ovarian_survival_lynch())

    def run_ovarian_survival_sporadic(self):
        yield self.env.event().fail(NotImplementedError("not yet implemented clinical presentation of sporadic ovarian cancer"))
    
    def run_ovarian_survival_lynch(self):
        if self.state.ovarian_cancer_observed_state in (OvarianCancerObservedState.STAGE_I, OvarianCancerObservedState.STAGE_II):
            logging.info('[%.2f, %s] ovarian cancer: Stage I or II, simulate time to recurrence', self.env.now, self.individual)
            x = self.rng.standard_exponential()
            alpha = self.params['ovarian.recurrence.lynch.rate']
            beta = self.params['ovarian.recurrence.lynch.shape']
            if beta / alpha * x > -1.0:
                t_recur = numpy.log(1 + beta / alpha * x) / beta
                e_recur = self.env.timeout(t_recur)
                yield e_recur | self.individual.died | self.individual.reach_time_horizon
                if e_recur.processed:
                    logging.info('[%.2f, %s] ovarian cancer: recurrence', self.env.now, self.individual)
                    yield self.env.process(self.individual.record_ovarian_cancer_recurrence(
                        CancerStage(self.state.ovarian_cancer_true_state)
                    ))
                    t_die = self.params['ovarian.survival.lynch.scale'] * self.rng.weibull(self.params['ovarian.survival.lynch.shape'])
                    e_die = self.env.timeout(t_die)
                    yield e_die | self.individual.died | self.individual.reach_time_horizon
                    if e_die.processed:
                        logging.info('[%.2f, %s] ovarian cancer: death from ovarian cancer', self.env.now, self.individual)
                        yield self.env.process(self.individual.record_death(CauseOfDeath.OVARIAN_CANCER))
            else:
                logging.info('[%.2f, %s] ovarian cancer: will never recur', self.env.now, self.individual)
                yield self.env.event().succeed()
        else:
            logging.info('[%.2f, %s] Stage III or IV, simulate time to death', self.env.now, self.individual)
            t_die = self.params['ovarian.survival.lynch.scale'] * self.rng.weibull(self.params['ovarian.survival.lynch.shape'])
            e_die = self.env.timeout(t_die)
            yield e_die | self.individual.died | self.individual.reach_time_horizon
            if e_die.processed:
                logging.info('[%.2f, %s] ovarian cancer: death from ovarian cancer', self.env.now, self.individual)
                yield self.env.process(self.individual.record_death(CauseOfDeath.OVARIAN_CANCER))
    
    def _lead_time(self, ovarian_state: OvarianState, lynch: bool):
        mappings = {
            OvarianCancerTrueState.STAGE_I: {
                'as_observed': OvarianCancerObservedState.STAGE_I,
                'as_generic': CancerStage.STAGE_I,
                'as_int': 1
            },
            OvarianCancerTrueState.STAGE_II: {
                'as_observed': OvarianCancerObservedState.STAGE_II,
                'as_generic': CancerStage.STAGE_II,
                'as_int': 2
            },
            OvarianCancerTrueState.STAGE_III: {
                'as_observed': OvarianCancerObservedState.STAGE_III,
                'as_generic': CancerStage.STAGE_III,
                'as_int': 3
            },
            OvarianCancerTrueState.STAGE_IV: {
                'as_observed': OvarianCancerObservedState.STAGE_IV,
                'as_generic': CancerStage.STAGE_IV,
                'as_int': 4
            }
        }[ovarian_state.ovarian_cancer_true_state]
        if not lynch:
            prog = self.params['ovarian.progression.genpop'][mappings['as_int']-1]
            pres = self.params['ovarian.presentation'][mappings['as_int']-1]
            return 1 / (prog + pres)
        else:
            if ovarian_state.ovarian_cancer_true_state is OvarianCancerTrueState.STAGE_I:
                prog0 = self.params['ovarian.progression.lynch'][0]
                prog1 = self.params['ovarian.progression.lynch'][1]
                pres0 = self.params['ovarian.presentation'][0]
                pres1 = self.params['ovarian.presentation'][1]
                return (prog0 + pres1 + prog1) / ((pres0 + prog0) * (pres1 + prog1))
            elif ovarian_state.ovarian_cancer_true_state is OvarianCancerTrueState.STAGE_II:
                prog = self.params['ovarian.progression.lynch'][1]
                pres = self.params['ovarian.presentation'][1]
                return 1 / (prog + pres)
            elif ovarian_state.ovarian_cancer_true_state is OvarianCancerTrueState.STAGE_III:
                prog0 = self.params['ovarian.progression.lynch'][2]
                pres0 = self.params['ovarian.presentation'][2]
                pres1 = self.params['ovarian.presentation'][3]
                return (prog0 + pres1) / ((pres0 + prog0) * pres1)
            elif ovarian_state.ovarian_cancer_true_state is OvarianCancerTrueState.STAGE_IV:
                pres = self.params['ovarian.presentation'][3]
                return 1 / pres

    
    @classmethod
    def incidence_spline(cls, alpha) -> BSpline:
        return BSpline(cls.incidence_spline_knots, alpha, cls.incidence_spline_degree)

    @staticmethod
    def _dydt(t, y, log_incidence_spline, progression, presentation):
        res = numpy.zeros(9)
        
        incidence = numpy.exp(log_incidence_spline(t))
        res[0] = -incidence * y[0]
        res[1] = incidence * y[0] - (progression[0] + presentation[0]) * y[1]
        res[2] = progression[0] * y[1] - (progression[1] + presentation[1]) * y[2]
        res[3] = progression[1] * y[2] - (progression[2] + presentation[2]) * y[3]
        res[4] = progression[2] * y[3] - presentation[3] * y[4]
        res[5] = presentation[0] * y[1]
        res[6] = presentation[1] * y[2]
        res[7] = presentation[2] * y[3]
        res[8] = presentation[3] * y[4]

        return res
    
    @staticmethod
    def _jac(t, y, log_incidence_spline, progression, presentation):
        res = numpy.zeros((9, 9))

        incidence = numpy.exp(log_incidence_spline(t))

        res[1,0] = incidence
        res[0,0] = -incidence
        res[1,1] = -(progression[0] + presentation[0])
        res[2,2] = -(progression[1] + presentation[1])
        res[3,3] = -(progression[2] + presentation[2])
        res[4,4] = -presentation[3]
        res[2,1] = progression[0]
        res[3,2] = progression[1]
        res[4,3] = progression[2]
        res[5,1] = presentation[0]
        res[6,2] = presentation[1]
        res[7,3] = presentation[2]
        res[8,4] = presentation[3]

        return res

    def _get_conditional_state_distribution(self):
        # Check the cache...
        key = (id(self.params), self.individual.genotype, numpy.round(self.individual.age))
        if key in Ovaries.csd_cache:
            return Ovaries.csd_cache[key]
        
        y0 = numpy.zeros(9)
        y0[0] = 1.0
        progression  = self.params['ovarian.progression.genpop'] if self.individual.genotype is ConstitutionalMMR.WILD_TYPE else self.params['ovarian.progression.lynch']
        presentation = self.params['ovarian.presentation']

        # Determine hazard function for ovarian cancer incidence
        incidence_mapping = {
            ConstitutionalMMR.PATH_MLH1: self.params['ovarian.incidence.MLH1'],
            ConstitutionalMMR.PATH_MSH2: self.params['ovarian.incidence.MSH2'],
            ConstitutionalMMR.PATH_MSH6: self.params['ovarian.incidence.MSH6'],
            ConstitutionalMMR.PATH_PMS2: self.params['ovarian.incidence.PMS2'],
            ConstitutionalMMR.WILD_TYPE: self.params['ovarian.incidence.genpop']
        }
        alpha = incidence_mapping[self.individual.genotype]
        bs = self.incidence_spline(alpha)

        ode_res = solve_ivp(
            fun=partial(
                Ovaries._dydt,
                log_incidence_spline=bs,
                progression=progression,
                presentation=presentation),
            t_span=(0.0, 100.0),
            t_eval=(self.individual.age,),
            y0=y0,
            method='LSODA',
            jac=partial(
                Ovaries._jac,
                log_incidence_spline=bs,
                progression=progression,
                presentation=presentation)
        )
        logging.debug('ovarian cancer: ode_res.y = %s', ode_res.y)
        state_probs = numpy.squeeze(ode_res.y[0:5,0])
        state_probs = numpy.clip(state_probs, 0.0, 1.0)
        state_probs = state_probs / numpy.sum(state_probs)

        Ovaries.csd_cache[key] = state_probs
        return state_probs
