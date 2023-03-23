from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import logging
import uuid
from collections.abc import Collection
from typing import Callable, Optional, Union

import numpy
import pandas
import simpy


from .bowel import Bowel, BowelState
from .death import CauseOfDeath, DeathMetadata
from .diagnosis import CancerSite, CancerStage, DiagnosisMetadata, RouteToDiagnosis
from .endometrium import Endometrium
from .genetics import ConstitutionalMMR
from .ovaries import MenopausalState, OvarianCancerObservedState, OvarianCancerTrueState, OvarianState, OvarianSurgicalState, Ovaries
from .recurrence import RecurrenceMetadata
from .reporting import EventMetadata, ReportingEvent
from .risk_reducing_surgery import RiskReducingSurgery, RiskReducingSurgeryMetadata
from .sex import Sex


@dataclass
class EnterModelMetadata(EventMetadata):
    uuid:     uuid.UUID
    age:      float
    genotype: ConstitutionalMMR
    sex:      Sex


class Individual:

    def __init__(
        self,
        env: simpy.Environment,
        rng: numpy.random.Generator,
        params
    ):
        self.env = env
        self.rng = rng
        self.params = params

        self.alive = True
        self.sex = None
        self.age = None
        self.bowel      : Optional[Bowel]             = None
        self.endometrium: Optional[Endometrium]       = None
        self.ovaries    : Optional[Ovaries]           = None
        self.genotype   : Optional[ConstitutionalMMR] = None
        self.uuid       : Optional[uuid.UUID]         = None

        self.event_trace = list()

        self.died = env.event()
        self.reach_time_horizon = env.event()

    def sample_baseline_state(self, init_rng: numpy.random.Generator):
        self.bowel.sample_baseline_state(init_rng)
        if self.sex is Sex.FEMALE:
            self.endometrium.sample_baseline_state(init_rng)
            self.ovaries.sample_baseline_state(init_rng)
    
    def __str__(self):
        return str(self.uuid)[0:8] + "..."
    
    @property
    def current_age(self) -> float:
        return self.age + self.env.now
    
    @property
    def has_ovaries(self) -> bool:
        if self.sex is Sex.MALE:
            return False
        
        return self.ovaries.state.ovarian_surgical_state is not OvarianSurgicalState.BOTH_OVARIES_REMOVED
    
    @property
    def has_uterus(self) -> bool:
        if self.sex is Sex.MALE:
            return False
        
        return not self.endometrium.hysterectomy.triggered
    
    @property
    def ever_diagnosed_ovarian_cancer(self) -> bool:
        if self.sex is Sex.MALE:
            return False
        
        return self.ovaries.ever_diagnosed_ovarian_cancer
    
    @property
    def ever_diagnosed_endometrial_cancer(self) -> bool:
        if self.sex is Sex.MALE:
            return False
        
        return self.endometrium.has_diagnosed_ec
    
    def run(self):
        self.event_trace.append((self.env.now, ReportingEvent.ENTER_MODEL, EnterModelMetadata(self.uuid, self.age, self.genotype, self.sex)))
        self.env.process(self.run_general_mortality())
        self.env.process(self.run_time_horizon())
        self.env.process(self.bowel.run())
        if self.sex is Sex.FEMALE:
            self.env.process(self.ovaries.run())
            self.env.process(self.endometrium.run())
        yield self.env.event().succeed()
    
    def run_time_horizon(self):
        # The time horizon for this individual is when they reach age 100 or
        # when self.env.now reaches analysis.time_horizon (whichever is
        # earlier)
        t_age_horizon = 100 - self.age
        t_time_horizon = self.params['analysis.time_horizon'] - self.env.now
        e_horizon = self.env.timeout(min(t_age_horizon, t_time_horizon))
        yield e_horizon | self.died
        if e_horizon.processed:
            self.event_trace.append( (self.env.now, ReportingEvent.REACH_TIME_HORIZON, EventMetadata()) )
            self.reach_time_horizon.succeed()
            # Cascade message to other classes as necessary
            self.bowel.signal_reach_time_horizon()
            if self.sex is Sex.FEMALE:
                self.endometrium.signal_reach_time_horizon()
                self.ovaries.signal_reach_time_horizon()
        
    
    def run_general_mortality(self):
        # Grab the relevant parameters
        beta = self.params['general_mortality.age.male'] if self.sex is Sex.MALE else self.params['general_mortality.age.female']
        coef_y0 = self.params['general_mortality.y0.male'] if self.sex is Sex.MALE else self.params['general_mortality.y0.female']
        coef_y1 = self.params['general_mortality.y1.male'] if self.sex is Sex.MALE else self.params['general_mortality.y1.female']
        coef_cons = self.params['general_mortality.cons.male'] if self.sex is Sex.MALE else self.params['general_mortality.cons.female']
        # Calculate y0 and y1 terms
        year = self.params['analysis.year']
        year_of_birth = year - self.age
        y0 = min(year_of_birth - 1975, 0)
        y1 = max(year_of_birth - 1975, 0)
        # Calculate alpha term
        alpha = numpy.exp(coef_y0 * y0 + coef_y1 * y1 + coef_cons)
        # Sample from Gompertz distribution
        x = self.rng.exponential(scale=beta/alpha)
        age_gen_mort = numpy.log(numpy.exp(beta * self.age) + x) / beta
        t_gen_mort = age_gen_mort - self.age
        logging.debug("[%.2f, %s] Sampled age of general mortality=%.2f (in %.2f years time)", self.env.now, self, age_gen_mort, t_gen_mort)
        e_gen_mort = self.env.timeout(t_gen_mort)
        yield e_gen_mort | self.died | self.reach_time_horizon
        if e_gen_mort.processed:
            logging.info("[%.2f, %s] Dying from general mortality", self.env.now, self)
            yield self.env.process(self.record_death(CauseOfDeath.OTHER_CAUSES))

    def run_rrgs(self, surgery: Optional[RiskReducingSurgery] = RiskReducingSurgery.HBSO):
        """Risk-reducing gynaecological surgery"""
        surgery = surgery.adjust(self.has_ovaries, self.has_uterus)

        yield self.env.process(self.record_event(
            ReportingEvent.RISK_REDUCING_SURGERY,
            RiskReducingSurgeryMetadata(surgery))
        )
        
        if surgery.affects_uterus():
            if self.endometrium.hysterectomy.triggered:
                logging.warning("[%.2f, %s] run_rrgs is about to call run_hysterectomy but hysterectomy event has already been triggered", self.env.now, self)
            yield self.env.process(self.endometrium.run_hysterectomy())
        
        if surgery.affects_ovaries():
            if not self.ovaries.has_occult_oc():
                self.ovaries.signal_oophorectomy()
            else:
                self.env.process(self.ovaries.run_oc_diagnosed(RouteToDiagnosis.RISK_REDUCING_SURGERY))
        
        yield self.env.event().succeed()
    
    def record_event(self, event: ReportingEvent, metadata: EventMetadata):
        """Process for recording a generic event
        
        Notes:
        - This is a generator, so must be called with env.process(...)
        - This process does not take any time, so it is safe to yield for synchronous execution
        - This process does not register any other processes with the environment
        """
        self.event_trace.append((self.env.now, event, metadata))
        yield self.env.event().succeed()
    
    def record_endometrial_cancer_diagnosis(self, route: RouteToDiagnosis, stage: CancerStage):
        """Process for recording an endometrial cancer diagnosis
        
        Notes:
        - This is a generator, so must be called with env.process(...)
        - This does not contain any endometrial cancer logic, it only adds to the event trace
        - This will raise a RunTime error if the individual has already died
        - This process does not take any time, so it is safe to yield for synchronous execution
        - This process does not register any other processes with the environment
        """
        if not self.alive:
            yield self.env.event().fail(RuntimeError("record_endometrial_cancer_diagnosis called but individual is already dead"))
        self.event_trace.append(
            (
                self.env.now,
                ReportingEvent.CANCER_DIAGNOSIS,
                DiagnosisMetadata(route, CancerSite.ENDOMETRIUM, stage)
            )
        )
        yield self.env.event().succeed()

    def record_endometrial_cancer_recurrence(self, stage_at_diagnosis: CancerStage):
        """Process for recording an endometrial cancer recurrence
        
        Notes:
        - This is a generator, so must be called with env.process(...)
        - This does not contain any endometrial cancer logic, it only adds to the event trace
        - This will raise a RunTime error if the individual has already died
        - This process does not take any time, so it is safe to yield for synchronous execution
        - This process does not register any other processes with the environment
        """
        if not self.alive:
            yield self.env.event().fail(RuntimeError("record_endometrial_cancer_recurrence called but individual is already dead"))
        self.event_trace.append(
            (
                self.env.now,
                ReportingEvent.CANCER_RECURRENCE,
                RecurrenceMetadata(CancerSite.ENDOMETRIUM, stage_at_diagnosis)
            )
        )
        yield self.env.event().succeed()

    def record_colorectal_cancer_diagnosis(self, route: RouteToDiagnosis, stage: CancerStage):
        """Process for recording a colorectal cancer diagnosis
        
        Notes:
        - This is a generator, so must be called with env.process(...)
        - This does not contain any colorectal cancer logic, it only adds to the event trace
        - This will raise a RunTime error if the individual has already died
        - This process does not take any time, so it is safe to yield for synchronous execution
        - This process does not register any other processes with the environment
        """
        if not self.alive:
            yield self.env.event().fail(RuntimeError("record_colorectal_cancer_diagnosis called but individual is already dead"))
        self.event_trace.append(
            (
                self.env.now,
                ReportingEvent.CANCER_DIAGNOSIS,
                DiagnosisMetadata(route, CancerSite.COLORECTUM, stage)
            )
        )
        yield self.env.event().succeed()

    def record_ovarian_cancer_diagnosis(self, route: RouteToDiagnosis, stage: CancerStage):
        """Process for recording an ovarian cancer diagnosis
        
        Notes:
        - This is a generator, so must be called with env.process(...)
        - This does not contain any ovarian cancer logic, it only adds to the event trace
        - This will raise a RunTime error if the individual has already died
        - This process does not take any time, so it is safe to yield for synchronous execution
        - This process does not register any other processes with the environment
        """
        if not self.alive:
            yield self.env.event().fail(RuntimeError("record_ovarian_cancer_diagnosis called but individual is already dead"))
        self.event_trace.append(
            (
                self.env.now,
                ReportingEvent.CANCER_DIAGNOSIS,
                DiagnosisMetadata(route, CancerSite.OVARIES, stage)
            )
        )
        yield self.env.event().succeed()
    
    def record_ovarian_cancer_recurrence(self, stage_at_diagnosis: CancerStage):
        """Process for recording an ovarian cancer recurrence
        
        Notes:
        - This is a generator, so must be called with env.process(...)
        - This does not contain any ovarian cancer logic, it only adds to the event trace
        - This will raise a RunTime error if the individual has already died
        - This process does not take any time, so it is safe to yield for synchronous execution
        - This process does not register any other processes with the environment
        """
        if not self.alive:
            yield self.env.event().fail(RuntimeError("record_ovarian_cancer_recurrence called but individual is already dead"))
        self.event_trace.append(
            (
                self.env.now,
                ReportingEvent.CANCER_RECURRENCE,
                RecurrenceMetadata(CancerSite.OVARIES, stage_at_diagnosis)
            )
        )
        yield self.env.event().succeed()

    
    def record_death(self, cause: CauseOfDeath):
        """Process for recording death of the individual
        
        Notes:
        - This is a generator, so must be called with env.process(...)
        - This will raise a RunTime error if the individual has already died
        - This process does not take any time, so it is safe to yield for synchronous execution
        - This process uses signal_XXX calls to inform Bowel, Endometrium and Ovaries that the individual has died
        - This process does not register any other processes with the environment
        """
        if self.alive:
            yield self.died.succeed()
            self.alive = False
            self.event_trace.append( (self.env.now, ReportingEvent.DEATH, DeathMetadata(cause)) )
            # Cascade message to other classes as necessary
            self.bowel.signal_mortality(cause)
            if self.sex is Sex.FEMALE:
                self.endometrium.signal_mortality(cause)
                self.ovaries.signal_mortality(cause)
        else:
            yield self.env.event().fail(RuntimeError("record_death called but individual is already dead"))
    
    def events(self, include_uuid: Optional[bool] = False) -> pandas.DataFrame:
        """Return the events recorded for this individual
        
        :param include_uuid:
            If True, the returned pandas.DataFrame will include a column `uuid`.
            If False (the default), this column will not be included.
        """
        if include_uuid:
            res = pandas.DataFrame(self.event_trace, columns=('time','event','metadata'))
            res.insert(0, 'uuid', self.uuid)
        else:
            return pandas.DataFrame(self.event_trace, columns=('time','event','metadata'))
    

class IndividualBuilder:
    """A Builder for the ``Individual`` class
    
    This class defines a fluent API for creating ``Individual`` instances
    allowing you to write code such as:

    .. code-block:: python

        env = simpy.Environment()
        builder = IndividualBuilder(env)
        individual = builder.set_sex(Sex.MALE).set_age(50).create()
    
    It will attempt to use sensible defaults for any attributes not specified.

    You can pass callables to the set_XXX functions (e.g. to sample from a
    probability distribution or read from a file) - they will be called once
    for each ``Individual`` created.

    You can create many ``Individual`` instances with the same signature using
    the ``create_many(n)`` function.
    """
    def __init__(
        self,
        env: simpy.Environment,
        init_rng: numpy.random.Generator,
        sim_rng: numpy.random.Generator,
        params
    ):
        self.env = env
        self.init_rng = init_rng
        self.sim_rng = sim_rng
        self.params = params
        self.genotype = ConstitutionalMMR.WILD_TYPE
        self.sex = Sex.FEMALE
        self.age = 30.0
        self.uuid = uuid.uuid4
        self.bowel_state = BowelState.UNDEFINED
        self.ovarian_state = OvarianState(
            OvarianCancerTrueState.UNDEFINED,
            OvarianCancerObservedState.NONE,
            MenopausalState.PREMENOPAUSAL,
            OvarianSurgicalState.OVARIES_INTACT
        )

    def set_sex(
        self,
        sex: Union[Sex, Callable[[], Sex]]
    ) -> IndividualBuilder:
        """Set the sex for objects created by the builder.

        :param sex: The Sex for a new Individual - this can be a constant or a
            Callable which will be called as ``sex()`` for each new Individual
            to determine the Sex
        :return: The `IndividualBuilder` instance with the sex updated so that
            calls can be chained in a fluent API
        """
        self.sex = sex
        return self

    def set_age(
        self,
        age: Union[float, Callable[[], float]]
    ) -> IndividualBuilder:
        """Set the age for objects created by the builder.
        
        :param age: The age for a new Individual - this can be a constant or a
            Callable which will be called as ``age()`` for each new Individual
            to determine the age
        :return: The `IndividualBuilder` instance with the age updated so that
            calls can be chained in a fluent API
        """
        self.age = age
        return self
    
    def set_bowel_state(
        self,
        bowel_state: Union[BowelState, Callable[[], BowelState]]
    ) -> IndividualBuilder:
        self.bowel_state = bowel_state
        return self
    
    def set_ovarian_state(
        self,
        ovarian_state: Union[OvarianState, Callable[[], OvarianState]]
    ) -> IndividualBuilder:
        self.ovarian_state = ovarian_state
        return self
    
    def set_genotype(
        self,
        genotype: Union[ConstitutionalMMR, Callable[[], ConstitutionalMMR]]
    ) -> IndividualBuilder:
        self.genotype = genotype
        return self

    def set_uuid(
        self,
        uuid: Union[uuid.UUID, Callable[[], uuid.UUID]]
    ) -> IndividualBuilder:
        """Set the UUID for `Individual` created by the builder.
        
        :param uuid: The UUID for a new Individual - this can be a constant or
            a Callable which will be called as ``uuid()`` for each new
            `Individual` to determine the UUID
        :return: The `IndividualBuilder` instance with the age updated so that
            calls can be chained in a fluent API
        """
        self.uuid = uuid
        return self

    def create(self) -> Individual:
        individual = Individual(self.env, self.sim_rng, self.params)

        if callable(self.sex):
            individual.sex = self.sex()
        else:
            individual.sex = self.sex

        if callable(self.age):
            individual.age = self.age()
        else:
            individual.age = self.age

        if callable(self.genotype):
            individual.genotype = self.genotype()
        else:
            individual.genotype = self.genotype

        if callable(self.uuid):
            individual.uuid = self.uuid()
        else:
            individual.uuid = self.uuid
        
        
        individual.bowel = Bowel(
            self.env,
            self.sim_rng,
            self.params,
            individual,
            self.bowel_state if not callable(self.bowel_state) else self.bowel_state()
        )

        # Create uterus if applicable
        # Create ovaries if applicable
        if individual.sex is Sex.FEMALE:
            individual.ovaries = Ovaries(
                self.env,
                self.ovarian_state,
                individual,
                self.sim_rng,
                self.params
            )
            individual.endometrium = Endometrium(
                self.env,
                self.sim_rng,
                self.params,
                individual
            )
        
        individual.sample_baseline_state(self.init_rng)

        return individual

    def create_many(self, n: int) -> Collection[Individual]:
        return [self.create() for _ in range(n)]
