from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Iterable, List, Mapping, Optional, Tuple
from uuid import UUID

import pandas

from lynchsyndrome.benefits import Benefits
from lynchsyndrome.costing import Costing
from lynchsyndrome.death import CauseOfDeath, DeathMetadata
from lynchsyndrome.diagnosis import CancerSite, DiagnosisMetadata
from lynchsyndrome.individual import EnterModelMetadata
from lynchsyndrome.recurrence import RecurrenceMetadata
from lynchsyndrome.reporting import ReportingEvent
from lynchsyndrome.risk_reducing_surgery import RiskReducingSurgery, RiskReducingSurgeryMetadata
from lynchsyndrome.surveillance import SurveillanceMetadata, SurveillanceSite

if TYPE_CHECKING:
    from lynchsyndrome.individual import Individual
    from lynchsyndrome.experiments.common.experiment import InterimResult


class ParameterReport:

    @staticmethod
    def get_nested_params(params: pandas.DataFrame) -> pandas.DataFrame:
        new_index = pandas.Index([str(uuid) for uuid in params.index.array], name='params_uuid')
        res = params
        res.index = new_index
        return res


class PatientLevelCostReport:
    """A Report which gives the total (discounted) cost for each simulated patient"""

    @staticmethod
    def process(event_traces: pandas.DataFrame, individuals: pandas.Series, params: Mapping[str, Any]) -> InterimResult:
        # Get ready to calculate costs by supplying params to the costing
        # calculator
        costing = Costing(params)
        
        # We need to supply to costing one individual at a time
        grouped_events = event_traces.groupby(level=['pathway_uuid', 'individual_uuid'])
        grouped_individuals = individuals.groupby(level=['pathway_uuid', 'individual_uuid'])
        total_costs = dict()
        for (pathway_uuid, individual_uuid), events_df in grouped_events:
            # Grab the individual
            individual = grouped_individuals.get_group((pathway_uuid, individual_uuid)).item()

            # Calculate the costs
            event_costs = costing.event_costs(events_df, individual)
            running_costs = costing.running_costs(events_df, individual)

            # Add to the dict
            total_costs[(pathway_uuid, individual_uuid)] = event_costs.sum() + running_costs.sum()
        
        # Concatenate the total costs
        return pandas.Series(
            total_costs.values(),
            index=pandas.MultiIndex.from_tuples(total_costs.keys(), names=['pathway_uuid', 'individual_uuid']),
            name='total_costs'
        )


    @staticmethod
    def aggregate(interim_results: Iterable[Tuple[UUID, InterimResult]]):
        return pandas.concat(
            (t[1] for t in interim_results),
            axis=0,
            keys=(t[0] for t in interim_results),
            names=['params_uuid', 'pathway_uuid', 'individual_uuid']
        )


class PatientLevelLifeYearsReport:
    """A Report which gives the total (discounted) life years for each simulated patient"""

    @staticmethod
    def process(event_traces: pandas.DataFrame, individuals: pandas.Series, params: Mapping[str, Any]) -> InterimResult:
        # Get ready to calculate costs by supplying params to the costing
        # calculator
        benefits = Benefits(params, None, None)
        
        # We need to supply to costing one individual at a time
        grouped_events = event_traces.groupby(level=['pathway_uuid', 'individual_uuid'])
        total_lys = dict()
        for (pathway_uuid, individual_uuid), events_df in grouped_events:
            # Add to the dict
            total_lys[(pathway_uuid, individual_uuid)] = benefits.life_years(events_df).sum()
        
        # Concatenate the total costs
        return pandas.Series(
            total_lys.values(),
            index=pandas.MultiIndex.from_tuples(total_lys.keys(), names=['pathway_uuid', 'individual_uuid']),
            name='total_life_years'
        )


    @staticmethod
    def aggregate(interim_results: Iterable[Tuple[UUID, InterimResult]]):
        return pandas.concat(
            (t[1] for t in interim_results),
            axis=0,
            keys=(t[0] for t in interim_results),
            names=['params_uuid', 'pathway_uuid', 'individual_uuid']
        )


class PatientLevelQALYsReport:
    """A Report which gives the total (discounted) QALYs for each simulated patient"""

    def __init__(self, utility_combiner, baseline_utility):
        self.utility_combiner = utility_combiner
        self.baseline_utility = baseline_utility

    def process(self, event_traces: pandas.DataFrame, individuals: pandas.Series, params: Mapping[str, Any]) -> InterimResult:
        # Get ready to calculate costs by supplying params to the costing
        # calculator
        benefits = Benefits(params, self.utility_combiner, self.baseline_utility)
        
        # We need to supply to costing one individual at a time
        grouped_events = event_traces.groupby(level=['pathway_uuid', 'individual_uuid'])
        grouped_individuals = individuals.groupby(level=['pathway_uuid', 'individual_uuid'])
        total_qalys = dict()
        for (pathway_uuid, individual_uuid), events_df in grouped_events:
            # Grab the individual
            individual = grouped_individuals.get_group((pathway_uuid, individual_uuid)).item()

            # Remove levels from MultiIndex to avoid bug
            events_df.index = events_df.index.droplevel(['pathway_uuid', 'individual_uuid'])

            # Add to the dict
            total_qalys[(pathway_uuid, individual_uuid)] = benefits.qalys(individual, events_df).sum()
        
        # Concatenate the total costs
        return pandas.Series(
            total_qalys.values(),
            index=pandas.MultiIndex.from_tuples(total_qalys.keys(), names=['pathway_uuid', 'individual_uuid']),
            name='total_qalys'
        )

    def aggregate(self, interim_results: Iterable[Tuple[UUID, InterimResult]]):
        return pandas.concat(
            (t[1] for t in interim_results),
            axis=0,
            keys=(t[0] for t in interim_results),
            names=['params_uuid', 'pathway_uuid', 'individual_uuid']
        )

class SurvivalType(Enum):
    ALL_CAUSE = auto()
    """Failure includes death from any cause"""

    CAUSE_SPECIFIC = auto()
    """Failure only includes the specific cause of interest"""

    CRUDE = ALL_CAUSE
    """Alias for ``SurvivalType.ALL_CAUSE``"""


class CancerOutcomes:

    @staticmethod
    def _cancer_count_row_transform(row):
        # Assume we have already selected rows so we only have events
        # relevant to the chosen cancer
        col_names = ['cancer_outcome', 'stage_at_diagnosis', 'route_to_diagnosis']
        if row.event is ReportingEvent.CANCER_DIAGNOSIS and isinstance(row.metadata, DiagnosisMetadata):
            return pandas.Series(['Incidence', str(row.metadata.stage), str(row.metadata.route)], index=col_names)
        elif row.event is ReportingEvent.CANCER_RECURRENCE and isinstance(row.metadata, RecurrenceMetadata):
            return pandas.Series(['Recurrence', str(row.metadata.stage_at_diagnosis), ''], index=col_names)
        elif row.event is ReportingEvent.DEATH and isinstance(row.metadata, DeathMetadata):
            return pandas.Series(['Mortality', '', ''], index=col_names)
        else:
            raise RuntimeError()

    @classmethod
    def colorectal_cancers(cls, population_events: pandas.DataFrame) -> pandas.DataFrame:
        """Return a data frame of the colorectal cancers detected

        The returned data frame will have the following columns:

        * cancer_outcome: 'Incidence', 'Recurrence', or 'Mortality'
        * stage_at_diagnosis: 'I', 'II', 'III', or 'IV'
        * route_to_diagnosis: 'Surveillance', 'Symptomatic'
        * n
        """
        eligible_events = (
            population_events['metadata'].apply(lambda md: isinstance(md, DiagnosisMetadata) and md.site is CancerSite.COLORECTUM)
            |
            population_events['metadata'].apply(lambda md: isinstance(md, RecurrenceMetadata) and md.site is CancerSite.COLORECTUM)
            |
            population_events['metadata'].apply(lambda md: isinstance(md, DeathMetadata) and md.cause is CauseOfDeath.COLORECTAL_CANCER)
        )
        events_transformed = population_events[eligible_events].apply(CancerOutcomes._cancer_count_row_transform, axis=1)
        
        return events_transformed.groupby(level=['params_uuid', 'pathway_uuid']).value_counts()

    @classmethod
    def endometrial_cancers(cls, population_events: pandas.DataFrame) -> pandas.DataFrame:
        """Return a data frame of the endometrial cancers detected

        The returned data frame will have the following columns:

        * cancer_outcome: 'Incidence', 'Recurrence', or 'Mortality'
        * stage_at_diagnosis: 'I', 'II', 'III', or 'IV'
        * route_to_diagnosis: 'Surveillance', 'Symptomatic'
        * n
        """
        eligible_events = (
            population_events['metadata'].apply(lambda md: isinstance(md, DiagnosisMetadata) and md.site is CancerSite.ENDOMETRIUM)
            |
            population_events['metadata'].apply(lambda md: isinstance(md, RecurrenceMetadata) and md.site is CancerSite.ENDOMETRIUM)
            |
            population_events['metadata'].apply(lambda md: isinstance(md, DeathMetadata) and md.cause is CauseOfDeath.ENDOMETRIAL_CANCER)
        )
        events_transformed = population_events[eligible_events].apply(CancerOutcomes._cancer_count_row_transform, axis=1)
        
        return events_transformed.groupby(level=['params_uuid', 'pathway_uuid']).value_counts()

    @classmethod
    def ovarian_cancers(cls, population_events: pandas.DataFrame) -> pandas.DataFrame:
        """Return a data frame of the ovarian cancers detected

        The returned data frame will have the following columns:

        * cancer_outcome: 'Incidence', 'Recurrence', or 'Mortality'
        * stage_at_diagnosis: 'I', 'II', 'III', or 'IV'
        * route_to_diagnosis: 'Surveillance', 'Symptomatic'
        * n
        """
        eligible_events = (
            population_events['metadata'].apply(lambda md: isinstance(md, DiagnosisMetadata) and md.site is CancerSite.OVARIES)
            |
            population_events['metadata'].apply(lambda md: isinstance(md, RecurrenceMetadata) and md.site is CancerSite.OVARIES)
            |
            population_events['metadata'].apply(lambda md: isinstance(md, DeathMetadata) and md.cause is CauseOfDeath.OVARIAN_CANCER)
        )
        events_transformed = population_events[eligible_events].apply(CancerOutcomes._cancer_count_row_transform, axis=1)
        
        return events_transformed.groupby(level=['params_uuid', 'pathway_uuid']).value_counts()
    
    @classmethod
    def cancer_free_survival(
        cls,
        events: pandas.DataFrame
    ) -> pandas.DataFrame:
        """Return a data frame from which cancer free survival can be estimated
        
        The returned data frame will have the following columns:

        * genotype:   the genotype of the individual
        * sex:        the sex of the individual
        * age_enter:  the age at which the individual enters the model
        * age_event:  the age at which the individual is diagnosed with cancer or becomes censored
        * event:      1 if age_event corresponds to a cancer diagnosis, 0 otherwise
        * cancer:     the cancer which the individual is affected by (if applicable)
        * stage:      the stage of cancer (if applicable)
        * route:      the route to diagnosis (if applicable)

        The possible reasons for censoring are:

        * Death
        * Reaching the time horizon

        Therefore the individual is considered at risk of cancer even if they
        undergo some risk-reducing surgery.

        Given the results of this function, a suitable analysis in Stata might
        look like:

        .. code-block:: stata

            stset age_event, origin(0) enter(age_enter) failure(event == 1)
            sts graph, survival by(sex genotype)

        The argument `events` is expected to have a
        ``pandas.MultiIndex`` of the form ([..., ] individual_uuid, record_id)

        The returned data frame will have an Index of 'individual_uuid' if the
        ``pandas.MultiIndex`` for `events` only has levels 'individual_uuid'
        and 'record_id', or it will have a `MultiIndex` with additional levels
        if additional levels are present in `events`
        """
        res: List[pandas.DataFrame] = list()

        for individual_uuid, individual_events in events.groupby(level='individual_uuid'):
            # For each individual we will process their events and construct a
            # single row DataFrame, which will then be concatenated with the
            # rows from the other individuals

            genotype = None
            sex = None
            age_enter = None
            age_event = None
            event = None
            cancer = None
            stage = None
            route = None

            # Step 1 - Find individual metadata in the EnterModel event
            ev, md = individual_events['event'].iloc[0], individual_events['metadata'].iloc[0]
            if ev is ReportingEvent.ENTER_MODEL and isinstance(md, EnterModelMetadata):
                genotype = md.genotype
                sex = md.sex
                age_enter = md.age
            else:
                raise ValueError("the event trace for each individual is expected to start with a ReportingEvent.ENTER_MODEL")
            
            # Step 2 - Find any cancer diagnoses
            idx_candx = None
            for i in range(1, len(individual_events)):
                ev, md = individual_events['event'].iloc[i], individual_events['metadata'].iloc[i]
                if ev is ReportingEvent.CANCER_DIAGNOSIS and isinstance(md, DiagnosisMetadata):
                    idx_candx = i
                    age_event = age_enter + individual_events['time'].iloc[i]
                    event = 1
                    cancer = md.site
                    stage = md.stage
                    route = md.route
                    break
            
            # Step 3 - If no cancer diagnoses, find when they die or reach the
            #          time horizon
            if idx_candx is None:
                for i in range(1, len(individual_events)):
                    ev, md = individual_events['event'].iloc[i], individual_events['metadata'].iloc[i]
                    if ev in [ReportingEvent.DEATH, ReportingEvent.REACH_TIME_HORIZON]:
                        event = 0
                        age_event = age_enter + individual_events['time'].iloc[i]
                        break
            
            # Step 4 - Construct the row and return
            res.append(pandas.DataFrame(
                {
                    'genotype': str(genotype),
                    'sex': str(sex),
                    'age_enter': age_enter,
                    'age_event': age_event,
                    'event': event,
                    'cancer': str(cancer),
                    'stage': str(stage),
                    'route': str(route)
                },
                index=individual_events.iloc[0:1].index.droplevel('record_id')
            ))

        if len(res) > 0:
            return pandas.concat(res, axis=0)
        else:
            return pandas.DataFrame(columns=['genotype', 'sex', 'age_enter', 'age_event', 'event', 'cancer', 'stage', 'route'])
    
    @classmethod
    def colorectal_cancer_survival(
        cls,
        events: pandas.DataFrame,
        survival_type: Optional[SurvivalType] = SurvivalType.ALL_CAUSE
    ) -> Optional[pandas.DataFrame]:
        """Return a data frame from which colorectal cancer survival can be calculated

        The returned data frame will have the following columns:

        * genotype:      the genotype of the individual
        * sex:           the sex of the individual
        * age_diagnosis: the age at which the individual is diagnosed with cancer
        * age_event:     the age at which the individual dies or becomes censored
        * event:         1 if age_event corresponds to an eligible death, 0 otherwise
        * stage:         the stage of the cancer
        * route:         the route to diagnosis of the cancer

        The returned data frame will only have entries for individuals who are
        diagnosed with colorectal cancer.

        The possible reasons for censoring are:

        * Death from a cause other than colorectal cancer (only if
          `survival_type` is ``SurvivalType.CAUSE_SPECIFIC``)
        * Reaching the time horizon

        Given the results of this function, a suitable analysis in Stata might
        look like:

        .. code-block:: stata

            stset age_event, origin(age_diagnosis) failure(event == 1)
            sts graph, survival
        
        The argument `events` is expected to have a
        ``pandas.MultiIndex`` of the form ([..., ] individual_uuid, record_id)
        """
        res: List[pandas.DataFrame] = list()

        for individual_uuid, individual_events in events.groupby(level='individual_uuid'):
            # For each individual we will process their events and construct a
            # single row DataFrame, which will then be concatenated with the
            # rows from the other individuals

            genotype = None
            sex = None
            age_enter = None
            age_diagnosis = None
            age_event = None
            event = None
            stage = None
            route = None

            # Step 1 - Find individual metadata in the EnterModel event
            ev, md = individual_events['event'].iloc[0], individual_events['metadata'].iloc[0]
            if ev is ReportingEvent.ENTER_MODEL and isinstance(md, EnterModelMetadata):
                genotype = md.genotype
                sex = md.sex
                age_enter = md.age
            else:
                raise ValueError("the event trace for each individual is expected to start with a ReportingEvent.ENTER_MODEL")
            
            # Step 2 - Find any colorectal cancer diagnosis
            idx_candx = None
            for i in range(1, len(individual_events)):
                ev, md = individual_events['event'].iloc[i], individual_events['metadata'].iloc[i]
                if ev is ReportingEvent.CANCER_DIAGNOSIS and isinstance(md, DiagnosisMetadata) and md.site is CancerSite.COLORECTUM:
                    idx_candx = i
                    age_diagnosis = age_enter + individual_events['time'].iloc[i]
                    event = 1
                    stage = md.stage
                    route = md.route
                    break
            
            # Step 3 - If no colorectal cancer diagnosis, skip this individual
            if idx_candx is None:
                continue

            # Step 4 - If there was a colorectal cancer diagnosis, find the
            #   record referring to failure or censoring
            for i in range(idx_candx + 1, len(individual_events)):
                ev, md = individual_events['event'].iloc[i], individual_events['metadata'].iloc[i]
                if ev in [ReportingEvent.DEATH, ReportingEvent.REACH_TIME_HORIZON]:
                    age_event = age_enter + individual_events['time'].iloc[i]
                    if (
                        ev is ReportingEvent.DEATH
                        and isinstance(md, DeathMetadata)
                        and (md.cause is CauseOfDeath.COLORECTAL_CANCER or survival_type is SurvivalType.ALL_CAUSE)
                    ):
                        event = 1
                    else:
                        event = 0
            
            # Step 4 - Construct the row and return
            res.append(pandas.DataFrame(
                {
                    'genotype': str(genotype),
                    'sex': str(sex),
                    'age_diagnosis': age_diagnosis,
                    'age_event': age_event,
                    'event': event,
                    'stage': str(stage),
                    'route': str(route)
                },
                index=individual_events.iloc[0:1].index.droplevel('record_id')
            ))

        if len(res) > 0:
            return pandas.concat(res, axis=0)
        else:
            return None

    @classmethod
    def endometrial_cancer_survival(
        cls,
        events: pandas.DataFrame,
        survival_type: Optional[SurvivalType] = SurvivalType.ALL_CAUSE
    ) -> Optional[pandas.DataFrame]:
        """Return a data frame from which endometrial cancer survival can be calculated

        The returned data frame will have the following columns:

        * genotype:      the genotype of the individual
        * sex:           the sex of the individual
        * age_diagnosis: the age at which the individual is diagnosed with cancer
        * age_event:     the age at which the individual dies or becomes censored
        * event:         1 if age_event corresponds to an eligible death, 0 otherwise
        * stage:         the stage of the cancer
        * route:         the route to diagnosis of the cancer

        The returned data frame will only have entries for individuals who are
        diagnosed with endometrial cancer.

        The possible reasons for censoring are:

        * Death from a cause other than endometrial cancer (only if
          `survival_type` is ``SurvivalType.CAUSE_SPECIFIC``)
        * Reaching the time horizon

        Given the results of this function, a suitable analysis in Stata might
        look like:

        .. code-block:: stata

            stset age_event, origin(age_diagnosis) failure(event == 1)
            sts graph, survival
        
        The argument `events` is expected to have a
        ``pandas.MultiIndex`` of the form ([..., ] individual_uuid, record_id)
        """
        res: List[pandas.DataFrame] = list()

        for individual_uuid, individual_events in events.groupby(level='individual_uuid'):
            # For each individual we will process their events and construct a
            # single row DataFrame, which will then be concatenated with the
            # rows from the other individuals

            genotype = None
            sex = None
            age_enter = None
            age_diagnosis = None
            age_event = None
            event = None
            stage = None
            route = None

            # Step 1 - Find individual metadata in the EnterModel event
            ev, md = individual_events['event'].iloc[0], individual_events['metadata'].iloc[0]
            if ev is ReportingEvent.ENTER_MODEL and isinstance(md, EnterModelMetadata):
                genotype = md.genotype
                sex = md.sex
                age_enter = md.age
            else:
                raise ValueError("the event trace for each individual is expected to start with a ReportingEvent.ENTER_MODEL")
            
            # Step 2 - Find any endometrial cancer diagnosis
            idx_candx = None
            for i in range(1, len(individual_events)):
                ev, md = individual_events['event'].iloc[i], individual_events['metadata'].iloc[i]
                if ev is ReportingEvent.CANCER_DIAGNOSIS and isinstance(md, DiagnosisMetadata) and md.site is CancerSite.ENDOMETRIUM:
                    idx_candx = i
                    age_diagnosis = age_enter + individual_events['time'].iloc[i]
                    event = 1
                    stage = md.stage
                    route = md.route
                    break
            
            # Step 3 - If no endometrial cancer diagnosis, skip this individual
            if idx_candx is None:
                continue

            # Step 4 - If there was a endometrial cancer diagnosis, find the
            #   record referring to failure or censoring
            for i in range(idx_candx + 1, len(individual_events)):
                ev, md = individual_events['event'].iloc[i], individual_events['metadata'].iloc[i]
                if ev in [ReportingEvent.DEATH, ReportingEvent.REACH_TIME_HORIZON]:
                    age_event = age_enter + individual_events['time'].iloc[i]
                    if (
                        ev is ReportingEvent.DEATH
                        and isinstance(md, DeathMetadata)
                        and (md.cause is CauseOfDeath.ENDOMETRIAL_CANCER or survival_type is SurvivalType.ALL_CAUSE)
                    ):
                        event = 1
                    else:
                        event = 0
            
            # Step 4 - Construct the row and return
            res.append(pandas.DataFrame(
                {
                    'genotype': str(genotype),
                    'sex': str(sex),
                    'age_diagnosis': age_diagnosis,
                    'age_event': age_event,
                    'event': event,
                    'stage': str(stage),
                    'route': str(route)
                },
                index=individual_events.iloc[0:1].index.droplevel('record_id')
            ))

        if len(res) > 0:
            return pandas.concat(res, axis=0)
        else:
            return None

    @classmethod
    def ovarian_cancer_survival(
        cls,
        events: pandas.DataFrame,
        survival_type: Optional[SurvivalType] = SurvivalType.ALL_CAUSE
    ) -> Optional[pandas.DataFrame]:
        """Return a data frame from which ovarian cancer survival can be calculated

        The returned data frame will have the following columns:

        * genotype:      the genotype of the individual
        * sex:           the sex of the individual
        * age_diagnosis: the age at which the individual is diagnosed with cancer
        * age_event:     the age at which the individual dies or becomes censored
        * event:         1 if age_event corresponds to an eligible death, 0 otherwise
        * stage:         the stage of the cancer
        * route:         the route to diagnosis of the cancer

        The returned data frame will only have entries for individuals who are
        diagnosed with ovarian cancer.

        The possible reasons for censoring are:

        * Death from a cause other than ovarian cancer (only if
          `survival_type` is ``SurvivalType.CAUSE_SPECIFIC``)
        * Reaching the time horizon

        Given the results of this function, a suitable analysis in Stata might
        look like:

        .. code-block:: stata

            stset age_event, origin(age_diagnosis) failure(event == 1)
            sts graph, survival
        
        The argument `events` is expected to have a
        ``pandas.MultiIndex`` of the form ([..., ] individual_uuid, record_id)
        """
        res: List[pandas.DataFrame] = list()

        for individual_uuid, individual_events in events.groupby(level='individual_uuid'):
            # For each individual we will process their events and construct a
            # single row DataFrame, which will then be concatenated with the
            # rows from the other individuals

            genotype = None
            sex = None
            age_enter = None
            age_diagnosis = None
            age_event = None
            event = None
            stage = None
            route = None

            # Step 1 - Find individual metadata in the EnterModel event
            ev, md = individual_events['event'].iloc[0], individual_events['metadata'].iloc[0]
            if ev is ReportingEvent.ENTER_MODEL and isinstance(md, EnterModelMetadata):
                genotype = md.genotype
                sex = md.sex
                age_enter = md.age
            else:
                raise ValueError("the event trace for each individual is expected to start with a ReportingEvent.ENTER_MODEL")
            
            # Step 2 - Find any ovarian cancer diagnosis
            idx_candx = None
            for i in range(1, len(individual_events)):
                ev, md = individual_events['event'].iloc[i], individual_events['metadata'].iloc[i]
                if ev is ReportingEvent.CANCER_DIAGNOSIS and isinstance(md, DiagnosisMetadata) and md.site is CancerSite.OVARIES:
                    idx_candx = i
                    age_diagnosis = age_enter + individual_events['time'].iloc[i]
                    event = 1
                    stage = md.stage
                    route = md.route
                    break
            
            # Step 3 - If no ovarian cancer diagnosis, skip this individual
            if idx_candx is None:
                continue

            # Step 4 - If there was a ovarian cancer diagnosis, find the
            #   record referring to failure or censoring
            for i in range(idx_candx + 1, len(individual_events)):
                ev, md = individual_events['event'].iloc[i], individual_events['metadata'].iloc[i]
                if ev in [ReportingEvent.DEATH, ReportingEvent.REACH_TIME_HORIZON]:
                    age_event = age_enter + individual_events['time'].iloc[i]
                    if (
                        ev is ReportingEvent.DEATH
                        and isinstance(md, DeathMetadata)
                        and (md.cause is CauseOfDeath.OVARIAN_CANCER or survival_type is SurvivalType.ALL_CAUSE)
                    ):
                        event = 1
                    else:
                        event = 0
            
            # Step 4 - Construct the row and return
            res.append(pandas.DataFrame(
                {
                    'genotype': str(genotype),
                    'sex': str(sex),
                    'age_diagnosis': age_diagnosis,
                    'age_event': age_event,
                    'event': event,
                    'stage': str(stage),
                    'route': str(route)
                },
                index=individual_events.iloc[0:1].index.droplevel('record_id')
            ))

        if len(res) > 0:
            return pandas.concat(res, axis=0)
        else:
            return None


class ResourceUseReport:
    
    @classmethod
    def gynae_risk_reduction(cls, events: pandas.DataFrame) -> pandas.DataFrame:
        """Count the number of gynaecological risk reduction procedures
        
        Will produce a :python:class:`pandas.DataFrame` with the following
        columns:

        * n_RRGS
        * n_gynae_surveillance

        These will be counted on a per-individual basis. Note that if
        cancer is found upon risk-reducing surgery it will not be counted in
        n_RRGS, it will instead be counted as a cancer diagnosis.
        """
        def process_events(individual_events: pandas.DataFrame) -> pandas.DataFrame:
            n_RRGS = 0
            n_gynae_surveillance = 0

            for i in range(len(individual_events)):
                ev, md = individual_events['event'].iloc[i], individual_events['metadata'].iloc[i]
                if ev is ReportingEvent.CANCER_SURVEILLANCE and isinstance(md, SurveillanceMetadata):
                    if md.site in [SurveillanceSite.ENDOMETRIUM, SurveillanceSite.OVARIES, SurveillanceSite.FEMALE_REPRODUCTIVE]:
                        n_gynae_surveillance += 1
                elif ev is ReportingEvent.RISK_REDUCING_SURGERY and isinstance(md, RiskReducingSurgeryMetadata):
                    n_RRGS += 1

            return pandas.DataFrame(
                { 'n_RRGS': n_RRGS, 'n_gynae_surveillance': n_gynae_surveillance },
                index=individual_events.iloc[0:1].index.droplevel(['individual_uuid', 'record_id'])
            )

        return events.groupby(level='individual_uuid').apply(process_events)
