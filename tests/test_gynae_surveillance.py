# -*- coding: utf-8 -*-
"""Tests for gynaecological surveillance."""

from typing import Any, Mapping

import numpy
import pandas
import pytest
import simpy
import simpy.util
from pytest_mock import MockerFixture
# from scipy.stats import ks_2samp

from lynchsyndrome.bowel import BowelState
from lynchsyndrome.death import CauseOfDeath
from lynchsyndrome.diagnosis import CancerSite, DiagnosisMetadata
from lynchsyndrome.genetics import ConstitutionalMMR
from lynchsyndrome.gynae_surveillance import AnnualGynaecologicalSurveillance, SingleSurveillance
from lynchsyndrome.individual import Individual, IndividualBuilder
from lynchsyndrome.ovaries import MenopausalState, OvarianCancerObservedState, OvarianCancerTrueState, OvarianState, OvarianSurgicalState
from lynchsyndrome.reporting import ReportingEvent
from lynchsyndrome.sex import Sex
from lynchsyndrome.surveillance import SurveillanceSite, SurveillanceTechnology

@pytest.fixture
def params():
    """A parameter set which led to a significant difference in survival
    between the surveillance and do nothing arms"""
    return {
        'analysis.year'        : 2022,
        'analysis.time_horizon': 100.0,

        'endometrial.aeh_management'                     : [0.3471, 0.6429, 40.68, 2.039],
        'endometrial.aeh_management.conservative_success': 0.8640,
        'endometrial.early_ec.response_rate'             : 0.7971,

        'endometrial.incidence.MLH1'        : [-10.17,-10.97,-1.158,-4.197,-10.86,-7.630],
        'endometrial.progression'           : [0.1189,0.0515,1.016,0.4015],
        'endometrial.symptomatic'           : [0.0543,0.2078,0.5506,0.7653,5.083],
        'endometrial.diagnosis'             : [0.8693,0.5302,0.9661,1.331,3.748],
        'endometrial.regression'            : 0.1045,
        'endometrial.survival.lynch.late'   : 0.2964,
        'endometrial.recurrence.lynch.early': 0.0496,
        'endometrial.postrecurrence.lynch'  : 0.1011,

        'ovarian.incidence.MLH1'        : [-21.94,-18.88,-3.224,-5.368,-20.54,-20.11],
        'ovarian.progression.lynch'     : [0.2589,3.327,0.3262],
        'ovarian.presentation'          : [0.6207,1.059,2.431,2.532],
        'ovarian.survival.lynch.shape'  : 0.4488,
        'ovarian.survival.lynch.scale'  : 20.48,
        'ovarian.recurrence.lynch.shape': -0.1537,
        'ovarian.recurrence.lynch.rate' : 0.0537,

        'gynae_surveillance.interval.m'                    : 1.034,
        'gynae_surveillance.interval.a'                    : 8.867,
        'gynae_surveillance.sensitivity.aeh.premenopausal' : 0.7480,
        'gynae_surveillance.sensitivity.aeh.postmenopausal': 0.8895,
        'gynae_surveillance.sensitivity.ec.premenopausal'  : 0.8750,
        'gynae_surveillance.sensitivity.ec.postmenopausal' : 0.9233,
        'gynae_surveillance.sensitivity.oc.premenopausal'  : 0.8046,
        'gynae_surveillance.sensitivity.oc.postmenopausal' : 0.7330,

        # These parameters shouldn't matter...
        'endometrial.incidence.MSH2'       : [-10,-10,-10,-10,-10,-10],
        'endometrial.incidence.MSH6'       : [-10,-10,-10,-10,-10,-10],
        'endometrial.incidence.PMS2'       : [-10,-10,-10,-10,-10,-10],
        'endometrial.incidence.genpop'     : [-10,-10,-10,-10,-10,-10],
        'endometrial.recurrence.genpop'    : [0.018, 0.084, 0.307, 2.69],
        'endometrial.postrecurrence.genpop': 3.13,
        'endometrial.frailty.genpop'       : 1.5,
        'general_mortality.age.male'       : .0902695,
        'general_mortality.y0.male'        : -.0107368,
        'general_mortality.y1.male'        : -.0040982,
        'general_mortality.cons.male'      : -10.39574,
        'general_mortality.age.female'     : .0972986,
        'general_mortality.y0.female'      : -.0098206,
        'general_mortality.y1.female'      : -.0039263,
        'general_mortality.cons.female'    : -11.23642,
        'ovarian.incidence.MSH2'           : [-10,-10,-10,-10,-10,-10],
        'ovarian.incidence.MSH6'           : [-10,-10,-10,-10,-10,-10],
        'ovarian.incidence.PMS2'           : [-10,-10,-10,-10,-10,-10],
        'ovarian.incidence.genpop'         : [-10,-10,-10,-10,-10,-10],
        'ovarian.progression.genpop'       : [0.2,0.2,0.2],
    }

def replace_run_general_mortality(self):
    e_gen_mort = self.env.timeout(45.0)
    yield e_gen_mort | self.died | self.reach_time_horizon
    if e_gen_mort.processed:
        yield self.env.process(self.record_death(CauseOfDeath.OTHER_CAUSES))

def test_surveillance_not_harmful_via_endometrial_cancer(mocker: MockerFixture, params: Mapping[str, Any]) -> None:
    sim_rng = numpy.random.default_rng()
    seed_sequence = numpy.random.SeedSequence()

    annual_surveillance = AnnualGynaecologicalSurveillance(25, 75)

    pathway_ages_at_death = dict()

    # Everyone dies exactly 45 years after model start
    mocker.patch('lynchsyndrome.individual.Individual.run_general_mortality', replace_run_general_mortality)

    # Mock out individual.sample_baseline_state to stop it doing anything
    # mocker.patch.object(Individual, 'sample_baseline_state')

    for pathway in ['Nothing', 'Surveillance']:
        env = simpy.Environment()
        init_rng = numpy.random.default_rng(seed_sequence)

        population = (
            IndividualBuilder(env, init_rng, sim_rng, params)
                .set_age(45)
                .set_sex(Sex.FEMALE)
                .set_genotype(ConstitutionalMMR.PATH_MLH1)
                .set_bowel_state(BowelState.NORMAL)
                .set_ovarian_state(
                    OvarianState(
                        OvarianCancerTrueState.NONE,
                        OvarianCancerObservedState.NONE,
                        MenopausalState.PREMENOPAUSAL,
                        OvarianSurgicalState.OVARIES_INTACT
                    )
                )
        ).create_many(8000)

        for ind in population:
            # Don't allow colorectal or ovarian cancer to interfere
            mocker.patch.object(ind.bowel, 'run')
            mocker.patch.object(ind.ovaries, 'run')

            # Add individual to the simulation
            env.process(ind.run())

            # Setup gynae surveillance if applicable
            if pathway == "Surveillance":
                env.process(annual_surveillance.run(env, sim_rng, params, ind))

        # Run the simulation
        env.run(until=params['analysis.time_horizon'] + 1e-3)

        # Extract ages of deaths
        ages = list()
        for ind in population:
            ages.append(next( (ind.age + t for t, ev, _ in ind.event_trace if ev is ReportingEvent.DEATH) ))

        pathway_ages_at_death[pathway] = pandas.Series(
            ages,
            index=[str(ind.uuid) for ind in population],
            name='ages_at_death'
        )

    # comp = ks_2samp(pathway_ages_at_death['Nothing'], pathway_ages_at_death['Surveillance'])
    res = pandas.concat(
        pathway_ages_at_death,
        axis=0,
        names=('competing_option', 'individual_uuid')
    )
    res.reset_index().to_feather('test_surveillance_not_harmful_via_endometrial_cancer.feather')

def test_surveillance_not_harmful_via_ovarian_cancer(mocker: MockerFixture, params: Mapping[str, Any]) -> None:
    sim_rng = numpy.random.default_rng()
    seed_sequence = numpy.random.SeedSequence()

    annual_surveillance = AnnualGynaecologicalSurveillance(25, 75)

    pathway_ages_at_death = dict()

    # Everyone dies exactly 45 years after model start
    mocker.patch('lynchsyndrome.individual.Individual.run_general_mortality', replace_run_general_mortality)

    # Mock out individual.sample_baseline_state to stop it doing anything
    # mocker.patch.object(Individual, 'sample_baseline_state')

    for pathway in ['Nothing', 'Surveillance']:
        env = simpy.Environment()
        init_rng = numpy.random.default_rng(seed_sequence)

        population = (
            IndividualBuilder(env, init_rng, sim_rng, params)
                .set_age(45)
                .set_sex(Sex.FEMALE)
                .set_genotype(ConstitutionalMMR.PATH_MLH1)
                .set_bowel_state(BowelState.NORMAL)
        ).create_many(8000)

        for ind in population:
            # Don't allow colorectal or endometrial cancer to interfere
            ind.endometrium.lesions[:] = 0
            mocker.patch.object(ind.bowel, 'run')
            mocker.patch.object(ind.endometrium, 'run')

            # Add individual to the simulation
            env.process(ind.run())

            # Setup gynae surveillance if applicable
            if pathway == "Surveillance":
                env.process(annual_surveillance.run(env, sim_rng, params, ind))

        # Run the simulation
        env.run(until=params['analysis.time_horizon'] + 1e-3)

        # Extract ages of deaths
        ages = list()
        for ind in population:
            ages.append(next( (ind.age + t for t, ev, _ in ind.event_trace if ev is ReportingEvent.DEATH) ))

        pathway_ages_at_death[pathway] = pandas.Series(
            ages,
            index=[str(ind.uuid) for ind in population],
            name='ages_at_death'
        )

    res = pandas.concat(
        pathway_ages_at_death,
        axis=0,
        names=('competing_option', 'individual_uuid')
    )
    res.reset_index().to_feather('test_surveillance_not_harmful_via_ovarian_cancer.feather')

def test_surveillance_not_harmful_via_gynaecological_cancer(mocker: MockerFixture, params: Mapping[str, Any]) -> None:
    sim_rng = numpy.random.default_rng()
    seed_sequence = numpy.random.SeedSequence()

    annual_surveillance = AnnualGynaecologicalSurveillance(25, 75)

    pathway_ages_at_death = dict()

    # Everyone dies exactly 45 years after model start
    mocker.patch('lynchsyndrome.individual.Individual.run_general_mortality', replace_run_general_mortality)

    # Mock out individual.sample_baseline_state to stop it doing anything
    # mocker.patch.object(Individual, 'sample_baseline_state')

    for pathway in ['Nothing', 'Surveillance']:
        env = simpy.Environment()
        init_rng = numpy.random.default_rng(seed_sequence)

        population = (
            IndividualBuilder(env, init_rng, sim_rng, params)
                .set_age(45)
                .set_sex(Sex.FEMALE)
                .set_genotype(ConstitutionalMMR.PATH_MLH1)
                .set_bowel_state(BowelState.NORMAL)
        ).create_many(8000)

        for ind in population:
            # Don't allow colorectal cancer to interfere
            mocker.patch.object(ind.bowel, 'run')

            # Add individual to the simulation
            env.process(ind.run())

            # Setup gynae surveillance if applicable
            if pathway == "Surveillance":
                env.process(annual_surveillance.run(env, sim_rng, params, ind))

        # Run the simulation
        env.run(until=params['analysis.time_horizon'] + 1e-3)

        # Extract ages of deaths
        ages = list()
        for ind in population:
            ages.append(next( (ind.age + t for t, ev, _ in ind.event_trace if ev is ReportingEvent.DEATH) ))

        pathway_ages_at_death[pathway] = pandas.Series(
            ages,
            index=[str(ind.uuid) for ind in population],
            name='ages_at_death'
        )

    res = pandas.concat(
        pathway_ages_at_death,
        axis=0,
        names=('competing_option', 'individual_uuid')
    )
    res.reset_index().to_feather('test_surveillance_not_harmful_via_gynaecological_cancer.feather')

def test_surveillance_extends_survival_endometrial(mocker: MockerFixture, params: Mapping[str, Any]) -> None:
    """Create a population with undiagnosed endometrial cancer, subject one to
    a single surveillance visit, and leave the other alone"""
    sim_rng = numpy.random.default_rng()
    seed_sequence = numpy.random.SeedSequence()

    pathway_results = dict()

    # Everyone dies exactly 45 years after model start
    mocker.patch('lynchsyndrome.individual.Individual.run_general_mortality', replace_run_general_mortality)

    # Mock out individual.sample_baseline_state to stop it doing anything
    mocker.patch.object(Individual, 'sample_baseline_state')

    for pathway in ['Nothing', 'Surveillance']:
        for idx in range(10):
            init_rng = numpy.random.default_rng(seed_sequence)
            env = simpy.Environment()
            population = (
                IndividualBuilder(env, init_rng, sim_rng, params)
                    .set_age(45)
                    .set_sex(Sex.FEMALE)
                    .set_genotype(ConstitutionalMMR.PATH_MLH1)
                    .set_bowel_state(BowelState.NORMAL)
                    .set_ovarian_state(
                        OvarianState(
                            OvarianCancerTrueState.NONE,
                            OvarianCancerObservedState.NONE,
                            MenopausalState.PREMENOPAUSAL,
                            OvarianSurgicalState.OVARIES_INTACT
                        )
                    )
            ).create_many(200)

            for ind in population:
                surveillance_visit = SingleSurveillance(
                    env,
                    sim_rng,
                    params,
                    ind,
                    SurveillanceSite.ENDOMETRIUM,
                    { SurveillanceTechnology.HYSTEROSCOPY, SurveillanceTechnology.TRANSVAGINAL_ULTRASOUND }
                )

                # Don't allow colorectal or ovarian cancer to interfere
                mocker.patch.object(ind.bowel, 'run')
                mocker.patch.object(ind.ovaries, 'run')

                # Start population with specific endometrial lesion
                ind.endometrium.lesions[:] = 0
                ind.endometrium.lesions.iloc[idx] = 1

                # Add individual to the simulation
                env.process(ind.run())

                # Setup gynae surveillance if applicable
                if pathway == "Surveillance":
                    env.process(surveillance_visit.run())

            # Run the simulation
            env.run(until=params['analysis.time_horizon'] + 1e-3)

            # Extract ages of deaths
            age = list()
            stage_at_diagnosis = dict()
            for ind in population:
                for _, ev, md in ind.event_trace:
                    if (
                        ev is ReportingEvent.CANCER_DIAGNOSIS
                        and isinstance(md, DiagnosisMetadata)
                        and md.site is CancerSite.ENDOMETRIUM
                    ):
                        stage_at_diagnosis[str(ind.uuid)] = md.stage.value
                age.append(next( (ind.age + t for t, ev, _ in ind.event_trace if ev is ReportingEvent.DEATH) ))

            age_at_death = pandas.Series(
                age,
                index=[str(ind.uuid) for ind in population],
                name='age_at_death'
            )
            stage_at_diagnosis_series = pandas.Series(
                stage_at_diagnosis,
                name='stage_at_diagnosis'
            )
            pathway_results[(idx, pathway)] = pandas.concat(
                (age_at_death, stage_at_diagnosis_series),
                axis=1
            )

    res = pandas.concat(
        pathway_results,
        axis=0,
        names=('lesion_index', 'competing_option', 'individual_uuid')
    )
    res.reset_index().to_feather('test_surveillance_extends_endometrial_cancer_survival.feather')
