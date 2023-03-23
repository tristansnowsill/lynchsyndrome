# -*- coding: utf-8 -*-
"""Tests for risk-reducing gynaecological surgery."""

from typing import Any, Mapping

import numpy
import pandas
import pytest
import simpy
import simpy.util
from pytest_mock import MockerFixture
from scipy.stats import binom_test

from lynchsyndrome.bowel import BowelState
from lynchsyndrome.death import CauseOfDeath
from lynchsyndrome.diagnosis import CancerSite, DiagnosisMetadata
from lynchsyndrome.experiments.common.parameters.sources import ConstantParameterSource, OverrideParameterSource, ParameterSource
from lynchsyndrome.experiments.common.providers.parameters import ParameterProvider
from lynchsyndrome.genetics import ConstitutionalMMR
from lynchsyndrome.gynae_surveillance import AnnualGynaecologicalSurveillance, SingleSurveillance
from lynchsyndrome.individual import Individual, IndividualBuilder
from lynchsyndrome.ovaries import MenopausalState, OvarianCancerObservedState, OvarianCancerTrueState, OvarianState, OvarianSurgicalState
from lynchsyndrome.reporting import ReportingEvent
from lynchsyndrome.risk_reducing_surgery import OfferRiskReducingHBSO
from lynchsyndrome.sex import Sex
from lynchsyndrome.surveillance import SurveillanceSite, SurveillanceTechnology

@pytest.fixture(scope='session')
def base_parameter_source():
    return ParameterProvider().provide_parameters(numpy.random.default_rng())

def replace_run_general_mortality(self):
    e_gen_mort = self.env.timeout(70.0)
    yield e_gen_mort | self.died | self.reach_time_horizon
    if e_gen_mort.processed:
        yield self.env.process(self.record_death(CauseOfDeath.OTHER_CAUSES))


def test_rrgs_happens_at_age_35(mocker: MockerFixture, base_parameter_source: ParameterSource) -> None:
    pop_size = 400

    sim_rng = numpy.random.default_rng()
    seed_sequence = numpy.random.SeedSequence()
    params_source = OverrideParameterSource(
        base_parameter_source,
        ConstantParameterSource({
            'analysis.time_horizon': 6.5,
            'risk_reducing_surgery.uptake.immediately': 1.0
        })
    )
    params = next(params_source)

    offer_hbso = OfferRiskReducingHBSO(35, False)

    # Everyone dies exactly 70 years after model start
    mocker.patch('lynchsyndrome.individual.Individual.run_general_mortality', replace_run_general_mortality)

    # Mock out individual.sample_baseline_state to stop it doing anything
    # mocker.patch.object(Individual, 'sample_baseline_state')

    for pathway in ['RRGS', 'Nothing']:
        env = simpy.Environment()
        init_rng = numpy.random.default_rng(seed_sequence)

        population = (
            IndividualBuilder(env, init_rng, sim_rng, params)
                .set_age(30.0)
                .set_sex(Sex.FEMALE)
                .set_genotype(ConstitutionalMMR.PATH_MLH1)
        ).create_many(pop_size)

        for ind in population:
            # Don't allow colorectal cancer to interfere
            # mocker.patch.object(ind.bowel, 'run')
            # mocker.patch.object(ind.endometrium, 'run')
            # mocker.patch.object(ind.ovaries, 'run')

            # Add individual to the simulation
            env.process(ind.run())

            # Setup gynae surveillance if applicable
            if pathway == "RRGS":
                env.process(offer_hbso.run(env, sim_rng, params, ind))

        # Run the simulation
        env.run(until=params['analysis.time_horizon'] + 1e-3)

        # Check how many had RRGS
        N_RRGS = sum([
            any((ev is ReportingEvent.RISK_REDUCING_SURGERY for t, ev, _ in ind.event_trace))
            for ind in population
        ])
        if pathway == 'Nothing':
            assert N_RRGS <= 0.05 * pop_size
        else:
            p = binom_test(
                N_RRGS,
                pop_size,
                0.025 + 0.95 * params['risk_reducing_surgery.uptake.immediately']
            )
            assert p > 0.05


def test_rrgs_happens_at_some_point(mocker: MockerFixture, base_parameter_source: ParameterSource) -> None:
    pop_size = 500

    sim_rng = numpy.random.default_rng()
    seed_sequence = numpy.random.SeedSequence()
    params_source = OverrideParameterSource(
        base_parameter_source,
        ConstantParameterSource({
            'analysis.time_horizon': 100.0,
            'risk_reducing_surgery.uptake.immediately': 0.0,
            'risk_reducing_surgery.uptake.never.MLH1': 0.0
        })
    )
    params = next(params_source)

    offer_hbso = OfferRiskReducingHBSO(35, False)

    # Everyone dies exactly 70 years after model start
    mocker.patch('lynchsyndrome.individual.Individual.run_general_mortality', replace_run_general_mortality)

    for pathway in ['RRGS', 'Nothing']:
        env = simpy.Environment()
        init_rng = numpy.random.default_rng(seed_sequence)

        population = (
            IndividualBuilder(env, init_rng, sim_rng, params)
                .set_age(30)
                .set_sex(Sex.FEMALE)
                .set_genotype(ConstitutionalMMR.PATH_MLH1)
        ).create_many(pop_size)

        for ind in population:
            # Don't allow endometrial or ovarian cancer to interfere
            mocker.patch.object(ind.endometrium, 'run')
            mocker.patch.object(ind.ovaries, 'run')

            # Don't let the individual die from bowel cancer
            mocker.patch.object(ind.bowel, 'run_crc_survival')

            # Add individual to the simulation
            env.process(ind.run())

            # Setup gynae surveillance if applicable
            if pathway == "RRGS":
                env.process(offer_hbso.run(env, sim_rng, params, ind))

        # Run the simulation
        env.run(until=params['analysis.time_horizon'] + 1e-3)

        # Check how many had RRGS
        N_RRGS = sum([
            any((ev is ReportingEvent.RISK_REDUCING_SURGERY for t, ev, _ in ind.event_trace))
            for ind in population
        ])
        if pathway == 'Nothing':
            assert N_RRGS == 0
        else:
            assert N_RRGS >= 0.98 * pop_size
