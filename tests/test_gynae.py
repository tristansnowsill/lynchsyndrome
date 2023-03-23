# -*- coding: utf-8 -*-
"""Tests for various gynaecological modules."""

import uuid
from typing import Any, Mapping

import numpy
import pytest
import simpy
import simpy.util
from pytest_mock import MockerFixture
from lynchsyndrome.bowel import BowelState

import lynchsyndrome.endometrium
from lynchsyndrome.endometrium import Endometrium
from lynchsyndrome.genetics import ConstitutionalMMR
from lynchsyndrome.gynae_surveillance import SingleSurveillance
from lynchsyndrome.individual import Individual, IndividualBuilder
from lynchsyndrome.ovaries import MenopausalState, OvarianCancerObservedState, OvarianCancerTrueState, OvarianState, OvarianSurgicalState
from lynchsyndrome.sex import Sex
from lynchsyndrome.surveillance import SurveillanceSite, SurveillanceTechnology

@pytest.fixture
def params():
    return {
        'analysis.year'        : 2022,
        'analysis.time_horizon': 100.0,

        'endometrial.incidence.MLH1'  : [-10,-10,-10,-10,-10,-10],
        'endometrial.incidence.MSH2'  : [-10,-10,-10,-10,-10,-10],
        'endometrial.incidence.MSH6'  : [-10,-10,-10,-10,-10,-10],
        'endometrial.incidence.PMS2'  : [-10,-10,-10,-10,-10,-10],
        'endometrial.incidence.genpop': [-10,-10,-10,-10,-10,-10],
        'endometrial.progression'     : [numpy.log(2)/2, 1e-10, 1e-10, 1e-10],
        'endometrial.symptomatic'     : [1e-10, 1e-10, 1e-10, 1e-10, 1e-10],
        'endometrial.diagnosis'       : [1e-10, 1e-10, 1e-10, 1e-10, 1e-10],
        'endometrial.regression'      : 1e-10,

        'general_mortality.age.male'   : .0902695,
        'general_mortality.y0.male'    : -.0107368,
        'general_mortality.y1.male'    : -.0040982,
        'general_mortality.cons.male'  : -10.39574,
        'general_mortality.age.female' : .0972986,
        'general_mortality.y0.female'  : -.0098206,
        'general_mortality.y1.female'  : -.0039263,
        'general_mortality.cons.female': -11.23642,

        'gynae_surveillance.sensitivity.aeh.premenopausal' : 1.0,
        'gynae_surveillance.sensitivity.aeh.postmenopausal': 1.0,
        'gynae_surveillance.sensitivity.ec.premenopausal'  : 1.0,
        'gynae_surveillance.sensitivity.ec.postmenopausal' : 1.0,
        'gynae_surveillance.sensitivity.oc.premenopausal'  : 1.0,
        'gynae_surveillance.sensitivity.oc.postmenopausal' : 1.0,

        'ovarian.incidence.MLH1'    : [-10,-10,-10,-10,-10,-10],
        'ovarian.incidence.MSH2'    : [-10,-10,-10,-10,-10,-10],
        'ovarian.incidence.MSH6'    : [-10,-10,-10,-10,-10,-10],
        'ovarian.incidence.PMS2'    : [-10,-10,-10,-10,-10,-10],
        'ovarian.incidence.genpop'  : [-10,-10,-10,-10,-10,-10],
        'ovarian.progression.lynch' : [numpy.log(2)/2,1e-10,1e-10],
        'ovarian.progression.genpop': [1e-10,1e-10,1e-10],
        'ovarian.presentation'      : [1e-10,1e-10,1e-10,1e-10],
        'ovarian.presentation'      : [1e-10,1e-10,1e-10,1e-10],
    }

class MedianGenerator(numpy.random.Generator):
    def __init__(self):
        super().__init__(bit_generator=numpy.random.PCG64())

    def random(self):
        return 0.5

    def exponential(self, scale: float):
        return scale * numpy.log(2)

def test_synchronous_cancers_detected_in_surveillance(mocker: MockerFixture, params: Mapping[str, Any]) -> None:
    """Test that when there are synchronous ovarian and endometrial cancers
    everything proceeds as expected
    
    This sets up the following:

    .. list-table:: Events
        :widths: 50 50
        :header-rows: 1

        * - Time
          - Event
        * - 0.0
          - Enter model aged 40
        * - 9.0
          - Ovarian cancer incidence
        * - 10.0
          - AEH incidence
        * - 11.0
          - AEH progression to Stage I EC
        * - 12.0
          - Ovarian cancer progression to Stage II OC
        * - 15.0
          - Gynaecological surveillance

    """

    env = simpy.Environment()
    init_rng = numpy.random.default_rng()

    # We create a mock RNG where instead of returning a random variate it
    # always returns the median value
    sim_rng = MedianGenerator()

    # Mock out individual.sample_baseline_state to stop it doing anything
    mocker.patch.object(Individual, 'sample_baseline_state')

    # Mock out Endometrium._get_params_hash
    mocker.patch('lynchsyndrome.endometrium.Endometrium._get_params_hash')

    # Create an individual
    individual = (
        IndividualBuilder(env, init_rng, sim_rng, params)
            .set_age(40.0)
            .set_sex(Sex.FEMALE)
            .set_genotype(ConstitutionalMMR.PATH_MSH2)
            .set_bowel_state(BowelState.NORMAL)
            .set_ovarian_state(
                OvarianState(
                    OvarianCancerTrueState.NONE,
                    OvarianCancerObservedState.NONE,
                    MenopausalState.PREMENOPAUSAL,
                    OvarianSurgicalState.OVARIES_INTACT
                )
            )
        ).create()
    mocker.patch.object(individual.bowel, 'run')
    individual.endometrium.lesions.loc[:] = 0

    # Develop an occult endometrial cancer and ovarian cancer
    #   We already have control over most event times by using the
    #   MedianGenerator but we need to also control when lesions initially
    #   appear by mocking InhomogeneousPoissonProcess
    aeh_mock_ipp = mocker.patch('lynchsyndrome.endometrium.InhomogeneousPoissonProcess', autospec=True)
    aeh_mock_ipp_instance = aeh_mock_ipp.return_value
    aeh_mock_ipp_instance.sample.return_value = [45.0, 50.0]
    ec_mock_ipp = mocker.patch('lynchsyndrome.ovaries.InhomogeneousPoissonProcess', autospec=True)
    ec_mock_ipp_instance = ec_mock_ipp.return_value
    ec_mock_ipp_instance.sample.return_value = [49.0]

    # Create a surveillance visit
    surveillance_visit = SingleSurveillance(
        env,
        sim_rng,
        params,
        individual,
        SurveillanceSite.FEMALE_REPRODUCTIVE,
        {
            SurveillanceTechnology.CA125,
            SurveillanceTechnology.HYSTEROSCOPY,
            SurveillanceTechnology.TRANSVAGINAL_ULTRASOUND
        }
    )
    simpy.util.start_delayed(env, surveillance_visit.run(), delay=15.0)
    
    # Run the simulation
    env.process(individual.run())
    env.run(until=4.9)
    assert individual.endometrium.lesions.sum() == 0
    env.run(until=5.1)
    assert individual.endometrium.lesions.loc[('asymptomatic', 0)] == 1
    env.run(until=7.1)
    assert individual.endometrium.lesions.loc[('asymptomatic', 1)] == 1
    env.run(until=8.9)
    assert individual.ovaries.state.ovarian_cancer_true_state is OvarianCancerTrueState.NONE
    env.run(until=9.5)
    assert individual.ovaries.state.ovarian_cancer_true_state is OvarianCancerTrueState.STAGE_I
    env.run(until=14.5)
    assert individual.ovaries.state.ovarian_cancer_true_state is OvarianCancerTrueState.STAGE_II
    assert individual.endometrium.has_occult_ec
    env.run(until=15.1)
    assert individual.ever_diagnosed_endometrial_cancer
    assert individual.ever_diagnosed_ovarian_cancer
    env.run(until=params['analysis.time_horizon']+1e-3)

