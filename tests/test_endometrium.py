# -*- coding: utf-8 -*-
"""Tests for `lynchsyndrome.endometrium` module."""

from unittest.mock import Mock
import uuid
from collections import namedtuple
from typing import Any, Mapping

import numpy
import pytest
import simpy
from pytest import approx
from pytest_mock import MockerFixture

import lynchsyndrome.endometrium
from lynchsyndrome.diagnosis import CancerStage
from lynchsyndrome.endometrium import Endometrium
from lynchsyndrome.genetics import ConstitutionalMMR
from lynchsyndrome.sex import Sex

Individual = namedtuple('Individual', ['age', 'sex', 'genotype', 'uuid'])

@pytest.fixture
def params():
    return {
        'endometrial.incidence.MLH1': [-10,-10,-10,-10,-10,-10],
        'endometrial.incidence.MSH2': [-10,-10,-10,-10,-10,-10],
        'endometrial.incidence.MSH6': [-10,-10,-10,-10,-10,-10],
        'endometrial.incidence.PMS2': [-10,-10,-10,-10,-10,-10],
        'endometrial.incidence.genpop': [-10,-10,-10,-10,-10,-10],
        'endometrial.progression': [1e-10, 1e-10, 1e-10, 1e-10],
        'endometrial.symptomatic': [1e-10, 1e-10, 1e-10, 1e-10, 1e-10],
        'endometrial.diagnosis': [1e-10, 1e-10, 1e-10, 1e-10, 1e-10],
        'endometrial.regression': 1e-10
    }

def test_it_creates_lesions(mocker: MockerFixture, params: Mapping[str, Any]) -> None:
    env = simpy.Environment()
    individual = Individual(40.0, Sex.FEMALE, ConstitutionalMMR.PATH_MSH2, uuid.uuid4())
    
    # We need to mock the InhomogeneousPoissonProcess class so it returns
    # predictable results
    mock_ipp = mocker.patch('lynchsyndrome.endometrium.InhomogeneousPoissonProcess', autospec=True)
    mock_ipp_instance = mock_ipp.return_value
    mock_ipp_instance.sample.return_value = [50.0, 60.0]

    # Mock Endometrium._get_params_hash
    mocker.patch('lynchsyndrome.endometrium.Endometrium._get_params_hash')
    
    endometrium = Endometrium(env, numpy.random.default_rng(), params, individual)
    endometrium.lesions[:] = 0
    env.process(endometrium.run())
    env.run(until=10.01)
    assert mock_ipp_instance.sample.called
    assert sum(endometrium.lesions) == 1
    env.run(until=20.01)
    assert sum(endometrium.lesions) == 2

def test_properties(mocker: MockerFixture) -> None:
    env = simpy.Environment()
    individual = Individual(40.0, Sex.FEMALE, ConstitutionalMMR.PATH_MLH1, uuid.uuid4())
    m = mocker.patch('lynchsyndrome.endometrium.Endometrium._get_params_hash')
    m.return_value = 1
    endometrium = Endometrium(env, numpy.random.default_rng(), dict(), individual)
    endometrium.lesions[:] = 0
    assert not endometrium.has_occult_ec
    assert not endometrium.is_symptomatic
    endometrium.lesions[('symptomatic'), 1] = 1
    assert endometrium.has_occult_ec
    assert endometrium.is_symptomatic
    assert endometrium.occult_ec_stage == 1
    endometrium.lesions[:] = 0
    endometrium.lesions[('asymptomatic'), 1] = 1
    assert endometrium.has_occult_ec
    assert not endometrium.is_symptomatic
    assert endometrium.occult_ec_stage == 1
    endometrium.lesions[:] = 0
    endometrium.lesions[('asymptomatic'), 3] = 1
    assert endometrium.has_occult_ec
    assert not endometrium.is_symptomatic
    assert endometrium.occult_ec_stage == 3
    endometrium.lesions[:] = 0
    endometrium.lesions[('symptomatic'), 4] = 1
    assert endometrium.has_occult_ec
    assert endometrium.is_symptomatic
    assert endometrium.occult_ec_stage == 4
    

def test_lead_time(mocker: MockerFixture) -> None:
    params = {
        'endometrial.progression': [0.25, 1.60, 2.00, 2.50],
        'endometrial.symptomatic': [0.80, 1.40, 1.50, 1.60, 1.80],
        'endometrial.diagnosis'  : [2.0, 2.5, 2.5, 2.5, 3.0]
    }
    env = simpy.Environment()
    individual = Individual(40.0, Sex.FEMALE, ConstitutionalMMR.PATH_MSH2, uuid.uuid4())
    mocker.patch('lynchsyndrome.endometrium.Endometrium._get_params_hash')
    endometrium = Endometrium(env, numpy.random.default_rng(), params, individual)

    assert endometrium._lead_time(CancerStage.STAGE_I, False, True) == approx(0.686481115)
    assert endometrium._lead_time(CancerStage.STAGE_I, True, True)  == approx(0.302190235)
    assert endometrium._lead_time(CancerStage.STAGE_II, False, True) == approx(0.507936508)
    assert endometrium._lead_time(CancerStage.STAGE_II, True, True) == approx(0.222222222)
    assert endometrium._lead_time(CancerStage.STAGE_III, False, True) == approx(0.928997290)
    assert endometrium._lead_time(CancerStage.STAGE_III, True, True) == approx(0.366666667)
    assert endometrium._lead_time(CancerStage.STAGE_IV, False, True) == approx(0.888888889)
    assert endometrium._lead_time(CancerStage.STAGE_IV, True, True) == approx(0.333333333)

