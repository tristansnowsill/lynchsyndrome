# -*- coding: utf-8 -*-
"""Tests for `lynchsyndrome` package class `IndividualBuilder`."""

import random

import numpy.random
import pytest
import simpy

from lynchsyndrome.bowel import BowelState
from lynchsyndrome.experiments.common.providers.parameters import ParameterProvider
from lynchsyndrome.genetics import ConstitutionalMMR
from lynchsyndrome.individual import IndividualBuilder
from lynchsyndrome.ovaries import healthy_premenopausal
from lynchsyndrome.sex import Sex


@pytest.fixture
def simpy_env():
    """Creates a SimPy environment."""
    return simpy.Environment()

@pytest.fixture(scope="session")
def params():
    return next(ParameterProvider().provide_parameters(numpy.random.default_rng()))


def test_it_creates_individual_with_specified_sex(simpy_env, params):
    """Test that the IndividualBuilder class creates Individual with specified sex"""
    builder = IndividualBuilder(simpy_env, numpy.random.default_rng(), numpy.random.default_rng(), params)
    builder.set_sex(Sex.FEMALE)
    builder.set_genotype(ConstitutionalMMR.PATH_MLH1)
    builder.set_bowel_state(BowelState.NORMAL)
    builder.set_ovarian_state(healthy_premenopausal)
    individual = builder.create()
    assert individual.sex == Sex.FEMALE

def test_it_creates_individual_with_callable_sex(simpy_env, params):
    """Test that the IndividualBuilder class creates Individual with specified sex as callable"""
    builder = IndividualBuilder(simpy_env, numpy.random.default_rng(), numpy.random.default_rng(), params)

    def get_next_sex():
        return Sex.MALE

    builder.set_sex(get_next_sex)
    builder.set_bowel_state(BowelState.NORMAL)
    individual = builder.create()
    assert individual.sex == Sex.MALE

def test_it_creates_individual_with_specified_age(simpy_env, params):
    """Test that the IndividualBuilder class creates Individual with specified age"""
    builder = IndividualBuilder(simpy_env, numpy.random.default_rng(), numpy.random.default_rng(), params)
    builder.set_age(50)
    builder.set_genotype(ConstitutionalMMR.PATH_MLH1)
    builder.set_bowel_state(BowelState.NORMAL)
    builder.set_ovarian_state(healthy_premenopausal)
    individual = builder.create()
    assert individual.age == 50.0

def test_it_creates_individual_with_callable_age(simpy_env, params):
    """Test that the IndividualBuilder class creates Individual with specified age as callable"""
    builder = IndividualBuilder(simpy_env, numpy.random.default_rng(), numpy.random.default_rng(), params)

    def get_next_age():
        return random.uniform(30, 50)

    builder.set_age(get_next_age)
    builder.set_genotype(ConstitutionalMMR.PATH_MLH1)
    builder.set_bowel_state(BowelState.NORMAL)
    builder.set_ovarian_state(healthy_premenopausal)
    individual = builder.create()
    assert individual.age >= 30 and individual.age <= 50
