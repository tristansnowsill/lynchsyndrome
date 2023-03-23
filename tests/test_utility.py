# -*- coding: utf-8 -*-
"""Tests for `lynchsyndrome.utility` module."""

import numpy.random
import pytest
import simpy

from lynchsyndrome.individual import Individual, IndividualBuilder
from lynchsyndrome.sex import Sex
from lynchsyndrome.utility import (AdditiveOverallUtility, BaselineUtility,
                                   BasicSingleConditionUtilityEffect,
                                   HSEBaselineUtility, MinimumOverallUtility,
                                   MultiplicativeOverallUtility)


class SimpleBaselineUtility(BaselineUtility):
    @staticmethod
    def utility(sex: Sex, age: float) -> float:
        return 0.7 if sex is Sex.FEMALE else 0.75


@pytest.fixture
def crc_utility_effect():
    return BasicSingleConditionUtilityEffect(0.2, 0.7, "colorectal cancer")


@pytest.fixture
def ec_utility_effect():
    return BasicSingleConditionUtilityEffect(0.02, 0.97, "endometrial cancer")


@pytest.fixture
def oc_utility_effect():
    return BasicSingleConditionUtilityEffect(0.2, 0.8, "ovarian cancer")


def test_it_calculates_baseline_utility():
    simple_baseline = SimpleBaselineUtility()
    hse_baseline = HSEBaselineUtility()

    assert simple_baseline.utility(Sex.FEMALE, 65.0) == pytest.approx(0.7)
    assert simple_baseline.utility(Sex.MALE, 75.0) == pytest.approx(0.75)

    assert hse_baseline.utility(Sex.FEMALE, 65.0) == pytest.approx(0.793771)
    assert hse_baseline.utility(Sex.MALE, 75.0) == pytest.approx(0.765917)


def test_it_creates_consistent_utilities():
    u_affected = 0.7
    u_unaffected = 0.9
    cond = BasicSingleConditionUtilityEffect.create_from_affected_and_unaffected(u_affected, u_unaffected)
    assert cond.utility_affected() == pytest.approx(u_affected)
    assert cond.utility_unaffected() == pytest.approx(u_unaffected)


def test_it_calculates_additive_utilities(
    crc_utility_effect, ec_utility_effect, oc_utility_effect
):
    assert AdditiveOverallUtility().utility(
        Sex.FEMALE, 65.0,
        HSEBaselineUtility(), crc_utility_effect, ec_utility_effect, oc_utility_effect
    ) == pytest.approx(0.37377110)


def test_it_calculates_multiplicative_utilities(
    crc_utility_effect, ec_utility_effect, oc_utility_effect
):
    assert MultiplicativeOverallUtility().utility(Sex.FEMALE, 65, HSEBaselineUtility(), crc_utility_effect, ec_utility_effect, oc_utility_effect) == pytest.approx(0.43117646)


def test_it_calculates_minimum_utilities(
    crc_utility_effect, ec_utility_effect, oc_utility_effect
):
    overall_utility = MinimumOverallUtility()
    assert overall_utility.utility(Sex.FEMALE, 65, HSEBaselineUtility(), crc_utility_effect, ec_utility_effect, oc_utility_effect) == pytest.approx(0.5556398)
