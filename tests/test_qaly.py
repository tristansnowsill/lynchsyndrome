# -*- coding: utf-8 -*-
"""Tests for `lynchsyndrome.qaly` module."""

from math import exp
import pytest

from lynchsyndrome.qaly import AdditiveQALY, ConstantUtilityQALY, LinearUtilityQALY, QALY, QuadraticUtilityQALY, UtilityFunctionQALY

def test_it_calculates_constant_utility_qalys():
    """Test that the ConstantUtilityQALY class works"""
    dr = 0.03
    t0 = 0.0
    t1 = 10.0
    u = 0.9
    qaly = ConstantUtilityQALY(t0, t1, u)
    assert qaly.undiscounted() == pytest.approx(9.0)
    assert qaly.discounted(dr) == pytest.approx(7.791773)

def test_it_calculates_linear_utility_qalys():
    """Test that the LinearUtilityQALY class works"""
    dr = 0.03
    t0 = 0.0
    t1 = 10.0
    u0 = 1.0
    u1 = 0.8
    qaly = LinearUtilityQALY(t0, t1, u0, u1)
    assert qaly.undiscounted() == pytest.approx(9.0)
    assert qaly.discounted(dr) == pytest.approx(7.834362)

def test_it_calculates_quadratic_utility_qalys():
    """Test that the QuadraticUtilityQALY class works"""
    dr = 0.03
    t0 = 0.0
    t1 = 20.0
    a0 = 0.97
    a1 = -1e-3
    a2 = -1e-5
    t_off = 40
    qaly = QuadraticUtilityQALY(t0, t1, a0, a1, a2, t_off)
    assert qaly.undiscounted() == pytest.approx(19.81333)
    assert qaly.discounted(dr) == pytest.approx(14.96447)

def test_it_calculates_utility_function_qalys():
    """Test that the UtilityFunctionQALY class works"""
    dr = 0.03
    t0 = 0.0
    t1 = 10.0
    u = lambda t: 0.6 + 0.4 / (1 + exp(t - 5.0))
    qaly = UtilityFunctionQALY(t0, t1, u)
    assert qaly.undiscounted() == pytest.approx(8.0)
    assert qaly.discounted(dr) == pytest.approx(7.037774)

def test_it_calculates_additive_qalys():
    """Test that the AdditiveQALY class works"""
    dr = 0.03
    t0 = 0.0
    t1 = 5.0
    t2 = 10.0
    u01 = 0.9
    u12 = 0.7
    qaly = ConstantUtilityQALY(t0, t1, u01) + ConstantUtilityQALY(t1, t2, u12)
    assert isinstance(qaly, AdditiveQALY)
    assert qaly.undiscounted() == pytest.approx(8.0)
    assert qaly.discounted(dr) == pytest.approx(6.989881)
