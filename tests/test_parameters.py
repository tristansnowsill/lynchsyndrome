# -*- coding: utf-8 -*-
"""Tests for `lynchsyndrome.parameters` module."""

import numpy as np
import pytest

from lynchsyndrome.experiments.common.parameters.sources import CallableParameterSource, ConstantParameterSource, ParameterSamplesSource

def test_it_stores_constant_values():
    params_source = ConstantParameterSource({
        'analysis.discount_rate.cost': 0.03,
        'analysis.discount_rate.qaly': 0.01
    })
    params = next(params_source)
    assert params['analysis.discount_rate.cost'] == 0.03
    params = next(params_source)
    assert params['analysis.discount_rate.qaly'] == 0.01

def test_it_works_with_callables():
    rng = np.random.default_rng()
    params_source = CallableParameterSource(
        rng,
        {
            'cost.colonoscopy': lambda _rng: _rng.gamma(shape=10, scale=50)
        }
    )
    params = next(params_source)
    c_colonoscopy = params['cost.colonoscopy']
    assert params['cost.colonoscopy'] == c_colonoscopy
    params = next(params_source)
    assert params['cost.colonoscopy'] != c_colonoscopy

def test_it_combines_parameter_sources():
    params_source_A = ConstantParameterSource({
        'analysis.discount_rate.cost': 0.03
    })
    params_source_B = ConstantParameterSource({
        'analysis.discount_rate.qaly': 0.01
    })
    params_source = params_source_A + params_source_B
    params = next(params_source)
    assert params['analysis.discount_rate.cost'] == 0.03
    assert params['analysis.discount_rate.qaly'] == 0.01

def test_it_uses_existing_samples():
    params_samples = {
        'cost.a': [500.0, 400.0, 600.0, 500.0],
        'cost.b': [20.0, 20.0, 10.0, 30.0]
    }
    params_source = ParameterSamplesSource(np.random.default_rng(), params_samples, 4)
    params = next(params_source)
    c_a = params['cost.a']
    c_b = params['cost.b']
    assert c_a in params_samples['cost.a']
    assert c_b in params_samples['cost.b']
    assert params['cost.a'] == c_a
