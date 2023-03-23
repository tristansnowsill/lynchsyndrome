# -*- coding: utf-8 -*-

import injector
import numpy
import pytest

from lynchsyndrome.experiments.common.experiment import setup as setup_experiment
from lynchsyndrome.experiments.common.parameters.sources import (
    ConstantParameterSource, OverrideParameterSource, ParameterSource)
from lynchsyndrome.experiments.common.providers.parameters import ParameterProvider


class AdditionalParameterProvider(injector.Module):

    @injector.singleton
    @injector.provider
    def provide_additional_parameters(self, binder: injector.Binder, rng: numpy.random.Generator) -> ParameterSource:
        original = ParameterProvider().provide_parameters(rng)
        return OverrideParameterSource(
            original,
            ConstantParameterSource({
                'risk_reducing_surgery.uptake.immediately': 1.0
            })
        )

def test_can_override_parameters() -> None:
    vanilla_app = setup_experiment()
    vanilla_params_source = vanilla_app.get(ParameterSource)
    vanilla_params = next(vanilla_params_source)
    assert vanilla_params['risk_reducing_surgery.uptake.immediately'] < 1.0

    app = setup_experiment([AdditionalParameterProvider])
    params_source = app.get(ParameterSource)
    params = next(params_source)
    assert params['risk_reducing_surgery.uptake.immediately'] == 1.0
