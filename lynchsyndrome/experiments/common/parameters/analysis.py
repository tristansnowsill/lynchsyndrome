import numpy.random

from .sources import ConstantParameterSource, ParameterSource

def params_analysis(_: numpy.random.Generator) -> ParameterSource:
    return ConstantParameterSource({
        'analysis.discount_rate.cost': 0.035,
        'analysis.discount_rate.qaly': 0.035,
        'analysis.discount_rate.ly': 0.035,
        'analysis.time_horizon': 100.0,
        'analysis.year': 2022
    })
