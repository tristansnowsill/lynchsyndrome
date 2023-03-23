import numpy.random

from .sources import ConstantParameterSource, ParameterSource

def params_general_mortality(rng: numpy.random.Generator) -> ParameterSource:
    return ConstantParameterSource({
        'general_mortality.age.male'   : .0902695,
        'general_mortality.y0.male'    : -.0107368,
        'general_mortality.y1.male'    : -.0040982,
        'general_mortality.cons.male'  : -10.39574,
        'general_mortality.age.female' : .0972986,
        'general_mortality.y0.female'  : -.0098206,
        'general_mortality.y1.female'  : -.0039263,
        'general_mortality.cons.female': -11.23642
    })
