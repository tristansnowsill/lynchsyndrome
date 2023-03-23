
import numpy

from .sources import CallableParameterSource, DependentParameterSource, ParameterSource

def params_gynae_surveillance(rng: numpy.random.Generator) -> ParameterSource:
    return _params_gynae_surveillance_sens(rng) + _params_gynae_surveillance_interval(rng)

def _params_gynae_surveillance_interval(rng: numpy.random.Generator) -> ParameterSource:
    return CallableParameterSource(rng, {
        'gynae_surveillance.interval.m': lambda _rng: _rng.normal(0.9, 0.1),
        'gynae_surveillance.interval.a': lambda _rng: _rng.gamma(shape=8, scale=0.5)
    })

def _params_gynae_surveillance_sens(rng: numpy.random.Generator) -> ParameterSource:
    def sample(_rng: numpy.random.Generator):
        mu = numpy.array([0.5, 1.5, 1.5, 2.0, 0.5, 0.5])
        chol = numpy.array([
            [1,0,0,0,0,0],
            [0.5,0.866025403784439,0,0,0,0],
            [0.5,0,0.866025403784439,0,0,0],
            [0.25,0.433012701892219,0.433012701892219,0.75,0,0],
            [0,0,0,0,1,0 ],
            [0,0,0,0,0.9,0.435889894354067]
        ])
        u = _rng.normal(size=6)
        x = mu + numpy.sqrt(0.5) * chol @ u
        return 1 / (1 + numpy.exp(-x))
    
    return DependentParameterSource(
        rng,
        [
            'gynae_surveillance.sensitivity.aeh.premenopausal',
            'gynae_surveillance.sensitivity.aeh.postmenopausal',
            'gynae_surveillance.sensitivity.ec.premenopausal',
            'gynae_surveillance.sensitivity.ec.postmenopausal',
            'gynae_surveillance.sensitivity.oc.premenopausal',
            'gynae_surveillance.sensitivity.oc.postmenopausal'
        ],
        sample
    )
