from typing import Callable, Iterable, Literal, Optional, Tuple

import numpy

class InhomogeneousPoissonProcess:

    """Create a (1-dimensional) inhomogeneous poisson process
    
    :param intensity_fn:    The intensity function for the process
    :param range:           The range of possible values
    :param pwc_envelope:    A piece-wise constant function which is
                                greater than or equal to intensity_fn
                                at all points inside each subrange,
                                and where the subranges form a complete
                                partition of ``range``
    """
    def __init__(
        self,
        intensity_fn: Callable[[float], float],
        range: Tuple[float, float],
        pwc_envelope: Iterable[Tuple[float, float, float]],
        rng: Optional[numpy.random.Generator] = numpy.random.default_rng()
    ):
        self.intensity_fn = intensity_fn
        self.range = range
        self.pwc_envelope = pwc_envelope
        self.rng = rng
    
    def sample(self, output_format: Literal['list', 'numpy'] = 'list'):
        res = []
        for lower, upper, intensity in self.pwc_envelope:
            # Number of potential points governed by poisson distribution
            n_potential = self.rng.poisson(intensity * (upper - lower))

            # Positions sampled uniformly
            x_potential = self.rng.uniform(lower, upper, size=(n_potential,))

            # Rejection sampler
            for x in x_potential:
                actual_intensity = self.intensity_fn(x)
                u = self.rng.random()
                if u <= actual_intensity / intensity:
                    res.append(x)
        
        if output_format == 'list':
            return numpy.sort(res).tolist()
        elif output_format == 'numpy':
            return numpy.sort(res)


