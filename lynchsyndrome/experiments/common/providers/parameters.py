
import numpy.random
from injector import Module, provider, singleton

from ..parameters.analysis import params_analysis
from ..parameters.colorectal import params_colorectal
from ..parameters.costs import params_costs
from ..parameters.endometrial import params_endometrial
from ..parameters.general_mortality import params_general_mortality
from ..parameters.gynae_surveillance import params_gynae_surveillance
from ..parameters.ovarian import params_ovarian
from ..parameters.risk_reducing_surgery import params_risk_reducing_surgery
from ..parameters.sources import CachedParameterSource, CompositeParameterSource, ParameterSource
from ..parameters.utility import params_utilities

class ParameterProvider(Module):
    @singleton
    @provider
    def provide_parameters(self, rng: numpy.random.Generator) -> ParameterSource:
        all_parameters = [ 
            params_endometrial(rng),
            params_colorectal(rng),
            params_ovarian(rng),
            params_gynae_surveillance(rng),
            params_risk_reducing_surgery(rng),
            params_costs(rng),
            params_utilities(rng),
            params_analysis(rng),
            params_general_mortality(rng)
        ]
        return CachedParameterSource(CompositeParameterSource(all_parameters))