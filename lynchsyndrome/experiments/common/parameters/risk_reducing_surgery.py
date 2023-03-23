
from functools import partial
import numpy

from .sources import CallableParameterSource, DependentParameterSource, ParameterSource

def params_risk_reducing_surgery(rng: numpy.random.Generator) -> ParameterSource:
    def sample(_rng: numpy.random.Generator, m, Sigma):
        x = _rng.multivariate_normal(m, Sigma)
        x[1] = numpy.exp(x[1])
        return x
    
    return (
        CallableParameterSource(
            rng,
            {
                # 36 of 474 women aged 30-39 had undergone RRGS in PLSD
                'risk_reducing_surgery.uptake.immediately': lambda _rng: _rng.beta(a=36, b=438),

                # Rough estimates based on PLSD (Seppala et al. 2021)
                'risk_reducing_surgery.uptake.never.MLH1' : lambda _rng: _rng.beta(a=10, b=15),
                'risk_reducing_surgery.uptake.never.MSH2' : lambda _rng: _rng.beta(a=7, b=18),
                'risk_reducing_surgery.uptake.never.MSH6' : lambda _rng: _rng.beta(a=12, b=13),
                'risk_reducing_surgery.uptake.never.PMS2' : lambda _rng: _rng.beta(a=15, b=10),

                # Pretty much speculation
                'risk_reducing_surgery.uptake.aft.no_surveillance': lambda _rng: _rng.gamma(shape=100, scale=0.008)
            }
        ) +
        # These are based on ABC (approximate Bayesian computation) fits to
        # the mean and standard deviation of age at (first) RRGS in PLSD
        # using the R library `abc`, vague priors (see below), 100,000
        # simulations, keeping 1,000, and using local linear regression to
        # adjust parameter values. Finally, cov.wt was used to estimate the
        # (weighted) centre and covariance of a multivariate normal
        # approximation to the posterior sample distribution.
        #
        # Priors:
        #   mu = rnorm(10000, mean=3, sd=2)
        #   ln_sigma = rnorm(10000, mean=-1, sd=1)
        #
        # Targets:
        #   Genotype   Mean  Standard deviation
        #   =========  ====  ==================
        #   path_MLH1  45.4  7.6
        #   path_MSH2  44.4  7.9
        #   path_MSH6  47.6  8.3
        #   path_PMS2  48.3  9.8
        DependentParameterSource(
            rng,
            ['risk_reducing_surgery.uptake.mu.MLH1', 'risk_reducing_surgery.uptake.sigma.MLH1'],
            partial(sample, m=numpy.array([3.799381,-1.799352]), Sigma=1e-4*numpy.array([[0.8528981,-0.4000783],[-0.4000783,17.17899039]]))
        ) +
        DependentParameterSource(
            rng,
            ['risk_reducing_surgery.uptake.mu.MSH2', 'risk_reducing_surgery.uptake.sigma.MSH2'],
            partial(sample, m=numpy.array([3.775470,-1.737908]), Sigma=1e-4*numpy.array([[1.1005779,-0.0841241],[-0.0841241,16.6887817]]))
        ) +
        DependentParameterSource(
            rng,
            ['risk_reducing_surgery.uptake.mu.MSH6', 'risk_reducing_surgery.uptake.sigma.MSH6'],
            partial(sample, m=numpy.array([3.774748,-1.723470]), Sigma=1e-4*numpy.array([[4.6085933,0.5730082],[0.5730082,77.8553582]]))
        ) +
        DependentParameterSource(
            rng,
            ['risk_reducing_surgery.uptake.mu.PMS2', 'risk_reducing_surgery.uptake.sigma.PMS2'],
            partial(sample, m=numpy.array([3.773635,-1.705027]), Sigma=1e-4*numpy.array([[11.505156,-1.433028],[-1.433028,185.298820]]))
        )
    )
