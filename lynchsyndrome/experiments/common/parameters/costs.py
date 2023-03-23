from typing import Callable
import numpy

from lynchsyndrome.experiments.common.parameters.sources import CallableParameterSource, CompositeParameterSource, DependentParameterSource, ParameterSource

def params_costs(rng: numpy.random.Generator) -> ParameterSource:
    return CompositeParameterSource([ 
        _params_colorectal_cancer_costs(rng),
        _params_endometrial_cancer_costs(rng),
        _params_ovarian_cancer_costs(rng),
        _params_aeh_medical_management_costs(rng),
        _params_gynae_surveillance_costs(rng),
        _params_colonoscopy_costs(rng),
        _params_risk_reducing_surgery_costs(rng),
        _params_aspirin_chemoprophylaxis_costs(rng)
    ])

def _params_colorectal_cancer_costs(rng: numpy.random.Generator) -> ParameterSource:
    
    def sample_and_transform(rng: numpy.random.Generator):
        year  = numpy.arange(1, 10)
        stage = numpy.array([1, 2])
        age   = numpy.array([1, 2])

        # See https://stackoverflow.com/a/49071149/526510
        cartesian_prod = numpy.transpose(
            numpy.meshgrid(age, stage, year, indexing='ij'),
            numpy.roll(numpy.arange(4), -1)
        ).reshape(-1, 3)

        sigma = numpy.zeros((36, 36))

        for i in range(36):
            age_i   = cartesian_prod[i,0]
            stage_i = cartesian_prod[i,1]
            year_i  = cartesian_prod[i,2]
            for j in range(36):
                age_j   = cartesian_prod[j,0]
                stage_j = cartesian_prod[j,1]
                year_j  = cartesian_prod[j,2]

                g_age   = 1 if   age_i == age_j   else 0.6
                g_stage = 1 if stage_i == stage_j else 0.6
                g_year  = 0.5 ** numpy.abs(year_j - year_i)

                sigma[i,j] = 0.04 * g_year * g_stage * g_age
        
        uncertainty_multipliers = rng.multivariate_normal(
            numpy.ones(36), sigma
        )

        base_costs = numpy.array([
            17722,4134,3426,2639,2371,1611,1676,1534,1318,
            22885,7468,5092,4151,2951,2463,2878,2197,1498,
            16592,3822,3116,2592,2631,2658,2416,2678,2236,
            18059,5662,4361,3405,3182,3019,1914,2499,1933
        ])

        costs_stacked = base_costs * uncertainty_multipliers

        return numpy.split(costs_stacked, 4)
    
    index = [
        'cost.colorectal_cancer.younger.early',
        'cost.colorectal_cancer.younger.late',
        'cost.colorectal_cancer.older.early',
        'cost.colorectal_cancer.older.late'
    ]

    return DependentParameterSource(rng, index, sample_and_transform)

def _params_endometrial_cancer_costs(rng: numpy.random.Generator) -> ParameterSource:
    return CallableParameterSource(rng, {
        'cost.endometrial_cancer.stage_I': lambda _rng: _rng.gamma(shape=608.2611,scale=19.14014),
        'cost.endometrial_cancer.stage_II': lambda _rng: _rng.gamma(shape=128.5314,scale=126.3273),
        'cost.endometrial_cancer.stage_III': lambda _rng: _rng.gamma(shape=87.08273,scale=348.2097),
        'cost.endometrial_cancer.stage_IV': lambda _rng: _rng.gamma(shape=24.31947,scale=1318.099),
        'cost.endometrial_cancer.recurrence': lambda _rng: _rng.gamma(shape=25,scale=857.967)
    })

def _params_ovarian_cancer_costs(rng: numpy.random.Generator) -> ParameterSource:
    return CallableParameterSource(rng, {
        'cost.ovarian_cancer.early': lambda _rng: _rng.gamma(shape=25,scale=241.7),
        'cost.ovarian_cancer.late': lambda _rng: _rng.gamma(shape=25,scale=481.856),
        'cost.ovarian_cancer.followup_1_3': lambda _rng: _rng.gamma(shape=25,scale=18.082),
        'cost.ovarian_cancer.followup_after_3': lambda _rng: _rng.gamma(shape=25,scale=4.1798),
        'cost.ovarian_cancer.recurrence': lambda _rng: _rng.gamma(shape=25,scale=481.856)
    })

def _params_gynae_surveillance_costs(rng: numpy.random.Generator) -> ParameterSource:
    return CallableParameterSource(rng, {
        'cost.gynae_surveillance.hysteroscopy_tvus_ca125': lambda _rng: _rng.gamma(shape=100,scale=3.2913),
        'cost.gynae_surveillance.tvus_ca125': lambda _rng: _rng.gamma(shape=100,scale=2.2086),
        'cost.gynae_surveillance.hysteroscopy_tvus': lambda _rng: _rng.gamma(shape=100,scale=2.7953)
    })

def _params_colonoscopy_costs(rng: numpy.random.Generator) -> ParameterSource:
    return CallableParameterSource(rng, {
        'cost.colonoscopy.therapeutic': lambda _rng: _rng.gamma(shape=100,scale=7.4929),
        'cost.colonoscopy.diagnostic': lambda _rng: _rng.gamma(shape=100,scale=5.9337)
    })

def _params_risk_reducing_surgery_costs(rng: numpy.random.Generator) -> ParameterSource:
    return CallableParameterSource(rng, {
        'cost.hysterectomy': lambda _rng: _rng.gamma(shape=100,scale=62.8103),
        'cost.hysterectomy_bso': lambda _rng: _rng.gamma(shape=100,scale=62.8103),
        'cost.bso': lambda _rng: _rng.gamma(shape=100,scale=46.6022),
        'cost.hrt.highdose': lambda _rng: _rng.gamma(shape=100,scale=0.7457),
        'cost.hrt.lowdose': lambda _rng: _rng.gamma(shape=100,scale=0.6005)
    })

def _params_aeh_medical_management_costs(rng: numpy.random.Generator) -> ParameterSource:
    return CallableParameterSource(rng, {
        'cost.aeh_management.medical.start': lambda _rng: _rng.gamma(shape=100,scale=3.1797),
        'cost.aeh_management.medical.stop' : lambda _rng: _rng.gamma(shape=100,scale=2.7953)
    })

def _params_aspirin_chemoprophylaxis_costs(rng: numpy.random.Generator) -> ParameterSource:
    return CallableParameterSource(rng, {
        'cost.aspirin_prophylaxis.annual': lambda _rng: _rng.gamma(shape=100,scale=0.7897)
    })
