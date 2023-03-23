import io
import logging
import pkgutil

import numpy
import numpy.random
import pandas

from .sources import CallableParameterSource, ConstantParameterSource, DependentParameterSource, ParameterSamplesSource

def params_colorectal(rng: numpy.random.Generator):
    return (
        _params_colorectal_calibration(rng) +
        _params_colorectal_mimic(rng) +
        _params_colorectal_survival_lynch(rng) +
        _params_colorectal_survival_sporadic(rng) +
        _params_colonoscopy(rng)
    )

def _params_colonoscopy(rng: numpy.random.Generator):
    return CallableParameterSource(rng, {
        'colonoscopy.interval.m': lambda _rng: _rng.normal(1.8, 0.1),
        'colonoscopy.interval.a': lambda _rng: _rng.gamma(shape=8, scale=0.5)
    })

def _params_colorectal_survival_lynch(rng: numpy.random.Generator):
    return CallableParameterSource(rng, {
        'colorectal.survival.lynch.hazardratio': lambda _rng: _rng.lognormal(-2.326252546,0.882090635)
    })

def _params_colorectal_survival_sporadic(rng: numpy.random.Generator):
    def sample(_rng: numpy.random.Generator):
        mu = numpy.array([.9829777,1.8599294,4.2866946,.07160443,-.50317042,-.44487237,-.33542473,.84756902,-4.2940466,-.25818546])
        sigma = numpy.array(
            [
                [0.00230676,0.00193064,0.00193978,0.00001272,0.000007437,-0.00002157,-0.00001068,-0.00003926,-0.00191449,0.00002783],
                [0.00193064,0.00207734,0.00196854,0.0000201,0.000001866,-0.00002546,-0.00001842,-0.00001765,-0.0019206,0.00008589],
                [0.00193978,0.00196854,0.00225333,0.00002792,-0.00004234,-0.00006586,-0.00004111,0.00006934,-0.0019266,0.00050331],
                [0.00001272,0.0000201,0.00002792,0.00017877,-0.00002953,-0.000007958,-0.000003882,-0.000009901,-0.00009903,0.000005434],
                [0.000007437,0.000001866,-0.00004234,-0.00002953,0.00110963,0.00015356,0.0001509,0.00013234,-0.00013512,-0.00008053],
                [-0.00002157,-0.00002546,-0.00006586,-0.000007958,0.00015356,0.0006125,0.00015024,0.00013309,-0.0001172,-0.00007139],
                [-0.00001068,-0.00001842,-0.00004111,-0.000003882,0.0001509,0.00015024,0.00035989,0.00013692,-0.00013002,-0.00005099],
                [-0.00003926,-0.00001765,0.00006934,-0.000009901,0.00013234,0.00013309,0.00013692,0.00031018,-0.00011482,0.0001882],
                [-0.00191449,-0.0019206,-0.0019266,-0.00009903,-0.00013512,-0.0001172,-0.00013002,-0.00011482,0.00205259,-0.000005143],
                [0.00002783,0.00008589,0.00050331,0.000005434,-0.00008053,-0.00007139,-0.00005099,0.0001882,-0.000005143,0.00102872]
            ]
        )
        z = _rng.multivariate_normal(mu, sigma)
        z[9] = numpy.exp(z[9])
        return z

    return DependentParameterSource(
        rng,
        [
            'colorectal.survival.sporadic.rate.stageII',
            'colorectal.survival.sporadic.rate.stageIII',
            'colorectal.survival.sporadic.rate.stageIV',
            'colorectal.survival.sporadic.rate.female',
            'colorectal.survival.sporadic.rate.aged_under45',
            'colorectal.survival.sporadic.rate.aged_45_54',
            'colorectal.survival.sporadic.rate.aged_55_64',
            'colorectal.survival.sporadic.rate.aged_over75',
            'colorectal.survival.sporadic.rate.cons',
            'colorectal.survival.sporadic.frailty',
        ],
        sample
    )

# def _sample_colorectal_survival(rng: numpy.random.Generator):
#     sample = rng.multivariate_normal(
#         mean=[.5433256,17.21469],
#         cov=[[.0062528,-.22811118],[-.22811118,11.049248]]
#     )
#     return sample

# def _params_colorectal_survival(rng: numpy.random.Generator):
#     return DependentParameterSource(
#         rng,
#         ['colorectal.survival.scale', 'colorectal.survival.shape'],
#         _sample_colorectal_survival
#     )

# def _sample_colorectal_recurrence(rng: numpy.random.Generator):
#     sample = rng.multivariate_normal(
#         mean=[-0.1472635,-3.3052684],
#         cov=[[0.01310566,-0.05597987],[-0.05597987,0.38196806]]
#     )
#     sample[0] = numpy.exp(sample[0])
#     return sample

# def _params_colorectal_recurrence(rng: numpy.random.Generator):
#     return DependentParameterSource(
#         rng,
#         ['colorectal.recurrence.shape', 'colorectal.recurrence.rate'],
#         _sample_colorectal_recurrence
#     )

def _params_colorectal_mimic(rng: numpy.random.Generator):
    return ConstantParameterSource({
        'colorectal.norm_lr.male': [
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            2.01e-3,2.01e-3,2.01e-3,2.01e-3,2.01e-3,2.01e-3,2.01e-3,2.01e-3,2.01e-3,2.01e-3,
            2.68e-2,2.68e-2,2.68e-2,2.68e-2,2.68e-2,2.68e-2,2.68e-2,2.68e-2,2.68e-2,2.68e-2,
            1.49e-2,1.49e-2,1.49e-2,1.49e-2,1.49e-2,1.49e-2,1.49e-2,1.49e-2,1.49e-2,1.49e-2,
            8.17e-3,8.17e-3,8.17e-3,8.17e-3,8.17e-3,8.17e-3,8.17e-3,8.17e-3,8.17e-3,8.17e-3,
            4.63e-3,4.63e-3,4.63e-3,4.63e-3,4.63e-3,4.63e-3,4.63e-3,4.63e-3,4.63e-3,4.63e-3,
            3.15e-3,3.15e-3,3.15e-3,3.15e-3,3.15e-3,3.15e-3,3.15e-3,3.15e-3,3.15e-3,3.15e-3,3.15e-3,3.15e-3,3.15e-3,3.15e-3,3.15e-3,3.15e-3
        ],
        'colorectal.lr_hr.male': [
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            2.82e-2,2.82e-2,2.82e-2,2.82e-2,2.82e-2,2.82e-2,2.82e-2,2.82e-2,2.82e-2,2.82e-2,
            3.13e-2,3.13e-2,3.13e-2,3.13e-2,3.13e-2,3.13e-2,3.13e-2,3.13e-2,3.13e-2,3.13e-2,
            2.06e-2,2.06e-2,2.06e-2,2.06e-2,2.06e-2,2.06e-2,2.06e-2,2.06e-2,2.06e-2,2.06e-2,
            1.21e-2,1.21e-2,1.21e-2,1.21e-2,1.21e-2,1.21e-2,1.21e-2,1.21e-2,1.21e-2,1.21e-2,
            1.55e-2,1.55e-2,1.55e-2,1.55e-2,1.55e-2,1.55e-2,1.55e-2,1.55e-2,1.55e-2,1.55e-2,
            9.27e-3,9.27e-3,9.27e-3,9.27e-3,9.27e-3,9.27e-3,9.27e-3,9.27e-3,9.27e-3,9.27e-3,9.27e-3,9.27e-3,9.27e-3,9.27e-3,9.27e-3,9.27e-3
        ],
        'colorectal.hr_crc.male': [
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            9.24e-3,9.24e-3,9.24e-3,9.24e-3,9.24e-3,9.24e-3,9.24e-3,9.24e-3,9.24e-3,9.24e-3,
            1.63e-2,1.63e-2,1.63e-2,1.63e-2,1.63e-2,1.63e-2,1.63e-2,1.63e-2,1.63e-2,1.63e-2,
            1.81e-2,1.81e-2,1.81e-2,1.81e-2,1.81e-2,1.81e-2,1.81e-2,1.81e-2,1.81e-2,1.81e-2,
            2.84e-2,2.84e-2,2.84e-2,2.84e-2,2.84e-2,2.84e-2,2.84e-2,2.84e-2,2.84e-2,2.84e-2,
            5.02e-2,5.02e-2,5.02e-2,5.02e-2,5.02e-2,5.02e-2,5.02e-2,5.02e-2,5.02e-2,5.02e-2,
            3.52e-2,3.52e-2,3.52e-2,3.52e-2,3.52e-2,3.52e-2,3.52e-2,3.52e-2,3.52e-2,3.52e-2,3.52e-2,3.52e-2,3.52e-2,3.52e-2,3.52e-2,3.52e-2
        ],
        'colorectal.norm_crc.male': 5.77e-4,
        
        'colorectal.norm_lr.female': [
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            1.15e-3,1.15e-3,1.15e-3,1.15e-3,1.15e-3,1.15e-3,1.15e-3,1.15e-3,1.15e-3,1.15e-3,
            1.62e-2,1.62e-2,1.62e-2,1.62e-2,1.62e-2,1.62e-2,1.62e-2,1.62e-2,1.62e-2,1.62e-2,
            1.15e-2,1.15e-2,1.15e-2,1.15e-2,1.15e-2,1.15e-2,1.15e-2,1.15e-2,1.15e-2,1.15e-2,
            8.28e-3,8.28e-3,8.28e-3,8.28e-3,8.28e-3,8.28e-3,8.28e-3,8.28e-3,8.28e-3,8.28e-3,
            4.01e-3,4.01e-3,4.01e-3,4.01e-3,4.01e-3,4.01e-3,4.01e-3,4.01e-3,4.01e-3,4.01e-3,
            3.05e-3,3.05e-3,3.05e-3,3.05e-3,3.05e-3,3.05e-3,3.05e-3,3.05e-3,3.05e-3,3.05e-3,3.05e-3,3.05e-3,3.05e-3,3.05e-3,3.05e-3,3.05e-3
        ],
        'colorectal.lr_hr.female': [
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            1.75e-2,1.75e-2,1.75e-2,1.75e-2,1.75e-2,1.75e-2,1.75e-2,1.75e-2,1.75e-2,1.75e-2,
            2.85e-2,2.85e-2,2.85e-2,2.85e-2,2.85e-2,2.85e-2,2.85e-2,2.85e-2,2.85e-2,2.85e-2,
            1.45e-2,1.45e-2,1.45e-2,1.45e-2,1.45e-2,1.45e-2,1.45e-2,1.45e-2,1.45e-2,1.45e-2,
            1.44e-2,1.44e-2,1.44e-2,1.44e-2,1.44e-2,1.44e-2,1.44e-2,1.44e-2,1.44e-2,1.44e-2,
            1.99e-2,1.99e-2,1.99e-2,1.99e-2,1.99e-2,1.99e-2,1.99e-2,1.99e-2,1.99e-2,1.99e-2,
            1.14e-2,1.14e-2,1.14e-2,1.14e-2,1.14e-2,1.14e-2,1.14e-2,1.14e-2,1.14e-2,1.14e-2,1.14e-2,1.14e-2,1.14e-2,1.14e-2,1.14e-2,1.14e-2
        ],
        'colorectal.hr_crc.female': [
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            4.69e-3,4.69e-3,4.69e-3,4.69e-3,4.69e-3,4.69e-3,4.69e-3,4.69e-3,4.69e-3,4.69e-3,
            2.08e-2,2.08e-2,2.08e-2,2.08e-2,2.08e-2,2.08e-2,2.08e-2,2.08e-2,2.08e-2,2.08e-2,
            2.72e-2,2.72e-2,2.72e-2,2.72e-2,2.72e-2,2.72e-2,2.72e-2,2.72e-2,2.72e-2,2.72e-2,
            3.59e-2,3.59e-2,3.59e-2,3.59e-2,3.59e-2,3.59e-2,3.59e-2,3.59e-2,3.59e-2,3.59e-2,
            6.50e-2,6.50e-2,6.50e-2,6.50e-2,6.50e-2,6.50e-2,6.50e-2,6.50e-2,6.50e-2,6.50e-2,
            5.31e-2,5.31e-2,5.31e-2,5.31e-2,5.31e-2,5.31e-2,5.31e-2,5.31e-2,5.31e-2,5.31e-2,5.31e-2,5.31e-2,5.31e-2,5.31e-2,5.31e-2,5.31e-2
        ],
        'colorectal.norm_crc.female': 5.74e-4,

        'colorectal.progression.genpop': [2.93e-1,5.54e-1,3.50e-1],
        'colorectal.presentation': [2.03e-2,1.43e-1,2.74e-1,2.50e-1],

        'colonoscopy.sensitivity.lowrisk': 0.7651,
        'colonoscopy.sensitivity.highrisk': 0.9791,
        'colonoscopy.sensitivity.crc': 0.9656
    })

def _params_colorectal_calibration(rng: numpy.random.Generator):
    # Load the CSV data
    logging.debug('Loading colorectal cancer calibration posterior samples')
    df = pandas.concat(
        (
            pandas.read_csv(
                io.BytesIO(
                    pkgutil.get_data(
                        'lynchsyndrome.experiments.common.parameters.data',
                        f'colorectal-{chain}.csv'
                    )
                ),
                comment='#'
            )
            for chain in [1,2,3,4]),
        axis=0
    )
    df.reset_index(drop=True, inplace=True)

    res = pandas.DataFrame(index=pandas.Index(range(len(df))))

    res['theta_cons.MLH1'] = df.loc[:,'theta_cons.1']
    res['theta_cons.MSH2'] = df.loc[:,'theta_cons.2']
    res['theta_cons.MSH6'] = df.loc[:,'theta_cons.3']
    res['theta_age.MLH1'] = df.loc[:,'theta_age.1']
    res['theta_age.MSH2'] = df.loc[:,'theta_age.2']
    res['theta_age.MSH6'] = df.loc[:,'theta_age.3']
    res['rho0'] = df.loc[:,'rho0']
    res['rho1'] = df.loc[:,'rho1']
    res['rho2'] = df.loc[:,'rho2']
    res['eta.MLH1'] = df.loc[:,'eta.1']
    res['eta.MSH2'] = df.loc[:,'eta.2']
    res['eta.MSH6'] = df.loc[:,'eta.3']
    res['eta.PMS2'] = df.loc[:,'eta.4']
    res['phi.MLH1'] = df.loc[:,'phi.1']
    res['phi.MSH2'] = df.loc[:,'phi.2']
    res['phi.MSH6'] = df.loc[:,'phi.3']
    res['psi.MLH1'] = df.loc[:,'psi.1']
    res['psi.MSH2'] = df.loc[:,'psi.2']
    res['psi.MSH6'] = df.loc[:,'psi.3']
    res['kappa'] = df.loc[:,'kappa']
    res['nu'] = df.loc[:,'nu']

    progression_lynch_names = ['rate_LS_progression_AB','rate_LS_progression_BC','rate_LS_progression_CD']
    progression_lynch = df.loc[:,progression_lynch_names]
    progression_lynch_series = pandas.Series([
        progression_lynch.xs(i).to_list()
        for i in range(len(progression_lynch))
    ])
    res.insert(20, 'progression.lynch', progression_lynch_series)

    res = res.add_prefix('colorectal.')

    return ParameterSamplesSource(rng, res, len(res))


