import io
import logging
import pkgutil

import numpy
import numpy.random
import pandas

from .sources import DependentParameterSource, ParameterSamplesSource

def params_ovarian(rng: numpy.random.Generator):
    return _params_ovarian_calibration(rng) + _params_ovarian_survival(rng) + _params_ovarian_recurrence(rng) + _params_ovarian_sporadic_survival(rng)

def _sample_ovarian_survival(rng: numpy.random.Generator):
    sample = rng.multivariate_normal(
        mean=[-.6100465,2.845763],
        cov=[[.00783712,-.00902379],[-.00902379,.01379546]]
    )
    return numpy.exp(sample)

def _params_ovarian_survival(rng: numpy.random.Generator):
    return DependentParameterSource(
        rng,
        ['ovarian.survival.lynch.shape', 'ovarian.survival.lynch.scale'],
        _sample_ovarian_survival
    )

def _params_ovarian_sporadic_survival(rng: numpy.random.Generator):
    # Load the CSV data
    logging.debug('Loading ovarian cancer calibration posterior samples')
    df = pandas.concat(
        (
            pandas.read_csv(
                io.BytesIO(
                    pkgutil.get_data(
                        'lynchsyndrome.experiments.common.parameters.data',
                        f'ovarian-sporadic-survival-{chain}.csv'
                    )
                ),
                comment='#'
            )
            for chain in [1,2,3,4]),
        axis=0
    )
    df.reset_index(drop=True, inplace=True)

    res = pandas.DataFrame(index=pandas.Index(range(len(df))))

    recurrence_names = [f'lambda.{i}' for i in range(1, 5)]
    recurrence = df.loc[:,recurrence_names]
    recurrence_series = pandas.Series([
        recurrence.xs(i).to_list()
        for i in range(len(recurrence))
    ])
    res.insert(0, 'recurrence.genpop', recurrence_series)

    res['postrecurrence.genpop'] = df['delta']
    res['frailty.genpop'] = df['theta']

    res = res.add_prefix('ovarian.')

    return ParameterSamplesSource(rng, res, len(res))


def _sample_ovarian_recurrence(rng: numpy.random.Generator):
    sample = rng.multivariate_normal(
        mean=[-0.1472635,-3.3052684],
        cov=[[0.01310566,-0.05597987],[-0.05597987,0.38196806]]
    )
    sample[1] = numpy.exp(sample[1])
    return sample

def _params_ovarian_recurrence(rng: numpy.random.Generator):
    return DependentParameterSource(
        rng,
        ['ovarian.recurrence.lynch.shape', 'ovarian.recurrence.lynch.rate'],
        _sample_ovarian_recurrence
    )

def _params_ovarian_calibration(rng: numpy.random.Generator):
    # Load the CSV data
    logging.debug('Loading ovarian cancer calibration posterior samples')
    df = pandas.concat(
        (
            pandas.read_csv(
                io.BytesIO(
                    pkgutil.get_data(
                        'lynchsyndrome.experiments.common.parameters.data',
                        f'ovarian-{chain}.csv'
                    )
                ),
                comment='#'
            )
            for chain in [1]),
        axis=0
    )
    df.reset_index(drop=True, inplace=True)

    res = pandas.DataFrame(index=pandas.Index(range(len(df))))

    incidence_MLH1_names = [f'alpha.1.{i}' for i in range(1, 7)]
    incidence_MLH1 = df.loc[:,incidence_MLH1_names]
    incidence_MLH1_series = pandas.Series([
        incidence_MLH1.xs(i).to_list()
        for i in range(len(incidence_MLH1))
    ])
    res.insert(0, 'incidence.MLH1', incidence_MLH1_series)

    incidence_MSH2_names = [f'alpha.2.{i}' for i in range(1, 7)]
    incidence_MSH2 = df.loc[:,incidence_MSH2_names]
    incidence_MSH2_series = pandas.Series([
        incidence_MSH2.xs(i).to_list()
        for i in range(len(incidence_MSH2))
    ])
    res.insert(1, 'incidence.MSH2', incidence_MSH2_series)

    incidence_MSH6_names = [f'alpha.3.{i}' for i in range(1, 7)]
    incidence_MSH6 = df.loc[:,incidence_MSH6_names]
    incidence_MSH6_series = pandas.Series([
        incidence_MSH6.xs(i).to_list()
        for i in range(len(incidence_MSH6))
    ])
    res.insert(2, 'incidence.MSH6', incidence_MSH6_series)

    incidence_PMS2_names = [f'alpha.4.{i}' for i in range(1, 7)]
    incidence_PMS2 = df.loc[:,incidence_PMS2_names]
    incidence_PMS2_series = pandas.Series([
        incidence_PMS2.xs(i).to_list()
        for i in range(len(incidence_PMS2))
    ])
    res.insert(3, 'incidence.PMS2', incidence_PMS2_series)

    incidence_genpop_names = [f'alpha.5.{i}' for i in range(1, 7)]
    incidence_genpop = df.loc[:,incidence_genpop_names]
    incidence_genpop_series = pandas.Series([
        incidence_genpop.xs(i).to_list()
        for i in range(len(incidence_genpop))
    ])
    res.insert(4, 'incidence.genpop', incidence_genpop_series)

    progression_lynch_names = [f'lambda_LS.{i}' for i in range(1, 4)]
    progression_lynch = df.loc[:,progression_lynch_names]
    progression_lynch_series = pandas.Series([
        progression_lynch.xs(i).to_list()
        for i in range(len(progression_lynch))
    ])
    res.insert(5, 'progression.lynch', progression_lynch_series)

    progression_genpop_names = [f'lambda_sporadic.{i}' for i in range(1, 4)]
    progression_genpop = df.loc[:,progression_genpop_names]
    progression_genpop_series = pandas.Series([
        progression_genpop.xs(i).to_list()
        for i in range(len(progression_genpop))
    ])
    res.insert(6, 'progression.genpop', progression_genpop_series)

    presentation_names = [f'xi.{i}' for i in range(1, 5)]
    presentation = df.loc[:,presentation_names]
    presentation_series = pandas.Series([
        presentation.xs(i).to_list()
        for i in range(len(presentation))
    ])
    res.insert(7, 'presentation', presentation_series)

    res = res.add_prefix('ovarian.')

    return ParameterSamplesSource(rng, res, len(res))


