import io
import logging
import pkgutil

import numpy
import pandas

from .sources import CallableParameterSource, CompositeParameterSource, ParameterSamplesSource, ParameterSource

def params_endometrial(rng: numpy.random.Generator):
    return CompositeParameterSource([
        _params_medical_management(rng),
        _params_endometrial_calibration(rng),
        _params_endometrial_survival(rng),
        _params_endometrial_sporadic_survival(rng)
    ])


def _params_medical_management(rng: numpy.random.Generator) -> ParameterSource:
    def sample(_rng: numpy.random.Generator):
        res = numpy.zeros(4)
        temp = _rng.dirichlet([1.0,8.0,1.0])
        res[0:2] = temp[0:2]
        res[2] = _rng.normal(40.0, 2.0)
        res[3] = _rng.gamma(shape=4.0, scale=1.0)
        return res

    return CallableParameterSource(rng, {
        'endometrial.aeh_management': sample,
        'endometrial.aeh_management.conservative_success': lambda _rng: _rng.beta(12.3161,3.2739),
        'endometrial.early_ec.response_rate': lambda _rng: _rng.beta(11, 5)
    })


def _params_endometrial_calibration(rng: numpy.random.Generator):
    # Load the CSV data
    logging.debug('Loading endometrial cancer calibration posterior samples')
    df = pandas.concat(
        (
            pandas.read_csv(
                io.BytesIO(
                    pkgutil.get_data(
                        'lynchsyndrome.experiments.common.parameters.data',
                        f'endometrial-{chain}.csv'
                    )
                ),
                comment='#'
            )
            for chain in [1,2,3]),
        axis=0
    )
    df.reset_index(drop=True, inplace=True)

    logging.debug('Reshaping into desired form')
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

    res['regression'] = df['rho']

    symptomatic_names = [f'xi.{i}' for i in range(1, 6)]
    symptomatic = df.loc[:,symptomatic_names]
    symptomatic_series = pandas.Series([
        symptomatic.xs(i).to_list()
        for i in range(len(symptomatic))
    ])
    res.insert(6, 'symptomatic', symptomatic_series)

    progression_names = [f'lambda.{i}' for i in range(1, 5)]
    progression = df.loc[:,progression_names]
    progression_series = pandas.Series([
        progression.xs(i).to_list()
        for i in range(len(progression))
    ])
    res.insert(7, 'progression', progression_series)

    diagnosis_names = [f'delta.{i}' for i in range(1, 6)]
    diagnosis = df.loc[:,diagnosis_names]
    diagnosis_series = pandas.Series([
        diagnosis.xs(i).to_list()
        for i in range(len(diagnosis))
    ])
    res.insert(8, 'diagnosis', diagnosis_series)

    res = res.add_prefix('endometrial.')

    logging.debug('Done')

    return ParameterSamplesSource(rng, res, len(res))


def _params_endometrial_survival(rng: numpy.random.Generator):
    return CallableParameterSource(rng, {
        'endometrial.survival.lynch.late': lambda _rng: _rng.lognormal(mean=-2.34,sigma=0.71),
        'endometrial.recurrence.lynch.early': lambda _rng: _rng.lognormal(mean=-3.48,sigma=0.58),
        'endometrial.postrecurrence.lynch': lambda _rng: _rng.lognormal(mean=-2.34,sigma=0.71)
    })


def _params_endometrial_sporadic_survival(rng: numpy.random.Generator):
    # Load the CSV data
    logging.debug('Loading endometrial cancer calibration posterior samples')
    df = pandas.concat(
        (
            pandas.read_csv(
                io.BytesIO(
                    pkgutil.get_data(
                        'lynchsyndrome.experiments.common.parameters.data',
                        f'endometrial-sporadic-survival-{chain}.csv'
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

    res = res.add_prefix('endometrial.')

    return ParameterSamplesSource(rng, res, len(res))

