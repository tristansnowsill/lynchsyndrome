import argparse
from dataclasses import dataclass
import logging
import os.path
import tarfile
import tempfile
from functools import partial
from typing import Any, Callable, List, Mapping, Tuple, Union
from uuid import UUID

import numpy
import pandas
import simpy

from lynchsyndrome.colonoscopy import BiennialColonoscopy
from lynchsyndrome.experiments.common.competing_options import Pathway, PathwayPoint
from lynchsyndrome.experiments.common.experiment import Experiment, InterimResult, setup
from lynchsyndrome.experiments.common.results import (
    CancerOutcomes, ParameterReport, PatientLevelCostReport, PatientLevelLifeYearsReport,
    PatientLevelQALYsReport, ResourceUseReport, SurvivalType)
from lynchsyndrome.genetics import ConstitutionalMMR
from lynchsyndrome.gynae_surveillance import AnnualGynaecologicalSurveillance
from lynchsyndrome.individual import EnterModelMetadata, IndividualBuilder
from lynchsyndrome.reporting import ReportingEvent
from lynchsyndrome.risk_reducing_surgery import OfferRiskReducingHBSO
from lynchsyndrome.sex import Sex
from lynchsyndrome.utility import HSEBaselineUtility, LinearIndexOverallUtilityBasu2009

from .risk_reducing_surgery import ForceRiskReducingHBSO, ForceTwoStageRRGS

def population(
    env: simpy.Environment,
    init_rng: numpy.random.Generator,
    sim_rng: numpy.random.Generator,
    params: Mapping[str, Any],
    n: int,
    age: Union[float, Callable[[], float]],
    sex: Union[Sex, Callable[[], Sex]],
    genotype: Union[ConstitutionalMMR, Callable[[], ConstitutionalMMR]]
):
    individual_builder = IndividualBuilder(env, init_rng, sim_rng, params)
    individual_builder.set_age(age).set_sex(sex).set_genotype(genotype)

    return individual_builder.create_many(n)

def sanitise_for_serialisation(df: pandas.DataFrame, competing_options: pandas.DataFrame) -> pandas.DataFrame:
    if 'pathway_uuid' in df.index.names:
        df = pandas.merge(competing_options, df, how="inner", left_index=True, right_index=True)
        df.reset_index('pathway_uuid', drop=True, inplace=True)
    
    if 'params_uuid' in df.index.names:
        df['params_uuid'] = df.index.get_level_values('params_uuid')
        df['params_uuid'] = df['params_uuid'].apply(str)
        df.reset_index('params_uuid', drop=True, inplace=True)

    if 'individual_uuid' in df.index.names:
        df['individual_uuid'] = df.index.get_level_values('individual_uuid')
        df['individual_uuid'] = df['individual_uuid'].apply(str)
        df.reset_index('individual_uuid', drop=True, inplace=True)
    
    if 'site' in df.index.names:
        df.reset_index('site', drop=False, inplace=True)
    
    if 'survival_type' in df.index.names:
        df.reset_index('survival_type', drop=False, inplace=True)

    df.index = df.index.to_flat_index()
    return df.reset_index()

def simulate(opts):
    events_dest = opts.events
    params_dest = opts.params

    n_individuals = opts.n_individuals
    n_psa = opts.n_psa
    age = opts.age
    genotype = {
        'path_MLH1': ConstitutionalMMR.PATH_MLH1,
        'path_MSH2': ConstitutionalMMR.PATH_MSH2,
        'path_MSH6': ConstitutionalMMR.PATH_MSH6,
        'path_PMS2': ConstitutionalMMR.PATH_PMS2
    }[opts.genotype]

    injector = setup()
    logging.warning('A warning message')
    logging.info('An info message')
    logging.debug('A debug message')
    
    print(">> Preparing Experiment instance...")
    experiment = injector.get(Experiment)
    experiment.set_population(partial(population, n=n_individuals, age=age, sex=Sex.FEMALE, genotype=genotype))
    experiment.set_psa_samples(n_psa)

    # Individual interventions
    biennial_colonoscopy = BiennialColonoscopy(25, 75)
    surv_30 = AnnualGynaecologicalSurveillance(30, 75)
    surv_35 = AnnualGynaecologicalSurveillance(35, 75)
    force_hbso_35 = ForceRiskReducingHBSO(35)
    force_hbso_40 = ForceRiskReducingHBSO(40)
    force_hbso_50 = ForceRiskReducingHBSO(50)
    force_two_stage = ForceTwoStageRRGS(40, 50)

    # Register competing options with experiment
    basic = {
        PathwayPoint.COLORECTAL_SURVEILLANCE: { biennial_colonoscopy }
    }
    
    experiment.add_competing_option(Pathway(basic, 'Nothing'))
    experiment.add_competing_option(Pathway(dict({
        PathwayPoint.GYNAECOLOGICAL_RISK_REDUCING_SURGERY: { force_hbso_35 }
    }, **basic), 'HBSO at 35'))
    experiment.add_competing_option(Pathway(dict({
        PathwayPoint.GYNAECOLOGICAL_RISK_REDUCING_SURGERY: { force_hbso_40 }
    }, **basic), 'HBSO at 40'))
    experiment.add_competing_option(Pathway(dict({
        PathwayPoint.GYNAECOLOGICAL_RISK_REDUCING_SURGERY: { force_hbso_50 }
    }, **basic), 'HBSO at 50'))
    experiment.add_competing_option(Pathway(dict({
        PathwayPoint.GYNAECOLOGICAL_SURVEILLANCE: { surv_30 }
    }, **basic), 'Surveillance from 30'))
    experiment.add_competing_option(Pathway(dict({
        PathwayPoint.GYNAECOLOGICAL_SURVEILLANCE: { surv_35 }
    }, **basic), 'Surveillance from 35'))
    experiment.add_competing_option(Pathway(dict({
        PathwayPoint.GYNECOLOGICAL_RISK_REDUCING_SURGERY: { force_two_stage }
    }, **basic), 'Two-stage surgical approach'))
    experiment.add_competing_option(Pathway(dict({
        PathwayPoint.GYNECOLOGICAL_RISK_REDUCING_SURGERY: { surv_30 },
        PathwayPoint.GYNAECOLOGICAL_RISK_REDUCING_SURGERY: { force_hbso_40 }
    }, **basic), 'Surveillance from 30, HBSO at 40'))

    # Set up simple reports to extract event traces and parameters
    experiment.add_events_report('events', lambda x: x)
    experiment.add_params_report('params', lambda x: x)

    # Run the experiment
    print(">> Running the Experiment...")
    reports = experiment.run()

    # Pickle the results
    print(">> Writing the experiment results to files...")
    reports['events'].to_pickle(events_dest, compression='gzip')
    reports['params'].to_pickle(params_dest, compression='gzip')
    experiment.get_competing_options().to_pickle(opts.competing_options, compression='gzip')

    print(">> DONE")
    
def analyse(opts):

    @dataclass
    class Ghost:
        """Key data about an individual that was simulated
        
        This will be used in functions instead of an :py:class:`Individual`.
        It is lightweight and can be constructed from a
        :py:class:`EnterModelMetadata`.
        """

        age: float
        sex: Sex
        genotype: ConstitutionalMMR
        uuid: UUID
    
    print(">> Loading events...")
    events: pandas.DataFrame = pandas.read_pickle(opts.events)
    print(">> Loading parameters...")
    params_df: pandas.DataFrame = pandas.read_pickle(opts.params)
    print(">> Loading competing options...")
    competing_options: pandas.Series = pandas.read_pickle(opts.competing_options)

    outputs = list()

    if opts.report_patient_level_outcomes:
        print(">> Reconstructing individuals from events...")
        individuals_running: List[Tuple[UUID, UUID, UUID, Ghost]] = list()
        for individual_uuid, individual_events in events.groupby(level='individual_uuid'):
            ev, md = individual_events['event'].iloc[0], individual_events['metadata'].iloc[0]
            pathway_uuid = individual_events.index.unique(level='pathway_uuid')
            params_uuid = individual_events.index.unique(level='params_uuid')
            if ev is ReportingEvent.ENTER_MODEL and isinstance(md, EnterModelMetadata):
                individuals_running.append((pathway_uuid.array[0], params_uuid.array[0], individual_uuid, Ghost(md.age, md.sex, md.genotype, md.uuid)))
        individuals = pandas.Series(
            (t[3] for t in individuals_running),
            index=pandas.MultiIndex.from_tuples(((t[0], t[1], t[2]) for t in individuals_running), names=['pathway_uuid', 'params_uuid', 'individual_uuid']),
            name='individual'
        )

        print(">> Calculating patient level outcomes (costs, QALYs, life years)...")
        reports: dict[str, Any] = dict()
        interim_results: dict[str, List[Tuple[UUID, InterimResult]]] = {
            'patient_level_costs': list(),
            'patient_level_life_years': list(),
            'patient_level_qalys': list()
        }
        qalys_report = PatientLevelQALYsReport(LinearIndexOverallUtilityBasu2009(), HSEBaselineUtility())
        
        events_by_params = events.groupby(level='params_uuid')
        individuals_by_params = individuals.groupby(level='params_uuid')
        for params_uuid, params in params_df.iterrows():
            evs = events_by_params.get_group(params_uuid).reset_index(level='params_uuid', drop=True)
            inds = individuals_by_params.get_group(params_uuid).reset_index(level='params_uuid', drop=True)
            interim_results['patient_level_costs'].append((
                params_uuid,
                PatientLevelCostReport.process(evs, inds, params)
            ))
            interim_results['patient_level_life_years'].append((
                params_uuid,
                PatientLevelLifeYearsReport.process(evs, inds, params)
            ))
            interim_results['patient_level_qalys'].append((
                params_uuid,
                qalys_report.process(evs, inds, params)
            ))
        
        reports['patient_level_costs'] = PatientLevelCostReport.aggregate(interim_results['patient_level_costs'])
        reports['patient_level_life_years'] = PatientLevelLifeYearsReport.aggregate(interim_results['patient_level_life_years'])
        reports['patient_level_qalys'] = qalys_report.aggregate(interim_results['patient_level_qalys'])
        patient_level_outcomes = pandas.concat(
            (reports[rep] for rep in ['patient_level_costs', 'patient_level_life_years', 'patient_level_qalys']),
            axis=1
        )
        
        outputs.append((patient_level_outcomes, 'patient-level-outcomes.feather'))

    if opts.report_cancer_outcomes:
        print(">> Counting cancer events...")
        colorectal_cancers  = CancerOutcomes.colorectal_cancers(events)
        endometrial_cancers = CancerOutcomes.endometrial_cancers(events)
        ovarian_cancers     = CancerOutcomes.ovarian_cancers(events)
        cancers = pandas.concat(
            (rep.rename('n') for rep in [colorectal_cancers, endometrial_cancers, ovarian_cancers] if len(rep) > 0),
            axis=0,
            keys=[l for l, a in zip(('colorectal', 'endometrial', 'ovarian'), (colorectal_cancers, endometrial_cancers, ovarian_cancers)) if len(a) > 0]
        )
        cancers.index.set_names('site', level=0, inplace=True)

        outputs.append((
            cancers.reset_index(level=['site','cancer_outcome','stage_at_diagnosis','route_to_diagnosis']),
            'cancer-outcomes.feather'))

    if opts.report_cancer_free_survival:
        print(">> Producing cancer free survival data...")
        cancer_free_survival = CancerOutcomes.cancer_free_survival(events)
        outputs.append((cancer_free_survival, 'cancer-free-survival.feather'))

    if opts.report_cancer_survival:
        print(">> Producing cancer survival data...")
        colorectal_cancer_specific_mortality  = CancerOutcomes.colorectal_cancer_survival(events, SurvivalType.CAUSE_SPECIFIC)
        endometrial_cancer_specific_mortality = CancerOutcomes.endometrial_cancer_survival(events, SurvivalType.CAUSE_SPECIFIC)
        ovarian_cancer_specific_mortality     = CancerOutcomes.ovarian_cancer_survival(events, SurvivalType.CAUSE_SPECIFIC)

        colorectal_cancer_crude_survival  = CancerOutcomes.colorectal_cancer_survival(events, SurvivalType.ALL_CAUSE)
        endometrial_cancer_crude_survival = CancerOutcomes.endometrial_cancer_survival(events, SurvivalType.ALL_CAUSE)
        ovarian_cancer_crude_survival     = CancerOutcomes.ovarian_cancer_survival(events, SurvivalType.ALL_CAUSE)

        cansurv_reps = [
            colorectal_cancer_specific_mortality, endometrial_cancer_specific_mortality, ovarian_cancer_specific_mortality,
            colorectal_cancer_crude_survival, endometrial_cancer_crude_survival, ovarian_cancer_crude_survival
        ]
        cansurv_keys = [
            ('colorectal', 'cause-specific'),
            ('endometrial', 'cause-specific'),
            ('ovarian', 'cause-specific'),
            ('colorectal', 'all-cause'),
            ('endometrial', 'all-cause'),
            ('ovarian', 'all-cause')
        ]
        cancer_survival = pandas.concat(
            (rep for rep in cansurv_reps if rep is not None),
            axis=0,
            keys=(cansurv_keys[i] for i in range(len(cansurv_reps)) if cansurv_reps[i] is not None),
            names=['site', 'survival_type']
        )

        outputs.append((cancer_survival, 'cancer-survival.feather'))

    if opts.report_gynae_risk_reduction:
        print(">> Counting gynaecological cancer risk reduction resources...")
        gynae_risk_reduction = ResourceUseReport.gynae_risk_reduction(events)
        outputs.append((gynae_risk_reduction, 'gynae-risk-reduction.feather'))

    # We can't serialise with a MultiIndex or with UUID objects
    competing_options.name = 'competing_option'
    competing_options.index.rename('pathway_uuid', inplace=True)

    # Create gzipped TAR for output
    print(f">> Writing out to {opts.outfile}")
    with tarfile.open(opts.outfile, mode='w:gz') as out:
        # Create temporary files and then add to the gzipped TAR
        with tempfile.TemporaryDirectory() as tmpdir:
            # Handle params
            dest = os.path.join(tmpdir, 'params.feather')
            new_params_index = pandas.Index(
                (str(uuid) for uuid in params_df.index.array),
                name='params_uuid'
            )
            params_df.index = new_params_index
            params_df.reset_index().to_feather(dest)
            out.add(dest, arcname='params.feather')

            # Handle everything else
            for df, fn in outputs:
                dest = os.path.join(tmpdir, fn)
                sanitise_for_serialisation(df, competing_options).to_feather(dest)
                out.add(dest, arcname=fn)

    print(">> DONE")

def nocommand(opts):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='python3 -m lynchsyndrome.experiments.NIHR129713',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='''
        This program has two main commands, simulate and analyse.

        Calling the `simulate` command will cause the program to run a
        simulation of the Lynch syndrome whole disease model and output a
        gzipped Python pickle file of the resulting event traces, a gzipped
        Python pickle file of the parameter set, and an unnecessarily
        gzipped Python pickle file of the competing options.

        The event traces only tell you what happened and when (and generally
        does not include details of events unknown to the health system, e.g.,
        onset of preclinical cancer).

        The `analyse` command takes as its first three arguments the gzipped
        Python pickle files created by the `simulate` command, and then runs
        various reports. These reports are written out to a single gzipped
        tarball containing a number of data files in the Feather format.
        '''
    )
    parser.set_defaults(func=nocommand)
    subparsers = parser.add_subparsers()
    simulate_subparser = subparsers.add_parser('simulate')
    analyse_subparser = subparsers.add_parser('analyse')

    simulate_subparser.add_argument('events', help='destination for event traces')
    simulate_subparser.add_argument('params', help='destination for parameters')
    simulate_subparser.add_argument('competing_options', help='destination for competing options')
    simulate_subparser.add_argument('-n', '--n-individuals', type=int, default=1000, help='number of individuals to simulate in each population')
    simulate_subparser.add_argument('-m', '--n-psa', type=int, default=500, help='number of parameter sets to sample')
    simulate_subparser.add_argument('--age', type=float, default=40.0, help='age of the population')
    simulate_subparser.add_argument('--genotype', choices=['path_MLH1', 'path_MSH2', 'path_MSH6', 'path_PMS2'], default='path_MSH2', help='genotype of the population')
    simulate_subparser.set_defaults(func=simulate)

    analyse_subparser.add_argument('events', help='location of the event traces')
    analyse_subparser.add_argument('params', help='location of the parameters')
    analyse_subparser.add_argument('competing_options', help='location of the competing options')
    analyse_subparser.add_argument('outfile', help='where to output reports')
    analyse_subparser.add_argument(
        '--no-patient-level-outcomes',
        action='store_false',
        dest='report_patient_level_outcomes',
        help='do not include patient level outcomes (costs, life years, QALYs) in outfile'
    )
    analyse_subparser.add_argument(
        '--no-cancer-outcomes',
        action='store_false',
        dest='report_cancer_outcomes',
        help='do not include cancer outcomes (incidence, recurrence, mortality) in outfile'
    )
    analyse_subparser.add_argument(
        '--no-cancer-free-survival',
        action='store_false',
        dest='report_cancer_free_survival',
        help='do not include cancer-free survival in outfile'
    )
    analyse_subparser.add_argument(
        '--no-cancer-survival',
        action='store_false',
        dest='report_cancer_survival',
        help='do not include cancer survival (cause-specific and crude) in outfile'
    )
    analyse_subparser.add_argument(
        '--no-gynae-risk-reduction',
        action='store_false',
        dest='report_gynae_risk_reduction',
        help='do not include use of gynaecological risk reduction resources in outfile'
    )
    analyse_subparser.set_defaults(func=analyse)

    opts = parser.parse_args()
    opts.func(opts)
