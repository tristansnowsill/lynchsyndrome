# -*- coding: utf-8 -*-
"""Tests for `lynchsyndrome.parameters` module."""

import numpy
import pandas
import pandas.testing
import pytest
import simpy
from lynchsyndrome.bowel import BowelState

from lynchsyndrome.costing import Costing
from lynchsyndrome.death import CauseOfDeath, DeathMetadata
from lynchsyndrome.diagnosis import CancerSite, CancerStage, DiagnosisMetadata, RouteToDiagnosis
from lynchsyndrome.experiments.common.providers.parameters import ParameterProvider
from lynchsyndrome.genetics import ConstitutionalMMR
from lynchsyndrome.individual import Individual, IndividualBuilder
from lynchsyndrome.ovaries import MenopausalState, OvarianCancerObservedState, OvarianCancerTrueState, OvarianState, OvarianSurgicalState
from lynchsyndrome.recurrence import RecurrenceMetadata
from lynchsyndrome.reporting import EventMetadata, ReportingEvent
from lynchsyndrome.surveillance import SurveillanceMetadata, SurveillanceSite, SurveillanceTechnology


@pytest.fixture
def events_surveillance():
    return pandas.DataFrame({ 
        'time': numpy.zeros(6),
        'event': [
            ReportingEvent.CANCER_SURVEILLANCE,
            ReportingEvent.CANCER_SURVEILLANCE,
            ReportingEvent.CANCER_SURVEILLANCE,
            ReportingEvent.CANCER_SURVEILLANCE,
            ReportingEvent.CANCER_SURVEILLANCE,
            ReportingEvent.CANCER_SURVEILLANCE
        ],
        'metadata': [
            SurveillanceMetadata(
                SurveillanceSite.ENDOMETRIUM,
                { SurveillanceTechnology.HYSTEROSCOPY, SurveillanceTechnology.TRANSVAGINAL_ULTRASOUND }
            ),
            SurveillanceMetadata(
                SurveillanceSite.FEMALE_REPRODUCTIVE,
                { SurveillanceTechnology.CA125, SurveillanceTechnology.HYSTEROSCOPY, SurveillanceTechnology.TRANSVAGINAL_ULTRASOUND }
            ),
            SurveillanceMetadata(
                SurveillanceSite.FEMALE_REPRODUCTIVE,
                { SurveillanceTechnology.CA125, SurveillanceTechnology.UNDIRECTED_ENDOMETRIAL_BIOPSY, SurveillanceTechnology.TRANSVAGINAL_ULTRASOUND }
            ),
            SurveillanceMetadata(
                SurveillanceSite.OVARIES,
                { SurveillanceTechnology.CA125, SurveillanceTechnology.TRANSVAGINAL_ULTRASOUND }
            ),
            SurveillanceMetadata(
                SurveillanceSite.COLORECTUM,
                { SurveillanceTechnology.DIAGNOSTIC_COLONOSCOPY }
            ),
            SurveillanceMetadata(
                SurveillanceSite.COLORECTUM,
                { SurveillanceTechnology.THERAPEUTIC_COLONOSCOPY }
            )
        ]
    })


@pytest.fixture
def events_endometrial():
    return pandas.DataFrame({
        'time': numpy.zeros(9),
        'event': [
            ReportingEvent.CANCER_DIAGNOSIS,
            ReportingEvent.CANCER_DIAGNOSIS,
            ReportingEvent.CANCER_DIAGNOSIS,
            ReportingEvent.CANCER_DIAGNOSIS,
            ReportingEvent.CANCER_RECURRENCE,
            ReportingEvent.CANCER_RECURRENCE,
            ReportingEvent.CANCER_RECURRENCE,
            ReportingEvent.CANCER_RECURRENCE,
            ReportingEvent.DEATH
        ],
        'metadata': [
            DiagnosisMetadata(RouteToDiagnosis.SYMPTOMATIC_PRESENTATION, CancerSite.ENDOMETRIUM, CancerStage.STAGE_I),
            DiagnosisMetadata(RouteToDiagnosis.SURVEILLANCE, CancerSite.ENDOMETRIUM, CancerStage.STAGE_II),
            DiagnosisMetadata(RouteToDiagnosis.SCREENING, CancerSite.ENDOMETRIUM, CancerStage.STAGE_III),
            DiagnosisMetadata(RouteToDiagnosis.SYMPTOMATIC_PRESENTATION, CancerSite.ENDOMETRIUM, CancerStage.STAGE_IV),
            RecurrenceMetadata(CancerSite.ENDOMETRIUM, CancerStage.STAGE_I),
            RecurrenceMetadata(CancerSite.ENDOMETRIUM, CancerStage.STAGE_II),
            RecurrenceMetadata(CancerSite.ENDOMETRIUM, CancerStage.STAGE_III),
            RecurrenceMetadata(CancerSite.ENDOMETRIUM, CancerStage.STAGE_IV),
            DeathMetadata(CauseOfDeath.ENDOMETRIAL_CANCER)
        ]
    })

@pytest.fixture
def events_ovarian_early():
    return pandas.DataFrame({
        'time': [2.0, 6.0, 7.0],
        'event': [
            ReportingEvent.CANCER_DIAGNOSIS,
            ReportingEvent.CANCER_RECURRENCE,
            ReportingEvent.DEATH
        ],
        'metadata': [
            DiagnosisMetadata(
                RouteToDiagnosis.SYMPTOMATIC_PRESENTATION,
                CancerSite.OVARIES,
                CancerStage.STAGE_II
            ),
            RecurrenceMetadata(CancerSite.OVARIES, CancerStage.STAGE_II),
            DeathMetadata(CauseOfDeath.OVARIAN_CANCER)
        ]
    })

@pytest.fixture(scope="session")
def params():
    return next(ParameterProvider().provide_parameters(numpy.random.default_rng()))

@pytest.fixture
def individual(params):
    builder = IndividualBuilder(
        simpy.Environment(),
        numpy.random.default_rng(),
        numpy.random.default_rng(),
        params
    )
    builder.set_age(40)
    builder.set_genotype(ConstitutionalMMR.PATH_MSH2)
    builder.set_bowel_state(BowelState.NORMAL)
    builder.set_ovarian_state(OvarianState(
        OvarianCancerTrueState.NONE,
        OvarianCancerObservedState.NONE,
        MenopausalState.PREMENOPAUSAL,
        OvarianSurgicalState.OVARIES_INTACT
    ))
    return builder.create()

@pytest.fixture
def costing(params) -> Costing:
    return Costing(params)


def test_it_calculates_endometrial_event_costs(events_endometrial, individual: Individual, costing: Costing, params):
    event_costs = costing.event_costs(events_endometrial, individual)
    assert event_costs[0] == params['cost.endometrial_cancer.stage_I']
    assert event_costs[1] == params['cost.endometrial_cancer.stage_II']
    assert event_costs[2] == params['cost.endometrial_cancer.stage_III']
    assert event_costs[3] == params['cost.endometrial_cancer.stage_IV']
    assert event_costs[4] == params['cost.endometrial_cancer.recurrence']
    assert event_costs[5] == params['cost.endometrial_cancer.recurrence']
    assert event_costs[6] == 0.0
    assert event_costs[7] == 0.0
    assert event_costs[8] == 0.0

def test_it_calculates_ovarian_cancer_costs(events_ovarian_early, individual: Individual, costing: Costing, params):
    event_costs = costing.event_costs(events_ovarian_early, individual)
    running_costs = costing.running_costs(events_ovarian_early, individual)
    expected_event_costs = pandas.Series([
        params['cost.ovarian_cancer.early'] / ((1 + params['analysis.discount_rate.cost']) ** 2.0),
        params['cost.ovarian_cancer.recurrence'] / ((1 + params['analysis.discount_rate.cost']) ** 6.0),
        0
    ])
    drc = numpy.log(1 + params['analysis.discount_rate.cost'])
    expected_running_costs = pandas.Series([
        0,
        params['cost.ovarian_cancer.followup_1_3'] * (numpy.exp(-drc * 2) - numpy.exp(-drc * 5)) / drc +
            params['cost.ovarian_cancer.followup_after_3'] * (numpy.exp(-drc * 5) - numpy.exp(-drc * 6)) / drc,
        params['cost.ovarian_cancer.followup_after_3'] * (numpy.exp(-drc * 6) - numpy.exp(-drc * 7)) / drc
    ])
    pandas.testing.assert_series_equal(event_costs, expected_event_costs)
    pandas.testing.assert_series_equal(running_costs, expected_running_costs)

def test_it_calculates_surveillance_costs(events_surveillance, individual: Individual, costing: Costing, params):
    event_costs = costing.event_costs(events_surveillance, individual)
    assert event_costs[0] == params['cost.gynae_surveillance.hysteroscopy_tvus']
    assert event_costs[1] == params['cost.gynae_surveillance.hysteroscopy_tvus_ca125']
    assert event_costs[2] == params['cost.gynae_surveillance.hysteroscopy_tvus_ca125']
    assert event_costs[3] == params['cost.gynae_surveillance.tvus_ca125']
    assert event_costs[4] == params['cost.colonoscopy.diagnostic']
    assert event_costs[5] == params['cost.colonoscopy.therapeutic']

def test_it_discounts_costs(individual: Individual, costing: Costing, params):
    events = pandas.DataFrame({
        'time': [10.0],
        'event': [ReportingEvent.CANCER_SURVEILLANCE],
        'metadata': [SurveillanceMetadata(SurveillanceSite.COLORECTUM, {SurveillanceTechnology.DIAGNOSTIC_COLONOSCOPY})]
    })
    event_costs = costing.event_costs(events, individual)
    assert pytest.approx(event_costs[0]) == params['cost.colonoscopy.diagnostic'] * (1.0 + params['analysis.discount_rate.cost']) ** (-10.0)

def test_it_calculates_aspirin_costs(individual: Individual, costing: Costing, params):
    events = pandas.DataFrame({
        'time': numpy.arange(10),
        'event': [
            ReportingEvent.CANCER_SURVEILLANCE,     # t = 0
            ReportingEvent.START_CHEMOPROPHYLAXIS,  # t = 1
            ReportingEvent.CANCER_SURVEILLANCE,     # t = 2
            ReportingEvent.CANCER_SURVEILLANCE,     # t = 3
            ReportingEvent.STOP_CHEMOPROPHYLAXIS,   # t = 4
            ReportingEvent.CANCER_SURVEILLANCE,     # t = 5
            ReportingEvent.START_CHEMOPROPHYLAXIS,  # t = 6
            ReportingEvent.CANCER_SURVEILLANCE,     # t = 7
            ReportingEvent.STOP_CHEMOPROPHYLAXIS,   # t = 8
            ReportingEvent.REACH_TIME_HORIZON       # t = 9
        ],
        'metadata': [EventMetadata() for _ in range(10)]
    })
    running_costs = costing.running_aspirin_costs(events, individual)
    expected_costs = params['cost.aspirin_prophylaxis.annual'] * \
        pandas.Series([0,0,0.949753473,0.917636206,0.88660503,0,0,0.79966694,0.772625063,0])
    pandas.testing.assert_series_equal(running_costs, expected_costs)
