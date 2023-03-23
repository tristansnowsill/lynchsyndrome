# -*- coding: utf-8 -*-
"""Tests for `lynchsyndrome.benefits` module."""

import numpy as np
import pandas
import pandas.testing
import pytest
import simpy

from lynchsyndrome.benefits import Benefits
from lynchsyndrome.death import CauseOfDeath, DeathMetadata
from lynchsyndrome.diagnosis import CancerSite, CancerStage, DiagnosisMetadata, RouteToDiagnosis
from lynchsyndrome.individual import Individual
from lynchsyndrome.recurrence import RecurrenceMetadata
from lynchsyndrome.reporting import EventMetadata, ReportingEvent
from lynchsyndrome.sex import Sex
from lynchsyndrome.surveillance import SurveillanceMetadata, SurveillanceSite, SurveillanceTechnology
from lynchsyndrome.utility import BaselineUtility, HSEBaselineUtility, MinimumOverallUtility, OverallUtility


@pytest.fixture
def events_surveillance():
    return pandas.DataFrame({ 
        'time': np.zeros(6),
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
        'time': np.zeros(9),
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

@pytest.fixture
def params():
    return {
        'analysis.discount_rate.ly': 0,
        'analysis.discount_rate.qaly': 0.035,
        'utility.baseline': 0.80,
        'utility.colorectal.nonmetastatic.year1': 0.87,
        'utility.colorectal.nonmetastatic.beyond': 0.92,
        'utility.colorectal.metastatic.year1': 0.68,
        'utility.colorectal.metastatic.beyond': 0.73,
        'utility.endometrial.nocancer': 1.0,
        'utility.endometrial.early': 0.98,
        'utility.endometrial.advanced': 0.89,
        'utility.endometrial.recurrence': 0.89,
        'utility.ovarian.nocancer': 0.80,
        'utility.ovarian.early': 0.78,
        'utility.ovarian.advanced': 0.72,
        'utility.ovarian.recurrence': 0.72
    }

@pytest.fixture
def overall_utility() -> OverallUtility:
    return MinimumOverallUtility()

@pytest.fixture
def baseline_utility(params) -> BaselineUtility:
    class ConstantBaselineUtility(BaselineUtility):
        def __init__(self, params):
            self.params = params

        def utility(self, sex: Sex, age: float) -> float:
            return self.params['utility.baseline']
    
    return ConstantBaselineUtility(params)

@pytest.fixture
def benefits(params, overall_utility, baseline_utility) -> Benefits:
    return Benefits(params, overall_utility, baseline_utility)


def test_it_calculates_ovarian_qalys(events_ovarian_early, benefits: Benefits):
    expected_life_years = pandas.Series([2.0, 4.0, 1.0], name='life_years')
    expected_qalys = pandas.Series([1.546198655,2.721045394,0.575760197], name='qalys')
    individual = Individual(simpy.Environment(), None, None)
    individual.sex = Sex.FEMALE
    individual.age = 60
    pandas.testing.assert_series_equal(expected_life_years, benefits.life_years(events_ovarian_early))
    pandas.testing.assert_series_equal(expected_qalys, benefits.qalys(individual, events_ovarian_early))

def test_it_calculates_combined_qalys(benefits: Benefits):
    events = pandas.DataFrame({
        'time': [2.0, 4.0, 6.0, 8.0, 10.0],
        'event': [
            ReportingEvent.CANCER_DIAGNOSIS,
            ReportingEvent.CANCER_DIAGNOSIS,
            ReportingEvent.CANCER_RECURRENCE,
            ReportingEvent.CANCER_RECURRENCE,
            ReportingEvent.REACH_TIME_HORIZON
        ],
        'metadata': [
            DiagnosisMetadata(RouteToDiagnosis.SYMPTOMATIC_PRESENTATION, CancerSite.ENDOMETRIUM, CancerStage.STAGE_II),
            DiagnosisMetadata(RouteToDiagnosis.SYMPTOMATIC_PRESENTATION, CancerSite.OVARIES, CancerStage.STAGE_II),
            RecurrenceMetadata(CancerSite.ENDOMETRIUM, CancerStage.STAGE_II),
            RecurrenceMetadata(CancerSite.OVARIES, CancerStage.STAGE_II),
            EventMetadata()
        ]
    })
    expected_qalys = pandas.Series([1.546198655,1.407308164,1.31373723,1.084881482,1.012748472], name='qalys')
    individual = Individual(simpy.Environment(), None, None)
    individual.sex = Sex.FEMALE
    individual.age = 60
    pandas.testing.assert_series_equal(expected_qalys, benefits.qalys(individual, events))

# def test_it_calculates_endometrial_event_costs(events_endometrial, costing: Costing, params):
#     event_costs = costing.event_costs(events_endometrial)
#     assert event_costs[0] == params['cost.endometrial_cancer.stage_I']
#     assert event_costs[1] == params['cost.endometrial_cancer.stage_II']
#     assert event_costs[2] == params['cost.endometrial_cancer.stage_III']
#     assert event_costs[3] == params['cost.endometrial_cancer.stage_IV']
#     assert event_costs[4] == params['cost.endometrial_cancer.recurrence']
#     assert event_costs[5] == params['cost.endometrial_cancer.recurrence']
#     assert event_costs[6] == 0.0
#     assert event_costs[7] == 0.0
#     assert event_costs[8] == 0.0

# def test_it_calculates_ovarian_cancer_costs(events_ovarian_early, costing: Costing, params):
#     event_costs = costing.event_costs(events_ovarian_early)
#     running_costs = costing.running_costs(events_ovarian_early)
#     expected_event_costs = pandas.Series([5601.064202,9762.007732,0])
#     expected_running_costs = pandas.Series([0,1147.11131,79.96669397 ])
#     pandas.testing.assert_series_equal(event_costs, expected_event_costs)
#     pandas.testing.assert_series_equal(running_costs, expected_running_costs)

# def test_it_calculates_surveillance_costs(events_surveillance, costing: Costing, params):
#     event_costs = costing.event_costs(events_surveillance)
#     assert event_costs[0] == params['cost.gynae_surveillance.hysteroscopy_tvus']
#     assert event_costs[1] == params['cost.gynae_surveillance.hysteroscopy_tvus_ca125']
#     assert event_costs[2] == params['cost.gynae_surveillance.hysteroscopy_tvus_ca125']
#     assert event_costs[3] == params['cost.gynae_surveillance.tvus_ca125']
#     assert event_costs[4] == params['cost.colonoscopy.diagnostic']
#     assert event_costs[5] == params['cost.colonoscopy.therapeutic']

# def test_it_discounts_costs(costing: Costing, params):
#     events = pandas.DataFrame({
#         'time': [10.0],
#         'event': [ReportingEvent.CANCER_SURVEILLANCE],
#         'metadata': [SurveillanceMetadata(SurveillanceSite.COLORECTUM, {SurveillanceTechnology.DIAGNOSTIC_COLONOSCOPY})]
#     })
#     event_costs = costing.event_costs(events)
#     assert pytest.approx(event_costs[0]) == params['cost.colonoscopy.diagnostic'] * (1.0 + params['analysis.discount_rate.cost']) ** (-10.0)

# def test_it_calculates_aspirin_costs(costing: Costing, params):
#     events = pandas.DataFrame({
#         'time': np.arange(10),
#         'event': [
#             ReportingEvent.CANCER_SURVEILLANCE,     # t = 0
#             ReportingEvent.START_CHEMOPROPHYLAXIS,  # t = 1
#             ReportingEvent.CANCER_SURVEILLANCE,     # t = 2
#             ReportingEvent.CANCER_SURVEILLANCE,     # t = 3
#             ReportingEvent.STOP_CHEMOPROPHYLAXIS,   # t = 4
#             ReportingEvent.CANCER_SURVEILLANCE,     # t = 5
#             ReportingEvent.START_CHEMOPROPHYLAXIS,  # t = 6
#             ReportingEvent.CANCER_SURVEILLANCE,     # t = 7
#             ReportingEvent.STOP_CHEMOPROPHYLAXIS,   # t = 8
#             ReportingEvent.REACH_TIME_HORIZON       # t = 9
#         ],
#         'metadata': [EventMetadata() for _ in range(10)]
#     })
#     running_costs = costing.running_aspirin_costs(events)
#     expected_costs = params['cost.aspirin_prophylaxis.annual'] * \
#         pandas.Series([0,0,0.949753473,0.917636206,0.88660503,0,0,0.79966694,0.772625063,0])
#     pandas.testing.assert_series_equal(running_costs, expected_costs)
