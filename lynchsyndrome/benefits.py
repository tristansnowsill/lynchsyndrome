import logging
import re
from typing import Any, List, Mapping

import numpy
import pandas
from scipy.interpolate import KroghInterpolator

from lynchsyndrome.diagnosis import DiagnosisMetadata
from lynchsyndrome.qaly import QuadraticUtilityQALY
from lynchsyndrome.utility import BaselineUtility, BasicSingleConditionUtilityEffect, OverallUtility, SingleConditionUtilityEffect

from .diagnosis import CancerSite, CancerStage, DiagnosisMetadata, RouteToDiagnosis, is_early
from .individual import Individual
from .recurrence import RecurrenceMetadata
from .reporting import EventMetadata, ReportingEvent
from .risk_reducing_surgery import RiskReducingSurgery, RiskReducingSurgeryMetadata
from .surveillance import SurveillanceMetadata, SurveillanceSite, SurveillanceTechnology


class Benefits:
    def __init__(
        self,
        params: Mapping[str, Any],
        utility_combiner: OverallUtility,
        baseline_utility: BaselineUtility
    ):
        self.params = params
        self.utility_combiner = utility_combiner
        self.baseline_utility = baseline_utility
        self.condition_callbacks = [
            self.colorectal_cancer,
            self.endometrial_cancer,
            self.ovarian_cancer,
            self.rrgs
        ]
        self.filter_conditions_callback = self.filter_conditions
    
    def life_years(self, events: pandas.DataFrame) -> pandas.Series:
        """Calculate (discounted) life years from events"""
        t_r = events['time']
        t_l = t_r.shift(periods=1, fill_value=0.0)
        res = self._audc(t_l, t_r, self.params['analysis.discount_rate.ly'])
        res.name = 'life_years'
        return res
    
    def qalys(self, individual: Individual, events: pandas.DataFrame) -> pandas.Series:
        """Calculate (discounted) QALYs from events
        
        This function allows multiple callbacks to process the events to
        determine single condition effects on the QALY weight. These are then
        combined with population norms based on age and sex.

        The callbacks should specifically return a pandas.Series where each
        element is either None or an instance of SingleConditionUtilityEffect
        and the length of the returned series is the same as events.

        In addition, a filter callback is provided so that additional logic
        can be incorporated.

        NOTE - As with running costs, these are applied from the time of the
        previous event (or 0 if this is the first event) to the time of the
        current event.
        """
        discount_rate = self.params['analysis.discount_rate.qaly']

        conditions = pandas.concat(
            [cb(individual, events) for cb in self.condition_callbacks],
            axis=0,
            keys=numpy.arange(len(self.condition_callbacks)),
            names=['condition','record']
        )
        conditions.index = conditions.index.swaplevel(0, 1)
        conditions = conditions.sort_index()
        
        res = pandas.Series(numpy.zeros(len(events)), name='qalys')
        for i in range(len(events.index)):
            # Sample the combined utility at three points and calculate QALYs
            # assuming it is at most order 2
            t0 = 0 if i == 0 else events['time'].iloc[i-1]
            t2 = events['time'].iloc[i]
            t1 = 0.5 * (t0 + t2)

            if t2 == t0:
                res[i] = 0.0
                continue

            records = self.filter_conditions_callback(conditions[(i,)].dropna().to_list())

            u0 = self.utility_combiner.utility(individual.sex, individual.age + t0, self.baseline_utility, *records)
            u1 = self.utility_combiner.utility(individual.sex, individual.age + t1, self.baseline_utility, *records)
            u2 = self.utility_combiner.utility(individual.sex, individual.age + t2, self.baseline_utility, *records)

            interp = KroghInterpolator([t0,t1,t2], [u0,u1,u2])
            derivs = interp.derivatives(0, 3)
            a0 = derivs[0]
            a1 = derivs[1]
            a2 = 0.5 * derivs[2]

            res[i] = QuadraticUtilityQALY(t0, t2, a0, a1, a2).discounted(discount_rate)
        
        return res
    
    def ovarian_cancer(self, _: Individual, events: pandas.DataFrame) -> pandas.Series:
        res = pandas.Series([None]).repeat(len(events))
        res.index = events.index
        for i in range(1, len(events)):
            ev, md = events['event'].iloc[i-1], events['metadata'].iloc[i-1]
            if ev is ReportingEvent.CANCER_DIAGNOSIS and isinstance(md, DiagnosisMetadata):
                if md.site is CancerSite.OVARIES:
                    res.iloc[i:] = BasicSingleConditionUtilityEffect.create_from_affected_and_unaffected(
                        self.params['utility.ovarian.early'] if is_early(md.stage) else self.params['utility.ovarian.advanced'],
                        self.params['utility.ovarian.nocancer'],
                        'Early stage ovarian cancer' if is_early(md.stage) else 'Advanced ovarian cancer'
                    )
            elif ev is ReportingEvent.CANCER_RECURRENCE and isinstance(md, RecurrenceMetadata):
                if md.site is CancerSite.OVARIES:
                    res.iloc[i:] = BasicSingleConditionUtilityEffect.create_from_affected_and_unaffected(
                        self.params['utility.ovarian.recurrence'],
                        self.params['utility.ovarian.nocancer'],
                        'Recurrent ovarian cancer'
                    )

        return res
    
    def endometrial_cancer(self, _: Individual, events: pandas.DataFrame) -> pandas.Series:
        res = pandas.Series([None]).repeat(len(events))
        res.index = events.index
        for i in range(1, len(events)):
            ev, md = events['event'].iloc[i-1], events['metadata'].iloc[i-1]
            if ev is ReportingEvent.CANCER_DIAGNOSIS and isinstance(md, DiagnosisMetadata):
                if md.site is CancerSite.ENDOMETRIUM:
                    res.iloc[i:] = BasicSingleConditionUtilityEffect.create_from_affected_and_unaffected(
                        self.params['utility.endometrial.early'] if is_early(md.stage) else self.params['utility.endometrial.advanced'],
                        self.params['utility.endometrial.nocancer'],
                        'Early stage endometrial cancer' if is_early(md.stage) else 'Advanced endometrial cancer'
                    )
            elif ev is ReportingEvent.CANCER_RECURRENCE and isinstance(md, RecurrenceMetadata):
                if md.site is CancerSite.ENDOMETRIUM:
                    res.iloc[i:] = BasicSingleConditionUtilityEffect.create_from_affected_and_unaffected(
                        self.params['utility.endometrial.recurrence'],
                        self.params['utility.endometrial.nocancer'],
                        'Recurrent endometrial cancer'
                    )

        return res
    
    def colorectal_cancer(self, _: Individual, events: pandas.DataFrame) -> pandas.Series:
        # The utility values for colorectal cancer are applied differently for
        # the first 12 months. Rather than attempt to insert a new record for
        # this change point, we instead return a weighted average utility if
        # the record spans the 12 month point
        um_nonmet_y1 = self.params['utility.colorectal.nonmetastatic.year1']
        um_nonmet_y2 = self.params['utility.colorectal.nonmetastatic.beyond']
        um_met_y1 = self.params['utility.colorectal.metastatic.year1']
        um_met_y2 = self.params['utility.colorectal.metastatic.beyond']
        res = pandas.Series([None]).repeat(len(events))
        res.index = events.index
        t_diag = None
        for i in range(1, len(events)):
            ev, md = events['event'].iloc[i-1], events['metadata'].iloc[i-1]
            if ev is ReportingEvent.CANCER_DIAGNOSIS and isinstance(md, DiagnosisMetadata):
                if md.site is CancerSite.COLORECTUM:
                    t_diag = events['time'].iloc[i-1]
                    for j in range(i, len(events)):
                        t_l, t_r = events['time'].iloc[j-1], events['time'].iloc[j]
                        w_y1, w_y2 = None, None
                        if t_r <= t_diag + 1.0:
                            w_y1, w_y2 = 1.0, 0.0
                        elif t_l >= t_diag + 1.0:
                            w_y1, w_y2 = 0.0, 1.0
                        else:
                            w_y1 = (t_diag + 1.0 - t_l) / (t_r - t_l)
                            w_y2 = 1.0 - w_y1
                        if md.stage is CancerStage.STAGE_IV:
                            res.iloc[j] = BasicSingleConditionUtilityEffect.create_from_affected_and_unaffected(
                                w_y1 * um_met_y1 + w_y2 * um_met_y2,
                                1.0,
                                'Metastatic colorectal cancer'
                            )
                        else:
                            res.iloc[j] = BasicSingleConditionUtilityEffect.create_from_affected_and_unaffected(
                                w_y1 * um_nonmet_y1 + w_y2 * um_nonmet_y2,
                                1.0,
                                'Non-metastatic colorectal cancer'
                            )

        return res
    
    def rrgs(self, individual: Individual, events: pandas.DataFrame) -> pandas.Series:
        res = pandas.Series([None]).repeat(len(events))
        res.index = events.index

        current_surg_str = ''
        for i in range(1, len(events)):
            ev, md = events['event'].iloc[i-1], events['metadata'].iloc[i-1]
            if ev is ReportingEvent.RISK_REDUCING_SURGERY and isinstance(md, RiskReducingSurgeryMetadata):
                if md.surgery is RiskReducingSurgery.HBSO:
                    current_surg_str = 'hbso'
                elif md.surgery is RiskReducingSurgery.HBS:
                    current_surg_str = 'hbs'
                elif md.surgery is RiskReducingSurgery.HYSTERECTOMY and current_surg_str == '':
                    current_surg_str = 'hyst'
                elif md.surgery is RiskReducingSurgery.HYSTERECTOMY and current_surg_str == 'bso':
                    current_surg_str = 'hbso'
                elif md.surgery is RiskReducingSurgery.BSO and current_surg_str == '':
                    current_surg_str = 'bso'
                elif md.surgery in [RiskReducingSurgery.BILATERAL_OOPHORECTOMY, RiskReducingSurgery.BSO] and current_surg_str in ['hyst', 'hbs']:
                    current_surg_str = 'hbso'
                else:
                    logging.warning(
                        'unable to estimate utility effect of risk-reducing surgery %s (pre-existing surgery: %s)',
                        md.surgery, current_surg_str
                    )
                    continue

                # For the current and future records, construct the utility
                # effect, taking into account the expected age of menopause
                for j in range(i, len(events)):
                    age0 = individual.age + events['time'].iloc[j-1]
                    age1 = individual.age + events['time'].iloc[j]

                    # Retrieve relevant utility values
                    u_baseline_pre = self.params[f'utility.rrgs.{current_surg_str}.premenopausal.baseline']
                    u_affected_pre = self.params[f'utility.rrgs.{current_surg_str}.premenopausal']
                    u_baseline_post = self.params[f'utility.rrgs.{current_surg_str}.postmenopausal.baseline']
                    u_affected_post = self.params[f'utility.rrgs.{current_surg_str}.postmenopausal']
                    
                    # Estimate proportion of time period which is premenopausal
                    # Careful to avoid division by zero
                    if age0 < age1:
                        prop_pre = min(max(50.0 - age0, 0.0) / (age1 - age0), 1.0)
                    else:
                        prop_pre = 1.0 if age0 < 50 else 0.0

                    # Construct average utility values
                    u_baseline = prop_pre * u_baseline_pre + (1 - prop_pre) * u_baseline_post
                    u_affected = prop_pre * u_affected_pre + (1 - prop_pre) * u_affected_post

                    # Create BasicSingleConditionUtilityEffect
                    if u_affected < u_baseline:
                        res.iloc[j] = BasicSingleConditionUtilityEffect.create_from_affected_and_unaffected(u_affected, u_baseline, 'Risk-reducing gynaecological surgery')
                    else:
                        res.iloc[j] = None
                
        return res

    def filter_conditions(self, conditions: List[SingleConditionUtilityEffect]) -> List[SingleConditionUtilityEffect]:
        # If someone has had risk-reducing surgery and cancer was discovered,
        # we do not want to count both conditions, so we only include the
        # cancers
        includes_gynae_cancer = any(
            bool(re.search(r'(ovarian|endometrial) cancer', cond.description))
            for cond in conditions
        )
        if not includes_gynae_cancer:
            return conditions
        else:
            return [c for c in conditions if c.description != 'Risk-reducing gynaecological surgery']
    
    def _audc(self, t_l, t_r, dr: float):
        """Evaluate the area under the continuous discount curve
        
        The continuous discount curve is given by :math:`\exp(-r t)` where
        :math:`r` is the continuous discount rate. If the annual discount rate
        is :math:`DR` then :math:`r = \ln(1 + DR)`

        Calculating the area under the continuous discount curve is useful
        when a payoff (i.e. cost or QALYs) is accrued at a constant rate per
        unit time.
        """
        if dr == 0.0:
            return t_r - t_l
        r = numpy.log(1 + dr)
        return (numpy.exp(-r * t_l) - numpy.exp(-r * t_r)) / r

