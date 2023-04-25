import logging
from typing import Any, Mapping

import numpy
import pandas

from lynchsyndrome.diagnosis import DiagnosisMetadata
from lynchsyndrome.individual import Individual

from .diagnosis import CancerSite, CancerStage, DiagnosisMetadata, RouteToDiagnosis, is_early
from .recurrence import RecurrenceMetadata
from .reporting import EventMetadata, ReportingEvent
from .risk_reducing_surgery import RiskReducingSurgery, RiskReducingSurgeryMetadata
from .surveillance import SurveillanceMetadata, SurveillanceSite, SurveillanceTechnology


class Costing:
    def __init__(
        self,
        params: Mapping[str, Any]
    ):
        self.params = params
        self.event_costs_callbacks = {
            ReportingEvent.AEH_DIAGNOSIS         : self.event_cost_null,
            ReportingEvent.CANCER_DIAGNOSIS      : self.event_cost_cancer_diagnosis,
            ReportingEvent.CANCER_RECURRENCE     : self.event_cost_cancer_recurrence,
            ReportingEvent.CANCER_SURVEILLANCE   : self.event_cost_cancer_surveillance,
            ReportingEvent.DEATH                 : self.event_cost_null,
            ReportingEvent.ENTER_MODEL           : self.event_cost_null,
            ReportingEvent.REACH_TIME_HORIZON    : self.event_cost_null,
            ReportingEvent.RISK_REDUCING_SURGERY : self.event_cost_risk_reducing_surgery,
            ReportingEvent.START_AEH_MEDICAL     : self.event_cost_start_aeh_medical,
            ReportingEvent.START_CHEMOPROPHYLAXIS: self.event_cost_null,
            ReportingEvent.STOP_AEH_MEDICAL      : self.event_cost_stop_aeh_medical,
            ReportingEvent.STOP_CHEMOPROPHYLAXIS : self.event_cost_null
        }
        self.running_costs_callbacks = [
            self.running_aspirin_costs,
            self.running_colorectal_cancer_costs,
            self.running_endometrial_cancer_costs,
            self.running_ovarian_cancer_costs,
            self.running_hrt_costs
        ]
    
    
    def event_costs(self, events: pandas.DataFrame, individual: Individual) -> pandas.Series:
        """Calculate (discounted) costs of events
        
        Event costs occur at a specific time and are discounted using the
        simple formula

        .. math::
        
            c_d = \\frac{c}{(1+r)^t}

        where :math:`c_d` is the discounted cost for the event, :math:`c` is
        the undiscounted cost for the event, :math:`t` is the time of the
        event (in years), and :math:`r` is the annual discount rate

        """
        undiscounted_costs = events.apply(lambda row: self.individual_event_cost(row.event, row.metadata, individual), axis=1)
        discount_rate = self.params['analysis.discount_rate.cost']
        discount_factors = (1 + discount_rate) ** (-events['time'])
        return discount_factors * undiscounted_costs


    def individual_event_cost(self, event: ReportingEvent, metadata: EventMetadata, individual: Individual) -> float:
        """Calculate the (undiscounted) cost of a single event"""
        logging.debug("individual_event_cost (event=%s) (metadata=%s) (individual=%s)", event, metadata, individual)
        return self.event_costs_callbacks[event](metadata, individual)
    
    def event_cost_cancer_diagnosis(self, metadata: DiagnosisMetadata, _: Individual):
        if metadata.site is CancerSite.COLORECTUM:
            return 0.0
        elif metadata.site is CancerSite.ENDOMETRIUM:
            mapping = {
                CancerStage.STAGE_I: self.params['cost.endometrial_cancer.stage_I'],
                CancerStage.STAGE_II: self.params['cost.endometrial_cancer.stage_II'],
                CancerStage.STAGE_III: self.params['cost.endometrial_cancer.stage_III'],
                CancerStage.STAGE_IV: self.params['cost.endometrial_cancer.stage_IV']
            }
            # If it was detected during risk-reducing surgery we don't want to
            # double-count the cost of surgery, so we remove if applicable
            if metadata.route is RouteToDiagnosis.RISK_REDUCING_SURGERY:
                return mapping[metadata.stage] - self.params['cost.hysterectomy']
            else:
                return mapping[metadata.stage]
        elif metadata.site is CancerSite.OVARIES:
            # If it was detected during risk-reducing surgery we don't want to
            # double-count the cost of surgery, so we remove if applicable
            res = self.params['cost.ovarian_cancer.early'] if is_early(metadata.stage) else self.params['cost.ovarian_cancer.late']
            if metadata.route is RouteToDiagnosis.RISK_REDUCING_SURGERY:
                return res - self.params['cost.hysterectomy_bso']
            else:
                return res
    
    def event_cost_cancer_surveillance(self, metadata: SurveillanceMetadata, _: Individual):
        gynae_sites = (
            SurveillanceSite.FEMALE_REPRODUCTIVE,
            SurveillanceSite.ENDOMETRIUM,
            SurveillanceSite.OVARIES
        )

        # We use set comparison <= (non-strict subset)
        gynae_technologies = { 
            SurveillanceTechnology.CA125,
            SurveillanceTechnology.HYSTEROSCOPY,
            SurveillanceTechnology.TRANSVAGINAL_ULTRASOUND,
            SurveillanceTechnology.UNDIRECTED_ENDOMETRIAL_BIOPSY
        }
        colorectal_technologies = { 
            SurveillanceTechnology.DIAGNOSTIC_COLONOSCOPY,
            SurveillanceTechnology.THERAPEUTIC_COLONOSCOPY,
            SurveillanceTechnology.FLEXIBLE_SIGMOIDOSCOPY
        }
        if metadata.site in gynae_sites and metadata.technology <= gynae_technologies:
            comb1 = set([SurveillanceTechnology.CA125, SurveillanceTechnology.HYSTEROSCOPY, SurveillanceTechnology.TRANSVAGINAL_ULTRASOUND])
            comb2 = set([SurveillanceTechnology.CA125, SurveillanceTechnology.UNDIRECTED_ENDOMETRIAL_BIOPSY, SurveillanceTechnology.TRANSVAGINAL_ULTRASOUND])
            comb3 = set([SurveillanceTechnology.HYSTEROSCOPY, SurveillanceTechnology.TRANSVAGINAL_ULTRASOUND])
            comb4 = set([SurveillanceTechnology.CA125, SurveillanceTechnology.TRANSVAGINAL_ULTRASOUND])
            if metadata.technology == comb1:
                return self.params['cost.gynae_surveillance.hysteroscopy_tvus_ca125']
            elif metadata.technology == comb2:
                return self.params['cost.gynae_surveillance.hysteroscopy_tvus_ca125']
            elif metadata.technology == comb3:
                return self.params['cost.gynae_surveillance.hysteroscopy_tvus']
            elif metadata.technology == comb4:
                return self.params['cost.gynae_surveillance.tvus_ca125']
        elif metadata.site is SurveillanceSite.COLORECTUM and metadata.technology <= colorectal_technologies:
            if metadata.technology == {SurveillanceTechnology.DIAGNOSTIC_COLONOSCOPY}:
                return self.params['cost.colonoscopy.diagnostic']
            elif metadata.technology == {SurveillanceTechnology.THERAPEUTIC_COLONOSCOPY}:
                return self.params['cost.colonoscopy.therapeutic']
        else:
            raise ValueError("could not determine cost for cancer surveillance (metadata=%s)", metadata)
        
    def event_cost_cancer_recurrence(self, metadata: RecurrenceMetadata, _: Individual):
        if metadata.site is CancerSite.COLORECTUM:
            raise NotImplementedError("recurrence event for colorectal cancer has been logged but recurrence is not modelled")
        elif metadata.site is CancerSite.ENDOMETRIUM:
            return self.params['cost.endometrial_cancer.recurrence'] if is_early(metadata.stage_at_diagnosis) else 0.0
        elif metadata.site is CancerSite.OVARIES:
            return self.params['cost.ovarian_cancer.recurrence'] if is_early(metadata.stage_at_diagnosis) else 0.0

    def event_cost_null(self, metadata: EventMetadata, _: Individual):
        return 0.0
    
    def event_cost_risk_reducing_surgery(self, metadata: RiskReducingSurgeryMetadata, _: Individual):
        mapping = {
            RiskReducingSurgery.HBSO: self.params['cost.hysterectomy_bso'],
            RiskReducingSurgery.HBS: self.params['cost.hysterectomy_bs'],
            RiskReducingSurgery.HYSTERECTOMY: self.params['cost.hysterectomy'],
            RiskReducingSurgery.BSO: self.params['cost.bso'],
            RiskReducingSurgery.BILATERAL_OOPHORECTOMY: self.params['cost.bo'],
        }
        return mapping[metadata.surgery]
    
    def event_cost_start_aeh_medical(self, metadata: EventMetadata, _: Individual):
        return self.params['cost.aeh_management.medical.start']
    
    def event_cost_stop_aeh_medical(self, metadata: EventMetadata, _: Individual):
        return self.params['cost.aeh_management.medical.stop']
    

    def running_costs(self, events: pandas.DataFrame, individual: Individual) -> pandas.Series:
        """Calculate running costs
        
        NOTE! Callbacks will generally assume that `events` refers to the
        event trace for a *single* ``lynchsyndrome.Individual`` and are likely
        to return invalid results if passed event traces for more than one.
        """
        return sum(cb(events, individual) for cb in self.running_costs_callbacks)


    def running_aspirin_costs(self, events: pandas.DataFrame, _: Individual) -> pandas.Series:
        cost_aspirin = self.params['cost.aspirin_prophylaxis.annual']
        return self._running_generic_costs(
            events,
            cost_aspirin,
            {ReportingEvent.START_CHEMOPROPHYLAXIS},
            {ReportingEvent.STOP_CHEMOPROPHYLAXIS,ReportingEvent.DEATH,ReportingEvent.REACH_TIME_HORIZON}
        )
    

    def running_endometrial_cancer_costs(self, events: pandas.DataFrame, _: Individual) -> pandas.Series:
        return pandas.Series(numpy.zeros(len(events)), index=events.index)
    

    def running_colorectal_cancer_costs(self, events: pandas.DataFrame, individual: Individual) -> pandas.Series:
        # Determine if there are any colorectal cancer diagnoses...
        all_diagnoses = events[events['event'].eq(ReportingEvent.CANCER_DIAGNOSIS)]
        
        if len(all_diagnoses) == 0:
            return pandas.Series(numpy.zeros(len(events)), index=events.index)
        
        crc_diagnoses = all_diagnoses[all_diagnoses['metadata'].apply(lambda md: md.site is CancerSite.COLORECTUM)]

        if len(crc_diagnoses) == 0:
            return pandas.Series(numpy.zeros(len(events)), index=events.index)
        
        # NOTE - Currently assuming there is only one CRC diagnosis because
        #   the model does not include metachronous CRC (this is subject to
        #   change)

        t_diag = crc_diagnoses['time'].iloc[0]
        t_end = events['time'].iloc[-1]

        dr = self.params['analysis.discount_rate.cost']
        res = pandas.Series(index=events.index, dtype=numpy.float64)
        t_l = events['time'].shift(periods=1, fill_value=0.0)
        t_l.name = 't_l'

        for record in pandas.concat([events, t_l], axis=1).itertuples():
            logging.debug("record=%s", record)
            timepoints = [ t for t in numpy.arange(t_diag, t_end, step=1.0) if t >= record.t_l and t < record.time ]
            logging.debug("timepoints=%s", timepoints)
            years = [ min(int(numpy.rint(t - t_diag)), 8) for t in timepoints ]
            logging.debug("years=%s", years)
            age = 'older' if individual.age + t_diag >= 65.0 else 'younger'
            stage = 'early' if is_early(crc_diagnoses['metadata'].iloc[0].stage) else 'late'
            costs = [ self.params[f'cost.colorectal_cancer.{age}.{stage}'][year] / ((1 + dr) ** time) for time, year in zip(timepoints, years) ]
            logging.debug("costs=%s", costs)
            res[record.Index] = sum(costs)

        return res
    
    def running_ovarian_cancer_costs(self, events: pandas.DataFrame, _: Individual) -> pandas.Series:
        try:
            record_dx = next(
                i
                for i in range(len(events))
                if events['event'][i] is ReportingEvent.CANCER_DIAGNOSIS and
                    events['metadata'][i].site is CancerSite.OVARIES
            )
            cost_short_followup = self.params['cost.ovarian_cancer.followup_1_3']
            cost_long_followup = self.params['cost.ovarian_cancer.followup_after_3']
            t_r = events['time']
            t_l = t_r.shift(periods=1, fill_value=0.0)
            t_dx = t_r.iloc[record_dx]
            t_end_short_fup = t_dx + 3
            # Will assume that there are no events after reaching the time
            # horizon or dying...
            res = pandas.Series(numpy.zeros(len(events)), index=events.index)
            for i in range(record_dx+1, len(events)):
                if t_r.iloc[i] < t_end_short_fup:
                    # Record entirely within short follow-up period
                    res.iloc[i] = cost_short_followup * self._audc(t_l.iloc[i], t_r.iloc[i])
                else:
                    # Record ends within long follow-up period
                    if t_l.iloc[i] > t_end_short_fup:
                        # Record begins after short follow-up period
                        res.iloc[i] = cost_long_followup * self._audc(t_l.iloc[i], t_r.iloc[i])
                    else:
                        # Record spans short and long follow-up periods
                        res.iloc[i] = (
                            cost_short_followup * self._audc(t_l.iloc[i], t_end_short_fup) +
                            cost_long_followup * self._audc(t_end_short_fup, t_r.iloc[i])
                        )
            
            return res

        except StopIteration:
            return pandas.Series(numpy.zeros(len(events)), index=events.index)
        
    def running_hrt_costs(self, events: pandas.DataFrame, individual: Individual) -> pandas.Series:
        res = pandas.Series(0.0, dtype=float, index=events.index)
        for i in range(len(events)):
            ev, md = events['event'].iloc[i], events['metadata'].iloc[i]
            if ev is ReportingEvent.RISK_REDUCING_SURGERY and isinstance(md, RiskReducingSurgeryMetadata):
                if md.surgery in [RiskReducingSurgery.BSO, RiskReducingSurgery.HBSO]:
                    age_rrgs = events['time'].iloc[i] + individual.age
                    t_lower_dose = events['time'].iloc[i] + 2.0
                    t_stop = events['time'].iloc[i] + max(50.0 - age_rrgs, 4.0)
                    c_high = self.params['cost.hrt.highdose']
                    c_low = self.params['cost.hrt.lowdose']
                    for j in range(i+1, len(events)):
                        t_l, t_r = events['time'].iloc[j-1], events['time'].iloc[j]
                        if t_stop <= t_l:
                            #   |<====|---->|
                            # start lower  stop
                            #                  |<-->|
                            #                 t_l  t_r
                            break
                        elif t_lower_dose >= t_r:
                            #   |<============|---->|
                            # start         lower  stop
                            #        |<-->|
                            #       t_l  t_r
                            res.iloc[j] = c_high * self._audc(t_l, t_r)
                        elif t_lower_dose < t_l and t_stop > t_r:
                            #   |<====|----------->|
                            # start lower         stop
                            #             |<-->|
                            #            t_l  t_r
                            res.iloc[j] = c_low * self._audc(t_l, t_r)
                        else:
                            # The record may include some time with high dose,
                            # some time with low dose, and/or some time with
                            # no dose
                            tmp = 0.0
                            if t_l < t_lower_dose:
                                tmp += c_high * self._audc(t_l, min(t_r, t_lower_dose))
                            tmp += c_low * self._audc(max(t_l, t_lower_dose), min(t_r, t_stop))
                            res.iloc[j] = tmp
        return res
    
    def _running_generic_costs(self, events: pandas.DataFrame, cost: float, start_events, stop_events) -> pandas.Series:
        t_r = events['time']
        t_l = t_r.shift(periods=1, fill_value=0)
        incurring_cost = pandas.Series(numpy.full(len(t_l), False, bool), index=events.index)
        starting_cost = events['event'].isin(start_events)
        stopping_cost = events['event'].isin(stop_events)
        for i in range(len(starting_cost)-1):
            if starting_cost[i]:
                incurring_cost[(i+1):] = True
            if stopping_cost[i]:
                incurring_cost[(i+1):] = False
        return cost * incurring_cost * self._audc(t_l, t_r)


    def _audc(self, t_l, t_r):
        """Evaluate the area under the continuous discount curve
        
        The continuous discount curve is given by :math:`\exp(-r t)` where
        :math:`r` is the continuous discount rate. If the annual discount rate
        is :math:`DR` then :math:`r = \ln(1 + DR)`

        Calculating the area under the continuous discount curve is useful
        when a payoff (i.e. cost or QALYs) is accrued at a constant rate per
        unit time.
        """
        discount_rate = self.params['analysis.discount_rate.cost']
        r = numpy.log(1 + discount_rate)
        return (numpy.exp(-r * t_l) - numpy.exp(-r * t_r)) / r
