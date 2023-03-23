from __future__ import annotations

import logging
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Mapping, Optional

import numpy
import simpy 

from lynchsyndrome.death import CauseOfDeath
from lynchsyndrome.diagnosis import CancerSite, CancerStage, RouteToDiagnosis
from lynchsyndrome.genetics import ConstitutionalMMR
from lynchsyndrome.sex import Sex

if TYPE_CHECKING:
    from lynchsyndrome.individual import Individual


class BowelState(IntEnum):
    UNDEFINED             = -1
    NORMAL                = 0
    LOW_RISK_MSS_ADENOMA  = 1
    HIGH_RISK_MSS_ADENOMA = 2
    FLAT_MSI_ADENOMA      = 3
    LOW_RISK_MSI_ADENOMA  = 4
    HIGH_RISK_MSI_ADENOMA = 5
    PRECLIN_STAGE_I       = 6
    PRECLIN_STAGE_II      = 7
    PRECLIN_STAGE_III     = 8
    PRECLIN_STAGE_IV      = 9
    CLIN_STAGE_I          = 10
    CLIN_STAGE_II         = 11
    CLIN_STAGE_III        = 12
    CLIN_STAGE_IV         = 13


class Bowel:

    csd_cache = dict()
    tm_cache = dict()
    
    def __init__(
        self,
        env: simpy.Environment,
        rng: numpy.random.Generator,
        params: Mapping[str, Any],
        individual: Individual,
        initial_state: Optional[BowelState] = BowelState.UNDEFINED
    ):
        self.env = env
        self.rng = rng
        self.params = params
        self.individual = individual
        self.state = initial_state

    def sample_baseline_state(self, init_rng: numpy.random.Generator):
        if self.state is BowelState.UNDEFINED:
            state_probs = self._get_conditional_state_distribution()
            state_index = init_rng.choice(14, p=state_probs)
            self.state = BowelState(state_index)
    
    def signal_mortality(self, cause: CauseOfDeath):
        pass
    
    def signal_reach_time_horizon(self):
        pass
    
    def run(self):
        while True:
            age = int(numpy.rint(self.individual.current_age))
            e_cycle = self.env.timeout(1.0)
            yield e_cycle | self.individual.died | self.individual.reach_time_horizon
            if e_cycle.processed:
                tm = self._trans_mat(age)
                tp = tm[self.state,:]
                next_index = self.rng.choice(14, p=tp)
                if next_index >= BowelState.CLIN_STAGE_I and self.state < BowelState.CLIN_STAGE_I:
                    yield self.env.process(self.individual.record_colorectal_cancer_diagnosis(
                        RouteToDiagnosis.SYMPTOMATIC_PRESENTATION,
                        CancerStage(BowelState(next_index))
                    ))
                    self.state = BowelState(next_index)
                    self.env.process(self.run_crc_survival())
                else:
                    self.state = BowelState(next_index)
            else:
                break
    
    def run_crc_survival(self):
        # Intercept
        b = self.params['colorectal.survival.sporadic.rate.cons']

        # Age
        if self.individual.current_age >= 75:
            b += self.params['colorectal.survival.sporadic.rate.aged_over75']
        elif self.individual.current_age < 45:
            b += self.params['colorectal.survival.sporadic.rate.aged_under45']
        elif self.individual.current_age < 55:
            b += self.params['colorectal.survival.sporadic.rate.aged_45_54']
        elif self.individual.current_age < 65:
            b += self.params['colorectal.survival.sporadic.rate.aged_55_64']
        
        # Stage
        if self.state is BowelState.CLIN_STAGE_II:
            b += self.params['colorectal.survival.sporadic.rate.stageII']
        elif self.state is BowelState.CLIN_STAGE_III:
            b += self.params['colorectal.survival.sporadic.rate.stageIII']
        elif self.state is BowelState.CLIN_STAGE_IV:
            b += self.params['colorectal.survival.sporadic.rate.stageIV']
        
        # Sex
        if self.individual.sex is Sex.FEMALE:
            b += self.params['colorectal.survival.sporadic.rate.female']
        
        # Genotype
        if self.individual.genotype is not ConstitutionalMMR.WILD_TYPE:
            b += numpy.log(self.params['colorectal.survival.lynch.hazardratio'])
        
        logging.debug("[%.2f, %s] Bowel.run_crc_survival log(Rate)=%.4f", self.env.now, self.individual, b)
        surv_rate = numpy.exp(b)
        logging.debug("[%.2f, %s] Bowel.run_crc_survival Rate=%.6f", self.env.now, self.individual, surv_rate)

        # Frailty
        theta = self.params['colorectal.survival.sporadic.frailty']
        logging.debug("[%.2f, %s] Bowel.run_crc_survival Var(Frailty)=%.4f", self.env.now, self.individual, theta)
        frailty = self.rng.gamma(shape=1/theta, scale=theta)
        logging.debug("[%.2f, %s] Bowel.run_crc_survival Frailty=%.4f", self.env.now, self.individual, frailty)
        surv_rate = surv_rate * frailty
        logging.debug("[%.2f, %s] Bowel.run_crc_survival Rate=%.6f", self.env.now, self.individual, surv_rate)

        # Time to event
        t_crc_mortality = self.rng.exponential(scale=1/surv_rate)
        e_crc_mortality = self.env.timeout(t_crc_mortality)
        yield e_crc_mortality | self.individual.died | self.individual.reach_time_horizon
        if e_crc_mortality.processed:
            logging.info("[%.2f, %s] colorectal cancer: death from colorectal cancer", self.env.now, self.individual)
            yield self.env.process(self.individual.record_death(CauseOfDeath.COLORECTAL_CANCER))


    def _trans_mat(self, age: int):
        age = min(age, 99)
        key = (id(self.params), self.individual.genotype, self.individual.sex, age)
        if key in Bowel.tm_cache:
            return Bowel.tm_cache[key]
        
        sex = 'male' if self.individual.sex is Sex.MALE else 'female'
        gt = {
            ConstitutionalMMR.PATH_MLH1: 'MLH1',
            ConstitutionalMMR.PATH_MSH2: 'MSH2',
            ConstitutionalMMR.PATH_MSH6: 'MSH6',
            ConstitutionalMMR.PATH_PMS2: 'PMS2',
            ConstitutionalMMR.WILD_TYPE: 'genpop'
        }[self.individual.genotype]
        mmr3 = {
            ConstitutionalMMR.PATH_MLH1,
            ConstitutionalMMR.PATH_MSH2,
            ConstitutionalMMR.PATH_MSH6
        }
        mmr4 = {
            ConstitutionalMMR.PATH_MLH1,
            ConstitutionalMMR.PATH_MSH2,
            ConstitutionalMMR.PATH_MSH6,
            ConstitutionalMMR.PATH_PMS2
        }

        _mu = self.params[f'colorectal.norm_lr.{sex}']
        _theta_cons = self.params[f'colorectal.theta_cons.{gt}'] if self.individual.genotype in mmr3 else -numpy.inf
        _theta_age = self.params[f'colorectal.theta_age.{gt}'] if self.individual.genotype in mmr3 else 0.0
        _beta_max = self.params[f'colorectal.norm_crc.{sex}']
        _alpha = self.params[f'colorectal.lr_hr.{sex}']
        _eta = self.params[f'colorectal.eta.{gt}'] if self.individual.genotype in mmr4 else 0.0
        _lambda0 = self.params[f'colorectal.hr_crc.{sex}']
        _rho0 = self.params['colorectal.rho0']
        _rho1 = self.params['colorectal.rho1']
        _rho2 = self.params['colorectal.rho2']
        _phi = self.params[f'colorectal.phi.{gt}'] if self.individual.genotype in mmr3 else 0.0
        _psi = self.params[f'colorectal.psi.{gt}'] if self.individual.genotype in mmr3 else 0.0
        _kappa = self.params['colorectal.kappa']
        _nu = self.params['colorectal.nu']
        _crc_pres = self.params['colorectal.presentation']
        _crc_prog = (1.0 - numpy.exp(-numpy.array(self.params['colorectal.progression.lynch']))) if self.individual.genotype in mmr4 else self.params['colorectal.progression.genpop']

        trans_mat = numpy.zeros((14, 14))
        # Off-diagonal entries
        trans_mat[BowelState.NORMAL,BowelState.LOW_RISK_MSS_ADENOMA] = _mu[age]
        trans_mat[BowelState.NORMAL,BowelState.FLAT_MSI_ADENOMA] = numpy.exp(_theta_cons + _theta_age * age)
        _beta = _beta_max * numpy.clip((age - 15)/85, a_min=0, a_max=1) 
        trans_mat[BowelState.NORMAL,BowelState.PRECLIN_STAGE_I] = _beta
        trans_mat[BowelState.LOW_RISK_MSS_ADENOMA,BowelState.HIGH_RISK_MSS_ADENOMA] = _alpha[age]
        trans_mat[BowelState.LOW_RISK_MSS_ADENOMA,BowelState.LOW_RISK_MSI_ADENOMA] = _eta
        trans_mat[BowelState.LOW_RISK_MSS_ADENOMA,BowelState.PRECLIN_STAGE_I] = _beta
        trans_mat[BowelState.HIGH_RISK_MSS_ADENOMA,BowelState.HIGH_RISK_MSI_ADENOMA] = _eta
        trans_mat[BowelState.HIGH_RISK_MSS_ADENOMA,BowelState.PRECLIN_STAGE_I] = _beta + _lambda0[age]
        _rho = _rho0 / (1 + numpy.exp(-(age - _rho1) / _rho2))
        trans_mat[BowelState.FLAT_MSI_ADENOMA,BowelState.NORMAL] = _rho
        trans_mat[BowelState.FLAT_MSI_ADENOMA,BowelState.HIGH_RISK_MSI_ADENOMA] = _phi
        trans_mat[BowelState.FLAT_MSI_ADENOMA,BowelState.PRECLIN_STAGE_I] = _beta + _psi
        trans_mat[BowelState.LOW_RISK_MSI_ADENOMA,BowelState.NORMAL] = _rho
        trans_mat[BowelState.LOW_RISK_MSI_ADENOMA,BowelState.HIGH_RISK_MSI_ADENOMA] = _kappa
        trans_mat[BowelState.LOW_RISK_MSI_ADENOMA,BowelState.PRECLIN_STAGE_I] = _beta
        trans_mat[BowelState.HIGH_RISK_MSI_ADENOMA,BowelState.NORMAL] = _rho
        trans_mat[BowelState.HIGH_RISK_MSI_ADENOMA,BowelState.PRECLIN_STAGE_I] = _beta + _nu
        trans_mat[BowelState.PRECLIN_STAGE_I,BowelState.PRECLIN_STAGE_II] = _crc_prog[0]
        trans_mat[BowelState.PRECLIN_STAGE_I,BowelState.CLIN_STAGE_I] = _crc_pres[0]
        trans_mat[BowelState.PRECLIN_STAGE_II,BowelState.PRECLIN_STAGE_III] = _crc_prog[1]
        trans_mat[BowelState.PRECLIN_STAGE_II,BowelState.CLIN_STAGE_II] = _crc_pres[1]
        trans_mat[BowelState.PRECLIN_STAGE_III,BowelState.PRECLIN_STAGE_IV] = _crc_prog[2]
        trans_mat[BowelState.PRECLIN_STAGE_III,BowelState.CLIN_STAGE_III] = _crc_pres[2]
        trans_mat[BowelState.PRECLIN_STAGE_IV,BowelState.CLIN_STAGE_IV] = _crc_pres[3]
        
        # Diagonal entries
        numpy.fill_diagonal(trans_mat, 1.0 - numpy.sum(trans_mat, axis=1))

        # If any diagonal entries are negative, set them to zero and rescale
        # the corresponding row
        for i in range(14):
            if trans_mat[i,i] < 0:
                logging.info("sum of probabilities of outgoing transitions exceed one, rescaling")
                trans_mat[i,i] = 0
                s = numpy.sum(trans_mat[i,:])
                trans_mat[i,:] = trans_mat[i,:] / s

        Bowel.tm_cache[key] = trans_mat
        return trans_mat

    def _get_conditional_state_distribution(self):
        age = numpy.rint(self.individual.age)
        # Check the cache
        key = (id(self.params), self.individual.genotype, self.individual.sex, age)
        if key in Bowel.csd_cache:
            return Bowel.csd_cache[key]

        y = numpy.zeros(14)
        y[0] = 1.0

        for i in range(int(age)):
            # Evolve state
            y = numpy.matmul(y, self._trans_mat(i))
        
        # Condition on no CRC diagnosis
        y[10:14] = 0.0
        y = y / numpy.sum(y)
        Bowel.csd_cache[key] = y
        return y
        