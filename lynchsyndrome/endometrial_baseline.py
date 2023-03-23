
import logging
from enum import Enum, auto
from typing import Any, Callable, List, Mapping

import numpy
import pandas
import scipy
import scipy.sparse
from more_itertools import unique_everseen
from scipy.integrate import solve_ivp


class BaselineLesionState(Enum):
    NONE                   = auto()

    ASYMPTOMATIC_AEH       = auto()
    ASYMPTOMATIC_STAGE_I   = auto()
    ASYMPTOMATIC_STAGE_II  = auto()
    ASYMPTOMATIC_STAGE_III = auto()
    ASYMPTOMATIC_STAGE_IV  = auto()

    SYMPTOMATIC_AEH        = auto()
    SYMPTOMATIC_STAGE_I    = auto()
    SYMPTOMATIC_STAGE_II   = auto()
    SYMPTOMATIC_STAGE_III  = auto()
    SYMPTOMATIC_STAGE_IV   = auto()

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            return {
                '0': BaselineLesionState.NONE,
                'A': BaselineLesionState.ASYMPTOMATIC_AEH,
                'B': BaselineLesionState.ASYMPTOMATIC_STAGE_I,
                'C': BaselineLesionState.ASYMPTOMATIC_STAGE_II,
                'D': BaselineLesionState.ASYMPTOMATIC_STAGE_III,
                'E': BaselineLesionState.ASYMPTOMATIC_STAGE_IV,
                'F': BaselineLesionState.SYMPTOMATIC_AEH,
                'G': BaselineLesionState.SYMPTOMATIC_STAGE_I,
                'H': BaselineLesionState.SYMPTOMATIC_STAGE_II,
                'I': BaselineLesionState.SYMPTOMATIC_STAGE_III,
                'J': BaselineLesionState.SYMPTOMATIC_STAGE_IV
            }[value]
        else:
            return super()._missing_(value)


class EndometrialBaselineCalculation:

    _state_names = None
    _state_index = dict()
    _ages = numpy.linspace(0.0, 100.0, 101)

    def __init__(self, params: Mapping[str, Any], aeh_incidence: Callable[[float], float]):
        self._params = params
        self._aeh_incidence = aeh_incidence

        if self._state_names is None:
            self._prepare_mappings()

        logging.debug("EndometrialBaselineCalculation: calculating constant components of Jacobian matrix")
        self._jac_constants = (
            self._jac_aeh_resolve() +
            self._jac_aeh_symptomatic() +
            self._jac_aeh_diagnosed() +
            self._jac_aeh_progression() +
            self._jac_ec_symptomatic() +
            self._jac_ec_diagnosis() +
            self._jac_ec_progression()
        )

        y0 = numpy.zeros(286)
        y0[0] = 1.0
        logging.debug("EndometrialBaselineCalculation: solving initial value problem")
        self._solution = solve_ivp(self._f, (self._ages.min(), self._ages.max()), y0, method='BDF', t_eval=self._ages, jac=self._jac)
        if self._solution.status != 0:
            raise RuntimeError()
    
    def get_dataframe(self) -> pandas.DataFrame:
        # The returned data frame will have the following shape
        #
        # Time  Lesion3: BaselineLesionState.NONE BaselineLesionState.NONE
        #       Lesion2: BaselineLesionState.NONE BaselineLesionState.NONE
        #       Lesion1: BaselineLesionState.NONE BaselineLesionState.ASYMPTOMATIC_AEH
        #  0.0           1.000                    0.000
        #  1.0
        #  2.0
        #  ...
        # 99.0
        #  100
        time_index = pandas.Index(self._ages)
        column_multiindex = pandas.MultiIndex.from_tuples(
            (BaselineLesionState(s[0]), BaselineLesionState(s[1]), BaselineLesionState(s[2]))
            for s in self._state_names
        )
        data = self._solution.y.T

        if data.min() < -1e-5:
            logging.warning("Some values are more negative than acceptable (%.3e)", data.min())
        
        data = data.clip(min=0, max=None)

        rowsums = data.sum(axis=1)
        res = pandas.DataFrame(data / rowsums[:, numpy.newaxis], time_index, column_multiindex)

        return res

    def _jac(self, t, y) -> scipy.sparse.coo_matrix:
        return self._jac_aeh_incidence(t, y) + self._jac_constants

    def _jac_aeh_incidence(self, t, y) -> scipy.sparse.coo_matrix:
        # Just focus on AEH incidence

        # AEH incidence happens when we go from 0 to A in any of the state
        # string positions, and whenever this happens it affects two entries
        # in the Jacobian matrix

        row = list()
        col = list()
        data = list()

        mu = self._aeh_incidence(t)

        state_from: List[str] = [s for s in self._state_names if '0' in s]
        state_to:   List[str] = [s[::-1].replace('0', 'A', 1)[::-1] for s in state_from]

        for sf, st in zip(state_from, state_to):
            row.append(self._state_index[sf])
            col.append(self._state_index[sf])
            data.append(-mu)

            row.append(self._state_index[st])
            col.append(self._state_index[sf])
            data.append(mu)
        
        return scipy.sparse.coo_matrix((data, (row, col)), shape=(286, 286))
    
    def _jac_aeh_resolve(self) -> scipy.sparse.coo_matrix:
        # Just focus on AEH resolving

        # AEH spontaneously resolve when we go from A or F to 0 in any of the
        # state string positions, and whenever this happens it affects two
        # entries in the Jacobian matrix

        # For added fun (!) the rate at which AEH resolve depends on the number
        # of AEH...

        row = list()
        col = list()
        data = list()

        mu = self._params['endometrial.regression']

        asymp_state_from: List[str] = [s for s in self._state_names if 'A' in s]
        asymp_muddled_to: List[str] = [s[::-1].replace('A', '0', 1)[::-1] for s in asymp_state_from]
        asymp_state_to:   List[str] = [''.join(sorted(list(s))) for s in asymp_muddled_to]

        symp_state_from: List[str] = [s for s in self._state_names if 'F' in s]
        symp_muddled_to: List[str] = [s[::-1].replace('F', '0', 1)[::-1] for s in symp_state_from]
        symp_state_to:   List[str] = [''.join(sorted(list(s))) for s in symp_muddled_to]

        # Consider the following instructive example
        #
        # From the state 0AF there are two possible transitions corresponding
        # to AEH resolution: 0AF -> 00F and 0AF -> 00A
        #
        # The first of these will be found in asymp_state_from and
        # asymp_state_to while the second will be found in symp_state_from and
        # symp_state_to.
        #
        # We only want to have one entry for the diagonal entry, and the entry
        # should reflect the multiplicity of transitions
        #
        # For off diagonal entries we may also have multiplicity!
        #
        # Consider an even more instructive example:
        #
        # From the state 0FF there is only one transition corresponding to AEH
        # resolution: 0FF -> 00F, but this in truth represents two possible
        # events, since either of the lesions could resolve. In this case we
        # have multiplicity 2

        for sf in unique_everseen(asymp_state_from + symp_state_from):
            mul = sf.count('A') + sf.count('F')

            row.append(self._state_index[sf])
            col.append(self._state_index[sf])
            data.append(-mu * mul)

        for sf, st in unique_everseen(zip(asymp_state_from, asymp_state_to)):
            mul = sf.count('A') # We only count asymptomatic transitions

            row.append(self._state_index[st])
            col.append(self._state_index[sf])
            data.append(mu * mul)

        for sf, st in unique_everseen(zip(symp_state_from, symp_state_to)):
            mul = sf.count('F') # We only count symptomatic transitions

            row.append(self._state_index[st])
            col.append(self._state_index[sf])
            data.append(mu * mul)
        
        res = scipy.sparse.coo_matrix((data, (row, col)), shape=(286, 286))

        # Sanity check - the total sum should be zero
        if numpy.fabs(res.sum()) > 1e-12:
            logging.warning('Spontaneously resolving rates not matching (%.3e)', res.sum())
        
        return res
    
    def _jac_aeh_symptomatic(self) -> scipy.sparse.coo_matrix:
        # Just focus on AEH becoming symptomatic

        # AEH become symptomatic when we go from A to F in any of the state
        # string positions, and whenever this happens it affects two entries
        # in the Jacobian matrix

        row = list()
        col = list()
        data = list()

        mu = self._params['endometrial.symptomatic'][0]

        state_from: List[str] = [s for s in self._state_names if 'A' in s]
        muddled_to: List[str] = [s[::-1].replace('A', 'F', 1)[::-1] for s in state_from]
        state_to:   List[str] = [''.join(sorted(list(s))) for s in muddled_to]

        for sf, st in zip(state_from, state_to):
            mul = sf.count('A')

            row.append(self._state_index[sf])
            col.append(self._state_index[sf])
            data.append(-mu * mul)

            row.append(self._state_index[st])
            col.append(self._state_index[sf])
            data.append(mu * mul)
        
        return scipy.sparse.coo_matrix((data, (row, col)), shape=(286, 286))

    def _jac_aeh_diagnosed(self) -> scipy.sparse.coo_matrix:
        # Just focus on symptomatic AEH becoming diagnosed

        # AEH are diagnosed when they go from having F in any of the state
        # string positions to just disappearing

        row = list()
        col = list()
        data = list()

        mu = self._params['endometrial.diagnosis'][0]

        state_from: List[str] = [s for s in self._state_names if 'F' in s]

        for sf in state_from:
            mul = sf.count('F')

            row.append(self._state_index[sf])
            col.append(self._state_index[sf])
            data.append(-mu * mul)
        
        return scipy.sparse.coo_matrix((data, (row, col)), shape=(286, 286))
    
    def _jac_aeh_progression(self) -> scipy.sparse.coo_matrix:
        # Just focus on AEH progressing

        # AEH progress when we go from A to B or from F to G in any of the
        # state string positions, and whenever this happens it affects two
        # entries in the Jacobian matrix

        row = list()
        col = list()
        data = list()

        mu = self._params['endometrial.progression'][0]

        asymp_state_from: List[str] = [s for s in self._state_names if 'A' in s]
        asymp_muddled_to: List[str] = [s[::-1].replace('A', 'B', 1)[::-1] for s in asymp_state_from]
        asymp_state_to:   List[str] = [''.join(sorted(list(s))) for s in asymp_muddled_to]

        symp_state_from: List[str] = [s for s in self._state_names if 'F' in s]
        symp_muddled_to: List[str] = [s[::-1].replace('F', 'G', 1)[::-1] for s in symp_state_from]
        symp_state_to:   List[str] = [''.join(sorted(list(s))) for s in symp_muddled_to]

        for sf in unique_everseen(asymp_state_from + symp_state_from):
            mul = sf.count('A') + sf.count('F')

            row.append(self._state_index[sf])
            col.append(self._state_index[sf])
            data.append(-mu * mul)

        for sf, st in unique_everseen(zip(asymp_state_from, asymp_state_to)):
            mul = sf.count('A') # We only count asymptomatic transitions

            row.append(self._state_index[st])
            col.append(self._state_index[sf])
            data.append(mu * mul)

        for sf, st in unique_everseen(zip(symp_state_from, symp_state_to)):
            mul = sf.count('F') # We only count symptomatic transitions

            row.append(self._state_index[st])
            col.append(self._state_index[sf])
            data.append(mu * mul)
        
        return scipy.sparse.coo_matrix((data, (row, col)), shape=(286, 286))

    def _jac_ec_symptomatic(self) -> scipy.sparse.coo_matrix:
        return (
            self._jac_ec_x_symptomatic(1) +
            self._jac_ec_x_symptomatic(2) +
            self._jac_ec_x_symptomatic(3) +
            self._jac_ec_x_symptomatic(4)
        )

    def _jac_ec_x_symptomatic(self, stage: int) -> scipy.sparse.coo_matrix:
        # Just focus on Stage x EC becoming symptomatic

        # Stage x EC become symptomatic when we go from ['B','C','D','E'][x-1]
        # to ['G','H','I','J'][x-1] in any of the state string positions, and
        # whenever this happens it affects two entries in the Jacobian matrix

        row = list()
        col = list()
        data = list()

        mu = self._params['endometrial.symptomatic'][stage]

        from_symbol: str = list('BCDE')[stage-1]
        to_symbol:   str = list('GHIJ')[stage-1]

        state_from: List[str] = [s for s in self._state_names if from_symbol in s]
        muddled_to: List[str] = [s[::-1].replace(from_symbol, to_symbol, 1)[::-1] for s in state_from]
        state_to:   List[str] = [''.join(sorted(list(s))) for s in muddled_to]

        for sf, st in unique_everseen(zip(state_from, state_to)):
            mul = sf.count(from_symbol)

            row.append(self._state_index[sf])
            col.append(self._state_index[sf])
            data.append(-mu * mul)

            row.append(self._state_index[st])
            col.append(self._state_index[sf])
            data.append(mu * mul)
        
        return scipy.sparse.coo_matrix((data, (row, col)), shape=(286, 286))

    def _jac_ec_diagnosis(self) -> scipy.sparse.coo_matrix:
        return (
            self._jac_ec_x_diagnosis(1) +
            self._jac_ec_x_diagnosis(2) +
            self._jac_ec_x_diagnosis(3) +
            self._jac_ec_x_diagnosis(4)
        )

    def _jac_ec_x_diagnosis(self, stage: int) -> scipy.sparse.coo_matrix:
        # Just focus on Stage x EC being diagnosed

        # Stage x EC are diagnosed when we go from ['G','H','I','J'][x-1] to
        # nothing

        row = list()
        col = list()
        data = list()

        mu = self._params['endometrial.diagnosis'][stage]

        from_symbol: str = list('GHIJ')[stage-1]

        state_from: List[str] = [s for s in self._state_names if from_symbol in s]

        for sf in state_from:
            mul = sf.count(from_symbol)

            row.append(self._state_index[sf])
            col.append(self._state_index[sf])
            data.append(-mu * mul)
        
        return scipy.sparse.coo_matrix((data, (row, col)), shape=(286, 286))

    def _jac_ec_progression(self) -> scipy.sparse.coo_matrix:
        return (
            self._jac_asymp_ec_x_progression(1) +
            self._jac_asymp_ec_x_progression(2) +
            self._jac_asymp_ec_x_progression(3) +
            self._jac_symp_ec_x_progression(1) +
            self._jac_symp_ec_x_progression(2) +
            self._jac_symp_ec_x_progression(3)
        )

    def _jac_asymp_ec_x_progression(self, stage: int) -> scipy.sparse.coo_matrix:
        # Just focus on asymptomatic Stage x EC progressing

        # Stage x EC progress when we go from ['B','C','D','E'][x-1]
        # to ['B','C','D','E'][x] in any of the state string positions, and
        # whenever this happens it affects two entries in the Jacobian matrix

        row = list()
        col = list()
        data = list()

        mu = self._params['endometrial.progression'][stage]

        from_symbol: str = list('BCDE')[stage-1]
        to_symbol:   str = list('BCDE')[stage]

        state_from: List[str] = [s for s in self._state_names if from_symbol in s]
        muddled_to: List[str] = [s[::-1].replace(from_symbol, to_symbol, 1)[::-1] for s in state_from]
        state_to:   List[str] = [''.join(sorted(list(s))) for s in muddled_to]

        for sf, st in unique_everseen(zip(state_from, state_to)):
            mul = sf.count(from_symbol)

            row.append(self._state_index[sf])
            col.append(self._state_index[sf])
            data.append(-mu * mul)

            row.append(self._state_index[st])
            col.append(self._state_index[sf])
            data.append(mu * mul)
        
        return scipy.sparse.coo_matrix((data, (row, col)), shape=(286, 286))

    def _jac_symp_ec_x_progression(self, stage: int) -> scipy.sparse.coo_matrix:
        # Just focus on symptomatic Stage x EC progressing

        # Stage x EC progress when we go from ['G','H','I','J'][x-1]
        # to ['G','H','I','J'][x] in any of the state string positions, and
        # whenever this happens it affects two entries in the Jacobian matrix

        row = list()
        col = list()
        data = list()

        mu = self._params['endometrial.progression'][stage]

        from_symbol: str = list('GHIJ')[stage-1]
        to_symbol:   str = list('GHIJ')[stage]

        state_from: List[str] = [s for s in self._state_names if from_symbol in s]
        muddled_to: List[str] = [s[::-1].replace(from_symbol, to_symbol, 1)[::-1] for s in state_from]
        state_to:   List[str] = [''.join(sorted(list(s))) for s in muddled_to]

        for sf, st in unique_everseen(zip(state_from, state_to)):
            mul = sf.count(from_symbol)

            row.append(self._state_index[sf])
            col.append(self._state_index[sf])
            data.append(-mu * mul)

            row.append(self._state_index[st])
            col.append(self._state_index[sf])
            data.append(mu * mul)
        
        return scipy.sparse.coo_matrix((data, (row, col)), shape=(286, 286))


    def _f(self, t, y) -> numpy.ndarray:
        return self._jac(t, y) @ y
    
    @classmethod
    def _prepare_mappings(cls):
        symbols = list('0ABCDEFGHIJ')
        i = 0
        combos = list()
        for c in range(11):
            for b in range(c+1):
                for a in range(b+1):
                    state_str = symbols[a] + symbols[b] + symbols[c]
                    cls._state_index[state_str] = i
                    combos.append(state_str)
                    i += 1
        cls._state_names = numpy.array(combos)


# if __name__ == "__main__":
#     params = {
#         'endometrial.regression' : 0.116894,
#         'endometrial.symptomatic': [0.128265,0.218129,0.542683,1.150071,2.981015],
#         'endometrial.diagnosis'  : [0.600479,1.632337,1.557966,2.165137,1.669742],
#         'endometrial.progression': [0.134738,0.074398,0.986091,0.509629]
#     }
#     ebc = EndometrialBaselineCalculation(params=params, aeh_incidence=lambda x: 0.1)
#     print(ebc.get_dataframe())
