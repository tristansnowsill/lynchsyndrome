# -*- coding: utf-8 -*-
"""Utility value calculations for the Lynch syndrome model"""

from __future__ import annotations
from abc import ABC, abstractmethod
import logging
from math import fsum, prod
from typing import Optional

import numpy

from .sex import Sex


class BaselineUtility(ABC):
    """Abstract base class for baseline utility.
    
    Baseline utility encapsulates the effects of conditions which are not
    explicitly modelled but are prevalent in the population of interest. It is
    important because even in the absence of conditions which are explicitly
    modelled it is unlikely that the population has perfect health.

    You can create a concrete subclass of this abstract base class, but we
    also supply the ``HSEBaselineUtility`` concrete class based on the Health
    Survey for England.

    You can evaluate the utility for a particular individual either by using
    the ``utility()`` method, or just calling the instance as the ``__call__``
    method defers to the ``utility()`` method.
    """
    
    @abstractmethod
    def utility(self, sex: Sex, age: float) -> float:
        raise NotImplementedError()
    
    def __call__(self, sex: Sex, age: float) -> float:
        return self.utility(sex, age)


class HSEBaselineUtility(BaselineUtility):
    """Baseline utility as estimated from the Health Survey for England by
    Ara and Brazier (2010)

    Ara R, Brazier J. Populating an economic model with health state utility
    values: moving towards better practice. Value in Health 2010; 13(5):509-518.
    doi: 10.1111/j.1524-4733.2010.00700.x
    """

    u_cons = 0.9508566
    """Intercept for utility equation"""

    u_male = 0.0212126
    """Utility difference for males vs females"""

    u_age  = -0.0002587
    """Linear coefficient for age in utility equation"""

    u_age2 = -0.0000332
    """Quadratic coefficient for age in utility equation"""
    
    def utility(self, sex: Sex, age: float) -> float:
        u_int = self.u_cons + self.u_male if sex is Sex.MALE else self.u_cons
        return u_int + age * (self.u_age + age * self.u_age2)


class SingleConditionUtilityEffect(ABC):
    """Abstract base class for a single condition utility effect.
    
    You can either create a concrete subclass of this abstract base class, or
    create an instance of ``BasicSingleConditionUtilityEffect``

    NOTE - Generally we are interested in the direct effect of a condition on
    utility, i.e., the difference between those with the condition and those
    without the condition after adjusting for other factors (e.g., age, sex,
    comorbidities).
    """

    @property
    @abstractmethod
    def description(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def utility_decrement(self) -> float:
        """The effect of the single condition on utility assuming a
        additive effect.

        NOTE - This should be a positive number for a condition which leads to
        a worsening of health-related quality of life. E.g., if utility
        without the condition is 0.9 and utility with the condition is 0.8,
        then ``utility_decrement()`` should return 0.1.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def utility_scale(self) -> float:
        """The effect of the single condition on utility assuming a
        multiplicative effect.
        """
        raise NotImplementedError()

    def utility_affected(self) -> float:
        """Return the implied utility for someone affected by the condition"""
        s = self.utility_scale()
        return self.utility_decrement() * s / (1.0 - s)
    
    def utility_unaffected(self) -> float:
        """Return the implied utility for someone unaffected by the condition"""
        return self.utility_decrement() / (1.0 - self.utility_scale())


class BasicSingleConditionUtilityEffect(SingleConditionUtilityEffect):
    def __init__(
        self,
        utility_decrement: float,
        utility_scale: float,
        description: Optional[str] = None
    ):
        if utility_decrement < 0.0:
            logging.warning("""utility_decrement (%f) is negative, but should be positive
                for a condition which worsens HRQoL""" % (utility_decrement))
        if utility_scale > 1.0:
            logging.warning("""utility_scale (%f) should be less than or equal to 1 for a
                condition which worsens HRQoL""" % (utility_scale))
        if utility_scale < 0.0:
            raise ValueError("utility_scale (%f) must be non-negative" % (utility_scale))
        self.__utility_decrement = utility_decrement
        self.__utility_scale = utility_scale
        self.__description = description
    
    @property
    def description(self) -> str:
        return self.__description
    
    def utility_decrement(self) -> float:
        return self.__utility_decrement
    
    def utility_scale(self) -> float:
        return self.__utility_scale
    
    @staticmethod
    def create_from_affected_and_unaffected(
        affected: float,
        unaffected: float,
        description: Optional[str] = None
    ) -> BasicSingleConditionUtilityEffect:
        if affected > unaffected:
            logging.warning("""affected (%f) should be less than unaffected (%f) for
                condition which worsens HRQoL""" % (affected, unaffected))
        if affected < 0:
            logging.info("affected state is worse than death, setting utility scale to 0")
            return BasicSingleConditionUtilityEffect(unaffected - affected, 0.0, description)
        return BasicSingleConditionUtilityEffect(
            unaffected - affected,
            affected / unaffected,
            description)


class OverallUtility(ABC):

    @classmethod
    @abstractmethod
    def utility(cls, sex: Sex, age: float, baseline_utility: BaselineUtility, *conditions: SingleConditionUtilityEffect) -> float:
        raise NotImplementedError()


class LinearIndexOverallUtilityBasu2009(OverallUtility):
    """Calculate the utility using the linear index method first described by
    Basu et al. (2009)
    
    Basu A, Dale W, Elstein A, Meltzer D. A linear index for predicting joint
    health-states utilities from single health-states utilities. Health
    Economics 2009;18(4):403-419

    The method as described calculates the utility by estimating the utility
    loss:

    .. math::

        E[l(JS)] = \\alpha_0 + \\alpha_1 \\max\\{l(SS1),l(SS2)\\} + \\alpha_2 \\min\\{l(SS1),l(SS2)\\} + \\alpha_3 l(SS1) l(SS2)
    
    Where :math:`l(JS)` is the utility loss for the joint state, and
    :math:`l(SS1)` and :math:`l(SS2)` are the utility losses for the two
    individual states.
    
    The method assumes that the utility in the absence of the single states is
    1, but in our case we have a baseline utility. Therefore if we have a
    single condition, we treat the baseline utility as a second condition and
    perform the calculation as follows:

    .. math::

        E[u(JS)] &= 1 - E[l(JS)] \\\\
        E[l(JS)] &= \\alpha_0 + \\alpha_1 \\max\\{l(NC),l(SS)\\} + \\alpha_2 \\min\\{l(NC),l(SS)\\} + \\alpha_3 l(NC) l(SS)

    If there are more than two conditions, we implement the method using the
    two conditions with the greatest utility decrement as follows:

    .. math:: E[u(JS)] = u(NC) - E[l(JS)]
    """

    a0 = 0.046
    a1 = 0.721
    a2 = 0.331
    a3 = -0.176

    @classmethod
    def utility(cls, sex: Sex, age: float, baseline_utility: BaselineUtility, *conditions: SingleConditionUtilityEffect) -> float:
        u_NC = baseline_utility(sex, age)
        num_conditions = len(conditions)
        u_baseline = None
        l_SS1 = None
        l_SS2 = None
        if num_conditions == 0:
            return u_NC
        elif num_conditions == 1:
            u_baseline = 1.0
            l_SS1 = 1.0 - u_NC
            l_SS2 = conditions[0].utility_decrement()
        elif num_conditions == 2:
            u_baseline = u_NC
            l_SS1 = conditions[0].utility_decrement()
            l_SS2 = conditions[0].utility_decrement()
        else:
            u_baseline = u_NC
            all_decrements = numpy.array([c.utility_decrement() for c in conditions])
            all_decrements.partition(-2)
            l_SS1 = all_decrements[-2]
            l_SS2 = all_decrements[-1]

        l_JS = cls.a0 + cls.a1 * max(l_SS1, l_SS2) + cls.a2 * min(l_SS1, l_SS2) + cls.a3 * l_SS1 * l_SS2
        return u_baseline - l_JS


class LinearIndexOverallUtilityThompson2019(OverallUtility):
    """Calculate the utility using the linear index method first described by
    Basu et al. (2008) and extended to four arguments and estimated from UK
    primary care population by Thompson et al. (2019)
    
    Basu A, Dale W, Elstein A, Meltzer D. A linear index for predicting joint
    health-states utilities from single health-states utilities. Health
    Economics 2009;18(4):403-419

    Thompson AJ, Sutton M, Payne K. Estimating joint health condition utility
    values. Value in Health 2019;22(4):482-490
    """

    a0 = { 2: 0.423629, 3: -0.04523, 4: -0.25723 }
    a1 = { 2: -0.44007, 3: 0.031313, 4: 0.342493 }
    a2 = { 2: -0.32722, 3: -0.20939, 4: -0.26215 }
    a3 = { 2: 0.541800, 3: 1.043063, 4: 1.260055 }

    @classmethod
    def utility(cls, sex: Sex, age: float, baseline_utility: BaselineUtility, *conditions: SingleConditionUtilityEffect) -> float:
        u_NC = baseline_utility(sex, age)
        num_conditions = len(conditions)
        if num_conditions == 0:
            return u_NC
        elif num_conditions == 1:
            # We don't know whether to use utility decrement or utility scale
            # so we calculate both and use the average...
            u_add = u_NC - conditions[0].utility_decrement()
            u_mul = u_NC * conditions[0].utility_scale()
            return 0.5 * (u_add + u_mul)
        elif num_conditions <= 4:
            x1 = max(c.utility_decrement() for c in conditions)
            x2 = min(c.utility_decrement() for c in conditions)
            x3 = u_NC * prod(c.utility_scale() for c in conditions)
            res = (
                cls.a0[num_conditions] + cls.a1[num_conditions] * x1 +
                cls.a2[num_conditions] * x2 + cls.a3[num_conditions] * x3
            )
            if res > u_NC:
                logging.info(
                    "predicted utility is greater than baseline utility (u_NC=%.4f, conditions=%s); replacing with baseline utility",
                    u_NC, [(c.utility_decrement(), c.utility_scale()) for c in conditions]
                )
                return u_NC
            else:
                return res
        else:
            raise ValueError("cannot process more than four joint health conditions")


class AdditiveOverallUtility(OverallUtility):
    """Calculate the utility by summing the absolute utility losses"""
    
    @classmethod
    def utility(cls, sex: Sex, age: float, baseline_utility: BaselineUtility, *conditions: SingleConditionUtilityEffect) -> float:
        u_baseline = baseline_utility(sex, age)
        if len(conditions) == 0:
            return u_baseline
        l_total = fsum([cond.utility_decrement() for cond in conditions])
        return u_baseline - l_total


class MultiplicativeOverallUtility(OverallUtility):
    """Calculate the utility by multiplying utility scale factors"""

    @classmethod
    def utility(cls, sex: Sex, age: float, baseline_utility: BaselineUtility, *conditions: SingleConditionUtilityEffect) -> float:
        u_baseline = baseline_utility(sex, age)
        if len(conditions) == 0:
            return u_baseline
        sf_total = prod([cond.utility_scale() for cond in conditions])
        return u_baseline * sf_total


class MinimumOverallUtility(OverallUtility):
    """Calculate the utility as the minimum utility obtained by applying each
    of the single condition utilities (tries as utility decrements and utility
    scales).
    """
    
    @classmethod
    def utility(cls, sex: Sex, age: float, baseline_utility: BaselineUtility, *conditions: SingleConditionUtilityEffect) -> float:
        u_baseline = baseline_utility(sex, age)
        if len(conditions) == 0:
            return u_baseline
        u_add = [u_baseline - cond.utility_decrement() for cond in conditions]
        u_mul = [u_baseline * cond.utility_scale() for cond in conditions]
        return min(u_add + u_mul)

