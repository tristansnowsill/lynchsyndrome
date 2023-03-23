# -*- coding: utf-8 -*-
"""QALY calculations for the Lynch syndrome model"""

from __future__ import annotations
from abc import ABC, abstractmethod
from math import exp, log
from typing import Callable, Optional

from scipy.integrate import quad

class QALY(ABC):
    @abstractmethod
    def discounted(self, discount_rate: float) -> float:
        raise NotImplementedError()

    @abstractmethod
    def undiscounted(self) -> float:
        raise NotImplementedError()

    def __add__(self, other: QALY) -> AdditiveQALY:
        return AdditiveQALY(self, other)


class ConstantUtilityQALY(QALY):
    def __init__(self, t0: float, t1: float, u: float):
        super(ConstantUtilityQALY, self).__init__()
        if t1 <= t0:
            raise ValueError("t1 (%f) must be greater than t0 (%f)" % (t1, t0))
        if u > 1.0:
            raise ValueError("u (%f) must be less than or equal to 1.0" % (u))
        self.t0 = t0
        self.t1 = t1
        self.u = u
    
    def discounted(self, discount_rate: float) -> float:
        if discount_rate <= 0.0:
            raise ValueError("discount_rate (%f) must be greater than 0" % (discount_rate))
        discount_rate_c = log(discount_rate + 1.0)
        return self.u / discount_rate_c * (exp(-discount_rate_c * self.t0) - exp(-discount_rate_c * self.t1))
    
    def undiscounted(self) -> float:
        return self.u * (self.t1 - self.t0)


class LinearUtilityQALY(QALY):
    """Defines a period where the utility value is a linear function."""

    def __init__(self, t0: float, t1: float, u0: float, u1: float):
        """Initialise a `LinearUtilityQALY` instance.
        
        :param t0: the start time for the period
        :param t1: the end time for the period
        :param u0: the utility at the start of the period
        :param u1: the utility at the end of the period
        """
        super(LinearUtilityQALY, self).__init__()
        if t1 <= t0:
            raise ValueError("t1 (%f) must be greater than t0 (%f)" % (t1, t0))
        if u0 > 1.0 or u1 > 1.0:
            raise ValueError("utilities (u0=%f, u1=%f) must be less than or equal to 1.0" % (u0, u1))
        self.t0 = t0
        self.t1 = t1
        self.u0 = u0
        self.u1 = u1
    
    def discounted(self, discount_rate: float) -> float:
        r = log(discount_rate + 1.0)
        rinv = 1.0 / r
        m = (self.u1 - self.u0) / (self.t1 - self.t0)
        c = self.u0 - m * self.t0
        return rinv * ((c+m*(self.t0+rinv)) * exp(-r*self.t0) - (c+m*(self.t1+rinv)) * exp(-r*self.t1))

    def undiscounted(self) -> float:
        return 0.5 * (self.u0 + self.u1) * (self.t1 - self.t0)
    
    @staticmethod
    def from_intercept_and_slope(
        t0: float, t1: float, intercept: float, slope: float
    ) -> LinearUtilityQALY:
        u0 = intercept + slope * t0
        u1 = intercept + slope * t1
        return LinearUtilityQALY(t0, t1, u0, u1)


class QuadraticUtilityQALY(QALY):
    """Defines a period where the utility value is a quadratic function."""

    def __init__(
        self,
        t0: float,
        t1: float,
        a0: float,
        a1: float,
        a2: float,
        t_off: Optional[float] = 0.0
    ):
        """Initialise a `QuadraticUtilityQALY` period.

        The utility is given by:

        .. math:: U(t) = a0 + a1 (t-t_off) + a2 (t-t_off)^2
        
        :param t0: the start time for the period
        :param t1: the end time for the period
        :param a0: the constant coefficient
        :param a1: the linear coefficient, i.e., for (t - t_off)
        :param a2: the quadratic coefficient, i.e., for (t - t_off)^2
        :param t_off: offset for quadratic formula (optional; default = 0)
        """
        super(QuadraticUtilityQALY, self).__init__()
        if t1 <= t0:
            raise ValueError("t1 (%f) must be greater than t0 (%f)" % (t1, t0))
        self.t0 = t0
        self.t1 = t1
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.t_off = t_off
        
    def discounted(self, discount_rate: float) -> float:
        r = log(discount_rate + 1.0)
        scale = exp(-r * self.t_off)
        tau0 = self.t0 - self.t_off
        tau1 = self.t1 - self.t_off
        helper = LinearUtilityQALY.from_intercept_and_slope(tau0, tau1, self.a1, 2*self.a2)
        c0 = (self.a0 + tau0 * (self.a1 + tau0 * self.a2)) * exp(-r * tau0)
        c1 = (self.a0 + tau1 * (self.a1 + tau1 * self.a2)) * exp(-r * tau1)
        quad = c0 - c1
        return scale / r * (quad + helper.discounted(discount_rate))
    
    def undiscounted(self) -> float:
        tau0 = self.t0 - self.t_off
        tau1 = self.t1 - self.t_off
        c0 = tau0 * (self.a0 + tau0 * (self.a1 / 2.0 + tau0 * self.a2 / 3.0))
        c1 = tau1 * (self.a0 + tau1 * (self.a1 / 2.0 + tau1 * self.a2 / 3.0))
        return c1 - c0


class UtilityFunctionQALY(QALY):
    def __init__(
        self,
        t0: float,
        t1: float,
        u: Callable[[float], float]
    ):
        super(UtilityFunctionQALY, self).__init__()
        if t1 <= t0:
            raise ValueError("t1 (%f) must be greater than t0 (%f)" % (t1, t0))
        self.t0 = t0
        self.t1 = t1
        self.u = u
    

    def discounted(self, discount_rate: float) -> float:
        r = log(discount_rate + 1.0)
        (val, _) = quad(lambda t: self.u(t) * exp(-r * t), self.t0, self.t1)
        return val


    def undiscounted(self) -> float:
        (val, _) = quad(self.u, self.t0, self.t1)
        return val


class AdditiveQALY(QALY):
    """Calculate total QALYs as the sum of sequential QALY periods.

    It is generally not intended that users will explicitly build
    ``AdditiveQALY`` instances, but will instead use the overloaded
    ``__add__`` method of the abstract base class ``QALY``.
    
    Note: This will not check that the periods are suitable, e.g., whether
    they overlap or leave gaps.

    Note: This will unpack any existing ``AdditiveQALY`` instances to limit
    the amount of function chaining. If you make any subsequent changes to a
    constituent ``AdditiveQALY`` instance this will NOT be reflected in this
    instance.
    """
    
    def __init__(self, *qalys):
        super(AdditiveQALY, self).__init__()
        self.qalys = []
        for q in qalys:
            self.qalys += q.qalys if isinstance(q, AdditiveQALY) else [q]
    
    def discounted(self, discount_rate: float) -> float:
        return sum([q.discounted(discount_rate) for q in self.qalys])

    def undiscounted(self) -> float:
        return sum([q.undiscounted() for q in self.qalys])
