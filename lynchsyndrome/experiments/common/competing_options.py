# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Set, Tuple

import numpy.random
import simpy

if TYPE_CHECKING:
    from lynchsyndrome.individual import Individual


class Intervention(ABC):
    
    @abstractmethod
    def run(
        self,
        env: simpy.Environment,
        rng: numpy.random.Generator,
        params: Mapping[str, Any],
        individual: Individual
    ):
        raise NotImplementedError()


class Pathway:
    def __init__(
        self,
        elements: Dict[PathwayPoint, Set[Intervention]],
        description: Optional[str] = None
    ):
        self._elements = elements # type: Dict[PathwayPoint, Set[Intervention]]
        if description is not None:
            self._description = description
        else:
            self._description = repr(self._elements)

    @property
    def description(self):
        return self._description

    @property
    def elements(self):
        return self._elements

    def __repr__(self) -> str:
        return repr(self._elements)


class Comparison:
    def __init__(
        self,
        x: Pathway,
        y: Pathway,
        description: Optional[str] = None
    ):
        self._x = x
        self._y = y
        self._description = description
    
    def difference(self) -> Dict[PathwayPoint, Tuple[Set[Intervention]]]:
        """Gives a representation of the differences between different
        :py:class:`Pathway` objects
        
        :return: a mapping from all :py:class:`PathwayPoint` where the
            :py:class:`Pathway` differ to a tuple of :py:class:`Intervention`
            sets, the first element of which is the set of
            :py:class:`Intervention` in ``x`` but not ``y``, and the second
            element of which is the set of the set of :py:class:`Intervention`
            in ``y`` but not ``x``
        """
        x = self._x.elements
        y = self._y.elements
        pp_x = set(x.keys())
        pp_y = set(y.keys())

        pp_only_x = pp_x.difference(pp_y)
        pp_only_y = pp_y.difference(pp_x)
        pp_both = pp_x.intersection(pp_y)

        res1 = dict()
        for pp in pp_only_x:
            res1[pp] = x[pp]

        res2 = dict()
        for pp in pp_only_y:
            res2[pp] = y[pp]

        for pp in pp_both:
            int_x = x[pp]
            int_y = y[pp]
            diff_x = int_x.difference(int_y)
            diff_y = int_y.difference(int_x)
            if len(diff_x) > 0:
                res1[pp] = diff_x
            if len(diff_y) > 0:
                res2[pp] = diff_y

        return (res1, res2)
        


class PathwayPoint(Enum):
    """The various points in pathways where interventions may be implemented

    This class enumerates the various points in pathways where we may seek to
    intervene to improve the health of people with Lynch syndrome.

    Current provision likely includes diagnosis of Lynch syndrome in newly
    diagnosed colorectal and endometrial cancer patients
    (:py:const:`PathwayPoint.CASE_FINDING`), colonoscopic surveillance
    (:py:const:`PathwayPoint.COLORECTAL_SURVEILLANCE`) and risk-reducing
    hysterectomy and bilateral salpingo-oophorectomy
    (:py:const:`PathwayPoint.GYNAECOLOGICAL_RISK_REDUCING_SURGERY`)
    """

    CASE_FINDING = auto()
    """Strategies to identify people with Lynch syndrome
    
    This could include:

    * diagnosis of Lynch syndrome in newly diagnosed cancer patients
    * targeted screening for Lynch syndrome in unaffected patients
    * cascade predictive testing in families
    """

    MEDICAL_RISK_REDUCTION = auto()
    """Medical strategies to reduce the risk of morbidity and mortality
    
    This could include:

    * chemoprophylaxis with aspirin
    * (not yet proven) vaccination against frameshift peptides

    Note - surveillance is specified elsewhere
    """

    COLORECTAL_SURVEILLANCE = auto()
    """Surveillance strategies for colorectal cancer
    
    This could include:

    * colonoscopy (potentially with different intervals)
    * faecal immunochemical testing
    * capsule endoscopy
    """

    COLORECTAL_RISK_REDUCING_SURGERY = auto()
    """Risk-reducing surgery for colorectal cancer
    
    Note - this refers to surgery to prevent colorectal cancer, which is not
    standard practice in Lynch syndrome (but would be recommended in familial
    adenomatous polyposis)
    """

    COLORECTAL_CANCER_TREATMENT = auto()
    """Strategies for treating colorectal cancer
    
    This could include:

    * extended colectomy for those with Lynch syndrome
    * immunotherapy
    """

    GYNAECOLOGICAL_SURVEILLANCE = auto()
    """Surveillance strategies for gynaecological cancer"""

    GYNECOLOGICAL_SURVEILLANCE = GYNAECOLOGICAL_SURVEILLANCE
    """American English synonym"""

    GYNAECOLOGICAL_RISK_REDUCING_SURGERY = auto()
    """Risk-reducing surgery for gynaecological cancer"""

    GYNECOLOGICAL_RISK_REDUCING_SURGERY = GYNAECOLOGICAL_RISK_REDUCING_SURGERY
    """American English synonym"""

    ENDOMETRIAL_CANCER_TREATMENT = auto()
    """Strategies for treating endometrial cancer
    
    This could include:

    * removal of fallopian tubes and ovaries at same time as hysterectomy
    """

    OVARIAN_CANCER_TREATMENT = auto()
    """Strategies for treating ovarian cancer"""
