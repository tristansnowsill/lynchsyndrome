
import numpy.random

from .sources import CallableParameterSource, ConstantParameterSource, DependentParameterSource, ParameterSource

def params_utilities(rng: numpy.random.Generator) -> ParameterSource:
    return (
        _params_ovarian_utilities(rng) +
        _params_endometrial_utilities(rng) +
        _params_colorectal_utilities(rng) +
        _params_rrgs_utilities(rng)
    )


def _params_ovarian_utilities(rng: numpy.random.Generator) -> ParameterSource:
    def sample(_rng: numpy.random.Generator):
        u_NC = _rng.normal(0.80, 0.03)
        ud_I_II = _rng.gamma(shape=1, scale=0.02)
        ud_III_IV = _rng.gamma(shape=1, scale=0.06)
        return [u_NC, u_NC - ud_I_II, u_NC - ud_I_II - ud_III_IV, u_NC - ud_I_II - ud_III_IV]
    
    return DependentParameterSource(
        rng,
        ['utility.ovarian.nocancer','utility.ovarian.early','utility.ovarian.advanced','utility.ovarian.recurrence'],
        sample
    )

def _params_endometrial_utilities(rng: numpy.random.Generator) -> ParameterSource:
    def sample(_rng: numpy.random.Generator):
        u_NC = 1.0
        ud_early = _rng.gamma(shape=1, scale=0.02)
        ud_adv = _rng.gamma(shape=1, scale=0.11)
        u_early = u_NC - ud_early
        u_adv = u_early - ud_adv
        return [u_NC, u_early, u_adv, u_adv]
    
    return DependentParameterSource(
        rng,
        ['utility.endometrial.nocancer','utility.endometrial.early','utility.endometrial.advanced','utility.endometrial.recurrence'],
        sample
    )

def _params_colorectal_utilities(rng: numpy.random.Generator) -> ParameterSource:
    return CallableParameterSource(
        rng,
        {
            'utility.colorectal.nonmetastatic.year1': lambda _rng: _rng.beta(8.579701,1.282024),
            'utility.colorectal.nonmetastatic.beyond': lambda _rng: _rng.beta(9.2406586,0.8035355),
            'utility.colorectal.metastatic.year1': lambda _rng: _rng.beta(8.061453,3.793625),
            'utility.colorectal.metastatic.beyond': lambda _rng: _rng.beta(11.227179,4.152518)
        }
    )

def _params_rrgs_utilities(rng: numpy.random.Generator) -> ParameterSource:
    return CallableParameterSource(
        rng,
        { 'utility.rrgs.hbso.premenopausal': lambda _rng: 0.9235248 * _rng.beta(94.2, 5.8) }
    ) + ConstantParameterSource({
        'utility.rrgs.hbso.premenopausal.baseline': 0.9235248,
        'utility.rrgs.bso.premenopausal.baseline' : 0.9235248,
        'utility.rrgs.hbs.premenopausal.baseline' : 0.9235248,
        'utility.rrgs.hbs.premenopausal'          : 0.9235248,
        'utility.rrgs.hyst.premenopausal.baseline': 0.9235248,
        'utility.rrgs.hyst.premenopausal'         : 0.9235248,

        'utility.rrgs.hbso.postmenopausal.baseline': 0.9235248,
        'utility.rrgs.hbso.postmenopausal'         : 0.9235248,
        'utility.rrgs.bso.postmenopausal.baseline' : 0.9235248,
        'utility.rrgs.bso.postmenopausal'          : 0.9235248,
        'utility.rrgs.hbs.postmenopausal.baseline' : 0.9235248,
        'utility.rrgs.hbs.postmenopausal'          : 0.9235248,
        'utility.rrgs.hyst.postmenopausal.baseline': 0.9235248,
        'utility.rrgs.hyst.postmenopausal'         : 0.9235248,
    })
