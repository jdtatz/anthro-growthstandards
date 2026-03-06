from fractions import Fraction

import numpy as np

from .gamlss_ext import BCCGModel, GAMLSSModelByCondition, LookupTable, SimpleBCCGModel
from .who_models import lhfa_female, lhfa_male, wfh_female, wfh_male, wfl_female, wfl_male


def _shift(lt: LookupTable, ds: int | Fraction, /):
    return LookupTable(
        start=lt.start + ds,
        stop=lt.stop + ds,
        step=lt.step,
        fp=lt.fp,
    )


def _concat(lhs: LookupTable, rhs: LookupTable, /):
    if lhs.stop + lhs.step != rhs.start:
        raise ValueError("can't concat non-contiguous lookup tables")
    elif lhs.step != rhs.step:
        raise ValueError(f"can't concat lookup tables with different steps ({lhs.step} != {rhs.step})")
    return LookupTable(
        start=lhs.start,
        stop=rhs.stop,
        step=lhs.step,
        fp=np.concat((lhs.fp, rhs.fp)),
    )


## undo the length/height split conversion

assert isinstance(lhfa_female.loc, LookupTable) and lhfa_female.loc.xp[731] == 731
lefa_female = SimpleBCCGModel(
    loc=LookupTable(
        start=lhfa_female.loc.start,
        stop=lhfa_female.loc.stop,
        step=lhfa_female.loc.step,
        # FIXME
        fp=np.concat((lhfa_female.loc.fp[:731], lhfa_female.loc.fp[731:] + 0.7)),
    ),
    scale=lhfa_female.scale,
    attrs={**lhfa_female.attrs, "name": "length", "long_name": "Recumbent Length (Extended)"},
    x_attrs=lhfa_female.x_attrs,
)
hefa_female = SimpleBCCGModel(
    loc=LookupTable(
        start=lhfa_female.loc.start,
        stop=lhfa_female.loc.stop,
        step=lhfa_female.loc.step,
        # FIXME
        fp=np.concat((lhfa_female.loc.fp[:731] - 0.7, lhfa_female.loc.fp[731:])),
    ),
    scale=lhfa_female.scale,
    attrs={**lhfa_female.attrs, "name": "height", "long_name": "Standing Height (Extended)"},
    x_attrs=lhfa_female.x_attrs,
)
assert isinstance(lhfa_male.loc, LookupTable) and lhfa_male.loc.xp[731] == 731
lefa_male = SimpleBCCGModel(
    loc=LookupTable(
        start=lhfa_male.loc.start,
        stop=lhfa_male.loc.stop,
        step=lhfa_male.loc.step,
        # FIXME
        fp=np.concat((lhfa_male.loc.fp[:731], lhfa_male.loc.fp[731:] + 0.7)),
    ),
    scale=lhfa_male.scale,
    attrs={**lhfa_male.attrs, "name": "length", "long_name": "Recumbent Length (Extended)"},
    x_attrs=lhfa_male.x_attrs,
)
hefa_male = SimpleBCCGModel(
    loc=LookupTable(
        start=lhfa_male.loc.start,
        stop=lhfa_male.loc.stop,
        step=lhfa_male.loc.step,
        # FIXME
        fp=np.concat((lhfa_male.loc.fp[:731] - 0.7, lhfa_male.loc.fp[731:])),
    ),
    scale=lhfa_male.scale,
    attrs={**lhfa_male.attrs, "name": "height", "long_name": "Standing Height (Extended)"},
    x_attrs=lhfa_male.x_attrs,
)

lefa = GAMLSSModelByCondition(lefa_female, lefa_male, cond_attrs={"name": "is_female", "long_name": "sex = Female"})
hefa = GAMLSSModelByCondition(hefa_female, hefa_male, cond_attrs={"name": "is_female", "long_name": "sex = Female"})


## undo the length/height split

assert isinstance(wfl_female.mu, LookupTable)
assert isinstance(wfh_female.mu, LookupTable)
assert isinstance(wfl_female.sigma, LookupTable)
assert isinstance(wfh_female.sigma, LookupTable)
assert np.all(wfl_female.mu.fp[-444:] == wfh_female.mu.fp[:444])
assert np.all(wfl_female.sigma.fp[-444:] == wfh_female.sigma.fp[:444])
assert wfl_female.nu == wfh_female.nu
wfle_female = BCCGModel(
    mu=_concat(wfl_female.mu, _shift(wfh_female.mu[444:], Fraction(7, 10))),
    sigma=_concat(wfl_female.sigma, _shift(wfh_female.sigma[444:], Fraction(7, 10))),
    nu=wfl_female.nu,
    attrs=wfl_female.attrs,
    x_attrs=wfl_female.x_attrs,
)
wfhe_female = BCCGModel(
    mu=_concat(_shift(wfl_female.mu, Fraction(-7, 10)), wfh_female.mu[444:]),
    sigma=_concat(_shift(wfl_female.sigma, Fraction(-7, 10)), wfh_female.sigma[444:]),
    nu=wfh_female.nu,
    attrs=wfh_female.attrs,
    x_attrs=wfh_female.x_attrs,
)

assert isinstance(wfl_male.mu, LookupTable)
assert isinstance(wfh_male.mu, LookupTable)
assert isinstance(wfl_male.sigma, LookupTable)
assert isinstance(wfh_male.sigma, LookupTable)
assert np.all(wfl_male.mu.fp[-444:] == wfh_male.mu.fp[:444])
assert np.all(wfl_male.sigma.fp[-444:] == wfh_male.sigma.fp[:444])
assert wfl_male.nu == wfh_male.nu
wfle_male = BCCGModel(
    mu=_concat(wfl_male.mu, _shift(wfh_male.mu[444:], Fraction(7, 10))),
    sigma=_concat(wfl_male.sigma, _shift(wfh_male.sigma[444:], Fraction(7, 10))),
    nu=wfl_male.nu,
    attrs=wfl_male.attrs,
    x_attrs=wfl_male.x_attrs,
)
wfhe_male = BCCGModel(
    mu=_concat(_shift(wfl_male.mu, Fraction(-7, 10)), wfh_male.mu[444:]),
    sigma=_concat(_shift(wfl_male.sigma, Fraction(-7, 10)), wfh_male.sigma[444:]),
    nu=wfh_male.nu,
    attrs=wfh_male.attrs,
    x_attrs=wfh_male.x_attrs,
)

wfle = GAMLSSModelByCondition(wfle_female, wfle_female, cond_attrs={"name": "is_female", "long_name": "sex = Female"})
wfhe = GAMLSSModelByCondition(wfhe_female, wfhe_female, cond_attrs={"name": "is_female", "long_name": "sex = Female"})
