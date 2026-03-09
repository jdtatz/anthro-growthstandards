from .gamlss_ext import GAMLSSModelByCondition, SimpleBCCGModel
from .gamlss_params import LookupTable
from .who_models import lhfa_female, lhfa_male

__all__ = ["hfa", "hfa_female", "hfa_male", "lfa", "lfa_female", "lfa_male"]

## split lhfa into lfa and hfa

assert isinstance(lhfa_female.loc, LookupTable) and lhfa_female.loc.xp[731] == 731
assert isinstance(lhfa_female.scale, LookupTable) and lhfa_female.scale.xp[731] == 731
lfa_female = SimpleBCCGModel(
    loc=lhfa_female.loc[:731],
    scale=lhfa_female.scale[:731],
    attrs={**lhfa_female.attrs, "name": "length", "long_name": "Recumbent Length"},
    x_attrs=lhfa_female.x_attrs,
)
hfa_female = SimpleBCCGModel(
    loc=lhfa_female.loc[731:],
    scale=lhfa_female.scale[731:],
    attrs={**lhfa_female.attrs, "name": "height", "long_name": "Standing Height"},
    x_attrs=lhfa_female.x_attrs,
)
assert isinstance(lhfa_male.loc, LookupTable) and lhfa_male.loc.xp[731] == 731
assert isinstance(lhfa_male.scale, LookupTable) and lhfa_male.scale.xp[731] == 731
lfa_male = SimpleBCCGModel(
    loc=lhfa_male.loc[:731],
    scale=lhfa_male.scale[:731],
    attrs={**lhfa_male.attrs, "name": "length", "long_name": "Recumbent Length"},
    x_attrs=lhfa_male.x_attrs,
)
hfa_male = SimpleBCCGModel(
    loc=lhfa_male.loc[731:],
    scale=lhfa_male.scale[731:],
    attrs={**lhfa_male.attrs, "name": "height", "long_name": "Standing Height"},
    x_attrs=lhfa_male.x_attrs,
)

lfa = GAMLSSModelByCondition(lfa_female, lfa_male, cond_attrs={"name": "is_female", "long_name": "sex = Female"})
hfa = GAMLSSModelByCondition(hfa_female, hfa_male, cond_attrs={"name": "is_female", "long_name": "sex = Female"})
