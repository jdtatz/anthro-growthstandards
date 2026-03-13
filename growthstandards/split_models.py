from .gamlss_ext import BCCGModel, GAMLSSModelByCondition, SimpleBCCGModel
from .gamlss_params import LookupTable
from .who_models import bfa_female, bfa_male, lhfa_female, lhfa_male

__all__ = [
    "bhfa",
    "bhfa_female",
    "bhfa_male",
    "blfa",
    "blfa_female",
    "blfa_male",
    "hfa",
    "hfa_female",
    "hfa_male",
    "lfa",
    "lfa_female",
    "lfa_male",
]

## split lhfa into lfa and hfa

assert isinstance(lhfa_female, SimpleBCCGModel)
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
assert isinstance(lhfa_male, SimpleBCCGModel)
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

## split bfa into blfa and bhfa

assert isinstance(bfa_female, BCCGModel)
assert isinstance(bfa_female.mu, LookupTable) and bfa_female.mu.xp[731] == 731
assert isinstance(bfa_female.sigma, LookupTable) and bfa_female.sigma.xp[731] == 731
assert isinstance(bfa_female.nu, LookupTable) and bfa_female.nu.xp[731] == 731
# NOTE: `bhfa_female.nu` was fit as a constant
assert (bfa_female.nu.fp[731] == bfa_female.nu.fp[731:]).all()
blfa_female = BCCGModel(
    mu=bfa_female.mu[:731],
    sigma=bfa_female.sigma[:731],
    nu=bfa_female.nu[:731],
    attrs={**bfa_female.attrs, "long_name": "Body Mass Index (Recumbent Length)"},
    x_attrs=bfa_female.x_attrs,
)
bhfa_female = BCCGModel(
    mu=bfa_female.mu[731:],
    sigma=bfa_female.sigma[731:],
    nu=bfa_female.nu.fp[731],
    attrs={**bfa_female.attrs, "long_name": "Body Mass Index (Standing Height)"},
    x_attrs=bfa_female.x_attrs,
)
assert isinstance(bfa_male, BCCGModel)
assert isinstance(bfa_male.mu, LookupTable) and bfa_male.mu.xp[731] == 731
assert isinstance(bfa_male.sigma, LookupTable) and bfa_male.sigma.xp[731] == 731
assert isinstance(bfa_male.nu, LookupTable) and bfa_male.nu.xp[731] == 731
blfa_male = BCCGModel(
    mu=bfa_male.mu[:731],
    sigma=bfa_male.sigma[:731],
    nu=bfa_male.nu[:731],
    attrs={**bfa_male.attrs, "long_name": "Body Mass Index (Recumbent Length)"},
    x_attrs=bfa_male.x_attrs,
)
bhfa_male = BCCGModel(
    mu=bfa_male.mu[731:],
    sigma=bfa_male.sigma[731:],
    nu=bfa_male.nu[731:],
    attrs={**bfa_male.attrs, "long_name": "Body Mass Index (Standing Height)"},
    x_attrs=bfa_male.x_attrs,
)

blfa = GAMLSSModelByCondition(blfa_female, blfa_male, cond_attrs={"name": "is_female", "long_name": "sex = Female"})
bhfa = GAMLSSModelByCondition(bhfa_female, bhfa_male, cond_attrs={"name": "is_female", "long_name": "sex = Female"})
