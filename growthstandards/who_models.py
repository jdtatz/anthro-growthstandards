import importlib.resources
from fractions import Fraction

import numpy as np

from .gamlss_ext import BCCGModel, GAMLSSModelByCondition, SimpleBCCGModel
from .gamlss_params import LookupTable

__all__ = [
    "acfa",
    "acfa_female",
    "acfa_male",
    "bfa",
    "bfa_female",
    "bfa_male",
    "hcfa",
    "hcfa_female",
    "hcfa_male",
    "lhfa",
    "lhfa_female",
    "lhfa_male",
    "ssfa",
    "ssfa_female",
    "ssfa_male",
    "tsfa",
    "tsfa_female",
    "tsfa_male",
    "wfa",
    "wfa_female",
    "wfa_male",
    "wfh",
    "wfh_female",
    "wfh_male",
    "wfl",
    "wfl_female",
    "wfl_male",
]

_traversable = importlib.resources.files(__package__)
with _traversable.joinpath("who_model_params.npz").open("rb") as _f, np.load(_f) as npz:
    _npz = dict(npz.items())


def _load(name: str) -> np.ndarray:
    return _npz[name]


acfa_female = BCCGModel(
    mu=LookupTable(start=91, stop=1856, step=1, fp=_load("acfa_female_mu")),
    sigma=LookupTable(start=91, stop=1856, step=1, fp=_load("acfa_female_sigma")),
    nu=-0.1733,
    attrs={"name": "arm_c", "long_name": "Arm Circumference", "units": "cm"},
    x_attrs={"name": "age", "long_name": "Age", "units": "days"},
)
acfa_male = BCCGModel(
    mu=LookupTable(start=91, stop=1856, step=1, fp=_load("acfa_male_mu")),
    sigma=LookupTable(start=91, stop=1856, step=1, fp=_load("acfa_male_sigma")),
    nu=LookupTable(start=91, stop=1856, step=1, fp=_load("acfa_male_nu")),
    attrs={"name": "arm_c", "long_name": "Arm Circumference", "units": "cm"},
    x_attrs={"name": "age", "long_name": "Age", "units": "days"},
)
acfa = GAMLSSModelByCondition(
    acfa_female,
    acfa_male,
    cond_attrs={"name": "is_female", "long_name": "sex = Female"},
)
bfa_female = BCCGModel(
    mu=LookupTable(start=0, stop=1856, step=1, fp=_load("bfa_female_mu")),
    sigma=LookupTable(start=0, stop=1856, step=1, fp=_load("bfa_female_sigma")),
    nu=LookupTable(start=0, stop=1856, step=1, fp=_load("bfa_female_nu")),
    attrs={"name": "bmi", "long_name": "Body Mass Index", "units": "kg/m^2"},
    x_attrs={"name": "age", "long_name": "Age", "units": "days"},
)
bfa_male = BCCGModel(
    mu=LookupTable(start=0, stop=1856, step=1, fp=_load("bfa_male_mu")),
    sigma=LookupTable(start=0, stop=1856, step=1, fp=_load("bfa_male_sigma")),
    nu=LookupTable(start=0, stop=1856, step=1, fp=_load("bfa_male_nu")),
    attrs={"name": "bmi", "long_name": "Body Mass Index", "units": "kg/m^2"},
    x_attrs={"name": "age", "long_name": "Age", "units": "days"},
)
bfa = GAMLSSModelByCondition(
    bfa_female,
    bfa_male,
    cond_attrs={"name": "is_female", "long_name": "sex = Female"},
)
hcfa_female = SimpleBCCGModel(
    loc=LookupTable(start=0, stop=1856, step=1, fp=_load("hcfa_female_loc")),
    scale=LookupTable(start=0, stop=1856, step=1, fp=_load("hcfa_female_scale")),
    attrs={"name": "head_c", "long_name": "Head Circumference", "units": "cm"},
    x_attrs={"name": "age", "long_name": "Age", "units": "days"},
)
hcfa_male = SimpleBCCGModel(
    loc=LookupTable(start=0, stop=1856, step=1, fp=_load("hcfa_male_loc")),
    scale=LookupTable(start=0, stop=1856, step=1, fp=_load("hcfa_male_scale")),
    attrs={"name": "head_c", "long_name": "Head Circumference", "units": "cm"},
    x_attrs={"name": "age", "long_name": "Age", "units": "days"},
)
hcfa = GAMLSSModelByCondition(
    hcfa_female,
    hcfa_male,
    cond_attrs={"name": "is_female", "long_name": "sex = Female"},
)
lhfa_female = SimpleBCCGModel(
    loc=LookupTable(start=0, stop=1856, step=1, fp=_load("lhfa_female_loc")),
    scale=LookupTable(start=0, stop=1856, step=1, fp=_load("lhfa_female_scale")),
    attrs={"name": "len_hi", "long_name": "Recumbent Length/Standing Height", "units": "cm"},
    x_attrs={"name": "age", "long_name": "Age", "units": "days"},
)
lhfa_male = SimpleBCCGModel(
    loc=LookupTable(start=0, stop=1856, step=1, fp=_load("lhfa_male_loc")),
    scale=LookupTable(start=0, stop=1856, step=1, fp=_load("lhfa_male_scale")),
    attrs={"name": "len_hi", "long_name": "Recumbent Length/Standing Height", "units": "cm"},
    x_attrs={"name": "age", "long_name": "Age", "units": "days"},
)
lhfa = GAMLSSModelByCondition(
    lhfa_female,
    lhfa_male,
    cond_attrs={"name": "is_female", "long_name": "sex = Female"},
)
ssfa_female = BCCGModel(
    mu=LookupTable(start=91, stop=1856, step=1, fp=_load("ssfa_female_mu")),
    sigma=LookupTable(start=91, stop=1856, step=1, fp=_load("ssfa_female_sigma")),
    nu=LookupTable(start=91, stop=1856, step=1, fp=_load("ssfa_female_nu")),
    attrs={"name": "ss", "long_name": "Subscapular Skinfold", "units": "mm"},
    x_attrs={"name": "age", "long_name": "Age", "units": "days"},
)
ssfa_male = BCCGModel(
    mu=LookupTable(start=91, stop=1856, step=1, fp=_load("ssfa_male_mu")),
    sigma=LookupTable(start=91, stop=1856, step=1, fp=_load("ssfa_male_sigma")),
    nu=LookupTable(start=91, stop=1856, step=1, fp=_load("ssfa_male_nu")),
    attrs={"name": "ss", "long_name": "Subscapular Skinfold", "units": "mm"},
    x_attrs={"name": "age", "long_name": "Age", "units": "days"},
)
ssfa = GAMLSSModelByCondition(
    ssfa_female,
    ssfa_male,
    cond_attrs={"name": "is_female", "long_name": "sex = Female"},
)
tsfa_female = BCCGModel(
    mu=LookupTable(start=91, stop=1856, step=1, fp=_load("tsfa_female_mu")),
    sigma=LookupTable(start=91, stop=1856, step=1, fp=_load("tsfa_female_sigma")),
    nu=LookupTable(start=91, stop=1856, step=1, fp=_load("tsfa_female_nu")),
    attrs={"name": "ts", "long_name": "Triceps Skinfold", "units": "mm"},
    x_attrs={"name": "age", "long_name": "Age", "units": "days"},
)
tsfa_male = BCCGModel(
    mu=LookupTable(start=91, stop=1856, step=1, fp=_load("tsfa_male_mu")),
    sigma=LookupTable(start=91, stop=1856, step=1, fp=_load("tsfa_male_sigma")),
    nu=LookupTable(start=91, stop=1856, step=1, fp=_load("tsfa_male_nu")),
    attrs={"name": "ts", "long_name": "Triceps Skinfold", "units": "mm"},
    x_attrs={"name": "age", "long_name": "Age", "units": "days"},
)
tsfa = GAMLSSModelByCondition(
    tsfa_female,
    tsfa_male,
    cond_attrs={"name": "is_female", "long_name": "sex = Female"},
)
wfa_female = BCCGModel(
    mu=LookupTable(start=0, stop=1856, step=1, fp=_load("wfa_female_mu")),
    sigma=LookupTable(start=0, stop=1856, step=1, fp=_load("wfa_female_sigma")),
    nu=LookupTable(start=0, stop=1856, step=1, fp=_load("wfa_female_nu")),
    attrs={"name": "weight", "long_name": "Weight", "units": "kg"},
    x_attrs={"name": "age", "long_name": "Age", "units": "days"},
)
wfa_male = BCCGModel(
    mu=LookupTable(start=0, stop=1856, step=1, fp=_load("wfa_male_mu")),
    sigma=LookupTable(start=0, stop=1856, step=1, fp=_load("wfa_male_sigma")),
    nu=LookupTable(start=0, stop=1856, step=1, fp=_load("wfa_male_nu")),
    attrs={"name": "weight", "long_name": "Weight", "units": "kg"},
    x_attrs={"name": "age", "long_name": "Age", "units": "days"},
)
wfa = GAMLSSModelByCondition(
    wfa_female,
    wfa_male,
    cond_attrs={"name": "is_female", "long_name": "sex = Female"},
)
wfh_female = BCCGModel(
    mu=LookupTable(start=65, stop=120, step=Fraction(1, 10), fp=_load("wfh_female_mu")),
    sigma=LookupTable(start=65, stop=120, step=Fraction(1, 10), fp=_load("wfh_female_sigma")),
    nu=-0.3833,
    attrs={"name": "weight", "long_name": "Weight", "units": "kg"},
    x_attrs={"name": "height", "long_name": "Standing Height", "units": "cm"},
)
wfh_male = BCCGModel(
    mu=LookupTable(start=65, stop=120, step=Fraction(1, 10), fp=_load("wfh_male_mu")),
    sigma=LookupTable(start=65, stop=120, step=Fraction(1, 10), fp=_load("wfh_male_sigma")),
    nu=-0.3521,
    attrs={"name": "weight", "long_name": "Weight", "units": "kg"},
    x_attrs={"name": "height", "long_name": "Standing Height", "units": "cm"},
)
wfh = GAMLSSModelByCondition(
    wfh_female,
    wfh_male,
    cond_attrs={"name": "is_female", "long_name": "sex = Female"},
)
wfl_female = BCCGModel(
    mu=LookupTable(start=45, stop=110, step=Fraction(1, 10), fp=_load("wfl_female_mu")),
    sigma=LookupTable(start=45, stop=110, step=Fraction(1, 10), fp=_load("wfl_female_sigma")),
    nu=-0.3833,
    attrs={"name": "weight", "long_name": "Weight", "units": "kg"},
    x_attrs={"name": "length", "long_name": "Recumbent Length", "units": "cm"},
)
wfl_male = BCCGModel(
    mu=LookupTable(start=45, stop=110, step=Fraction(1, 10), fp=_load("wfl_male_mu")),
    sigma=LookupTable(start=45, stop=110, step=Fraction(1, 10), fp=_load("wfl_male_sigma")),
    nu=-0.3521,
    attrs={"name": "weight", "long_name": "Weight", "units": "kg"},
    x_attrs={"name": "length", "long_name": "Recumbent Length", "units": "cm"},
)
wfl = GAMLSSModelByCondition(
    wfl_female,
    wfl_male,
    cond_attrs={"name": "is_female", "long_name": "sex = Female"},
)
