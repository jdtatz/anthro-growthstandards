#!/usr/bin/env python3
from fractions import Fraction
from functools import partial
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from growthstandards.gamlss_ext import BCCGModel, LookupTable, SimpleBCCGModel

index_attr_map = {
    "age": {"name": "age", "long_name": "Age", "units": "days"},
    "length": {"name": "length", "long_name": "Recumbent Length", "units": "cm"},
    "height": {"name": "height", "long_name": "Standing Height", "units": "cm"},
}


var_attr_map = {
    "ac": {"name": "arm_c", "long_name": "Arm Circumference", "units": "cm"},
    "b": {"name": "bmi", "long_name": "Body Mass Index", "units": "kg/m^2"},
    "hc": {"name": "head_c", "long_name": "Head Circumference", "units": "cm"},
    # lh=dict(name="len_hi", units="cm"),
    "lh": {"name": "len_hi", "long_name": "Recumbent Length/Standing Height", "units": "cm"},
    "ss": {"name": "ss", "long_name": "Subscapular Skinfold", "units": "mm"},
    "ts": {"name": "ts", "long_name": "Triceps Skinfold", "units": "mm"},
    "w": {"name": "weight", "long_name": "Weight", "units": "kg"},
    "wfl": {"name": "wfl", "long_name": "Weight for Recumbent Length", "units": "kg"},
    "wfh": {"name": "wfh", "long_name": "Weight for Standing Height", "units": "kg"},
}


# lambda, df(mu), df(sigma), df(nu)
pow_df_map = {
    "acfa": {"M": (0.35, 7, 4, 2), "F": (0.35, 8, 4, 1)},
    # FIXME: BMI fit is split at 24 months, for by length or by height
    # "bfa": {"M": (), "F": ()},
    "blfa": {"M": (0.05, 10, 4, 3), "F": (0.05, 10, 3, 3)},
    "bhfa": {"M": (None, 4, 3, 3), "F": (None, 4, 4, 1)},
    "hcfa": (0.2, 9, 5, 0),
    "lhfa": {"M": (0.35, 12, 6, 0), "F": (0.35, 10, 5, 0)},
    "ssfa": {"M": (0.65, 6, 2, 2), "F": (0.15, 5, 4, 2)},
    "tsfa": {"M": (0.3, 7, 5, 2), "F": (0.15, 7, 5, 3)},
    "wfa": {"M": (0.35, 13, 6, 1), "F": (0.35, 11, 7, 3)},
    "wfh": {"M": (None, 13, 6, 1), "F": (None, 12, 4, 1)},
    "wfl": {"M": (None, 13, 6, 1), "F": (None, 12, 4, 1)},
}


def to_scalar(v):
    value = float(v)
    return int(value) if value.is_integer() else value


def unique_scalar_or_array(v: npt.ArrayLike):
    uniq = np.unique(v)
    return uniq.item() if uniq.size == 1 else np.asarray(v)


def scalar_or_lookuptable(x: npt.NDArray, v: npt.ArrayLike):
    # return v if np.size(v) == 1 else LookupTable(x, v)
    if np.size(v) == 1:
        return v
    dx = np.diff(x)
    udx = np.unique(dx)
    start = to_scalar(x[0])
    stop = to_scalar(x[-1])
    # print(udx)
    # print(x == np.linspace(x[0], x[-1], len(x)))
    # print(np.allclose(x, np.linspace(x[0], x[-1], len(x))))
    if udx.size == 1:
        step = udx.item()
        if isinstance(step, int):
            # xr = range(x[0], x[-1] + step, step)
            # print(xr)
            # return LookupTable(xr, v)
            # return LookupTable(start=start, stop=stop, step=step, fp=v)
            pass
        else:
            raise TypeError(f"{step} is not an integer")
        # return f"LookupTable(xp=np.arange({x[0]}, {x[-1] + step}, {step}), fp=np.array({v.tolist()}))"
    elif np.allclose(udx, 0.1):
        # return f"LookupTable(xp=np.arange({int(10 * x[0])}, {int(10 * x[-1]) + 1})/10, fp=np.array({v.tolist()}))"
        step = Fraction(1, 10)
        # return LookupTable(start=start, stop=stop, step=step, fp=v)
    return LookupTable(start=start, stop=stop, step=step, fp=v)
    # return LookupTable(start=start, stop=stop, step=step, fp=np.asarray(v).tolist())
    # assert False, f"{udx}"
    # return LookupTable(x, v)start=x[0], stop=x[-1]


dt = xr.DataTree()
models = {}
std_model_names = set()

prelude = """\
from fractions import Fraction

from .gamlss_ext import BCCGModel, LookupTable, SimpleBCCGModel
"""
print(prelude)

table_dir = Path("who-expanded-tables")
for p in sorted(table_dir.glob("*.xlsx")):
    print(f"# {p.stem}")
    std_name, sex, _zscore, _expanded, _tables = p.stem.split("-")
    assert sex in ("boys", "girls")
    sex = "male" if sex == "boys" else "female"
    assert _zscore == "zscore"
    assert _expanded == "expanded"
    assert _tables in {"table", "tables"}
    v_name, c_name = std_name.rsplit("f", 1)
    v_attrs = var_attr_map[v_name]

    df = pd.read_excel(p, index_col=0)
    # df = pd.read_excel(p, dtype=np.str_)
    # df = df.set_index(df.columns[0])

    assert all(c in ("L", "M", "S") or c.startswith("SD") for c in df.columns)
    index_name = df.index.name.lower()
    if index_name == "day":
        index_name = "age"
        assert c_name == "a"
    else:
        assert index_name in ("length", "height")
        if index_name == "length":
            assert c_name == "l"
        else:
            assert c_name == "h"
    index_attrs = index_attr_map[index_name]

    x = np.asarray(df.index)
    mu = unique_scalar_or_array(df["M"])
    sigma = unique_scalar_or_array(df["S"])
    nu = unique_scalar_or_array(df["L"])
    # print(v_name)
    # print(v_attrs)
    # print(index_name)
    # print(index_attrs)
    # print(attrs)
    # print(x)
    # print(mu)
    # print(sigma)
    # print(nu)
    if np.size(nu) == 1 and nu == 1:
        """For Y ∼ BCS(μ, σ, 𝜈; r), if 𝜈 = 1 then Y has a truncated symmetric distribution with parameters μ and μσ and support (0, ∞)."""
        model = SimpleBCCGModel(
            loc=scalar_or_lookuptable(x, mu),
            scale=scalar_or_lookuptable(x, mu * sigma),
            attrs=v_attrs,
            x_attrs=index_attrs,
        )
    else:
        model = BCCGModel(
            mu=scalar_or_lookuptable(x, mu),
            sigma=scalar_or_lookuptable(x, sigma),
            nu=scalar_or_lookuptable(x, nu),
            attrs=v_attrs,
            x_attrs=index_attrs,
        )
    print(f"{std_name}_{sex} = {model}")
    models[f"{std_name}_{sex}"] = model
    # print()
    std_model_names.add(std_name)

    # dims = (index_name,)
    # ds = xr.Dataset(
    #     dict(
    #         mu=(dims if np.ndim(mu) > 0 else (), mu),
    #         sigma=(dims if np.ndim(sigma) > 0 else (), sigma),
    #         nu=(dims if np.ndim(nu) > 0 else (), nu),
    #     ),
    #     coords={index_name: (dims, x, index_attrs)},
    #     attrs=v_attrs,
    # )
    # # print(std_name, ds)
    # dt[f"/{std_name}/{sex}"] = ds

# print(dt)
# dt.to_zarr("test2.zarr", consolidated=False)

all_model_names = []
std_models = {}
for std_name in sorted(std_model_names):
    f_model = models[f"{std_name}_female"]
    m_model = models[f"{std_name}_male"]
    assert f_model.attrs == m_model.attrs
    assert f_model.x_attrs == m_model.x_attrs
    std_models[std_name] = (f_model, m_model)
    all_model_names.append(std_name)
    all_model_names.append(f"{std_name}_female")
    all_model_names.append(f"{std_name}_male")


prelude = f"""\
import importlib.resources
from fractions import Fraction

import numpy as np

from .gamlss_ext import BCCGModel, GAMLSSModelByCondition, SimpleBCCGModel
from .gamlss_params import LookupTable

__all__ = {sorted(all_model_names)!r}

_traversable = importlib.resources.files(__package__)
with _traversable.joinpath("who_model_params.npz").open("rb") as _f, np.load(_f) as npz:
    _npz = dict(npz.items())


def _load(name: str) -> np.ndarray:
    return _npz[name]

"""

with open("growthstandards/who_models.py", "w") as f:
    fprint = partial(print, file=f)
    fprint(prelude)
    npz = {}
    for std_name, (f_model, m_model) in std_models.items():
        for name, model in ((f"{std_name}_female", f_model), (f"{std_name}_male", m_model)):
            if isinstance(model, BCCGModel):
                fprint(f"{name} = BCCGModel(")
                _param_names = "mu", "sigma", "nu"
            else:
                fprint(f"{name} = SimpleBCCGModel(")
                _param_names = "loc", "scale"
            for p in _param_names:
                param = getattr(model, p)
                if isinstance(param, LookupTable):
                    npz_name = f"{name}_{p}"
                    npz[npz_name] = param.fp
                    sval = f'LookupTable(start={param.start!r}, stop={param.stop!r}, step={param.step!r}, fp=_load("{npz_name}"))'
                else:
                    sval = str(param)
                fprint(f"    {p}={sval},")
            fprint(f"    attrs={model.attrs!r},")
            fprint(f"    x_attrs={model.x_attrs!r},")
            fprint(")")
        cond_attrs = {"name": "is_female", "long_name": "sex = Female"}
        fprint(f"{std_name} = GAMLSSModelByCondition(")
        fprint(f"    {std_name}_female,")
        fprint(f"    {std_name}_male,")
        fprint(f"    cond_attrs={cond_attrs!r},")
        fprint(")")
np.savez_compressed("growthstandards/who_model_params.npz", allow_pickle=False, **npz)
