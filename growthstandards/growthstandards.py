import importlib.resources
from collections.abc import Mapping
from functools import lru_cache
from itertools import chain
from typing import Iterator

import numpy as np
import scipy.optimize as optimize
import scipy.stats as stats
import xarray as xr
import xarray_einstats.stats as xr_stats
import zarr

from .bcs_ext.scipy_ext import BCCG

try:
    if not hasattr(xr_stats.XrRV, "median"):
        # FIXME: upstream
        xr_stats._add_documented_method(
            xr_stats.XrRV, "rv_generic", ["median"], xr_stats.doc_extras
        )
except:
    pass


def rv_to_ds(rv: xr_stats.XrRV):
    return xr.Dataset(
        {**{da.name: da for da in rv.args}, **rv.kwargs},
        attrs=getattr(rv, "attrs", None),
    )


def ds_to_rv(distr: stats.rv_continuous, ds: xr.Dataset):
    args = [ds[s.name] for s in distr._shape_info()]
    if "loc" in ds:
        args.append(ds["loc"])
    if "scale" in ds:
        args.append(ds["scale"])
    rv = xr_stats.XrContinuousRV(distr, *args)
    # rv = xr_stats.XrContinuousRV(distr, **ds.items())
    rv.attrs = ds.attrs
    return rv


def rv_coords(rv: xr_stats.XrRV) -> xr.Dataset:
    return xr.merge(
        (
            da.coords.to_dataset()
            for da in chain(rv.args, rv.kwargs.values())
            if isinstance(da, xr.DataArray)
        ),
        compat="broadcast_equals",
        combine_attrs="drop_conflicts",
    )


# TODO: Once tanh-sinh quadrature is added to scipy, add the compound_log* functions,
#  which allows evaluating the log of the integral based on the log of the integrand


def compound_pdf(
    x: xr.DataArray,
    cond_rv: xr_stats.XrContinuousRV,
    marginal_rv: xr_stats.XrContinuousRV,
    marginal: xr.DataArray,
    *,
    marginal_dim=None,
):
    if marginal_dim is None:
        marginal_dim = marginal.dims
    elif isinstance(marginal_dim, str):
        marginal_dim = [marginal_dim]
    p_x: xr.DataArray = cond_rv.pdf(x)
    p_marginal = marginal_rv.pdf(marginal)
    return (p_x * p_marginal).integrate(marginal_dim)


def compound_cdf(
    x: xr.DataArray,
    cond_rv: xr_stats.XrContinuousRV,
    marginal_rv: xr_stats.XrContinuousRV,
    marginal: xr.DataArray,
    *,
    marginal_dim=None,
):
    if marginal_dim is None:
        marginal_dim = marginal.dims
    elif isinstance(marginal_dim, str):
        marginal_dim = [marginal_dim]
    cdf_x: xr.DataArray = cond_rv.cdf(x)
    p_marginal = marginal_rv.pdf(marginal)
    return (cdf_x * p_marginal).integrate(marginal_dim)


def compound_ppf(
    q: xr.DataArray,
    cond_rv: xr_stats.XrContinuousRV,
    marginal_rv: xr_stats.XrContinuousRV,
    marginal: xr.DataArray,
    *,
    marginal_dim=None,
):
    def inner_compound_ppf(
        q, x_0, x_min, x_max, m, p_m, *shape_params, dist, shape_keys
    ):
        kwds = dict(zip(shape_keys, shape_params))

        def fun(x):
            return np.trapz(p_m * dist.cdf(x[..., None], **kwds), m) - q

        def dfun(x):
            return np.trapz(p_m * dist.pdf(x[..., None], **kwds), m)

        # return optimize.root_scalar(fun, x0=x_0, bracket=(x_min, x_max), fprime=dfun)
        # return optimize._chandrupatla(fun, x_min, x_max)
        return np.clip(optimize.newton(fun, x_0, fprime=dfun), x_min, x_max)

    if marginal_dim is None:
        marginal_dim = marginal.dims
    elif isinstance(marginal_dim, str):
        marginal_dim = [marginal_dim]
    ppf_q = cond_rv.ppf(q)
    p_marginal = marginal_rv.pdf(marginal)
    x_0: xr.DataArray = ppf_q.weighted(p_marginal).mean(marginal_dim)
    shape_ds = rv_to_ds(cond_rv)
    shape_params = list(shape_ds.values())
    return xr.apply_ufunc(
        inner_compound_ppf,
        q,
        x_0,
        ppf_q.min(marginal_dim),
        ppf_q.max(marginal_dim),
        marginal,
        p_marginal,
        *shape_params,
        input_core_dims=[[], [], [], [], marginal_dim, marginal_dim]
        + [marginal_dim] * len(shape_params),
        kwargs=dict(dist=cond_rv.dist, shape_keys=shape_ds.keys()),
    )


GROWTHSTANDARD_KEYS = (
    "arm_c",
    "bmi_height",
    "bmi_length",
    "head_c",
    "height",
    "length",
    "ss",
    "ts",
    "weight",
    "wfh",
    "wfl",
)

GROWTHSTANDARD_NAMES = (*GROWTHSTANDARD_KEYS, "len_hei", "bmi", "gfl", "gfh")


@lru_cache
def load_growthstandard_ds(g: str) -> xr.Dataset:
    if g in GROWTHSTANDARD_KEYS:
        traversable = importlib.resources.files(__package__)
        store = zarr.DirectoryStore(traversable.joinpath("growthstandards.zarr"))
        return xr.open_zarr(store=store, group=g, decode_times=False).load()
    elif g == "len_hei":
        return xr.combine_by_coords(
            [load_growthstandard_ds("length"), load_growthstandard_ds("height")],
            combine_attrs="drop_conflicts",
        ).assign_attrs(long_name="Recumbent Length / Standing Height")
    elif g == "bmi":
        return xr.combine_by_coords(
            [
                load_growthstandard_ds("bmi_length"),
                load_growthstandard_ds("bmi_height"),
            ],
            combine_attrs="drop_conflicts",
        ).assign_attrs(long_name="Body Mass Index")
    elif g == "gfl":
        # Do not mutate
        gfl = load_growthstandard_ds("wfl").copy()
        gfl["m"] = gfl["m"] / gfl.coords["length"]
        return gfl.assign_attrs(long_name="Growth Metric for Length", units="kg/cm")
    elif g == "gfh":
        # Do not mutate
        gfh = load_growthstandard_ds("wfh").copy()
        gfh["m"] = gfh["m"] / gfh.coords["height"]
        return gfh.assign_attrs(long_name="Growth Metric for Height", units="kg/cm")
    else:
        raise KeyError(g)


class _GrowthStandards(Mapping):
    def __getitem__(self, key: str) -> xr_stats.XrContinuousRV:
        gds = load_growthstandard_ds(key)
        return ds_to_rv(BCCG, gds.rename_vars({"m": "mu", "s": "sigma", "l": "nu"}))

    def __iter__(self) -> Iterator:
        return iter(GROWTHSTANDARD_NAMES)

    def __len__(self) -> int:
        return len(GROWTHSTANDARD_NAMES)


GrowthStandards = _GrowthStandards()
