from itertools import chain
from typing import Hashable, Optional, Sequence, Union

import numpy as np
import scipy.optimize as optimize
import scipy.stats as stats
import xarray as xr
import xarray_einstats.stats as xr_stats

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


def _vectorized_root_scalar(f, x0, bracket, fprime):
    x_min, x_max = bracket
    # return optimize._chandrupatla(f, x_min, x_max)
    # need scipy 1.12 for _chandrupatla
    # see https://github.com/scipy/scipy/issues/7242 for stabilization progress
    return np.clip(optimize.newton(f, x0, fprime=fprime), x_min, x_max)


def _compound_ppf_root(
    q,
    x_0,
    x_min,
    x_max,
    m,
    p_m,
    *shape_params,
    dist: stats.rv_continuous,
    shape_keys: tuple[str, ...],
):
    kwds = dict(zip(shape_keys, shape_params))

    def fun(x):
        return np.trapz(p_m * dist.cdf(x[..., None], **kwds), m) - q

    def dfun(x):
        return np.trapz(p_m * dist.pdf(x[..., None], **kwds), m)

    return _vectorized_root_scalar(fun, x0=x_0, bracket=(x_min, x_max), fprime=dfun)


def _compound_isf_root(
    q,
    x_0,
    x_min,
    x_max,
    m,
    p_m,
    *shape_params,
    dist: stats.rv_continuous,
    shape_keys: tuple[str, ...],
):
    kwds = dict(zip(shape_keys, shape_params))

    def fun(x):
        return q - np.trapz(p_m * dist.sf(x[..., None], **kwds), m)

    def dfun(x):
        return np.trapz(p_m * dist.pdf(x[..., None], **kwds), m)

    return _vectorized_root_scalar(fun, x0=x_0, bracket=(x_min, x_max), fprime=dfun)


class XrCompoundRV:
    def __init__(
        self,
        cond_rv: xr_stats.XrContinuousRV,
        marginal_rv: xr_stats.XrContinuousRV,
        marginal: Optional[Union[xr.DataArray, xr.Dataset]] = None,
        *,
        marginal_dim: Optional[Union[Hashable, Sequence[Hashable]]] = None,
    ):
        if marginal is None and marginal_dim is None:
            raise ValueError("one of `marginal` or `marginal_dim` must not be `None`")
        if marginal_dim is None:
            self.marginal_dim = marginal.dims
        elif isinstance(marginal_dim, Hashable):
            self.marginal_dim = [marginal_dim]
        else:
            self.marginal_dim = marginal_dim
        if marginal is None:
            self.marginal = rv_coords(cond_rv)[self.marginal_dim]
        else:
            self.marginal = marginal
        self.cond_rv = cond_rv
        self.marginal_rv = marginal_rv
        self.p_marginal: xr.DataArray = self.marginal_rv.pdf(self.marginal)

    def pdf(self, x: Union[float, xr.DataArray]):
        p_x: xr.DataArray = self.cond_rv.pdf(x)
        return (p_x * self.p_marginal).integrate(self.marginal_dim)

    def cdf(self, x: Union[float, xr.DataArray]):
        cdf_x: xr.DataArray = self.cond_rv.cdf(x)
        return (cdf_x * self.p_marginal).integrate(self.marginal_dim)

    def sf(self, x: Union[float, xr.DataArray]):
        sf_x: xr.DataArray = self.cond_rv.sf(x)
        return (sf_x * self.p_marginal).integrate(self.marginal_dim)

    def ppf(self, q: Union[float, xr.DataArray]):
        ppf_q = self.cond_rv.ppf(q)
        x_0: xr.DataArray = ppf_q.weighted(self.p_marginal).mean(self.marginal_dim)
        shape_ds = rv_to_ds(self.cond_rv)
        shape_params = list(shape_ds.values())
        return xr.apply_ufunc(
            _compound_ppf_root,
            q,
            x_0,
            ppf_q.min(self.marginal_dim),
            ppf_q.max(self.marginal_dim),
            self.marginal,
            self.p_marginal,
            *shape_params,
            input_core_dims=[[], [], [], [], self.marginal_dim, self.marginal_dim]
            + [self.marginal_dim] * len(shape_params),
            kwargs=dict(dist=self.cond_rv.dist, shape_keys=shape_ds.keys()),
        )

    def isf(self, q: Union[float, xr.DataArray]):
        isf_q = self.cond_rv.isf(q)
        x_0: xr.DataArray = isf_q.weighted(self.p_marginal).mean(self.marginal_dim)
        shape_ds = rv_to_ds(self.cond_rv)
        shape_params = list(shape_ds.values())
        return xr.apply_ufunc(
            _compound_isf_root,
            q,
            x_0,
            isf_q.min(self.marginal_dim),
            isf_q.max(self.marginal_dim),
            self.marginal,
            self.p_marginal,
            *shape_params,
            input_core_dims=[[], [], [], [], self.marginal_dim, self.marginal_dim]
            + [self.marginal_dim] * len(shape_params),
            kwargs=dict(dist=self.cond_rv.dist, shape_keys=shape_ds.keys()),
        )

    def median(self):
        return self.ppf(0.5)

    # TODO: fully impl the log* functions once tanh-sinh quadrature is added to scipy,
    #  which allows evaluating the log of the integral based on the log of the integrand
    #  https://github.com/scipy/scipy/pull/18650

    def logpdf(self, x: Union[float, xr.DataArray]):
        return np.log(self.pdf(x))

    def logcdf(self, x: Union[float, xr.DataArray]):
        return np.log(self.cdf(x))

    def logsf(self, x: Union[float, xr.DataArray]):
        return np.log(self.sf(x))


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
