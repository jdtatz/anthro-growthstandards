from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import scipy.special
import xarray as xr
import xarray_einstats.stats as xr_stats
from array_api_extra import apply_where
from scipy import interpolate, stats
from scipy.integrate import tanhsinh
from scipy.optimize.elementwise import find_root

from .bcs_ext.scipy_ext import BCCG, BCPE

type GAMLSSParam = int | float | Callable[[npt.ArrayLike], npt.ArrayLike]


def _interpolate(x: npt.ArrayLike, p: GAMLSSParam):
    return p if isinstance(p, (int, float)) else p(x)


def _interpolate_tuple(x: npt.ArrayLike, *ps: GAMLSSParam):
    return tuple(_interpolate(x, p) for p in ps)


def _param_domain(p: GAMLSSParam) -> None | tuple[int | float, int | float]:
    return None if isinstance(p, (int, float)) or not hasattr(p, "domain") else p.domain


def _merge_param_domains(*ps: GAMLSSParam) -> tuple[int | float, int | float]:
    ls, us = zip(*filter(lambda d: d is not None, map(_param_domain, ps)), strict=True)
    l = min(ls, default=-np.inf)
    u = max(us, default=+np.inf)
    return l, u


class GAMLSSModel(ABC):
    def __init_subclass__(
        cls,
        distr: stats.rv_continuous,
        rv_type: Optional[type["stats._distribution_infrastructure.ContinuousDistribution"]] = None,
    ):
        cls._distr = distr
        cls._rv_type = stats.make_distribution(distr) if rv_type is None else rv_type

    @property
    @abstractmethod
    def _param_names(self) -> tuple[str, ...]: ...
    @abstractmethod
    def _interpolate_params(self, x: npt.ArrayLike, /) -> tuple[npt.ArrayLike, ...]: ...

    def _interpolate_param_dict(self, x: npt.ArrayLike, /) -> dict[str, npt.ArrayLike]:
        return dict(zip(self._param_names, self._interpolate_params(x), strict=True))

    def interpolate_distr(self, x: npt.ArrayLike, /) -> stats.rv_continuous:
        return self._distr(**self._interpolate_param_dict(x))

    def interpolate_rv(self, x: npt.ArrayLike, /) -> "stats._distribution_infrastructure.ContinuousDistribution":
        params = self._interpolate_param_dict(x)
        loc = params.pop("loc", None)
        scale = params.pop("scale", None)
        rv = self._rv_type(**params)
        if loc is not None and scale is not None:
            return loc + scale * rv
        if loc is not None:
            return loc + rv
        if scale is not None:
            return scale * rv
        return rv

    def interpolate_xr_rv(self, x: xr.DataArray) -> xr_stats.XrContinuousRV:
        params = xr.apply_ufunc(self._interpolate_params, x)
        return xr_stats.XrContinuousRV(self._distr, *params)

    @abstractmethod
    def _domain(self) -> tuple[int | float, int | float]: ...

    ## Convenience Methods

    def mean(self, x: npt.ArrayLike, /) -> npt.NDArray:
        return self.interpolate_rv(x).mean()

    def median(self, x: npt.ArrayLike, /) -> npt.NDArray:
        return self.interpolate_rv(x).median()

    def pdf(self, x: npt.ArrayLike, v: npt.ArrayLike, /) -> npt.NDArray:
        return self.interpolate_rv(x).pdf(v)

    def logpdf(self, x: npt.ArrayLike, v: npt.ArrayLike, /) -> npt.NDArray:
        return self.interpolate_rv(x).logpdf(v)

    def cdf(self, x: npt.ArrayLike, v: npt.ArrayLike, /) -> npt.NDArray:
        return self.interpolate_rv(x).cdf(v)

    def icdf(self, x: npt.ArrayLike, q: npt.ArrayLike, /) -> npt.NDArray:
        return self.interpolate_rv(x).icdf(q)

    def ccdf(self, x: npt.ArrayLike, v: npt.ArrayLike, /) -> npt.NDArray:
        return self.interpolate_rv(x).ccdf(v)

    def iccdf(self, x: npt.ArrayLike, q: npt.ArrayLike, /) -> npt.NDArray:
        return self.interpolate_rv(x).iccdf(q)

    def logcdf(self, x: npt.ArrayLike, v: npt.ArrayLike, /) -> npt.NDArray:
        return self.interpolate_rv(x).logcdf(v)

    def ilogcdf(self, x: npt.ArrayLike, logp: npt.ArrayLike, /) -> npt.NDArray:
        return self.interpolate_rv(x).ilogcdf(logp)

    def logccdf(self, x: npt.ArrayLike, v: npt.ArrayLike, /) -> npt.NDArray:
        return self.interpolate_rv(x).logccdf(v)

    def ilogccdf(self, x: npt.ArrayLike, logp: npt.ArrayLike, /) -> npt.NDArray:
        return self.interpolate_rv(x).ilogccdf(logp)


class GAMLSSLinkFunction(ABC):
    def __init__(self, inner: GAMLSSParam):
        self.inner = inner

    def __repr__(self):
        name = type(self).__name__
        return f"{name}({self.inner!r})"

    @property
    def domain(self) -> None | tuple[int | float, int | float]:
        return _param_domain(self.inner)

    @abstractmethod
    def forward(self, value: npt.ArrayLike) -> npt.ArrayLike: ...

    def __call__(self, x: npt.ArrayLike, /):
        return self.forward(_interpolate(x, self.inner))


class LogLink(GAMLSSLinkFunction):
    def forward(self, value: npt.ArrayLike) -> npt.ArrayLike:
        return np.exp(value)


class LogitLink(GAMLSSLinkFunction):
    def forward(self, value: npt.ArrayLike) -> npt.ArrayLike:
        return scipy.special.expit(value)


@dataclass
class LookupTable:
    start: int
    stop: int
    step: int | Fraction
    fp: npt.ArrayLike
    xp: npt.ArrayLike = field(init=False, repr=False)

    def __post_init__(self):
        if isinstance(self.step, Fraction):
            n, d = self.step.numerator, self.step.denominator
            self.xp = np.arange(d * self.start, d * self.stop + 1, n) / d
        else:
            self.xp = np.arange(self.start, self.stop + self.step, self.step)

    @property
    def domain(self) -> tuple[int, int]:
        return self.start, self.stop

    def __call__(self, x: npt.ArrayLike, /):
        x = np.asanyarray(x)
        return np.interp(x, self.xp, self.fp)

    def __getitem__(self, key: slice):
        if not isinstance(key, slice):
            raise TypeError("LookupTable only supports subslicing")
        if not isinstance(self.step, int):
            raise NotImplementedError("Can only slice LookupTables with `int` steps")
        new_range = range(self.start, self.stop + self.step, self.step)[key]
        return LookupTable(
            start=new_range.start, stop=new_range.stop - new_range.step, step=new_range.step, fp=self.fp[key]
        )


@dataclass
class FractionalPolynomial:
    intercept: float
    coefficients: tuple[float, ...]
    fpowers: tuple[float, ...]
    domain: tuple[float, float]
    shift: float = 0
    inv_scale: float = field(init=False, repr=False)

    def __post_init__(self):
        assert len(self.coefficients) == len(self.fpowers)
        x_min, x_max = self.domain
        self.inv_scale = 10 ** int(-np.trunc(np.log10(x_max - x_min)))

    def __call__(self, x: npt.ArrayLike, /):
        x = np.asanyarray(x)
        if self.shift:
            x = x + self.shift
        x = x * self.inv_scale

        y = self.intercept
        last = None
        for c, p in zip(self.coefficients, self.fpowers, strict=True):
            xp = np.log(x) if p == 0 else x**p
            if last == p:
                xp *= np.log(x)
            y += c * xp
            last = p
        return y


## FIXME: Bikeshed (include linear component in name)
@dataclass
class PSpline:
    domain: tuple[float, float]
    intercept: float
    slope: float
    spline_coefficients: tuple[float, ...]
    spline_degree: int
    spline: Callable = field(init=False, repr=False)

    def __post_init__(self):
        ndx = len(self.spline_coefficients) - self.spline_degree
        xmin, xmax = self.domain
        xmin, xmax = (xmin - 0.01 * (xmax - xmin)), (xmax + 0.01 * (xmax - xmin))
        dx = (xmax - xmin) / ndx
        knots = xmin + np.arange(-self.spline_degree, ndx + self.spline_degree + 1) * dx

        self.spline = interpolate.BSpline(knots, self.spline_coefficients, self.spline_degree)

    def __call__(self, x: npt.ArrayLike, /):
        x = np.asanyarray(x)
        return self.intercept + self.slope * x + self.spline(x)


## FIXME: Use a better name. Would `UnitaryBCCGModel` be correct?
@dataclass
class SimpleBCCGModel(GAMLSSModel, distr=stats.truncnorm):
    """For Y ∼ BCS(μ, σ, 𝜈; r), if 𝜈 = 1 then Y has a truncated symmetric distribution with parameters μ and μσ and support (0, ∞)."""

    loc: GAMLSSParam
    scale: GAMLSSParam
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def _param_names(self) -> tuple[str, ...]:
        return "a", "b", "loc", "scale"

    def _interpolate_params(self, x: npt.ArrayLike, /) -> tuple[npt.ArrayLike, ...]:
        loc, scale = _interpolate_tuple(x, self.loc, self.scale)
        # NOTE: `a, b = (lb - loc) / scale, (ub - loc) / scale`
        # lb, ub = 0, np.inf
        a = -loc / scale
        b = np.inf
        return a, b, loc, scale

    ## TODO: Is the only benefit to `truncnorm` over `truncate(Normal(...))` moment computation?
    ## TODO: Is `truncate(Normal(...), lb=0)` better numerically than `truncnorm(a=-loc/scale, ...)`?
    # def interpolate_rv(self, x: npt.ArrayLike, /):
    #     loc, scale = _interpolate_tuple(x, self.loc, self.scale)
    #     return stats.truncate(stats.Normal(mu=loc, sigma=scale), lb=0)

    def _domain(self) -> tuple[int | float, int | float]:
        return _merge_param_domains(self.loc, self.scale)


@dataclass
class BCCGModel(GAMLSSModel, distr=BCCG):
    mu: GAMLSSParam
    sigma: GAMLSSParam
    nu: GAMLSSParam
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def _param_names(self) -> tuple[str, ...]:
        return "mu", "sigma", "nu"

    def _interpolate_params(self, x: npt.ArrayLike, /) -> tuple[npt.ArrayLike, ...]:
        return _interpolate_tuple(x, self.mu, self.sigma, self.nu)

    def _domain(self) -> tuple[int | float, int | float]:
        return _merge_param_domains(self.mu, self.sigma, self.nu)


@dataclass
class BCPEModel(GAMLSSModel, distr=BCPE):
    mu: GAMLSSParam
    sigma: GAMLSSParam
    nu: GAMLSSParam
    tau: GAMLSSParam
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def _param_names(self) -> tuple[str, ...]:
        return "mu", "sigma", "nu", "beta"

    def _interpolate_params(self, x: npt.ArrayLike, /) -> tuple[npt.ArrayLike, ...]:
        return _interpolate_tuple(x, self.mu, self.sigma, self.nu, self.tau)

    def _domain(self) -> tuple[int | float, int | float]:
        return _merge_param_domains(self.mu, self.sigma, self.nu, self.tau)


@dataclass
class BetaModel(GAMLSSModel, distr=stats.beta):
    mu: GAMLSSParam
    sigma: GAMLSSParam
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def _param_names(self) -> tuple[str, ...]:
        return "a", "b"

    def _interpolate_params(self, x: npt.ArrayLike, /) -> tuple[npt.ArrayLike, ...]:
        mu, sigma = _interpolate_tuple(x, self.mu, self.sigma)
        s2 = sigma**2
        # var = s2 * mu * (1 - mu)
        # nu = mu * (1 - mu) / var - 1
        nu = 1 / s2 - 1
        a = mu * nu
        b = (1 - mu) * nu
        return a, b

    def _domain(self) -> tuple[int | float, int | float]:
        return _merge_param_domains(self.mu, self.sigma)


# WIP
@dataclass
class CompoundGAMLSSModel:
    cond_model: GAMLSSModel
    # TODO: multiple marginal models?
    marginal_model: GAMLSSModel
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.attrs |= {k: v for k, v in self.cond_model.attrs.items() if not k.startswith("x_")} | {
            k: v for k, v in self.marginal_model.attrs.items() if k.startswith("x_")
        }

    @property
    def _param_names(self) -> tuple[str, ...]:
        raise NotImplementedError

    def _interpolate_params(self, x: npt.ArrayLike, /) -> tuple[npt.ArrayLike, ...]:
        raise NotImplementedError

    def _interpolate_param_dict(self, x: npt.ArrayLike, /) -> dict[str, npt.ArrayLike]:
        raise NotImplementedError

    def interpolate_distr(self, x: npt.ArrayLike, /) -> stats.rv_continuous:
        raise NotImplementedError

    def interpolate_rv(self, x: npt.ArrayLike, /) -> "stats._distribution_infrastructure.ContinuousDistribution":
        raise NotImplementedError

    def interpolate_xr_rv(self, x: xr.DataArray) -> xr_stats.XrContinuousRV:
        raise NotImplementedError

    def _integrate(
        self,
        fn: str,
        log: bool,
        x: npt.ArrayLike,
        v: npt.ArrayLike,
        *,
        maxlevel: int | None = None,
        minlevel: int = 2,
        atol: float | None = None,
        rtol: float | None = None,
    ):
        ifn: Callable[[npt.ArrayLike, npt.ArrayLike], npt.NDArray] = getattr(self.cond_model, fn)

        def _integrand(m, x, v):
            if log:
                return self.marginal_model.logpdf(x, m) + ifn(m, v)
            else:
                return self.marginal_model.pdf(x, m) * ifn(m, v)

        a, b = self.cond_model._domain()
        res = tanhsinh(
            _integrand, a, b, args=(x, v), log=log, maxlevel=maxlevel, minlevel=minlevel, atol=atol, rtol=rtol
        )
        # TODO: return other info?
        return res

    def _iroot(
        self,
        bfn: str,
        ifn: str,
        log: bool,
        x: npt.ArrayLike,
        q: npt.ArrayLike,
        *,
        tolerances: dict | None = None,
        maxiter: int | None = None,
        **integrate_kwargs,
    ):
        bfn: Callable[[npt.ArrayLike, npt.ArrayLike], npt.NDArray] = getattr(self.cond_model, bfn)

        def _inner(v, x, q):
            return self._integrate(ifn, log, x, v, **integrate_kwargs).integral - q

        a, b = self.cond_model._domain()
        min_q = bfn(a, q)
        max_q = bfn(b, q)
        res = find_root(_inner, (min_q, max_q), args=(x, q), tolerances=tolerances, maxiter=maxiter)
        # TODO: return other info?
        return res

    ##

    def mean(self, x: npt.ArrayLike, /) -> npt.NDArray:
        raise NotImplementedError

    def median(self, x: npt.ArrayLike, /) -> npt.NDArray:
        # TODO: icdf, iccdf, ilogcdf, or ilogccdf?
        return self.icdf(x, 0.5)

    def pdf(self, x: npt.ArrayLike, v: npt.ArrayLike, /) -> npt.NDArray:
        return self._integrate("pdf", False, x, v)

    def logpdf(self, x: npt.ArrayLike, v: npt.ArrayLike, /) -> npt.NDArray:
        return self._integrate("logpdf", True, x, v)

    def cdf(self, x: npt.ArrayLike, v: npt.ArrayLike, /) -> npt.NDArray:
        return self._integrate("cdf", False, x, v)

    def icdf(self, x: npt.ArrayLike, q: npt.ArrayLike, /) -> npt.NDArray:
        return self._iroot("icdf", "cdf", False, x, q)

    def ccdf(self, x: npt.ArrayLike, v: npt.ArrayLike, /) -> npt.NDArray:
        return self._integrate("ccdf", False, x, v)

    def iccdf(self, x: npt.ArrayLike, q: npt.ArrayLike, /) -> npt.NDArray:
        return self._iroot("iccdf", "ccdf", False, x, q)

    def logcdf(self, x: npt.ArrayLike, v: npt.ArrayLike, /) -> npt.NDArray:
        return self._integrate("logcdf", True, x, v)

    def ilogcdf(self, x: npt.ArrayLike, logp: npt.ArrayLike, /) -> npt.NDArray:
        return self._iroot("ilogcdf", "logcdf", True, x, logp)

    def logccdf(self, x: npt.ArrayLike, v: npt.ArrayLike, /) -> npt.NDArray:
        return self._integrate("logccdf", True, x, v)

    def ilogccdf(self, x: npt.ArrayLike, logp: npt.ArrayLike, /) -> npt.NDArray:
        return self._iroot("ilogccdf", "logccdf", True, x, logp)


@dataclass
class GAMLSSModelByCondition:
    model1: GAMLSSModel
    model2: GAMLSSModel
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.model1._distr is not self.model2._distr:
            raise TypeError(f"(model1 ~ {self.model1._distr.name}) != (model2 ~ {self.model2._distr.name})")

    @property
    def _param_names(self) -> tuple[str, ...]:
        return self.model1._param_names

    @abstractmethod
    def _interpolate_params(self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, /) -> tuple[npt.ArrayLike, ...]:
        ## TODO: use `apply_where` when it accepts tuple output
        # return apply_where(cond, (x,), self.model1._interpolate_params, self.model2._interpolate_params, xp=np)
        params1 = self.model1._interpolate_params(x)
        params2 = self.model2._interpolate_params(x)
        return tuple(np.where(cond, p1, p2) for p1, p2 in zip(params1, params2, strict=True))

    def _interpolate_param_dict(self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, /) -> dict[str, npt.ArrayLike]:
        return dict(zip(self._param_names, self._interpolate_params(cond, x), strict=True))

    def interpolate_distr(self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, /) -> stats.rv_continuous:
        return self.model1._distr(**self._interpolate_param_dict(cond, x))

    def interpolate_rv(
        self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, /
    ) -> "stats._distribution_infrastructure.ContinuousDistribution":
        params = self._interpolate_param_dict(cond, x)
        loc = params.pop("loc", None)
        scale = params.pop("scale", None)
        rv = self.model1._rv_type(**params)
        if loc is not None and scale is not None:
            return loc + scale * rv
        if loc is not None:
            return loc + rv
        if scale is not None:
            return scale * rv
        return rv

    def _domain(self):
        # TODO: how to handle when models' domains differ
        # l1, u1 = self.model1._domain()
        # l2, u2 = self.model2._domain()
        raise NotImplementedError

    ## Convenience Methods

    def mean(self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, /) -> npt.NDArray:
        return apply_where(cond, (x,), self.model1.mean, self.model2.mean, xp=np)

    def median(self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, /) -> npt.NDArray:
        return apply_where(cond, (x,), self.model1.median, self.model2.median, xp=np)

    def pdf(self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, v: npt.ArrayLike, /) -> npt.NDArray:
        return apply_where(cond, (x, v), self.model1.pdf, self.model2.pdf, xp=np)

    def logpdf(self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, v: npt.ArrayLike, /) -> npt.NDArray:
        return apply_where(cond, (x, v), self.model1.logpdf, self.model2.logpdf, xp=np)

    def cdf(self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, v: npt.ArrayLike, /) -> npt.NDArray:
        return apply_where(cond, (x, v), self.model1.cdf, self.model2.cdf, xp=np)

    def icdf(self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, q: npt.ArrayLike, /) -> npt.NDArray:
        return apply_where(cond, (x, q), self.model1.icdf, self.model2.icdf, xp=np)

    def ccdf(self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, v: npt.ArrayLike, /) -> npt.NDArray:
        return apply_where(cond, (x, v), self.model1.ccdf, self.model2.ccdf, xp=np)

    def iccdf(self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, q: npt.ArrayLike, /) -> npt.NDArray:
        return apply_where(cond, (x, q), self.model1.iccdf, self.model2.iccdf, xp=np)

    def logcdf(self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, v: npt.ArrayLike, /) -> npt.NDArray:
        return apply_where(cond, (x, v), self.model1.logcdf, self.model2.logcdf, xp=np)

    def ilogcdf(self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, logp: npt.ArrayLike, /) -> npt.NDArray:
        return apply_where(cond, (x, logp), self.model1.ilogcdf, self.model2.ilogcdf, xp=np)

    def logccdf(self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, v: npt.ArrayLike, /) -> npt.NDArray:
        return apply_where(cond, (x, v), self.model1.logccdf, self.model2.logccdf, xp=np)

    def ilogccdf(self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, logp: npt.ArrayLike, /) -> npt.NDArray:
        return apply_where(cond, (x, logp), self.model1.ilogccdf, self.model2.ilogccdf, xp=np)
