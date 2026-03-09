from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing_extensions import TypedDict

import numpy as np
import numpy.typing as npt
from array_api_extra import apply_where
from scipy import stats
from scipy.integrate import tanhsinh
from scipy.optimize.elementwise import find_root

from .bcs_ext.scipy_ext import BCCG, BCPE
from .gamlss_params import GAMLSSParam, LookupTable, _interpolate, _param_domain, _RealArrayLike


class Attributes(TypedDict, total=False, closed=False):
    name: str
    long_name: str
    units: str


def _interpolate_tuple(x: npt.ArrayLike, *ps: GAMLSSParam):
    return tuple(_interpolate(x, p) for p in ps)


def _merge_param_domains(*ps: GAMLSSParam) -> tuple[int | float, int | float]:
    ls, us = zip(*filter(lambda d: d is not None, map(_param_domain, ps)), strict=True)
    lb = min(ls, default=-np.inf)
    ub = max(us, default=+np.inf)
    return (int(lb) if lb.is_integer() else float(lb)), (int(ub) if ub.is_integer() else float(ub))


class GAMLSSModel(ABC):
    def __init_subclass__(
        cls,
        distr: stats.rv_continuous,
        rv_type: type["stats._distribution_infrastructure.ContinuousDistribution"] | None = None,
    ):
        cls._distr = distr
        cls._rv_type = stats.make_distribution(distr) if rv_type is None else rv_type

    attrs: Attributes
    x_attrs: Attributes

    @property
    @abstractmethod
    def _param_names(self) -> tuple[str, ...]: ...
    @abstractmethod
    def _interpolate_params(self, x: npt.ArrayLike, /) -> tuple[_RealArrayLike, ...]: ...

    def _interpolate_param_dict(self, x: npt.ArrayLike, /) -> dict[str, _RealArrayLike]:
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


## FIXME: Use a better name. Would `UnitaryBCCGModel` be correct?
@dataclass
class SimpleBCCGModel(GAMLSSModel, distr=stats.truncnorm):
    """For Y ∼ BCS(μ, σ, 𝜈; r), if 𝜈 = 1 then Y has a truncated symmetric distribution with parameters μ and μσ and support (0, ∞)."""

    loc: GAMLSSParam
    scale: GAMLSSParam
    attrs: Attributes = field(default_factory=dict)
    x_attrs: Attributes = field(default_factory=dict)

    @property
    def _param_names(self) -> tuple[str, ...]:
        return "a", "b", "loc", "scale"

    def _interpolate_params(self, x: npt.ArrayLike, /):
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
    attrs: Attributes = field(default_factory=dict)
    x_attrs: Attributes = field(default_factory=dict)

    @property
    def _param_names(self) -> tuple[str, ...]:
        return "mu", "sigma", "nu"

    def _interpolate_params(self, x: npt.ArrayLike, /):
        return _interpolate_tuple(x, self.mu, self.sigma, self.nu)

    def _domain(self) -> tuple[int | float, int | float]:
        return _merge_param_domains(self.mu, self.sigma, self.nu)


@dataclass
class BCPEModel(GAMLSSModel, distr=BCPE):
    mu: GAMLSSParam
    sigma: GAMLSSParam
    nu: GAMLSSParam
    tau: GAMLSSParam
    attrs: Attributes = field(default_factory=dict)
    x_attrs: Attributes = field(default_factory=dict)

    @property
    def _param_names(self) -> tuple[str, ...]:
        return "mu", "sigma", "nu", "beta"

    def _interpolate_params(self, x: npt.ArrayLike, /):
        return _interpolate_tuple(x, self.mu, self.sigma, self.nu, self.tau)

    def _domain(self) -> tuple[int | float, int | float]:
        return _merge_param_domains(self.mu, self.sigma, self.nu, self.tau)


@dataclass
class BetaModel(GAMLSSModel, distr=stats.beta):
    mu: GAMLSSParam
    sigma: GAMLSSParam
    attrs: Attributes = field(default_factory=dict)
    x_attrs: Attributes = field(default_factory=dict)

    @property
    def _param_names(self) -> tuple[str, ...]:
        return "a", "b"

    def _interpolate_params(self, x: npt.ArrayLike, /):
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
    attrs: Attributes = field(default_factory=dict)
    x_attrs: Attributes = field(default_factory=dict)

    def __post_init__(self):
        for k, v in self.cond_model.attrs.items():
            if k not in self.attrs:
                self.attrs[k] = v
        for k, v in self.marginal_model.x_attrs.items():
            if k not in self.x_attrs:
                self.x_attrs[k] = v

    @property
    def _param_names(self) -> tuple[str, ...]:
        raise NotImplementedError

    def _interpolate_params(self, x: npt.ArrayLike, /) -> tuple[_RealArrayLike, ...]:
        raise NotImplementedError

    def _interpolate_param_dict(self, x: npt.ArrayLike, /) -> dict[str, _RealArrayLike]:
        raise NotImplementedError

    def interpolate_distr(self, x: npt.ArrayLike, /) -> stats.rv_continuous:
        raise NotImplementedError

    def interpolate_rv(self, x: npt.ArrayLike, /) -> "stats._distribution_infrastructure.ContinuousDistribution":
        raise NotImplementedError

    def _integrate(
        self,
        fn: str,
        log: bool,  # noqa: FBT001
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
        return res.integral

    def _iroot(
        self,
        bfn: str,
        ifn: str,
        log: bool,  # noqa: FBT001
        x: npt.ArrayLike,
        q: npt.ArrayLike,
        *,
        tolerances: dict | None = None,
        maxiter: int | None = None,
        **integrate_kwargs,
    ):
        bfn: Callable[[npt.ArrayLike, npt.ArrayLike], npt.NDArray] = getattr(self.cond_model, bfn)

        def _inner(v, x, q):
            return self._integrate(ifn, log, x, v, **integrate_kwargs) - q

        a, b = self.cond_model._domain()
        min_q = bfn(a, q)
        max_q = bfn(b, q)
        res = find_root(_inner, (min_q, max_q), args=(x, q), tolerances=tolerances, maxiter=maxiter)
        # TODO: return other info?
        return res.x

    ##

    def mean(self, x: npt.ArrayLike, /) -> npt.NDArray:
        raise NotImplementedError

    def median(self, x: npt.ArrayLike, /) -> npt.NDArray:
        # TODO: icdf, iccdf, ilogcdf, or ilogccdf?
        return self.icdf(x, 0.5)

    # ruff: disable[FBT003]

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

    # ruff: enable[FBT003]


@dataclass
class GAMLSSModelByCondition:
    model1: GAMLSSModel
    model2: GAMLSSModel
    attrs: Attributes = field(default_factory=dict)
    x_attrs: Attributes = field(default_factory=dict)
    cond_attrs: Attributes = field(default_factory=dict)

    def __post_init__(self):
        for k in set(self.model1.attrs.keys()) & set(self.model2.attrs.keys()):
            # FIXME: equality check can raise Exception
            if k not in self.attrs and self.model1.attrs[k] == self.model2.attrs[k]:
                self.attrs[k] = self.model1.attrs[k]
        for k in set(self.model1.x_attrs.keys()) & set(self.model2.x_attrs.keys()):
            # FIXME: equality check can raise Exception
            if k not in self.x_attrs and self.model1.x_attrs[k] == self.model2.x_attrs[k]:
                self.x_attrs[k] = self.model1.x_attrs[k]

    @property
    def _param_names(self) -> tuple[str, ...]:
        return self.model1._param_names

    @abstractmethod
    def _interpolate_params(self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, /) -> tuple[_RealArrayLike, ...]:
        if self.model1._distr is not self.model2._distr:
            raise TypeError(f"(model1 ~ {self.model1._distr.name}) != (model2 ~ {self.model2._distr.name})")
        ## TODO: use `apply_where` when it accepts tuple output
        # return apply_where(cond, (x,), self.model1._interpolate_params, self.model2._interpolate_params, xp=np)
        params1 = self.model1._interpolate_params(x)
        params2 = self.model2._interpolate_params(x)
        return tuple(np.where(cond, p1, p2) for p1, p2 in zip(params1, params2, strict=True))

    def _interpolate_param_dict(self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, /) -> dict[str, _RealArrayLike]:
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
