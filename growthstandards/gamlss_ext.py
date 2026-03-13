from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Literal
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


def _interpolate_tuple(x: npt.ArrayLike, *ps: GAMLSSParam, extrapolate: bool):
    return tuple(_interpolate(x, p, extrapolate=extrapolate) for p in ps)


def _merge_param_domains(*ps: GAMLSSParam) -> tuple[int | float, int | float]:
    ls, us = zip(*filter(lambda d: d is not None, map(_param_domain, ps)), strict=True)
    lb = min(ls, default=-np.inf)
    ub = max(us, default=+np.inf)
    return (int(lb) if lb.is_integer() else float(lb)), (int(ub) if ub.is_integer() else float(ub))


class GAMLSSModel(ABC):
    attrs: Attributes
    x_attrs: Attributes

    @abstractmethod
    def interpolate_distr(self, x: npt.ArrayLike, /, *, extrapolate: bool = False) -> stats.rv_continuous: ...

    @abstractmethod
    def interpolate_rv(
        self, x: npt.ArrayLike, /, *, extrapolate: bool = False
    ) -> "stats._distribution_infrastructure.ContinuousDistribution": ...

    @abstractmethod
    def _domain(self) -> tuple[int | float, int | float]: ...

    ## Convenience Methods

    def support(self, x: npt.ArrayLike, /, *, extrapolate: bool = False) -> tuple[npt.NDArray, npt.NDArray]:
        return self.interpolate_rv(x, extrapolate=extrapolate).support()

    def sample(self, x: npt.ArrayLike, /, shape=(), *, extrapolate: bool = False, rng=None) -> npt.NDArray:
        return self.interpolate_rv(x, extrapolate=extrapolate).sample(shape, rng=rng)

    def moment(
        self,
        x: npt.ArrayLike,
        /,
        *,
        extrapolate: bool = False,
        order: int = 1,
        kind: Literal["raw", "central", "standardized"] = "raw",
    ) -> npt.NDArray:
        return self.interpolate_rv(x, extrapolate=extrapolate).moment(order, kind)

    def mean(self, x: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self.interpolate_rv(x, extrapolate=extrapolate).mean()

    def median(self, x: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self.interpolate_rv(x, extrapolate=extrapolate).median()

    def mode(self, x: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self.interpolate_rv(x, extrapolate=extrapolate).mode()

    def variance(self, x: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self.interpolate_rv(x, extrapolate=extrapolate).variance()

    def standard_deviation(self, x: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self.interpolate_rv(x, extrapolate=extrapolate).standard_deviation()

    def skewness(self, x: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self.interpolate_rv(x, extrapolate=extrapolate).skewness()

    def kurtosis(self, x: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self.interpolate_rv(x, extrapolate=extrapolate).kurtosis()

    def pdf(self, x: npt.ArrayLike, v: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self.interpolate_rv(x, extrapolate=extrapolate).pdf(v)

    def logpdf(self, x: npt.ArrayLike, v: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self.interpolate_rv(x, extrapolate=extrapolate).logpdf(v)

    def cdf(self, x: npt.ArrayLike, v: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self.interpolate_rv(x, extrapolate=extrapolate).cdf(v)

    def icdf(self, x: npt.ArrayLike, q: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self.interpolate_rv(x, extrapolate=extrapolate).icdf(q)

    def ccdf(self, x: npt.ArrayLike, v: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self.interpolate_rv(x, extrapolate=extrapolate).ccdf(v)

    def iccdf(self, x: npt.ArrayLike, q: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self.interpolate_rv(x, extrapolate=extrapolate).iccdf(q)

    def logcdf(self, x: npt.ArrayLike, v: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self.interpolate_rv(x, extrapolate=extrapolate).logcdf(v)

    def ilogcdf(self, x: npt.ArrayLike, logp: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self.interpolate_rv(x, extrapolate=extrapolate).ilogcdf(logp)

    def logccdf(self, x: npt.ArrayLike, v: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self.interpolate_rv(x, extrapolate=extrapolate).logccdf(v)

    def ilogccdf(self, x: npt.ArrayLike, logp: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self.interpolate_rv(x, extrapolate=extrapolate).ilogccdf(logp)

    def logentropy(self, x: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self.interpolate_rv(x, extrapolate=extrapolate).logentropy()

    def entropy(self, x: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self.interpolate_rv(x, extrapolate=extrapolate).entropy()


class _BaseGAMLSSModel(GAMLSSModel):
    def __init_subclass__(
        cls,
        distr: stats.rv_continuous,
        rv_type: type["stats._distribution_infrastructure.ContinuousDistribution"] | None = None,
    ):
        cls._distr = distr
        cls._rv_type = stats.make_distribution(distr) if rv_type is None else rv_type

    @property
    @abstractmethod
    def _param_names(self) -> tuple[str, ...]: ...
    @abstractmethod
    def _interpolate_params(self, x: npt.ArrayLike, /, *, extrapolate: bool) -> tuple[_RealArrayLike, ...]: ...

    def _interpolate_param_dict(self, x: npt.ArrayLike, /, *, extrapolate: bool) -> dict[str, _RealArrayLike]:
        return dict(zip(self._param_names, self._interpolate_params(x, extrapolate=extrapolate), strict=True))

    def interpolate_distr(self, x: npt.ArrayLike, /, *, extrapolate: bool = False) -> stats.rv_continuous:
        return self._distr(**self._interpolate_param_dict(x, extrapolate=extrapolate))

    def interpolate_rv(
        self, x: npt.ArrayLike, /, *, extrapolate: bool = False
    ) -> "stats._distribution_infrastructure.ContinuousDistribution":
        params = self._interpolate_param_dict(x, extrapolate=extrapolate)
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


## FIXME: Use a better name. Would `UnitaryBCCGModel` be correct?
@dataclass
class SimpleBCCGModel(_BaseGAMLSSModel, distr=stats.truncnorm):
    """For Y ∼ BCS(μ, σ, 𝜈; r), if 𝜈 = 1 then Y has a truncated symmetric distribution with parameters μ and μσ and support (0, ∞)."""

    loc: GAMLSSParam
    scale: GAMLSSParam
    attrs: Attributes = field(default_factory=dict)
    x_attrs: Attributes = field(default_factory=dict)

    @property
    def _param_names(self) -> tuple[str, ...]:
        return "a", "b", "loc", "scale"

    def _interpolate_params(self, x: npt.ArrayLike, /, *, extrapolate: bool):
        loc, scale = _interpolate_tuple(x, self.loc, self.scale, extrapolate=extrapolate)
        # NOTE: `a, b = (lb - loc) / scale, (ub - loc) / scale`
        # lb, ub = 0, np.inf
        a = -loc / scale
        b = np.inf
        return a, b, loc, scale

    ## TODO: Is the only benefit to `truncnorm` over `truncate(Normal(...))` moment computation?
    ## TODO: Is `truncate(Normal(...), lb=0)` better numerically than `truncnorm(a=-loc/scale, ...)`?
    # def interpolate_rv(self, x: npt.ArrayLike, /, *, extrapolate: bool = False):
    #     loc, scale = _interpolate_tuple(x, self.loc, self.scale, extrapolate=extrapolate)
    #     return stats.truncate(stats.Normal(mu=loc, sigma=scale), lb=0)

    def _domain(self) -> tuple[int | float, int | float]:
        return _merge_param_domains(self.loc, self.scale)


@dataclass
class BCCGModel(_BaseGAMLSSModel, distr=BCCG):
    mu: GAMLSSParam
    sigma: GAMLSSParam
    nu: GAMLSSParam
    attrs: Attributes = field(default_factory=dict)
    x_attrs: Attributes = field(default_factory=dict)

    @property
    def _param_names(self) -> tuple[str, ...]:
        return "mu", "sigma", "nu"

    def _interpolate_params(self, x: npt.ArrayLike, /, *, extrapolate: bool):
        return _interpolate_tuple(x, self.mu, self.sigma, self.nu, extrapolate=extrapolate)

    def _domain(self) -> tuple[int | float, int | float]:
        return _merge_param_domains(self.mu, self.sigma, self.nu)


@dataclass
class BCPEModel(_BaseGAMLSSModel, distr=BCPE):
    mu: GAMLSSParam
    sigma: GAMLSSParam
    nu: GAMLSSParam
    tau: GAMLSSParam
    attrs: Attributes = field(default_factory=dict)
    x_attrs: Attributes = field(default_factory=dict)

    @property
    def _param_names(self) -> tuple[str, ...]:
        return "mu", "sigma", "nu", "beta"

    def _interpolate_params(self, x: npt.ArrayLike, /, *, extrapolate: bool):
        return _interpolate_tuple(x, self.mu, self.sigma, self.nu, self.tau, extrapolate=extrapolate)

    def _domain(self) -> tuple[int | float, int | float]:
        return _merge_param_domains(self.mu, self.sigma, self.nu, self.tau)


@dataclass
class BetaModel(_BaseGAMLSSModel, distr=stats.beta):
    mu: GAMLSSParam
    sigma: GAMLSSParam
    attrs: Attributes = field(default_factory=dict)
    x_attrs: Attributes = field(default_factory=dict)

    @property
    def _param_names(self) -> tuple[str, ...]:
        return "a", "b"

    def _interpolate_params(self, x: npt.ArrayLike, /, *, extrapolate: bool):
        mu, sigma = _interpolate_tuple(x, self.mu, self.sigma, extrapolate=extrapolate)
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
class CompoundGAMLSSModel(GAMLSSModel):
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

    def _interpolate_params(self, x: npt.ArrayLike, /, *, extrapolate: bool) -> tuple[_RealArrayLike, ...]:
        raise NotImplementedError

    def _interpolate_param_dict(self, x: npt.ArrayLike, /, *, extrapolate: bool) -> dict[str, _RealArrayLike]:
        raise NotImplementedError

    def interpolate_distr(self, x: npt.ArrayLike, /, *, extrapolate: bool = False) -> stats.rv_continuous:
        raise NotImplementedError

    def interpolate_rv(
        self, x: npt.ArrayLike, /, *, extrapolate: bool = False
    ) -> "stats._distribution_infrastructure.ContinuousDistribution":
        raise NotImplementedError

    def _domain(self) -> tuple[int | float, int | float]:
        return self.marginal_model._domain()

    def _integrate(
        self,
        fn: str,
        log: bool,  # noqa: FBT001
        x: npt.ArrayLike,
        v: npt.ArrayLike,
        *,
        extrapolate: bool,
        maxlevel: int | None = None,
        minlevel: int = 2,
        atol: float | None = None,
        rtol: float | None = None,
    ):
        ifn: Callable[[npt.ArrayLike, npt.ArrayLike], npt.NDArray] = getattr(self.cond_model, fn)

        def _integrand(m, x, v):
            if log:
                return self.marginal_model.logpdf(x, m, extrapolate=extrapolate) + ifn(m, v, extrapolate=False)
            else:
                return self.marginal_model.pdf(x, m, extrapolate=extrapolate) * ifn(m, v, extrapolate=False)

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
        extrapolate: bool,
        tolerances: dict | None = None,
        maxiter: int | None = None,
        **integrate_kwargs,
    ):
        bfn: Callable[[npt.ArrayLike, npt.ArrayLike], npt.NDArray] = getattr(self.cond_model, bfn)

        def _inner(v, x, q):
            return self._integrate(ifn, log, x, v, extrapolate=extrapolate, **integrate_kwargs) - q

        a, b = self.cond_model._domain()
        min_q = bfn(a, q, extrapolate=False)
        max_q = bfn(b, q, extrapolate=False)
        res = find_root(_inner, (min_q, max_q), args=(x, q), tolerances=tolerances, maxiter=maxiter)
        # TODO: return other info?
        return res.x

    ##

    def support(self, x: npt.ArrayLike, /, *, extrapolate: bool = False) -> tuple[npt.NDArray, npt.NDArray]:
        raise NotImplementedError

    def sample(self, x: npt.ArrayLike, /, shape=(), *, extrapolate: bool = False, rng=None) -> npt.NDArray:
        raise NotImplementedError

    def moment(
        self,
        x: npt.ArrayLike,
        /,
        *,
        extrapolate: bool = False,
        order: int = 1,
        kind: Literal["raw", "central", "standardized"] = "raw",
    ) -> npt.NDArray:
        raise NotImplementedError

    def mean(self, x: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        raise NotImplementedError

    def median(self, x: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        # TODO: icdf, iccdf, ilogcdf, or ilogccdf?
        return self.icdf(x, 0.5, extrapolate=extrapolate)

    def mode(self, x: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        raise NotImplementedError

    def variance(self, x: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        raise NotImplementedError

    def standard_deviation(self, x: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        raise NotImplementedError

    def skewness(self, x: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        raise NotImplementedError

    def kurtosis(self, x: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        raise NotImplementedError

    # ruff: disable[FBT003]

    def pdf(self, x: npt.ArrayLike, v: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self._integrate("pdf", False, x, v, extrapolate=extrapolate)

    def logpdf(self, x: npt.ArrayLike, v: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self._integrate("logpdf", True, x, v, extrapolate=extrapolate)

    def cdf(self, x: npt.ArrayLike, v: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self._integrate("cdf", False, x, v, extrapolate=extrapolate)

    def icdf(self, x: npt.ArrayLike, q: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self._iroot("icdf", "cdf", False, x, q, extrapolate=extrapolate)

    def ccdf(self, x: npt.ArrayLike, v: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self._integrate("ccdf", False, x, v, extrapolate=extrapolate)

    def iccdf(self, x: npt.ArrayLike, q: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self._iroot("iccdf", "ccdf", False, x, q, extrapolate=extrapolate)

    def logcdf(self, x: npt.ArrayLike, v: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self._integrate("logcdf", True, x, v, extrapolate=extrapolate)

    def ilogcdf(self, x: npt.ArrayLike, logp: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self._iroot("ilogcdf", "logcdf", True, x, logp, extrapolate=extrapolate)

    def logccdf(self, x: npt.ArrayLike, v: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self._integrate("logccdf", True, x, v, extrapolate=extrapolate)

    def ilogccdf(self, x: npt.ArrayLike, logp: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self._iroot("ilogccdf", "logccdf", True, x, logp, extrapolate=extrapolate)

    def logentropy(self, x: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        raise NotImplementedError

    def entropy(self, x: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        raise NotImplementedError

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
        assert isinstance(self.model1, _BaseGAMLSSModel)
        return self.model1._param_names

    @abstractmethod
    def _interpolate_params(
        self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, /, *, extrapolate: bool
    ) -> tuple[_RealArrayLike, ...]:
        assert isinstance(self.model1, _BaseGAMLSSModel)
        assert isinstance(self.model2, _BaseGAMLSSModel)
        if self.model1._distr is not self.model2._distr:
            raise TypeError(f"(model1 ~ {self.model1._distr.name}) != (model2 ~ {self.model2._distr.name})")
        ## TODO: use `apply_where` when it accepts tuple output
        # return apply_where(cond, (x,), self.model1._interpolate_params, self.model2._interpolate_params, xp=np)
        params1 = self.model1._interpolate_params(x, extrapolate=extrapolate)
        params2 = self.model2._interpolate_params(x, extrapolate=extrapolate)
        return tuple(np.where(cond, p1, p2) for p1, p2 in zip(params1, params2, strict=True))

    def _interpolate_param_dict(
        self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, /, *, extrapolate: bool
    ) -> dict[str, _RealArrayLike]:
        return dict(zip(self._param_names, self._interpolate_params(cond, x, extrapolate=extrapolate), strict=True))

    def interpolate_distr(
        self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, /, *, extrapolate: bool = False
    ) -> stats.rv_continuous:
        assert isinstance(self.model1, _BaseGAMLSSModel)
        return self.model1._distr(**self._interpolate_param_dict(cond, x, extrapolate=extrapolate))

    def interpolate_rv(
        self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, /, *, extrapolate: bool = False
    ) -> "stats._distribution_infrastructure.ContinuousDistribution":
        assert isinstance(self.model1, _BaseGAMLSSModel)
        params = self._interpolate_param_dict(cond, x, extrapolate=extrapolate)
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

    def _apply(
        self, fn: str, cond: npt.NDArray[np.bool_], *args: npt.ArrayLike, extrapolate: bool, **kwargs
    ) -> npt.NDArray:
        return apply_where(
            cond,
            args,
            partial(getattr(self.model1, fn), extrapolate=extrapolate, **kwargs),
            partial(getattr(self.model2, fn), extrapolate=extrapolate, **kwargs),
            xp=np,
        )

    ## Convenience Methods

    def support(
        self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, /, *, extrapolate: bool = False
    ) -> tuple[npt.NDArray, npt.NDArray]:
        ## TODO: use `apply_where` when it accepts tuple output
        return self.interpolate_rv(cond, x, extrapolate=extrapolate).support()

    def sample(
        self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, /, shape=(), *, extrapolate: bool = False, rng=None
    ) -> npt.NDArray:
        return self._apply("sample", cond, x, shape=shape, extrapolate=extrapolate, rng=rng)

    def moment(
        self,
        cond: npt.NDArray[np.bool_],
        x: npt.ArrayLike,
        /,
        order: int = 1,
        kind: Literal["raw", "central", "standardized"] = "raw",
        *,
        extrapolate: bool = False,
    ) -> npt.NDArray:
        return self._apply("moment", cond, x, order=order, kind=kind, extrapolate=extrapolate)

    def mean(self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self._apply("mean", cond, x, extrapolate=extrapolate)

    def median(self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self._apply("median", cond, x, extrapolate=extrapolate)

    def mode(self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self._apply("mode", cond, x, extrapolate=extrapolate)

    def variance(self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self._apply("variance", cond, x, extrapolate=extrapolate)

    def standard_deviation(
        self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, /, *, extrapolate: bool = False
    ) -> npt.NDArray:
        return self._apply("standard_deviation", cond, x, extrapolate=extrapolate)

    def skewness(self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self._apply("skewness", cond, x, extrapolate=extrapolate)

    def kurtosis(self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self._apply("kurtosis", cond, x, extrapolate=extrapolate)

    def pdf(
        self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, v: npt.ArrayLike, /, *, extrapolate: bool = False
    ) -> npt.NDArray:
        return self._apply("pdf", cond, x, v, extrapolate=extrapolate)

    def logpdf(
        self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, v: npt.ArrayLike, /, *, extrapolate: bool = False
    ) -> npt.NDArray:
        return self._apply("logpdf", cond, x, v, extrapolate=extrapolate)

    def cdf(
        self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, v: npt.ArrayLike, /, *, extrapolate: bool = False
    ) -> npt.NDArray:
        return self._apply("cdf", cond, x, v, extrapolate=extrapolate)

    def icdf(
        self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, q: npt.ArrayLike, /, *, extrapolate: bool = False
    ) -> npt.NDArray:
        return self._apply("icdf", cond, x, q, extrapolate=extrapolate)

    def ccdf(
        self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, v: npt.ArrayLike, /, *, extrapolate: bool = False
    ) -> npt.NDArray:
        return self._apply("ccdf", cond, x, v, extrapolate=extrapolate)

    def iccdf(
        self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, q: npt.ArrayLike, /, *, extrapolate: bool = False
    ) -> npt.NDArray:
        return self._apply("iccdf", cond, x, q, extrapolate=extrapolate)

    def logcdf(
        self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, v: npt.ArrayLike, /, *, extrapolate: bool = False
    ) -> npt.NDArray:
        return self._apply("logcdf", cond, x, v, extrapolate=extrapolate)

    def ilogcdf(
        self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, logp: npt.ArrayLike, /, *, extrapolate: bool = False
    ) -> npt.NDArray:
        return self._apply("ilogcdf", cond, x, logp, extrapolate=extrapolate)

    def logccdf(
        self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, v: npt.ArrayLike, /, *, extrapolate: bool = False
    ) -> npt.NDArray:
        return self._apply("logccdf", cond, x, v, extrapolate=extrapolate)

    def ilogccdf(
        self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, logp: npt.ArrayLike, /, *, extrapolate: bool = False
    ) -> npt.NDArray:
        return self._apply("ilogccdf", cond, x, logp, extrapolate=extrapolate)

    def logentropy(self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self._apply("logentropy", cond, x, extrapolate=extrapolate)

    def entropy(self, cond: npt.NDArray[np.bool_], x: npt.ArrayLike, /, *, extrapolate: bool = False) -> npt.NDArray:
        return self._apply("entropy", cond, x, extrapolate=extrapolate)
