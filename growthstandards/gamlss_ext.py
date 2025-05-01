from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional, TypeAlias

import numpy as np
import numpy.typing as npt
import scipy.special
import scipy.stats as stats
import xarray as xr
import xarray_einstats.stats as xr_stats

from .bcs_ext.scipy_ext import BCPE

GAMLSSParam: TypeAlias = int | float | Callable[[npt.ArrayLike], npt.ArrayLike]


def _interpolate(x: npt.ArrayLike, p: GAMLSSParam):
    return p if isinstance(p, (int, float)) else p(x)


def _interpolate_tuple(x: npt.ArrayLike, *ps: GAMLSSParam):
    return tuple(_interpolate(x, p) for p in ps)


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
        return self._rv_type(**self._interpolate_param_dict(x))

    def interpolate_xr_rv(self, x: xr.DataArray) -> xr_stats.XrContinuousRV:
        params = xr.apply_ufunc(self._interpolate_params, x)
        rv = xr_stats.XrContinuousRV(self._distr, *params)
        return rv

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
