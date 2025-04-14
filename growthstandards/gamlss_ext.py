from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.stats as stats
import xarray as xr
import xarray_einstats.stats as xr_stats

from .bcs_ext.scipy_ext import BCPE


class GAMLSSModel(ABC):
    @abstractmethod
    def interpolate_distr(self, x: npt.ArrayLike, /) -> stats.rv_continuous: ...
    @abstractmethod
    def interpolate_rv(self, x: npt.ArrayLike, /) -> "stats._distribution_infrastructure.ContinuousDistribution": ...
    @abstractmethod
    def interpolate_xr_rv(self, x: xr.DataArray) -> xr_stats.XrContinuousRV: ...

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
class BCPEModel(GAMLSSModel):
    mu: FractionalPolynomial
    sigma: float
    nu: float
    tau: float
    attrs: dict[str, Any] = field(default_factory=dict)

    def interpolate_distr(self, x: npt.ArrayLike, /) -> stats.rv_continuous:
        mu = self.mu(x)
        return BCPE(mu=mu, sigma=self.sigma, nu=self.nu, beta=self.tau)

    def interpolate_rv(self, x: npt.ArrayLike, /) -> "stats._distribution_infrastructure.ContinuousDistribution":
        mu = self.mu(x)
        return stats.make_distribution(BCPE)(mu=mu, sigma=self.sigma, nu=self.nu, beta=self.tau)

    def interpolate_xr_rv(self, x: xr.DataArray, /) -> xr_stats.XrContinuousRV:
        mu = xr.apply_ufunc(self.mu, x)
        rv = xr_stats.XrContinuousRV(BCPE, mu, self.sigma, self.nu, self.tau)
        return rv
