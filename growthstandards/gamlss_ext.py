from typing import Any
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import scipy.stats as stats
import xarray as xr
import xarray_einstats.stats as xr_stats

from .bcs_ext.scipy_ext import BCPE


@dataclass
class FractionalPolynomial:
    intercept: float
    coefficients: tuple[float, ...]
    fpowers: tuple[float, ...]
    domain: tuple[float, float]
    inv_scale: float = field(init=False, repr=False)

    def __post_init__(self):
        assert len(self.coefficients) == len(self.fpowers)
        x_min, x_max = self.domain
        self.inv_scale = 10 ** int(-np.trunc(np.log10(x_max - x_min)))

    def __call__(self, x: npt.ArrayLike):
        x = np.asanyarray(x)
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
class BCPEModel:
    mu: FractionalPolynomial
    sigma: float
    nu: float
    tau: float
    attrs: dict[str, Any] = field(default_factory=dict)

    def interpolate_distr(self, x: npt.ArrayLike) -> stats.rv_continuous:
        mu = self.mu(x)
        return BCPE(mu=mu, sigma=self.sigma, nu=self.nu, beta=self.tau)

    def interpolate_rv(self, x: npt.ArrayLike):
        mu = self.mu(x)
        return stats.make_distribution(BCPE)(mu=mu, sigma=self.sigma, nu=self.nu, beta=self.tau)

    def interpolate_xr_rv(self, x: xr.DataArray) -> xr_stats.XrContinuousRV:
        mu = xr.apply_ufunc(self.mu, x)
        rv = xr_stats.XrContinuousRV(BCPE, mu, self.sigma, self.nu, self.tau)
        return rv
