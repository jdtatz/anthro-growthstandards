from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from math import lcm
from typing import Protocol

import numpy as np
import numpy.typing as npt
import scipy
from scipy import interpolate


class CallableGAMLSSParam(Protocol):
    domain: tuple[int | float | Fraction, int | float | Fraction]

    def __call__(self, x: npt.ArrayLike, /) -> npt.ArrayLike: ...


type GAMLSSParam = int | float | CallableGAMLSSParam


def _interpolate(x: npt.ArrayLike, p: GAMLSSParam):
    return p if isinstance(p, (int, float)) else p(x)


def _param_domain(p: GAMLSSParam) -> None | tuple[int | float | Fraction, int | float | Fraction]:
    return None if isinstance(p, (int, float)) or not hasattr(p, "domain") else p.domain


class LinkFunction(ABC):
    def __init__(self, inner: GAMLSSParam):
        self.inner = inner

    def __repr__(self):
        name = type(self).__name__
        return f"{name}({self.inner!r})"

    @property
    def domain(self) -> None | tuple[int | float | Fraction, int | float | Fraction]:
        return _param_domain(self.inner)

    @abstractmethod
    def forward(self, value: npt.ArrayLike) -> npt.ArrayLike: ...

    def __call__(self, x: npt.ArrayLike, /):
        return self.forward(_interpolate(x, self.inner))


class LogLink(LinkFunction):
    def forward(self, value: npt.ArrayLike) -> npt.ArrayLike:
        return np.exp(value)


class LogitLink(LinkFunction):
    def forward(self, value: npt.ArrayLike) -> npt.ArrayLike:
        return scipy.special.expit(value)


@dataclass
class LookupTable:
    start: int | Fraction
    stop: int | Fraction
    step: int | Fraction
    fp: npt.ArrayLike
    xp: npt.ArrayLike = field(init=False, repr=False)

    def _parts(self):
        parts = (self.start, self.stop, self.step)
        if any(isinstance(v, Fraction) for v in parts):
            d = lcm(*(v.denominator for v in parts))
            assert all((d * v).is_integer() for v in parts)
            return *(int(d * v) for v in parts), 1, lambda v: Fraction(v, d) if isinstance(v, int) else v / d
        else:
            return *parts, self.step, lambda v: v

    def __post_init__(self):
        n, r = divmod(self.stop - self.start, self.step)
        if r != 0:
            raise ValueError("`stop - start` is not an integral multiple of `step`")
        start, stop, step, offset, apply_denom = self._parts()
        self.xp = apply_denom(np.arange(start, stop + offset, step))
        assert self.xp.shape == (1 + n,), f"{self.xp.shape} != {(1 + n,)}"

    @property
    def domain(self) -> tuple[int | Fraction, int | Fraction]:
        return self.start, self.stop

    def __call__(self, x: npt.ArrayLike, /):
        x = np.asanyarray(x)
        return np.interp(x, self.xp, self.fp)

    def __getitem__(self, key: slice):
        if not isinstance(key, slice):
            raise TypeError("LookupTable only supports subslicing")
        elif key.step is not None and key.step != 1:
            raise NotImplementedError("LookupTable only supports contiguous subslicing")
        start, stop, step, offset, apply_denom = self._parts()
        new_range = range(start, stop + offset, step)[key]
        res = LookupTable(
            start=apply_denom(new_range.start),
            stop=apply_denom(new_range.stop - offset),
            step=apply_denom(new_range.step),
            fp=self.fp[key],
        )
        assert np.all(self.xp[key] == res.xp)
        return res


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
