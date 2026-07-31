from functools import partial

import numpy as np
from array_api_extra import apply_where
from numpy.typing import ArrayLike
from scipy.special import log_ndtr, ndtr, ndtri, ndtri_exp
from scipy.stats._distribution_infrastructure import ContinuousDistribution

from growthstandards.gamlss_ext import GAMLSSModel, GAMLSSModelByCondition


def _ccdf_z_score(rv, *args: ArrayLike, log: bool, **kwargs):
    return (-ndtri_exp(rv.logccdf(*args, **kwargs))) if log else (-ndtri(rv.ccdf(*args, **kwargs)))


def _cdf_z_score(rv, *args: ArrayLike, log: bool, **kwargs):
    return ndtri_exp(rv.logcdf(*args, **kwargs)) if log else ndtri(rv.cdf(*args, **kwargs))


def _z_score(rv, *args: ArrayLike, log: bool, **kwargs):
    *rv_args, v = args

    return apply_where(
        v >= rv.median(*rv_args),
        args,
        partial(_ccdf_z_score, rv, log=log, **kwargs),
        partial(_cdf_z_score, rv, log=log, **kwargs),
        xp=np,
    )


def _iccdf_z_score(rv, *args: ArrayLike, log: bool, **kwargs):
    *rv_args, z = args
    return rv.ilogccdf(*rv_args, log_ndtr(-z), **kwargs) if log else rv.iccdf(*rv_args, ndtr(-z), **kwargs)


def _icdf_z_score(rv, *args: ArrayLike, log: bool, **kwargs):
    *rv_args, z = args
    return rv.ilogcdf(*rv_args, log_ndtr(z), **kwargs) if log else rv.icdf(*rv_args, ndtr(z), **kwargs)


def _inverse_z_score(rv, *args: ArrayLike, log: bool, **kwargs):
    *_, z = args

    return apply_where(
        z >= 0,
        args,
        partial(_iccdf_z_score, rv, log=log, **kwargs),
        partial(_icdf_z_score, rv, log=log, **kwargs),
        xp=np,
    )


def z_score(rv: ContinuousDistribution, v: ArrayLike, *, log: bool = True):
    """Generalized z-score

    References
    ----------
    .. [1] Rigby, R.A. and Stasinopoulos, D.M. (2004), Smooth centile curves for skew and kurtotic data modelled using the Box–Cox power exponential distribution. Statist. Med., 23: 3053-3076. https://doi.org/10.1002/sim.1861
    """
    return _z_score(rv, v, log=log)


def inverse_z_score(rv: ContinuousDistribution, z: ArrayLike, *, log: bool = True):
    """Generalized inverse z-score

    References
    ----------
    .. [1] Rigby, R.A. and Stasinopoulos, D.M. (2004), Smooth centile curves for skew and kurtotic data modelled using the Box–Cox power exponential distribution. Statist. Med., 23: 3053-3076. https://doi.org/10.1002/sim.1861
    """
    return _inverse_z_score(rv, z, log=log)


def gamlss_z_score(model: GAMLSSModel, x: ArrayLike, v: ArrayLike, *, log: bool = True, extrapolate: bool = False):
    """Generalized z-score

    References
    ----------
    .. [1] Rigby, R.A. and Stasinopoulos, D.M. (2004), Smooth centile curves for skew and kurtotic data modelled using the Box–Cox power exponential distribution. Statist. Med., 23: 3053-3076. https://doi.org/10.1002/sim.1861
    """
    return _z_score(model, x, v, log=log, extrapolate=extrapolate)


def gamlss_inverse_z_score(
    model: GAMLSSModel, x: ArrayLike, z: ArrayLike, *, log: bool = True, extrapolate: bool = False
):
    """Generalized inverse z-score

    References
    ----------
    .. [1] Rigby, R.A. and Stasinopoulos, D.M. (2004), Smooth centile curves for skew and kurtotic data modelled using the Box–Cox power exponential distribution. Statist. Med., 23: 3053-3076. https://doi.org/10.1002/sim.1861
    """
    return _inverse_z_score(model, x, z, log=log, extrapolate=extrapolate)


def gamlss_z_score_by_cond(
    model: GAMLSSModelByCondition,
    cond: ArrayLike,
    x: ArrayLike,
    v: ArrayLike,
    *,
    log: bool = True,
    extrapolate: bool = False,
):
    """Generalized z-score

    References
    ----------
    .. [1] Rigby, R.A. and Stasinopoulos, D.M. (2004), Smooth centile curves for skew and kurtotic data modelled using the Box–Cox power exponential distribution. Statist. Med., 23: 3053-3076. https://doi.org/10.1002/sim.1861
    """
    return _z_score(model, cond, x, v, log=log, extrapolate=extrapolate)


def gamlss_inverse_z_score_by_cond(
    model: GAMLSSModelByCondition,
    cond: ArrayLike,
    x: ArrayLike,
    z: ArrayLike,
    *,
    log: bool = True,
    extrapolate: bool = False,
):
    """Generalized inverse z-score

    References
    ----------
    .. [1] Rigby, R.A. and Stasinopoulos, D.M. (2004), Smooth centile curves for skew and kurtotic data modelled using the Box–Cox power exponential distribution. Statist. Med., 23: 3053-3076. https://doi.org/10.1002/sim.1861
    """
    return _inverse_z_score(model, cond, x, z, log=log, extrapolate=extrapolate)
