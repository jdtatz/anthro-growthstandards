from functools import wraps, update_wrapper, partial, partialmethod

import xarray as xr
import scipy.stats as stats

# from scipy.stats._distn_infrastructure import rv_generic


class _WrappedRVHelper:
    def __set_name__(self, owner, name):
        self.name = name
        self.wrapped = getattr(owner._kind, self.name)
        update_wrapper(self, self.wrapped)

    def __get__(self, instance, owner=None):
        if instance is None:
            assert owner is not None
            return update_wrapper(
                partialmethod(owner._xr_wrapped_rv_call, self.name), self.wrapped
            )
        return update_wrapper(
            partial(instance._xr_wrapped_rv_call, self.name), self.wrapped
        )


class XrContinuousRV:
    _kind = stats.rv_continuous

    def __init__(self, distr: stats.rv_continuous, *shape_params: xr.DataArray):
        self._distr = distr
        self._shape_params = shape_params

    @classmethod
    def from_ds(cls, distr: stats.rv_continuous, ds: xr.Dataset):
        shape_params = [ds[s.name] for s in distr._shape_info()]
        if "loc" in ds.keys():
            shape_params.append(ds["loc"])
        if "scale" in ds.keys():
            shape_params.append(ds["scale"])
        return cls(distr, *shape_params)

    def as_ds(self) -> xr.Dataset:
        keys = [s.name for s in self._distr._shape_info()] + ["loc", "scale"]
        return xr.Dataset(dict(zip(keys, self._shape_params)))

    def _xr_wrapped_rv_call(self, method: str, *args, **kwargs):
        assert "loc" not in kwargs
        assert "scale" not in kwargs
        return xr.apply_ufunc(
            getattr(self._distr, method), *args, *self._shape_params, kwargs=kwargs
        )

    @wraps(stats.rv_continuous.stats)
    def stats(self, *args, moments="mv", **kwargs):
        assert "loc" not in kwargs
        assert "scale" not in kwargs
        assert all(c in "mvsk" for c in moments)
        return xr.apply_ufunc(
            self._distr.stats,
            *args,
            *self._shape_params,
            output_core_dims=[() for _c in moments],
            kwargs={**kwargs, "moments": moments}
        )

    # pdf = update_wrapper(partialmethod(_xr_wrapped_rv_call, "pdf"), stats.rv_continuous.pdf)
    # pdf = partialmethod(update_wrapper(_xr_wrapped_rv_call, stats.rv_continuous.pdf), "pdf")
    # pdf = update_wrapper(_xr_wrapped_rv_call, stats.rv_continuous.pdf)

    rvs = _WrappedRVHelper()
    pdf = _WrappedRVHelper()
    logpdf = _WrappedRVHelper()
    cdf = _WrappedRVHelper()
    logcdf = _WrappedRVHelper()
    sf = _WrappedRVHelper()
    logsf = _WrappedRVHelper()
    ppf = _WrappedRVHelper()
    isf = _WrappedRVHelper()
    moment = _WrappedRVHelper()
    entropy = _WrappedRVHelper()
    # expect = _WrappedRVHelper()
    median = _WrappedRVHelper()
    mean = _WrappedRVHelper()
    std = _WrappedRVHelper()
    var = _WrappedRVHelper()
    interval = _WrappedRVHelper()
    support = _WrappedRVHelper()
