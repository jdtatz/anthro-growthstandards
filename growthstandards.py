import scipy.stats as stats
import xarray as xr
import xarray_einstats.stats as xr_stats
import zarr

from bcs_ext.scipy_ext import BCCG

try:
    if "median" not in xr_stats.XrRV:
        # FIXME: upstream
        xr_stats._add_documented_method(xr_stats.XrRV, "rv_generic", ["median"], xr_stats.doc_extras)
except:
    pass



def rv_to_ds(rv: xr_stats.XrRV):
    return xr.Dataset({
        **{da.name: da for da in rv.args},
        **rv.kwargs
    })


def ds_to_rv(distr: stats.rv_continuous, ds: xr.Dataset):
    args = [ds[s.name] for s in distr._shape_info()]
    if "loc" in ds:
        args.append(ds["loc"])
    if "scale" in ds:
        args.append(ds["scale"])
    return xr_stats.XrContinuousRV(distr, *args)


def _xr_bcs_mul(rv: xr_stats.XrContinuousRV, c):
    distr = rv.dist
    ds = rv_to_ds(rv)
    ds["mu"] = ds["mu"] * c
    return ds_to_rv(distr, ds)



GROWTHSTANDARD_KEYS = (
    'arm_c',
    'bmi_height',
    'bmi_length',
    'head_c',
    'height',
    'length',
    'ss',
    'ts',
    'weight',
    'wfh',
    'wfl'
)


def load_growthstandard_dss() -> dict[str, xr.Dataset]:
    store = zarr.DirectoryStore("growthstandards.zarr")
    growthstandard_dss = {
        k: xr.open_zarr(store=store, group=k, decode_times=False).load()
        for k in GROWTHSTANDARD_KEYS
    }

    growthstandard_dss["len_hei"] = xr.combine_by_coords(
        [growthstandard_dss["length"], growthstandard_dss["height"]],
        combine_attrs="drop_conflicts"
    ).assign_attrs(long_name="Recumbent Length / Standing Height")

    growthstandard_dss["bmi"] = xr.combine_by_coords(
        [growthstandard_dss["bmi_length"], growthstandard_dss["bmi_height"]],
        combine_attrs="drop_conflicts"
    ).assign_attrs(long_name="Body Mass Index")
    return growthstandard_dss


def load_growthstandard_rvs() -> dict[str, xr_stats.XrContinuousRV]:
    growthstandard_dss = load_growthstandard_dss()

    growthstandard_rvs = {
        k: ds_to_rv(BCCG, gds.rename_vars({"m": "mu", "s": "sigma", "l": "nu"}))
        for k, gds in growthstandard_dss.items()
    }
    length_da = growthstandard_dss["wfl"].length
    height_da = growthstandard_dss["wfh"].height
    growthstandard_rvs["gfl"] = _xr_bcs_mul(growthstandard_rvs["wfl"], 1 / length_da)
    growthstandard_rvs["gfh"] = _xr_bcs_mul(growthstandard_rvs["wfh"], 1 / height_da)
    return growthstandard_rvs



