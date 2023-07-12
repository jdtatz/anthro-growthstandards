import importlib.resources
from collections.abc import Mapping
from functools import lru_cache
from typing import Iterator

import xarray as xr
import xarray_einstats.stats as xr_stats
import zarr

from .bcs_ext.scipy_ext import BCCG, BCPE
from .xr_stats_ext import ds_to_rv

GROWTHSTANDARD_KEYS = (
    "arm_c",
    "bmi_height",
    "bmi_length",
    "brain",
    "csf",
    "head_c",
    "height",
    "length",
    "ss",
    "ts",
    "weight",
    "wfh",
    "wfl",
)

GROWTHSTANDARD_NAMES = (*GROWTHSTANDARD_KEYS, "len_hei", "bmi", "gfl", "gfh")


@lru_cache
def load_growthstandard_ds(g: str) -> xr.Dataset:
    if g in GROWTHSTANDARD_KEYS:
        traversable = importlib.resources.files(__package__)
        store = zarr.DirectoryStore(traversable.joinpath("growthstandards.zarr"))
        return xr.open_zarr(store=store, group=g, decode_times=False).load()
    elif g == "len_hei":
        return xr.combine_by_coords(
            [load_growthstandard_ds("length"), load_growthstandard_ds("height")],
            combine_attrs="drop_conflicts",
        ).assign_attrs(long_name="Recumbent Length / Standing Height")
    elif g == "bmi":
        return xr.combine_by_coords(
            [
                load_growthstandard_ds("bmi_length"),
                load_growthstandard_ds("bmi_height"),
            ],
            combine_attrs="drop_conflicts",
        ).assign_attrs(long_name="Body Mass Index")
    elif g == "gfl":
        # Do not mutate
        gfl = load_growthstandard_ds("wfl").copy()
        gfl["m"] = gfl["m"] / gfl.coords["length"]
        return gfl.assign_attrs(long_name="Growth Metric (Recumbent Length)", units="kg/cm")
    elif g == "gfh":
        # Do not mutate
        gfh = load_growthstandard_ds("wfh").copy()
        gfh["m"] = gfh["m"] / gfh.coords["height"]
        return gfh.assign_attrs(long_name="Growth Metric (Standing Height)", units="kg/cm")
    else:
        raise KeyError(g)


class _GrowthStandards(Mapping):
    def __getitem__(self, key: str) -> xr_stats.XrContinuousRV:
        gds = load_growthstandard_ds(key)
        if key in ("brain", "csf"):
            return ds_to_rv(BCPE, gds.rename_vars({"tau": "beta"}))
        else:
            return ds_to_rv(BCCG, gds.rename_vars({"m": "mu", "s": "sigma", "l": "nu"}))

    def __iter__(self) -> Iterator:
        return iter(GROWTHSTANDARD_NAMES)

    def __len__(self) -> int:
        return len(GROWTHSTANDARD_NAMES)


GrowthStandards = _GrowthStandards()
