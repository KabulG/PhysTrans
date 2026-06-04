#!/usr/bin/env python3


import argparse
import os
import sys
from typing import Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def choose_var(ds: xr.Dataset, candidates=None) -> Optional[str]:
    if candidates is None:
        candidates = ["Rad", "albedo_01", "albedo_02"]
    for cand in candidates:
        if cand in ds:
            return cand
   
    for name, da in ds.data_vars.items():
        if da.dtype.kind in {"f", "i"} and da.ndim >= 2:
            return name
    return None


def squeeze_to_2d(da: xr.DataArray) -> xr.DataArray:
    while da.ndim > 2:
        dim0 = da.dims[0]
        da = da.isel({dim0: 0})
    return da


def summarize(arr: np.ndarray):
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    med = float(np.nanmedian(arr))
    cover = float(np.nanmean(arr > med))
    return mean, std, vmin, vmax, med, cover


def main():
    parser = argparse.ArgumentParser(description="Visualize a NetCDF cloud image and print stats.")
    parser.add_argument("--nc_path", required=True, help="Path to NetCDF file")
    parser.add_argument("--out_png", required=False, help="Where to save PNG (default: nc_path + .png)")
    parser.add_argument("--var", required=False, help="Variable name to plot (optional)")
    args = parser.parse_args()

    if not os.path.exists(args.nc_path):
        print(f"NC file not found: {args.nc_path}", file=sys.stderr)
        sys.exit(1)

    ds = xr.open_dataset(args.nc_path)
    var_name = args.var or choose_var(ds)
    if var_name is None:
        print("No suitable variable found in dataset.", file=sys.stderr)
        sys.exit(1)

    da = ds[var_name]
    da2d = squeeze_to_2d(da)
    data = da2d.values

    mean, std, vmin, vmax, med, cover = summarize(data)
    print(f"Variable: {var_name}")
    print(f"Shape: {data.shape}")
    print(f"mean={mean:.4f}, std={std:.4f}, min={vmin:.4f}, max={vmax:.4f}, median={med:.4f}")
    print(f"Cloud cover (>median) = {cover*100:.2f}%")

    out_path = args.out_png or (args.nc_path + ".png")
    plt.figure(figsize=(8, 6))
    im = plt.imshow(data, cmap="gray")
    plt.colorbar(im, shrink=0.8, label=var_name)
    plt.title(os.path.basename(args.nc_path))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved image to: {out_path}")


if __name__ == "__main__":
    main()
