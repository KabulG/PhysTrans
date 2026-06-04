#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import math
import argparse
from typing import Optional, Tuple, Sequence

import numpy as np
import xarray as xr
from PIL import Image



DEFAULT_SITE_LAT = 24.4
DEFAULT_SITE_LON = 103.4

DEFAULT_PATCH_SIZE = 16
DEFAULT_WINDOW_KM = 40.0

DEFAULT_CLOUD_HEIGHT_KM = 2.5     
DEFAULT_MAX_SHIFT_KM = 30.0       


DEFAULT_SZA_FREEZE_DEG = 90.0      
DEFAULT_SZA_TAPER_START = 85.0    


CLOUD_VAR_CANDIDATES = [
    "tbb_13", "TBB_13", "BT_13", "B13", "band13", "Band13",
    "tbb", "bt", "brightness_temperature",
]
EXCLUDE_VARS = {"SOZ", "SOA"} 


def _squeeze_time(da: xr.DataArray) -> xr.DataArray:
    
    for d in da.dims:
        if d.lower() in ("time", "t", "datetime"):
            return da.isel({d: 0})
    return da


def scalar_at(ds: xr.Dataset, var: str, y: int, x: int) -> float:
    
    da = _squeeze_time(ds[var])

    
    val = float(np.asarray(da.values)[y, x])

    
    sf = da.attrs.get("scale_factor", None)
    off = da.attrs.get("add_offset", None)
    if sf is None:
        sf = da.encoding.get("scale_factor", None)
    if off is None:
        off = da.encoding.get("add_offset", None)

    if (sf is not None or off is not None) and abs(val) > 360:
        sf = 1.0 if sf is None else float(sf)
        off = 0.0 if off is None else float(off)
        val = val * sf + off

   
    units = str(da.attrs.get("units", "")).lower()
    if "rad" in units:
        val = math.degrees(val)

    return val


def daylight_weight(sza_deg: float, freeze_deg: float, taper_start: float) -> float:
    
    if not np.isfinite(sza_deg):
        return 0.0
    if sza_deg >= freeze_deg:
        return 0.0
    if sza_deg > taper_start:
        
        return max(0.0, (freeze_deg - sza_deg) / (freeze_deg - taper_start))
    return 1.0


def pick_cloud_var(ds: xr.Dataset) -> str:
    
    for v in CLOUD_VAR_CANDIDATES:
        if v in ds.data_vars:
            return v

    
    for v in ds.data_vars:
        if v in EXCLUDE_VARS:
            continue
        da = ds[v]
        if da.ndim >= 2:
           
            if da.ndim == 2:
                return v
            if da.ndim == 3:
                return v

    raise ValueError("No suitable cloud image variable found. Please edit CLOUD_VAR_CANDIDATES.")


def get_lat_lon_1d(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
   
    lat_names = ["lat", "latitude", "LAT", "Latitude"]
    lon_names = ["lon", "longitude", "LON", "Longitude"]

    lat_da = None
    lon_da = None
    for n in lat_names:
        if n in ds:
            lat_da = ds[n]
            break
        if n in ds.coords:
            lat_da = ds.coords[n]
            break
    for n in lon_names:
        if n in ds:
            lon_da = ds[n]
            break
        if n in ds.coords:
            lon_da = ds.coords[n]
            break

    if lat_da is None or lon_da is None:
        raise ValueError("lat/lon coordinate arrays not found (expected variables/coords named lat/lon).")

    lats = np.asarray(lat_da.values)
    lons = np.asarray(lon_da.values)

    if lats.ndim != 1 or lons.ndim != 1:
        raise ValueError("This script expects 1D lat and 1D lon arrays (regular lat/lon grid).")

    if len(lats) < 2 or len(lons) < 2:
        raise ValueError("lat/lon arrays are too short to estimate resolution.")

    return lats, lons


def nearest_1d(arr: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(arr - value)))


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def extract_with_padding(a: np.ndarray, y0: int, y1: int, x0: int, x1: int, fill: float) -> np.ndarray:
    """Extract [y0:y1, x0:x1] with padding if out-of-bounds."""
    h, w = a.shape
    yy0, yy1 = max(0, y0), min(h, y1)
    xx0, xx1 = max(0, x0), min(w, x1)

    cut = a[yy0:yy1, xx0:xx1]
    top = yy0 - y0
    left = xx0 - x0
    bottom = y1 - yy1
    right = x1 - xx1

    if any(v > 0 for v in (top, bottom, left, right)):
        cut = np.pad(cut, ((top, bottom), (left, right)), mode="constant", constant_values=fill)
    return cut


def normalize_to_uint8(data: np.ndarray, var_name: str, da: Optional[xr.DataArray] = None) -> np.ndarray:
    """Normalize cloud image to uint8 for PNG."""
    # Fill NaN
    finite = np.isfinite(data)
    if not finite.any():
        data = np.zeros_like(data, dtype=np.float32)
    else:
        fill_val = float(np.nanmedian(data))
        data = np.nan_to_num(data, nan=fill_val)

    units = ""
    if da is not None:
        units = str(da.attrs.get("units", "")).lower()

    name_l = var_name.lower()

    
    if ("k" in units and ("tbb" in name_l or "bt" in name_l or "brightness_temperature" in name_l)) or ("tbb" in name_l or "bt" in name_l):
        t_low, t_high = 180.0, 330.0
        v = (t_high - data) / (t_high - t_low)  # invert
        v = np.clip(v, 0.0, 1.0)
        return (v * 255.0).astype(np.uint8)

   
    p1, p99 = np.percentile(data, [1, 99])
    if p99 <= p1:
        p1, p99 = float(np.min(data)), float(np.max(data) + 1e-6)
    v = (data - p1) / (p99 - p1)
    v = np.clip(v, 0.0, 1.0)
    return (v * 255.0).astype(np.uint8)


def process_one_file(
    nc_path: str,
    output_dir: str,
    site_lat: float,
    site_lon: float,
    patch_size: int,
    window_km: float,
    cloud_height_km: float,
    max_shift_km: float,
    sza_freeze_deg: float,
    sza_taper_start: float,
) -> None:
    ds = None
    try:
        
        try:
            ds = xr.open_dataset(nc_path, engine="netcdf4")
        except Exception:
            try:
                ds = xr.open_dataset(nc_path, engine="h5netcdf")
            except Exception:
                ds = xr.open_dataset(nc_path, engine="scipy")

       
        lats, lons = get_lat_lon_1d(ds)
        lat_idx = nearest_1d(lats, site_lat)
        lon_idx = nearest_1d(lons, site_lon)

       
        if "SOZ" in ds and "SOA" in ds:
            sza = scalar_at(ds, "SOZ", lat_idx, lon_idx)
            azi = scalar_at(ds, "SOA", lat_idx, lon_idx)
        else:
           
            sza = float("nan")
            azi = float("nan")

       
        if np.isfinite(sza):
            sza = abs(float(sza))  
        if np.isfinite(azi):
            azi = float(azi) % 360.0

        
        w_day = daylight_weight(sza, sza_freeze_deg, sza_taper_start)
        if w_day == 0.0:
            dist_shift = 0.0
        else:
            sza_clip = min(sza, 80.0)  
            dist_shift = cloud_height_km * math.tan(math.radians(sza_clip))
            dist_shift *= w_day
            dist_shift = clamp(dist_shift, -max_shift_km, max_shift_km)

       
        if np.isfinite(azi):
            shadow_azi = math.radians((azi + 180.0) % 360.0)
        else:
            shadow_azi = 0.0

        dx_km = dist_shift * math.sin(shadow_azi)  
        dy_km = dist_shift * math.cos(shadow_azi)  

        
        lat_step = float(lats[1] - lats[0]) 
        lon_step = float(lons[1] - lons[0])

        km_per_lat = 111.0
        km_per_lon = 111.0 * math.cos(math.radians(site_lat))

        shift_y = int(round(dy_km / (lat_step * km_per_lat)))
        shift_x = int(round(dx_km / (lon_step * km_per_lon)))

        center_y = int(lat_idx + shift_y)
        center_x = int(lon_idx + shift_x)

        
        half_y = max(1, int(round((window_km / 2.0) / (abs(lat_step) * km_per_lat))))
        half_x = max(1, int(round((window_km / 2.0) / (abs(lon_step) * km_per_lon))))

        
        cloud_var = pick_cloud_var(ds)
        cloud_da = _squeeze_time(ds[cloud_var])
        data2d = np.asarray(cloud_da.values)
        if data2d.ndim != 2:
            raise ValueError(f"Cloud var {cloud_var} is not 2D after squeezing time. dims={cloud_da.dims}")

        y0, y1 = center_y - half_y, center_y + half_y
        x0, x1 = center_x - half_x, center_x + half_x

        cut = extract_with_padding(data2d, y0, y1, x0, x1, fill=float(np.nanmedian(data2d[np.isfinite(data2d)]) if np.isfinite(data2d).any() else 0.0))

        img_arr = normalize_to_uint8(cut, cloud_var, cloud_da)

        img = Image.fromarray(img_arr)
        img = img.resize((patch_size, patch_size), Image.BICUBIC)

        
        fname_nc = os.path.basename(nc_path)
        fname = fname_nc.replace(".nc", ".png")
        day_str = None
        for token in fname_nc.split("_"):
            if token.isdigit() and len(token) == 8:
                day_str = token
                break
        sub_dir = os.path.join(output_dir, day_str) if day_str else output_dir
        os.makedirs(sub_dir, exist_ok=True)
        save_path = os.path.join(sub_dir, fname)
        img.save(save_path)

    except Exception as e:
        print(f"[ERROR] {os.path.basename(nc_path)}: {e}")
    finally:
        try:
            if ds is not None:
                ds.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Directory containing .nc files")
    ap.add_argument("--output_dir", required=True, help="Directory to write .png patches")
    ap.add_argument("--site_lat", type=float, default=DEFAULT_SITE_LAT)
    ap.add_argument("--site_lon", type=float, default=DEFAULT_SITE_LON)
    ap.add_argument("--patch_size", type=int, default=DEFAULT_PATCH_SIZE)
    ap.add_argument("--window_km", type=float, default=DEFAULT_WINDOW_KM)
    ap.add_argument("--cloud_height_km", type=float, default=DEFAULT_CLOUD_HEIGHT_KM)
    ap.add_argument("--max_shift_km", type=float, default=DEFAULT_MAX_SHIFT_KM)
    ap.add_argument("--sza_freeze_deg", type=float, default=DEFAULT_SZA_FREEZE_DEG)
    ap.add_argument("--sza_taper_start", type=float, default=DEFAULT_SZA_TAPER_START)
    ap.add_argument("--suffix", type=str, default=".nc", help="File suffix to scan (default: .nc)")
    args = ap.parse_args()

    nc_files = []
    for root, _, files in os.walk(args.input_dir):
        for f in files:
            if f.lower().endswith(args.suffix.lower()):
                nc_files.append(os.path.join(root, f))
    nc_files.sort()

    if not nc_files:
        raise SystemExit(f"No files with suffix {args.suffix} found under {args.input_dir}")

    for idx, p in enumerate(nc_files, 1):
        print(f"[{idx}/{len(nc_files)}] {os.path.basename(p)}")
        process_one_file(
            nc_path=p,
            output_dir=args.output_dir,
            site_lat=args.site_lat,
            site_lon=args.site_lon,
            patch_size=args.patch_size,
            window_km=args.window_km,
            cloud_height_km=args.cloud_height_km,
            max_shift_km=args.max_shift_km,
            sza_freeze_deg=args.sza_freeze_deg,
            sza_taper_start=args.sza_taper_start,
        )


if __name__ == "__main__":
    main()
