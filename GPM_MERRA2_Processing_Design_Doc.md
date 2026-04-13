# GPM DPR + MERRA-2 Data Processing Pipeline
## Design Documentation for Replicating Seela et al. (2026)
### *"Influence of atmospheric pollution on precipitation microphysics: Insights from GPM DPR analysis of clean vs. polluted events"*

---

> **Document Version:** 1.0  
> **Target Paper:** Seela et al., Urban Climate 65 (2026) 102778  
> **Study Region:** Taiwan (21–26°N, 119–123°E)  
> **Study Period:** 2014–2023  
> **Data Sources:** GPM DPR Level-2 Version 7 + MERRA-2 Version 5.12.4 AOD  

---

## Table of Contents

1. [Project Directory Structure](#1-project-directory-structure)
2. [Environment and Dependencies](#2-environment-and-dependencies)
3. [Data Inventory and Native Formats](#3-data-inventory-and-native-formats)
4. [GPM DPR — Internal Structure and Variable Extraction](#4-gpm-dpr--internal-structure-and-variable-extraction)
5. [MERRA-2 — Internal Structure and Variable Extraction](#5-merra-2--internal-structure-and-variable-extraction)
6. [Collocation Strategy](#6-collocation-strategy)
7. [Quality Control and Filtering](#7-quality-control-and-filtering)
8. [Clean vs. Polluted Classification](#8-clean-vs-polluted-classification)
9. [Height Reconstruction for Vertical Profiles](#9-height-reconstruction-for-vertical-profiles)
10. [Processed Data Schema and Storage Design](#10-processed-data-schema-and-storage-design)
11. [Per-Figure Processing Logic](#11-per-figure-processing-logic)
12. [Plotting Specifications](#12-plotting-specifications)
13. [Parallelization Strategy](#13-parallelization-strategy)
14. [Known Hurdles and Solutions](#14-known-hurdles-and-solutions)
15. [End-to-End Run Order](#15-end-to-end-run-order)
16. [Validation Checkpoints](#16-validation-checkpoints)

---

## 1. Project Directory Structure

Design the project as a flat, reproducible structure. Every script is numbered to enforce run order. Raw data is never modified.

```
project_root/
│
├── data/
│   ├── raw/
│   │   ├── gpm/                        # Original GPM HDF5 files (read-only)
│   │   │   └── YYYY/MM/DD/
│   │   │       └── 2A.GPM.DPR.V9-20211125.YYYYMMDD-SHHMMSS-EHHMMSS.ORBIT.V07A.HDF5
│   │   └── merra2/                     # Original MERRA-2 NetCDF4 files (read-only)
│   │       └── YYYY/
│   │           └── MERRA2_400.tavg1_2d_aer_Nx.YYYYMMDD.nc4
│   │
│   ├── interim/                        # Mid-pipeline outputs (can be deleted after final)
│   │   ├── merra2_aod_taiwan.nc        # Preprocessed, cropped MERRA-2 AOD
│   │   ├── gpm_pixels_clean.parquet    # Per-pixel near-surface data, clean
│   │   ├── gpm_pixels_polluted.parquet # Per-pixel near-surface data, polluted
│   │   ├── profiles_clean.zarr/        # Vertical profile data, clean events
│   │   └── profiles_polluted.zarr/     # Vertical profile data, polluted events
│   │
│   └── final/
│       ├── figure_data/                # Pre-computed arrays for each figure (.npz)
│       │   ├── fig1_pdf_data.npz
│       │   ├── fig2_2dhist_data.npz
│       │   ├── fig3_cfad_total.npz
│       │   ├── fig4_cfad_stratiform.npz
│       │   ├── fig5_cfad_convective.npz
│       │   ├── fig6_kde_echotopp.npz
│       │   ├── fig7_warmrain_delta.npz
│       │   └── fig8_thermodynamic.npz
│       └── plots/
│           ├── fig1_pdf.png
│           ├── fig2_2dhist.png
│           ├── fig3_cfad_total.png
│           ├── fig4_cfad_stratiform.png
│           ├── fig5_cfad_convective.png
│           ├── fig6_kde_echotop.png
│           ├── fig7_warmrain.png
│           └── fig8_boxplots.png
│
├── scripts/
│   ├── 00_check_environment.py         # Verify all packages installed and data accessible
│   ├── 01_preprocess_merra2.py         # Crop and consolidate MERRA-2 AOD
│   ├── 02_extract_gpm_pixels.py        # Extract near-surface pixels from GPM HDF5
│   ├── 03_extract_gpm_profiles.py      # Extract vertical profiles from GPM HDF5
│   ├── 04_collocate_and_classify.py    # Match GPM pixels to MERRA-2 AOD, assign class
│   ├── 05_compute_figure_data.py       # All intermediate computations for figures
│   ├── 06_plot_figures.py              # Generate all figures from figure_data/
│   └── utils/
│       ├── gpm_reader.py               # GPM HDF5 reading utilities
│       ├── merra2_reader.py            # MERRA-2 xarray utilities
│       ├── collocator.py               # Spatial/temporal matching logic
│       ├── cfad_utils.py               # CFAD generation
│       └── plot_utils.py               # Shared plotting helpers
│
├── config/
│   └── config.yaml                     # All paths, thresholds, constants in one place
│
├── logs/
│   └── pipeline_YYYYMMDD.log
│
├── notebooks/                          # Optional: exploratory analysis
│   └── explore_single_file.ipynb
│
└── requirements.txt
```

**Why this structure?**
- Raw data directories are read-only by convention — scripts never write there
- `interim/` is the only directory where scripts write during processing
- `final/figure_data/` decouples computation from plotting — you can regenerate plots without reprocessing data
- Every figure's data is saved as `.npz` before plotting, so plots can be tweaked instantly

---

## 2. Environment and Dependencies

### 2.1 Python Version

Use **Python 3.10+**. Avoid 3.12 for now — some scientific packages have edge-case issues there.

### 2.2 Recommended Installation

```bash
conda create -n gpm_env python=3.10
conda activate gpm_env

# Core scientific
conda install -c conda-forge numpy scipy pandas matplotlib

# Data I/O
conda install -c conda-forge h5py xarray netcdf4 zarr fastparquet pyarrow

# Parallel processing
conda install -c conda-forge dask distributed

# Geospatial (for Taiwan bbox clipping)
conda install -c conda-forge pyproj

# Optional but useful
conda install -c conda-forge tqdm pyyaml jupyter
```

### 2.3 requirements.txt

```
numpy>=1.24
scipy>=1.10
pandas>=2.0
matplotlib>=3.7
h5py>=3.8
xarray>=2023.1
netCDF4>=1.6
zarr>=2.14
fastparquet>=2023.4
pyarrow>=12.0
dask>=2023.5
distributed>=2023.5
pyproj>=3.5
tqdm>=4.65
pyyaml>=6.0
```

### 2.4 config.yaml

Centralise every threshold and path here. Scripts import this file — never hardcode paths in scripts.

```yaml
paths:
  gpm_raw:      "data/raw/gpm"
  merra2_raw:   "data/raw/merra2"
  interim:      "data/interim"
  final:        "data/final"
  logs:         "logs"

taiwan_bbox:
  lat_min: 21.0
  lat_max: 26.0
  lon_min: 119.0
  lon_max: 123.0

collocation:
  time_window_minutes: 30
  spatial_tolerance_deg: 0.625    # MERRA-2 longitude resolution

classification:
  aod_clean_threshold:    0.1
  aod_polluted_threshold: 0.5

gpm:
  quality_flag_value: 1           # keep only pixels where flagPrecip == 1
  range_bin_size_km:  0.125       # 125 m per bin for NS scan
  n_range_bins:       88
  surface_bin_offset: 5           # bins above surface to use as "near surface" (~2 km proxy — verify with actual data)
  precip_type_convective:  [2, 20, 21, 25]   # typePrecip values meaning convective
  precip_type_stratiform:  [1, 10, 11, 14]   # typePrecip values meaning stratiform
  # NOTE: verify exact integer values from DPR ATBD V07 document

cfad:
  height_min_km:  2.0
  height_max_km: 14.0
  height_bin_km:  0.5
  param_bins:
    R:   [-1.0, 1.6, 0.05]      # log10R, [min, max, step]
    Z:   [10, 50, 1.0]           # dBZ
    Dm:  [0.5, 2.5, 0.05]        # mm
    Nw:  [20, 45, 0.5]           # dBNw

warm_rain:
  lower_km: 2.0
  upper_km: 3.0

plotting:
  dpi: 300
  figsize_single: [6, 5]
  figsize_triple: [14, 5]
  color_clean:    "#2ca02c"      # green
  color_polluted: "#1f77b4"      # blue
  cmap_cfad:      "jet"
  cmap_2dhist:    "jet"
```

---

## 3. Data Inventory and Native Formats

### 3.1 GPM DPR Files

| Property | Value |
|---|---|
| Format | HDF5 (`.HDF5`) |
| Product | 2A.DPR, Level-2, Version 7 (V07A) |
| Coverage | Global 65°S–65°N |
| Temporal | One file per orbit (~98 min) |
| Spatial | ~5 km horizontal footprint, 125 m vertical bins |
| File size | ~200–400 MB per orbit |
| Access | NASA GES DISC: `https://disc.gsfc.nasa.gov/datasets/GPM_2ADPR_07/` |
| Download tool | `wget` with `.netrc` credentials or `earthaccess` Python library |

**File naming convention:**
```
2A.GPM.DPR.V9-20211125.20200615-S013002-E030235.035694.V07A.HDF5
                         |date    |start  |end     |orbit
```

### 3.2 MERRA-2 Files

| Property | Value |
|---|---|
| Format | NetCDF4 (`.nc4`) |
| Collection | `tavg1_2d_aer_Nx` (M2T1NXAER) |
| Variable needed | `TOTEXTTAU` — Total Aerosol Extinction AOD at 550 nm |
| Temporal | 1-hourly, time-averaged from 00:30 UTC |
| Spatial | 0.5° lat × 0.625° lon, global |
| File size | ~476 MB per daily file |
| Access | NASA GES DISC: `https://disc.gsfc.nasa.gov/datasets/M2T1NXAER_5.12.4` |

**File naming convention:**
```
MERRA2_400.tavg1_2d_aer_Nx.20200615.nc4
           |collection      |date
```

**Note on MERRA-2 version numbers in filename:**
- `MERRA2_100` → 1980–1991
- `MERRA2_200` → 1992–2000
- `MERRA2_300` → 2001–2010
- `MERRA2_400` → 2011–present

Your 2014–2023 data will all be `MERRA2_400`.

---

## 4. GPM DPR — Internal Structure and Variable Extraction

### 4.1 HDF5 Internal Layout

The DPR Level-2 file contains multiple scan types. You want the **NS (Normal Scan)** group, which uses Ku-band only and has the widest swath (245 km).

```
Root/
├── NS/                          ← USE THIS — Normal Scan (Ku-band, full swath)
│   ├── Latitude                 shape: (nray, nscan)
│   ├── Longitude                shape: (nray, nscan)
│   ├── PRE/                     ← Pre-processing group
│   │   ├── flagPrecip           shape: (nray, nscan)       — quality flag
│   │   ├── binRealSurface       shape: (nray, nscan)       — surface bin index
│   │   ├── heightZeroDeg        shape: (nray, nscan)       — freezing level (m)
│   │   └── localZenithAngle     shape: (nray, nscan)
│   ├── SLV/                     ← Single-frequency algorithm outputs
│   │   ├── precipRate           shape: (nray, nscan, nbin) — R vertical profile
│   │   ├── precipRateNearSurface shape: (nray, nscan)      — R at 2 km
│   │   ├── zFactorCorrected     shape: (nray, nscan, nbin) — Z vertical profile
│   │   ├── zFactorCorrectedNearSurface shape: (nray, nscan) — Z at 2 km
│   │   └── paramDSD             shape: (nray, nscan, nbin, 2)
│   │                                   index 0 = Dm (mm)
│   │                                   index 1 = log10(Nw) ← NOTE: already log10
│   ├── CSF/                     ← Classification group
│   │   ├── typePrecip           shape: (nray, nscan)       — precip type flag
│   │   ├── heightStormTop       shape: (nray, nscan)       — storm top height (m)
│   │   └── piaFinal             shape: (nray, nscan)       — path integrated attenuation
│   └── VER/                     ← Verification group
│       └── binEchoTop           shape: (nray, nscan)       — echo top bin index
│
├── MS/                          ← Match Scan (dual-freq) — not needed for this paper
└── HS/                          ← High Sensitivity Scan (Ka-band) — not needed
```

**Dimensions for NS scan:**
- `nray` = 49 (cross-track rays per scan)
- `nscan` = varies per orbit file (~7000–9000 scans)
- `nbin` = 88 (range bins, 125 m each, top-to-bottom)

### 4.2 typePrecip Decoding

`typePrecip` is a packed integer. The tens digit encodes precipitation type:

```python
precip_type_major = (typePrecip // 10000000).astype(int)
# 0 = no precipitation
# 1 = stratiform
# 2 = convective
# 3 = other (hail, shallow, etc.) ← EXCLUDE these per the paper
```

More precisely, use:
```python
precip_type_major = typePrecip // 10000000
mask_convective = (precip_type_major == 2)
mask_stratiform = (precip_type_major == 1)
mask_valid_type = mask_convective | mask_stratiform
```

**Always verify this with the V07 ATBD document for your specific version.**

### 4.3 Near-Surface Variable Extraction (for PDFs, Fig 1–2)

```python
import h5py
import numpy as np

def extract_near_surface(filepath, taiwan_bbox):
    with h5py.File(filepath, 'r') as f:
        lat = f['NS/Latitude'][:]              # (nray, nscan)
        lon = f['NS/Longitude'][:]
        
        # Spatial filter — Taiwan only
        mask_geo = (
            (lat >= taiwan_bbox['lat_min']) & (lat <= taiwan_bbox['lat_max']) &
            (lon >= taiwan_bbox['lon_min']) & (lon <= taiwan_bbox['lon_max'])
        )
        
        if mask_geo.sum() == 0:
            return None  # This orbit doesn't cover Taiwan
        
        flag    = f['NS/PRE/flagPrecip'][:]    # (nray, nscan)
        ttype   = f['NS/CSF/typePrecip'][:]    # (nray, nscan)
        R_ns    = f['NS/SLV/precipRateNearSurface'][:]
        Z_ns    = f['NS/SLV/zFactorCorrectedNearSurface'][:]
        dsd     = f['NS/SLV/paramDSD'][:]      # (nray, nscan, nbin, 2)
        htop    = f['NS/CSF/heightStormTop'][:]
        
        # Near-surface DSD (2 km ~ bin index 2 from surface, but use NearSurface product directly)
        Dm_ns   = dsd[:, :, -6, 0]   # ~2 km from surface — see Section 9 for exact bin calculation
        Nw_ns   = dsd[:, :, -6, 1]   # log10(Nw) — already log, convert to dBNw = 10 * log10(Nw)
        
        # Quality + type mask
        mask_quality = (flag == 1)
        precip_major = ttype // 10000000
        mask_valid   = (precip_major == 1) | (precip_major == 2)
        
        final_mask = mask_geo & mask_quality & mask_valid
        
        # Extract scan timestamp (get from metadata)
        # GPM DPR stores scan time in NS/ScanTime group
        year   = f['NS/ScanTime/Year'][:]       # (nscan,)
        month  = f['NS/ScanTime/Month'][:]
        day    = f['NS/ScanTime/DayOfMonth'][:]
        hour   = f['NS/ScanTime/Hour'][:]
        minute = f['NS/ScanTime/Minute'][:]
        second = f['NS/ScanTime/Second'][:]
        
        # Broadcast to (nray, nscan) — same time for all rays in a scan
        year_2d = np.broadcast_to(year[np.newaxis, :], lat.shape)
        # ... repeat for all time components
        
        return {
            'lat':       lat[final_mask],
            'lon':       lon[final_mask],
            'R_ns':      R_ns[final_mask],
            'Z_ns':      Z_ns[final_mask],
            'Dm_ns':     Dm_ns[final_mask],
            'Nw_ns':     Nw_ns[final_mask],    # log10(Nw)
            'dBNw_ns':   10 * Nw_ns[final_mask], # dBNw = 10*log10(Nw)
            'htop':      htop[final_mask],
            'precip_type': precip_major[final_mask],
            # timestamp as pandas Timestamp — one per pixel
        }
```

**Critical note on fill values:** GPM uses `-9999.9` as fill value for most float fields and `-9999` for integers. Always mask these:

```python
R_ns = np.where(R_ns < 0, np.nan, R_ns)
Z_ns = np.where(Z_ns < 0, np.nan, Z_ns)
Dm   = np.where((dsd[:,:,:,0] < 0) | (dsd[:,:,:,0] > 10), np.nan, dsd[:,:,:,0])
Nw   = np.where((dsd[:,:,:,1] < -10) | (dsd[:,:,:,1] > 10), np.nan, dsd[:,:,:,1])
```

### 4.4 Vertical Profile Extraction (for CFADs, Fig 3–5)

Profiles are heavier — extract only pixels that pass quality + geo filter.

```python
def extract_profiles(filepath, taiwan_bbox):
    with h5py.File(filepath, 'r') as f:
        lat = f['NS/Latitude'][:]
        lon = f['NS/Longitude'][:]
        
        mask_geo     = (lat >= ...) & (lat <= ...) & (lon >= ...) & (lon <= ...)
        flag         = f['NS/PRE/flagPrecip'][:]
        ttype        = f['NS/CSF/typePrecip'][:]
        precip_major = ttype // 10000000
        mask_valid   = (precip_major == 1) | (precip_major == 2)
        final_mask   = mask_geo & (flag == 1) & mask_valid
        
        if final_mask.sum() == 0:
            return None
        
        # Full profiles — only for filtered pixels
        # Shape: (nray, nscan, nbin) → filter to (n_valid_pixels, nbin)
        R_prof  = f['NS/SLV/precipRate'][:]        [final_mask, :]
        Z_prof  = f['NS/SLV/zFactorCorrected'][:] [final_mask, :]
        dsd_all = f['NS/SLV/paramDSD'][:]          [final_mask, :, :]
        Dm_prof = dsd_all[:, :, 0]   # (n_pixels, nbin)
        Nw_prof = dsd_all[:, :, 1]   # (n_pixels, nbin) — log10(Nw)
        
        # Also need surface bin to compute height
        binSurf = f['NS/PRE/binRealSurface'][:] [final_mask]
        
        return {
            'R':          R_prof,
            'Z':          Z_prof,
            'Dm':         Dm_prof,
            'Nw':         Nw_prof,
            'dBNw':       10 * Nw_prof,
            'binSurface': binSurf,
            'precip_type': precip_major[final_mask],
            'lat':        lat[final_mask],
            'lon':        lon[final_mask],
        }
```

---

## 5. MERRA-2 — Internal Structure and Variable Extraction

### 5.1 Preprocessing — Build a Single Consolidated File

Processing 365 × 10 = 3650 daily files every time you run is slow. Instead, preprocess once:

```python
# script: 01_preprocess_merra2.py
import xarray as xr
import glob

files = sorted(glob.glob("data/raw/merra2/MERRA2_400.tavg1_2d_aer_Nx.*.nc4"))

ds = xr.open_mfdataset(
    files,
    combine='by_coords',
    engine='netcdf4',
    data_vars=['TOTEXTTAU'],          # load only what we need
    chunks={'time': 24, 'lat': 91, 'lon': 144}  # dask chunking
)

# Crop to Taiwan + generous buffer for interpolation
taiwan = ds.sel(
    lat=slice(20.0, 27.0),
    lon=slice(118.0, 124.0)
)

taiwan[['TOTEXTTAU']].to_netcdf("data/interim/merra2_aod_taiwan.nc")
```

**Result:** A ~2 GB file instead of 3650 files. Loads in seconds. All subsequent scripts read only this file.

### 5.2 MERRA-2 Time Coordinate Warning

MERRA-2 `tavg1` files have time stamps at `00:30, 01:30, ..., 23:30 UTC` — these are the *center* of 1-hour averaging windows, not the start. Keep this in mind during collocation: the 00:30 UTC stamp represents the average over 00:00–01:00 UTC.

```python
# After opening, verify:
print(ds.time.values[:3])
# Should show: 2014-01-01T00:30:00, 2014-01-01T01:30:00, ...
```

### 5.3 Coordinate Names

MERRA-2 uses lowercase `lat` and `lon`. GPM uses `Latitude` and `Longitude`. Be consistent in your lookups:

```python
aod_ds = xr.open_dataset("data/interim/merra2_aod_taiwan.nc")
# Access: aod_ds['TOTEXTTAU'] with dims (time, lat, lon)
```

---

## 6. Collocation Strategy

### 6.1 Temporal Matching

For each GPM pixel, find the closest MERRA-2 hourly time step within ±30 minutes:

```python
import pandas as pd
import numpy as np

def get_merra2_aod_for_pixel(pixel_time, pixel_lat, pixel_lon, aod_ds):
    """
    pixel_time: pandas Timestamp (UTC)
    pixel_lat, pixel_lon: float
    aod_ds: xarray Dataset with TOTEXTTAU(time, lat, lon)
    """
    # Find nearest MERRA-2 time within ±30 min
    time_diffs = abs(aod_ds.time.values - np.datetime64(pixel_time))
    nearest_idx = time_diffs.argmin()
    nearest_time = aod_ds.time.values[nearest_idx]
    
    # Enforce ±30 min window
    diff_minutes = abs((pd.Timestamp(nearest_time) - pixel_time).total_seconds()) / 60
    if diff_minutes > 30:
        return np.nan
    
    # Spatial nearest-neighbor (MERRA-2 grid is coarse, nearest is fine)
    aod_val = aod_ds['TOTEXTTAU'].sel(
        time=nearest_time,
        lat=pixel_lat,
        lon=pixel_lon,
        method='nearest'
    ).values
    
    return float(aod_val)
```

### 6.2 Batch Vectorized Collocation (Fast)

Doing pixel-by-pixel lookups is extremely slow. Vectorize:

```python
def collocate_batch(pixel_times, pixel_lats, pixel_lons, aod_ds):
    """
    Vectorized collocation for a batch of pixels from one GPM orbit.
    All pixels in one orbit share approximately the same overpass time,
    so we only need to find one or a few MERRA-2 time steps.
    """
    # For a single orbit, all pixels share the same time within ~90 min
    # Find the unique MERRA-2 time steps that cover this orbit's time range
    t_min = pd.Timestamp(pixel_times.min())
    t_max = pd.Timestamp(pixel_times.max())
    
    # Subset MERRA-2 to only relevant time window
    aod_window = aod_ds.sel(
        time=slice(
            t_min - pd.Timedelta('30min'),
            t_max + pd.Timedelta('30min')
        )
    )
    
    # Interpolate spatially to all pixel locations at once using xarray
    # Build DataArrays of target coordinates
    import xarray as xr
    target_lats = xr.DataArray(pixel_lats, dims='pixel')
    target_lons = xr.DataArray(pixel_lons, dims='pixel')
    
    # For each pixel, find nearest MERRA-2 time
    aod_values = np.full(len(pixel_times), np.nan)
    
    for i, (pt, plat, plon) in enumerate(zip(pixel_times, pixel_lats, pixel_lons)):
        pt_ts = pd.Timestamp(pt)
        time_diffs = np.abs(
            (aod_window.time.values - pt_ts.to_datetime64()) / np.timedelta64(1, 'm')
        )
        if time_diffs.min() <= 30:
            best_t = aod_window.time.values[time_diffs.argmin()]
            aod_values[i] = float(
                aod_window['TOTEXTTAU'].sel(
                    time=best_t, lat=plat, lon=plon, method='nearest'
                )
            )
    
    return aod_values
```

**For maximum speed:** If you have hundreds of orbits, use `scipy.interpolate.RegularGridInterpolator` on the MERRA-2 grid, which is much faster than repeated xarray `.sel()` calls.

```python
from scipy.interpolate import RegularGridInterpolator

def build_merra2_interpolator(aod_ds, target_time):
    """Build a fast 2D interpolator for a specific MERRA-2 time step."""
    aod_slice = aod_ds['TOTEXTTAU'].sel(time=target_time, method='nearest').values
    lats = aod_ds.lat.values
    lons = aod_ds.lon.values
    interp = RegularGridInterpolator(
        (lats, lons), aod_slice,
        method='linear', bounds_error=False, fill_value=np.nan
    )
    return interp

# Then for a batch of pixels at the same time:
interp = build_merra2_interpolator(aod_ds, overpass_time)
aod_values = interp(np.column_stack([pixel_lats, pixel_lons]))
```

---

## 7. Quality Control and Filtering

Apply these filters in strict order. Each filter is logged so you know how many pixels each step removes.

```
Filter 1: Geographic — Taiwan bbox (21–26°N, 119–123°E)
Filter 2: Quality    — flagPrecip == 1 (NASA quality flag)
Filter 3: Type       — typePrecip major class == 1 (stratiform) or 2 (convective)
                       EXCLUDE major class 3 ("other")
Filter 4: Fill value — R > 0, Z > 0, Dm > 0, Dm < 10, Nw finite
Filter 5: AOD valid  — MERRA-2 AOD is not NaN (collocation succeeded)
Filter 6: Class      — AOD < 0.1 (clean) OR AOD > 0.5 (polluted)
                       DROP intermediate (0.1 ≤ AOD ≤ 0.5)
```

Log pixel counts at each stage:

```python
import logging

logger = logging.getLogger(__name__)

def apply_filters_and_log(data, filename):
    n0 = len(data)
    logger.info(f"{filename}: {n0} pixels after geo filter")
    
    data = data[data['flagPrecip'] == 1]
    logger.info(f"  After quality flag: {len(data)} ({len(data)/n0*100:.1f}%)")
    
    data = data[data['precip_major'].isin([1, 2])]
    logger.info(f"  After type filter: {len(data)}")
    
    data = data.dropna(subset=['R_ns', 'Z_ns', 'Dm_ns', 'Nw_ns', 'AOD'])
    logger.info(f"  After fill value: {len(data)}")
    
    data = data[data['AOD'] < 0] = np.nan   # should not happen
    data_classified = data[(data['AOD'] < 0.1) | (data['AOD'] > 0.5)]
    logger.info(f"  After AOD classification filter: {len(data_classified)}")
    
    return data_classified
```

---

## 8. Clean vs. Polluted Classification

The classification is at the **overpass level**, not pixel level. One GPM overpass = one AOD value (the spatial median of all MERRA-2 pixels within the Taiwan domain at that time).

```python
def classify_overpass(pixel_aod_values, config):
    """
    Given AOD values for all pixels in one overpass,
    compute the representative overpass-level AOD and classify.
    """
    # Use median AOD across Taiwan domain for this overpass
    overpass_aod = np.nanmedian(pixel_aod_values)
    
    if overpass_aod < config['classification']['aod_clean_threshold']:
        return 'clean', overpass_aod
    elif overpass_aod > config['classification']['aod_polluted_threshold']:
        return 'polluted', overpass_aod
    else:
        return 'intermediate', overpass_aod   # excluded
```

**This gives you ~96 clean and ~114 polluted overpasses as in the paper.**

Each pixel in the overpass inherits the overpass classification. This means all pixels from a "clean" overpass are labeled clean, regardless of local AOD variation — consistent with the paper's approach.

---

## 9. Height Reconstruction for Vertical Profiles

### 9.1 GPM DPR Ranging

The DPR NS scan has 88 range bins of 125 m each. Bin index 0 is at the top (farthest from surface), bin 87 is closest to the surface (but may be below ground for elevated terrain).

```
Bin 0   → highest altitude   (~17+ km)
Bin 87  → nearest to surface (may be clutter-contaminated)
```

### 9.2 Height of Each Bin

```python
def compute_bin_heights(binSurface, elevation_m, n_bins=88, bin_size_km=0.125):
    """
    Compute altitude (km) for each range bin.
    
    binSurface: integer index of the surface bin (per pixel)
    elevation_m: surface elevation in meters (from DEM or GPM's own elevation field)
    
    Returns: array of shape (n_bins,) with heights in km
    """
    surface_km = elevation_m / 1000.0
    
    # Height increases as bin index decreases (bin 0 = top)
    bin_heights = np.zeros(n_bins)
    for i in range(n_bins):
        bins_above_surface = binSurface - i
        bin_heights[i] = surface_km + bins_above_surface * bin_size_km
    
    return bin_heights
```

**Practical note:** For CFADs, bin heights vary pixel-to-pixel due to terrain. You need to compute heights per pixel and then map each bin's value to a common altitude grid (e.g., 0.5 km intervals from 2 to 14 km) by interpolation or nearest-bin assignment.

### 9.3 Identifying the 2 km and 3 km Levels

For warm rain analysis (Fig. 7), you need Dm and Z at exactly 2 km and 3 km:

```python
def get_value_at_altitude(profile, bin_heights, target_km, tolerance_km=0.125):
    """
    Extract profile value at target altitude using nearest-bin approach.
    """
    diffs = np.abs(bin_heights - target_km)
    nearest_bin = np.argmin(diffs)
    if diffs[nearest_bin] > tolerance_km:
        return np.nan   # no bin close enough
    return profile[nearest_bin]
```

### 9.4 Freezing Level and Warm Rain Region

The paper's warm rain analysis is in the region below the melting layer. Use `heightZeroDeg` from GPM:

```python
freezing_level_km = f['NS/PRE/heightZeroDeg'][:] / 1000.0
# Warm rain region: surface up to freezing level
# Specifically, the paper uses 2 km vs 3 km within this region
```

---

## 10. Processed Data Schema and Storage Design

### 10.1 Near-Surface Pixel Data → Parquet

Stored as two files: `gpm_pixels_clean.parquet` and `gpm_pixels_polluted.parquet`.

**Schema (one row per pixel):**

| Column | Type | Description |
|---|---|---|
| `lat` | float32 | Pixel latitude |
| `lon` | float32 | Pixel longitude |
| `timestamp` | datetime64 | Scan timestamp (UTC) |
| `orbit_id` | int32 | GPM orbit number |
| `R_ns` | float32 | Near-surface rain rate (mm/h) |
| `logR_ns` | float32 | log10(R_ns) — precomputed |
| `Z_ns` | float32 | Near-surface reflectivity (dBZ) |
| `Dm_ns` | float32 | Near-surface Dm (mm) |
| `Nw_ns` | float32 | log10(Nw) — raw from DPR |
| `dBNw_ns` | float32 | 10*log10(Nw) — dBNw |
| `htop_m` | float32 | Storm top height (m) |
| `h30dBZ_m` | float32 | 30 dBZ echo top height (m) |
| `precip_type` | int8 | 1=stratiform, 2=convective |
| `AOD` | float32 | Collocated MERRA-2 TOTEXTTAU |
| `class` | category | 'clean' or 'polluted' |
| `Dm_2km` | float32 | Dm at 2 km (for Fig 7) |
| `Dm_3km` | float32 | Dm at 3 km (for Fig 7) |
| `Z_2km` | float32 | Z at 2 km (for Fig 7) |
| `Z_3km` | float32 | Z at 3 km (for Fig 7) |

**Why Parquet?**
- Columnar: loading only `R_ns` + `class` is instant — doesn't read other columns
- Compressed by default (snappy): 5–10× size reduction
- pandas/dask native: `pd.read_parquet()` in one line
- Preserves dtypes

```python
import pandas as pd

df_clean = pd.DataFrame(pixel_data_clean)
df_clean.to_parquet("data/interim/gpm_pixels_clean.parquet", engine='pyarrow', compression='snappy')

# Loading later:
df = pd.read_parquet("data/interim/gpm_pixels_clean.parquet", columns=['R_ns', 'logR_ns', 'Dm_ns'])
```

### 10.2 Vertical Profile Data → Zarr

Zarr is the best choice for chunked n-dimensional array storage. CFADs need fast access across all pixels at a specific height level.

**Store layout:**

```
profiles_clean.zarr/
├── R/              # float32, shape: (n_pixels, 88)
├── Z/              # float32, shape: (n_pixels, 88)
├── Dm/             # float32, shape: (n_pixels, 88)
├── dBNw/           # float32, shape: (n_pixels, 88)
├── bin_heights/    # float32, shape: (n_pixels, 88) — altitude for each bin
├── precip_type/    # int8,    shape: (n_pixels,)
└── orbit_id/       # int32,   shape: (n_pixels,)
```

**Chunking strategy:**

```python
import zarr
import numpy as np

store = zarr.open("data/interim/profiles_clean.zarr", mode='w')

# Chunk along pixels for height-slice access (CFAD computation)
# Each chunk = 1000 pixels × 88 bins
store.create_dataset('R',    shape=(n_pixels, 88), chunks=(1000, 88), dtype='float32')
store.create_dataset('Z',    shape=(n_pixels, 88), chunks=(1000, 88), dtype='float32')
store.create_dataset('Dm',   shape=(n_pixels, 88), chunks=(1000, 88), dtype='float32')
store.create_dataset('dBNw', shape=(n_pixels, 88), chunks=(1000, 88), dtype='float32')
store.create_dataset('bin_heights', shape=(n_pixels, 88), chunks=(1000, 88), dtype='float32')
store.create_dataset('precip_type', shape=(n_pixels,), chunks=(10000,), dtype='int8')
```

### 10.3 Figure Data → NumPy NPZ

Before plotting, all computation results are saved to `.npz`. This separates heavy computation (run once) from plotting (run many times for tweaking).

```python
# Example: saving CFAD data
np.savez_compressed(
    "data/final/figure_data/fig3_cfad_total.npz",
    cfad_R_clean=cfad_R_clean,         # 2D array (height_bins, param_bins)
    cfad_R_polluted=cfad_R_polluted,
    cfad_Z_clean=cfad_Z_clean,
    cfad_Z_polluted=cfad_Z_polluted,
    cfad_Dm_clean=cfad_Dm_clean,
    cfad_Dm_polluted=cfad_Dm_polluted,
    cfad_dBNw_clean=cfad_dBNw_clean,
    cfad_dBNw_polluted=cfad_dBNw_polluted,
    height_edges=height_edges,
    R_edges=R_edges,
    Z_edges=Z_edges,
    Dm_edges=Dm_edges,
    dBNw_edges=dBNw_edges,
)

# Loading for plotting:
d = np.load("data/final/figure_data/fig3_cfad_total.npz")
cfad_R_clean = d['cfad_R_clean']
```

---

## 11. Per-Figure Processing Logic

### Figure 1 — PDF of Near-Surface Parameters

**Variables:** log₁₀R, Z, Dm, dBNw  
**Breakdown:** Total / Stratiform / Convective × Clean / Polluted = 6 groups

```python
def compute_pdfs(df_clean, df_polluted, param_col, bins):
    """
    Compute normalized PDFs using histogram.
    Returns bin centers and density values for clean and polluted.
    """
    from scipy.stats import gaussian_kde
    
    # Method 1: KDE (smooth, matches paper visual style)
    kde_clean    = gaussian_kde(df_clean[param_col].dropna(), bw_method='scott')
    kde_polluted = gaussian_kde(df_polluted[param_col].dropna(), bw_method='scott')
    
    x = np.linspace(bins[0], bins[-1], 500)
    pdf_clean    = kde_clean(x)
    pdf_polluted = kde_polluted(x)
    
    return x, pdf_clean, pdf_polluted
```

**Compute for each of 6 groups × 4 parameters = 24 curves total.**

**Log transform note:**
```python
# Do NOT take log10 of already-log10 values
# log10R: apply log10 to R_ns (mm/h), excluding zeros
logR = np.log10(df['R_ns'].replace(0, np.nan))

# dBNw: DPR already gives log10(Nw); multiply by 10 for dBNw
dBNw = 10 * df['Nw_ns']   # Nw_ns is log10(Nw) from paramDSD[:,:,:,1]

# Z_ns is already in dBZ — use as-is
# Dm_ns is in mm — use as-is
```

### Figure 2 — 2D Histogram: Dm vs. dBNw

**Variables:** Dm (x-axis), dBNw (y-axis)  
**Breakdown:** Total / Stratiform / Convective × Clean / Polluted = 6 panels

```python
def compute_2d_histogram(df, dm_bins, nw_bins):
    """Normalized 2D histogram (density)."""
    H, xedges, yedges = np.histogram2d(
        df['Dm_ns'].dropna(),
        df['dBNw_ns'].dropna(),
        bins=[dm_bins, nw_bins],
        density=True
    )
    return H, xedges, yedges
```

### Figure 3–5 — CFADs

**The CFAD computation is the most memory-intensive step.**

```python
def compute_cfad(zarr_store, precip_type_filter, param_key, 
                 height_edges, param_edges):
    """
    Compute Contoured Frequency by Altitude Diagram.
    
    zarr_store: opened zarr store with profiles
    precip_type_filter: None (total), 1 (stratiform), 2 (convective)
    param_key: 'R', 'Z', 'Dm', or 'dBNw'
    height_edges: 1D array of altitude bin edges (km)
    param_edges: 1D array of parameter bin edges
    
    Returns: 2D array of shape (n_height_bins, n_param_bins)
             values are occurrence frequency in percent
    """
    n_height = len(height_edges) - 1
    n_param  = len(param_edges) - 1
    cfad     = np.zeros((n_height, n_param))
    counts_per_height = np.zeros(n_height)
    
    # Load precip type filter
    ptype = zarr_store['precip_type'][:]
    if precip_type_filter is not None:
        pixel_mask = (ptype == precip_type_filter)
    else:
        pixel_mask = np.ones(len(ptype), dtype=bool)
    
    # Process in chunks to avoid memory overflow
    chunk_size = 5000
    n_pixels   = pixel_mask.sum()
    pixel_indices = np.where(pixel_mask)[0]
    
    for i in range(0, len(pixel_indices), chunk_size):
        idx_chunk = pixel_indices[i:i+chunk_size]
        
        heights = zarr_store['bin_heights'].oindex[idx_chunk, :]  # (chunk, 88)
        values  = zarr_store[param_key].oindex[idx_chunk, :]       # (chunk, 88)
        
        # For log-scale params
        if param_key == 'R':
            values = np.log10(np.where(values > 0, values, np.nan))
        
        # Loop over height bins
        for h_idx in range(n_height):
            h_lo = height_edges[h_idx]
            h_hi = height_edges[h_idx + 1]
            
            # Mask bins within this height range
            height_mask = (heights >= h_lo) & (heights < h_hi)
            
            valid_vals = values[height_mask]
            valid_vals = valid_vals[np.isfinite(valid_vals)]
            
            if len(valid_vals) == 0:
                continue
            
            # Accumulate histogram
            hist, _ = np.histogram(valid_vals, bins=param_edges)
            cfad[h_idx, :] += hist
            counts_per_height[h_idx] += len(valid_vals)
    
    # Normalize: convert to percentage per height level
    for h_idx in range(n_height):
        if counts_per_height[h_idx] > 0:
            cfad[h_idx, :] = cfad[h_idx, :] / counts_per_height[h_idx] * 100.0
    
    return cfad
```

### Figure 6 — KDE of 30 dBZ Echo Top vs. Parameters

**30 dBZ echo top height** is the altitude where Z drops below 30 dBZ from above (storm intensity proxy).

```python
def compute_30dbz_echotop(Z_profile, bin_heights):
    """
    Find the highest altitude where Z >= 30 dBZ.
    Z_profile: 1D array (88 bins), top-to-bottom
    bin_heights: 1D array (88 bins), altitude in km
    """
    valid = np.isfinite(Z_profile) & (Z_profile >= 30.0)
    if valid.sum() == 0:
        return np.nan
    # Highest altitude (minimum bin index) where Z >= 30
    top_bin = np.where(valid)[0].min()
    return bin_heights[top_bin]
```

Then KDE of scatter: `h30dBZ` (y) vs. R, Dm, dBNw (x) — 3 panels × total/stratiform/convective.

### Figure 7 — Warm Rain ΔDm vs. ΔZ Scatter

```python
def compute_warm_rain_deltas(df):
    """
    Compute ΔDm and ΔZ between 2 km and 3 km levels.
    df must have columns: Dm_2km, Dm_3km, Z_2km, Z_3km
    """
    df['delta_Dm'] = df['Dm_2km'] - df['Dm_3km']   # ΔDm = (Dm)_2km - (Dm)_3km
    df['delta_Z']  = df['Z_2km']  - df['Z_3km']    # ΔZ  = (Z)_2km  - (Z)_3km
    return df
```

The four quadrants encode microphysical processes:

| Quadrant | ΔDm | ΔZ | Process |
|---|---|---|---|
| Q1 (upper right) | + | + | Coalescence |
| Q2 (upper left)  | + | − | Size-sorting / evaporation |
| Q3 (lower left)  | − | − | Breakup |
| Q4 (lower right) | − | + | Breakup-coalescence balance |

Compute percentage of points in each quadrant per group.

### Figure 8 — Thermodynamic Box Plots

This requires additional MERRA-2 variables beyond AOD. You need `tavg1_2d_slv_Nx` (M2T1NXSLV) for:

| Variable | MERRA-2 name |
|---|---|
| Dew point temperature (2m) | `T2MDEW` |
| Temperature (2m) | `T2M` |
| Relative humidity | Derive: `RH = f(T2M, T2MDEW)` or use 3D field |
| K-index | Derive from pressure level fields |
| CAPE | Derive from `tavg3_3d_mst_Nv` or use reanalysis product |
| Cloud inhibition (CIN) | Derive or use ERA5 (MERRA-2 doesn't directly provide CAPE/CIN) |
| Total column water | `TQL + TQI + TQV` from `M2T1NXSLV` |
| Total column water vapor | `TQV` from `M2T1NXSLV` |

**Note:** MERRA-2 does not directly provide CAPE and CIN as output variables. You have two options:
1. Use ERA5 reanalysis which does provide CAPE/CIN (simpler)
2. Compute CAPE/CIN from MERRA-2 pressure-level T, q fields using MetPy:

```python
from metpy.calc import cape_cin, dewpoint_from_specific_humidity
from metpy.units import units

# From M2I3NPASM (pressure-level fields):
# T(tzyx), QV(tzyx), pressure levels
cape, cin = cape_cin(pressure_levels * units.Pa,
                     temperature * units.kelvin,
                     dewpoint * units.kelvin)
```

---

## 12. Plotting Specifications

### 12.1 General Style

```python
import matplotlib.pyplot as plt
import matplotlib as mpl

# Global style
mpl.rcParams.update({
    'font.family':     'serif',
    'font.size':       10,
    'axes.labelsize':  11,
    'axes.titlesize':  12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi':      300,
    'savefig.dpi':     300,
    'savefig.bbox':    'tight',
    'savefig.format':  'png',
})

COLOR_CLEAN    = '#2ca02c'   # green (matches paper)
COLOR_POLLUTED = '#1f77b4'   # blue  (matches paper)
```

### 12.2 Figure 1 — PDF (3×4 panel grid)

```
Layout: 3 rows (Total, Stratiform, Convective) × 4 cols (logR, Z, Dm, dBNw)
figsize: (14, 10)
Line: solid, linewidth=1.5
X-axis labels: log₁₀R (mm h⁻¹), Z (dBZ), Dm (mm), dBNw (m⁻³ mm⁻¹)
Y-axis: PDF (shared within row)
Legend: only in first panel — Clean (green), Polluted (blue)
```

### 12.3 Figure 2 — 2D Histogram (2×3 panel grid)

```
Layout: 2 rows (Clean, Polluted) × 3 cols (Total, Stratiform, Convective)
figsize: (12, 8)
Colormap: jet
X-axis: Dm (mm), range 0–3
Y-axis: dBNw (m⁻³ mm⁻¹), range 10–60
Colorbar: shared, label = "Density"
Title per col: Total, Stratiform, Convective
Row label: Clean / Polluted
```

### 12.4 Figures 3–5 — CFADs (2×4 panel grid each)

```
Layout: 2 rows (Clean, Polluted) × 4 cols (R, Z, Dm, dBNw)
figsize: (14, 8)
Colormap: jet
Y-axis: Height (km), 2–14
X-axis: parameter range per variable
Colorbar: shared per row, label = "%"
CFAD range: 0–12% per the paper's colorbar
Panel labels: (a), (b), etc.
```

```python
def plot_cfad(ax, cfad_2d, param_edges, height_edges, cmap='jet', vmax=12):
    """Plot a single CFAD panel."""
    from matplotlib.colors import BoundaryNorm
    
    # cfad_2d: (n_heights, n_params)
    h_centers = (height_edges[:-1] + height_edges[1:]) / 2
    p_centers = (param_edges[:-1]  + param_edges[1:])  / 2
    
    im = ax.pcolormesh(
        p_centers, h_centers, cfad_2d,
        cmap=cmap, vmin=0, vmax=vmax, shading='auto'
    )
    ax.set_ylim([2, 14])
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
    ax.grid(True, alpha=0.3, linestyle='--')
    return im
```

### 12.5 Figure 7 — Warm Rain Scatter (3×2 panel grid)

```
Layout: 2 rows (Clean, Polluted) × 3 cols (Total, Stratiform, Convective)
figsize: (12, 8)
Scatter: small dots, colored by density (use plt.hexbin or 2D KDE)
Quadrant lines: dashed at ΔDm=0, ΔZ=0
Quadrant labels: percentage text in each quadrant
X-axis: ΔZ (dBZ), range -20 to +20
Y-axis: ΔDm (mm), range -1 to +1
Colormap: jet (density), range 0–30%
```

### 12.6 Figure 8 — Box Plots

```
Layout: 2×4 subplots
figsize: (14, 8)
Box: clean=green, polluted=blue (facecolor)
Whiskers: 5th–95th percentile
Outliers: not shown (or small dots)
Significance annotation: add ** for p<0.001
```

---

## 13. Parallelization Strategy

### 13.1 GPM File Processing (Most Time-Consuming Step)

Each GPM HDF5 file is independent. Use `concurrent.futures.ProcessPoolExecutor`:

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import glob

gpm_files = sorted(glob.glob("data/raw/gpm/**/*.HDF5", recursive=True))

def process_one_file(filepath):
    """Wrapper: extract pixels + profiles from one GPM file."""
    try:
        pixels   = extract_near_surface(filepath, taiwan_bbox)
        profiles = extract_profiles(filepath, taiwan_bbox)
        return filepath, pixels, profiles
    except Exception as e:
        return filepath, None, str(e)

# Use n_workers = number of CPU cores - 1
n_workers = 7   # adjust to your machine

results = []
with ProcessPoolExecutor(max_workers=n_workers) as executor:
    futures = {executor.submit(process_one_file, f): f for f in gpm_files}
    for future in tqdm(as_completed(futures), total=len(futures)):
        filepath, pixels, profiles = future.result()
        if pixels is not None:
            results.append((filepath, pixels, profiles))
```

**Memory warning:** Each GPM file loads ~200–400 MB into memory. With 7 workers, that's ~2.8 GB peak RAM. If you have less memory, reduce `n_workers` or stream one variable at a time.

### 13.2 MERRA-2 Processing (Dask)

```python
import xarray as xr
import dask

# Open with dask for lazy evaluation
aod_ds = xr.open_dataset(
    "data/interim/merra2_aod_taiwan.nc",
    chunks={'time': 48, 'lat': 91, 'lon': 144}
)

# Operations are lazy — only computed when .compute() or .values is called
# This avoids loading all 10 years into memory at once
```

### 13.3 CFAD Computation (Chunked Zarr)

The CFAD loop in Section 11 already processes profiles in chunks of 5000 pixels to stay within RAM limits. Adjust `chunk_size` based on available memory.

---

## 14. Known Hurdles and Solutions

### H1: GPM typePrecip Encoding

**Hurdle:** The `typePrecip` integer encodes multiple levels of classification (major/minor/detail) in a packed format. Different papers use different decoding schemes.

**Solution:** Always verify against the **DPR Level-2 Algorithm Theoretical Basis Document (ATBD) Version 07**. The major type is: `typePrecip // 10000000`. The value 0 = no precip, 1 = stratiform, 2 = convective, 3 = other. Do not assume minor type encodings without reading the ATBD.

---

### H2: Mismatched paramDSD Indices

**Hurdle:** `paramDSD` has shape `(nray, nscan, nbin, 2)`. Index 0 is Dm in mm; index 1 is log₁₀(Nw) — but only when valid retrievals exist. In regions without rain, both are fill values (−9999.9).

**Solution:**
```python
Dm  = np.where(dsd[:,:,:,0] > 0,   dsd[:,:,:,0], np.nan)
logNw = np.where((dsd[:,:,:,1] > -5) & (dsd[:,:,:,1] < 6), dsd[:,:,:,1], np.nan)
dBNw = 10 * logNw   # convert to dBNw
```

---

### H3: GPM Scan Time Broadcast

**Hurdle:** `NS/ScanTime/Hour` has shape `(nscan,)` but pixels have shape `(nray, nscan)`. You must broadcast scan time to all rays.

**Solution:**
```python
hour_2d = np.broadcast_to(hour[np.newaxis, :], (nray, nscan))
# Note: np.broadcast_to returns a read-only view — copy if you need to modify
hour_2d = hour_2d.copy()
```

---

### H4: MERRA-2 Version Number in Filename

**Hurdle:** For the period 2014–2023, files are named `MERRA2_400.*`. If your download covers 2011 only, the version number is still 400. If you ever add pre-2011 data, you'll encounter `MERRA2_300` files. `xr.open_mfdataset` will handle them transparently, but glob patterns must be adjusted.

**Solution:** Use a glob that doesn't hardcode the version:
```python
files = sorted(glob.glob("data/raw/merra2/MERRA2_*.tavg1_2d_aer_Nx.*.nc4"))
```

---

### H5: Height Varies Per Pixel in CFADs

**Hurdle:** Because terrain elevation varies across Taiwan (sea level to 3952 m), the altitude corresponding to a given range bin varies per pixel. You cannot simply use a fixed bin-to-height mapping.

**Solution:** Precompute `bin_heights` per pixel (Section 9) and store in the Zarr profile store. During CFAD computation, use `bin_heights` to map each bin to the correct altitude bin.

---

### H6: Memory Overflow When Loading All Profiles

**Hurdle:** All 210 overpasses × ~thousands of pixels per overpass × 88 bins × 4 variables = potentially several GB in RAM.

**Solution:** Use Zarr chunked storage (Section 10.2). Never load the entire profile dataset at once. Process the CFAD in height-bin chunks as shown in Section 11.

---

### H7: GPM Clutter Near Surface

**Hurdle:** The bottom 2–3 range bins near the surface are often contaminated by ground clutter. Using `precipRateNearSurface` avoids this (it's a DPR-computed near-surface estimate at ~2 km), but profile data for bins near `binRealSurface` should be masked.

**Solution:**
```python
# Mask bins within 3 bins of surface (375 m clutter zone)
for i in range(n_pixels):
    clutter_start = binSurface[i] - 3
    profiles[i, clutter_start:] = np.nan
```

---

### H8: Taiwan-Only Orbits

**Hurdle:** Most GPM orbits do not pass over Taiwan. Of ~5000+ orbits per year, only a fraction cover the Taiwan region. Opening every file just to check wastes time.

**Solution:** Pre-filter files using orbit track metadata. The GPM file names encode start/end times — you can check the geographic coverage from a pre-downloaded orbit manifest, or do a quick bounding-box check:

```python
def orbit_covers_taiwan(filepath, bbox):
    """Fast check: does this orbit contain any Taiwan pixels?"""
    with h5py.File(filepath, 'r') as f:
        lat = f['NS/Latitude'][:]
        lon = f['NS/Longitude'][:]
        return (
            ((lat >= bbox['lat_min']) & (lat <= bbox['lat_max']) &
             (lon >= bbox['lon_min']) & (lon <= bbox['lon_max'])).any()
        )
```

---

### H9: AOD Intermediate Values

**Hurdle:** Pixels with 0.1 ≤ AOD ≤ 0.5 are excluded from analysis. These could be a large fraction of your data.

**Solution:** Assign a three-class label (`clean`, `polluted`, `intermediate`) at the overpass level, then simply filter out `intermediate` when doing analysis. Never delete them from the parquet — keep the label so you can reconstruct the full dataset later if thresholds change.

---

### H10: CAPE/CIN Not Directly in MERRA-2 Aerosol Collection

**Hurdle:** The paper shows CAPE and CIN box plots (Fig. 8), but `tavg1_2d_aer_Nx` doesn't contain these. They require thermodynamic profiles.

**Solution:** Download `tavg1_2d_flx_Nx` (M2T1NXFLX) which has some surface flux diagnostics, but for CAPE/CIN specifically, compute from `inst3_3d_asm_Np` (M2I3NPASM) pressure-level T and Q fields using MetPy. Alternatively, cross-validate by downloading ERA5 CAPE/CIN directly from Copernicus.

---

### H11: Coordinate System Differences

**Hurdle:** GPM uses `(nray, nscan)` ordering (cross-track × along-track), while MERRA-2 uses `(time, lat, lon)`. Converting between them requires careful indexing.

**Solution:** Flatten GPM `(nray, nscan)` to a 1D pixel list immediately after reading. Never work with the 2D grid unless you need the spatial context.

---

### H12: dBNw Definition Consistency

**Hurdle:** `dBNw = 10 × log₁₀(Nw)`. GPM's `paramDSD[:,:,:,1]` already gives log₁₀(Nw). Multiplying by 10 gives dBNw. Forgetting this factor of 10 shifts the entire distribution.

**Solution:** Define a single utility function and use it everywhere:
```python
def to_dBNw(log10Nw):
    """Convert log10(Nw) from GPM paramDSD to dBNw."""
    return 10.0 * log10Nw
```

---

## 15. End-to-End Run Order

```
Step 0:  python scripts/00_check_environment.py
         → Verify all packages, check raw data file counts, estimate disk space needed

Step 1:  python scripts/01_preprocess_merra2.py
         → Input:  data/raw/merra2/**/*.nc4
         → Output: data/interim/merra2_aod_taiwan.nc
         → Time:   ~10–30 min (depends on disk speed)

Step 2:  python scripts/02_extract_gpm_pixels.py
         → Input:  data/raw/gpm/**/*.HDF5
                   data/interim/merra2_aod_taiwan.nc (for collocation)
         → Output: data/interim/gpm_pixels_clean.parquet
                   data/interim/gpm_pixels_polluted.parquet
         → Time:   ~2–4 hours (parallelized over orbits)

Step 3:  python scripts/03_extract_gpm_profiles.py
         → Input:  data/raw/gpm/**/*.HDF5
                   data/interim/gpm_pixels_clean.parquet   (for orbit list)
                   data/interim/gpm_pixels_polluted.parquet
         → Output: data/interim/profiles_clean.zarr/
                   data/interim/profiles_polluted.zarr/
         → Time:   ~3–6 hours (parallelized)
         → NOTE:   Steps 2 and 3 can be merged into one pass per file to avoid
                   reading each HDF5 twice

Step 4:  python scripts/04_collocate_and_classify.py
         → Input:  parquet files from Step 2
                   merra2_aod_taiwan.nc
         → Output: updated parquet files with 'AOD' and 'class' columns
         → Time:   ~30 min

Step 5:  python scripts/05_compute_figure_data.py
         → Input:  parquet files, zarr stores
         → Output: data/final/figure_data/*.npz
         → Time:   ~1–2 hours (CFAD computation is the bottleneck)

Step 6:  python scripts/06_plot_figures.py
         → Input:  data/final/figure_data/*.npz
         → Output: data/final/plots/*.png
         → Time:   ~2–5 min
```

**Recommended merge for Step 2+3:** Read each GPM HDF5 file exactly once, extract both near-surface pixels AND vertical profiles in a single `h5py` open call, then write to both parquet and zarr. This halves total I/O time.

---

## 16. Validation Checkpoints

After each major step, verify these expected values against the paper:

| Check | Expected (Paper) | How to Verify |
|---|---|---|
| Total clean overpasses | 96 | `df_clean['orbit_id'].nunique()` |
| Total polluted overpasses | 114 | `df_polluted['orbit_id'].nunique()` |
| Mean R (clean) | 1.28 mm/h | `df_clean['R_ns'].mean()` |
| Mean R (polluted) | 2.78 mm/h | `df_polluted['R_ns'].mean()` |
| Mean Z (clean) | 22.83 dBZ | `df_clean['Z_ns'].mean()` |
| Mean Z (polluted) | 26.8 dBZ | `df_polluted['Z_ns'].mean()` |
| Mean Dm (clean) | 1.11 mm | `df_clean['Dm_ns'].mean()` |
| Mean Dm (polluted) | 1.30 mm | `df_polluted['Dm_ns'].mean()` |
| Mean dBNw (clean) | 34.56 | `df_clean['dBNw_ns'].mean()` |
| Mean dBNw (polluted) | 43.06 | `df_polluted['dBNw_ns'].mean()` |
| Welch's t-test on R | p < 0.001 | `scipy.stats.ttest_ind(..., equal_var=False)` |
| Clean breakup dominant | ~46% of total | quadrant percentages in Fig 7 |
| Polluted coalescence | ~50% of total | quadrant percentages in Fig 7 |

If any of these diverge significantly, the issue is almost always one of:
1. Wrong `typePrecip` decoding
2. Wrong fill value masking
3. Wrong AOD threshold application
4. Overpass-level vs. pixel-level AOD assignment mismatch

---

*End of Design Documentation*

---

**Document generated for:** Seela et al. (2026), Urban Climate 65, 102778  
**Pipeline designed for:** Full reproduction of Figures 1–8  
**Estimated total processing time (8-core machine, 32 GB RAM):** 6–10 hours  
**Estimated disk space (raw + interim + final):** 500 GB raw GPM + 5 GB MERRA-2 (Taiwan) + ~20 GB interim
