# app/src/uteis/clim_diaria_olr.py
# -*- coding: utf-8 -*-
"""
OLR diario (CPC Blended OLR 2.5°) — media observada e climatologia diaria (LTM).

Fonte: olr.day.mean.nc (CPC Blended OLR, media diaria, todos os anos).
A partir do MEAN calcula-se a climatologia diaria (LTM por dia-do-ano, base
1991-2020) — assim a anomalia de OLR (reanalise OU previsao GFS) usa a MESMA
climatologia ("igual com igual"), em vez do arquivo de anomalia pronto do PSL.

  - clim_olr_daily(dates, lat, lon) -> LTM por dia-do-ano, interpolada p/ a grade alvo
  - olr_obs_daily(dates, lat, lon)  -> OLR observado (mean) das datas, na grade alvo
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from app.common.download_helper import DownloadEngine, download_with_progress
from app.shared.logger import get_logger
from app.shared.settings_factory import settings

logger = get_logger(__name__)

OLR_MEAN_URL = 'https://downloads.psl.noaa.gov/Datasets/cpc_blended_olr-2.5deg/olr.day.mean.nc'
CLIM_BASE = (1991, 2020)


def _ensure_olr_mean_file() -> Path:
    out = Path(settings.DIR_FILE_NC) / 'olr.day.mean.nc'
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        logger.info('Baixando OLR diario (CPC Blended mean): {}', OLR_MEAN_URL)
        download_with_progress(
            url=OLR_MEAN_URL, output_path=str(out),
            description='OLR diaria (CPC Blended mean)', max_retries=5,
            engine=DownloadEngine.AUTO,
        )
    return out


def _open_olr() -> xr.DataArray:
    ds = xr.open_dataset(str(_ensure_olr_mean_file()))
    da = ds['olr'] if 'olr' in ds.data_vars else next(iter(ds.data_vars.values()))
    ren = {}
    for name in list(da.dims) + list(da.coords):
        low = name.lower()
        if low == 'latitude' and 'lat' not in da.dims:
            ren[name] = 'lat'
        elif low == 'longitude' and 'lon' not in da.dims:
            ren[name] = 'lon'
    if ren:
        da = da.rename(ren)
    da = da.assign_coords(lon=(da['lon'] % 360)).sortby('lon').sortby('lat')
    return da


def _interp(da2d: xr.DataArray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    tlat = xr.DataArray(lat, dims=['lat'])
    tlon = xr.DataArray(lon, dims=['lon'])
    return da2d.interp(lat=tlat, lon=tlon, method='linear').values


def _doy(dt64) -> int:
    """Dia-do-ano com 29/fev mapeado para 28/fev (cap em 365)."""
    d = pd.Timestamp(dt64)
    doy = d.dayofyear
    if d.is_leap_year and doy >= 60:  # apos 29/fev
        doy -= 1
    return min(doy, 365)


def clim_olr_daily(dates: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """LTM diaria de OLR (base 1991-2020) p/ cada data, interpolada p/ (lat, lon)."""
    da = _open_olr()
    base = da.sel(time=slice(f'{CLIM_BASE[0]}-01-01', f'{CLIM_BASE[1]}-12-31'))
    if base.sizes.get('time', 0) == 0:
        base = da
    t = pd.DatetimeIndex(pd.to_datetime(base['time'].values))
    keep = ~((t.month == 2) & (t.day == 29))
    base = base.isel(time=np.asarray(keep))
    doy_all = np.array([_doy(x) for x in base['time'].values])
    base = base.assign_coords(doy=('time', doy_all))
    clim = base.groupby('doy').mean('time')  # (doy, lat, lon)
    out = []
    for d in dates:
        out.append(_interp(clim.sel(doy=_doy(d)), lat, lon))
    return np.stack(out, axis=0)


def olr_obs_daily(dates: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """OLR observado (mean) das `dates`, interpolado p/ (lat, lon). Usado em reanalise."""
    da = _open_olr()
    out = []
    for d in dates:
        sub = da.sel(time=np.datetime64(pd.Timestamp(d).date()), method='nearest')
        out.append(_interp(sub, lat, lon))
    return np.stack(out, axis=0)
