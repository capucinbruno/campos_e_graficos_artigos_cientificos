# app/src/uteis/downloaders_gfs_tmp850.py
# -*- coding: utf-8 -*-
"""
Downloader GFS (previsao) de temperatura em 850 hPa via NOMADS Grib Filter.

Um NetCDF por dia valido com as horas sinoticas, variavel 't' (Kelvin — o s34
normaliza p/ °C). Reusa os helpers _download_grb2 / _steps_for_day do downloader 200.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import xarray as xr

from app.shared.logger import get_logger
from app.src.uteis.downloaders_gfs_fcst200 import (
    DEFAULT_SYNOPTIC_HOURS,
    DIR_DADOS_BASE,
    _download_grb2,
    _steps_for_day,
)

logger = get_logger(__name__)

DIR_GFS_TMP850 = DIR_DADOS_BASE / 'GFS_TMP850'


def _build_params(init: datetime, fhr: int) -> dict:
    return {
        'file': f'gfs.t{init.hour:02d}z.pgrb2.0p25.f{fhr:03d}',
        'lev_850_mb': 'on',
        'var_TMP': 'on',
        'dir': f'/gfs.{init.strftime("%Y%m%d")}/{init.hour:02d}/atmos',
    }


def _open_gfs_tmp850(path: Path) -> xr.Dataset:
    """Abre o GRIB2 do TMP@850 e normaliza para 't' (K) em (lat, lon)."""
    ds = xr.open_dataset(
        path, engine='cfgrib', backend_kwargs={'indexpath': ''},
        filter_by_keys={'typeOfLevel': 'isobaricInhPa'},
    )
    ren = {}
    for name in list(ds.dims) + list(ds.coords):
        low = name.lower()
        if low == 'latitude' and 'lat' not in ds.dims:
            ren[name] = 'lat'
        elif low == 'longitude' and 'lon' not in ds.dims:
            ren[name] = 'lon'
    if ren:
        ds = ds.rename(ren)
    var = next((v for v in ('t', 'tmp', 'air', 'temperature') if v in ds.data_vars),
               list(ds.data_vars)[0])
    da = ds[var].rename('t')
    for dim in ('isobaricInhPa', 'level', 'pressure_level'):
        if dim in da.dims:
            da = da.isel({dim: 0}, drop=True)
        elif dim in da.coords:
            da = da.drop_vars(dim, errors='ignore')
    for coord in ('time', 'step', 'valid_time'):
        if coord in da.coords and coord not in da.dims:
            da = da.drop_vars(coord, errors='ignore')
    da.attrs['units'] = 'K'
    return da.to_dataset(name='t')


def _download_day(init: datetime, day: date, steps: List[Tuple[int, datetime]], force: bool) -> Path:
    fname = f'gfs_tmp850_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
    nc_path = DIR_GFS_TMP850 / fname
    if nc_path.exists() and not force:
        logger.info('GFS T850 valido {} (init {}Z) ja existe — pulando.', day, init.hour)
        return nc_path
    DIR_GFS_TMP850.mkdir(parents=True, exist_ok=True)
    parts = []
    for fhr, vt in steps:
        grb = DIR_GFS_TMP850 / f'gfs_tmp850_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
        if not grb.exists() or force:
            _download_grb2(_build_params(init, fhr), grb)
        ds = _open_gfs_tmp850(grb).expand_dims(time=[np.datetime64(vt)])
        parts.append(ds.load())
    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    ds_day.to_netcdf(nc_path, engine='netcdf4')
    for fhr, _ in steps:
        grb = DIR_GFS_TMP850 / f'gfs_tmp850_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
        if grb.exists():
            grb.unlink()
    logger.info('GFS T850 valido {} salvo: {}', day, nc_path.name)
    return nc_path


def ensure_gfs_tmp850_fcst_for_period(
    init: datetime, lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de T850 (K) do GFS para a janela [init, init+lead_hours]."""
    files: List[Path] = []
    end = init + timedelta(hours=lead_hours)
    day = init.date()
    while day <= end.date():
        steps = _steps_for_day(init, day, hours, lead_hours)
        if steps:
            files.append(_download_day(init, day, steps, force_redownload))
        day += timedelta(days=1)
    logger.info('GFS T850: {} arquivos | init {:%Y-%m-%d %H}Z + {}h', len(files), init, lead_hours)
    return files
