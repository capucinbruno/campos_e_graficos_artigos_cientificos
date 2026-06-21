# app/src/uteis/downloaders_gfs_hgt700.py
# -*- coding: utf-8 -*-
"""
Downloader GFS (previsao) de altura geopotencial em 700 hPa via NOMADS Grib Filter.

Um NetCDF por dia valido com as horas sinoticas, variavel 'hgt' (metros geopotenciais).
Reusa os helpers _download_grb2 / _steps_for_day do downloader 200. Espelha o
downloaders_gfs_tmp850, mudando so a variavel/nivel (HGT @ 700 hPa).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import xarray as xr

from app.common.forecast_download import StepNotAvailable, download_days_parallel, save_netcdf
from app.shared.logger import get_logger
from app.src.uteis.downloaders_gfs_fcst200 import (
    DEFAULT_SYNOPTIC_HOURS,
    DIR_DADOS_BASE,
    _download_grb2,
    _steps_for_day,
)

logger = get_logger(__name__)

DIR_GFS_HGT700 = DIR_DADOS_BASE / 'GFS_HGT700'


def _build_params(init: datetime, fhr: int) -> dict:
    return {
        'file': f'gfs.t{init.hour:02d}z.pgrb2.0p25.f{fhr:03d}',
        'lev_700_mb': 'on',
        'var_HGT': 'on',
        'dir': f'/gfs.{init.strftime("%Y%m%d")}/{init.hour:02d}/atmos',
    }


def _open_gfs_hgt700(path: Path) -> xr.Dataset:
    """Abre o GRIB2 do HGT@700 e normaliza para 'hgt' (m) em (lat, lon)."""
    from app.common.forecast_download import GRIB_NETCDF_LOCK
    with GRIB_NETCDF_LOCK:  # ecCodes nao e thread-safe entre downloads paralelos
        ds = xr.open_dataset(
            path, engine='cfgrib', backend_kwargs={'indexpath': ''},
            filter_by_keys={'typeOfLevel': 'isobaricInhPa'},
        ).load()
    ren = {}
    for name in list(ds.dims) + list(ds.coords):
        low = name.lower()
        if low == 'latitude' and 'lat' not in ds.dims:
            ren[name] = 'lat'
        elif low == 'longitude' and 'lon' not in ds.dims:
            ren[name] = 'lon'
    if ren:
        ds = ds.rename(ren)
    var = next((v for v in ('gh', 'hgt', 'z', 'HGT') if v in ds.data_vars), list(ds.data_vars)[0])
    da = ds[var].rename('hgt')
    for dim in ('isobaricInhPa', 'level', 'pressure_level'):
        if dim in da.dims:
            da = da.isel({dim: 0}, drop=True)
        elif dim in da.coords:
            da = da.drop_vars(dim, errors='ignore')
    for coord in ('time', 'step', 'valid_time'):
        if coord in da.coords and coord not in da.dims:
            da = da.drop_vars(coord, errors='ignore')
    da.attrs['units'] = 'm'
    return da.to_dataset(name='hgt')


def _download_day(init: datetime, day: date, steps: List[Tuple[int, datetime]], force: bool) -> Path:
    fname = f'gfs_hgt700_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
    nc_path = DIR_GFS_HGT700 / fname
    if nc_path.exists() and not force:
        logger.info('GFS Z700 valido {} (init {}Z) ja existe — pulando.', day, init.hour)
        return nc_path
    DIR_GFS_HGT700.mkdir(parents=True, exist_ok=True)
    parts = []
    for fhr, vt in steps:
        grb = DIR_GFS_HGT700 / f'gfs_hgt700_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
        if not grb.exists() or force:
            try:
                _download_grb2(_build_params(init, fhr), grb)
            except StepNotAvailable:
                logger.warning('  GFS Z700 f{:03d} ainda nao publicado (404) — pulando passo', fhr)
                continue
        ds = _open_gfs_hgt700(grb).expand_dims(time=[np.datetime64(vt)])
        parts.append(ds.load())
    if not parts:
        logger.warning('GFS Z700 valido {} sem passos publicados — dia ignorado.', day)
        return None
    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    save_netcdf(ds_day, nc_path)
    for fhr, _ in steps:
        grb = DIR_GFS_HGT700 / f'gfs_hgt700_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
        if grb.exists():
            grb.unlink()
    logger.info('GFS Z700 valido {} salvo: {}', day, nc_path.name)
    return nc_path


def ensure_gfs_hgt700_fcst_for_period(
    init: datetime, lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de Z700 (m) do GFS para a janela [init, init+lead_hours]."""
    end = init + timedelta(hours=lead_hours)
    jobs = []
    day = init.date()
    while day <= end.date():
        steps = _steps_for_day(init, day, hours, lead_hours)
        if steps:
            jobs.append((day, steps))
        day += timedelta(days=1)
    files = download_days_parallel(
        jobs, lambda day, steps: _download_day(init, day, steps, force_redownload), logger)
    logger.info('GFS Z700: {} arquivos | init {:%Y-%m-%d %H}Z + {}h', len(files), init, lead_hours)
    return files
