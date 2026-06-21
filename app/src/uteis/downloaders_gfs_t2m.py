# app/src/uteis/downloaders_gfs_t2m.py
# -*- coding: utf-8 -*-
"""
Downloader GFS (previsao) de temperatura do ar a 2 m via NOMADS Grib Filter.

T2m e variavel de SUPERFICIE (heightAboveGround=2), nao nivel de pressao: o filtro usa
`lev_2_m_above_ground=on&var_TMP=on`. Um NetCDF por dia valido, variavel 't2m' (Kelvin —
o s34 normaliza p/ °C). Reusa _download_grb2 / _steps_for_day do downloader 200.
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

DIR_GFS_T2M = DIR_DADOS_BASE / 'GFS_T2M'


def _build_params(init: datetime, fhr: int) -> dict:
    return {
        'file': f'gfs.t{init.hour:02d}z.pgrb2.0p25.f{fhr:03d}',
        'lev_2_m_above_ground': 'on',
        'var_TMP': 'on',
        'dir': f'/gfs.{init.strftime("%Y%m%d")}/{init.hour:02d}/atmos',
    }


def _open_t2m_grb2(path: Path) -> xr.Dataset:
    """Abre o GRIB2 do TMP@2m e normaliza para 't2m' (K) em (lat, lon)."""
    from app.common.forecast_download import GRIB_NETCDF_LOCK
    with GRIB_NETCDF_LOCK:  # ecCodes nao e thread-safe entre downloads paralelos
        ds = xr.open_dataset(
            path, engine='cfgrib', backend_kwargs={'indexpath': ''},
            filter_by_keys={'typeOfLevel': 'heightAboveGround'},
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
    var = next((v for v in ('t2m', 't', '2t', 'tmp') if v in ds.data_vars), list(ds.data_vars)[0])
    da = ds[var].rename('t2m')
    for coord in ('time', 'step', 'valid_time', 'heightAboveGround', 'surface', 'level'):
        if coord in da.coords and coord not in da.dims:
            da = da.drop_vars(coord, errors='ignore')
    da.attrs['units'] = 'K'
    return da.to_dataset(name='t2m')


def _download_day(init: datetime, day: date, steps: List[Tuple[int, datetime]], force: bool) -> Path:
    fname = f'gfs_t2m_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
    nc_path = DIR_GFS_T2M / fname
    if nc_path.exists() and not force:
        logger.info('GFS T2m valido {} (init {}Z) ja existe — pulando.', day, init.hour)
        return nc_path
    DIR_GFS_T2M.mkdir(parents=True, exist_ok=True)
    parts = []
    for fhr, vt in steps:
        grb = DIR_GFS_T2M / f'gfs_t2m_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
        if not grb.exists() or force:
            try:
                _download_grb2(_build_params(init, fhr), grb)
            except StepNotAvailable:
                logger.warning('  GFS T2m f{:03d} ainda nao publicado (404) — pulando passo', fhr)
                continue
        ds = _open_t2m_grb2(grb).expand_dims(time=[np.datetime64(vt)])
        parts.append(ds.load())
    if not parts:
        logger.warning('GFS T2m valido {} sem passos publicados — dia ignorado.', day)
        return None
    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    save_netcdf(ds_day, nc_path)
    for fhr, _ in steps:
        grb = DIR_GFS_T2M / f'gfs_t2m_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
        if grb.exists():
            grb.unlink()
    logger.info('GFS T2m valido {} salvo: {}', day, nc_path.name)
    return nc_path


def ensure_gfs_t2m_fcst_for_period(
    init: datetime, lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de T2m (K) do GFS para a janela [init, init+lead_hours]."""
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
    logger.info('GFS T2m: {} arquivos | init {:%Y-%m-%d %H}Z + {}h', len(files), init, lead_hours)
    return files
