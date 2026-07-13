# app/src/uteis/downloaders_gfs_pwat.py
# -*- coding: utf-8 -*-
"""
Downloader GFS (previsao) de agua precipitavel (PWAT, camada unica) via NOMADS Grib Filter.

Mesma estrutura do downloaders_gfs_olr, mas PWAT (diferente de ULWRF) esta disponivel
tambem no passo f000 (analise) -- nao precisa pular esse passo.
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

DIR_GFS_PWAT = DIR_DADOS_BASE / 'GFS_PWAT'


def _build_params(init: datetime, fhr: int) -> dict:
    return {
        'file': f'gfs.t{init.hour:02d}z.pgrb2.0p25.f{fhr:03d}',
        'lev_entire_atmosphere_(considered_as_a_single_layer)': 'on',
        'var_PWAT': 'on',
        'dir': f'/gfs.{init.strftime("%Y%m%d")}/{init.hour:02d}/atmos',
    }


def _open_gfs_pwat(path: Path) -> xr.Dataset:
    """Abre o GRIB2 do PWAT e normaliza para 'pwat' (kg/m2) em (lat, lon)."""
    from app.common.forecast_download import GRIB_NETCDF_LOCK
    ds = None
    last = None
    with GRIB_NETCDF_LOCK:  # ecCodes nao e thread-safe entre downloads paralelos
        for fbk in ({'typeOfLevel': 'atmosphereSingleLayer'}, {}):
            try:
                ds = xr.open_dataset(
                    path, engine='cfgrib', backend_kwargs={'indexpath': ''}, filter_by_keys=fbk)
                if len(ds.data_vars):
                    ds = ds.load()
                    break
            except Exception as exc:
                last = exc
                ds = None
    if ds is None or not len(ds.data_vars):
        raise RuntimeError(f'cfgrib nao conseguiu abrir o PWAT do GFS: {last}')

    ren = {}
    for name in list(ds.dims) + list(ds.coords):
        low = name.lower()
        if low == 'latitude' and 'lat' not in ds.dims:
            ren[name] = 'lat'
        elif low == 'longitude' and 'lon' not in ds.dims:
            ren[name] = 'lon'
    if ren:
        ds = ds.rename(ren)

    var = next((v for v in ds.data_vars if 'pwat' in v.lower()), list(ds.data_vars)[0])
    da = ds[var].rename('pwat')
    for coord in ('time', 'step', 'valid_time', 'atmosphere', 'level'):
        if coord in da.coords and coord not in da.dims:
            da = da.drop_vars(coord, errors='ignore')
    da.attrs['units'] = 'kg m-2'
    return da.to_dataset(name='pwat')


def _download_day(init: datetime, day: date, steps: List[Tuple[int, datetime]], force: bool):
    fname = f'gfs_pwat_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
    nc_path = DIR_GFS_PWAT / fname
    if nc_path.exists() and not force:
        logger.info('GFS PWAT valido {} (init {}Z) ja existe — pulando.', day, init.hour)
        return nc_path
    DIR_GFS_PWAT.mkdir(parents=True, exist_ok=True)
    parts = []
    for fhr, vt in steps:
        grb = DIR_GFS_PWAT / f'gfs_pwat_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
        if not grb.exists() or force:
            try:
                _download_grb2(_build_params(init, fhr), grb)
            except StepNotAvailable:
                logger.warning('  GFS PWAT f{:03d} ainda nao publicado (404) — pulando passo', fhr)
                continue
        try:
            ds = _open_gfs_pwat(grb).expand_dims(time=[np.datetime64(vt)])
            parts.append(ds.load())
        except Exception as exc:
            logger.warning('PWAT f{:03d} sem mensagem valida — pulando ({})', fhr, exc)
            if grb.exists():
                grb.unlink()
    if not parts:
        logger.warning('GFS PWAT {} sem passos validos — dia ignorado.', day)
        return None
    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    save_netcdf(ds_day, nc_path)
    for fhr, _ in steps:
        grb = DIR_GFS_PWAT / f'gfs_pwat_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
        if grb.exists():
            grb.unlink()
    logger.info('GFS PWAT valido {} salvo: {}', day, nc_path.name)
    return nc_path


def ensure_gfs_pwat_fcst_for_period(
    init: datetime, lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de agua precipitavel (kg/m2) do GFS para [init, init+lead_hours]."""
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
    logger.info('GFS PWAT: {} arquivos | init {:%Y-%m-%d %H}Z + {}h', len(files), init, lead_hours)
    return files
