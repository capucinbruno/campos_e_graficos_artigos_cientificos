# app/src/uteis/downloaders_gfs_hgt250.py
# -*- coding: utf-8 -*-
"""
Downloader GFS (previsao) de altura geopotencial em 250 hPa via NOMADS Grib Filter.

Um NetCDF por dia valido, variavel 'hgt' (m). Espelha o downloaders_gfs_hgt500, trocando
so o nivel (250 hPa). Reusa o parser de HGT (`_open_gfs_hgt500`, agnostico ao nivel).
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
from app.src.uteis.downloaders_gfs_hgt500 import _open_gfs_hgt500

logger = get_logger(__name__)

DIR_GFS_HGT250 = DIR_DADOS_BASE / 'GFS_HGT250'


def _build_params(init: datetime, fhr: int) -> dict:
    return {
        'file': f'gfs.t{init.hour:02d}z.pgrb2.0p25.f{fhr:03d}',
        'lev_250_mb': 'on',
        'var_HGT': 'on',
        'dir': f'/gfs.{init.strftime("%Y%m%d")}/{init.hour:02d}/atmos',
    }


def _download_day(init: datetime, day: date, steps: List[Tuple[int, datetime]], force: bool) -> Path:
    fname = f'gfs_hgt250_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
    nc_path = DIR_GFS_HGT250 / fname
    if nc_path.exists() and not force:
        logger.info('GFS Z250 valido {} (init {}Z) ja existe — pulando.', day, init.hour)
        return nc_path
    DIR_GFS_HGT250.mkdir(parents=True, exist_ok=True)
    parts = []
    for fhr, vt in steps:
        grb = DIR_GFS_HGT250 / f'gfs_hgt250_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
        if not grb.exists() or force:
            try:
                _download_grb2(_build_params(init, fhr), grb)
            except StepNotAvailable:
                logger.warning('  GFS Z250 f{:03d} ainda nao publicado (404) — pulando passo', fhr)
                continue
        ds = _open_gfs_hgt500(grb).expand_dims(time=[np.datetime64(vt)])
        parts.append(ds.load())
    if not parts:
        logger.warning('GFS Z250 valido {} sem passos publicados — dia ignorado.', day)
        return None
    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    save_netcdf(ds_day, nc_path)
    for fhr, _ in steps:
        grb = DIR_GFS_HGT250 / f'gfs_hgt250_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
        if grb.exists():
            grb.unlink()
    logger.info('GFS Z250 valido {} salvo: {}', day, nc_path.name)
    return nc_path


def ensure_gfs_hgt250_fcst_for_period(
    init: datetime, lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de Z250 (m) do GFS para a janela [init, init+lead_hours]."""
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
    logger.info('GFS Z250: {} arquivos | init {:%Y-%m-%d %H}Z + {}h', len(files), init, lead_hours)
    return files
