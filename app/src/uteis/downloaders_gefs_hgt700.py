# app/src/uteis/downloaders_gefs_hgt700.py
# -*- coding: utf-8 -*-
"""
Downloader GEFS (previsao) de altura geopotencial em 700 hPa via NOMADS Grib Filter.

Espelha o downloaders_gefs_tmp850, mas extrai HGT @ 700 hPa (MEDIA DO ENSEMBLE, membro
`geavg`, no mesmo arquivo pgrb2a 0.5°). Um NetCDF por dia valido, variavel 'hgt' (m).
Reusa os helpers do downloader GEFS 200 e o parser de HGT do GFS Z700.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import xarray as xr

from app.common.forecast_download import StepNotAvailable, download_days_parallel, save_netcdf
from app.shared.logger import get_logger
from app.src.uteis.downloaders_gefs_fcst200 import (
    DEFAULT_SYNOPTIC_HOURS,
    DIR_DADOS_BASE,
    _download_grb2,
    _gefs_dir,
    _gefs_file,
    _steps_for_day,
)
# Parser de GRIB do HGT@700 (model-agnostico) reusado do downloader GFS Z700.
from app.src.uteis.downloaders_gfs_hgt700 import _open_gfs_hgt700 as _open_gefs_hgt700

logger = get_logger(__name__)

DIR_GEFS_HGT700 = DIR_DADOS_BASE / 'GEFS_HGT700'


def _build_params(init: datetime, fhr: int) -> dict:
    return {
        'file': _gefs_file(init, fhr),
        'lev_700_mb': 'on',
        'var_HGT': 'on',
        'dir': _gefs_dir(init),
    }


def _download_day(init: datetime, day: date, steps: List[Tuple[int, datetime]], force: bool):
    fname = f'gefs_hgt700_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
    nc_path = DIR_GEFS_HGT700 / fname
    if nc_path.exists() and not force:
        logger.info('GEFS Z700 valido {} (init {}Z) ja existe — pulando.', day, init.hour)
        return nc_path
    DIR_GEFS_HGT700.mkdir(parents=True, exist_ok=True)
    parts = []
    for fhr, vt in steps:
        grb = DIR_GEFS_HGT700 / f'gefs_hgt700_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
        if not grb.exists() or force:
            try:
                _download_grb2(_build_params(init, fhr), grb)
            except StepNotAvailable:
                logger.warning('  GEFS Z700 f{:03d} ainda nao publicado (404) — pulando passo', fhr)
                continue
        ds = _open_gefs_hgt700(grb).expand_dims(time=[np.datetime64(vt)])
        parts.append(ds.load())
    if not parts:
        logger.warning('GEFS Z700 valido {} sem passos publicados — dia ignorado.', day)
        return None
    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    save_netcdf(ds_day, nc_path)
    for fhr, _ in steps:
        grb = DIR_GEFS_HGT700 / f'gefs_hgt700_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
        if grb.exists():
            grb.unlink()
    logger.info('GEFS Z700 valido {} salvo: {}', day, nc_path.name)
    return nc_path


def ensure_gefs_hgt700_fcst_for_period(
    init: datetime, lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de Z700 (m) do GEFS (media do ensemble) para [init, init+lead_hours]."""
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
    logger.info('GEFS Z700: {} arquivos | init {:%Y-%m-%d %H}Z + {}h', len(files), init, lead_hours)
    return files
