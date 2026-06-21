# app/src/uteis/downloaders_gefs_t2m.py
# -*- coding: utf-8 -*-
"""
Downloader GEFS (previsao) de temperatura do ar a 2 m via NOMADS Grib Filter (media geavg).

Espelha o downloaders_gfs_t2m (variavel de superficie heightAboveGround=2). Um NetCDF por dia
valido, variavel 't2m' (K). Reusa os helpers do GEFS 200 e o parser de T2m do GFS.
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
from app.src.uteis.downloaders_gfs_t2m import _open_t2m_grb2

logger = get_logger(__name__)

DIR_GEFS_T2M = DIR_DADOS_BASE / 'GEFS_T2M'


def _build_params(init: datetime, fhr: int) -> dict:
    return {
        'file': _gefs_file(init, fhr),
        'lev_2_m_above_ground': 'on',
        'var_TMP': 'on',
        'dir': _gefs_dir(init),
    }


def _download_day(init: datetime, day: date, steps: List[Tuple[int, datetime]], force: bool):
    fname = f'gefs_t2m_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
    nc_path = DIR_GEFS_T2M / fname
    if nc_path.exists() and not force:
        logger.info('GEFS T2m valido {} (init {}Z) ja existe — pulando.', day, init.hour)
        return nc_path
    DIR_GEFS_T2M.mkdir(parents=True, exist_ok=True)
    parts = []
    for fhr, vt in steps:
        grb = DIR_GEFS_T2M / f'gefs_t2m_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
        if not grb.exists() or force:
            try:
                _download_grb2(_build_params(init, fhr), grb)
            except StepNotAvailable:
                logger.warning('  GEFS T2m f{:03d} ainda nao publicado (404) — pulando passo', fhr)
                continue
        ds = _open_t2m_grb2(grb).expand_dims(time=[np.datetime64(vt)])
        parts.append(ds.load())
    if not parts:
        logger.warning('GEFS T2m valido {} sem passos publicados — dia ignorado.', day)
        return None
    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    save_netcdf(ds_day, nc_path)
    for fhr, _ in steps:
        grb = DIR_GEFS_T2M / f'gefs_t2m_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
        if grb.exists():
            grb.unlink()
    logger.info('GEFS T2m valido {} salvo: {}', day, nc_path.name)
    return nc_path


def ensure_gefs_t2m_fcst_for_period(
    init: datetime, lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de T2m (K) do GEFS (media do ensemble) para [init, init+lead_hours]."""
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
    logger.info('GEFS T2m: {} arquivos | init {:%Y-%m-%d %H}Z + {}h', len(files), init, lead_hours)
    return files
