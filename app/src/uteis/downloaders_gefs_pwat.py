# app/src/uteis/downloaders_gefs_pwat.py
# -*- coding: utf-8 -*-
"""
Downloader GEFS (previsao) de agua precipitavel (PWAT, camada unica) via NOMADS Grib Filter.

Espelha o downloaders_gefs_olr, mas usa a MEDIA DO ENSEMBLE do GEFS (membro `geavg`).
PWAT esta no MESMO arquivo pgrb2a 0.5° das demais variaveis (200/850 hPa/OLR), e
diferente do ULWRF esta disponivel tambem no passo f000 (analise). Reusa os helpers
_download_grb2 / _steps_for_day do downloader GEFS 200 e o parser de GRIB do GFS.
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
# Parser de GRIB do PWAT (model-agnostico) reusado do downloader GFS.
from app.src.uteis.downloaders_gfs_pwat import _open_gfs_pwat as _open_gefs_pwat

logger = get_logger(__name__)

DIR_GEFS_PWAT = DIR_DADOS_BASE / 'GEFS_PWAT'


def _build_params(init: datetime, fhr: int) -> dict:
    return {
        'file': _gefs_file(init, fhr),
        'lev_entire_atmosphere_(considered_as_a_single_layer)': 'on',
        'var_PWAT': 'on',
        'dir': _gefs_dir(init),
    }


def _download_day(init: datetime, day: date, steps: List[Tuple[int, datetime]], force: bool):
    fname = f'gefs_pwat_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
    nc_path = DIR_GEFS_PWAT / fname
    if nc_path.exists() and not force:
        logger.info('GEFS PWAT valido {} (init {}Z) ja existe — pulando.', day, init.hour)
        return nc_path
    DIR_GEFS_PWAT.mkdir(parents=True, exist_ok=True)
    parts = []
    for fhr, vt in steps:
        grb = DIR_GEFS_PWAT / f'gefs_pwat_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
        if not grb.exists() or force:
            try:
                _download_grb2(_build_params(init, fhr), grb)
            except StepNotAvailable:
                logger.warning('  GEFS PWAT f{:03d} ainda nao publicado (404) — pulando passo', fhr)
                continue
        try:
            ds = _open_gefs_pwat(grb).expand_dims(time=[np.datetime64(vt)])
            parts.append(ds.load())
        except Exception as exc:
            logger.warning('PWAT f{:03d} sem mensagem valida — pulando ({})', fhr, exc)
            if grb.exists():
                grb.unlink()
    if not parts:
        logger.warning('GEFS PWAT {} sem passos validos — dia ignorado.', day)
        return None
    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    save_netcdf(ds_day, nc_path)
    for fhr, _ in steps:
        grb = DIR_GEFS_PWAT / f'gefs_pwat_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
        if grb.exists():
            grb.unlink()
    logger.info('GEFS PWAT valido {} salvo: {}', day, nc_path.name)
    return nc_path


def ensure_gefs_pwat_fcst_for_period(
    init: datetime, lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de agua precipitavel (kg/m2) do GEFS (media do ensemble) para
    [init, init+lead_hours]."""
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
    logger.info('GEFS PWAT: {} arquivos | init {:%Y-%m-%d %H}Z + {}h', len(files), init, lead_hours)
    return files
