# app/src/uteis/downloaders_aifs_hgt250.py
# -*- coding: utf-8 -*-
"""
Downloader AIFS-single (IA deterministica do ECMWF) de altura geopotencial em 250 hPa.

Espelha o downloaders_aifs_hgt500, trocando so o nivel (250 hPa). Reusa o opener generico
`_open_hgt500(..., level=250, model=AIFS)`. Um NetCDF por dia valido, variavel 'hgt' (m).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import xarray as xr

from app.common.forecast_download import StepNotAvailable, download_days_parallel, save_netcdf
from app.shared.logger import get_logger
from app.src.uteis.downloaders_aifs_fcst200 import AIFS_MODEL, AIFS_STREAM, AIFS_TYPE
from app.src.uteis.downloaders_ecmwf_fcst200 import DEFAULT_SYNOPTIC_HOURS, DIR_DADOS_BASE, _steps_for_day
from app.src.uteis.downloaders_ecmwf_hgt500 import _open_hgt500

logger = get_logger(__name__)

DIR_AIFS_HGT250 = DIR_DADOS_BASE / 'AIFS_HGT250'


def _download_day(init: datetime, day: date, steps: List[Tuple[int, datetime]], force: bool) -> Path:
    fname = f'aifs_hgt250_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
    nc_path = DIR_AIFS_HGT250 / fname
    if nc_path.exists() and not force:
        logger.info('AIFS Z250 valido {} (init {}Z) ja existe — pulando.', day, init.hour)
        return nc_path
    DIR_AIFS_HGT250.mkdir(parents=True, exist_ok=True)
    tmp = DIR_AIFS_HGT250 / f'aifs_z250_{init.strftime("%Y%m%d%H")}_{day.strftime("%Y%m%d")}_tmp.grb2'
    parts = []
    for step, vt in steps:
        try:
            ds = _open_hgt500(
                init, step, tmp, model=AIFS_MODEL, stream=AIFS_STREAM, ftype=AIFS_TYPE, level=250,
            ).expand_dims(time=[np.datetime64(vt)])
        except StepNotAvailable:
            logger.warning('  AIFS Z250 step {:03d}h ainda nao publicado (404) — pulando', step)
            continue
        parts.append(ds.load())
    if tmp.exists():
        tmp.unlink()
    if not parts:
        logger.warning('AIFS Z250 valido {} sem passos publicados — dia ignorado.', day)
        return None
    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    save_netcdf(ds_day, nc_path)
    logger.info('AIFS Z250 valido {} salvo: {}', day, nc_path.name)
    return nc_path


def ensure_aifs_hgt250_fcst_for_period(
    init: datetime, lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de Z250 (m) do AIFS-single para [init, init+lead_hours]."""
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
    logger.info('AIFS Z250: {} arquivos | init {:%Y-%m-%d %H}Z + {}h', len(files), init, lead_hours)
    return files
