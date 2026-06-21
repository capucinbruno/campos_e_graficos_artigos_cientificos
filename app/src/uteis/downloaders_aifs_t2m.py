# app/src/uteis/downloaders_aifs_t2m.py
# -*- coding: utf-8 -*-
"""
Downloader AIFS-single (IA deterministica do ECMWF) de temperatura do ar a 2 m.

Espelha o ECMWF HRES T2m (open data / byte-range), reusando `_open_t2m` com o `model` do AIFS.
Um NetCDF por dia valido, variavel 't2m' (K).
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
from app.src.uteis.downloaders_ecmwf_t2m import _open_t2m

logger = get_logger(__name__)

DIR_AIFS_T2M = DIR_DADOS_BASE / 'AIFS_T2M'


def _download_day(init: datetime, day: date, steps: List[Tuple[int, datetime]], force: bool) -> Path:
    fname = f'aifs_t2m_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
    nc_path = DIR_AIFS_T2M / fname
    if nc_path.exists() and not force:
        logger.info('AIFS T2m valido {} (init {}Z) ja existe — pulando.', day, init.hour)
        return nc_path
    DIR_AIFS_T2M.mkdir(parents=True, exist_ok=True)
    tmp = DIR_AIFS_T2M / f'aifs_t2m_{init.strftime("%Y%m%d%H")}_{day.strftime("%Y%m%d")}_tmp.grb2'
    parts = []
    for step, vt in steps:
        try:
            ds = _open_t2m(
                init, step, tmp, model=AIFS_MODEL, stream=AIFS_STREAM, ftype=AIFS_TYPE,
            ).expand_dims(time=[np.datetime64(vt)])
        except StepNotAvailable:
            logger.warning('  AIFS T2m step {:03d}h ainda nao publicado (404) — pulando', step)
            continue
        parts.append(ds.load())
    if tmp.exists():
        tmp.unlink()
    if not parts:
        logger.warning('AIFS T2m valido {} sem passos publicados — dia ignorado.', day)
        return None
    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    save_netcdf(ds_day, nc_path)
    logger.info('AIFS T2m valido {} salvo: {}', day, nc_path.name)
    return nc_path


def ensure_aifs_t2m_fcst_for_period(
    init: datetime, lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de T2m (K) do AIFS-single para [init, init+lead_hours]."""
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
    logger.info('AIFS T2m: {} arquivos | init {:%Y-%m-%d %H}Z + {}h', len(files), init, lead_hours)
    return files
