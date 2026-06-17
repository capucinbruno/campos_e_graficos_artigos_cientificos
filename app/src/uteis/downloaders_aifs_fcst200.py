# app/src/uteis/downloaders_aifs_fcst200.py
# -*- coding: utf-8 -*-
"""
Downloader AIFS-single (previsao DETERMINISTICA de IA do ECMWF) de u/v/hgt em 200 hPa.

O AIFS e o modelo de IA do ECMWF, publicado no MESMO open data do IFS (data.ecmwf.int),
sob o caminho `aifs-single/0p25/oper` e com a MESMA nomenclatura `oper-fc` + acesso por
byte-range. Por isso reusa toda a maquinaria do downloader ECMWF HRES, mudando so o `model`.

Alcance: 6-horario ate 360 h (15 dias). Variaveis: tem u/v/HGT@200 e T@850, mas NAO tem OLR
(o AIFS nao emite radiacao no topo) — por isso o s34 gera 4 dos 5 mapas para o AIFS.

Gera um NetCDF por dia valido (u/v/hgt 200 hPa), mesma estrutura dos demais downloaders.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import xarray as xr

from app.common.forecast_download import StepNotAvailable, download_days_parallel, save_netcdf
from app.shared.logger import get_logger
from app.src.uteis.downloaders_ecmwf_fcst200 import (
    DEFAULT_SYNOPTIC_HOURS,
    DIR_DADOS_BASE,
    _open_uvhgt200,
    _steps_for_day,
)

logger = get_logger(__name__)

AIFS_MODEL = 'aifs-single/0p25'  # IA deterministica do ECMWF
AIFS_STREAM = 'oper'
AIFS_TYPE = 'fc'

DIR_AIFS_FCST200 = DIR_DADOS_BASE / 'AIFS_FCST200'


def _download_day(init: datetime, day: date, steps: List[Tuple[int, datetime]], force: bool) -> Path:
    fname = f'aifs_fcst200_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
    nc_path = DIR_AIFS_FCST200 / fname
    if nc_path.exists() and not force:
        logger.info('AIFS 200 hPa valido {} (init {}Z) ja existe — pulando.', day, init.hour)
        return nc_path
    DIR_AIFS_FCST200.mkdir(parents=True, exist_ok=True)
    tmp = DIR_AIFS_FCST200 / f'aifs_{init.strftime("%Y%m%d%H")}_{day.strftime("%Y%m%d")}_tmp.grb2'
    parts = []
    for step, vt in steps:
        try:
            ds = _open_uvhgt200(
                init, step, tmp, model=AIFS_MODEL, stream=AIFS_STREAM, ftype=AIFS_TYPE,
            ).expand_dims(time=[np.datetime64(vt)])
        except StepNotAvailable:
            logger.warning('  AIFS step {:03d}h ainda nao publicado (404) — pulando', step)
            continue
        parts.append(ds.load())
    if tmp.exists():
        tmp.unlink()
    if not parts:
        logger.warning('AIFS 200 hPa valido {} sem passos publicados — dia ignorado.', day)
        return None
    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    save_netcdf(ds_day, nc_path)
    logger.info('AIFS 200 hPa valido {} salvo: {}', day, nc_path.name)
    return nc_path


def ensure_aifs_fcst200_for_period(
    init: datetime, lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios (u/v/hgt 200 hPa) do AIFS-single para [init, init+lead_hours]."""
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
    logger.info('AIFS FCST200: {} arquivos | init {:%Y-%m-%d %H}Z + {}h', len(files), init, lead_hours)
    return files
