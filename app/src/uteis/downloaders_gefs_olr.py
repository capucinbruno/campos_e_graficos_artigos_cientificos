# app/src/uteis/downloaders_gefs_olr.py
# -*- coding: utf-8 -*-
"""
Downloader GEFS (previsao) de OLR (ULWRF no topo da atmosfera) via NOMADS Grib Filter.

Espelha o downloaders_gfs_olr, mas usa a MEDIA DO ENSEMBLE do GEFS (membro `geavg`).
No GEFS, o ULWRF do topo esta no MESMO arquivo pgrb2a 0.5° das variaveis de 200/850 hPa.
Um NetCDF por dia valido com as horas sinoticas, variavel 'olr' (W/m2). Reusa os helpers
_download_grb2 / _steps_for_day do downloader GEFS 200 e o parser de GRIB do GFS.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import xarray as xr

from app.shared.logger import get_logger
from app.src.uteis.downloaders_gefs_fcst200 import (
    DEFAULT_SYNOPTIC_HOURS,
    DIR_DADOS_BASE,
    _download_grb2,
    _gefs_dir,
    _gefs_file,
    _steps_for_day,
)
# Parser de GRIB do ULWRF (model-agnostico) reusado do downloader GFS.
from app.src.uteis.downloaders_gfs_olr import _open_gfs_olr as _open_gefs_olr

logger = get_logger(__name__)

DIR_GEFS_OLR = DIR_DADOS_BASE / 'GEFS_OLR'


def _build_params(init: datetime, fhr: int) -> dict:
    return {
        'file': _gefs_file(init, fhr),
        'lev_top_of_atmosphere': 'on',
        'var_ULWRF': 'on',
        'dir': _gefs_dir(init),
    }


def _download_day(init: datetime, day: date, steps: List[Tuple[int, datetime]], force: bool):
    fname = f'gefs_olr_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
    nc_path = DIR_GEFS_OLR / fname
    if nc_path.exists() and not force:
        logger.info('GEFS OLR valido {} (init {}Z) ja existe — pulando.', day, init.hour)
        return nc_path
    DIR_GEFS_OLR.mkdir(parents=True, exist_ok=True)
    parts = []
    for fhr, vt in steps:
        # f000 (analise) nao tem ULWRF (fluxo medio precisa de intervalo) -> pula
        if fhr == 0:
            continue
        grb = DIR_GEFS_OLR / f'gefs_olr_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
        if not grb.exists() or force:
            _download_grb2(_build_params(init, fhr), grb)
        try:
            ds = _open_gefs_olr(grb).expand_dims(time=[np.datetime64(vt)])
            parts.append(ds.load())
        except Exception as exc:
            logger.warning('OLR f{:03d} sem mensagem valida — pulando ({})', fhr, exc)
            if grb.exists():
                grb.unlink()
    if not parts:
        logger.warning('GEFS OLR {} sem passos validos — dia ignorado.', day)
        return None
    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    ds_day.to_netcdf(nc_path, engine='netcdf4')
    for fhr, _ in steps:
        grb = DIR_GEFS_OLR / f'gefs_olr_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
        if grb.exists():
            grb.unlink()
    logger.info('GEFS OLR valido {} salvo: {}', day, nc_path.name)
    return nc_path


def ensure_gefs_olr_fcst_for_period(
    init: datetime, lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de OLR (W/m2) do GEFS (media do ensemble) para [init, init+lead_hours]."""
    files: List[Path] = []
    end = init + timedelta(hours=lead_hours)
    day = init.date()
    while day <= end.date():
        steps = _steps_for_day(init, day, hours, lead_hours)
        if steps:
            nc = _download_day(init, day, steps, force_redownload)
            if nc is not None:
                files.append(nc)
        day += timedelta(days=1)
    logger.info('GEFS OLR: {} arquivos | init {:%Y-%m-%d %H}Z + {}h', len(files), init, lead_hours)
    return files
