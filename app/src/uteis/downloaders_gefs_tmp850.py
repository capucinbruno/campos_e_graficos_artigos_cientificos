# app/src/uteis/downloaders_gefs_tmp850.py
# -*- coding: utf-8 -*-
"""
Downloader GEFS (previsao) de temperatura em 850 hPa via NOMADS Grib Filter.

Espelha o downloaders_gfs_tmp850, mas usa a MEDIA DO ENSEMBLE do GEFS (membro `geavg`),
no mesmo arquivo pgrb2a 0.5°. Um NetCDF por dia valido com as horas sinoticas, variavel
't' (Kelvin — o s34 normaliza p/ °C). Reusa os helpers do downloader GEFS 200 e o parser
de GRIB do GFS.
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
# Parser de GRIB do TMP@850 (model-agnostico) reusado do downloader GFS.
from app.src.uteis.downloaders_gfs_tmp850 import _open_gfs_tmp850 as _open_gefs_tmp850

logger = get_logger(__name__)

DIR_GEFS_TMP850 = DIR_DADOS_BASE / 'GEFS_TMP850'


def _build_params(init: datetime, fhr: int) -> dict:
    return {
        'file': _gefs_file(init, fhr),
        'lev_850_mb': 'on',
        'var_TMP': 'on',
        'dir': _gefs_dir(init),
    }


def _download_day(init: datetime, day: date, steps: List[Tuple[int, datetime]], force: bool) -> Path:
    fname = f'gefs_tmp850_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
    nc_path = DIR_GEFS_TMP850 / fname
    if nc_path.exists() and not force:
        logger.info('GEFS T850 valido {} (init {}Z) ja existe — pulando.', day, init.hour)
        return nc_path
    DIR_GEFS_TMP850.mkdir(parents=True, exist_ok=True)
    parts = []
    for fhr, vt in steps:
        grb = DIR_GEFS_TMP850 / f'gefs_tmp850_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
        if not grb.exists() or force:
            _download_grb2(_build_params(init, fhr), grb)
        ds = _open_gefs_tmp850(grb).expand_dims(time=[np.datetime64(vt)])
        parts.append(ds.load())
    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    ds_day.to_netcdf(nc_path, engine='netcdf4')
    for fhr, _ in steps:
        grb = DIR_GEFS_TMP850 / f'gefs_tmp850_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
        if grb.exists():
            grb.unlink()
    logger.info('GEFS T850 valido {} salvo: {}', day, nc_path.name)
    return nc_path


def ensure_gefs_tmp850_fcst_for_period(
    init: datetime, lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de T850 (K) do GEFS (media do ensemble) para [init, init+lead_hours]."""
    files: List[Path] = []
    end = init + timedelta(hours=lead_hours)
    day = init.date()
    while day <= end.date():
        steps = _steps_for_day(init, day, hours, lead_hours)
        if steps:
            files.append(_download_day(init, day, steps, force_redownload))
        day += timedelta(days=1)
    logger.info('GEFS T850: {} arquivos | init {:%Y-%m-%d %H}Z + {}h', len(files), init, lead_hours)
    return files
