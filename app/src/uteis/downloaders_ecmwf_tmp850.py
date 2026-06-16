# app/src/uteis/downloaders_ecmwf_tmp850.py
# -*- coding: utf-8 -*-
"""
Downloader ECMWF HRES (previsao) de temperatura em 850 hPa via ECMWF Open Data (byte-range).

Espelha o downloaders_ecmwf_fcst200, mas extrai `t` @ 850 hPa. Um NetCDF por dia valido com
as horas sinoticas, variavel 't' (Kelvin — o s34 normaliza p/ °C). Reusa os helpers de acesso
ao open data (index + byte-range) do downloader ECMWF 200.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import xarray as xr

from app.shared.logger import get_logger
from app.src.uteis.downloaders_ecmwf_fcst200 import (
    DEFAULT_SYNOPTIC_HOURS,
    DIR_DADOS_BASE,
    _steps_for_day,
    ecmwf_grib_url,
    fetch_index,
    match_record,
    open_grib_bytes,
    range_bytes,
)

logger = get_logger(__name__)

DIR_ECMWF_TMP850 = DIR_DADOS_BASE / 'ECMWF_TMP850'


def _open_t850(init: datetime, step: int, tmp_path: Path) -> xr.Dataset:
    recs = fetch_index(init, step)
    raw = range_bytes(ecmwf_grib_url(init, step), match_record(recs, 't', 850))
    ds = open_grib_bytes(raw, tmp_path)
    da = ds['t'] if 't' in ds.data_vars else ds[list(ds.data_vars)[0]].rename('t')
    for coord in ('time', 'step', 'valid_time', 'isobaricInhPa', 'level', 'heightAboveGround'):
        if coord in da.coords and coord not in da.dims:
            da = da.drop_vars(coord, errors='ignore')
    da.attrs['units'] = 'K'
    return da.to_dataset(name='t')


def _download_day(init: datetime, day: date, steps: List[Tuple[int, datetime]], force: bool) -> Path:
    fname = f'ecmwf_tmp850_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
    nc_path = DIR_ECMWF_TMP850 / fname
    if nc_path.exists() and not force:
        logger.info('ECMWF T850 valido {} (init {}Z) ja existe — pulando.', day, init.hour)
        return nc_path
    DIR_ECMWF_TMP850.mkdir(parents=True, exist_ok=True)
    tmp = DIR_ECMWF_TMP850 / f'ecmwf_t850_{init.strftime("%Y%m%d%H")}_tmp.grb2'
    parts = []
    for step, vt in steps:
        ds = _open_t850(init, step, tmp).expand_dims(time=[np.datetime64(vt)])
        parts.append(ds.load())
    if tmp.exists():
        tmp.unlink()
    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    ds_day.to_netcdf(nc_path, engine='netcdf4')
    logger.info('ECMWF T850 valido {} salvo: {}', day, nc_path.name)
    return nc_path


def ensure_ecmwf_tmp850_fcst_for_period(
    init: datetime, lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de T850 (K) do ECMWF HRES para [init, init+lead_hours]."""
    files: List[Path] = []
    end = init + timedelta(hours=lead_hours)
    day = init.date()
    while day <= end.date():
        steps = _steps_for_day(init, day, hours, lead_hours)
        if steps:
            files.append(_download_day(init, day, steps, force_redownload))
        day += timedelta(days=1)
    logger.info('ECMWF T850: {} arquivos | init {:%Y-%m-%d %H}Z + {}h', len(files), init, lead_hours)
    return files
