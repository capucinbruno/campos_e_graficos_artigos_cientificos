# app/src/uteis/downloaders_gefs_mslp.py
# -*- coding: utf-8 -*-
"""
Downloader GEFS (previsao) de PNMM (pressao ao nivel medio do mar) via NOMADS Grib Filter.

Espelha o downloaders_gefs_tmp850, mas pega a variavel PRMSL (mean sea level) do MESMO
arquivo pgrb2a 0.5° (membro `geavg`, media do ensemble). Um NetCDF por dia valido com as
horas sinoticas, variavel `prmsl` (Pa — o daily_mslp_on_grid converte p/ hPa). Usado pelo
s38/s39 (campo tmp850_mslp) para desenhar as isolinhas de PNMM na parte de PREVISAO.

Fonte: https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p50a.pl
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

logger = get_logger(__name__)

DIR_GEFS_MSLP = DIR_DADOS_BASE / 'GEFS_MSLP'


def _build_params(init: datetime, fhr: int) -> dict:
    return {
        'file': _gefs_file(init, fhr),
        'lev_mean_sea_level': 'on',
        'var_PRMSL': 'on',
        'dir': _gefs_dir(init),
    }


def _open_gefs_mslp(path: Path) -> xr.Dataset:
    """Abre o GRIB2 do PRMSL (mean sea level) e normaliza para 'prmsl' (Pa) em (lat, lon)."""
    from app.common.forecast_download import GRIB_NETCDF_LOCK
    with GRIB_NETCDF_LOCK:  # ecCodes nao e thread-safe entre downloads paralelos
        ds = xr.open_dataset(
            path, engine='cfgrib', backend_kwargs={'indexpath': ''},
            filter_by_keys={'typeOfLevel': 'meanSea'},
        ).load()
    ren = {}
    for name in list(ds.dims) + list(ds.coords):
        low = name.lower()
        if low == 'latitude' and 'lat' not in ds.dims:
            ren[name] = 'lat'
        elif low == 'longitude' and 'lon' not in ds.dims:
            ren[name] = 'lon'
    if ren:
        ds = ds.rename(ren)
    var = next((v for v in ('prmsl', 'msl', 'mslet', 'psl') if v in ds.data_vars),
               list(ds.data_vars)[0])
    da = ds[var].rename('prmsl')
    for coord in ('time', 'step', 'valid_time', 'meanSea', 'level'):
        if coord in da.coords and coord not in da.dims:
            da = da.drop_vars(coord, errors='ignore')
    da.attrs['units'] = 'Pa'
    return da.to_dataset(name='prmsl')


def _download_day(init: datetime, day: date, steps: List[Tuple[int, datetime]], force: bool):
    fname = f'gefs_mslp_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
    nc_path = DIR_GEFS_MSLP / fname
    if nc_path.exists() and not force:
        logger.info('GEFS MSLP valido {} (init {}Z) ja existe — pulando.', day, init.hour)
        return nc_path
    DIR_GEFS_MSLP.mkdir(parents=True, exist_ok=True)
    parts = []
    for fhr, vt in steps:
        grb = DIR_GEFS_MSLP / f'gefs_mslp_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
        if not grb.exists() or force:
            try:
                _download_grb2(_build_params(init, fhr), grb)
            except StepNotAvailable:
                logger.warning('  GEFS MSLP f{:03d} ainda nao publicado (404) — pulando passo', fhr)
                continue
        ds = _open_gefs_mslp(grb).expand_dims(time=[np.datetime64(vt)])
        parts.append(ds.load())
    if not parts:
        logger.warning('GEFS MSLP valido {} sem passos publicados — dia ignorado.', day)
        return None
    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    save_netcdf(ds_day, nc_path)
    for fhr, _ in steps:
        grb = DIR_GEFS_MSLP / f'gefs_mslp_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
        if grb.exists():
            grb.unlink()
    logger.info('GEFS MSLP valido {} salvo: {}', day, nc_path.name)
    return nc_path


def ensure_gefs_mslp_fcst_for_period(
    init: datetime, lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de PNMM (Pa) do GEFS (media do ensemble) para [init, init+lead_hours]."""
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
    logger.info('GEFS MSLP: {} arquivos | init {:%Y-%m-%d %H}Z + {}h', len(files), init, lead_hours)
    return files
