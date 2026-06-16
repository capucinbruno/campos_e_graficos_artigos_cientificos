# app/src/uteis/downloaders_gfs_olr.py
# -*- coding: utf-8 -*-
"""
Downloader GFS (previsao) de OLR (ULWRF no topo da atmosfera) via NOMADS Grib Filter.

Mesma estrutura do downloaders_gfs_fcst200: um NetCDF por dia valido com as horas
sinoticas, variavel 'olr' (W/m2). Reusa os helpers _download_grb2 / _steps_for_day.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import xarray as xr

from app.shared.logger import get_logger
from app.src.uteis.downloaders_gfs_fcst200 import (
    DEFAULT_SYNOPTIC_HOURS,
    DIR_DADOS_BASE,
    _download_grb2,
    _steps_for_day,
)

logger = get_logger(__name__)

DIR_GFS_OLR = DIR_DADOS_BASE / 'GFS_OLR'


def _build_params(init: datetime, fhr: int) -> dict:
    return {
        'file': f'gfs.t{init.hour:02d}z.pgrb2.0p25.f{fhr:03d}',
        'lev_top_of_atmosphere': 'on',
        'var_ULWRF': 'on',
        'dir': f'/gfs.{init.strftime("%Y%m%d")}/{init.hour:02d}/atmos',
    }


def _open_gfs_olr(path: Path) -> xr.Dataset:
    """Abre o GRIB2 do ULWRF (topo) e normaliza para 'olr' em (lat, lon)."""
    ds = None
    last = None
    for fbk in ({'typeOfLevel': 'nominalTop'}, {'stepType': 'avg'}, {}):
        try:
            ds = xr.open_dataset(
                path, engine='cfgrib', backend_kwargs={'indexpath': ''}, filter_by_keys=fbk)
            if len(ds.data_vars):
                break
        except Exception as exc:
            last = exc
            ds = None
    if ds is None or not len(ds.data_vars):
        raise RuntimeError(f'cfgrib nao conseguiu abrir o OLR do GFS: {last}')

    ren = {}
    for name in list(ds.dims) + list(ds.coords):
        low = name.lower()
        if low == 'latitude' and 'lat' not in ds.dims:
            ren[name] = 'lat'
        elif low == 'longitude' and 'lon' not in ds.dims:
            ren[name] = 'lon'
    if ren:
        ds = ds.rename(ren)

    var = next((v for v in ds.data_vars if 'ulwrf' in v.lower() or v.lower() == 'olr'),
               list(ds.data_vars)[0])
    da = ds[var].rename('olr')
    for coord in ('time', 'step', 'valid_time', 'nominalTop', 'atmosphere', 'level'):
        if coord in da.coords and coord not in da.dims:
            da = da.drop_vars(coord, errors='ignore')
    da.attrs['units'] = 'W m-2'
    return da.to_dataset(name='olr')


def _download_day(init: datetime, day: date, steps: List[Tuple[int, datetime]], force: bool):
    fname = f'gfs_olr_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
    nc_path = DIR_GFS_OLR / fname
    if nc_path.exists() and not force:
        logger.info('GFS OLR valido {} (init {}Z) ja existe — pulando.', day, init.hour)
        return nc_path
    DIR_GFS_OLR.mkdir(parents=True, exist_ok=True)
    parts = []
    for fhr, vt in steps:
        # f000 (analise) nao tem ULWRF (fluxo medio precisa de intervalo) -> pula
        if fhr == 0:
            continue
        grb = DIR_GFS_OLR / f'gfs_olr_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
        if not grb.exists() or force:
            _download_grb2(_build_params(init, fhr), grb)
        try:
            ds = _open_gfs_olr(grb).expand_dims(time=[np.datetime64(vt)])
            parts.append(ds.load())
        except Exception as exc:
            logger.warning('OLR f{:03d} sem mensagem valida — pulando ({})', fhr, exc)
            if grb.exists():
                grb.unlink()
    if not parts:
        logger.warning('GFS OLR {} sem passos validos — dia ignorado.', day)
        return None
    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    ds_day.to_netcdf(nc_path, engine='netcdf4')
    for fhr, _ in steps:
        grb = DIR_GFS_OLR / f'gfs_olr_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
        if grb.exists():
            grb.unlink()
    logger.info('GFS OLR valido {} salvo: {}', day, nc_path.name)
    return nc_path


def ensure_gfs_olr_fcst_for_period(
    init: datetime, lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de OLR (W/m2) do GFS para a janela [init, init+lead_hours]."""
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
    logger.info('GFS OLR: {} arquivos | init {:%Y-%m-%d %H}Z + {}h', len(files), init, lead_hours)
    return files
