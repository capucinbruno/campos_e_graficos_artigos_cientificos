# app/src/uteis/downloaders_aifs_ens.py
# -*- coding: utf-8 -*-
"""
Downloader AIFS-ENS (ensemble de IA do ECMWF) — MEDIA DOS 50 MEMBROS.

Espelha o ECMWF-ENS: no open data, o AIFS-ENS expoe os membros perturbados (stream `enfo`,
type `pf`, `number` 1..50) em `aifs-ens/0p25/enfo` — sem media pronta. Calcula-se a media
baixando cada membro por byte-range e mediando (reusa `_ens_mean_2d`, com paralelismo por
membro `ECMWF_ENS_WORKERS` e nº de membros `ECMWF_ENS_MEMBERS`).

Alcance: 6-horario ate 360 h (15 dias). Tem u/v/HGT@200 e T@850, mas NAO tem OLR (o AIFS nao
emite radiacao no topo) — o s34 gera 4 dos 5 mapas para o AIFS-ENS.

Obs.: o open data tambem expoe o controle (`enfo-cf`); por consistencia com o ECMWF-ENS a
media usa os 50 perturbados (a diferenca vs incluir o controle e desprezivel).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import xarray as xr

from app.common.forecast_download import StepNotAvailable
from app.shared.logger import get_logger
from app.src.uteis.downloaders_ecmwf_ens import _ens_mean_2d, _n_members
from app.src.uteis.downloaders_ecmwf_fcst200 import (
    DEFAULT_SYNOPTIC_HOURS,
    DIR_DADOS_BASE,
    _steps_for_day,
    ecmwf_grib_url,
    fetch_index,
)

logger = get_logger(__name__)

AIFS_ENS_MODEL = 'aifs-ens/0p25'
AIFS_ENS_STREAM = 'enfo'
AIFS_ENS_TYPE = 'pf'   # membros perturbados (1..50)

DIR_AIFS_ENS_FCST200 = DIR_DADOS_BASE / 'AIFS_ENS_FCST200'
DIR_AIFS_ENS_TMP850 = DIR_DADOS_BASE / 'AIFS_ENS_TMP850'
DIR_AIFS_ENS_HGT500 = DIR_DADOS_BASE / 'AIFS_ENS_HGT500'
DIR_AIFS_ENS_HGT250 = DIR_DADOS_BASE / 'AIFS_ENS_HGT250'
DIR_AIFS_ENS_UV250 = DIR_DADOS_BASE / 'AIFS_ENS_UV250'
DIR_AIFS_ENS_UV850 = DIR_DADOS_BASE / 'AIFS_ENS_UV850'
DIR_AIFS_ENS_T2M = DIR_DADOS_BASE / 'AIFS_ENS_T2M'
_G = 9.80665  # AIFS-ENS expoe `z` (geopotencial m2/s2); hgt = z/g (o single ja traz `gh` em m)

_MKW = {'stream': AIFS_ENS_STREAM, 'ftype': AIFS_ENS_TYPE, 'model': AIFS_ENS_MODEL}


def _download_day_200(init: datetime, day: date, steps: List[Tuple[int, datetime]], force: bool) -> Path:
    fname = f'aifs_ens_fcst200_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
    nc_path = DIR_AIFS_ENS_FCST200 / fname
    if nc_path.exists() and not force:
        logger.info('AIFS-ENS 200 hPa valido {} (init {}Z) ja existe — pulando.', day, init.hour)
        return nc_path
    DIR_AIFS_ENS_FCST200.mkdir(parents=True, exist_ok=True)
    parts = []
    for step, vt in steps:
        try:
            recs = fetch_index(init, step, **_MKW)
            grib_url = ecmwf_grib_url(init, step, **_MKW)
            logger.info('  AIFS-ENS init {}Z step {:03d}h: media de {} membros (u/v/gh@200)',
                        init.hour, step, _n_members())
            u, lat, lon = _ens_mean_2d(init, step, 'u', 200, DIR_AIFS_ENS_FCST200, recs, grib_url)
            v, _, _ = _ens_mean_2d(init, step, 'v', 200, DIR_AIFS_ENS_FCST200, recs, grib_url)
            z, _, _ = _ens_mean_2d(init, step, 'z', 200, DIR_AIFS_ENS_FCST200, recs, grib_url)
        except StepNotAvailable:
            logger.warning('  AIFS-ENS step {:03d}h ainda nao publicado (404) — pulando', step)
            continue
        ds = xr.Dataset(
            {'u': (('lat', 'lon'), u), 'v': (('lat', 'lon'), v), 'hgt': (('lat', 'lon'), z / _G)},
            coords={'lat': lat, 'lon': lon},
        ).expand_dims(time=[np.datetime64(vt)])
        parts.append(ds)
    if not parts:
        logger.warning('AIFS-ENS 200 hPa valido {} sem passos publicados — dia ignorado.', day)
        return None
    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    ds_day.to_netcdf(nc_path, engine='netcdf4')
    logger.info('AIFS-ENS 200 hPa valido {} salvo: {}', day, nc_path.name)
    return nc_path


def ensure_aifs_ens_fcst200_for_period(
    init: datetime, lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios (u/v/hgt 200 hPa) da MEDIA do AIFS-ENS para [init, init+lead_hours]."""
    files: List[Path] = []
    end = init + timedelta(hours=lead_hours)
    day = init.date()
    while day <= end.date():
        steps = _steps_for_day(init, day, hours, lead_hours)
        if steps:
            nc = _download_day_200(init, day, steps, force_redownload)
            if nc is not None:
                files.append(nc)
        day += timedelta(days=1)
    logger.info('AIFS-ENS FCST200: {} arquivos | init {:%Y-%m-%d %H}Z + {}h', len(files), init, lead_hours)
    return files


def _download_day_t850(init: datetime, day: date, steps: List[Tuple[int, datetime]], force: bool) -> Path:
    fname = f'aifs_ens_tmp850_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
    nc_path = DIR_AIFS_ENS_TMP850 / fname
    if nc_path.exists() and not force:
        logger.info('AIFS-ENS T850 valido {} (init {}Z) ja existe — pulando.', day, init.hour)
        return nc_path
    DIR_AIFS_ENS_TMP850.mkdir(parents=True, exist_ok=True)
    parts = []
    for step, vt in steps:
        try:
            t, lat, lon = _ens_mean_2d(init, step, 't', 850, DIR_AIFS_ENS_TMP850, **_MKW)
        except StepNotAvailable:
            logger.warning('  AIFS-ENS T850 step {:03d}h ainda nao publicado (404) — pulando', step)
            continue
        ds = xr.Dataset({'t': (('lat', 'lon'), t)}, coords={'lat': lat, 'lon': lon})
        ds['t'].attrs['units'] = 'K'
        parts.append(ds.expand_dims(time=[np.datetime64(vt)]))
    if not parts:
        logger.warning('AIFS-ENS T850 valido {} sem passos publicados — dia ignorado.', day)
        return None
    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    ds_day.to_netcdf(nc_path, engine='netcdf4')
    logger.info('AIFS-ENS T850 valido {} salvo: {}', day, nc_path.name)
    return nc_path


def ensure_aifs_ens_tmp850_fcst_for_period(
    init: datetime, lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de T850 (K) da MEDIA do AIFS-ENS para [init, init+lead_hours]."""
    files: List[Path] = []
    end = init + timedelta(hours=lead_hours)
    day = init.date()
    while day <= end.date():
        steps = _steps_for_day(init, day, hours, lead_hours)
        if steps:
            nc = _download_day_t850(init, day, steps, force_redownload)
            if nc is not None:
                files.append(nc)
        day += timedelta(days=1)
    logger.info('AIFS-ENS T850: {} arquivos | init {:%Y-%m-%d %H}Z + {}h', len(files), init, lead_hours)
    return files


def _download_day_hgt500(init: datetime, day: date, steps: List[Tuple[int, datetime]], force: bool) -> Path:
    fname = f'aifs_ens_hgt500_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
    nc_path = DIR_AIFS_ENS_HGT500 / fname
    if nc_path.exists() and not force:
        logger.info('AIFS-ENS Z500 valido {} (init {}Z) ja existe — pulando.', day, init.hour)
        return nc_path
    DIR_AIFS_ENS_HGT500.mkdir(parents=True, exist_ok=True)
    parts = []
    for step, vt in steps:
        try:
            z, lat, lon = _ens_mean_2d(init, step, 'z', 500, DIR_AIFS_ENS_HGT500, **_MKW)
        except StepNotAvailable:
            logger.warning('  AIFS-ENS Z500 step {:03d}h ainda nao publicado (404) — pulando', step)
            continue
        ds = xr.Dataset({'hgt': (('lat', 'lon'), z / _G)}, coords={'lat': lat, 'lon': lon})
        ds['hgt'].attrs['units'] = 'm'
        parts.append(ds.expand_dims(time=[np.datetime64(vt)]))
    if not parts:
        logger.warning('AIFS-ENS Z500 valido {} sem passos publicados — dia ignorado.', day)
        return None
    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    ds_day.to_netcdf(nc_path, engine='netcdf4')
    logger.info('AIFS-ENS Z500 valido {} salvo: {}', day, nc_path.name)
    return nc_path


def ensure_aifs_ens_hgt500_fcst_for_period(
    init: datetime, lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de Z500 (m) da MEDIA do AIFS-ENS para [init, init+lead_hours]."""
    files: List[Path] = []
    end = init + timedelta(hours=lead_hours)
    day = init.date()
    while day <= end.date():
        steps = _steps_for_day(init, day, hours, lead_hours)
        if steps:
            nc = _download_day_hgt500(init, day, steps, force_redownload)
            if nc is not None:
                files.append(nc)
        day += timedelta(days=1)
    logger.info('AIFS-ENS Z500: {} arquivos | init {:%Y-%m-%d %H}Z + {}h', len(files), init, lead_hours)
    return files


def _download_day_hgt250(init: datetime, day: date, steps: List[Tuple[int, datetime]], force: bool) -> Path:
    fname = f'aifs_ens_hgt250_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
    nc_path = DIR_AIFS_ENS_HGT250 / fname
    if nc_path.exists() and not force:
        logger.info('AIFS-ENS Z250 valido {} (init {}Z) ja existe — pulando.', day, init.hour)
        return nc_path
    DIR_AIFS_ENS_HGT250.mkdir(parents=True, exist_ok=True)
    parts = []
    for step, vt in steps:
        try:
            z, lat, lon = _ens_mean_2d(init, step, 'z', 250, DIR_AIFS_ENS_HGT250, **_MKW)
        except StepNotAvailable:
            logger.warning('  AIFS-ENS Z250 step {:03d}h ainda nao publicado (404) — pulando', step)
            continue
        ds = xr.Dataset({'hgt': (('lat', 'lon'), z / _G)}, coords={'lat': lat, 'lon': lon})
        ds['hgt'].attrs['units'] = 'm'
        parts.append(ds.expand_dims(time=[np.datetime64(vt)]))
    if not parts:
        logger.warning('AIFS-ENS Z250 valido {} sem passos publicados — dia ignorado.', day)
        return None
    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    ds_day.to_netcdf(nc_path, engine='netcdf4')
    logger.info('AIFS-ENS Z250 valido {} salvo: {}', day, nc_path.name)
    return nc_path


def ensure_aifs_ens_hgt250_fcst_for_period(
    init: datetime, lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de Z250 (m) da MEDIA do AIFS-ENS para [init, init+lead_hours]."""
    files: List[Path] = []
    end = init + timedelta(hours=lead_hours)
    day = init.date()
    while day <= end.date():
        steps = _steps_for_day(init, day, hours, lead_hours)
        if steps:
            nc = _download_day_hgt250(init, day, steps, force_redownload)
            if nc is not None:
                files.append(nc)
        day += timedelta(days=1)
    logger.info('AIFS-ENS Z250: {} arquivos | init {:%Y-%m-%d %H}Z + {}h', len(files), init, lead_hours)
    return files


def _download_day_uv250(init: datetime, day: date, steps: List[Tuple[int, datetime]], force: bool) -> Path:
    fname = f'aifs_ens_uv250_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
    nc_path = DIR_AIFS_ENS_UV250 / fname
    if nc_path.exists() and not force:
        logger.info('AIFS-ENS u/v 250 valido {} (init {}Z) ja existe — pulando.', day, init.hour)
        return nc_path
    DIR_AIFS_ENS_UV250.mkdir(parents=True, exist_ok=True)
    parts = []
    for step, vt in steps:
        try:
            u, lat, lon = _ens_mean_2d(init, step, 'u', 250, DIR_AIFS_ENS_UV250, **_MKW)
            v, _, _ = _ens_mean_2d(init, step, 'v', 250, DIR_AIFS_ENS_UV250, **_MKW)
        except StepNotAvailable:
            logger.warning('  AIFS-ENS u/v 250 step {:03d}h ainda nao publicado (404) — pulando', step)
            continue
        ds = xr.Dataset({'u': (('lat', 'lon'), u), 'v': (('lat', 'lon'), v)},
                        coords={'lat': lat, 'lon': lon}).expand_dims(time=[np.datetime64(vt)])
        parts.append(ds)
    if not parts:
        logger.warning('AIFS-ENS u/v 250 valido {} sem passos publicados — dia ignorado.', day)
        return None
    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    ds_day.to_netcdf(nc_path, engine='netcdf4')
    logger.info('AIFS-ENS u/v 250 valido {} salvo: {}', day, nc_path.name)
    return nc_path


def ensure_aifs_ens_uv250_fcst_for_period(
    init: datetime, lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de u/v 250 hPa (m/s) da MEDIA do AIFS-ENS para [init, init+lead_hours]."""
    files: List[Path] = []
    end = init + timedelta(hours=lead_hours)
    day = init.date()
    while day <= end.date():
        steps = _steps_for_day(init, day, hours, lead_hours)
        if steps:
            nc = _download_day_uv250(init, day, steps, force_redownload)
            if nc is not None:
                files.append(nc)
        day += timedelta(days=1)
    logger.info('AIFS-ENS u/v 250: {} arquivos | init {:%Y-%m-%d %H}Z + {}h', len(files), init, lead_hours)
    return files


def _download_day_uv850(init: datetime, day: date, steps: List[Tuple[int, datetime]], force: bool) -> Path:
    fname = f'aifs_ens_uv850_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
    nc_path = DIR_AIFS_ENS_UV850 / fname
    if nc_path.exists() and not force:
        logger.info('AIFS-ENS u/v 850 valido {} (init {}Z) ja existe — pulando.', day, init.hour)
        return nc_path
    DIR_AIFS_ENS_UV850.mkdir(parents=True, exist_ok=True)
    parts = []
    for step, vt in steps:
        try:
            u, lat, lon = _ens_mean_2d(init, step, 'u', 850, DIR_AIFS_ENS_UV850, **_MKW)
            v, _, _ = _ens_mean_2d(init, step, 'v', 850, DIR_AIFS_ENS_UV850, **_MKW)
        except StepNotAvailable:
            logger.warning('  AIFS-ENS u/v 850 step {:03d}h ainda nao publicado (404) — pulando', step)
            continue
        ds = xr.Dataset({'u': (('lat', 'lon'), u), 'v': (('lat', 'lon'), v)},
                        coords={'lat': lat, 'lon': lon}).expand_dims(time=[np.datetime64(vt)])
        parts.append(ds)
    if not parts:
        logger.warning('AIFS-ENS u/v 850 valido {} sem passos publicados — dia ignorado.', day)
        return None
    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    ds_day.to_netcdf(nc_path, engine='netcdf4')
    logger.info('AIFS-ENS u/v 850 valido {} salvo: {}', day, nc_path.name)
    return nc_path


def ensure_aifs_ens_uv850_fcst_for_period(
    init: datetime, lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de u/v 850 hPa (m/s) da MEDIA do AIFS-ENS para [init, init+lead_hours]."""
    files: List[Path] = []
    end = init + timedelta(hours=lead_hours)
    day = init.date()
    while day <= end.date():
        steps = _steps_for_day(init, day, hours, lead_hours)
        if steps:
            nc = _download_day_uv850(init, day, steps, force_redownload)
            if nc is not None:
                files.append(nc)
        day += timedelta(days=1)
    logger.info('AIFS-ENS u/v 850: {} arquivos | init {:%Y-%m-%d %H}Z + {}h', len(files), init, lead_hours)
    return files


def _download_day_t2m(init: datetime, day: date, steps: List[Tuple[int, datetime]], force: bool) -> Path:
    fname = f'aifs_ens_t2m_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
    nc_path = DIR_AIFS_ENS_T2M / fname
    if nc_path.exists() and not force:
        logger.info('AIFS-ENS T2m valido {} (init {}Z) ja existe — pulando.', day, init.hour)
        return nc_path
    DIR_AIFS_ENS_T2M.mkdir(parents=True, exist_ok=True)
    parts = []
    for step, vt in steps:
        try:
            t, lat, lon = _ens_mean_2d(init, step, '2t', None, DIR_AIFS_ENS_T2M, **_MKW)
        except StepNotAvailable:
            logger.warning('  AIFS-ENS T2m step {:03d}h ainda nao publicado (404) — pulando', step)
            continue
        ds = xr.Dataset({'t2m': (('lat', 'lon'), t)}, coords={'lat': lat, 'lon': lon})
        ds['t2m'].attrs['units'] = 'K'
        parts.append(ds.expand_dims(time=[np.datetime64(vt)]))
    if not parts:
        logger.warning('AIFS-ENS T2m valido {} sem passos publicados — dia ignorado.', day)
        return None
    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    ds_day.to_netcdf(nc_path, engine='netcdf4')
    logger.info('AIFS-ENS T2m valido {} salvo: {}', day, nc_path.name)
    return nc_path


def ensure_aifs_ens_t2m_fcst_for_period(
    init: datetime, lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de T2m (K) da MEDIA do AIFS-ENS para [init, init+lead_hours]."""
    files: List[Path] = []
    end = init + timedelta(hours=lead_hours)
    day = init.date()
    while day <= end.date():
        steps = _steps_for_day(init, day, hours, lead_hours)
        if steps:
            nc = _download_day_t2m(init, day, steps, force_redownload)
            if nc is not None:
                files.append(nc)
        day += timedelta(days=1)
    logger.info('AIFS-ENS T2m: {} arquivos | init {:%Y-%m-%d %H}Z + {}h', len(files), init, lead_hours)
    return files
