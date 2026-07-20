# app/src/uteis/downloaders_gfs_ptype.py
# -*- coding: utf-8 -*-
"""
Downloader GFS (previsao) de TIPO de precipitacao via NOMADS Grib Filter.

O GFS nao tem um codigo categorico unico como o `ptype` do ECMWF (ver `downloaders_ecmwf_ptype.py`)
-- publica 4 flags BINARIAS independentes na superficie (CRAIN/CFRZR/CICEP/CSNOW, 0 ou 1), uma por
tipo. Aqui so baixa `CSNOW` (categorical snow), a unica usada hoje pela camada de NEVE do `NEVE`
(settings) -- ver `_ptype_neve_forecast_series` em `globo_3d_anim.py`.

CSNOW e' instantaneo, amostrado nos MESMOS horarios de FIM dos buckets de 6h do APCP (06/12/18/00Z,
ver `ensure_gfs_precip_native_fcst_for_period` em `downloaders_gfs_precip.py`) -- classifica cada
bucket de chuva pelo tipo vigente no fim daquele intervalo, mesmo principio do `ptype` do ECMWF (que
tambem classifica o passo pelo FIM do intervalo).

Forecast-only (NOMADS nao publica reanalise).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import xarray as xr

from app.common.forecast_download import StepNotAvailable, save_netcdf
from app.shared.logger import get_logger
from app.src.uteis.downloaders_gfs_fcst200 import DIR_DADOS_BASE, GFS_MAX_FHR, _download_grb2

logger = get_logger(__name__)

DIR_GFS_PTYPE = DIR_DADOS_BASE / 'GFS_PTYPE'

_NATIVE_STEP_HOURS = 6   # mesma cadencia dos buckets de APCP (ver downloaders_gfs_precip.py)


def _build_params(init: datetime, fhr: int) -> dict:
    return {
        'file': f'gfs.t{init.hour:02d}z.pgrb2.0p25.f{fhr:03d}',
        'lev_surface': 'on',
        'var_CSNOW': 'on',
        'dir': f'/gfs.{init.strftime("%Y%m%d")}/{init.hour:02d}/atmos',
    }


def _open_gfs_csnow(path: Path) -> xr.DataArray:
    """Abre o GRIB2 do CSNOW e devolve o DataArray (lat, lon) 0/1.

    O GFS publica CSNOW com 2 `stepType` no mesmo arquivo (`instant` = flag no instante exato do
    passo; `avg` = fracao do intervalo com neve) -- sem filtrar por `stepType`, o cfgrib recusa por
    chave ambigua. Pega `instant` (equivalente ao `ptype` instantaneo do ECMWF); cai pra `avg` se o
    passo nao tiver `instant` publicado."""
    from app.common.forecast_download import GRIB_NETCDF_LOCK
    ds = None
    last = None
    with GRIB_NETCDF_LOCK:  # ecCodes nao e thread-safe entre downloads paralelos
        for fbk in ({'typeOfLevel': 'surface', 'stepType': 'instant'},
                    {'typeOfLevel': 'surface', 'stepType': 'avg'}):
            try:
                ds = xr.open_dataset(path, engine='cfgrib', backend_kwargs={'indexpath': ''},
                                     filter_by_keys=fbk)
                if len(ds.data_vars):
                    ds = ds.load()
                    break
            except Exception as exc:
                last = exc
                ds = None
    if ds is None or not len(ds.data_vars):
        raise RuntimeError(f'cfgrib nao conseguiu abrir o CSNOW do GFS: {last}')
    ren = {}
    for name in list(ds.dims) + list(ds.coords):
        low = name.lower()
        if low == 'latitude' and 'lat' not in ds.dims:
            ren[name] = 'lat'
        elif low == 'longitude' and 'lon' not in ds.dims:
            ren[name] = 'lon'
    if ren:
        ds = ds.rename(ren)
    var = next((v for v in ds.data_vars if v.lower() in ('csnow', 'unknown')),
               list(ds.data_vars)[0])
    da = ds[var]
    for coord in ('time', 'step', 'valid_time', 'surface', 'heightAboveGround'):
        if coord in da.coords and coord not in da.dims:
            da = da.drop_vars(coord, errors='ignore')
    return da.astype('int8').rename('csnow')  # 0/1


def _fetch_csnow(init: datetime, fhr: int, tmp_grb: Path) -> np.ndarray | None:
    """CSNOW (0/1, instantaneo) no passo `fhr`, em (lat, lon). None se indisponivel."""
    grb = tmp_grb.with_name(f'gfs_csnow_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2')
    if not grb.exists():
        try:
            _download_grb2(_build_params(init, fhr), grb)
        except StepNotAvailable:
            logger.warning('  GFS CSNOW f{:03d} ainda nao publicado (404) — passo ausente', fhr)
            return None
    try:
        return _open_gfs_csnow(grb).values.astype('int8')
    except Exception as exc:
        logger.warning('GFS CSNOW f{:03d} sem mensagem valida — passo ignorado ({})', fhr, exc)
        return None
    finally:
        if grb.exists():
            grb.unlink()


def _open_grid(init: datetime, fhr: int, tmp_grb: Path) -> tuple[np.ndarray, np.ndarray] | None:
    grb = tmp_grb.with_name(f'gfs_csnow_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2')
    if not grb.exists():
        try:
            _download_grb2(_build_params(init, fhr), grb)
        except StepNotAvailable:
            return None
    try:
        da = _open_gfs_csnow(grb)
        return da['lat'].values, da['lon'].values
    except Exception:
        return None
    finally:
        if grb.exists():
            grb.unlink()


def ensure_gfs_csnow_native_fcst_for_period(
    init: datetime, lead_hours: int, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de CSNOW (0/1) do GFS em CADA passo de 6h NATIVO -- MESMA cadencia/agrupamento
    por dia do `ensure_gfs_precip_native_fcst_for_period` (identicos horarios de fim de bucket), pra
    a classificacao de neve cruzar os dois eixos `time` sem align aproximado."""
    DIR_GFS_PTYPE.mkdir(parents=True, exist_ok=True)
    tmp = DIR_GFS_PTYPE / f'gfs_csnow_{init.strftime("%Y%m%d%H")}_tmp.grb2'
    out: List[Path] = []
    lat = lon = None

    por_dia: dict[date, list] = {}
    h = _NATIVE_STEP_HOURS
    while h <= min(lead_hours, GFS_MAX_FHR):
        vt = init + timedelta(hours=h)
        por_dia.setdefault(vt.date(), []).append((vt, h))
        h += _NATIVE_STEP_HOURS

    # 2 dias vazios seguidos == borda real do horizonte (mesma logica do APCP/MSLP nativos) -- para
    # de sondar em vez de bater em todos os dias ate o fim (NOMADS tambem faz throttling em rajada).
    dias_vazios_seguidos = 0

    for day in sorted(por_dia):
        steps = por_dia[day]
        nc_path = DIR_GFS_PTYPE / f'gfs_csnow_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
        if nc_path.exists() and not force_redownload:
            logger.info('GFS CSNOW valido {} (init {}Z) ja existe — pulando.', day, init.hour)
            out.append(nc_path)
            dias_vazios_seguidos = 0
            continue

        if lat is None:
            grid = _open_grid(init, steps[0][1], tmp)
            if grid is not None:
                lat, lon = grid
        parts = []
        for vt, fhr in steps:
            v = _fetch_csnow(init, fhr, tmp)
            if v is not None and lat is not None:
                parts.append(xr.DataArray(
                    v[None, :, :], dims=['time', 'lat', 'lon'],
                    coords={'time': [np.datetime64(vt)], 'lat': lat, 'lon': lon}, name='csnow'))
        if not parts:
            logger.warning('GFS CSNOW {} sem nenhum passo publicado — dia ignorado.', day)
            dias_vazios_seguidos += 1
            if dias_vazios_seguidos >= 2:
                logger.info('GFS CSNOW: {} dias vazios seguidos -- borda do horizonte, parando.',
                           dias_vazios_seguidos)
                break
            continue
        dias_vazios_seguidos = 0

        ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
        if nc_path.exists():
            nc_path.unlink()
        save_netcdf(ds_day.to_dataset(name='csnow'), nc_path)
        logger.info('GFS CSNOW valido {} salvo ({} passo(s)): {}', day, len(parts), nc_path.name)
        out.append(nc_path)

    if tmp.exists():
        tmp.unlink()
    logger.info('GFS CSNOW: {} arquivo(s) | init {:%Y-%m-%d %H}Z + {}h', len(out), init, lead_hours)
    return out
