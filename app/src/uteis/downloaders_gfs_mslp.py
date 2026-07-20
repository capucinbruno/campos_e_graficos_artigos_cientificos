# app/src/uteis/downloaders_gfs_mslp.py
# -*- coding: utf-8 -*-
"""
Downloader GFS (previsao) de PNMM (PRMSL, pressao ao nivel medio do mar) via NOMADS Grib Filter.

Variavel `PRMSL` (mean sea level pressure, Pa), instantanea. Espelha o `downloaders_ecmwf_mslp`:
`ensure_gfs_mslp_fcst_for_period` (diario, horas sinoticas 00/06/12/18Z, p/ isolinha MEDIA do dia)
e `ensure_gfs_mslp_native_fcst_for_period` (passo a passo, p/ animar junto da chuva quando
ACUM_HORARIO=true). O GFS 0.25 nao tem cadenciamento "nativo" mais fino que os buckets de 6h do
APCP (ver `downloaders_gfs_precip.py`) -- por isso o passo NATIVO aqui e' 6 em 6h (0,6,12,...),
os MESMOS horarios dos buckets de chuva, pra as duas series animarem no mesmo eixo `time`.

Forecast-only (NOMADS nao publica reanalise).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Sequence

import numpy as np
import xarray as xr

from app.common.forecast_download import StepNotAvailable, save_netcdf
from app.shared.logger import get_logger
from app.src.uteis.downloaders_gfs_fcst200 import DIR_DADOS_BASE, GFS_MAX_FHR, _download_grb2

logger = get_logger(__name__)

DIR_GFS_MSLP = DIR_DADOS_BASE / 'GFS_MSLP'

DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)
_NATIVE_STEP_HOURS = 6   # mesma cadencia dos buckets de APCP (ver downloaders_gfs_precip.py)


def _build_params(init: datetime, fhr: int) -> dict:
    return {
        'file': f'gfs.t{init.hour:02d}z.pgrb2.0p25.f{fhr:03d}',
        'lev_mean_sea_level': 'on',
        'var_PRMSL': 'on',
        'dir': f'/gfs.{init.strftime("%Y%m%d")}/{init.hour:02d}/atmos',
    }


def _open_gfs_prmsl(path: Path) -> xr.DataArray:
    """Abre o GRIB2 do PRMSL e devolve o DataArray (lat, lon) em Pa."""
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
    var = next((v for v in ds.data_vars if v.lower() in ('prmsl', 'msl', 'mslet')),
               list(ds.data_vars)[0])
    da = ds[var]
    for coord in ('time', 'step', 'valid_time', 'meanSea', 'surface', 'heightAboveGround'):
        if coord in da.coords and coord not in da.dims:
            da = da.drop_vars(coord, errors='ignore')
    return da.rename('msl')  # Pa


def _fetch_prmsl(init: datetime, fhr: int, tmp_grb: Path) -> np.ndarray | None:
    """PRMSL (Pa, instantaneo) no passo `fhr`, em (lat, lon). None se indisponivel."""
    grb = tmp_grb.with_name(f'gfs_prmsl_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2')
    if not grb.exists():
        try:
            _download_grb2(_build_params(init, fhr), grb)
        except StepNotAvailable:
            logger.warning('  GFS PRMSL f{:03d} ainda nao publicado (404) — passo ausente', fhr)
            return None
    try:
        return _open_gfs_prmsl(grb).values.astype('float32')
    except Exception as exc:
        logger.warning('GFS PRMSL f{:03d} sem mensagem valida — passo ignorado ({})', fhr, exc)
        return None
    finally:
        if grb.exists():
            grb.unlink()


def _open_grid(init: datetime, fhr: int, tmp_grb: Path) -> tuple[np.ndarray, np.ndarray] | None:
    grb = tmp_grb.with_name(f'gfs_prmsl_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2')
    if not grb.exists():
        try:
            _download_grb2(_build_params(init, fhr), grb)
        except StepNotAvailable:
            return None
    try:
        da = _open_gfs_prmsl(grb)
        return da['lat'].values, da['lon'].values
    except Exception:
        return None
    finally:
        if grb.exists():
            grb.unlink()


def ensure_gfs_mslp_fcst_for_period(
    init: datetime, lead_hours: int, hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS,
    force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de PNMM (Pa) do GFS p/ [init, init+lead_hours].

    Um arquivo por dia com `msl` nas horas sinoticas `hours` (default 00/06/12/18Z) -- MESMA forma
    do `ensure_ecmwf_mslp_fcst_for_period`. O `daily_mslp_on_grid` do motor faz a media diaria e
    converte Pa->hPa."""
    DIR_GFS_MSLP.mkdir(parents=True, exist_ok=True)
    end = init + timedelta(hours=lead_hours)
    tmp = DIR_GFS_MSLP / f'gfs_mslp_{init.strftime("%Y%m%d%H")}_tmp.grb2'
    out: List[Path] = []
    lat = lon = None

    day = init.date()
    while day <= end.date():
        vts = [datetime(day.year, day.month, day.day, h) for h in hours]
        steps = [(vt, int((vt - init).total_seconds() // 3600)) for vt in vts]
        steps = [(vt, s) for vt, s in steps if 0 <= s <= min(GFS_MAX_FHR, lead_hours)]
        if not steps:
            day += timedelta(days=1)
            continue

        nc_path = DIR_GFS_MSLP / f'gfs_mslp_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
        if nc_path.exists() and not force_redownload:
            logger.info('GFS MSLP valido {} (init {}Z) ja existe — pulando.', day, init.hour)
            out.append(nc_path)
            day += timedelta(days=1)
            continue

        if lat is None:
            grid = _open_grid(init, steps[0][1], tmp)
            if grid is not None:
                lat, lon = grid
        parts = []
        for vt, s in steps:
            v = _fetch_prmsl(init, s, tmp)
            if v is not None and lat is not None:
                parts.append(xr.DataArray(
                    v[None, :, :], dims=['time', 'lat', 'lon'],
                    coords={'time': [np.datetime64(vt)], 'lat': lat, 'lon': lon}, name='msl'))
        if not parts:
            logger.warning('GFS MSLP {} sem nenhum passo sinotico — dia ignorado.', day)
            day += timedelta(days=1)
            continue

        ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
        if nc_path.exists():
            nc_path.unlink()
        save_netcdf(ds_day.to_dataset(name='msl'), nc_path)
        logger.info('GFS MSLP valido {} salvo ({} passo(s) sinoticos): {}', day, len(parts), nc_path.name)
        out.append(nc_path)
        day += timedelta(days=1)

    if tmp.exists():
        tmp.unlink()
    logger.info('GFS MSLP: {} arquivo(s) | init {:%Y-%m-%d %H}Z + {}h', len(out), init, lead_hours)
    return out


def ensure_gfs_mslp_native_fcst_for_period(
    init: datetime, lead_hours: int, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de PNMM (Pa) do GFS em CADA passo de 6 em 6h (`_NATIVE_STEP_HOURS`) -- MESMA
    cadencia dos buckets de chuva (`ensure_gfs_precip_fcst_for_period`), pra isolinha de PNMM e
    chuva animarem no mesmo eixo `time` quando ACUM_HORARIO=true. Nao faz o resample diario que
    `ensure_gfs_mslp_fcst_for_period` faz -- mantem cada passo como veio. MESMA forma de arquivo do
    `ensure_ecmwf_mslp_native_fcst_for_period` (1 NetCDF por dia com os passos daquele dia)."""
    DIR_GFS_MSLP.mkdir(parents=True, exist_ok=True)
    end = init + timedelta(hours=lead_hours)
    tmp = DIR_GFS_MSLP / f'gfs_mslp_native_{init.strftime("%Y%m%d%H")}_tmp.grb2'
    out: List[Path] = []
    lat = lon = None

    por_dia: dict[date, list] = {}
    fhr = 0
    limite = min(lead_hours, GFS_MAX_FHR)
    while fhr <= limite:
        vt = init + timedelta(hours=fhr)
        por_dia.setdefault(vt.date(), []).append((vt, fhr))
        fhr += _NATIVE_STEP_HOURS

    # 2 dias vazios seguidos == borda real do horizonte (mesma logica do ECMWF nativo) -- para de
    # sondar em vez de bater em todos os dias ate `end` (NOMADS tambem faz throttling em rajada).
    dias_vazios_seguidos = 0

    for day in sorted(por_dia):
        steps = por_dia[day]
        nc_path = DIR_GFS_MSLP / f'gfs_mslp_native_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
        if nc_path.exists() and not force_redownload:
            logger.info('GFS MSLP nativo valido {} (init {}Z) ja existe — pulando.', day, init.hour)
            out.append(nc_path)
            dias_vazios_seguidos = 0
            continue

        if lat is None:
            grid = _open_grid(init, steps[0][1], tmp)
            if grid is not None:
                lat, lon = grid
        parts = []
        for vt, s in steps:
            v = _fetch_prmsl(init, s, tmp)
            if v is not None and lat is not None:
                parts.append(xr.DataArray(
                    v[None, :, :], dims=['time', 'lat', 'lon'],
                    coords={'time': [np.datetime64(vt)], 'lat': lat, 'lon': lon}, name='msl'))
        if not parts:
            logger.warning('GFS MSLP nativo {} sem nenhum passo publicado — dia ignorado.', day)
            dias_vazios_seguidos += 1
            if dias_vazios_seguidos >= 2:
                logger.info('GFS MSLP nativo: {} dias vazios seguidos -- borda do horizonte, parando.',
                           dias_vazios_seguidos)
                break
            continue
        dias_vazios_seguidos = 0

        ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
        if nc_path.exists():
            nc_path.unlink()
        save_netcdf(ds_day.to_dataset(name='msl'), nc_path)
        logger.info('GFS MSLP nativo valido {} salvo ({} passo(s)): {}', day, len(parts), nc_path.name)
        out.append(nc_path)

    if tmp.exists():
        tmp.unlink()
    logger.info('GFS MSLP nativo: {} arquivo(s) | init {:%Y-%m-%d %H}Z + {}h', len(out), init, lead_hours)
    return out
