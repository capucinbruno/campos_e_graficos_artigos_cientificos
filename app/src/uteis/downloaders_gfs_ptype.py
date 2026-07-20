# app/src/uteis/downloaders_gfs_ptype.py
# -*- coding: utf-8 -*-
"""
Downloader GFS (previsao) de TIPO de precipitacao via NOMADS Grib Filter.

O GFS nao tem um codigo categorico unico como o `ptype` do ECMWF (ver `downloaders_ecmwf_ptype.py`)
-- publica 4 flags BINARIAS independentes na superficie, uma por tipo:
    CRAIN  = chuva
    CFRZR  = chuva congelante (freezing rain)
    CICEP  = granizo/ice pellets
    CSNOW  = neve
Cada uma e' instantanea, amostrada nos MESMOS horarios de FIM dos buckets de 6h do APCP (06/12/18/00Z,
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

# Nome da flag no GRIB (`var_<NOME>=on` no grib filter) -> nome curto usado nos arquivos/variavel.
_FLAGS = {
    'crain': 'CRAIN',
    'cfrzr': 'CFRZR',
    'cicep': 'CICEP',
    'csnow': 'CSNOW',
}


def _build_params(init: datetime, fhr: int, flag: str) -> dict:
    return {
        'file': f'gfs.t{init.hour:02d}z.pgrb2.0p25.f{fhr:03d}',
        'lev_surface': 'on',
        f'var_{_FLAGS[flag]}': 'on',
        'dir': f'/gfs.{init.strftime("%Y%m%d")}/{init.hour:02d}/atmos',
    }


def _open_gfs_flag(path: Path, flag: str) -> xr.DataArray:
    """Abre o GRIB2 de UMA flag categorica (CRAIN/CFRZR/CICEP/CSNOW) e devolve o DataArray (lat, lon) 0/1.

    O GFS publica cada flag com 2 `stepType` no mesmo arquivo (`instant` = valor no instante exato do
    passo; `avg` = fracao do intervalo com aquele tipo) -- sem filtrar por `stepType`, o cfgrib recusa
    por chave ambigua. Pega `instant` (equivalente ao `ptype` instantaneo do ECMWF); cai pra `avg` se o
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
        raise RuntimeError(f'cfgrib nao conseguiu abrir o {_FLAGS[flag]} do GFS: {last}')
    ren = {}
    for name in list(ds.dims) + list(ds.coords):
        low = name.lower()
        if low == 'latitude' and 'lat' not in ds.dims:
            ren[name] = 'lat'
        elif low == 'longitude' and 'lon' not in ds.dims:
            ren[name] = 'lon'
    if ren:
        ds = ds.rename(ren)
    var = next((v for v in ds.data_vars if v.lower() in (flag, 'unknown')),
               list(ds.data_vars)[0])
    da = ds[var]
    for coord in ('time', 'step', 'valid_time', 'surface', 'heightAboveGround'):
        if coord in da.coords and coord not in da.dims:
            da = da.drop_vars(coord, errors='ignore')
    return da.astype('int8').rename(flag)  # 0/1


def _fetch_flag(init: datetime, fhr: int, tmp_grb: Path, flag: str) -> np.ndarray | None:
    """Flag categorica (0/1, instantanea) no passo `fhr`, em (lat, lon). None se indisponivel."""
    grb = tmp_grb.with_name(f'gfs_{flag}_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2')
    if not grb.exists():
        try:
            _download_grb2(_build_params(init, fhr, flag), grb)
        except StepNotAvailable:
            logger.warning('  GFS {} f{:03d} ainda nao publicado (404) — passo ausente',
                           _FLAGS[flag], fhr)
            return None
    try:
        return _open_gfs_flag(grb, flag).values.astype('int8')
    except Exception as exc:
        logger.warning('GFS {} f{:03d} sem mensagem valida — passo ignorado ({})',
                       _FLAGS[flag], fhr, exc)
        return None
    finally:
        if grb.exists():
            grb.unlink()


def _open_grid(init: datetime, fhr: int, tmp_grb: Path, flag: str) -> tuple[np.ndarray, np.ndarray] | None:
    grb = tmp_grb.with_name(f'gfs_{flag}_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2')
    if not grb.exists():
        try:
            _download_grb2(_build_params(init, fhr, flag), grb)
        except StepNotAvailable:
            return None
    try:
        da = _open_gfs_flag(grb, flag)
        return da['lat'].values, da['lon'].values
    except Exception:
        return None
    finally:
        if grb.exists():
            grb.unlink()


def _ensure_gfs_flag_native_fcst_for_period(
    flag: str, init: datetime, lead_hours: int, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de UMA flag categorica (0/1) do GFS em CADA passo de 6h NATIVO -- MESMA
    cadencia/agrupamento por dia do `ensure_gfs_precip_native_fcst_for_period` (identicos horarios
    de fim de bucket), pra a classificacao de tipo cruzar os dois eixos `time` sem align aproximado."""
    DIR_GFS_PTYPE.mkdir(parents=True, exist_ok=True)
    tmp = DIR_GFS_PTYPE / f'gfs_{flag}_{init.strftime("%Y%m%d%H")}_tmp.grb2'
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
        nc_path = DIR_GFS_PTYPE / f'gfs_{flag}_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
        if nc_path.exists() and not force_redownload:
            logger.info('GFS {} valido {} (init {}Z) ja existe — pulando.', _FLAGS[flag], day, init.hour)
            out.append(nc_path)
            dias_vazios_seguidos = 0
            continue

        if lat is None:
            grid = _open_grid(init, steps[0][1], tmp, flag)
            if grid is not None:
                lat, lon = grid
        parts = []
        for vt, fhr in steps:
            v = _fetch_flag(init, fhr, tmp, flag)
            if v is not None and lat is not None:
                parts.append(xr.DataArray(
                    v[None, :, :], dims=['time', 'lat', 'lon'],
                    coords={'time': [np.datetime64(vt)], 'lat': lat, 'lon': lon}, name=flag))
        if not parts:
            logger.warning('GFS {} {} sem nenhum passo publicado — dia ignorado.', _FLAGS[flag], day)
            dias_vazios_seguidos += 1
            if dias_vazios_seguidos >= 2:
                logger.info('GFS {}: {} dias vazios seguidos -- borda do horizonte, parando.',
                           _FLAGS[flag], dias_vazios_seguidos)
                break
            continue
        dias_vazios_seguidos = 0

        ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
        if nc_path.exists():
            nc_path.unlink()
        save_netcdf(ds_day.to_dataset(name=flag), nc_path)
        logger.info('GFS {} valido {} salvo ({} passo(s)): {}', _FLAGS[flag], day, len(parts), nc_path.name)
        out.append(nc_path)

    if tmp.exists():
        tmp.unlink()
    logger.info('GFS {}: {} arquivo(s) | init {:%Y-%m-%d %H}Z + {}h', _FLAGS[flag], len(out), init, lead_hours)
    return out


def ensure_gfs_csnow_native_fcst_for_period(
    init: datetime, lead_hours: int, force_redownload: bool = False,
) -> List[Path]:
    """CSNOW (neve) — ver `_ensure_gfs_flag_native_fcst_for_period`."""
    return _ensure_gfs_flag_native_fcst_for_period('csnow', init, lead_hours, force_redownload)


def ensure_gfs_crain_native_fcst_for_period(
    init: datetime, lead_hours: int, force_redownload: bool = False,
) -> List[Path]:
    """CRAIN (chuva) — ver `_ensure_gfs_flag_native_fcst_for_period`."""
    return _ensure_gfs_flag_native_fcst_for_period('crain', init, lead_hours, force_redownload)


def ensure_gfs_cfrzr_native_fcst_for_period(
    init: datetime, lead_hours: int, force_redownload: bool = False,
) -> List[Path]:
    """CFRZR (chuva congelante) — ver `_ensure_gfs_flag_native_fcst_for_period`."""
    return _ensure_gfs_flag_native_fcst_for_period('cfrzr', init, lead_hours, force_redownload)


def ensure_gfs_cicep_native_fcst_for_period(
    init: datetime, lead_hours: int, force_redownload: bool = False,
) -> List[Path]:
    """CICEP (granizo/ice pellets) — ver `_ensure_gfs_flag_native_fcst_for_period`."""
    return _ensure_gfs_flag_native_fcst_for_period('cicep', init, lead_hours, force_redownload)
