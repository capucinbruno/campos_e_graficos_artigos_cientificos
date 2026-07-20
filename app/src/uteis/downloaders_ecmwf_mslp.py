# app/src/uteis/downloaders_ecmwf_mslp.py
# -*- coding: utf-8 -*-
"""
Downloader ECMWF HRES (previsao) de PNMM (pressao ao nivel medio do mar) via ECMWF Open Data.

Variavel `msl` (mean sea level pressure), instantanea, em Pa. Espelha o `downloaders_gefs_mslp`:
um NetCDF por dia com a MSLP nas horas sinoticas (00/06/12/18Z), variavel `msl` (Pa) -- o
`daily_mslp_on_grid` do motor filtra as horas sinoticas, faz a MEDIA DIARIA e converte p/ hPa.

Usado pelas isolinhas de PNMM (ficha com `isolinha_mslp`, ex.: rajada_abs do s47). Forecast-only.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Sequence

import numpy as np
import xarray as xr

from app.common.forecast_download import StepNotAvailable, save_netcdf
from app.shared.logger import get_logger
from app.src.uteis.downloaders_ecmwf_fcst200 import (
    ECMWF_MAX_FHR,
    DIR_DADOS_BASE,
    ECMWF_MODEL,
    ECMWF_STREAM,
    ECMWF_TYPE,
    ecmwf_grib_url,
    ecmwf_native_steps,
    fetch_index,
    match_record,
    open_grib_bytes,
    range_bytes,
)

logger = get_logger(__name__)

DIR_ECMWF_MSLP = DIR_DADOS_BASE / 'ECMWF_MSLP'

DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)


def _fetch_msl(init: datetime, step: int, tmp_path: Path) -> xr.DataArray | None:
    """`msl` (Pa, instantaneo) no passo `step`, em (lat, lon). None se indisponivel."""
    try:
        recs = fetch_index(init, step, stream=ECMWF_STREAM, ftype=ECMWF_TYPE, model=ECMWF_MODEL)
        raw = range_bytes(ecmwf_grib_url(init, step, stream=ECMWF_STREAM, ftype=ECMWF_TYPE,
                                         model=ECMWF_MODEL), match_record(recs, 'msl'))
    except StepNotAvailable:
        logger.warning('  ECMWF msl step {:03d}h ainda nao publicado (404) — passo ausente', step)
        return None
    ds = open_grib_bytes(raw, tmp_path)
    da = ds['msl'] if 'msl' in ds.data_vars else ds[list(ds.data_vars)[0]]
    ren = {}
    for name in list(da.dims) + list(da.coords):
        low = name.lower()
        if low == 'latitude' and 'lat' not in da.dims:
            ren[name] = 'lat'
        elif low == 'longitude' and 'lon' not in da.dims:
            ren[name] = 'lon'
    if ren:
        da = da.rename(ren)
    for coord in ('step', 'valid_time', 'surface', 'number', 'heightAboveGround'):
        if coord in da.coords and coord not in da.dims:
            da = da.drop_vars(coord, errors='ignore')
    return da.rename('msl')  # Pa


def ensure_ecmwf_mslp_fcst_for_period(
    init: datetime, lead_hours: int, hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS,
    force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de PNMM (Pa) do ECMWF HRES p/ [init, init+lead_hours].

    Um arquivo por dia com `msl` nas horas sinoticas `hours` (default 00/06/12/18Z). O
    `daily_mslp_on_grid` do motor faz a media diaria e converte Pa->hPa."""
    DIR_ECMWF_MSLP.mkdir(parents=True, exist_ok=True)
    end = init + timedelta(hours=lead_hours)
    tmp = DIR_ECMWF_MSLP / f'ecmwf_mslp_{init.strftime("%Y%m%d%H")}_tmp.grb2'
    out: List[Path] = []

    day = init.date()
    while day <= end.date():
        # passos (hora sinotica -> step) dentro do horizonte, a partir do init
        vts = [datetime(day.year, day.month, day.day, h) for h in hours]
        steps = [(vt, int((vt - init).total_seconds() // 3600)) for vt in vts]
        steps = [(vt, s) for vt, s in steps if 0 <= s <= min(ECMWF_MAX_FHR, 10 ** 6)]
        if not steps:
            day += timedelta(days=1)
            continue

        nc_path = DIR_ECMWF_MSLP / (f'ecmwf_mslp_{init.strftime("%Y%m%d%H")}'
                                    f'_valid{day.strftime("%Y%m%d")}.nc')
        if nc_path.exists() and not force_redownload:
            logger.info('ECMWF MSLP valido {} (init {}Z) ja existe — pulando.', day, init.hour)
            out.append(nc_path)
            day += timedelta(days=1)
            continue

        parts = []
        for vt, s in steps:
            da = _fetch_msl(init, s, tmp)
            if da is not None:
                parts.append(da.expand_dims(time=[np.datetime64(vt)]))
        if not parts:
            logger.warning('ECMWF MSLP {} sem nenhum passo sinotico — dia ignorado.', day)
            day += timedelta(days=1)
            continue

        ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
        if nc_path.exists():
            nc_path.unlink()
        save_netcdf(ds_day.to_dataset(name='msl') if isinstance(ds_day, xr.DataArray) else ds_day,
                    nc_path)
        logger.info('ECMWF MSLP valido {} salvo ({} passo(s) sinoticos): {}',
                    day, len(parts), nc_path.name)
        out.append(nc_path)
        day += timedelta(days=1)

    if tmp.exists():
        tmp.unlink()
    logger.info('ECMWF MSLP: {} arquivo(s) | init {:%Y-%m-%d %H}Z + {}h', len(out), init, lead_hours)
    return out


def ensure_ecmwf_mslp_native_fcst_for_period(
    init: datetime, lead_hours: int, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de PNMM (Pa) do ECMWF HRES em CADA passo NATIVO (`ecmwf_native_steps`:
    3/3h ate 144h, 6/6h dai em diante) -- nao faz o resample diario que
    `ensure_ecmwf_mslp_fcst_for_period` faz (`daily_mslp_on_grid`); mantem cada passo como veio.
    MESMA forma de arquivo do `downloaders_ecmwf_ptype`/`ensure_ecmwf_tp_native_fcst_for_period`
    (mesmo `time` por dia) -- os tres alinham direto por `time`. Usado por fichas que animam
    passo a passo (ex.: chuva/neve por tipo), nao pelas isolinhas diarias (`rajada_abs`)."""
    DIR_ECMWF_MSLP.mkdir(parents=True, exist_ok=True)
    tmp = DIR_ECMWF_MSLP / f'ecmwf_mslp_native_{init.strftime("%Y%m%d%H")}_tmp.grb2'
    out: List[Path] = []

    por_dia: dict = {}
    for s in ecmwf_native_steps(lead_hours):
        vt = init + timedelta(hours=s)
        por_dia.setdefault(vt.date(), []).append((vt, s))

    # Ver comentario equivalente em `ensure_ecmwf_tp_native_fcst_for_period`: 2 dias vazios
    # seguidos == borda real do horizonte, para de sondar o resto (evita 429 do open data).
    dias_vazios_seguidos = 0

    for day in sorted(por_dia):
        steps = por_dia[day]
        nc_path = DIR_ECMWF_MSLP / (f'ecmwf_mslp_native_{init.strftime("%Y%m%d%H")}'
                                    f'_valid{day.strftime("%Y%m%d")}.nc')
        if nc_path.exists() and not force_redownload:
            logger.info('ECMWF MSLP nativo valido {} (init {}Z) ja existe — pulando.', day, init.hour)
            out.append(nc_path)
            dias_vazios_seguidos = 0
            continue

        parts = []
        for vt, s in steps:
            da = _fetch_msl(init, s, tmp)
            if da is not None:
                parts.append(da.expand_dims(time=[np.datetime64(vt)]))
        if not parts:
            logger.warning('ECMWF MSLP nativo {} sem nenhum passo publicado — dia ignorado.', day)
            dias_vazios_seguidos += 1
            if dias_vazios_seguidos >= 2:
                logger.info('ECMWF MSLP nativo: {} dias vazios seguidos -- borda do horizonte, parando.', dias_vazios_seguidos)
                break
            continue
        dias_vazios_seguidos = 0

        ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
        if nc_path.exists():
            nc_path.unlink()
        save_netcdf(ds_day.to_dataset(name='msl') if isinstance(ds_day, xr.DataArray) else ds_day,
                    nc_path)
        logger.info('ECMWF MSLP nativo valido {} salvo ({} passo(s)): {}',
                    day, len(parts), nc_path.name)
        out.append(nc_path)

    if tmp.exists():
        tmp.unlink()
    logger.info('ECMWF MSLP nativo: {} arquivo(s) | init {:%Y-%m-%d %H}Z + {}h', len(out), init, lead_hours)
    return out
