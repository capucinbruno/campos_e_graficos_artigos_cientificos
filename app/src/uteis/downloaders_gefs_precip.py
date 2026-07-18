# app/src/uteis/downloaders_gefs_precip.py
# -*- coding: utf-8 -*-
"""
Downloader GEFS (media do ensemble, `geavg`) de CHUVA (precipitacao acumulada) via NOMADS Grib Filter.

Espelha o `downloaders_gfs_precip`: o GEFS publica `APCP` (surface) no pgrb2a 0.5 em BUCKETS de 6 h
que resetam nos sinoticos -> a chuva do DIA (00-24 UTC) e a SOMA dos quatro buckets que terminam em
06, 12, 18 (do dia) e 00 (do dia seguinte). APCP e kg/m2 == mm (sem conversao).

Usa o membro `geavg` (media do ensemble PRONTA no NOMADS, ao contrario do ECMWF ENS, que exige
baixar os 50 membros e mediar) -> 1 download por bucket, mesmo custo do GFS.

NAO VALIDADO AO VIVO: o NOMADS respondeu 403 (throttling) a todas as sondagens do GEFS durante a
implementacao, entao o formato dos buckets aqui e o do GFS/GEFS documentado, nao medido. O
`_http_get` do projeto ja faz backoff p/ 403 -- se falhar na sua maquina, o erro sai claro.

Um NetCDF por dia UTC completo, variavel 'precip' (mm), 1 passo de tempo (o dia, rotulado 00Z).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import xarray as xr

from app.common.forecast_download import StepNotAvailable, save_netcdf
from app.shared.logger import get_logger
from app.src.uteis.downloaders_gefs_fcst200 import (
    DIR_DADOS_BASE,
    _download_grb2,
    _gefs_dir,
    _gefs_file,
    _gefs_max_fhr,
)
from app.src.uteis.downloaders_gfs_precip import _open_gfs_apcp

logger = get_logger(__name__)

DIR_GEFS_PRECIP = DIR_DADOS_BASE / 'GEFS_PRECIP'

_BUCKET_HOURS = (6, 12, 18, 24)   # fim de cada balde de 6 h que compoe o dia UTC


def _build_params(init: datetime, fhr: int) -> dict:
    return {
        'file': _gefs_file(init, fhr),   # geavg.tHHz.pgrb2a.0p50.fFFF
        'lev_surface': 'on',
        'var_APCP': 'on',
        'dir': _gefs_dir(init),
    }


def _fetch_bucket(init: datetime, vt: datetime) -> np.ndarray | None:
    """Balde de 6 h de APCP com fim no tempo valido `vt` (mm). None se indisponivel."""
    fhr = int((vt - init).total_seconds() // 3600)
    if fhr <= 0 or fhr > _gefs_max_fhr(init):
        return None
    grb = DIR_GEFS_PRECIP / f'gefs_apcp_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
    if not grb.exists():
        try:
            _download_grb2(_build_params(init, fhr), grb)
        except StepNotAvailable:
            logger.warning('  GEFS APCP f{:03d} ainda nao publicado (404) — bucket ausente', fhr)
            return None
    try:
        return _open_gfs_apcp(grb).values.astype('float32')   # mesmo leitor do GFS (APCP identico)
    except Exception as exc:
        logger.warning('GEFS APCP f{:03d} sem mensagem valida — bucket ignorado ({})', fhr, exc)
        return None
    finally:
        if grb.exists():
            grb.unlink()


def _open_grid(init: datetime, vt: datetime) -> tuple[np.ndarray, np.ndarray] | None:
    """(lat, lon) de um bucket qualquer, p/ montar o DataArray diario com coordenadas."""
    fhr = int((vt - init).total_seconds() // 3600)
    grb = DIR_GEFS_PRECIP / f'gefs_apcp_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
    if not grb.exists():
        try:
            _download_grb2(_build_params(init, fhr), grb)
        except StepNotAvailable:
            return None
    try:
        da = _open_gfs_apcp(grb)
        return da['lat'].values, da['lon'].values
    except Exception:
        return None
    finally:
        if grb.exists():
            grb.unlink()


def ensure_gefs_precip_fcst_for_period(
    init: datetime, lead_hours: int, hours=None, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs de CHUVA ACUMULADA DIARIA (mm) do GEFS p/ os dias UTC completos em [init, init+lead].

    `hours` e' ignorado (compat com a assinatura dos demais downloaders do globo): a chuva e'
    ACUMULADO DIARIO (00-24 UTC), nao snapshot sinotico.

    Cada dia = soma dos quatro buckets de 6 h; so e' salvo se os QUATRO estiverem disponiveis."""
    DIR_GEFS_PRECIP.mkdir(parents=True, exist_ok=True)
    end = init + timedelta(hours=lead_hours)
    out: List[Path] = []
    lat = lon = None

    day = init.date() if init.hour == 0 else (init.date() + timedelta(days=1))
    while True:
        vts = [datetime(day.year, day.month, day.day) + timedelta(hours=h) for h in _BUCKET_HOURS]
        if vts[-1] > end:
            break

        nc_path = DIR_GEFS_PRECIP / (f'gefs_precip_{init.strftime("%Y%m%d%H")}'
                                     f'_valid{day.strftime("%Y%m%d")}.nc')
        if nc_path.exists() and not force_redownload:
            logger.info('GEFS chuva valido {} (init {}Z) ja existe — pulando.', day, init.hour)
            out.append(nc_path)
            day += timedelta(days=1)
            continue

        if lat is None:
            grid = _open_grid(init, vts[0])
            if grid is not None:
                lat, lon = grid
        buckets = [_fetch_bucket(init, vt) for vt in vts]
        if any(b is None for b in buckets) or lat is None:
            logger.warning('GEFS chuva {} incompleto (bucket ausente) — dia ignorado.', day)
            day += timedelta(days=1)
            continue

        acum = np.sum(buckets, axis=0).astype('float32')
        da = xr.DataArray(
            acum[None, :, :], dims=['time', 'lat', 'lon'],
            coords={'time': [np.datetime64(datetime(day.year, day.month, day.day))],
                    'lat': lat, 'lon': lon}, name='precip')
        da.attrs['units'] = 'mm'
        da.attrs['long_name'] = 'chuva acumulada diaria (00-24 UTC)'
        if nc_path.exists():
            nc_path.unlink()
        save_netcdf(da.to_dataset(name='precip'), nc_path)
        logger.info('GEFS chuva valido {} salvo ({:.1f} mm max): {}',
                    day, float(np.nanmax(acum)), nc_path.name)
        out.append(nc_path)
        day += timedelta(days=1)

    logger.info('GEFS chuva: {} dia(s) | init {:%Y-%m-%d %H}Z + {}h', len(out), init, lead_hours)
    return out
