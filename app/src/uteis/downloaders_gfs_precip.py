# app/src/uteis/downloaders_gfs_precip.py
# -*- coding: utf-8 -*-
"""
Downloader GFS (previsao) de CHUVA (precipitacao acumulada) via NOMADS Grib Filter.

NAO confundir com PWAT: aqui e a precipitacao que CAIU (`APCP`, surface), nao a agua
precipitavel (vapor na coluna). Saida = ACUMULADO DIARIO em mm (kg/m2 == mm de chuva).

Acumulacao no GFS 0.25deg: o `APCP` vem em BUCKETS de 6 h que RESETAM nos sinoticos
(f006 = 00-06h, f012 = 06-12h, f018 = 12-18h, f024 = 18-24h, ...). Cada bucket nos
multiplos de 6 h e' o total de 6 h daquele intervalo -> a chuva do DIA (UTC 00-24) e' a
SOMA dos quatro buckets que terminam em 06, 12, 18 (do dia) e 00 (do dia seguinte).

Um NetCDF por dia UTC completo, variavel 'precip' (mm), 1 passo de tempo (o dia, rotulado 00Z).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import xarray as xr

from app.common.forecast_download import StepNotAvailable, save_netcdf
from app.shared.logger import get_logger
from app.src.uteis.downloaders_gfs_fcst200 import (
    DIR_DADOS_BASE,
    GFS_MAX_FHR,
    _download_grb2,
)

logger = get_logger(__name__)

DIR_GFS_PRECIP = DIR_DADOS_BASE / 'GFS_PRECIP'

# Buckets de 6 h (fim do intervalo, hora UTC) que somam a chuva de UM dia UTC:
#   06, 12, 18 do proprio dia + 00 do dia seguinte (bucket 18-24h).
_BUCKET_HOURS = (6, 12, 18, 24)


def _build_params(init: datetime, fhr: int) -> dict:
    return {
        'file': f'gfs.t{init.hour:02d}z.pgrb2.0p25.f{fhr:03d}',
        'lev_surface': 'on',
        'var_APCP': 'on',
        'dir': f'/gfs.{init.strftime("%Y%m%d")}/{init.hour:02d}/atmos',
    }


def _open_gfs_apcp(path: Path) -> xr.DataArray:
    """Abre o GRIB2 do APCP (bucket de 6 h) e devolve o DataArray (lat, lon) em mm."""
    from app.common.forecast_download import GRIB_NETCDF_LOCK
    ds = None
    last = None
    with GRIB_NETCDF_LOCK:  # ecCodes nao e thread-safe entre downloads paralelos
        for fbk in ({'stepType': 'accum', 'typeOfLevel': 'surface'}, {}):
            try:
                ds = xr.open_dataset(
                    path, engine='cfgrib', backend_kwargs={'indexpath': ''}, filter_by_keys=fbk)
                if len(ds.data_vars):
                    ds = ds.load()
                    break
            except Exception as exc:
                last = exc
                ds = None
    if ds is None or not len(ds.data_vars):
        raise RuntimeError(f'cfgrib nao conseguiu abrir o APCP do GFS: {last}')

    ren = {}
    for name in list(ds.dims) + list(ds.coords):
        low = name.lower()
        if low == 'latitude' and 'lat' not in ds.dims:
            ren[name] = 'lat'
        elif low == 'longitude' and 'lon' not in ds.dims:
            ren[name] = 'lon'
    if ren:
        ds = ds.rename(ren)

    # APCP costuma sair como 'tp' ou 'unknown'/'acpcp'; pega a 1a variavel de dados.
    var = next((v for v in ds.data_vars if v.lower() in ('tp', 'apcp', 'acpcp', 'unknown')),
               list(ds.data_vars)[0])
    da = ds[var]
    for coord in ('time', 'step', 'valid_time', 'surface', 'level', 'heightAboveGround'):
        if coord in da.coords and coord not in da.dims:
            da = da.drop_vars(coord, errors='ignore')
    return da  # kg/m2 == mm (o bucket ja e' a chuva das 6 h)


def _fetch_bucket(init: datetime, vt: datetime) -> np.ndarray | None:
    """Baixa o bucket de 6 h de APCP com fim no tempo valido `vt`. None se indisponivel."""
    fhr = int((vt - init).total_seconds() // 3600)
    if fhr <= 0 or fhr > min(GFS_MAX_FHR, 10**6):
        return None
    grb = DIR_GFS_PRECIP / f'gfs_apcp_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
    if not grb.exists():
        try:
            _download_grb2(_build_params(init, fhr), grb)
        except StepNotAvailable:
            logger.warning('  GFS APCP f{:03d} ainda nao publicado (404) — bucket ausente', fhr)
            return None
    try:
        arr = _open_gfs_apcp(grb).values.astype('float32')
        return arr
    except Exception as exc:
        logger.warning('GFS APCP f{:03d} sem mensagem valida — bucket ignorado ({})', fhr, exc)
        if grb.exists():
            grb.unlink()
        return None
    finally:
        if grb.exists():
            grb.unlink()


def _open_grid(init: datetime, vt: datetime) -> tuple[np.ndarray, np.ndarray] | None:
    """(lat, lon) de um bucket qualquer, p/ montar o DataArray diario com coordenadas."""
    fhr = int((vt - init).total_seconds() // 3600)
    grb = DIR_GFS_PRECIP / f'gfs_apcp_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
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


def ensure_gfs_precip_fcst_for_period(
    init: datetime, lead_hours: int, hours=None, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs de CHUVA ACUMULADA DIARIA (mm) do GFS para os dias UTC completos em [init, init+lead].

    `hours` e' ignorado (compat com a assinatura dos demais downloaders do globo, que passam as horas
    sinoticas): a chuva e' ACUMULADO DIARIO (00-24 UTC), nao snapshot sinotico -- os buckets internos
    de 6 h ja sao fixos (06/12/18/00Z).

    Cada dia UTC = soma dos quatro buckets de 6 h (06/12/18 do dia + 00 do dia seguinte). Um dia so
    e' salvo se TODOS os quatro buckets estao disponiveis (dia completo)."""
    DIR_GFS_PRECIP.mkdir(parents=True, exist_ok=True)
    end = init + timedelta(hours=lead_hours)
    out: List[Path] = []

    # Primeiro dia UTC completo: se o init nao e' 00Z, o dia do init fica parcial -> comeca no proximo.
    first_day = init.date() if init.hour == 0 else (init.date() + timedelta(days=1))
    day = first_day
    lat = lon = None
    while True:
        # buckets que terminam em 06,12,18 do dia e 00 do dia seguinte
        vts = [datetime(day.year, day.month, day.day) + timedelta(hours=h) for h in _BUCKET_HOURS]
        if vts[-1] > end:
            break  # dia incompleto (ultimo bucket alem do horizonte) -> para

        fname = f'gfs_precip_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
        nc_path = DIR_GFS_PRECIP / fname
        if nc_path.exists() and not force_redownload:
            logger.info('GFS chuva valido {} (init {}Z) ja existe — pulando.', day, init.hour)
            out.append(nc_path)
            day += timedelta(days=1)
            continue

        if lat is None:
            grid = _open_grid(init, vts[0])
            if grid is not None:
                lat, lon = grid
        buckets = [_fetch_bucket(init, vt) for vt in vts]
        if any(b is None for b in buckets) or lat is None:
            logger.warning('GFS chuva {} incompleto (bucket ausente) — dia ignorado.', day)
            day += timedelta(days=1)
            continue

        acum = np.sum(buckets, axis=0).astype('float32')  # mm no dia
        da = xr.DataArray(
            acum[None, :, :], dims=['time', 'lat', 'lon'],
            coords={'time': [np.datetime64(datetime(day.year, day.month, day.day))],
                    'lat': lat, 'lon': lon}, name='precip')
        da.attrs['units'] = 'mm'
        da.attrs['long_name'] = 'chuva acumulada diaria (00-24 UTC)'
        ds_day = da.to_dataset(name='precip')
        if nc_path.exists():
            nc_path.unlink()
        save_netcdf(ds_day, nc_path)
        logger.info('GFS chuva valido {} salvo ({:.1f} mm max): {}',
                    day, float(np.nanmax(acum)), nc_path.name)
        out.append(nc_path)
        day += timedelta(days=1)

    logger.info('GFS chuva: {} dia(s) | init {:%Y-%m-%d %H}Z + {}h', len(out), init, lead_hours)
    return out
