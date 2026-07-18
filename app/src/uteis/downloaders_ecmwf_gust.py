# app/src/uteis/downloaders_ecmwf_gust.py
# -*- coding: utf-8 -*-
"""
Downloader ECMWF HRES (previsao) de RAJADA DE VENTO a 10 m via ECMWF Open Data.

Variavel `10fg` (10 metre wind gust) -- confirmado na fonte (2026-07-17, init 00Z): e a UNICA
variavel de rajada do HRES (nao ha rajada em outro nivel nem convectiva separada). `stepType=max`,
unidade m/s: cada passo traz o MAXIMO da rajada no intervalo desde o passo anterior, NAO um
acumulado. Logo a rajada do DIA (00-24 UTC) e o MAXIMO ELEMENTO-A-ELEMENTO dos quatro intervalos de
6 h (06/12/18 do dia + 00 do dia seguinte) -- diferente da chuva, que SOMA os intervalos.

Saida: um NetCDF por dia UTC completo, variavel 'gust' em KM/H (m/s x 3.6), rotulado 00Z, 1 passo de
tempo. km/h porque e como rajada e comunicada no Brasil; a conversao e feita AQUI (o motor plota o
valor como vem).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import List

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
    fetch_index,
    match_record,
    open_grib_bytes,
    range_bytes,
)

logger = get_logger(__name__)

DIR_ECMWF_GUST = DIR_DADOS_BASE / 'ECMWF_GUST'

# Intervalos de 6 h (fim, hora UTC) cujo MAXIMO e a rajada do dia -- 06/12/18 do dia + 00 do seguinte.
_BUCKET_HOURS = (6, 12, 18, 24)

MS_PARA_KMH = 3.6


def _fetch_10fg(init: datetime, step: int, tmp_path: Path) -> np.ndarray | None:
    """`10fg` (m/s, max no intervalo) no passo `step`, em (lat, lon). None se indisponivel."""
    try:
        recs = fetch_index(init, step, stream=ECMWF_STREAM, ftype=ECMWF_TYPE, model=ECMWF_MODEL)
        raw = range_bytes(ecmwf_grib_url(init, step, stream=ECMWF_STREAM, ftype=ECMWF_TYPE,
                                         model=ECMWF_MODEL), match_record(recs, '10fg'))
    except StepNotAvailable:
        logger.warning('  ECMWF 10fg step {:03d}h ainda nao publicado (404) — intervalo ausente', step)
        return None
    ds = open_grib_bytes(raw, tmp_path)
    da = ds['10fg'] if '10fg' in ds.data_vars else ds[list(ds.data_vars)[0]]
    ren = {}
    for name in list(da.dims) + list(da.coords):
        low = name.lower()
        if low == 'latitude' and 'lat' not in da.dims:
            ren[name] = 'lat'
        elif low == 'longitude' and 'lon' not in da.dims:
            ren[name] = 'lon'
    if ren:
        da = da.rename(ren)
    return da.values.astype('float32')  # m/s


def _grid(init: datetime, step: int, tmp_path: Path):
    """(lat, lon) de um passo qualquer, p/ montar o DataArray diario."""
    recs = fetch_index(init, step, stream=ECMWF_STREAM, ftype=ECMWF_TYPE, model=ECMWF_MODEL)
    raw = range_bytes(ecmwf_grib_url(init, step, stream=ECMWF_STREAM, ftype=ECMWF_TYPE,
                                     model=ECMWF_MODEL), match_record(recs, '10fg'))
    ds = open_grib_bytes(raw, tmp_path)
    da = ds['10fg'] if '10fg' in ds.data_vars else ds[list(ds.data_vars)[0]]
    latn = next((n for n in list(da.dims) + list(da.coords) if n.lower() == 'latitude'), None)
    lonn = next((n for n in list(da.dims) + list(da.coords) if n.lower() == 'longitude'), None)
    return (da[latn].values if latn else da['lat'].values,
            da[lonn].values if lonn else da['lon'].values)


def ensure_ecmwf_gust_fcst_for_period(
    init: datetime, lead_hours: int, hours=None, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs de RAJADA MAXIMA DIARIA (km/h) do ECMWF HRES p/ os dias UTC completos em [init, init+lead].

    `hours` e' ignorado (compat com a assinatura dos demais downloaders do globo): a rajada e' o
    MAXIMO DIARIO (00-24 UTC), nao um snapshot sinotico.

    rajada[D] = max(10fg nos 4 intervalos de 6 h do dia) * 3.6. Um dia so e' salvo se os QUATRO
    intervalos estao disponiveis (dia completo)."""
    DIR_ECMWF_GUST.mkdir(parents=True, exist_ok=True)
    end = init + timedelta(hours=lead_hours)
    tmp = DIR_ECMWF_GUST / f'ecmwf_gust_{init.strftime("%Y%m%d%H")}_tmp.grb2'
    out: List[Path] = []
    lat = lon = None

    first_day = init.date() if init.hour == 0 else (init.date() + timedelta(days=1))
    day = first_day
    while True:
        vts = [datetime(day.year, day.month, day.day) + timedelta(hours=h) for h in _BUCKET_HOURS]
        steps = [int((vt - init).total_seconds() // 3600) for vt in vts]
        if steps[-1] > min(ECMWF_MAX_FHR, 10 ** 6):
            break  # ultimo intervalo do dia alem do horizonte -> para

        nc_path = DIR_ECMWF_GUST / (f'ecmwf_gust_{init.strftime("%Y%m%d%H")}'
                                    f'_valid{day.strftime("%Y%m%d")}.nc')
        if nc_path.exists() and not force_redownload:
            logger.info('ECMWF rajada valido {} (init {}Z) ja existe — pulando.', day, init.hour)
            out.append(nc_path)
            day += timedelta(days=1)
            continue

        if lat is None:
            try:
                lat, lon = _grid(init, steps[0], tmp)
            except Exception:
                pass
        campos = [_fetch_10fg(init, s, tmp) for s in steps]
        if any(c is None for c in campos) or lat is None:
            logger.warning('ECMWF rajada {} incompleto (intervalo ausente) — dia ignorado.', day)
            day += timedelta(days=1)
            continue

        # MAXIMO elemento-a-elemento dos 4 intervalos (nao soma!) -> m/s, depois km/h.
        gmax = np.maximum.reduce(campos).astype('float32') * MS_PARA_KMH
        da = xr.DataArray(
            gmax[None, :, :], dims=['time', 'lat', 'lon'],
            coords={'time': [np.datetime64(datetime(day.year, day.month, day.day))],
                    'lat': lat, 'lon': lon}, name='gust')
        da.attrs['units'] = 'km/h'
        da.attrs['long_name'] = 'rajada maxima diaria a 10 m (00-24 UTC)'
        if nc_path.exists():
            nc_path.unlink()
        save_netcdf(da.to_dataset(name='gust'), nc_path)
        logger.info('ECMWF rajada valido {} salvo ({:.0f} km/h max): {}',
                    day, float(np.nanmax(gmax)), nc_path.name)
        out.append(nc_path)
        day += timedelta(days=1)

    if tmp.exists():
        tmp.unlink()
    logger.info('ECMWF rajada: {} dia(s) | init {:%Y-%m-%d %H}Z + {}h', len(out), init, lead_hours)
    return out
