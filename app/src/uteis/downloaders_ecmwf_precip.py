# app/src/uteis/downloaders_ecmwf_precip.py
# -*- coding: utf-8 -*-
"""
Downloader ECMWF HRES (previsao) de CHUVA (precipitacao acumulada) via ECMWF Open Data.

NAO confundir com PWAT: aqui e a precipitacao que CAIU (`tp`, total precipitation), nao a
agua precipitavel (`tcwv`). Saida = ACUMULADO DIARIO em mm.

Acumulacao no ECMWF: `tp` e' acumulado desde o INIT, em METROS, de forma CONTINUA (nao reseta).
Logo a chuva de um dia UTC = tp(fim do dia) - tp(inicio do dia), convertido de m para mm (x1000):
    chuva[D] = ( tp(D+1 00Z) - tp(D 00Z) ) * 1000
No dia do init (00Z), tp(init) = 0, entao chuva[dia_init] = tp(init+24h) * 1000.

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
from app.src.uteis.downloaders_ecmwf_fcst200 import (
    ECMWF_MAX_FHR,
    DIR_DADOS_BASE,
    ecmwf_grib_url,
    fetch_index,
    match_record,
    open_grib_bytes,
    range_bytes,
)

logger = get_logger(__name__)

DIR_ECMWF_PRECIP = DIR_DADOS_BASE / 'ECMWF_PRECIP'


def _fetch_tp(init: datetime, step: int, tmp_path: Path) -> np.ndarray | None:
    """`tp` (metros, acumulado desde o init) no passo `step`, em (lat, lon). None se indisponivel.

    step 0 = analise: tp=0 (inicio da acumulacao) -> devolve zeros na 1a grade conhecida (tratado
    pelo chamador, que ja tem a grade)."""
    from app.src.uteis.downloaders_ecmwf_fcst200 import ECMWF_MODEL, ECMWF_STREAM, ECMWF_TYPE
    try:
        recs = fetch_index(init, step, stream=ECMWF_STREAM, ftype=ECMWF_TYPE, model=ECMWF_MODEL)
        raw = range_bytes(ecmwf_grib_url(init, step, stream=ECMWF_STREAM, ftype=ECMWF_TYPE,
                                         model=ECMWF_MODEL), match_record(recs, 'tp'))
    except StepNotAvailable:
        logger.warning('  ECMWF tp step {:03d}h ainda nao publicado (404) — passo ausente', step)
        return None
    ds = open_grib_bytes(raw, tmp_path)
    da = ds['tp'] if 'tp' in ds.data_vars else ds[list(ds.data_vars)[0]]
    ren = {}
    for name in list(da.dims) + list(da.coords):
        low = name.lower()
        if low == 'latitude' and 'lat' not in da.dims:
            ren[name] = 'lat'
        elif low == 'longitude' and 'lon' not in da.dims:
            ren[name] = 'lon'
    if ren:
        da = da.rename(ren)
    return da.values.astype('float32')  # metros


def ensure_ecmwf_precip_fcst_for_period(
    init: datetime, lead_hours: int, hours=None, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs de CHUVA ACUMULADA DIARIA (mm) do ECMWF HRES p/ os dias UTC completos em [init, init+lead].

    `hours` e' ignorado (compat com a assinatura dos demais downloaders do globo): a chuva e' ACUMULADO
    DIARIO (00-24 UTC), nao snapshot sinotico.

    chuva[D] = (tp(D+1 00Z) - tp(D 00Z)) * 1000. Um dia so e' salvo com AMBAS as fronteiras (00Z do
    dia e do dia seguinte) disponiveis."""
    DIR_ECMWF_PRECIP.mkdir(parents=True, exist_ok=True)
    end = init + timedelta(hours=lead_hours)
    tmp = DIR_ECMWF_PRECIP / f'ecmwf_precip_{init.strftime("%Y%m%d%H")}_tmp.grb2'

    # Grade + coordenadas (lidas 1x do 1o passo valido).
    lat = lon = None

    def _tp_lonlat(step: int):
        nonlocal lat, lon
        from app.src.uteis.downloaders_ecmwf_fcst200 import ECMWF_MODEL, ECMWF_STREAM, ECMWF_TYPE
        try:
            recs = fetch_index(init, step, stream=ECMWF_STREAM, ftype=ECMWF_TYPE, model=ECMWF_MODEL)
            raw = range_bytes(ecmwf_grib_url(init, step, stream=ECMWF_STREAM, ftype=ECMWF_TYPE,
                                             model=ECMWF_MODEL), match_record(recs, 'tp'))
            ds = open_grib_bytes(raw, tmp)
            da = ds['tp'] if 'tp' in ds.data_vars else ds[list(ds.data_vars)[0]]
            latn = next((n for n in list(da.dims) + list(da.coords) if n.lower() == 'latitude'), None)
            lonn = next((n for n in list(da.dims) + list(da.coords) if n.lower() == 'longitude'), None)
            lat = da[latn].values if latn else da['lat'].values
            lon = da[lonn].values if lonn else da['lon'].values
        except Exception:
            pass

    out: List[Path] = []
    # Primeiro dia UTC completo (o dia do init se 00Z; senao o proximo).
    first_day = init.date() if init.hour == 0 else (init.date() + timedelta(days=1))
    day = first_day
    tp_cache: dict[int, np.ndarray] = {}

    def _tp_at(vt: datetime) -> np.ndarray | None:
        step = int((vt - init).total_seconds() // 3600)
        if step < 0 or step > min(ECMWF_MAX_FHR, 10**6):
            return None
        if step == 0:
            if lat is None:
                _tp_lonlat(24)
            return np.zeros((len(lat), len(lon)), dtype='float32') if lat is not None else None
        if step not in tp_cache:
            v = _fetch_tp(init, step, tmp)
            if v is None:
                return None
            tp_cache[step] = v
        return tp_cache[step]

    while True:
        d0 = datetime(day.year, day.month, day.day)              # 00Z do dia
        d1 = d0 + timedelta(days=1)                              # 00Z do dia seguinte
        if d1 > end:
            break  # dia incompleto (fronteira final alem do horizonte)

        fname = f'ecmwf_precip_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
        nc_path = DIR_ECMWF_PRECIP / fname
        if nc_path.exists() and not force_redownload:
            logger.info('ECMWF chuva valido {} (init {}Z) ja existe — pulando.', day, init.hour)
            out.append(nc_path)
            day += timedelta(days=1)
            continue

        if lat is None:
            _tp_lonlat(int((d1 - init).total_seconds() // 3600))
        tp0, tp1 = _tp_at(d0), _tp_at(d1)
        if tp0 is None or tp1 is None or lat is None:
            logger.warning('ECMWF chuva {} incompleto (fronteira ausente) — dia ignorado.', day)
            day += timedelta(days=1)
            continue

        acum = np.clip((tp1 - tp0) * 1000.0, 0.0, None).astype('float32')  # m->mm, sem negativos
        da = xr.DataArray(
            acum[None, :, :], dims=['time', 'lat', 'lon'],
            coords={'time': [np.datetime64(d0)], 'lat': lat, 'lon': lon}, name='precip')
        da.attrs['units'] = 'mm'
        da.attrs['long_name'] = 'chuva acumulada diaria (00-24 UTC)'
        if nc_path.exists():
            nc_path.unlink()
        save_netcdf(da.to_dataset(name='precip'), nc_path)
        logger.info('ECMWF chuva valido {} salvo ({:.1f} mm max): {}',
                    day, float(np.nanmax(acum)), nc_path.name)
        out.append(nc_path)
        day += timedelta(days=1)

    if tmp.exists():
        tmp.unlink()
    logger.info('ECMWF chuva: {} dia(s) | init {:%Y-%m-%d %H}Z + {}h', len(out), init, lead_hours)
    return out
