# app/src/uteis/downloaders_ecmwf_ptype.py
# -*- coding: utf-8 -*-
"""
Downloader ECMWF HRES (previsao) de TIPO de precipitacao via ECMWF Open Data.

Variavel `ptype` (precipitation type), instantanea, codigo da tabela OMM 4.201 (ver `PTYPE_CODES`
abaixo). Confirmado por consulta real ao `.index` do open data (2026-07-18): existe no MESMO
cadenciamento nativo do resto do HRES -- 3 em 3h ate 144h, 6 em 6h dai ate 360h -- sem corte
proprio (mesma cobertura do `tp`/`msl`).

Diferente de `tp`/`msl` (que agregam num NetCDF diario por hora SINOTICA ou por soma do dia),
`ptype` e um instantaneo categorico: nao faz sentido "media diaria" (misturar codigo 1=chuva com
5=neve nao produz nada util). Por isso o NetCDF diario aqui guarda TODOS os passos nativos
publicados naquele dia (3h ou 6h, conforme o horizonte) na dimensao `time` -- quem for plotar
(estilo Tropical Tidbits: chuva/chuva-congelante/neve/granizo por cor, num horario especifico)
escolhe o passo que quer, sem o downloader decidir por ele.

Forecast-only (open data nao publica reanalise ERA5 de tipo de precipitacao).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import xarray as xr

from app.common.forecast_download import StepNotAvailable, save_netcdf
from app.shared.logger import get_logger
from app.src.uteis.downloaders_ecmwf_fcst200 import (
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

DIR_ECMWF_PTYPE = DIR_DADOS_BASE / 'ECMWF_PTYPE'

# Tabela OMM 4.201 (subconjunto que o ECMWF HRES realmente publica -- confirmado por amostragem
# real do campo: 0/1/3/5/6/7/8/12 apareceram todos num unico passo).
PTYPE_CODES: Dict[int, str] = {
    0: 'sem precipitacao',
    1: 'chuva',
    3: 'chuva congelante',
    5: 'neve',
    6: 'neve molhada',
    7: 'chuva e neve mista',
    8: 'granizo (ice pellets)',
    12: 'garoa congelante',
}


def _fetch_ptype(init: datetime, step: int, tmp_path: Path) -> xr.DataArray | None:
    """`ptype` (codigo OMM 4.201, instantaneo) no passo `step`, em (lat, lon). None se indisponivel."""
    try:
        recs = fetch_index(init, step, stream=ECMWF_STREAM, ftype=ECMWF_TYPE, model=ECMWF_MODEL)
        raw = range_bytes(ecmwf_grib_url(init, step, stream=ECMWF_STREAM, ftype=ECMWF_TYPE,
                                         model=ECMWF_MODEL), match_record(recs, 'ptype'))
    except StepNotAvailable:
        logger.warning('  ECMWF ptype step {:03d}h ainda nao publicado (404) — passo ausente', step)
        return None
    ds = open_grib_bytes(raw, tmp_path)
    da = ds['ptype'] if 'ptype' in ds.data_vars else ds[list(ds.data_vars)[0]]
    for coord in ('step', 'valid_time', 'surface', 'number', 'heightAboveGround'):
        if coord in da.coords and coord not in da.dims:
            da = da.drop_vars(coord, errors='ignore')
    return da.astype('int8').rename('ptype')  # codigos 0-12 cabem sobrando em int8


def ensure_ecmwf_ptype_fcst_for_period(
    init: datetime, lead_hours: int, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de tipo de precipitacao (`ptype`, codigo OMM 4.201) do ECMWF HRES p/
    [init, init+lead_hours]. Um arquivo por dia com TODOS os passos nativos publicados nesse dia
    (ver `_native_steps`) na dimensao `time` -- ver docstring do modulo p/ o motivo de nao agregar."""
    DIR_ECMWF_PTYPE.mkdir(parents=True, exist_ok=True)
    tmp = DIR_ECMWF_PTYPE / f'ecmwf_ptype_{init.strftime("%Y%m%d%H")}_tmp.grb2'
    out: List[Path] = []

    por_dia: Dict[date, list] = {}
    for s in ecmwf_native_steps(lead_hours):
        vt = init + timedelta(hours=s)
        por_dia.setdefault(vt.date(), []).append((vt, s))

    # Ver comentario equivalente em `ensure_ecmwf_tp_native_fcst_for_period`: 2 dias vazios
    # seguidos == borda real do horizonte, para de sondar o resto (evita 429 do open data).
    dias_vazios_seguidos = 0

    for day in sorted(por_dia):
        steps = por_dia[day]
        nc_path = DIR_ECMWF_PTYPE / (f'ecmwf_ptype_{init.strftime("%Y%m%d%H")}'
                                     f'_valid{day.strftime("%Y%m%d")}.nc')
        if nc_path.exists() and not force_redownload:
            logger.info('ECMWF ptype valido {} (init {}Z) ja existe — pulando.', day, init.hour)
            out.append(nc_path)
            dias_vazios_seguidos = 0
            continue

        parts = []
        for vt, s in steps:
            da = _fetch_ptype(init, s, tmp)
            if da is not None:
                parts.append(da.expand_dims(time=[np.datetime64(vt)]))
        if not parts:
            logger.warning('ECMWF ptype {} sem nenhum passo publicado — dia ignorado.', day)
            dias_vazios_seguidos += 1
            if dias_vazios_seguidos >= 2:
                logger.info('ECMWF ptype: {} dias vazios seguidos -- borda do horizonte, parando.', dias_vazios_seguidos)
                break
            continue
        dias_vazios_seguidos = 0

        ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
        if nc_path.exists():
            nc_path.unlink()
        save_netcdf(ds_day.to_dataset(name='ptype') if isinstance(ds_day, xr.DataArray) else ds_day,
                    nc_path)
        logger.info('ECMWF ptype valido {} salvo ({} passo(s)): {}', day, len(parts), nc_path.name)
        out.append(nc_path)

    if tmp.exists():
        tmp.unlink()
    logger.info('ECMWF ptype: {} arquivo(s) | init {:%Y-%m-%d %H}Z + {}h', len(out), init, lead_hours)
    return out
