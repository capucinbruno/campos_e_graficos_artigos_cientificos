# app/src/uteis/downloaders_ecmwf_olr.py
# -*- coding: utf-8 -*-
"""
Downloader ECMWF HRES (previsao) de OLR via ECMWF Open Data (byte-range).

No ECMWF o OLR NAO vem pronto: usa-se `ttr` (top net thermal radiation), que e ACUMULADO
em J/m² desde o passo 0 (stepType='accum'). Para obter o OLR medio (W/m²) na janela de 6 h
que termina no tempo valido, DESACUMULA-SE entre passos consecutivos e inverte-se o sinal:

    OLR(S) = -(ttr(S) - ttr(S-6)) / (6*3600)        [W/m²]

(no topo da atmosfera nao ha radiacao termica descendente, entao o fluxo termico liquido
e -OLR; o sinal negativo devolve o OLR como grandeza positiva, ~200-300 W/m²).

Um NetCDF por dia valido com as horas sinoticas, variavel 'olr' (W/m2) — mesma estrutura
dos downloaders GFS/GEFS. Reusa os helpers de acesso ao open data do downloader ECMWF 200.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from app.shared.logger import get_logger
from app.src.uteis.downloaders_ecmwf_fcst200 import (
    DEFAULT_SYNOPTIC_HOURS,
    DIR_DADOS_BASE,
    _steps_for_day,
    ecmwf_grib_url,
    fetch_index,
    match_record,
    open_grib_bytes,
    range_bytes,
)

logger = get_logger(__name__)

DIR_ECMWF_OLR = DIR_DADOS_BASE / 'ECMWF_OLR'
_ACC_SECONDS = 6 * 3600  # janela de acumulacao entre passos sinoticos (6 h)


def _fetch_ttr(init: datetime, step: int, tmp_path: Path) -> xr.DataArray:
    """Campo `ttr` (J/m², acumulado desde o passo 0) no passo `step`, em (lat, lon)."""
    recs = fetch_index(init, step)
    raw = range_bytes(ecmwf_grib_url(init, step), match_record(recs, 'ttr'))
    ds = open_grib_bytes(raw, tmp_path)
    da = ds['ttr'] if 'ttr' in ds.data_vars else ds[list(ds.data_vars)[0]]
    for c in ('time', 'step', 'valid_time', 'nominalTop', 'atmosphere', 'level', 'surface'):
        if c in da.coords and c not in da.dims:
            da = da.drop_vars(c, errors='ignore')
    return da.load()


def ensure_ecmwf_olr_fcst_for_period(
    init: datetime, lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de OLR (W/m2) do ECMWF HRES para [init, init+lead_hours].

    Deriva o OLR desacumulando `ttr` entre passos sinoticos (6 h). Pula o passo 0 (analise,
    sem janela de acumulacao). ttr(0)=0 (inicio da acumulacao) — usado quando S=6.
    """
    end = init + timedelta(hours=lead_hours)
    days: List[Tuple[date, List[Tuple[int, datetime]]]] = []
    day = init.date()
    while day <= end.date():
        steps = _steps_for_day(init, day, hours, lead_hours)
        if steps:
            days.append((day, steps))
        day += timedelta(days=1)
    if not days:
        return []

    DIR_ECMWF_OLR.mkdir(parents=True, exist_ok=True)
    tmp = DIR_ECMWF_OLR / f'ecmwf_olr_{init.strftime("%Y%m%d%H")}_tmp.grb2'
    ttr_cache: Dict[int, xr.DataArray] = {}

    def ttr(step: int) -> Optional[xr.DataArray]:
        if step <= 0:
            return None  # acumulacao comeca em 0 -> tratado como zero no calculo
        if step not in ttr_cache:
            ttr_cache[step] = _fetch_ttr(init, step, tmp)
        return ttr_cache[step]

    files: List[Path] = []
    for day, steps in days:
        fname = f'ecmwf_olr_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
        nc_path = DIR_ECMWF_OLR / fname
        if nc_path.exists() and not force_redownload:
            logger.info('ECMWF OLR valido {} (init {}Z) ja existe — pulando.', day, init.hour)
            files.append(nc_path)
            continue

        parts = []
        for step, vt in steps:
            if step == 0:
                continue  # analise: sem OLR (precisa de intervalo de acumulacao)
            cur = ttr(step)
            prev = ttr(step - 6)
            prev_vals = 0.0 if prev is None else prev.values
            olr = -(cur.values - prev_vals) / _ACC_SECONDS  # J/m² -> W/m², sinal -> outgoing
            da = xr.DataArray(olr, dims=cur.dims, coords=cur.coords, name='olr')
            da.attrs['units'] = 'W m-2'
            parts.append(da.expand_dims(time=[np.datetime64(vt)]))
        if not parts:
            logger.warning('ECMWF OLR {} sem passos validos — dia ignorado.', day)
            continue
        ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
        if nc_path.exists():
            nc_path.unlink()
        ds_day.to_dataset(name='olr').to_netcdf(nc_path, engine='netcdf4')
        logger.info('ECMWF OLR valido {} salvo: {}', day, nc_path.name)
        files.append(nc_path)

    if tmp.exists():
        tmp.unlink()
    logger.info('ECMWF OLR: {} arquivos | init {:%Y-%m-%d %H}Z + {}h', len(files), init, lead_hours)
    return files
