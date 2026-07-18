# app/src/uteis/downloaders_ecmwf_precip.py
# -*- coding: utf-8 -*-
"""
Downloader ECMWF HRES (previsao) de CHUVA (precipitacao acumulada) via ECMWF Open Data.

NAO confundir com PWAT: aqui e a precipitacao que CAIU (`tp`, total precipitation), nao a
agua precipitavel (`tcwv`). Saida = ACUMULADO DIARIO em mm.

Acumulacao no ECMWF: `tp` e' acumulado desde o INIT, de forma CONTINUA (nao reseta). Logo a chuva
de um dia UTC = tp(fim do dia) - tp(inicio do dia):
    chuva[D] = tp(D+1 00Z) - tp(D 00Z)
ATENCAO A UNIDADE: o HRES publica `tp` em METROS e o AIFS em kg/m2 (== mm) -- mesmo campo, mesma
fonte, unidades diferentes (medido em 2026-07-17). A conversao p/ mm e feita por `_tp_para_mm`,
que le a unidade DECLARADA no GRIB; um x1000 fixo inflava o AIFS em 1000x.
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


def _tp_para_mm(da) -> np.ndarray:
    """`tp` -> MILIMETROS, decidido pela UNIDADE DECLARADA no GRIB, nao pelo modelo.

    O ECMWF publica o MESMO campo `tp` em unidades DIFERENTES conforme o modelo (medido na rodada
    2026-07-17 00Z, passo +24h):
        IFS HRES -> units='m'         (max 0.2796, media global 0.00253)
        AIFS     -> units='kg m**-2'  (max 184.34, media global 2.46)
    kg/m2 == mm; metros == mm/1000. Converter por unidade (e nao por um x1000 fixo) faz o AIFS entrar
    certo hoje e protege se o ECMWF trocar a unidade de algum deles amanha. Unidade desconhecida ->
    assume metros (o historico do HRES) e avisa."""
    u = str(da.attrs.get('units', '')).strip().lower()
    vals = da.values.astype('float32')
    if u in ('kg m**-2', 'kg/m**2', 'kg m-2', 'kg/m2', 'mm'):
        return vals
    if u not in ('m', 'metres', 'meters'):
        logger.warning('tp com unidade inesperada {!r} — assumindo METROS (comportamento do HRES)', u)
    return vals * 1000.0


def _fetch_tp(init: datetime, step: int, tmp_path: Path, *, model: str | None = None,
              stream: str | None = None, ftype: str | None = None,
              tag: str = 'ECMWF') -> np.ndarray | None:
    """`tp` (metros, acumulado desde o init) no passo `step`, em (lat, lon). None se indisponivel.

    step 0 = analise: tp=0 (inicio da acumulacao) -> devolve zeros na 1a grade conhecida (tratado
    pelo chamador, que ja tem a grade).

    model/stream/ftype: default = HRES (ifs/0p25, oper, fc). Parametrizados porque o AIFS publica
    `tp` no MESMO formato (open data, byte-range, acumulado em metros) -- muda so o caminho."""
    from app.src.uteis.downloaders_ecmwf_fcst200 import ECMWF_MODEL, ECMWF_STREAM, ECMWF_TYPE
    model, stream = model or ECMWF_MODEL, stream or ECMWF_STREAM
    ftype = ftype or ECMWF_TYPE
    try:
        recs = fetch_index(init, step, stream=stream, ftype=ftype, model=model)
        raw = range_bytes(ecmwf_grib_url(init, step, stream=stream, ftype=ftype,
                                         model=model), match_record(recs, 'tp'))
    except StepNotAvailable:
        logger.warning('  {} tp step {:03d}h ainda nao publicado (404) — passo ausente', tag, step)
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
    return _tp_para_mm(da)


def ensure_tp_precip_daily(
    init: datetime, lead_hours: int, *, dir_out: Path, prefixo: str, tag: str,
    max_fhr: int, model: str | None = None, stream: str | None = None, ftype: str | None = None,
    force_redownload: bool = False,
) -> List[Path]:
    """Motor comum dos modelos do ECMWF Open Data que publicam `tp` (IFS HRES e AIFS).

    A convencao e a MESMA nos dois (acumulado desde o init, em metros, sem reset), entao a unica
    diferenca e o caminho (`model`/`stream`/`ftype`), a pasta e o rotulo. Ver os wrappers no fim
    deste modulo e em `downloaders_aifs_precip`."""
    dir_out.mkdir(parents=True, exist_ok=True)
    end = init + timedelta(hours=lead_hours)
    tmp = dir_out / f'{prefixo}_{init.strftime("%Y%m%d%H")}_tmp.grb2'

    # Grade + coordenadas (lidas 1x do 1o passo valido).
    lat = lon = None

    def _tp_lonlat(step: int):
        nonlocal lat, lon
        from app.src.uteis.downloaders_ecmwf_fcst200 import ECMWF_MODEL, ECMWF_STREAM, ECMWF_TYPE
        _m, _s = model or ECMWF_MODEL, stream or ECMWF_STREAM
        _f = ftype or ECMWF_TYPE
        try:
            recs = fetch_index(init, step, stream=_s, ftype=_f, model=_m)
            raw = range_bytes(ecmwf_grib_url(init, step, stream=_s, ftype=_f,
                                             model=_m), match_record(recs, 'tp'))
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
        if step < 0 or step > max_fhr:
            return None
        if step == 0:
            if lat is None:
                _tp_lonlat(24)
            return np.zeros((len(lat), len(lon)), dtype='float32') if lat is not None else None
        if step not in tp_cache:
            v = _fetch_tp(init, step, tmp, model=model, stream=stream, ftype=ftype, tag=tag)
            if v is None:
                return None
            tp_cache[step] = v
        return tp_cache[step]

    while True:
        d0 = datetime(day.year, day.month, day.day)              # 00Z do dia
        d1 = d0 + timedelta(days=1)                              # 00Z do dia seguinte
        if d1 > end:
            break  # dia incompleto (fronteira final alem do horizonte)

        fname = f'{prefixo}_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
        nc_path = dir_out / fname
        if nc_path.exists() and not force_redownload:
            logger.info('{} chuva valido {} (init {}Z) ja existe — pulando.', tag, day, init.hour)
            out.append(nc_path)
            day += timedelta(days=1)
            continue

        if lat is None:
            _tp_lonlat(int((d1 - init).total_seconds() // 3600))
        tp0, tp1 = _tp_at(d0), _tp_at(d1)
        if tp0 is None or tp1 is None or lat is None:
            logger.warning('{} chuva {} incompleto (fronteira ausente) — dia ignorado.', tag, day)
            day += timedelta(days=1)
            continue

        # tp0/tp1 ja vem em mm (`_tp_para_mm`); clip em 0 mata ruido negativo da diferenca.
        acum = np.clip(tp1 - tp0, 0.0, None).astype('float32')
        da = xr.DataArray(
            acum[None, :, :], dims=['time', 'lat', 'lon'],
            coords={'time': [np.datetime64(d0)], 'lat': lat, 'lon': lon}, name='precip')
        da.attrs['units'] = 'mm'
        da.attrs['long_name'] = 'chuva acumulada diaria (00-24 UTC)'
        if nc_path.exists():
            nc_path.unlink()
        save_netcdf(da.to_dataset(name='precip'), nc_path)
        logger.info('{} chuva valido {} salvo ({:.1f} mm max): {}',
                    tag, day, float(np.nanmax(acum)), nc_path.name)
        out.append(nc_path)
        day += timedelta(days=1)

    if tmp.exists():
        tmp.unlink()
    logger.info('{} chuva: {} dia(s) | init {:%Y-%m-%d %H}Z + {}h', tag, len(out), init, lead_hours)
    return out


def ensure_ecmwf_precip_fcst_for_period(
    init: datetime, lead_hours: int, hours=None, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs de CHUVA ACUMULADA DIARIA (mm) do ECMWF HRES p/ os dias UTC completos em [init, init+lead].

    `hours` e' ignorado (compat com a assinatura dos demais downloaders do globo): a chuva e' ACUMULADO
    DIARIO (00-24 UTC), nao snapshot sinotico.

    chuva[D] = (tp(D+1 00Z) - tp(D 00Z)) * 1000. Um dia so e' salvo com AMBAS as fronteiras (00Z do
    dia e do dia seguinte) disponiveis."""
    return ensure_tp_precip_daily(
        init, lead_hours, dir_out=DIR_ECMWF_PRECIP, prefixo='ecmwf_precip', tag='ECMWF',
        max_fhr=ECMWF_MAX_FHR, force_redownload=force_redownload)
