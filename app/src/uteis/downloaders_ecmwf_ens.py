# app/src/uteis/downloaders_ecmwf_ens.py
# -*- coding: utf-8 -*-
"""
Downloader ECMWF-ENS (previsao por ensemble) — MEDIA DOS 50 MEMBROS — via ECMWF Open Data.

O open data do ECMWF NAO expoe o membro de controle nem a media pronta do ENS: so os 50
membros PERTURBADOS (stream `enfo`, type `ef`, `number` 1..50). Aqui calcula-se a media do
ensemble baixando cada membro por byte-range e mediando (analogo ao `geavg` do GEFS, porem
computado por nos). Reusa os helpers de acesso do downloader HRES (`downloaders_ecmwf_fcst200`).

Custo: ~50 requisicoes por campo/passo — os membros sao baixados em paralelo (ThreadPool,
`ECMWF_ENS_WORKERS`, default 8). Cada membro usa um arquivo temporario UNICO (cfgrib precisa
de path em disco), evitando colisao entre threads. Numero de membros: `ECMWF_ENS_MEMBERS`
(default 50, pode reduzir para acelerar ao custo de representatividade).

OLR: como a desacumulacao de `ttr` e LINEAR, media(OLR) = desacumular(media(ttr)). Entao
media-se o `ttr` por passo (cache) e desacumula-se a media: OLR(S) = -(ttr_S - ttr_{S-6})/21600.

Gera **um NetCDF por dia valido** (u/v/hgt@200, t@850 e olr), mesma estrutura dos demais.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from app.common.forecast_download import StepNotAvailable
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
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

ENS_STREAM = 'enfo'
ENS_TYPE = 'ef'
ENS_N_MEMBERS = 50           # membros perturbados (1..50)
# Downloads de membros em paralelo. O ECMWF open data (S3/CDN) escala com conexoes —
# 32-48 ficam ~5-6x mais rapido que 8, sem perda de qualidade. (NAO vale p/ NOMADS, que bane.)
ENS_WORKERS_DEFAULT = 32
_ACC_SECONDS = 6 * 3600      # janela de acumulacao do ttr entre passos sinoticos (6 h)

DIR_ECMWF_ENS_FCST200 = DIR_DADOS_BASE / 'ECMWF_ENS_FCST200'
DIR_ECMWF_ENS_OLR = DIR_DADOS_BASE / 'ECMWF_ENS_OLR'
DIR_ECMWF_ENS_TMP850 = DIR_DADOS_BASE / 'ECMWF_ENS_TMP850'


def _n_members() -> int:
    return int(settings.get('ECMWF_ENS_MEMBERS', ENS_N_MEMBERS))


def _workers() -> int:
    return max(1, int(settings.get('ECMWF_ENS_WORKERS', ENS_WORKERS_DEFAULT)))


def _fetch_member(grib_url: str, rec: dict, tmp_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Baixa o campo de UM membro (byte-range) e devolve (valores 2D, lat, lon)."""
    ds = open_grib_bytes(range_bytes(grib_url, rec), tmp_path)
    try:
        da = ds[list(ds.data_vars)[0]]
        vals = np.asarray(da.values, dtype='float64')
        lat, lon = da['lat'].values, da['lon'].values
    finally:
        ds.close()
        if tmp_path.exists():
            tmp_path.unlink()
    return vals, lat, lon


def _ens_mean_2d(
    init: datetime, step: int, param: str, levelist: Optional[int], tmp_dir: Path,
    recs: Optional[List[dict]] = None, grib_url: Optional[str] = None,
    stream: str = ENS_STREAM, ftype: str = ENS_TYPE, model: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Media de `param` (e `levelist`) sobre os membros do ENS no passo `step`. (vals, lat, lon).

    Generico no caminho do modelo: IFS ENS (default) ou AIFS-ENS (model='aifs-ens/0p25',
    stream='enfo', ftype='pf')."""
    n = _n_members()
    _mkw = {'stream': stream, 'ftype': ftype}
    if model is not None:
        _mkw['model'] = model
    if recs is None:
        recs = fetch_index(init, step, **_mkw)
    if grib_url is None:
        grib_url = ecmwf_grib_url(init, step, **_mkw)
    tasks = []
    for m in range(1, n + 1):
        rec = match_record(recs, param, levelist, number=m)
        tmp = tmp_dir / f'_ens_{step}_{param}_{levelist}_{m}.grb2'
        tasks.append((grib_url, rec, tmp))

    acc: Optional[np.ndarray] = None
    lat = lon = None
    with ThreadPoolExecutor(max_workers=_workers()) as ex:
        for vals, la, lo in ex.map(lambda t: _fetch_member(*t), tasks):
            acc = vals if acc is None else acc + vals
            if lat is None:
                lat, lon = la, lo
    return acc / n, lat, lon


# ---------------------------------------------------------------------------
# u/v/hgt 200 hPa (media do ensemble)
# ---------------------------------------------------------------------------
def _download_day_200(init: datetime, day: date, steps: List[Tuple[int, datetime]], force: bool) -> Path:
    fname = f'ecmwf_ens_fcst200_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
    nc_path = DIR_ECMWF_ENS_FCST200 / fname
    if nc_path.exists() and not force:
        logger.info('ECMWF-ENS 200 hPa valido {} (init {}Z) ja existe — pulando.', day, init.hour)
        return nc_path
    DIR_ECMWF_ENS_FCST200.mkdir(parents=True, exist_ok=True)
    parts = []
    for step, vt in steps:
        try:
            recs = fetch_index(init, step, stream=ENS_STREAM, ftype=ENS_TYPE)
            grib_url = ecmwf_grib_url(init, step, stream=ENS_STREAM, ftype=ENS_TYPE)
            logger.info('  ECMWF-ENS init {}Z step {:03d}h: media de {} membros (u/v/gh@200)',
                        init.hour, step, _n_members())
            u, lat, lon = _ens_mean_2d(init, step, 'u', 200, DIR_ECMWF_ENS_FCST200, recs, grib_url)
            v, _, _ = _ens_mean_2d(init, step, 'v', 200, DIR_ECMWF_ENS_FCST200, recs, grib_url)
            gh, _, _ = _ens_mean_2d(init, step, 'gh', 200, DIR_ECMWF_ENS_FCST200, recs, grib_url)
        except StepNotAvailable:
            logger.warning('  ECMWF-ENS step {:03d}h ainda nao publicado (404) — pulando', step)
            continue
        ds = xr.Dataset(
            {'u': (('lat', 'lon'), u), 'v': (('lat', 'lon'), v), 'hgt': (('lat', 'lon'), gh)},
            coords={'lat': lat, 'lon': lon},
        ).expand_dims(time=[np.datetime64(vt)])
        parts.append(ds)
    if not parts:
        logger.warning('ECMWF-ENS 200 hPa valido {} sem passos publicados — dia ignorado.', day)
        return None
    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    ds_day.to_netcdf(nc_path, engine='netcdf4')
    logger.info('ECMWF-ENS 200 hPa valido {} salvo: {}', day, nc_path.name)
    return nc_path


def ensure_ecmwf_ens_fcst200_for_period(
    init: datetime, lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios (u/v/hgt 200 hPa) da MEDIA do ECMWF-ENS para [init, init+lead_hours]."""
    files: List[Path] = []
    end = init + timedelta(hours=lead_hours)
    day = init.date()
    while day <= end.date():
        steps = _steps_for_day(init, day, hours, lead_hours)
        if steps:
            nc = _download_day_200(init, day, steps, force_redownload)
            if nc is not None:
                files.append(nc)
        day += timedelta(days=1)
    logger.info('ECMWF-ENS FCST200: {} arquivos | init {:%Y-%m-%d %H}Z + {}h', len(files), init, lead_hours)
    return files


# ---------------------------------------------------------------------------
# T850 (media do ensemble)
# ---------------------------------------------------------------------------
def _download_day_t850(init: datetime, day: date, steps: List[Tuple[int, datetime]], force: bool) -> Path:
    fname = f'ecmwf_ens_tmp850_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
    nc_path = DIR_ECMWF_ENS_TMP850 / fname
    if nc_path.exists() and not force:
        logger.info('ECMWF-ENS T850 valido {} (init {}Z) ja existe — pulando.', day, init.hour)
        return nc_path
    DIR_ECMWF_ENS_TMP850.mkdir(parents=True, exist_ok=True)
    parts = []
    for step, vt in steps:
        try:
            t, lat, lon = _ens_mean_2d(init, step, 't', 850, DIR_ECMWF_ENS_TMP850)
        except StepNotAvailable:
            logger.warning('  ECMWF-ENS T850 step {:03d}h ainda nao publicado (404) — pulando', step)
            continue
        ds = xr.Dataset({'t': (('lat', 'lon'), t)}, coords={'lat': lat, 'lon': lon})
        ds['t'].attrs['units'] = 'K'
        parts.append(ds.expand_dims(time=[np.datetime64(vt)]))
    if not parts:
        logger.warning('ECMWF-ENS T850 valido {} sem passos publicados — dia ignorado.', day)
        return None
    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    ds_day.to_netcdf(nc_path, engine='netcdf4')
    logger.info('ECMWF-ENS T850 valido {} salvo: {}', day, nc_path.name)
    return nc_path


def ensure_ecmwf_ens_tmp850_fcst_for_period(
    init: datetime, lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de T850 (K) da MEDIA do ECMWF-ENS para [init, init+lead_hours]."""
    files: List[Path] = []
    end = init + timedelta(hours=lead_hours)
    day = init.date()
    while day <= end.date():
        steps = _steps_for_day(init, day, hours, lead_hours)
        if steps:
            nc = _download_day_t850(init, day, steps, force_redownload)
            if nc is not None:
                files.append(nc)
        day += timedelta(days=1)
    logger.info('ECMWF-ENS T850: {} arquivos | init {:%Y-%m-%d %H}Z + {}h', len(files), init, lead_hours)
    return files


# ---------------------------------------------------------------------------
# OLR (media do ensemble; ttr desacumulado)
# ---------------------------------------------------------------------------
def ensure_ecmwf_ens_olr_fcst_for_period(
    init: datetime, lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs diarios de OLR (W/m2) da MEDIA do ECMWF-ENS para [init, init+lead_hours].

    media(OLR) = desacumular(media(ttr)): media-se o `ttr` por passo e desacumula-se a media.
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

    DIR_ECMWF_ENS_OLR.mkdir(parents=True, exist_ok=True)
    ttr_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def mean_ttr(step: int):
        if step <= 0:
            return None
        if step not in ttr_cache:
            logger.info('  ECMWF-ENS step {:03d}h: media de {} membros (ttr/OLR)', step, _n_members())
            try:
                ttr_cache[step] = _ens_mean_2d(init, step, 'ttr', None, DIR_ECMWF_ENS_OLR)
            except StepNotAvailable:
                ttr_cache[step] = None  # passo ainda nao publicado
        return ttr_cache[step]

    files: List[Path] = []
    for day, steps in days:
        fname = f'ecmwf_ens_olr_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
        nc_path = DIR_ECMWF_ENS_OLR / fname
        if nc_path.exists() and not force_redownload:
            logger.info('ECMWF-ENS OLR valido {} (init {}Z) ja existe — pulando.', day, init.hour)
            files.append(nc_path)
            continue
        parts = []
        for step, vt in steps:
            if step == 0:
                continue
            cur = mean_ttr(step)
            if cur is None:  # passo ainda nao publicado -> pula
                logger.warning('  ECMWF-ENS OLR step {:03d}h ainda nao publicado (404) — pulando', step)
                continue
            prev = mean_ttr(step - 6)
            cur_vals, lat, lon = cur
            prev_vals = 0.0 if prev is None else prev[0]
            olr = -(cur_vals - prev_vals) / _ACC_SECONDS
            ds = xr.Dataset({'olr': (('lat', 'lon'), olr)}, coords={'lat': lat, 'lon': lon})
            ds['olr'].attrs['units'] = 'W m-2'
            parts.append(ds.expand_dims(time=[np.datetime64(vt)]))
        if not parts:
            logger.warning('ECMWF-ENS OLR {} sem passos validos — dia ignorado.', day)
            continue
        ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
        if nc_path.exists():
            nc_path.unlink()
        ds_day.to_netcdf(nc_path, engine='netcdf4')
        logger.info('ECMWF-ENS OLR valido {} salvo: {}', day, nc_path.name)
        files.append(nc_path)

    logger.info('ECMWF-ENS OLR: {} arquivos | init {:%Y-%m-%d %H}Z + {}h', len(files), init, lead_hours)
    return files
