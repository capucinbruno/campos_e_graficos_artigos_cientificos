# app/src/uteis/backfill_hgt700_aws.py
# -*- coding: utf-8 -*-
"""Backfill de Z700 (GFS/GEFS) a partir dos buckets AWS S3 (Open Data) da NOAA.

O NOMADS so retem ~10 dias de rodadas; para preencher o passado (~60 dias) do arquivo de
verificacao do s35 usamos os buckets S3, que retem meses:
- GFS  : ``noaa-gfs-bdp-pds``  (0.25°, ``gfs.tHHz.pgrb2.0p25.fFFF``)
- GEFS : ``noaa-gefs-pds``     (0.5°, media do ensemble ``geavg.tHHz.pgrb2a.0p50.fFFF``)

So a chamada de FETCH muda em relacao aos downloaders NOMADS: aqui baixamos por byte-range a
partir do ``.idx`` (mesma tecnica do downloader CFS). Todo o resto (passos sinoticos por dia,
parser do GRIB, escrita do NetCDF, paralelismo) e REUSADO dos modulos existentes, e o NetCDF
de saida tem o MESMO nome/dir do downloader NOMADS — logo os arquivos sao intercambiaveis e
cacheados (uma data ja baixada via NOMADS nao e re-baixada do S3).
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Sequence, Tuple

import httpx
import numpy as np
import xarray as xr

from app.common.forecast_download import (
    StepNotAvailable,
    download_days_parallel,
    save_netcdf,
)
from app.shared.logger import get_logger
from app.src.uteis.forecast_daily import DEFAULT_SYNOPTIC_HOURS
# Reuso direto da maquinaria dos downloaders NOMADS (passos + parser + dir de saida):
from app.src.uteis.downloaders_gfs_fcst200 import _steps_for_day as _gfs_steps_for_day
from app.src.uteis.downloaders_gefs_fcst200 import _steps_for_day as _gefs_steps_for_day
from app.src.uteis.downloaders_gfs_hgt700 import DIR_GFS_HGT700, _open_gfs_hgt700
from app.src.uteis.downloaders_gefs_hgt700 import DIR_GEFS_HGT700

logger = get_logger(__name__)

_HTTP_RETRIES = 4

# Config por modelo: como montar a URL S3, onde salvar e qual cadencia de passos usar.
_MODELS = {
    'gfs': {
        'base': 'https://noaa-gfs-bdp-pds.s3.amazonaws.com',
        'subdir': lambda i: f'gfs.{i:%Y%m%d}/{i.hour:02d}/atmos',
        'fname': lambda i, f: f'gfs.t{i.hour:02d}z.pgrb2.0p25.f{f:03d}',
        'dir': DIR_GFS_HGT700,
        'prefix': 'gfs_hgt700',
        'steps': _gfs_steps_for_day,
    },
    'gefs': {
        'base': 'https://noaa-gefs-pds.s3.amazonaws.com',
        'subdir': lambda i: f'gefs.{i:%Y%m%d}/{i.hour:02d}/atmos/pgrb2ap5',
        'fname': lambda i, f: f'geavg.t{i.hour:02d}z.pgrb2a.0p50.f{f:03d}',
        'dir': DIR_GEFS_HGT700,
        'prefix': 'gefs_hgt700',
        'steps': _gefs_steps_for_day,
    },
}

BACKFILL_MODELS = tuple(_MODELS)  # ('gfs', 'gefs') — unicos com retencao S3 + downloader Z700


def _http_get(url: str, headers=None, timeout: int = 120) -> httpx.Response:
    """GET com backoff exponencial; 404/403 no .idx -> StepNotAvailable (rodada ausente)."""
    last: Exception | None = None
    for attempt in range(1, _HTTP_RETRIES + 1):
        try:
            r = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
            r.raise_for_status()
            return r
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (403, 404) and 'Range' not in (headers or {}):
                raise StepNotAvailable(url) from None
            last = exc
        except Exception as exc:  # rede instavel -> tenta de novo
            last = exc
        if attempt < _HTTP_RETRIES:
            time.sleep(min(2 ** (attempt - 1), 15))
    raise RuntimeError(f'Falha AWS S3 apos {_HTTP_RETRIES} tentativas: {url}') from last


def _hgt700_byte_range(idx_text: str) -> Tuple[int, int | None] | None:
    """Acha o registro HGT @ 700 mb no `.idx` e devolve (offset_inicial, offset_final) em bytes.
    O fim e o inicio do PROXIMO registro - 1 (ou aberto, se for o ultimo do arquivo)."""
    lines = idx_text.splitlines()
    for i, ln in enumerate(lines):
        f = ln.split(':')
        if len(f) >= 5 and f[3] == 'HGT' and f[4] == '700 mb':
            start = int(f[1])
            for nxt in lines[i + 1:]:
                nf = nxt.split(':')
                if len(nf) >= 2 and nf[1].isdigit():
                    return start, int(nf[1]) - 1
            return start, None
    return None


def _aws_fetch_grb2(model: str, init: datetime, fhr: int, out_path: Path) -> None:
    """Baixa SO o campo HGT@700 do passo `fhr` (byte-range via .idx) e grava no `.grb2`."""
    cfg = _MODELS[model]
    url = f"{cfg['base']}/{cfg['subdir'](init)}/{cfg['fname'](init, fhr)}"
    rng = _hgt700_byte_range(_http_get(url + '.idx').text)
    if rng is None:
        raise StepNotAvailable(url)  # registro HGT700 ausente nesse passo
    start, end = rng
    hdr = {'Range': f'bytes={start}-' + ('' if end is None else str(end))}
    out_path.write_bytes(_http_get(url, headers=hdr).content)


def _download_day_aws(model: str, init: datetime, day: date,
                      steps: List[Tuple[int, datetime]], force: bool):
    """Espelha o `_download_day` do downloader NOMADS, trocando so o FETCH por S3 byte-range.
    Salva o NetCDF com o mesmo nome/dir do NOMADS (intercambiavel / cacheado)."""
    cfg = _MODELS[model]
    nc_path = cfg['dir'] / f"{cfg['prefix']}_{init:%Y%m%d%H}_valid{day:%Y%m%d}.nc"
    if nc_path.exists() and not force:
        return nc_path
    cfg['dir'].mkdir(parents=True, exist_ok=True)
    parts = []
    for fhr, vt in steps:
        grb = cfg['dir'] / f"{cfg['prefix']}_{init:%Y%m%d%H}_f{fhr:03d}.grb2"
        if not grb.exists() or force:
            try:
                _aws_fetch_grb2(model, init, fhr, grb)
            except StepNotAvailable:
                logger.warning('  {} Z700 f{:03d} ausente no S3 — pulando passo', model.upper(), fhr)
                continue
        ds = _open_gfs_hgt700(grb).expand_dims(time=[np.datetime64(vt)])
        parts.append(ds.load())
    if not parts:
        logger.warning('{} Z700 valido {} sem passos no S3 — dia ignorado.', model.upper(), day)
        return None
    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    save_netcdf(ds_day, nc_path)
    for fhr, _ in steps:
        grb = cfg['dir'] / f"{cfg['prefix']}_{init:%Y%m%d%H}_f{fhr:03d}.grb2"
        if grb.exists():
            grb.unlink()
    return nc_path


def backfill_hgt700_aws(model: str, init: datetime, lead_hours: int,
                        hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS) -> List[Path]:
    """NetCDFs diarios de Z700 (m) do `model` ('gfs'/'gefs') a partir do S3, para
    [init, init+lead_hours]. Mesma interface/saida do `ensure_*_hgt700_fcst_for_period`."""
    if model not in _MODELS:
        raise ValueError(f"backfill S3 nao suportado para '{model}' (so {BACKFILL_MODELS})")
    cfg = _MODELS[model]
    end = init + timedelta(hours=lead_hours)
    jobs = []
    day = init.date()
    while day <= end.date():
        steps = cfg['steps'](init, day, hours, lead_hours)
        if steps:
            jobs.append((day, steps))
        day += timedelta(days=1)
    files = download_days_parallel(
        jobs, lambda day, steps: _download_day_aws(model, init, day, steps, False), logger)
    logger.info('{} Z700 (AWS S3): {} arquivos | init {:%Y-%m-%d %H}Z + {}h',
                model.upper(), len(files), init, lead_hours)
    return files
