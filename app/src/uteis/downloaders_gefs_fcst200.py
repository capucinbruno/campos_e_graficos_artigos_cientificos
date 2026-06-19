# app/src/uteis/downloaders_gefs_fcst200.py
# -*- coding: utf-8 -*-
"""
Downloader GEFS (previsao) de u/v/altura geopotencial em 200 hPa via NOMADS Grib Filter.

Espelha o downloaders_gfs_fcst200, mas usa o GEFS — modelo de ENSEMBLE. Por decisao
do projeto, usa-se a MEDIA DO ENSEMBLE (membro `geavg`), que e o analogo direto ao GFS
deterministico (um campo so, porem mais suave). Resolucao 0.5° (pgrb2a). As variaveis
u/v/HGT@200, TMP@850 e ULWRF (topo) estao TODAS no mesmo arquivo `pgrb2a` — cada
downloader (200/olr/tmp850) faz a sua requisicao filtrada do mesmo arquivo, como no GFS.

GEFS entrega HGT (altura geopotencial) ja em metros — nao precisa converter.

Gera **um NetCDF por dia valido** com as horas sinoticas (00/06/12/18) e as variaveis
`u`, `v`, `hgt` em 200 hPa — a MESMA estrutura dos arquivos GFS/GDAS, para o pipeline
do s34 rodar sem alteracao.

Fonte: https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p50a.pl
"""

from __future__ import annotations

import random
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Sequence, Tuple

import httpx
import numpy as np
import xarray as xr

from app.common.forecast_download import StepNotAvailable, download_days_parallel, save_netcdf
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
# Reusa os parsers de GRIB (model-agnosticos) do downloader GFS, evitando duplicacao.
from app.src.uteis.downloaders_gfs_fcst200 import _open_gfs_grb2 as _open_gefs_grb2

logger = get_logger(__name__)

NOMADS_FILTER_URL = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p50a.pl'
DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)
# Horizonte do GEFS v12 depende do ciclo: o 00Z roda ATE 840 h (35 dias, subsazonal); os
# demais (06/12/18Z) so ate 384 h (16 dias). No trecho estendido (>240 h) a saida e 6-horaria
# — as horas sinoticas (multiplos de 6) seguem disponiveis.
GEFS_MAX_FHR = 384       # ciclos 06/12/18Z: 16 dias
GEFS_MAX_FHR_00Z = 840   # ciclo 00Z: 35 dias
GEFS_MEMBER = 'geavg'  # media do ensemble (analogo ao GFS deterministico)


def _gefs_max_fhr(init: datetime) -> int:
    """Maior passo de previsao do GEFS para o ciclo de `init` (840 h no 00Z, senao 384 h)."""
    return GEFS_MAX_FHR_00Z if init.hour == 0 else GEFS_MAX_FHR

try:
    DIR_DADOS_BASE = Path(settings.DIR_DADOS)
except Exception:
    DIR_DADOS_BASE = Path('dados')

DIR_GEFS_FCST200 = DIR_DADOS_BASE / 'GEFS_FCST200'


def _gefs_dir(init: datetime) -> str:
    """Subpasta do ciclo no NOMADS (produto pgrb2a 0.5°: pasta pgrb2ap5)."""
    return f'/gefs.{init.strftime("%Y%m%d")}/{init.hour:02d}/atmos/pgrb2ap5'


def _gefs_file(init: datetime, fhr: int) -> str:
    return f'{GEFS_MEMBER}.t{init.hour:02d}z.pgrb2a.0p50.f{fhr:03d}'


def _build_filter_params(init: datetime, fhr: int) -> dict:
    return {
        'file': _gefs_file(init, fhr),
        'lev_200_mb': 'on',
        'var_UGRD': 'on',
        'var_VGRD': 'on',
        'var_HGT': 'on',
        'dir': _gefs_dir(init),
    }


# NOMADS faz throttling (HTTP 403/429/503 ou 302 p/ pagina de erro) quando recebe muitas
# requisicoes rapidas — tipico do ciclo 00Z (35 dias = muitos passos estendidos) e da variavel
# baixada por ULTIMO no pipeline (OLR), que pega o servidor ja sob carga. A janela de throttle
# pode durar MINUTOS, entao o backoff precisa ser longo o bastante p/ atravessa-la; o jitter
# dessincroniza as threads paralelas (DOWNLOAD_WORKERS) que reincidem juntas no rate-limit.
_GEFS_HTTP_RETRIES = 8       # antes 6: 403/302 do NOMADS podem persistir por varios minutos
_GEFS_BACKOFF_CAP = 60       # teto do backoff exponencial (s): 1,2,4,8,16,32,60 (~123s + jitter)


def _download_grb2(params: dict, target: Path, timeout: int = 180) -> None:
    """Baixa um passo GRIB do GEFS via NOMADS grib filter.

    Tratamento de erro (NAO fatal — sempre pula o passo e segue, gerando figuras ate onde ha dado):
      - **404** (passo ainda nao publicado) -> `StepNotAvailable` imediato.
      - **403/429/503/302/timeout** (throttling do NOMADS) -> backoff exponencial COM JITTER
        (ate ~123s no total, 8 tentativas); se ainda assim esgotar, levanta `StepNotAvailable`
        (o passo e pulado). Antes isso virava `RuntimeError` fatal e abortava o s34 inteiro.

    O backoff foi alongado (era 1..15s/6 tentativas) porque o throttle do NOMADS chega a durar
    minutos: com o backoff curto, dias inteiros de OLR eram pulados e — pela intersecao de datas
    do s34 — derrubavam ate os mapas de z200/Ks/RWS/T850 daquele dia.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.part')
    last = None
    for attempt in range(1, _GEFS_HTTP_RETRIES + 1):
        try:
            with httpx.stream(
                'GET', NOMADS_FILTER_URL, params=params, timeout=timeout, follow_redirects=True,
            ) as r:
                r.raise_for_status()
                with open(tmp, 'wb') as f:
                    for chunk in r.iter_bytes(chunk_size=65536):
                        f.write(chunk)
            tmp.rename(target)
            return
        except httpx.HTTPStatusError as exc:
            if tmp.exists():
                tmp.unlink()
            if exc.response.status_code == 404:  # passo ainda nao publicado -> pula
                raise StepNotAvailable(target.name) from None
            last = exc  # 403/503 = throttling -> espera e tenta de novo
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            last = exc
        if attempt < _GEFS_HTTP_RETRIES:
            base = min(2 ** (attempt - 1), _GEFS_BACKOFF_CAP)  # 1,2,4,8,16,32,60
            wait = base + random.uniform(0, min(base, 5))  # jitter: dessincroniza threads paralelas
            logger.warning(
                '  GEFS {} tentativa {}/{} falhou ({}). Aguardando {:.1f}s (throttling NOMADS).',
                target.name, attempt, _GEFS_HTTP_RETRIES, last, wait)
            time.sleep(wait)
    # Esgotou as tentativas num erro nao-404 (throttling persistente): pula o passo e segue.
    logger.warning(
        '  GEFS {} falhou apos {} tentativas ({}) — pulando passo (throttling NOMADS).',
        target.name, _GEFS_HTTP_RETRIES, last)
    raise StepNotAvailable(target.name) from last


def _steps_for_day(
    init: datetime, day: date, hours: Sequence[int], lead_hours: int,
) -> List[Tuple[int, datetime]]:
    """Passos de previsao (fhr, valid_time) das horas sinoticas de `day` a partir de `init`."""
    max_fhr = _gefs_max_fhr(init)
    steps = []
    for h in hours:
        vt = datetime(day.year, day.month, day.day, h)
        fhr = int((vt - init).total_seconds() // 3600)
        if 0 <= fhr <= min(lead_hours, max_fhr):
            steps.append((fhr, vt))
    return steps


def _download_gefs_day(
    init: datetime, day: date, steps: List[Tuple[int, datetime]], force_redownload: bool,
):
    fname = f'gefs_fcst200_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
    nc_path = DIR_GEFS_FCST200 / fname
    if nc_path.exists() and not force_redownload:
        logger.info('GEFS 200 hPa valido {} (init {}Z) ja existe — pulando.', day, init.hour)
        return nc_path

    DIR_GEFS_FCST200.mkdir(parents=True, exist_ok=True)
    parts = []
    for fhr, vt in steps:
        grb = DIR_GEFS_FCST200 / f'gefs_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
        if not grb.exists() or force_redownload:
            logger.info('  Baixando GEFS init {}Z f{:03d} (valido {:%Y-%m-%d %HZ})', init.hour, fhr, vt)
            try:
                _download_grb2(_build_filter_params(init, fhr), grb)
            except StepNotAvailable:  # passo ainda nao publicado -> pula
                logger.warning('  GEFS f{:03d} ainda nao publicado (404) — pulando passo', fhr)
                continue
        ds = _open_gefs_grb2(grb).expand_dims(time=[np.datetime64(vt)])
        parts.append(ds.load())

    if not parts:  # dia inteiro ainda nao publicado -> ignora (figuras vao ate onde ha dado)
        logger.warning('GEFS 200 hPa valido {} sem passos publicados — dia ignorado.', day)
        return None
    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    save_netcdf(ds_day, nc_path)

    for fhr, _ in steps:
        grb = DIR_GEFS_FCST200 / f'gefs_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
        if grb.exists():
            grb.unlink()

    logger.info('GEFS 200 hPa valido {} salvo: {}', day, nc_path.name)
    return nc_path


def ensure_gefs_fcst200_for_period(
    init: datetime,
    lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS,
    force_redownload: bool = False,
) -> List[Path]:
    """Garante NetCDFs diarios (u/v/hgt 200 hPa) do GEFS (media do ensemble) para [init, init+lead].

    Um arquivo por dia valido, com as horas sinoticas disponiveis. Retorna a lista de paths.
    """
    end = init + timedelta(hours=lead_hours)
    jobs = []
    day = init.date()
    while day <= end.date():
        steps = _steps_for_day(init, day, hours, lead_hours)
        if steps:
            jobs.append((day, steps))
        day += timedelta(days=1)

    files = download_days_parallel(
        jobs, lambda day, steps: _download_gefs_day(init, day, steps, force_redownload), logger)
    logger.info(
        'GEFS FCST200: {} arquivos diarios | init {:%Y-%m-%d %H}Z + {}h',
        len(files), init, lead_hours,
    )
    return files
