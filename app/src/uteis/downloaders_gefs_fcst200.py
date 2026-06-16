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

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Sequence, Tuple

import httpx
import numpy as np
import xarray as xr

from app.shared.logger import get_logger
from app.shared.settings_factory import settings
# Reusa os parsers de GRIB (model-agnosticos) do downloader GFS, evitando duplicacao.
from app.src.uteis.downloaders_gfs_fcst200 import _open_gfs_grb2 as _open_gefs_grb2

logger = get_logger(__name__)

NOMADS_FILTER_URL = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p50a.pl'
DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)
GEFS_MAX_FHR = 384  # GEFS 0.5° vai ate 384 h (16 dias)
GEFS_MEMBER = 'geavg'  # media do ensemble (analogo ao GFS deterministico)

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


def _download_grb2(params: dict, target: Path, timeout: int = 180) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.part')
    for attempt in range(1, 4):
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
        except Exception as exc:
            logger.warning('Tentativa {}/{} falhou para {}: {}', attempt, 3, target.name, exc)
            if tmp.exists():
                tmp.unlink()
            if attempt == 3:
                raise RuntimeError(
                    f'Falha ao baixar GEFS 200 hPa apos 3 tentativas: {target.name}'
                ) from exc


def _steps_for_day(
    init: datetime, day: date, hours: Sequence[int], lead_hours: int,
) -> List[Tuple[int, datetime]]:
    """Passos de previsao (fhr, valid_time) das horas sinoticas de `day` a partir de `init`."""
    steps = []
    for h in hours:
        vt = datetime(day.year, day.month, day.day, h)
        fhr = int((vt - init).total_seconds() // 3600)
        if 0 <= fhr <= min(lead_hours, GEFS_MAX_FHR):
            steps.append((fhr, vt))
    return steps


def _download_gefs_day(
    init: datetime, day: date, steps: List[Tuple[int, datetime]], force_redownload: bool,
) -> Path:
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
            _download_grb2(_build_filter_params(init, fhr), grb)
        ds = _open_gefs_grb2(grb).expand_dims(time=[np.datetime64(vt)])
        parts.append(ds.load())

    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    ds_day.to_netcdf(nc_path, engine='netcdf4')

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
    files: List[Path] = []
    end = init + timedelta(hours=lead_hours)
    day = init.date()
    while day <= end.date():
        steps = _steps_for_day(init, day, hours, lead_hours)
        if steps:
            files.append(_download_gefs_day(init, day, steps, force_redownload))
        day += timedelta(days=1)

    logger.info(
        'GEFS FCST200: {} arquivos diarios | init {:%Y-%m-%d %H}Z + {}h',
        len(files), init, lead_hours,
    )
    return files
