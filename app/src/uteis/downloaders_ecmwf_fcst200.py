# app/src/uteis/downloaders_ecmwf_fcst200.py
# -*- coding: utf-8 -*-
"""
Downloader ECMWF HRES (previsao deterministica) de u/v/altura geopotencial em 200 hPa.

Fonte: ECMWF Open Data (https://data.ecmwf.int/forecasts), stream `oper` (HRES, IFS 0.25°).
Diferente do NOMADS (GFS/GEFS), o ECMWF nao tem um "grib filter": baixa-se cada campo por
REQUISICAO DE BYTE-RANGE no GRIB, usando os offsets do arquivo `.index` (uma linha JSON por
campo, com `_offset`/`_length`). Sem dependencia extra (so httpx, ja usado no projeto).

O HRES e o analogo deterministico do ECMWF (o membro de CONTROLE do ENS nao e exposto no
open data). Tem todas as variaveis do s34: u/v/HGT@200, TMP@850 e OLR (via `ttr`), ate 360h.

ECMWF entrega `gh` (altura geopotencial) ja em metros — nao precisa converter.

Gera **um NetCDF por dia valido** com as horas sinoticas (00/06/12/18) e as variaveis
`u`, `v`, `hgt` em 200 hPa — a MESMA estrutura dos arquivos GFS/GEFS/GDAS, para o pipeline
do s34 rodar sem alteracao.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import httpx
import numpy as np
import xarray as xr

from app.shared.logger import get_logger
from app.shared.settings_factory import settings

logger = get_logger(__name__)

ECMWF_BASE = 'https://data.ecmwf.int/forecasts'
DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)
ECMWF_MAX_FHR = 360  # HRES open data vai ate 360 h (15 dias)
ECMWF_STREAM = 'oper'   # HRES deterministico
ECMWF_TYPE = 'fc'

try:
    DIR_DADOS_BASE = Path(settings.DIR_DADOS)
except Exception:
    DIR_DADOS_BASE = Path('dados')

DIR_ECMWF_FCST200 = DIR_DADOS_BASE / 'ECMWF_FCST200'


# ---------------------------------------------------------------------------
# URLs e acesso ao open data (index + byte-range)
# ---------------------------------------------------------------------------
def _stamp(init: datetime) -> str:
    return init.strftime('%Y%m%d%H%M%S')


def _dir_url(init: datetime, stream: str = ECMWF_STREAM) -> str:
    return f'{ECMWF_BASE}/{init:%Y%m%d}/{init:%H}z/ifs/0p25/{stream}'


def ecmwf_grib_url(init: datetime, step: int, stream: str = ECMWF_STREAM, ftype: str = ECMWF_TYPE) -> str:
    return f'{_dir_url(init, stream)}/{_stamp(init)}-{step}h-{stream}-{ftype}.grib2'


def ecmwf_index_url(init: datetime, step: int, stream: str = ECMWF_STREAM, ftype: str = ECMWF_TYPE) -> str:
    return f'{_dir_url(init, stream)}/{_stamp(init)}-{step}h-{stream}-{ftype}.index'


def fetch_index(
    init: datetime, step: int, stream: str = ECMWF_STREAM, ftype: str = ECMWF_TYPE, timeout: int = 120,
) -> List[dict]:
    """Le o `.index` (uma linha JSON por campo do GRIB daquele passo). stream/ftype:
    'oper'/'fc' (HRES) ou 'enfo'/'ef' (ENS, 50 membros perturbados)."""
    url = ecmwf_index_url(init, step, stream, ftype)
    for attempt in range(1, 4):
        try:
            r = httpx.get(url, timeout=timeout, follow_redirects=True)
            r.raise_for_status()
            return [json.loads(ln) for ln in r.text.splitlines() if ln.strip()]
        except Exception as exc:
            logger.warning('Tentativa {}/3 falhou no index ECMWF {}: {}', attempt, url, exc)
            if attempt == 3:
                raise RuntimeError(f'Falha ao ler index ECMWF apos 3 tentativas: {url}') from exc


def match_record(
    records: List[dict], param: str, levelist: Optional[int] = None, number: Optional[int] = None,
) -> dict:
    """Acha o registro de `param` (e `levelist`/`number`, se dados) no index.

    `number` = membro do ensemble (1..50) para o stream ENS; None ignora (HRES nao tem)."""
    lev = None if levelist is None else str(levelist)
    num = None if number is None else str(number)
    for r in records:
        if (r.get('param') == param
                and (lev is None or r.get('levelist') == lev)
                and (num is None or r.get('number') == num)):
            return r
    raise KeyError(f'Campo {param} (nivel {levelist}, membro {number}) nao encontrado no index ECMWF.')


def range_bytes(grib_url: str, rec: dict, timeout: int = 180) -> bytes:
    """Baixa só os bytes do campo (HTTP Range) a partir do offset/length do index."""
    off, ln = int(rec['_offset']), int(rec['_length'])
    headers = {'Range': f'bytes={off}-{off + ln - 1}'}
    for attempt in range(1, 4):
        try:
            r = httpx.get(grib_url, headers=headers, timeout=timeout, follow_redirects=True)
            r.raise_for_status()
            return r.content
        except Exception as exc:
            logger.warning('Tentativa {}/3 falhou no range ECMWF {}: {}', attempt, grib_url, exc)
            if attempt == 3:
                raise RuntimeError(f'Falha ao baixar campo ECMWF apos 3 tentativas: {grib_url}') from exc


def open_grib_bytes(raw: bytes, tmp_path: Path) -> xr.Dataset:
    """Escreve bytes GRIB (1+ mensagens concatenadas) e abre com cfgrib, normalizando lat/lon."""
    tmp_path.write_bytes(raw)
    ds = xr.open_dataset(tmp_path, engine='cfgrib', backend_kwargs={'indexpath': ''})
    ren = {}
    for name in list(ds.dims) + list(ds.coords):
        low = name.lower()
        if low == 'latitude' and 'lat' not in ds.dims:
            ren[name] = 'lat'
        elif low == 'longitude' and 'lon' not in ds.dims:
            ren[name] = 'lon'
    if ren:
        ds = ds.rename(ren)
    return ds


# ---------------------------------------------------------------------------
# Serie diaria u/v/hgt 200 hPa
# ---------------------------------------------------------------------------
def _steps_for_day(
    init: datetime, day: date, hours: Sequence[int], lead_hours: int,
) -> List[Tuple[int, datetime]]:
    """Passos de previsao (step_h, valid_time) das horas sinoticas de `day` a partir de `init`."""
    steps = []
    for h in hours:
        vt = datetime(day.year, day.month, day.day, h)
        step = int((vt - init).total_seconds() // 3600)
        if 0 <= step <= min(lead_hours, ECMWF_MAX_FHR):
            steps.append((step, vt))
    return steps


def _open_uvhgt200(init: datetime, step: int, tmp_path: Path) -> xr.Dataset:
    """Baixa u/v/gh @ 200 (3 byte-ranges concatenados) e devolve u/v/hgt em (lat, lon)."""
    recs = fetch_index(init, step)
    grib_url = ecmwf_grib_url(init, step)
    raw = b''.join(range_bytes(grib_url, match_record(recs, p, 200)) for p in ('u', 'v', 'gh'))
    ds = open_grib_bytes(raw, tmp_path)
    if 'gh' in ds.data_vars:
        ds = ds.rename({'gh': 'hgt'})
    for coord in ('time', 'step', 'valid_time', 'isobaricInhPa', 'level', 'heightAboveGround'):
        if coord in ds.coords and coord not in ds.dims:
            ds = ds.drop_vars(coord, errors='ignore')
    keep = [v for v in ('u', 'v', 'hgt') if v in ds.data_vars]
    if not keep:
        raise KeyError(f'u/v/hgt nao encontrados no GRIB ECMWF. Disponiveis: {list(ds.data_vars)}')
    return ds[keep]


def _download_ecmwf_day(
    init: datetime, day: date, steps: List[Tuple[int, datetime]], force_redownload: bool,
) -> Path:
    fname = f'ecmwf_fcst200_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
    nc_path = DIR_ECMWF_FCST200 / fname
    if nc_path.exists() and not force_redownload:
        logger.info('ECMWF 200 hPa valido {} (init {}Z) ja existe — pulando.', day, init.hour)
        return nc_path

    DIR_ECMWF_FCST200.mkdir(parents=True, exist_ok=True)
    tmp = DIR_ECMWF_FCST200 / f'ecmwf_{init.strftime("%Y%m%d%H")}_tmp.grb2'
    parts = []
    for step, vt in steps:
        logger.info('  Baixando ECMWF init {}Z step {:03d}h (valido {:%Y-%m-%d %HZ})', init.hour, step, vt)
        ds = _open_uvhgt200(init, step, tmp).expand_dims(time=[np.datetime64(vt)])
        parts.append(ds.load())
    if tmp.exists():
        tmp.unlink()

    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    ds_day.to_netcdf(nc_path, engine='netcdf4')
    logger.info('ECMWF 200 hPa valido {} salvo: {}', day, nc_path.name)
    return nc_path


def ensure_ecmwf_fcst200_for_period(
    init: datetime,
    lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS,
    force_redownload: bool = False,
) -> List[Path]:
    """Garante NetCDFs diarios (u/v/hgt 200 hPa) do ECMWF HRES para [init, init+lead_hours]."""
    files: List[Path] = []
    end = init + timedelta(hours=lead_hours)
    day = init.date()
    while day <= end.date():
        steps = _steps_for_day(init, day, hours, lead_hours)
        if steps:
            files.append(_download_ecmwf_day(init, day, steps, force_redownload))
        day += timedelta(days=1)

    logger.info(
        'ECMWF FCST200: {} arquivos diarios | init {:%Y-%m-%d %H}Z + {}h',
        len(files), init, lead_hours,
    )
    return files
