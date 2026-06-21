# app/src/uteis/downloaders_ai_nomads.py
# -*- coding: utf-8 -*-
"""
Downloaders dos modelos de IA do NCEP (NOMADS): AIGFS (deterministico) e AIGEFS (ensemble).

Diferente do GFS/GEFS, AIGFS/AIGEFS NAO tem "grib filter" — so HTTPS. Mas tem arquivo `.idx`
(indice classico do NOMADS: `registro:offset:d=data:PARAM:nivel:fcst:`), entao baixa-se SO os
campos necessarios por **byte-range** (Range HTTP a partir do offset do `.idx`).

AIGEFS tem **media do ensemble PRONTA** em `ensstat/products/...avg...` (analogo ao geavg do
GEFS) — usa-se o `avg` direto, 1 download por campo (nao baixa os 31 membros).

Ambos: 6-horario ate 384 h (16 dias); u/v/HGT@200 e TMP@850. **NAO tem OLR** (ULWRF) — como o
AIFS, geram 4 dos 5 mapas no s34. Um NetCDF por dia valido, mesma estrutura dos demais.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import httpx
import numpy as np
import xarray as xr

from app.common.forecast_download import StepNotAvailable, download_days_parallel, save_netcdf
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.src.uteis.downloaders_ecmwf_fcst200 import open_grib_bytes  # cfgrib+lock, normaliza lat/lon

logger = get_logger(__name__)

NOMADS_BASE = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com'
DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)
AI_MAX_FHR = 384  # AIGFS/AIGEFS vao ate 384 h (16 dias)

WANTED_200 = (('UGRD', '200 mb'), ('VGRD', '200 mb'), ('HGT', '200 mb'))
WANTED_T850 = (('TMP', '850 mb'),)
WANTED_HGT500 = (('HGT', '500 mb'),)
WANTED_HGT250 = (('HGT', '250 mb'),)
WANTED_UV250 = (('UGRD', '250 mb'), ('VGRD', '250 mb'))
WANTED_UV850 = (('UGRD', '850 mb'), ('VGRD', '850 mb'))
WANTED_T2M = (('TMP', '2 m above ground'),)

try:
    DIR_DADOS_BASE = Path(settings.DIR_DADOS)
except Exception:
    DIR_DADOS_BASE = Path('dados')


# ---------------------------------------------------------------------------
# URLs (AIGFS deterministico; AIGEFS usa a media `avg` pronta)
# ---------------------------------------------------------------------------
def _aigfs_urls(init: datetime, fhr: int) -> Tuple[str, str]:
    d, h = init.strftime('%Y%m%d'), init.strftime('%H')
    g = f'{NOMADS_BASE}/aigfs/prod/aigfs.{d}/{h}/model/atmos/grib2/aigfs.t{h}z.pres.f{fhr:03d}.grib2'
    return g, g + '.idx'


def _aigefs_urls(init: datetime, fhr: int) -> Tuple[str, str]:
    d, h = init.strftime('%Y%m%d'), init.strftime('%H')
    g = (f'{NOMADS_BASE}/aigefs/prod/aigefs.{d}/{h}/ensstat/products/atmos/grib2/'
         f'aigefs.t{h}z.pres.avg.f{fhr:03d}.grib2')
    return g, g + '.idx'


# ---------------------------------------------------------------------------
# Byte-range via .idx do NOMADS
# ---------------------------------------------------------------------------
def _parse_idx(text: str) -> List[Tuple[int, str, str]]:
    """Le o `.idx` do NOMADS -> lista (offset_inicio, PARAM, nivel) na ordem do arquivo."""
    recs = []
    for ln in text.splitlines():
        f = ln.split(':')
        if len(f) >= 5 and f[1].isdigit():
            recs.append((int(f[1]), f[3], f[4]))
    return recs


def _byte_range(recs: List[Tuple[int, str, str]], param: str, level: str) -> Tuple[int, int]:
    """Faixa de bytes (inicio, fim) do registro (param, nivel). fim=-1 -> ate o fim do arquivo."""
    for i, (start, p, lev) in enumerate(recs):
        if p == param and lev == level:
            end = recs[i + 1][0] - 1 if i + 1 < len(recs) else -1
            return start, end
    # Registro ausente no .idx = essa variavel/nivel NAO existe nesse modelo de IA (ex.: AIGFS sem
    # TMP@2m). Trata como StepNotAvailable -> o passo e PULADO (nao-fatal): a variavel simplesmente
    # nao sai e o s34 pula so aquele campo (desacoplamento), em vez de abortar a rodada.
    raise StepNotAvailable(f'{param}:{level} nao encontrado no .idx')


_HTTP_RETRIES = 6  # NOMADS faz throttling -> backoff exponencial recupera o passo


def _http_get(url: str, headers=None, timeout: int = 120) -> httpx.Response:
    for attempt in range(1, _HTTP_RETRIES + 1):
        try:
            r = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
            r.raise_for_status()
            return r
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:  # passo ainda nao publicado -> pula
                raise StepNotAvailable(url) from None
            last = exc  # 403/503 = throttling -> espera e tenta de novo
        except Exception as exc:
            last = exc
        if attempt < _HTTP_RETRIES:
            time.sleep(min(2 ** (attempt - 1), 15))  # backoff: 1,2,4,8,15s
    raise RuntimeError(f'Falha NOMADS apos {_HTTP_RETRIES} tentativas: {url}') from last


def _download_records(grib_url: str, idx_url: str, wanted: Sequence[Tuple[str, str]]) -> bytes:
    """Baixa SO os registros `wanted` (param, nivel) por byte-range; concatena as mensagens GRIB."""
    recs = _parse_idx(_http_get(idx_url).text)
    raw = b''
    for param, level in wanted:
        start, end = _byte_range(recs, param, level)
        rng = f'bytes={start}-' + ('' if end < 0 else str(end))
        raw += _http_get(grib_url, headers={'Range': rng}).content
    return raw


# ---------------------------------------------------------------------------
# Serie diaria
# ---------------------------------------------------------------------------
def _steps_for_day(init, day, hours, lead_hours):
    steps = []
    for h in hours:
        vt = datetime(day.year, day.month, day.day, h)
        fhr = int((vt - init).total_seconds() // 3600)
        if 0 <= fhr <= min(lead_hours, AI_MAX_FHR):
            steps.append((fhr, vt))
    return steps


def _open_200(raw: bytes, tmp: Path) -> xr.Dataset:
    ds = open_grib_bytes(raw, tmp)
    if 'gh' in ds.data_vars:
        ds = ds.rename({'gh': 'hgt'})
    for c in ('time', 'step', 'valid_time', 'isobaricInhPa', 'level', 'heightAboveGround'):
        if c in ds.coords and c not in ds.dims:
            ds = ds.drop_vars(c, errors='ignore')
    keep = [v for v in ('u', 'v', 'hgt') if v in ds.data_vars]
    if not keep:
        raise KeyError(f'u/v/hgt nao encontrados no GRIB AI. Disponiveis: {list(ds.data_vars)}')
    return ds[keep]


def _open_t850(raw: bytes, tmp: Path) -> xr.Dataset:
    ds = open_grib_bytes(raw, tmp)
    da = ds['t'] if 't' in ds.data_vars else ds[list(ds.data_vars)[0]].rename('t')
    for c in ('time', 'step', 'valid_time', 'isobaricInhPa', 'level', 'heightAboveGround'):
        if c in da.coords and c not in da.dims:
            da = da.drop_vars(c, errors='ignore')
    da.attrs['units'] = 'K'
    return da.to_dataset(name='t')


def _open_hgt500(raw: bytes, tmp: Path) -> xr.Dataset:
    ds = open_grib_bytes(raw, tmp)
    var = next((v for v in ('gh', 'hgt', 'z') if v in ds.data_vars), list(ds.data_vars)[0])
    da = ds[var].rename('hgt')
    for c in ('time', 'step', 'valid_time', 'isobaricInhPa', 'level', 'heightAboveGround'):
        if c in da.coords and c not in da.dims:
            da = da.drop_vars(c, errors='ignore')
    da.attrs['units'] = 'm'
    return da.to_dataset(name='hgt')


def _open_t2m(raw: bytes, tmp: Path) -> xr.Dataset:
    ds = open_grib_bytes(raw, tmp)
    var = next((v for v in ('t2m', 't', '2t') if v in ds.data_vars), list(ds.data_vars)[0])
    da = ds[var].rename('t2m')
    for c in ('time', 'step', 'valid_time', 'isobaricInhPa', 'level', 'heightAboveGround', 'surface'):
        if c in da.coords and c not in da.dims:
            da = da.drop_vars(c, errors='ignore')
    da.attrs['units'] = 'K'
    return da.to_dataset(name='t2m')


def _download_day(
    out_dir: Path, prefix: str, urls_fn: Callable, wanted, opener,
    init: datetime, day: date, steps: List[Tuple[int, datetime]], force: bool,
):
    fname = f'{prefix}_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
    nc_path = out_dir / fname
    if nc_path.exists() and not force:
        logger.info('{} valido {} (init {}Z) ja existe — pulando.', prefix, day, init.hour)
        return nc_path
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f'_tmp_{init.strftime("%Y%m%d%H")}_{day.strftime("%Y%m%d")}.grb2'
    parts = []
    for fhr, vt in steps:
        g, idx = urls_fn(init, fhr)
        try:
            raw = _download_records(g, idx, wanted)
        except StepNotAvailable as exc:
            logger.warning('  {} f{:03d} indisponivel ({}) — pulando passo', prefix, fhr, exc)
            continue
        except RuntimeError as exc:  # throttling persistente -> pula e segue (nao aborta a rodada)
            logger.warning('  {} f{:03d} indisponivel apos retries ({}) — pulando passo', prefix, fhr, exc)
            continue
        parts.append(opener(raw, tmp).expand_dims(time=[np.datetime64(vt)]))
    if tmp.exists():
        tmp.unlink()
    if not parts:
        logger.warning('{} valido {} sem passos publicados — dia ignorado.', prefix, day)
        return None
    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    save_netcdf(ds_day, nc_path)
    logger.info('{} valido {} salvo: {}', prefix, day, nc_path.name)
    return nc_path


def _ensure_period(out_dir, prefix, urls_fn, wanted, opener, init, lead_hours, hours, force):
    end = init + timedelta(hours=lead_hours)
    jobs = []
    day = init.date()
    while day <= end.date():
        steps = _steps_for_day(init, day, hours, lead_hours)
        if steps:
            jobs.append((day, steps))
        day += timedelta(days=1)
    # NOMADS faz throttling com muitas conexoes -> teto baixo (<=4), independente de DOWNLOAD_WORKERS
    workers = min(4, int(settings.get('DOWNLOAD_WORKERS', 6)))
    files = download_days_parallel(
        jobs, lambda day, steps: _download_day(out_dir, prefix, urls_fn, wanted, opener,
                                                init, day, steps, force), logger, workers=workers)
    logger.info('{}: {} arquivos | init {:%Y-%m-%d %H}Z + {}h', prefix, len(files), init, lead_hours)
    return files


# ---------------------------------------------------------------------------
# Interface publica (4 ensure_*, assinatura igual aos demais modelos)
# ---------------------------------------------------------------------------
DIR_AIGFS_FCST200 = DIR_DADOS_BASE / 'AIGFS_FCST200'
DIR_AIGFS_TMP850 = DIR_DADOS_BASE / 'AIGFS_TMP850'
DIR_AIGFS_HGT500 = DIR_DADOS_BASE / 'AIGFS_HGT500'
DIR_AIGFS_HGT250 = DIR_DADOS_BASE / 'AIGFS_HGT250'
DIR_AIGFS_UV250 = DIR_DADOS_BASE / 'AIGFS_UV250'
DIR_AIGFS_UV850 = DIR_DADOS_BASE / 'AIGFS_UV850'
DIR_AIGFS_T2M = DIR_DADOS_BASE / 'AIGFS_T2M'
DIR_AIGEFS_FCST200 = DIR_DADOS_BASE / 'AIGEFS_FCST200'
DIR_AIGEFS_TMP850 = DIR_DADOS_BASE / 'AIGEFS_TMP850'
DIR_AIGEFS_HGT500 = DIR_DADOS_BASE / 'AIGEFS_HGT500'
DIR_AIGEFS_HGT250 = DIR_DADOS_BASE / 'AIGEFS_HGT250'
DIR_AIGEFS_UV250 = DIR_DADOS_BASE / 'AIGEFS_UV250'
DIR_AIGEFS_UV850 = DIR_DADOS_BASE / 'AIGEFS_UV850'
DIR_AIGEFS_T2M = DIR_DADOS_BASE / 'AIGEFS_T2M'


def ensure_aigfs_fcst200_for_period(init, lead_hours, hours=DEFAULT_SYNOPTIC_HOURS, force_redownload=False):
    """NetCDFs diarios (u/v/hgt 200 hPa) do AIGFS para [init, init+lead_hours]."""
    return _ensure_period(DIR_AIGFS_FCST200, 'aigfs_fcst200', _aigfs_urls, WANTED_200, _open_200,
                          init, lead_hours, hours, force_redownload)


def ensure_aigfs_tmp850_fcst_for_period(init, lead_hours, hours=DEFAULT_SYNOPTIC_HOURS, force_redownload=False):
    """NetCDFs diarios de T850 (K) do AIGFS para [init, init+lead_hours]."""
    return _ensure_period(DIR_AIGFS_TMP850, 'aigfs_tmp850', _aigfs_urls, WANTED_T850, _open_t850,
                          init, lead_hours, hours, force_redownload)


def ensure_aigefs_fcst200_for_period(init, lead_hours, hours=DEFAULT_SYNOPTIC_HOURS, force_redownload=False):
    """NetCDFs diarios (u/v/hgt 200 hPa) da MEDIA do AIGEFS (produto `avg`) para o periodo."""
    return _ensure_period(DIR_AIGEFS_FCST200, 'aigefs_fcst200', _aigefs_urls, WANTED_200, _open_200,
                          init, lead_hours, hours, force_redownload)


def ensure_aigefs_tmp850_fcst_for_period(init, lead_hours, hours=DEFAULT_SYNOPTIC_HOURS, force_redownload=False):
    """NetCDFs diarios de T850 (K) da MEDIA do AIGEFS (produto `avg`) para o periodo."""
    return _ensure_period(DIR_AIGEFS_TMP850, 'aigefs_tmp850', _aigefs_urls, WANTED_T850, _open_t850,
                          init, lead_hours, hours, force_redownload)


def ensure_aigfs_hgt500_fcst_for_period(init, lead_hours, hours=DEFAULT_SYNOPTIC_HOURS, force_redownload=False):
    """NetCDFs diarios de Z500 (m) do AIGFS para [init, init+lead_hours]."""
    return _ensure_period(DIR_AIGFS_HGT500, 'aigfs_hgt500', _aigfs_urls, WANTED_HGT500, _open_hgt500,
                          init, lead_hours, hours, force_redownload)


def ensure_aigefs_hgt500_fcst_for_period(init, lead_hours, hours=DEFAULT_SYNOPTIC_HOURS, force_redownload=False):
    """NetCDFs diarios de Z500 (m) da MEDIA do AIGEFS (produto `avg`) para o periodo."""
    return _ensure_period(DIR_AIGEFS_HGT500, 'aigefs_hgt500', _aigefs_urls, WANTED_HGT500, _open_hgt500,
                          init, lead_hours, hours, force_redownload)


def ensure_aigfs_hgt250_fcst_for_period(init, lead_hours, hours=DEFAULT_SYNOPTIC_HOURS, force_redownload=False):
    """NetCDFs diarios de Z250 (m) do AIGFS para [init, init+lead_hours]."""
    return _ensure_period(DIR_AIGFS_HGT250, 'aigfs_hgt250', _aigfs_urls, WANTED_HGT250, _open_hgt500,
                          init, lead_hours, hours, force_redownload)


def ensure_aigefs_hgt250_fcst_for_period(init, lead_hours, hours=DEFAULT_SYNOPTIC_HOURS, force_redownload=False):
    """NetCDFs diarios de Z250 (m) da MEDIA do AIGEFS (produto `avg`) para o periodo."""
    return _ensure_period(DIR_AIGEFS_HGT250, 'aigefs_hgt250', _aigefs_urls, WANTED_HGT250, _open_hgt500,
                          init, lead_hours, hours, force_redownload)


def ensure_aigfs_uv250_fcst_for_period(init, lead_hours, hours=DEFAULT_SYNOPTIC_HOURS, force_redownload=False):
    """NetCDFs diarios de u/v 250 hPa (m/s) do AIGFS para [init, init+lead_hours]."""
    return _ensure_period(DIR_AIGFS_UV250, 'aigfs_uv250', _aigfs_urls, WANTED_UV250, _open_200,
                          init, lead_hours, hours, force_redownload)


def ensure_aigefs_uv250_fcst_for_period(init, lead_hours, hours=DEFAULT_SYNOPTIC_HOURS, force_redownload=False):
    """NetCDFs diarios de u/v 250 hPa (m/s) da MEDIA do AIGEFS (produto `avg`) para o periodo."""
    return _ensure_period(DIR_AIGEFS_UV250, 'aigefs_uv250', _aigefs_urls, WANTED_UV250, _open_200,
                          init, lead_hours, hours, force_redownload)


def ensure_aigfs_uv850_fcst_for_period(init, lead_hours, hours=DEFAULT_SYNOPTIC_HOURS, force_redownload=False):
    """NetCDFs diarios de u/v 850 hPa (m/s) do AIGFS para [init, init+lead_hours]."""
    return _ensure_period(DIR_AIGFS_UV850, 'aigfs_uv850', _aigfs_urls, WANTED_UV850, _open_200,
                          init, lead_hours, hours, force_redownload)


def ensure_aigefs_uv850_fcst_for_period(init, lead_hours, hours=DEFAULT_SYNOPTIC_HOURS, force_redownload=False):
    """NetCDFs diarios de u/v 850 hPa (m/s) da MEDIA do AIGEFS (produto `avg`) para o periodo."""
    return _ensure_period(DIR_AIGEFS_UV850, 'aigefs_uv850', _aigefs_urls, WANTED_UV850, _open_200,
                          init, lead_hours, hours, force_redownload)


def ensure_aigfs_t2m_fcst_for_period(init, lead_hours, hours=DEFAULT_SYNOPTIC_HOURS, force_redownload=False):
    """NetCDFs diarios de T2m (K) do AIGFS para [init, init+lead_hours]."""
    return _ensure_period(DIR_AIGFS_T2M, 'aigfs_t2m', _aigfs_urls, WANTED_T2M, _open_t2m,
                          init, lead_hours, hours, force_redownload)


def ensure_aigefs_t2m_fcst_for_period(init, lead_hours, hours=DEFAULT_SYNOPTIC_HOURS, force_redownload=False):
    """NetCDFs diarios de T2m (K) da MEDIA do AIGEFS (produto `avg`) para o periodo."""
    return _ensure_period(DIR_AIGEFS_T2M, 'aigefs_t2m', _aigefs_urls, WANTED_T2M, _open_t2m,
                          init, lead_hours, hours, force_redownload)
