# app/src/uteis/downloaders_cfs_ensemble.py
# -*- coding: utf-8 -*-
"""
Downloader CFSv2 (NOMADS) — PSEUDO-ENSEMBLE DEFASADO (lagged), à la NCICS/Carl Schreck.

O CFSv2 operacional roda, a cada ciclo, 4 membros (`time_grib_01..04`), e há 4 ciclos/dia
(00/06/12/18Z). Combinam-se os 4 membros × 4 ciclos de um MESMO dia D = **16 previsões**, e
tira-se a média (alinhada por data válida) — o "pseudo-ensemble" do NCICS. Como os 4 ciclos
partem de horas diferentes (defasados 6h), a média é por data, não por lead.

Fonte: .../com/cfs/prod/cfs.{YYYYMMDD}/{HH}/time_grib_{0M}/{var}.{0M}.{YYYYMMDD}{HH}.daily.grb2
- `wnd200` = UGRD+VGRD @ 200 mb (CHI200 via Poisson no s31). `wnd850` = UGRD+VGRD @ 850 mb.
- 6-horário; os membros mais curtos vão a 1080h = **45 dias** (janela do NCICS). Byte-range via
  `.idx` corta os ~12 MB/membro dos primeiros 45 dias (registros contiguos a partir do offset 0).

Gera **um NetCDF da MÉDIA do ensemble** (u/v diário, lat/lon), formato compatível com o que o
s31 já consome (mesma `_daily_series_uv200`). Resiliente: membro que falhar é descartado.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import httpx
import numpy as np
import pandas as pd
import xarray as xr

from app.common.forecast_download import StepNotAvailable, save_netcdf
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.src.uteis.downloaders_ecmwf_fcst200 import open_grib_bytes

logger = get_logger(__name__)

CFS_BASE = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod'
# Bucket AWS Open Data (mesma estrutura do prod) — retem meses, p/ backfill de inits passados
# (o NOMADS prod so retem ~7 dias). Ver `base_url=` nas funcoes ensure_cfs_*.
CFS_BASE_AWS = 'https://noaa-cfs-pds.s3.amazonaws.com'
CFS_CYCLES = (0, 6, 12, 18)        # 4 ciclos/dia
CFS_MEMBERS = (1, 2, 3, 4)         # 4 membros/ciclo -> 16 no total
CFS_LEAD_DAYS = 45                 # piso dos membros (1080h) = janela do NCICS
CFS_MIN_MEMBERS = 8                # minimo de membros p/ aceitar o ensemble (de 16)
CFS_WORKERS_DEFAULT = 8

try:
    DIR_DADOS_BASE = Path(settings.DIR_DADOS)
except Exception:
    DIR_DADOS_BASE = Path('dados')

DIR_CFS_FCST200 = DIR_DADOS_BASE / 'CFS_FCST200'
DIR_CFS_UV850 = DIR_DADOS_BASE / 'CFS_UV850'
DIR_CFS_OLR = DIR_DADOS_BASE / 'CFS_OLR'

_HTTP_RETRIES = 6  # NOMADS faz throttling (403/503) -> backoff exponencial


def _workers() -> int:
    return max(1, int(settings.get('CFS_WORKERS', CFS_WORKERS_DEFAULT)))


def _member_urls(D: date, cycle: int, member: int, var: str,
                 base_url: str = CFS_BASE) -> Tuple[str, str]:
    d = D.strftime('%Y%m%d')
    g = (f'{base_url}/cfs.{d}/{cycle:02d}/time_grib_{member:02d}/'
         f'{var}.{member:02d}.{d}{cycle:02d}.daily.grb2')
    return g, g + '.idx'


def _http_get(url: str, headers=None, timeout: int = 120) -> httpx.Response:
    """GET com backoff exponencial; 404 -> StepNotAvailable (membro/ciclo ausente)."""
    last: Optional[Exception] = None
    for attempt in range(1, _HTTP_RETRIES + 1):
        try:
            r = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
            r.raise_for_status()
            return r
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (403, 404) and 'Range' not in (headers or {}):
                # .idx 404/403 = ciclo/membro nao publicado -> pula esse membro
                raise StepNotAvailable(url) from None
            last = exc
        except Exception as exc:
            last = exc
        if attempt < _HTTP_RETRIES:
            time.sleep(min(2 ** (attempt - 1), 15))
    raise RuntimeError(f'Falha NOMADS apos {_HTTP_RETRIES} tentativas: {url}') from last


def _parse_cfs_idx(text: str) -> List[Tuple[int, str, str, Optional[int]]]:
    """Le o `.idx` do CFS -> lista (offset, PARAM, nivel, fcst_hour) na ordem do arquivo."""
    recs = []
    for ln in text.splitlines():
        f = ln.split(':')
        if len(f) >= 6 and f[1].isdigit():
            m = re.match(r'(\d+)\s+hour', f[5])
            fhr = int(m.group(1)) if m else None
            recs.append((int(f[1]), f[3], f[4], fhr))
    return recs


def _fetch_member_daily(
    D: date, cycle: int, member: int, file_var: str, idx_params: Tuple[str, ...],
    level_label: str, out_rename: dict, lead_hours: int, tmp_dir: Path,
    base_url: str = CFS_BASE,
) -> Optional[xr.Dataset]:
    """Baixa os campos `idx_params`@`level_label` (byte-range ate `lead_hours`) de UM membro e
    devolve a MÉDIA DIÁRIA (time,lat,lon). `out_rename` mapeia o nome do cfgrib -> nome de saida
    (ex.: {'sulwrf':'olr'}); vazio = mantem os nomes (u/v). Retorna None se o membro nao publicado."""
    grib_url, idx_url = _member_urls(D, cycle, member, file_var, base_url)
    try:
        recs = _parse_cfs_idx(_http_get(idx_url).text)
    except StepNotAvailable:
        return None

    sel = [i for i, (o, p, lev, fh) in enumerate(recs)
           if p in idx_params and lev == level_label and fh is not None and fh <= lead_hours]
    if not sel:
        return None
    # Registros <= lead sao contiguos a partir do offset 0 (arquivo ordenado por dia).
    start = recs[sel[0]][0]
    last = max(sel)
    end = recs[last + 1][0] - 1 if last + 1 < len(recs) else -1
    rng = f'bytes={start}-' + ('' if end < 0 else str(end))
    raw = _http_get(grib_url, headers={'Range': rng}).content

    tmp = tmp_dir / f'_cfs_{file_var}_{D:%Y%m%d}{cycle:02d}_{member:02d}.grb2'
    try:
        ds = open_grib_bytes(raw, tmp)
    finally:
        if tmp.exists():
            tmp.unlink()

    if out_rename:
        ds = ds.rename({k: v for k, v in out_rename.items() if k in ds.data_vars})
        out_vars = [v for v in out_rename.values() if v in ds.data_vars]
    else:
        out_vars = list(ds.data_vars)
    if not out_vars:
        return None
    vt = pd.DatetimeIndex(pd.to_datetime(ds['valid_time'].values))
    data = {v: (('time', 'lat', 'lon'), np.asarray(ds[v].values, dtype='float32')) for v in out_vars}
    out = xr.Dataset(
        data, coords={'time': vt, 'lat': ds['lat'].values, 'lon': ds['lon'].values},
    ).sortby('lat').sortby('lon')
    ds.close()
    # media diaria por data valida
    return out.resample(time='1D').mean()


def _lagged_ensemble_mean(
    D: date, file_var: str, idx_params: Tuple[str, ...], level_label: str, out_rename: dict,
    lead_hours: int, tmp_dir: Path, base_url: str = CFS_BASE,
) -> Tuple[xr.Dataset, int]:
    """Média do pseudo-ensemble (16 membros = 4 ciclos × 4) alinhada por data válida.

    Retorna (ds_media, n_membros_validos)."""
    jobs = [(c, m) for c in CFS_CYCLES for m in CFS_MEMBERS]  # 16
    tmp_dir.mkdir(parents=True, exist_ok=True)

    def _one(cm):
        c, m = cm
        try:
            return _fetch_member_daily(
                D, c, m, file_var, idx_params, level_label, out_rename, lead_hours, tmp_dir, base_url)
        except Exception as exc:  # membro com erro persistente -> descarta (nao aborta o ensemble)
            logger.warning('CFS {} {:%Y%m%d} {:02d}Z m{:02d} falhou ({}) — descartado.',
                           file_var, D, c, m, exc)
            return None

    with ThreadPoolExecutor(max_workers=_workers()) as ex:
        results = list(ex.map(_one, jobs))
    members = [r for r in results if r is not None]
    n = len(members)
    if n < CFS_MIN_MEMBERS:
        raise RuntimeError(
            f'CFS {file_var}: só {n}/16 membros disponíveis p/ {D} (mínimo {CFS_MIN_MEMBERS}). '
            f'A rodada de {D} pode não estar completa no NOMADS.')
    logger.info('CFS {} {:%Y-%m-%d}: média de {}/16 membros do pseudo-ensemble.', file_var, D, n)
    # alinha por data (outer) e media ignorando NaN (1o dia tem cobertura parcial dos ciclos)
    combined = xr.concat(members, dim='member', join='outer')
    return combined.mean(dim='member', skipna=True), n


def _ensure_cfs_level(
    init: datetime, lead_hours: int, file_var: str, idx_params: Tuple[str, ...], level_label: str,
    out_rename: dict, out_dir: Path, prefix: str, force_redownload: bool,
    base_url: str = CFS_BASE,
) -> List[Path]:
    """Garante o NetCDF da média do pseudo-ensemble CFS (diário) p/ o dia D=init.date()."""
    D = init.date()
    out_dir.mkdir(parents=True, exist_ok=True)
    nc_path = out_dir / f'{prefix}_{D:%Y%m%d}.nc'
    if nc_path.exists() and not force_redownload:
        logger.info('CFS {} (pseudo-ensemble {:%Y-%m-%d}) ja existe — pulando.', file_var, D)
        return [nc_path]

    lead = min(lead_hours, CFS_LEAD_DAYS * 24)
    ds_mean, n = _lagged_ensemble_mean(
        D, file_var, idx_params, level_label, out_rename, lead, out_dir, base_url)
    ds_mean = ds_mean.sel(time=slice(np.datetime64(D), np.datetime64(D + timedelta(days=CFS_LEAD_DAYS))))
    ds_mean.attrs['cfs_pseudo_ensemble_members'] = n
    ds_mean.attrs['cfs_ensemble_day'] = D.strftime('%Y-%m-%d')
    if nc_path.exists():
        nc_path.unlink()
    save_netcdf(ds_mean, nc_path)
    logger.info('CFS {} pseudo-ensemble salvo: {} ({} dias, {} membros)',
                file_var, nc_path.name, ds_mean.sizes.get('time', 0), n)
    return [nc_path]


def ensure_cfs_fcst200_for_period(
    init: datetime, lead_hours: int, hours: Sequence[int] = CFS_CYCLES,
    force_redownload: bool = False,
) -> List[Path]:
    """NetCDF da média do pseudo-ensemble CFS de u/v 200 hPa (dia D=init.date(), ~45 dias)."""
    return _ensure_cfs_level(init, lead_hours, 'wnd200', ('UGRD', 'VGRD'), '200 mb', {},
                             DIR_CFS_FCST200, 'cfs_fcst200', force_redownload)


def ensure_cfs_uv850_for_period(
    init: datetime, lead_hours: int, hours: Sequence[int] = CFS_CYCLES,
    force_redownload: bool = False,
) -> List[Path]:
    """NetCDF da média do pseudo-ensemble CFS de u/v 850 hPa (dia D=init.date(), ~45 dias)."""
    return _ensure_cfs_level(init, lead_hours, 'wnd850', ('UGRD', 'VGRD'), '850 mb', {},
                             DIR_CFS_UV850, 'cfs_uv850', force_redownload)


def ensure_cfs_olr_for_period(
    init: datetime, lead_hours: int, hours: Sequence[int] = CFS_CYCLES,
    force_redownload: bool = False,
) -> List[Path]:
    """NetCDF da média do pseudo-ensemble CFS de OLR (ULWRF no topo; var 'olr', W/m²; ~45 dias)."""
    return _ensure_cfs_level(init, lead_hours, 'ulwtoa', ('ULWRF',), 'top of atmosphere',
                             {'sulwrf': 'olr'}, DIR_CFS_OLR, 'cfs_olr', force_redownload)


def ensure_cfs_pwat_for_period(
    init: datetime, lead_hours: int, hours: Sequence[int] = CFS_CYCLES,
    force_redownload: bool = False,
) -> List[Path]:
    """NetCDF da média do pseudo-ensemble CFS de agua precipitavel (var 'pwat', kg/m2;
    produto dedicado `pwat.<membro>.<data><ciclo>.daily.grb2`; ~45 dias)."""
    return _ensure_cfs_level(init, lead_hours, 'pwat', ('PWAT',),
                             'entire atmosphere (considered as a single layer)', {},
                             DIR_CFS_PWAT, 'cfs_pwat', force_redownload)


# ---------------------------------------------------------------------------
# Variaveis adicionais do CFS usadas pelo s34 (modelo americano de horizonte estendido).
# O CFS tem 200/250/500/850 hPa, hgt (z200/z500), tmp2m e OLR — mas NAO tem z250 nem T850.
# ---------------------------------------------------------------------------
DIR_CFS_HGT500 = DIR_DADOS_BASE / 'CFS_HGT500'
DIR_CFS_HGT700 = DIR_DADOS_BASE / 'CFS_HGT700'
DIR_CFS_UV250 = DIR_DADOS_BASE / 'CFS_UV250'
DIR_CFS_T2M = DIR_DADOS_BASE / 'CFS_T2M'
DIR_CFS_PWAT = DIR_DADOS_BASE / 'CFS_PWAT'


def ensure_cfs_fcst200_uvz_for_period(
    init: datetime, lead_hours: int, hours: Sequence[int] = CFS_CYCLES,
    force_redownload: bool = False,
) -> List[Path]:
    """CFS u/v/HGT @ 200 hPa (s34): junta wnd200 (u/v) + z200 (HGT->hgt) num NetCDF por dia D.

    O s34 le u/v E hgt do MESMO arquivo fcst200 — por isso os dois produtos CFS sao mesclados."""
    D = init.date()
    nc = DIR_CFS_FCST200 / f'cfs_fcst200_uvz_{D:%Y%m%d}.nc'
    if nc.exists() and not force_redownload:
        logger.info('CFS fcst200 (u/v/hgt 200, {:%Y-%m-%d}) ja existe — pulando.', D)
        return [nc]
    uv = _ensure_cfs_level(init, lead_hours, 'wnd200', ('UGRD', 'VGRD'), '200 mb', {},
                           DIR_CFS_FCST200, 'cfs_wnd200', force_redownload)
    hg = _ensure_cfs_level(init, lead_hours, 'z200', ('HGT',), '200 mb', {'gh': 'hgt'},
                           DIR_CFS_FCST200, 'cfs_z200', force_redownload)
    with xr.open_dataset(uv[0]) as a, xr.open_dataset(hg[0]) as b:
        ds = xr.merge([a, b], join='inner', compat='override').load()
    if nc.exists():
        nc.unlink()
    save_netcdf(ds, nc)
    logger.info('CFS fcst200 (u/v/hgt 200) salvo: {} ({} dias)', nc.name, ds.sizes.get('time', 0))
    return [nc]


def ensure_cfs_hgt500_for_period(
    init: datetime, lead_hours: int, hours: Sequence[int] = CFS_CYCLES,
    force_redownload: bool = False,
) -> List[Path]:
    """NetCDF do pseudo-ensemble CFS de altura geopotencial 500 hPa (var 'hgt', m; ~45 dias)."""
    return _ensure_cfs_level(init, lead_hours, 'z500', ('HGT',), '500 mb', {'gh': 'hgt'},
                             DIR_CFS_HGT500, 'cfs_hgt500', force_redownload)


def ensure_cfs_hgt700_for_period(
    init: datetime, lead_hours: int, hours: Sequence[int] = CFS_CYCLES,
    force_redownload: bool = False, base_url: str = CFS_BASE,
) -> List[Path]:
    """NetCDF do pseudo-ensemble CFS de altura geopotencial 700 hPa (var 'hgt', m; ~45 dias).

    `base_url`: troque por `CFS_BASE_AWS` para baixar inits passados (NOMADS so retem ~7 dias)."""
    return _ensure_cfs_level(init, lead_hours, 'z700', ('HGT',), '700 mb', {'gh': 'hgt'},
                             DIR_CFS_HGT700, 'cfs_hgt700', force_redownload, base_url)


def ensure_cfs_uv250_for_period(
    init: datetime, lead_hours: int, hours: Sequence[int] = CFS_CYCLES,
    force_redownload: bool = False,
) -> List[Path]:
    """NetCDF do pseudo-ensemble CFS de u/v 250 hPa (var 'u'/'v', m/s; ~45 dias)."""
    return _ensure_cfs_level(init, lead_hours, 'wnd250', ('UGRD', 'VGRD'), '250 mb', {},
                             DIR_CFS_UV250, 'cfs_uv250', force_redownload)


def ensure_cfs_t2m_for_period(
    init: datetime, lead_hours: int, hours: Sequence[int] = CFS_CYCLES,
    force_redownload: bool = False,
) -> List[Path]:
    """NetCDF do pseudo-ensemble CFS de T2m (TMP @ 2 m; var 't2m', K; ~45 dias)."""
    return _ensure_cfs_level(init, lead_hours, 'tmp2m', ('TMP',), '2 m above ground', {},
                             DIR_CFS_T2M, 'cfs_t2m', force_redownload)


if __name__ == '__main__':
    from datetime import datetime as _dt
    d0 = _dt.now() - timedelta(days=1)
    paths = ensure_cfs_fcst200_for_period(d0, CFS_LEAD_DAYS * 24, force_redownload=True)
    ds = xr.open_dataset(paths[0])
    print('OK:', paths[0], '| vars:', list(ds.data_vars), '| dims:', dict(ds.sizes))
