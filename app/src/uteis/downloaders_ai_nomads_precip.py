# app/src/uteis/downloaders_ai_nomads_precip.py
# -*- coding: utf-8 -*-
"""
Downloaders de CHUVA (precipitacao acumulada) dos modelos de IA do NCEP no NOMADS:
AIGFS (deterministico) e AIGEFS (media do ensemble, `avg` pronta).

FONTE (confirmado na rodada 2026-07-17 00Z): o APCP esta no arquivo **`sfc`**, NAO no `pres` que o
`downloaders_ai_nomads` usa p/ as demais variaveis -- `pres` so tem HGT/TMP/UGRD/VGRD/SPFH/VVEL.
Foi por isso que "a IA nao tinha chuva": o modulo antigo so conhecia o `pres`.

ACUMULACAO (lida no .idx, nao suposta):
  AIGFS  f024 -> '5:...:APCP:surface:18-24 hour acc fcst:'  (balde de 6 h)
                 '6:...:APCP:surface:0-1 day acc fcst:'     (acumulado desde o init)
  AIGEFS f024 -> '5:...:APCP:surface:18-24 hour acc fcst:ens mean'  (SO o balde de 6 h)
Como o AIGEFS nao tem o acumulado, os DOIS usam o BALDE de 6 h e somam os quatro do dia
(06/12/18 do dia + 00 do dia seguinte) -- mesma logica do GFS, um caminho so p/ a familia.

APCP em GRIB2 e kg/m2 == mm -> nao ha conversao de unidade. Saida: um NetCDF por dia UTC completo,
variavel 'precip' (mm), 1 passo de tempo (o dia, rotulado 00Z).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import xarray as xr

from app.common.forecast_download import StepNotAvailable, save_netcdf
from app.shared.logger import get_logger
from app.src.uteis.downloaders_ai_nomads import (
    AI_MAX_FHR,
    DIR_DADOS_BASE,
    NOMADS_BASE,
    _http_get,
)

logger = get_logger(__name__)

DIR_AIGFS_PRECIP = DIR_DADOS_BASE / 'AIGFS_PRECIP'
DIR_AIGEFS_PRECIP = DIR_DADOS_BASE / 'AIGEFS_PRECIP'

# Baldes de 6 h (fim do intervalo, hora UTC) que somam a chuva de UM dia UTC -- igual ao GFS.
_BUCKET_HOURS = (6, 12, 18, 24)


def _aigfs_sfc_urls(init: datetime, fhr: int) -> Tuple[str, str]:
    d, h = init.strftime('%Y%m%d'), init.strftime('%H')
    g = f'{NOMADS_BASE}/aigfs/prod/aigfs.{d}/{h}/model/atmos/grib2/aigfs.t{h}z.sfc.f{fhr:03d}.grib2'
    return g, g + '.idx'


def _aigefs_sfc_urls(init: datetime, fhr: int) -> Tuple[str, str]:
    d, h = init.strftime('%Y%m%d'), init.strftime('%H')
    g = (f'{NOMADS_BASE}/aigefs/prod/aigefs.{d}/{h}/ensstat/products/atmos/grib2/'
         f'aigefs.t{h}z.sfc.avg.f{fhr:03d}.grib2')
    return g, g + '.idx'


def _range_bucket_apcp(idx_text: str, fhr: int) -> Tuple[int, int]:
    """Faixa de bytes do APCP do BALDE de 6 h que termina em `fhr`.

    Precisa do 5o campo do .idx (o descritor, ex.: '18-24 hour acc fcst'): o `_parse_idx` do
    `downloaders_ai_nomads` guarda so (offset, param, nivel), e o AIGFS tem DOIS registros APCP em
    'surface' -- o balde e o acumulado desde o init. Sem o descritor eles sao indistinguiveis e
    pegariamos o errado (o acumulado), inflando a chuva do dia."""
    linhas = [ln.split(':') for ln in idx_text.splitlines()]
    recs = [(int(f[1]), f[3], f[4], f[5] if len(f) > 5 else '') for f in linhas
            if len(f) >= 6 and f[1].isdigit()]
    alvo = f'{fhr - 6}-{fhr} hour acc fcst'
    for i, (start, param, nivel, desc) in enumerate(recs):
        if param == 'APCP' and nivel == 'surface' and desc == alvo:
            end = recs[i + 1][0] - 1 if i + 1 < len(recs) else -1
            return start, end
    raise StepNotAvailable(f'APCP:surface:{alvo} nao encontrado no .idx')


def _fetch_bucket(urls: Callable[[datetime, int], Tuple[str, str]], init: datetime,
                  vt: datetime, tmp: Path, tag: str) -> np.ndarray | None:
    """Balde de 6 h de APCP (mm) terminando em `vt`, em (lat, lon). None se indisponivel."""
    fhr = int((vt - init).total_seconds() // 3600)
    if fhr <= 0 or fhr > AI_MAX_FHR:
        return None
    grib_url, idx_url = urls(init, fhr)
    try:
        start, end = _range_bucket_apcp(_http_get(idx_url).text, fhr)
        rng = f'bytes={start}-' + ('' if end < 0 else str(end))
        raw = _http_get(grib_url, headers={'Range': rng}).content
    except StepNotAvailable:
        logger.warning('  {} APCP f{:03d} ausente (passo nao publicado ou sem o balde) — dia cai',
                       tag, fhr)
        return None
    da = _open_apcp(raw, tmp)
    return da.values.astype('float32')


def _open_apcp(raw: bytes, tmp: Path) -> xr.DataArray:
    """Abre o GRIB2 do APCP (bytes) e devolve (lat, lon) em mm."""
    from app.common.forecast_download import GRIB_NETCDF_LOCK
    tmp.write_bytes(raw)
    with GRIB_NETCDF_LOCK:  # ecCodes nao e thread-safe
        ds = xr.open_dataset(tmp, engine='cfgrib', backend_kwargs={'indexpath': ''}).load()
    ren = {}
    for name in list(ds.dims) + list(ds.coords):
        low = name.lower()
        if low == 'latitude' and 'lat' not in ds.dims:
            ren[name] = 'lat'
        elif low == 'longitude' and 'lon' not in ds.dims:
            ren[name] = 'lon'
    if ren:
        ds = ds.rename(ren)
    var = next((v for v in ds.data_vars if v.lower() in ('tp', 'apcp', 'acpcp', 'unknown')),
               list(ds.data_vars)[0])
    da = ds[var]
    for coord in ('time', 'step', 'valid_time', 'surface', 'level'):
        if coord in da.coords and coord not in da.dims:
            da = da.drop_vars(coord, errors='ignore')
    return da  # kg/m2 == mm


def _ensure_ai_nomads_precip(
    init: datetime, lead_hours: int, *, urls: Callable[[datetime, int], Tuple[str, str]],
    dir_out: Path, prefixo: str, tag: str, force_redownload: bool = False,
) -> List[Path]:
    """Motor comum AIGFS/AIGEFS: soma os quatro baldes de 6 h de cada dia UTC completo."""
    dir_out.mkdir(parents=True, exist_ok=True)
    end = init + timedelta(hours=lead_hours)
    tmp = dir_out / f'{prefixo}_{init.strftime("%Y%m%d%H")}_tmp.grb2'
    out: List[Path] = []
    lat = lon = None

    # Se o init nao e' 00Z o dia do init fica parcial -> comeca no proximo.
    day = init.date() if init.hour == 0 else (init.date() + timedelta(days=1))
    while True:
        vts = [datetime(day.year, day.month, day.day) + timedelta(hours=h) for h in _BUCKET_HOURS]
        if vts[-1] > end:
            break  # dia incompleto (ultimo balde alem do horizonte)

        nc_path = dir_out / f'{prefixo}_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
        if nc_path.exists() and not force_redownload:
            logger.info('{} chuva valido {} (init {}Z) ja existe — pulando.', tag, day, init.hour)
            out.append(nc_path)
            day += timedelta(days=1)
            continue

        baldes = []
        for vt in vts:
            b = _fetch_bucket(urls, init, vt, tmp, tag)
            if b is None:
                baldes = None
                break
            baldes.append(b)
            if lat is None:
                da = _open_apcp(tmp.read_bytes(), tmp)
                lat, lon = da['lat'].values, da['lon'].values
        if baldes is None or lat is None:
            logger.warning('{} chuva {} incompleto (balde ausente) — dia ignorado.', tag, day)
            day += timedelta(days=1)
            continue

        acum = np.sum(baldes, axis=0).astype('float32')
        da = xr.DataArray(
            acum[None, :, :], dims=['time', 'lat', 'lon'],
            coords={'time': [np.datetime64(datetime(day.year, day.month, day.day))],
                    'lat': lat, 'lon': lon}, name='precip')
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


def ensure_aigfs_precip_fcst_for_period(
    init: datetime, lead_hours: int, hours=None, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs de CHUVA ACUMULADA DIARIA (mm) do AIGFS. `hours` ignorado (compat da assinatura)."""
    return _ensure_ai_nomads_precip(
        init, lead_hours, urls=_aigfs_sfc_urls, dir_out=DIR_AIGFS_PRECIP,
        prefixo='aigfs_precip', tag='AIGFS', force_redownload=force_redownload)


def ensure_aigefs_precip_fcst_for_period(
    init: datetime, lead_hours: int, hours=None, force_redownload: bool = False,
) -> List[Path]:
    """NetCDFs de CHUVA ACUMULADA DIARIA (mm) do AIGEFS (media do ensemble, `avg` pronta).

    `hours` ignorado (compat da assinatura)."""
    return _ensure_ai_nomads_precip(
        init, lead_hours, urls=_aigefs_sfc_urls, dir_out=DIR_AIGEFS_PRECIP,
        prefixo='aigefs_precip', tag='AIGEFS', force_redownload=force_redownload)
