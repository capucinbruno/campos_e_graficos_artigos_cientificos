# app/src/uteis/downloaders_gfs_fcst200.py
# -*- coding: utf-8 -*-
"""
Downloader GFS (previsao) de u/v/altura geopotencial em 200 hPa via NOMADS Grib Filter.

Modo PROGNOSTICO do projeto: baixa os passos de previsao do GFS 0.25° (ciclos
00/06/12/18 UTC) e monta **um NetCDF por dia valido** contendo as horas sinoticas
(00/06/12/18) com as variaveis `u`, `v`, `hgt` em 200 hPa — a MESMA estrutura dos
arquivos GDAS, para o pipeline do s34 (media diaria, anomalia, Ks/RWS/WAF) rodar
sem alteracao.

GFS entrega HGT (altura geopotencial) ja em metros — nao precisa converter.

Fonte: https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl
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

logger = get_logger(__name__)

NOMADS_FILTER_URL = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl'
DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)
GFS_MAX_FHR = 384  # GFS 0.25° vai ate 384 h

try:
    DIR_DADOS_BASE = Path(settings.DIR_DADOS)
except Exception:
    DIR_DADOS_BASE = Path('dados')

DIR_GFS_FCST200 = DIR_DADOS_BASE / 'GFS_FCST200'


def _build_filter_params(init: datetime, fhr: int) -> dict:
    hh = init.hour
    return {
        'file': f'gfs.t{hh:02d}z.pgrb2.0p25.f{fhr:03d}',
        'lev_200_mb': 'on',
        'var_UGRD': 'on',
        'var_VGRD': 'on',
        'var_HGT': 'on',
        'dir': f'/gfs.{init.strftime("%Y%m%d")}/{hh:02d}/atmos',
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
                    f'Falha ao baixar GFS 200 hPa apos 3 tentativas: {target.name}'
                ) from exc


def _open_gfs_grb2(path: Path) -> xr.Dataset:
    """Abre o GRIB2 do GFS (u/v/gh em 200 hPa) e normaliza para u/v/hgt em (lat, lon)."""
    ds = xr.open_dataset(
        path, engine='cfgrib', backend_kwargs={'indexpath': ''},
        filter_by_keys={'typeOfLevel': 'isobaricInhPa'},
    )
    rename = {}
    for name in list(ds.dims) + list(ds.coords):
        low = name.lower()
        if low == 'latitude' and 'lat' not in ds.dims:
            rename[name] = 'lat'
        elif low == 'longitude' and 'lon' not in ds.dims:
            rename[name] = 'lon'
    if rename:
        ds = ds.rename(rename)
    if 'gh' in ds.data_vars:
        ds = ds.rename({'gh': 'hgt'})

    for coord in ('time', 'step', 'valid_time', 'isobaricInhPa', 'level', 'heightAboveGround'):
        if coord in ds.coords and coord not in ds.dims:
            ds = ds.drop_vars(coord, errors='ignore')

    keep = [v for v in ('u', 'v', 'hgt') if v in ds.data_vars]
    if not keep:
        raise KeyError(f'u/v/hgt nao encontrados no GFS GRIB. Disponiveis: {list(ds.data_vars)}')
    return ds[keep]


def _steps_for_day(
    init: datetime, day: date, hours: Sequence[int], lead_hours: int,
) -> List[Tuple[int, datetime]]:
    """Passos de previsao (fhr, valid_time) das horas sinoticas de `day` a partir de `init`."""
    steps = []
    for h in hours:
        vt = datetime(day.year, day.month, day.day, h)
        fhr = int((vt - init).total_seconds() // 3600)
        if 0 <= fhr <= min(lead_hours, GFS_MAX_FHR):
            steps.append((fhr, vt))
    return steps


def _download_gfs_day(
    init: datetime, day: date, steps: List[Tuple[int, datetime]], force_redownload: bool,
) -> Path:
    fname = f'gfs_fcst200_{init.strftime("%Y%m%d%H")}_valid{day.strftime("%Y%m%d")}.nc'
    nc_path = DIR_GFS_FCST200 / fname
    if nc_path.exists() and not force_redownload:
        logger.info('GFS 200 hPa valido {} (init {}Z) ja existe — pulando.', day, init.hour)
        return nc_path

    DIR_GFS_FCST200.mkdir(parents=True, exist_ok=True)
    parts = []
    for fhr, vt in steps:
        grb = DIR_GFS_FCST200 / f'gfs_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
        if not grb.exists() or force_redownload:
            logger.info('  Baixando GFS init {}Z f{:03d} (valido {:%Y-%m-%d %HZ})', init.hour, fhr, vt)
            _download_grb2(_build_filter_params(init, fhr), grb)
        ds = _open_gfs_grb2(grb).expand_dims(time=[np.datetime64(vt)])
        parts.append(ds.load())

    ds_day = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    if nc_path.exists():
        nc_path.unlink()
    ds_day.to_netcdf(nc_path, engine='netcdf4')

    for fhr, _ in steps:
        grb = DIR_GFS_FCST200 / f'gfs_{init.strftime("%Y%m%d%H")}_f{fhr:03d}.grb2'
        if grb.exists():
            grb.unlink()

    logger.info('GFS 200 hPa valido {} salvo: {}', day, nc_path.name)
    return nc_path


def ensure_gfs_fcst200_for_period(
    init: datetime,
    lead_hours: int,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS,
    force_redownload: bool = False,
) -> List[Path]:
    """Garante NetCDFs diarios (u/v/hgt 200 hPa) do GFS para a janela [init, init+lead_hours].

    Um arquivo por dia valido, com as horas sinoticas disponiveis. Retorna a lista de paths.
    """
    files: List[Path] = []
    end = init + timedelta(hours=lead_hours)
    day = init.date()
    while day <= end.date():
        steps = _steps_for_day(init, day, hours, lead_hours)
        if steps:
            files.append(_download_gfs_day(init, day, steps, force_redownload))
        day += timedelta(days=1)

    logger.info(
        'GFS FCST200: {} arquivos diarios | init {:%Y-%m-%d %H}Z + {}h',
        len(files), init, lead_hours,
    )
    return files
