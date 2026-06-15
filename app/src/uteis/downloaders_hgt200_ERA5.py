# app/src/uteis/downloaders_hgt200_ERA5.py
# -*- coding: utf-8 -*-
"""
Downloader ERA5 de altura geopotencial em 200 hPa (global, NetCDF).

Espelha o padrao do downloader de 250 hPa (downloaders_hgt250_ERA5), porem
compacto: baixa geopotential (z, m2/s2) do CDS mensalmente e ja converte para
altura geopotencial (hgt, m) = z / g, salvando um NetCDF com a variavel 'hgt'.

Isso garante compatibilidade com `_compute_period_mean_streaming_hgt` do
plot_rossby_waf, que promedia diretamente a variavel sem converter unidades —
logo ERA5 (hgt em m) e GDAS (hgt em m) ficam na mesma unidade.
"""

from __future__ import annotations

import calendar
import os
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import cdsapi
import xarray as xr

from app.shared.logger import get_logger
from app.shared.settings_factory import settings

logger = get_logger(__name__)

G = 9.80665  # m s-2
URL_API_COPERNICUS = 'https://cds.climate.copernicus.eu/api'
DATASET_ERA5_PRESSURE_LEVELS = 'reanalysis-era5-pressure-levels'
DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)
MIN_BYTES_FILE = 50_000

try:
    DIR_DADOS_BASE = Path(settings.DIR_DADOS)
except Exception:
    DIR_DADOS_BASE = Path('dados')

DIR_ERA5_HGT200 = DIR_DADOS_BASE / 'ERA5_HGT200'


def _get_cds_client() -> cdsapi.Client:
    url = os.environ.get('CDSAPI_URL', URL_API_COPERNICUS)
    key = os.environ.get('CDSAPI_KEY', getattr(settings, 'KEY_CDS', None))
    if not key:
        raise RuntimeError(
            'Chave do CDS nao encontrada. Defina CDSAPI_KEY no ambiente ou KEY_CDS no settings.'
        )
    return cdsapi.Client(url=url, key=key, debug=False, progress=True, retry_max=8, sleep_max=120)


def _iter_year_month(start: datetime, end: datetime) -> List[Tuple[int, int]]:
    months = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def _normalize_hours(hours_utc: Sequence[int] | None) -> Tuple[int, ...]:
    hours = DEFAULT_SYNOPTIC_HOURS if hours_utc is None else hours_utc
    uniq = sorted(set(int(h) for h in hours))
    if not uniq:
        raise ValueError('Lista de horas UTC vazia.')
    return tuple(uniq)


def _convert_z_to_hgt(raw_path: Path, out_path: Path) -> Path:
    """Converte geopotential (z, m2/s2) para altura geopotencial (hgt, m) e salva."""
    with xr.open_dataset(raw_path, engine='netcdf4') as ds:
        zname = next((v for v in ('z', 'geopotential') if v in ds.data_vars), None)
        if zname is None:
            # ja e altura? mantem
            zname = list(ds.data_vars)[0]
            da = ds[zname]
        else:
            da = ds[zname] / G
        da = da.rename('hgt')
        da.attrs['units'] = 'm'
        da.attrs['long_name'] = 'geopotential height at 200 hPa'
        out = da.to_dataset(name='hgt')
        out.load()
    if out_path.exists():
        out_path.unlink()
    out.to_netcdf(out_path, engine='netcdf4')
    return out_path


def _download_era5_hgt200_month(
    year: int, month: int, end_day: int | None = None,
    hours_utc: Sequence[int] | None = None, force_redownload: bool = False,
    grid: str = '1.0/1.0',
) -> Path:
    hours = _normalize_hours(hours_utc)
    DIR_ERA5_HGT200.mkdir(parents=True, exist_ok=True)
    hours_tag = ''.join(f'{h:02d}' for h in hours)
    out_path = DIR_ERA5_HGT200 / f'era5_hgt200_{year:04d}{month:02d}_h{hours_tag}.nc'

    if out_path.exists() and out_path.stat().st_size >= MIN_BYTES_FILE and not force_redownload:
        logger.info('ERA5 HGT200 {}-{:02d} ja existe, pulando download.', year, month)
        return out_path

    last_day = calendar.monthrange(year, month)[1]
    if end_day is not None:
        last_day = min(last_day, end_day)
    days = [f'{d:02d}' for d in range(1, last_day + 1)]

    request = {
        'product_type': ['reanalysis'],
        'variable': ['geopotential'],
        'pressure_level': ['200'],
        'year': f'{year:04d}',
        'month': f'{month:02d}',
        'day': days,
        'time': [f'{h:02d}:00' for h in hours],
        'data_format': 'netcdf',
        'grid': grid,
    }

    raw_path = DIR_ERA5_HGT200 / f'era5_z200_{year:04d}{month:02d}_h{hours_tag}.nc'
    logger.info('Baixando ERA5 geopotential 200 hPa {}-{:02d} (dias 01..{:02d})', year, month, last_day)
    client = _get_cds_client()
    tmp = raw_path.with_suffix('.part')
    client.retrieve(DATASET_ERA5_PRESSURE_LEVELS, request, str(tmp))
    tmp.rename(raw_path)

    _convert_z_to_hgt(raw_path, out_path)
    if raw_path.exists():
        raw_path.unlink()
    logger.info('ERA5 HGT200 {}-{:02d} salvo: {}', year, month, out_path.name)
    return out_path


def ensure_era5_hgt200_for_period(
    start: datetime, end: datetime,
    hours_utc: Sequence[int] | None = None, force_redownload: bool = False,
) -> List[Path]:
    """Garante NetCDFs mensais de hgt 200 hPa (em metros) do ERA5 para [start, end]."""
    files: List[Path] = []
    for y, m in _iter_year_month(start, end):
        end_day = end.day if (y == end.year and m == end.month) else None
        files.append(_download_era5_hgt200_month(y, m, end_day, hours_utc, force_redownload))
    return files
