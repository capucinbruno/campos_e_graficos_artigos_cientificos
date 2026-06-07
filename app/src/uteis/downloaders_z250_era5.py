# app/src/uteis/downloaders_z250_era5.py
"""
Downloader ERA5 para geopotencial em 250 hPa (NetCDF, mensal).

Baixa somente o campo de geopotencial ('z') em 250 hPa para uso em
cálculo de altura geopotencial média de período (plot_z250_mean.py).
Uma variável, estrutura mais simples que downloaders_wind250.
"""

from __future__ import annotations

import calendar
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import cdsapi
import pandas as pd
import xarray as xr

SETTINGS_AVAILABLE = False
SETTINGS_KEY_CDS: Optional[str] = None

try:
    from app.shared.settings_factory import settings  # type: ignore

    DIR_DADOS_BASE = Path(settings.DIR_DADOS)
    SETTINGS_KEY_CDS = getattr(settings, 'KEY_CDS', None)
    SETTINGS_AVAILABLE = True
except Exception:
    DIR_DADOS_BASE = Path('dados')

LOGGER = logging.getLogger('ERA5_Z250')
if not LOGGER.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'))
    LOGGER.addHandler(_h)
LOGGER.setLevel(logging.INFO)

DIR_ERA5_Z250 = DIR_DADOS_BASE / 'ERA5_Z250'

URL_API_COPERNICUS = 'https://cds.climate.copernicus.eu/api'
DATASET_ERA5_PRESSURE_LEVELS = 'reanalysis-era5-pressure-levels'
MIN_BYTES_FILE = 30_000
DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)


def _get_cds_client() -> cdsapi.Client:
    url = os.environ.get('CDSAPI_URL', URL_API_COPERNICUS)
    key = os.environ.get('CDSAPI_KEY', SETTINGS_KEY_CDS)
    if not key:
        raise RuntimeError(
            'Chave do CDS não encontrada. Defina CDSAPI_KEY ou settings.KEY_CDS.'
        )
    return cdsapi.Client(url=url, key=key, debug=False, progress=True,
                         retry_max=8, sleep_max=120, delete=False)


def _file_ok(path: Path) -> bool:
    try:
        return path.stat().st_size >= MIN_BYTES_FILE
    except FileNotFoundError:
        return False


def _file_covers_period(path: Path, year: int, month: int, end_day: int) -> bool:
    """Verifica se o arquivo existente cobre o mês/período solicitado."""
    if not _file_ok(path):
        return False
    try:
        ds = xr.open_dataset(path, engine='netcdf4')
        try:
            tv = ds['time'].values if 'time' in ds.coords else ds['valid_time'].values
            t_idx = pd.DatetimeIndex(pd.to_datetime(tv))
            # Verifica se o arquivo tem dados para o dia final pedido
            target = pd.Timestamp(year=year, month=month, day=end_day)
            return t_idx.date.max() >= target.date()
        finally:
            ds.close()
    except Exception:
        return False


def _download_month(
    year: int,
    month: int,
    end_day: Optional[int],
    hours_utc: Sequence[int],
    target_file: Path,
    force: bool = False,
) -> Path:
    last_day = end_day or calendar.monthrange(year, month)[1]

    if not force and _file_covers_period(target_file, year, month, last_day):
        LOGGER.info('ERA5_Z250: reaproveitando %s', target_file.name)
        return target_file

    LOGGER.info('ERA5_Z250: baixando %04d-%02d (dias 1..%02d)', year, month, last_day)
    DIR_ERA5_Z250.mkdir(parents=True, exist_ok=True)

    part_file = target_file.with_suffix('.nc.part')
    if part_file.exists():
        part_file.unlink()

    client = _get_cds_client()
    request = {
        'product_type': ['reanalysis'],
        'variable': ['geopotential'],
        'pressure_level': ['250'],
        'year': [str(year)],
        'month': [f'{month:02d}'],
        'day': [f'{d:02d}' for d in range(1, last_day + 1)],
        'time': [f'{h:02d}:00' for h in sorted(set(hours_utc))],
        'data_format': 'netcdf',
        'download_format': 'unarchived',
    }

    client.retrieve(DATASET_ERA5_PRESSURE_LEVELS, request, str(part_file))

    if not _file_ok(part_file):
        raise RuntimeError(f'Arquivo baixado muito pequeno ou ausente: {part_file}')

    if target_file.exists():
        target_file.unlink()
    part_file.rename(target_file)
    LOGGER.info('ERA5_Z250: salvo → %s (%.1f MB)', target_file.name,
                target_file.stat().st_size / 1e6)
    return target_file


def ensure_era5_z250_for_period(
    start: datetime,
    end: datetime,
    hours_utc: Sequence[int] = DEFAULT_SYNOPTIC_HOURS,
    force_redownload: bool = False,
) -> List[Path]:
    """Baixa geopotencial ERA5 250 hPa para cada mês em [start, end]."""
    periods: list[tuple[int, int, Optional[int]]] = []
    cur = pd.Timestamp(start.year, start.month, 1)
    end_ts = pd.Timestamp(end.year, end.month, end.day)

    while cur <= end_ts:
        y, m = cur.year, cur.month
        last_of_month = calendar.monthrange(y, m)[1]
        end_day = min(last_of_month, end_ts.day if (y == end_ts.year and m == end_ts.month) else last_of_month)
        periods.append((y, m, end_day))
        if m == 12:
            cur = pd.Timestamp(y + 1, 1, 1)
        else:
            cur = pd.Timestamp(y, m + 1, 1)

    files: List[Path] = []
    for y, m, ed in periods:
        fname = DIR_ERA5_Z250 / f'era5_z250_{y}{m:02d}.nc'
        fp = _download_month(y, m, ed, hours_utc, fname, force=force_redownload)
        files.append(fp)

    return files
