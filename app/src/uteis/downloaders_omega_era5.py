# app/src/uteis/downloaders_omega_era5.py
"""Downloader ERA5 velocidade vertical (omega, Pa/s) em múltiplos níveis de pressão.

Arquivo de saída por mês:
  dados/ERA5_OMEGA/era5_omega_{YYYYMM}_h{hours}_g{grid}_p{levels}.nc

A CDS API aceita múltiplos níveis em uma única requisição — sem necessidade de merge.
Variável no arquivo resultante: 'w' (vertical_velocity, Pa/s).
"""

from __future__ import annotations

# Bibliotecas padrão
import calendar
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# HDF5/netCDF4 nao e thread-safe: serializa qualquer abertura de arquivo HDF5
_HDF5_LOCK = threading.Lock()

# Bibliotecas de terceiros
import cdsapi
import xarray as xr

# Módulos locais (com fallback para uso standalone)
try:
    from app.shared.settings_factory import settings  # type: ignore

    DIR_DADOS_BASE = Path(settings.DIR_DADOS)
    _KEY_CDS: Optional[str] = getattr(settings, 'KEY_CDS', None)
except Exception:
    DIR_DADOS_BASE = Path('dados')
    _KEY_CDS = None

# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
LOGGER = logging.getLogger('ERA5_OMEGA')
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
    )
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Constantes
# -----------------------------------------------------------------------------
DIR_ERA5_OMEGA = DIR_DADOS_BASE / 'ERA5_OMEGA'
URL_API = 'https://cds.climate.copernicus.eu/api'
DATASET = 'reanalysis-era5-pressure-levels'
DEFAULT_LEVELS_HPA: List[int] = [100, 250, 300, 500, 850]
DEFAULT_HOURS: Tuple[int, ...] = (0, 6, 12, 18)
MIN_BYTES = 50_000


# -----------------------------------------------------------------------------
# API pública
# -----------------------------------------------------------------------------
def ensure_era5_omega_for_period(
    start: datetime,
    end: datetime,
    levels_hpa: List[int] = DEFAULT_LEVELS_HPA,
    hours_utc: Sequence[int] | None = None,
    grid: str = '1.0/1.0',
    force_redownload: bool = False,
) -> List[Path]:
    """Garante arquivos mensais de omega ERA5 cobrindo [start, end].

    Retorna lista de Paths ordenada por mês. Pula meses cujo arquivo já é válido.
    """
    norm_hours = tuple(sorted(set(int(h) for h in (hours_utc or DEFAULT_HOURS))))
    DIR_ERA5_OMEGA.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    for year, month in _iter_months(start, end):
        end_day = end.day if (year == end.year and month == end.month) else None
        paths.append(
            _download_omega_month(year, month, end_day, levels_hpa, norm_hours, grid, force_redownload)
        )
    return paths


# -----------------------------------------------------------------------------
# Internos
# -----------------------------------------------------------------------------
def _iter_months(start: datetime, end: datetime) -> List[Tuple[int, int]]:
    y, m = start.year, start.month
    out = []
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        m += 1
        if m > 12:
            y += 1
            m = 1
    return out


def _target_path(
    year: int, month: int,
    levels_hpa: List[int],
    norm_hours: Tuple[int, ...],
    grid: str,
) -> Path:
    hours_tag = ''.join(f'{h:02d}' for h in norm_hours)
    grid_tag = grid.replace('/', 'x').replace('.', '')
    levels_tag = ''.join(str(l) for l in sorted(levels_hpa))
    return DIR_ERA5_OMEGA / f'era5_omega_{year:04d}{month:02d}_h{hours_tag}_g{grid_tag}_p{levels_tag}.nc'


def _file_covers_period(
    path: Path,
    year: int, month: int,
    end_day: Optional[int],
    norm_hours: Tuple[int, ...],
) -> bool:
    if not path.exists() or path.stat().st_size < MIN_BYTES:
        return False
    try:
        import pandas as pd
        with _HDF5_LOCK:
            ds = xr.open_dataset(str(path), engine='netcdf4')
        t_name = 'valid_time' if 'valid_time' in ds.coords else 'time'
        if t_name not in ds:
            ds.close()
            return False
        times = ds[t_name].values
        ds.close()

        t_idx = pd.DatetimeIndex(pd.to_datetime(times))
        t_sel = t_idx[t_idx.hour.isin(set(norm_hours))]
        if not len(t_sel):
            return False

        required_day = end_day or calendar.monthrange(year, month)[1]
        last_day = t_sel.max().day
        ok = (t_sel.max().year, t_sel.max().month) == (year, month) and last_day >= required_day
        if ok:
            LOGGER.info('[SKIP] %s (cobre até dia %02d)', path.name, last_day)
        return ok
    except Exception as exc:
        LOGGER.warning('Falha ao validar %s: %s', path.name, exc)
        return False


def _get_cds_client() -> cdsapi.Client:
    url = os.environ.get('CDSAPI_URL', URL_API)
    key = os.environ.get('CDSAPI_KEY') or _KEY_CDS
    if not key:
        raise RuntimeError(
            'Chave CDS não encontrada. Configure KEY_CDS no settings ou CDSAPI_KEY no ambiente.'
        )
    return cdsapi.Client(url=url, key=key, debug=False, progress=True, retry_max=8, sleep_max=120, delete=False)


def _download_omega_month(
    year: int, month: int,
    end_day: Optional[int],
    levels_hpa: List[int],
    norm_hours: Tuple[int, ...],
    grid: str,
    force_redownload: bool,
) -> Path:
    target = _target_path(year, month, levels_hpa, norm_hours, grid)

    if not force_redownload and _file_covers_period(target, year, month, end_day, norm_hours):
        return target

    last_day = end_day or calendar.monthrange(year, month)[1]
    days = [f'{d:02d}' for d in range(1, last_day + 1)]
    times = [f'{h:02d}:00' for h in norm_hours]

    LOGGER.info(
        'Baixando ERA5 omega %s — %04d-%02d dias 01..%02d',
        sorted(levels_hpa), year, month, last_day,
    )

    tmp = target.with_suffix('.nc.part')
    if tmp.exists():
        tmp.unlink()

    client = _get_cds_client()
    client.retrieve(
        DATASET,
        {
            'product_type': 'reanalysis',
            'variable': 'vertical_velocity',
            'pressure_level': [str(l) for l in sorted(levels_hpa)],
            'year': [f'{year:04d}'],
            'month': [f'{month:02d}'],
            'day': days,
            'time': times,
            'data_format': 'netcdf',
            'download_format': 'unarchived',
            'grid': grid.split('/'),
        },
        str(tmp),
    )

    if not tmp.exists() or tmp.stat().st_size < MIN_BYTES:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f'Arquivo inválido após download: {tmp}')

    if target.exists():
        target.unlink()
    tmp.rename(target)

    LOGGER.info('[OK] omega ERA5: %s', target.name)
    return target
