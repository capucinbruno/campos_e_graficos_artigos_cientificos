# app/src/uteis/downloaders_era5_pwat.py
# -*- coding: utf-8 -*-
"""
Downloader para ERA5 single-levels (NetCDF) com:
- total_column_water_vapour (agua precipitavel, tcwv -> renomeada 'pwat'), dominio global

Espelha o downloaders_tmp850_ERA5.py (mesma robustez: arquivo .part, validação de
tamanho, interpretação de erros do CDS sobre disponibilidade parcial), trocando o
dataset de pressure-levels por single-levels (tcwv nao tem nivel de pressao).
"""

from __future__ import annotations

# Bibliotecas padrão
import calendar
import logging
import os
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Bibliotecas de terceiros
import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

# -----------------------------------------------------------------------------
# Integração opcional com settings
# -----------------------------------------------------------------------------
try:
    from app.shared.settings_factory import settings  # type: ignore

    DIR_DADOS_BASE = Path(settings.DIR_DADOS)
except Exception:
    DIR_DADOS_BASE = Path('dados')

# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
LOGGER = logging.getLogger('ERA5_PWAT_GLOBAL')
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Diretórios
# -----------------------------------------------------------------------------
DIR_ERA5_PWAT = DIR_DADOS_BASE / 'ERA5_PWAT_GLOBAL' / 'pwat_hourly_global'

# -----------------------------------------------------------------------------
# Credenciais CDS
# -----------------------------------------------------------------------------
URL_API_COPERNICUS = 'https://cds.climate.copernicus.eu/api'

try:
    KEY_COPERNICUS_UTILIZADA = settings.KEY_CDS  # type: ignore
except Exception:
    KEY_COPERNICUS_UTILIZADA = os.environ.get('CDSAPI_KEY', '')

MIN_BYTES = 50_000

DATASET_ERA5_SINGLE_LEVELS = 'reanalysis-era5-single-levels'
VARIABLES_PWAT = ['total_column_water_vapour']

DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)

CDS_UI_CONSERVATIVE_MODE = True


# -----------------------------------------------------------------------------
# Utilitários básicos
# -----------------------------------------------------------------------------
def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _file_ok(path: Path, min_bytes: int = MIN_BYTES) -> bool:
    try:
        return path.stat().st_size >= min_bytes
    except FileNotFoundError:
        return False


def _safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        LOGGER.warning('Não consegui remover arquivo: %s', path)


def _get_cds_client() -> cdsapi.Client:
    url = os.environ.get('CDSAPI_URL', URL_API_COPERNICUS)
    key = os.environ.get('CDSAPI_KEY', KEY_COPERNICUS_UTILIZADA)
    return cdsapi.Client(
        url=url,
        key=key,
        debug=False,
        progress=True,
        retry_max=8,
        sleep_max=120,
        delete=False,
    )


def _day_list_for_month(year: int, month: int) -> List[str]:
    last_day = calendar.monthrange(year, month)[1]
    return [f'{d:02d}' for d in range(1, last_day + 1)]


def _normalize_hours(hours_utc: Sequence[int] | None) -> Tuple[int, ...]:
    if hours_utc is None:
        hours_utc = DEFAULT_SYNOPTIC_HOURS
    uniq = sorted(set(int(h) for h in hours_utc))
    if not uniq:
        raise ValueError('Lista de horas UTC vazia.')
    for h in uniq:
        if h < 0 or h > 23:
            raise ValueError(f'Hora UTC inválida: {h}')
    return tuple(uniq)


def _time_list_from_hours(hours_utc: Sequence[int]) -> List[str]:
    return [f'{int(h):02d}:00' for h in hours_utc]


def _is_current_utc_month(year: int, month: int) -> bool:
    today_utc = datetime.now(timezone.utc).date()
    return (today_utc.year == year) and (today_utc.month == month)


def _apply_conservative_cutoff(
    *,
    last_complete_day: Optional[int],
    year: int,
    month: int,
    requested_end_day: Optional[int],
) -> Optional[int]:
    if last_complete_day is None:
        return None
    if not CDS_UI_CONSERVATIVE_MODE:
        return last_complete_day
    if requested_end_day is None or not _is_current_utc_month(year, month):
        return last_complete_day
    if requested_end_day <= last_complete_day:
        return last_complete_day
    adjusted = max(1, last_complete_day - 1)
    if adjusted != last_complete_day:
        LOGGER.warning(
            '[CDS UI mode] Ajustando último dia disponível de %02d para %02d.',
            last_complete_day, adjusted,
        )
    return adjusted


# -----------------------------------------------------------------------------
# Leitura robusta de tempo
# -----------------------------------------------------------------------------
def _ensure_time_coord(obj):
    if hasattr(obj, 'dims') and 'time' not in obj.dims and 'valid_time' in obj.dims:
        obj = obj.rename({'valid_time': 'time'})
    elif hasattr(obj, 'coords') and 'time' not in obj.coords and 'valid_time' in obj.coords:
        obj = obj.rename({'valid_time': 'time'})
    if 'time' not in obj.coords:
        raise KeyError("Nem 'time' nem 'valid_time' encontrados.")
    return obj


def _choose_main_var(ds: xr.Dataset) -> xr.DataArray:
    for vn in ('tcwv', 'total_column_water_vapour', 'pwat'):
        if vn in ds.data_vars:
            return ds[vn]
    for vn in ds.data_vars:
        if np.issubdtype(ds[vn].dtype, np.number):
            return ds[vn]
    raise KeyError(f'Nenhuma variável de agua precipitavel encontrada. Disponíveis: {list(ds.data_vars)}')


def _drop_or_collapse_aux_dims(da: xr.DataArray) -> xr.DataArray:
    rename_dims = {}
    for d in da.dims:
        dl = d.lower()
        if dl == 'expver' and d != 'expver':
            rename_dims[d] = 'expver'
        elif dl == 'number' and d != 'number':
            rename_dims[d] = 'number'
    if rename_dims:
        da = da.rename(rename_dims)
    if 'expver' in da.dims:
        da = da.bfill('expver').ffill('expver').isel(expver=0, drop=True)
    if 'number' in da.dims:
        da = da.isel(number=0, drop=True)
    for c in ('expver', 'number'):
        if c in da.coords and c not in da.dims:
            try:
                da = da.drop_vars(c)
            except Exception:
                pass
    return da


def _extract_time_index(path: Path) -> pd.DatetimeIndex:
    ds = xr.open_dataset(path, engine='netcdf4')
    try:
        ds = _ensure_time_coord(ds)
        da = _choose_main_var(ds)
        da = _ensure_time_coord(da)
        da = _drop_or_collapse_aux_dims(da)
        return pd.DatetimeIndex(pd.to_datetime(da['time'].values))
    finally:
        ds.close()


def _last_complete_synoptic_date(
    path: Path,
    required_hours: Sequence[int],
) -> Optional[date]:
    if not _file_ok(path):
        return None
    try:
        required_set = set(int(h) for h in required_hours)
        t_idx = _extract_time_index(path)
        if len(t_idx) == 0:
            return None
        t_sel = t_idx[t_idx.hour.isin(required_set)]
        if len(t_sel) == 0:
            return None

        hours_by_day: Dict[Any, List[int]] = {}
        for day, grp in pd.Series(t_sel.hour, index=t_sel.floor('D')).groupby(level=0):
            hours_by_day[pd.Timestamp(day)] = sorted(set(int(h) for h in grp.values if int(h) in required_set))

        complete_days = [d for d, hs in hours_by_day.items() if set(hs) == required_set]
        return max(complete_days).date() if complete_days else None
    except Exception as e:
        LOGGER.warning('Falha ao validar %s: %s', path, e)
        return None


def _existing_file_satisfies(
    path: Path,
    *,
    year: int,
    month: int,
    end_day: Optional[int],
    required_hours: Sequence[int],
) -> bool:
    if not _file_ok(path):
        return False
    required_day = end_day if end_day is not None else calendar.monthrange(year, month)[1]
    last = _last_complete_synoptic_date(path, required_hours)
    if last is None or (last.year, last.month) != (year, month):
        return False
    ok = last.day >= required_day
    if ok:
        LOGGER.info('[SKIP] %s cobre até dia %02d. Último completo: %s', path.name, required_day, last)
    else:
        LOGGER.info('Arquivo %s desatualizado (precisa dia %02d, tem até %02d). Rebaixando.', path.name, required_day, last.day)
    return ok


# -----------------------------------------------------------------------------
# Interpretação de erros CDS
# -----------------------------------------------------------------------------
def _is_cds_no_data_error(error_text: str) -> bool:
    t = error_text.lower()
    return (
        'none of the data you have requested is available yet' in t
        or 'multiadapternodataerror' in t
        or 'no matching data' in t
        or 'requested data is not available' in t
        or 'latest date available' in t
    )


def _extract_latest_dt_from_cds_error(error_text: str) -> Optional[datetime]:
    m = re.search(
        r'latest date available.*?:\s*(\d{4}-\d{2}-\d{2})\s+(\d{2}):(\d{2})',
        error_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    try:
        base = datetime.strptime(m.group(1), '%Y-%m-%d')
        return base.replace(hour=int(m.group(2)), minute=int(m.group(3)))
    except ValueError:
        return None


# -----------------------------------------------------------------------------
# Download principal
# -----------------------------------------------------------------------------
def download_era5_pwat_hourly(
    year: int,
    month: int,
    end_day: Optional[int] = None,
    hours_utc: Sequence[int] | None = None,
    force_redownload: bool = False,
) -> Path:
    """Baixa ERA5 agua precipitavel (tcwv) global como NetCDF mensal."""
    norm_hours = _normalize_hours(hours_utc)
    time_list = _time_list_from_hours(norm_hours)

    _ensure_dir(DIR_ERA5_PWAT)

    hours_tag = ''.join(f'{h:02d}' for h in norm_hours)
    fname = f'era5_pwat_global_hourly_{year:04d}{month:02d}_h{hours_tag}.nc'
    target = DIR_ERA5_PWAT / fname

    if not force_redownload and _existing_file_satisfies(
        target, year=year, month=month, end_day=end_day, required_hours=norm_hours
    ):
        return target

    month_last = calendar.monthrange(year, month)[1]
    if end_day is not None and (end_day < 1 or end_day > month_last):
        raise ValueError(f'end_day inválido ({end_day}) para {month:02d}/{year:04d}.')

    days = (
        [f'{d:02d}' for d in range(1, end_day + 1)]
        if end_day is not None
        else _day_list_for_month(year, month)
    )

    client = _get_cds_client()
    request = {
        'product_type': ['reanalysis'],
        'variable': VARIABLES_PWAT,
        'year': [f'{year:04d}'],
        'month': [f'{month:02d}'],
        'day': days,
        'time': time_list,
        'data_format': 'netcdf',
        'download_format': 'unarchived',
    }

    LOGGER.info(
        'Baixando ERA5 pwat (tcwv) global %04d-%02d, dias 01..%s, horas=%s → %s',
        year, month, days[-1], ','.join(time_list), fname,
    )

    tmp_path = target.with_suffix(target.suffix + '.part')
    _safe_unlink(tmp_path)

    try:
        client.retrieve(DATASET_ERA5_SINGLE_LEVELS, request, str(tmp_path))
    except Exception as e:
        msg = str(e)
        if _is_cds_no_data_error(msg):
            latest_dt = _extract_latest_dt_from_cds_error(msg)
            if latest_dt is None:
                raise
            max_req_hour = max(int(h) for h in norm_hours)
            last_day = latest_dt.day if latest_dt.hour >= max_req_hour else latest_dt.day - 1
            last_day = _apply_conservative_cutoff(
                last_complete_day=last_day, year=year, month=month, requested_end_day=end_day
            )
            if last_day is None or last_day <= 0:
                raise RuntimeError(
                    f'ERA5 pwat: sem dia completo disponível para {month:02d}/{year:04d}.'
                ) from e
            if end_day is not None:
                raise RuntimeError(
                    f'ERA5 pwat: solicitado até {end_day:02d}/{month:02d}/{year:04d}, '
                    f'disponível até {last_day:02d}/{month:02d}/{year:04d}.'
                ) from e
            trimmed = [d for d in days if int(d) <= last_day]
            if not trimmed:
                raise RuntimeError(
                    f'ERA5 pwat: nenhum dia disponível para {month:02d}/{year:04d}.'
                ) from e
            LOGGER.warning('CDS período incompleto. Rebaixando com dias 01..%s.', trimmed[-1])
            _safe_unlink(tmp_path)
            request['day'] = trimmed
            client.retrieve(DATASET_ERA5_SINGLE_LEVELS, request, str(tmp_path))
        else:
            raise

    if not _file_ok(tmp_path):
        _safe_unlink(tmp_path)
        raise RuntimeError(f'ERA5 pwat: arquivo temporário inválido após download: {tmp_path}')

    _safe_unlink(target)
    tmp_path.rename(target)

    LOGGER.info('[OK] ERA5 pwat %04d-%02d salvo: %s', year, month, target)
    return target


# -----------------------------------------------------------------------------
# Helper por período
# -----------------------------------------------------------------------------
def _iter_year_month_range(start: datetime, end: datetime) -> List[tuple[int, int]]:
    if end < start:
        raise ValueError("Período inválido: 'end' é anterior a 'start'.")
    y, m = start.year, start.month
    out: List[tuple[int, int]] = []
    while (y < end.year) or (y == end.year and m <= end.month):
        out.append((y, m))
        m = m + 1 if m < 12 else 1
        y = y if m > 1 else y + 1
    return out


def ensure_era5_pwat_for_period(
    start: datetime,
    end: datetime,
    hours_utc: Sequence[int] | None = None,
    force_redownload: bool = False,
) -> List[Path]:
    """Garante arquivos mensais de agua precipitavel ERA5 (global) para o período [start, end]."""
    files: List[Path] = []
    for y, m in _iter_year_month_range(start, end):
        end_day = end.day if (y == end.year and m == end.month) else None
        arq = download_era5_pwat_hourly(y, m, end_day=end_day, hours_utc=hours_utc, force_redownload=force_redownload)
        files.append(arq)
    return files


if __name__ == '__main__':
    path = download_era5_pwat_hourly(year=2026, month=2, end_day=20)
    print(f'Arquivo salvo em: {path}')
