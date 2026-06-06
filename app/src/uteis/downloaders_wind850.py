# app/src/uteis/downloaders_wind850.py
# -*- coding: utf-8 -*-
"""
Downloader para ERA5 pressure-levels (NetCDF) com:
- u_component_of_wind
- v_component_of_wind
- pressure_level = 850 hPa

Objetivo:
- Baixar dados em NetCDF com robustez (arquivo .part, validação de tamanho)
- Reaproveitar arquivo existente quando possível
- Interpretar erros do CDS sobre disponibilidade parcial
- Validar o CONTEÚDO REAL do arquivo baixado
- Seguir um critério CONSERVADOR alinhado à UI do CDS para meses correntes
- Permitir download por mês e por período (start/end)
- Horas sinóticas padrão: 00, 06, 12, 18 UTC
"""

from __future__ import annotations

# Bibliotecas padrão
import calendar
import logging
import os
import re
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# HDF5/netCDF4 nao e thread-safe: serializa qualquer abertura de arquivo HDF5
_HDF5_LOCK = threading.Lock()

# Bibliotecas de terceiros
import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

# -----------------------------------------------------------------------------
# Integração opcional com projeto (settings)
# -----------------------------------------------------------------------------
SETTINGS_AVAILABLE = False
SETTINGS_KEY_CDS: Optional[str] = None

try:
    # Módulos locais
    from app.shared.settings_factory import settings  # type: ignore

    DIR_DADOS_BASE = Path(settings.DIR_DADOS)
    SETTINGS_KEY_CDS = getattr(settings, 'KEY_CDS', None)
    SETTINGS_AVAILABLE = True
except Exception:
    DIR_DADOS_BASE = Path('dados')

# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
LOGGER = logging.getLogger('ERA5_UV850')
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Diretórios
# -----------------------------------------------------------------------------
DIR_ERA5_BASE = DIR_DADOS_BASE / 'ERA5_UV850'
DIR_ERA5_UV850_HOURLY = DIR_ERA5_BASE / 'uv850_hourly'

# -----------------------------------------------------------------------------
# Credenciais CDS
# -----------------------------------------------------------------------------
URL_API_COPERNICUS = 'https://cds.climate.copernicus.eu/api'
KEY_COPERNICUS_UTILIZADA = SETTINGS_KEY_CDS

MIN_BYTES_FILE = 50_000

# Dataset e variáveis alvo
DATASET_ERA5_PRESSURE_LEVELS = 'reanalysis-era5-pressure-levels'
VARIABLES_UV850 = [
    'u_component_of_wind',
    'v_component_of_wind',
]
PRESSURE_LEVEL_850 = '850'

# Horas sinóticas padrão do projeto (00/06/12/18)
DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)

CDS_UI_CONSERVATIVE_MODE = True


# -----------------------------------------------------------------------------
# Utilitários básicos
# -----------------------------------------------------------------------------
def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _file_ok(path: Path, min_bytes: int) -> bool:
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

    if not key:
        raise RuntimeError(
            'Chave do CDS não encontrada. Defina CDSAPI_KEY no ambiente '
            'ou disponibilize settings.KEY_CDS.'
        )

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


def _hours_label(hours_utc: Sequence[int]) -> str:
    return ','.join(f'{int(h):02d}Z' for h in hours_utc)


def _is_current_utc_month(year: int, month: int) -> bool:
    today_utc = datetime.now(timezone.utc).date()
    return (today_utc.year == year) and (today_utc.month == month)


def _apply_cds_ui_conservative_cutoff_if_needed(
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

    if requested_end_day is None:
        return last_complete_day

    if not _is_current_utc_month(year, month):
        return last_complete_day

    if requested_end_day <= last_complete_day:
        return last_complete_day

    adjusted = max(1, last_complete_day - 1)
    if adjusted != last_complete_day:
        LOGGER.warning(
            '[CDS UI mode] Ajustando último dia disponível de %02d para %02d '
            '(mês corrente, modo conservador alinhado à UI do CDS).',
            last_complete_day,
            adjusted,
        )
    return adjusted


# -----------------------------------------------------------------------------
# Leitura robusta de tempo no arquivo
# -----------------------------------------------------------------------------
def _open_dataset(path_file: Path) -> xr.Dataset:
    suffix = path_file.suffix.lower()
    # _HDF5_LOCK: libhdf5 nao e thread-safe; serializa abertura entre threads paralelas
    with _HDF5_LOCK:
        if suffix in {'.nc', '.nc4'}:
            return xr.open_dataset(path_file, engine='netcdf4')
        return xr.open_dataset(
            path_file,
            engine='cfgrib',
            backend_kwargs={'indexpath': ''},
        )


def _ensure_time_coord(obj):
    if hasattr(obj, 'dims') and 'time' not in obj.dims and 'valid_time' in obj.dims:
        obj = obj.rename({'valid_time': 'time'})
    elif hasattr(obj, 'coords') and 'time' not in obj.coords and 'valid_time' in obj.coords:
        obj = obj.rename({'valid_time': 'time'})

    if 'time' not in obj.coords:
        raise KeyError("Nem 'time' nem 'valid_time' encontrados no arquivo.")

    if not np.issubdtype(obj['time'].dtype, np.datetime64):
        if isinstance(obj, xr.DataArray):
            obj = xr.decode_cf(obj.to_dataset(name='_tmp'))['_tmp']
        else:
            obj = xr.decode_cf(obj)

    return obj


def _choose_main_var(ds: xr.Dataset) -> xr.DataArray:
    preferred = ['u', 'v', 'u_component_of_wind', 'v_component_of_wind']

    for vn in preferred:
        if vn in ds.data_vars:
            return ds[vn]

    for vn in ds.data_vars:
        if np.issubdtype(ds[vn].dtype, np.number):
            return ds[vn]

    raise KeyError('Nenhuma variável numérica encontrada para validação temporal.')


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


def _extract_time_index_from_file(path_file: Path) -> pd.DatetimeIndex:
    ds = _open_dataset(path_file)
    try:
        ds = _ensure_time_coord(ds)
        da = _choose_main_var(ds)
        da = _ensure_time_coord(da)
        da = _drop_or_collapse_aux_dims(da)

        t_idx = pd.DatetimeIndex(pd.to_datetime(da['time'].values))
        return t_idx
    finally:
        ds.close()


def _summarize_synoptic_coverage_in_file(
    path_file: Path,
    required_hours_utc: Sequence[int],
) -> Dict[str, Any]:
    if not _file_ok(path_file, MIN_BYTES_FILE):
        return {
            'n_total_timestamps': 0, 'min_time': None, 'max_time': None,
            'n_days_with_records': 0, 'n_complete_days': 0,
            'last_any_date': None, 'last_complete_date_raw': None,
            'hours_count_by_day': pd.Series(dtype=int),
            'hours_present_by_day': {}, 'hours_present_last_any_day': [],
        }

    required_hours = tuple(sorted(set(int(h) for h in required_hours_utc)))
    required_set = set(required_hours)

    t_idx = _extract_time_index_from_file(path_file)
    if len(t_idx) == 0:
        return {
            'n_total_timestamps': 0, 'min_time': None, 'max_time': None,
            'n_days_with_records': 0, 'n_complete_days': 0,
            'last_any_date': None, 'last_complete_date_raw': None,
            'hours_count_by_day': pd.Series(dtype=int),
            'hours_present_by_day': {}, 'hours_present_last_any_day': [],
        }

    t_sel = t_idx[t_idx.hour.isin(required_set)]

    if len(t_sel) == 0:
        return {
            'n_total_timestamps': int(len(t_idx)), 'min_time': t_idx.min(), 'max_time': t_idx.max(),
            'n_days_with_records': 0, 'n_complete_days': 0,
            'last_any_date': None, 'last_complete_date_raw': None,
            'hours_count_by_day': pd.Series(dtype=int),
            'hours_present_by_day': {}, 'hours_present_last_any_day': [],
        }

    t_sel = pd.DatetimeIndex(sorted(pd.unique(t_sel)))

    hours_present_by_day: Dict[pd.Timestamp, List[int]] = {}
    for day, group in pd.Series(t_sel.hour, index=t_sel.floor('D')).groupby(level=0):
        unique_hours = sorted(set(int(h) for h in group.values if int(h) in required_set))
        hours_present_by_day[pd.Timestamp(day)] = unique_hours

    hours_count_by_day = pd.Series({
        day: len(hours) for day, hours in hours_present_by_day.items()
    }).sort_index()

    complete_days = [day for day, hours in hours_present_by_day.items() if set(hours) == required_set]

    last_any_ts = t_sel.max()
    last_any_date = last_any_ts.date() if pd.notna(last_any_ts) else None
    last_complete_date_raw = max(complete_days).date() if complete_days else None

    last_any_day_key = pd.Timestamp(last_any_ts.floor('D'))
    hours_present_last_any_day = hours_present_by_day.get(last_any_day_key, [])

    return {
        'n_total_timestamps': int(len(t_sel)),
        'min_time': t_sel.min(), 'max_time': t_sel.max(),
        'n_days_with_records': int(len(hours_count_by_day)),
        'n_complete_days': int(len(complete_days)),
        'last_any_date': last_any_date,
        'last_complete_date_raw': last_complete_date_raw,
        'hours_count_by_day': hours_count_by_day,
        'hours_present_by_day': hours_present_by_day,
        'hours_present_last_any_day': hours_present_last_any_day,
    }


def _last_complete_synoptic_date_in_file(
    path_file: Path,
    required_hours_utc: Sequence[int],
) -> Optional[date]:
    try:
        summary = _summarize_synoptic_coverage_in_file(path_file, required_hours_utc)
        return summary['last_complete_date_raw']
    except Exception as e:
        LOGGER.warning('Falha ao validar arquivo %s: %s', path_file, e)
        return None


def _target_last_day_for_request(year: int, month: int, end_day: Optional[int]) -> int:
    if end_day is None:
        return calendar.monthrange(year, month)[1]
    return end_day


def _existing_file_satisfies_period(
    target_file: Path,
    *,
    year: int,
    month: int,
    end_day: Optional[int],
    required_hours_utc: Sequence[int],
) -> bool:
    if not _file_ok(target_file, MIN_BYTES_FILE):
        return False

    required_day = _target_last_day_for_request(year, month, end_day)
    last_complete = _last_complete_synoptic_date_in_file(
        target_file, required_hours_utc=required_hours_utc,
    )

    if last_complete is None:
        return False

    if (last_complete.year, last_complete.month) != (year, month):
        return False

    ok = last_complete.day >= required_day
    if ok:
        LOGGER.info(
            '[SKIP] %s cobre período solicitado (até dia %02d). Último dia completo no arquivo: %02d/%02d/%04d',
            target_file.name, required_day, last_complete.day, last_complete.month, last_complete.year,
        )
    else:
        LOGGER.info(
            'Arquivo %s está desatualizado para pedido atual (precisa dia %02d, tem até %02d). Rebaixando.',
            target_file.name, required_day, last_complete.day,
        )
    return ok


# -----------------------------------------------------------------------------
# Interpretação do erro CDS
# -----------------------------------------------------------------------------
def _extract_latest_dt_from_cds_error(error_text: str) -> Optional[datetime]:
    m = re.search(
        r'latest date available.*?:\s*(\d{4}-\d{2}-\d{2})\s+(\d{2}):(\d{2})',
        error_text, flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    ymd = m.group(1)
    hh, mm = int(m.group(2)), int(m.group(3))
    try:
        return datetime.strptime(ymd, '%Y-%m-%d').replace(hour=hh, minute=mm)
    except ValueError:
        return None


def _last_complete_day_from_latest_dt(latest_dt: datetime, required_hours_utc: Sequence[int]) -> int:
    max_required_hour = max(int(h) for h in required_hours_utc)
    if latest_dt.hour >= max_required_hour:
        return latest_dt.day
    return latest_dt.day - 1


def _format_incomplete_period_message(
    year: int, month: int, requested_end_day: Optional[int], last_complete_day: int,
) -> str:
    if requested_end_day is None:
        return (
            f'[ERA5_UV850] O período solicitado ainda não está completo no CDS. '
            f'Para {month:02d}/{year:04d}, há dado completo disponível apenas até '
            f'{last_complete_day:02d}/{month:02d}/{year:04d}.'
        )
    return (
        f'[ERA5_UV850] Você solicitou até {requested_end_day:02d}/{month:02d}/{year:04d}, '
        f'mas há dado completo disponível apenas até '
        f'{last_complete_day:02d}/{month:02d}/{year:04d}. '
        f'Ajuste a data final para {last_complete_day:02d}/{month:02d}/{year:04d}.'
    )


def _is_cds_no_data_error(error_text: str) -> bool:
    t = error_text.lower()
    return (
        'none of the data you have requested is available yet' in t
        or 'multiadapternodataerror' in t
        or 'no matching data' in t
        or 'requested data is not available' in t
        or 'latest date available' in t
    )


# -----------------------------------------------------------------------------
# Validação pós-download
# -----------------------------------------------------------------------------
def _validate_downloaded_file_coverage(
    path_file: Path,
    *,
    year: int,
    month: int,
    requested_end_day: Optional[int],
    required_hours_utc: Sequence[int],
) -> None:
    summary = _summarize_synoptic_coverage_in_file(path_file, required_hours_utc)

    last_complete_raw: Optional[date] = summary['last_complete_date_raw']
    last_any: Optional[date] = summary['last_any_date']

    last_complete_day = last_complete_raw.day if last_complete_raw else None
    last_complete_day_adj = _apply_cds_ui_conservative_cutoff_if_needed(
        last_complete_day=last_complete_day, year=year, month=month,
        requested_end_day=requested_end_day,
    )
    last_complete_adj = (
        date(year, month, last_complete_day_adj) if last_complete_day_adj is not None else None
    )

    LOGGER.info(
        '[Validação pós-download | CONTEÚDO REAL] timestamps=%s | %s -> %s | '
        'dias_completos=%s | último_dia_completo_considerado(UI)=%s',
        summary['n_total_timestamps'], summary['min_time'], summary['max_time'],
        summary['n_complete_days'], last_complete_adj,
    )

    if requested_end_day is None:
        return

    if last_complete_adj is None:
        raise RuntimeError(
            f'[ERA5_UV850] Você solicitou até {requested_end_day:02d}/{month:02d}/{year:04d}, '
            'mas o arquivo baixado não contém nenhum dia completo com as horas sinóticas requeridas '
            f'({_hours_label(required_hours_utc)}).'
        )

    if last_complete_adj.day >= requested_end_day:
        return

    hours_last_any = summary.get('hours_present_last_any_day', []) or []
    hours_last_any_str = ', '.join(f'{h:02d}Z' for h in hours_last_any) if hours_last_any else 'nenhuma'

    if last_any is None:
        extra = 'O arquivo não possui registros de tempo válidos.'
    elif last_complete_raw is not None and last_any == last_complete_raw:
        extra = (
            f'Observação: o arquivo contém registros até {last_any.strftime("%d/%m/%Y")} '
            f'com horas {hours_last_any_str}, porém no modo conservador esse dia não é considerado liberado.'
        )
    else:
        extra = (
            f'O arquivo possui registros até {last_any.strftime("%d/%m/%Y")}, '
            f'mas no último dia há apenas horas {hours_last_any_str} '
            f'(requeridas: {_hours_label(required_hours_utc)}).'
        )

    raise RuntimeError(
        f'[ERA5_UV850] Você solicitou até {requested_end_day:02d}/{month:02d}/{year:04d}, '
        f'mas há dado completo disponível apenas até {last_complete_adj.day:02d}/{month:02d}/{year:04d}. '
        f'{extra} Ajuste a data final para {last_complete_adj.day:02d}/{month:02d}/{year:04d}.'
    )


# -----------------------------------------------------------------------------
# Download seguro
# -----------------------------------------------------------------------------
def _safe_download_with_cds_interpretation(
    client: cdsapi.Client,
    dataset: str,
    request: dict,
    target_file: Path,
    *,
    year: int,
    month: int,
    requested_end_day: Optional[int],
    required_hours_utc: Sequence[int],
    allow_auto_trim_when_end_day_none: bool = True,
) -> Tuple[Path, List[str]]:
    days_requested = list(request.get('day', []))
    if not days_requested:
        raise RuntimeError("[ERA5_UV850] Request sem campo 'day'.")

    tmp_path = target_file.with_suffix(target_file.suffix + '.part')
    _safe_unlink(tmp_path)

    def _do_retrieve(req: dict) -> None:
        client.retrieve(dataset, req, str(tmp_path))

    try:
        _do_retrieve(request)
    except Exception as e:
        msg = str(e)

        if _is_cds_no_data_error(msg):
            latest_dt = _extract_latest_dt_from_cds_error(msg)
            if latest_dt is None:
                raise

            last_complete = _last_complete_day_from_latest_dt(latest_dt, required_hours_utc=required_hours_utc)
            last_complete = _apply_cds_ui_conservative_cutoff_if_needed(
                last_complete_day=last_complete, year=year, month=month,
                requested_end_day=requested_end_day,
            )

            if last_complete is None or last_complete <= 0:
                raise RuntimeError(
                    f'[ERA5_UV850] O CDS indicou disponibilidade parcial para {month:02d}/{year:04d}, '
                    'mas não há dia completo disponível para as horas solicitadas.'
                ) from e

            if requested_end_day is not None:
                raise RuntimeError(
                    _format_incomplete_period_message(
                        year=year, month=month,
                        requested_end_day=requested_end_day,
                        last_complete_day=last_complete,
                    )
                ) from e

            if allow_auto_trim_when_end_day_none:
                trimmed_days = [d for d in days_requested if int(d) <= last_complete]
                if not trimmed_days:
                    raise RuntimeError(
                        _format_incomplete_period_message(
                            year=year, month=month,
                            requested_end_day=None,
                            last_complete_day=last_complete,
                        )
                    ) from e

                LOGGER.warning(
                    'CDS retornou período incompleto. Ajustando download para dias 01..%s e tentando novamente.',
                    trimmed_days[-1],
                )
                retry_req = dict(request)
                retry_req['day'] = trimmed_days
                _safe_unlink(tmp_path)
                _do_retrieve(retry_req)
                days_requested = trimmed_days
            else:
                raise RuntimeError(
                    _format_incomplete_period_message(
                        year=year, month=month,
                        requested_end_day=None,
                        last_complete_day=last_complete,
                    )
                ) from e
        else:
            raise

    if not _file_ok(tmp_path, MIN_BYTES_FILE):
        _safe_unlink(tmp_path)
        raise RuntimeError(f'[ERA5_UV850] Arquivo temporário inválido após download: {tmp_path}')

    _safe_unlink(target_file)
    tmp_path.rename(target_file)

    _validate_downloaded_file_coverage(
        target_file, year=year, month=month,
        requested_end_day=requested_end_day,
        required_hours_utc=required_hours_utc,
    )

    return target_file, days_requested


# -----------------------------------------------------------------------------
# Downloader principal: U/V em 850 hPa (NetCDF)
# -----------------------------------------------------------------------------
def _download_single_variable(
    variable: str,
    year: int,
    month: int,
    end_day: Optional[int],
    norm_hours: Tuple[int, ...],
    grid: str,
    force_redownload: bool,
) -> Path:
    time_list = _time_list_from_hours(norm_hours)
    _ensure_dir(DIR_ERA5_UV850_HOURLY)

    hours_tag = ''.join(f'{h:02d}' for h in norm_hours)
    grid_tag = grid.replace('/', 'x').replace('.', '')
    var_short = variable.split('_', maxsplit=1)[0]  # 'u' ou 'v'
    fname_nc = f'era5_{var_short}850_hourly_{year:04d}{month:02d}_h{hours_tag}_g{grid_tag}.nc'
    target_file = DIR_ERA5_UV850_HOURLY / fname_nc

    if not force_redownload:
        if _existing_file_satisfies_period(
            target_file, year=year, month=month,
            end_day=end_day, required_hours_utc=norm_hours,
        ):
            return target_file

    month_last = calendar.monthrange(year, month)[1]
    if end_day is not None and (end_day < 1 or end_day > month_last):
        raise ValueError(f'[ERA5_UV850] end_day inválido ({end_day}) para {month:02d}/{year:04d}.')

    client = _get_cds_client()

    days = (
        [f'{d:02d}' for d in range(1, end_day + 1)]
        if end_day is not None
        else _day_list_for_month(year, month)
    )

    request = {
        'product_type': ['reanalysis'],
        'variable': [variable],
        'year': [f'{year:04d}'],
        'month': [f'{month:02d}'],
        'day': days,
        'time': time_list,
        'pressure_level': [PRESSURE_LEVEL_850],
        'data_format': 'netcdf',
        'download_format': 'unarchived',
        'grid': grid,
    }

    LOGGER.info(
        'Baixando %s -> %s (%s 850 hPa %04d-%02d, dias 01..%s, horas=%s)',
        DATASET_ERA5_PRESSURE_LEVELS, target_file.name,
        var_short, year, month, days[-1], ','.join(time_list),
    )

    final_path, used_days = _safe_download_with_cds_interpretation(
        client=client, dataset=DATASET_ERA5_PRESSURE_LEVELS,
        request=request, target_file=target_file,
        year=year, month=month, requested_end_day=end_day,
        required_hours_utc=norm_hours, allow_auto_trim_when_end_day_none=True,
    )

    LOGGER.info(
        '[OK] %s850 %04d-%02d salvo (%s). Dias: 01..%s | Horas=%s',
        var_short, year, month, final_path, used_days[-1], ','.join(time_list),
    )
    return final_path


def download_era5_uv850_hourly(
    year: int,
    month: int,
    end_day: Optional[int] = None,
    hours_utc: Sequence[int] | None = None,
    grid: str = '1.0/1.0',
    force_redownload: bool = False,
) -> Path:
    """Baixa ERA5 pressure-levels global em NetCDF com u/v em 850 hPa."""
    norm_hours = _normalize_hours(hours_utc)
    _ensure_dir(DIR_ERA5_UV850_HOURLY)

    hours_tag = ''.join(f'{h:02d}' for h in norm_hours)
    grid_tag = grid.replace('/', 'x').replace('.', '')
    fname_merged = f'era5_uv850_hourly_{year:04d}{month:02d}_h{hours_tag}_g{grid_tag}.nc'
    target_merged = DIR_ERA5_UV850_HOURLY / fname_merged

    if not force_redownload:
        if _existing_file_satisfies_period(
            target_merged, year=year, month=month,
            end_day=end_day, required_hours_utc=norm_hours,
        ):
            return target_merged

    LOGGER.info('Iniciando download de u e v 850 hPa (%04d-%02d, grid=%s)', year, month, grid)

    var_paths: Dict[str, Path] = {}
    for var in VARIABLES_UV850:
        var_paths[var] = _download_single_variable(
            variable=var, year=year, month=month,
            end_day=end_day, norm_hours=norm_hours,
            grid=grid, force_redownload=force_redownload,
        )
        LOGGER.info('[OK] %s baixado: %s', var.split('_')[0], var_paths[var].name)

    LOGGER.info('Mesclando u e v em %s ...', fname_merged)
    # _HDF5_LOCK serializa abertura de arquivos HDF5 entre threads (libhdf5 nao e thread-safe)
    with _HDF5_LOCK:
        datasets = []
        for var in VARIABLES_UV850:
            ds = xr.open_dataset(var_paths[var], engine='netcdf4')
            datasets.append(ds)

        # Remove dims auxiliares do ERA5 (expver, number) antes de mesclar para evitar
        # atributos HDF5 inconsistentes que causam falhas em leituras posteriores.
        cleaned = []
        for ds in datasets:
            if 'expver' in ds.dims:
                ds = ds.isel(expver=0, drop=True)
            if 'number' in ds.dims:
                ds = ds.isel(number=0, drop=True)
            for c in ('expver', 'number'):
                if c in ds.coords and c not in ds.dims:
                    try:
                        ds = ds.drop_vars(c)
                    except Exception:
                        pass
            cleaned.append(ds)

        merged = xr.merge(cleaned, compat='override')
        for ds in datasets:
            ds.close()

        _safe_unlink(target_merged)
        merged.to_netcdf(target_merged)
        merged.close()

    LOGGER.info('[OK] uv850 %04d-%02d mesclado -> %s', year, month, target_merged.name)
    return target_merged


# -----------------------------------------------------------------------------
# Helpers por período
# -----------------------------------------------------------------------------
def _iter_year_month_range(start: datetime, end: datetime) -> List[tuple[int, int]]:
    if end < start:
        raise ValueError("Período inválido: 'end' é anterior a 'start'.")

    y, m = start.year, start.month
    end_y, end_m = end.year, end.month
    out: List[tuple[int, int]] = []

    while (y < end_y) or (y == end_y and m <= end_m):
        out.append((y, m))
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1

    return out


def ensure_era5_uv850_for_period(
    start: datetime,
    end: datetime,
    hours_utc: Sequence[int] | None = None,
    grid: str = '1.0/1.0',
    force_redownload: bool = False,
    max_parallel: int = 2,
) -> List[Path]:
    """Garante arquivos mensais do período [start, end], com download paralelo de até max_parallel meses."""
    months = _iter_year_month_range(start, end)

    if len(months) == 1:
        y, m = months[0]
        end_day = end.day if (y == end.year and m == end.month) else None
        return [
            download_era5_uv850_hourly(
                y, m, end_day=end_day, hours_utc=hours_utc,
                grid=grid, force_redownload=force_redownload,
            )
        ]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    LOGGER.info('Download paralelo de %d meses (max_parallel=%d)', len(months), max_parallel)

    tasks = []
    for y, m in months:
        end_day = end.day if (y == end.year and m == end.month) else None
        tasks.append((y, m, end_day))

    results: dict[tuple[int, int], Path] = {}

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_ym = {}
        for y, m, ed in tasks:
            fut = executor.submit(
                download_era5_uv850_hourly,
                y, m, end_day=ed, hours_utc=hours_utc,
                grid=grid, force_redownload=force_redownload,
            )
            future_to_ym[fut] = (y, m)

        for fut in as_completed(future_to_ym):
            ym = future_to_ym[fut]
            results[ym] = fut.result()
            LOGGER.info('[OK] Download concluido: %04d-%02d', ym[0], ym[1])

    return [results[(y, m)] for y, m in months]
