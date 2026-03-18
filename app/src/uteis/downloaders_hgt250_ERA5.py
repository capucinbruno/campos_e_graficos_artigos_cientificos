# app/src/uteis/downloaders_era5_altura_geopotencial_250_global_grib.py
# -*- coding: utf-8 -*-
"""
Downloader para ERA5 pressure-levels (GRIB) com:
- geopotential (z) em 250 hPa, domínio global

Objetivo:
- Baixar dados em GRIB com robustez (arquivo .part, validação de tamanho)
- Reaproveitar arquivo existente quando possível
- Interpretar erros do CDS sobre disponibilidade parcial
- Validar o CONTEÚDO REAL do arquivo baixado
- Seguir um critério CONSERVADOR alinhado à UI do CDS para meses correntes:
  se a UI ainda não liberou o "dia D", mesmo que a API/cache entregue 4 sinóticas de D,
  considerar como último dia disponível o dia D-1.
- Permitir download por mês e por período (start/end)
- Horas sinóticas padrão: 00, 06, 12, 18 UTC

Observações importantes:
- O CDS entrega a variável "geopotential" (z) em m²/s².
- A altura geopotencial (m) é obtida por:
      altura_geopotencial = z / 9.80665
- Para download global, o request NÃO envia o campo "area".
- Para leitura de GRIB no Python, este script usa xarray + cfgrib.
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
# Integração opcional com seu projeto (settings)
# -----------------------------------------------------------------------------
try:
    # Módulos locais
    from app.shared.settings_factory import settings  # type: ignore

    DIR_DADOS_BASE = Path(settings.DIR_DADOS)
except Exception:
    DIR_DADOS_BASE = Path('dados')

# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
LOGGER = logging.getLogger('ERA5_ALTURA_GEOPOTENCIAL_250_GLOBAL_GRIB')
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Diretórios
# -----------------------------------------------------------------------------
DIR_ERA5_BASE = DIR_DADOS_BASE / 'ERA5_ALTURA_GEOPOTENCIAL_250_GLOBAL'
DIR_ERA5_Z250 = DIR_ERA5_BASE / 'z250_hourly_global'

# -----------------------------------------------------------------------------
# Credenciais CDS
# -----------------------------------------------------------------------------
URL_API_COPERNICUS = 'https://cds.climate.copernicus.eu/api'

try:
    KEY_COPERNICUS_UTILIZADA = settings.KEY_CDS  # type: ignore
except Exception:
    KEY_COPERNICUS_UTILIZADA = os.environ.get('CDSAPI_KEY', '')

MIN_BYTES_GRIB = 50_000

# Dataset e variável alvo
DATASET_ERA5_PRESSURE_LEVELS = 'reanalysis-era5-pressure-levels'
VARIABLES_Z250 = ['geopotential']
PRESSURE_LEVEL_Z250 = '250'

# Horas sinóticas padrão
DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)

# Modo conservador alinhado à UI do CDS
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
# Leitura robusta de tempo no GRIB
# -----------------------------------------------------------------------------
def _open_grib_dataset(path_grib: Path) -> xr.Dataset:
    """
    Abre GRIB com cfgrib.

    indexpath="" evita criação de arquivos .idx persistentes ao lado do grib.
    """
    return xr.open_dataset(
        path_grib,
        engine='cfgrib',
        backend_kwargs={
            'indexpath': '',
        },
    )


def _ensure_time_coord(obj):
    if hasattr(obj, 'dims') and 'time' not in obj.dims and 'valid_time' in obj.dims:
        obj = obj.rename({'valid_time': 'time'})
    elif hasattr(obj, 'coords') and 'time' not in obj.coords and 'valid_time' in obj.coords:
        obj = obj.rename({'valid_time': 'time'})

    if 'time' not in obj.coords:
        raise KeyError("Nem 'time' nem 'valid_time' encontrados no arquivo.")
    return obj


def _choose_main_var(ds: xr.Dataset) -> xr.DataArray:
    preferred = [
        'z',
        'geopotential',
    ]
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


def _extract_time_index_from_file(path_grib: Path) -> pd.DatetimeIndex:
    """
    Lê o GRIB e devolve índice temporal robusto.
    """
    ds = _open_grib_dataset(path_grib)
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
    path_grib: Path,
    required_hours_utc: Sequence[int],
) -> Dict[str, Any]:
    if not _file_ok(path_grib, MIN_BYTES_GRIB):
        return {
            'n_total_timestamps': 0,
            'min_time': None,
            'max_time': None,
            'n_days_with_records': 0,
            'n_complete_days': 0,
            'last_any_date': None,
            'last_complete_date_raw': None,
            'hours_count_by_day': pd.Series(dtype=int),
            'hours_present_by_day': {},
            'hours_present_last_any_day': [],
        }

    required_hours = tuple(sorted(set(int(h) for h in required_hours_utc)))
    required_set = set(required_hours)

    t_idx = _extract_time_index_from_file(path_grib)
    if len(t_idx) == 0:
        return {
            'n_total_timestamps': 0,
            'min_time': None,
            'max_time': None,
            'n_days_with_records': 0,
            'n_complete_days': 0,
            'last_any_date': None,
            'last_complete_date_raw': None,
            'hours_count_by_day': pd.Series(dtype=int),
            'hours_present_by_day': {},
            'hours_present_last_any_day': [],
        }

    t_sel = t_idx[t_idx.hour.isin(required_set)]

    if len(t_sel) == 0:
        return {
            'n_total_timestamps': int(len(t_idx)),
            'min_time': t_idx.min(),
            'max_time': t_idx.max(),
            'n_days_with_records': 0,
            'n_complete_days': 0,
            'last_any_date': None,
            'last_complete_date_raw': None,
            'hours_count_by_day': pd.Series(dtype=int),
            'hours_present_by_day': {},
            'hours_present_last_any_day': [],
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

    if complete_days:
        last_complete_date_raw = max(complete_days).date()
    else:
        last_complete_date_raw = None

    last_any_day_key = pd.Timestamp(last_any_ts.floor('D'))
    hours_present_last_any_day = hours_present_by_day.get(last_any_day_key, [])

    return {
        'n_total_timestamps': int(len(t_sel)),
        'min_time': t_sel.min(),
        'max_time': t_sel.max(),
        'n_days_with_records': int(len(hours_count_by_day)),
        'n_complete_days': int(len(complete_days)),
        'last_any_date': last_any_date,
        'last_complete_date_raw': last_complete_date_raw,
        'hours_count_by_day': hours_count_by_day,
        'hours_present_by_day': hours_present_by_day,
        'hours_present_last_any_day': hours_present_last_any_day,
    }


def _last_complete_synoptic_date_in_file(
    path_grib: Path,
    required_hours_utc: Sequence[int],
) -> Optional[date]:
    try:
        summary = _summarize_synoptic_coverage_in_file(path_grib, required_hours_utc)
        return summary['last_complete_date_raw']
    except Exception as e:
        LOGGER.warning('Falha ao validar arquivo %s: %s', path_grib, e)
        return None


def _target_last_day_for_request(year: int, month: int, end_day: Optional[int]) -> int:
    if end_day is None:
        return calendar.monthrange(year, month)[1]
    return end_day


def _existing_file_satisfies_period(
    target_grib: Path,
    *,
    year: int,
    month: int,
    end_day: Optional[int],
    required_hours_utc: Sequence[int],
) -> bool:
    if not _file_ok(target_grib, MIN_BYTES_GRIB):
        return False

    required_day = _target_last_day_for_request(year, month, end_day)
    last_complete = _last_complete_synoptic_date_in_file(
        target_grib,
        required_hours_utc=required_hours_utc,
    )

    if last_complete is None:
        return False

    if (last_complete.year, last_complete.month) != (year, month):
        return False

    ok = last_complete.day >= required_day
    if ok:
        LOGGER.info(
            '[SKIP] %s cobre período solicitado (até dia %02d). Último dia completo no arquivo: %02d/%02d/%04d',
            target_grib.name,
            required_day,
            last_complete.day,
            last_complete.month,
            last_complete.year,
        )
    else:
        LOGGER.info(
            'Arquivo %s está desatualizado para pedido atual (precisa dia %02d, tem até %02d). Rebaixando.',
            target_grib.name,
            required_day,
            last_complete.day,
        )
    return ok


# -----------------------------------------------------------------------------
# Interpretação do erro CDS
# -----------------------------------------------------------------------------
def _extract_latest_dt_from_cds_error(error_text: str) -> Optional[datetime]:
    m = re.search(
        r'latest date available.*?:\s*(\d{4}-\d{2}-\d{2})\s+(\d{2}):(\d{2})',
        error_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None

    ymd = m.group(1)
    hh = int(m.group(2))
    mm = int(m.group(3))
    try:
        base = datetime.strptime(ymd, '%Y-%m-%d')
        return base.replace(hour=hh, minute=mm)
    except ValueError:
        return None


def _last_complete_day_from_latest_dt(
    latest_dt: datetime,
    required_hours_utc: Sequence[int],
) -> int:
    max_required_hour = max(int(h) for h in required_hours_utc)
    if latest_dt.hour >= max_required_hour:
        return latest_dt.day
    return latest_dt.day - 1


def _format_incomplete_period_message(
    year: int,
    month: int,
    requested_end_day: Optional[int],
    last_complete_day: int,
) -> str:
    if requested_end_day is None:
        return (
            f'[ERA5_ALTURA_GEOPOTENCIAL_250_GLOBAL_GRIB] O período solicitado ainda não está completo no CDS. '
            f'Para {month:02d}/{year:04d}, há dado completo disponível apenas até '
            f'{last_complete_day:02d}/{month:02d}/{year:04d}.'
        )

    return (
        f'[ERA5_ALTURA_GEOPOTENCIAL_250_GLOBAL_GRIB] Você solicitou até {requested_end_day:02d}/{month:02d}/{year:04d}, '
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
    path_grib: Path,
    *,
    year: int,
    month: int,
    requested_end_day: Optional[int],
    required_hours_utc: Sequence[int],
) -> None:
    summary = _summarize_synoptic_coverage_in_file(path_grib, required_hours_utc)

    last_complete_raw: Optional[date] = summary['last_complete_date_raw']
    last_any: Optional[date] = summary['last_any_date']

    last_complete_day = last_complete_raw.day if last_complete_raw else None
    last_complete_day_adj = _apply_cds_ui_conservative_cutoff_if_needed(
        last_complete_day=last_complete_day,
        year=year,
        month=month,
        requested_end_day=requested_end_day,
    )
    last_complete_adj = (
        date(year, month, last_complete_day_adj) if last_complete_day_adj is not None else None
    )

    LOGGER.info(
        '[Validação pós-download | CONTEÚDO REAL] Cobertura no arquivo: '
        'timestamps=%s | %s -> %s | dias_com_registros=%s | dias_completos=%s | '
        'último_dia_completo_bruto=%s | último_dia_completo_considerado(UI)=%s',
        summary['n_total_timestamps'],
        summary['min_time'],
        summary['max_time'],
        summary['n_days_with_records'],
        summary['n_complete_days'],
        last_complete_raw,
        last_complete_adj,
    )

    if requested_end_day is None:
        return

    if last_complete_adj is None:
        raise RuntimeError(
            f'[ERA5_ALTURA_GEOPOTENCIAL_250_GLOBAL_GRIB] Você solicitou até {requested_end_day:02d}/{month:02d}/{year:04d}, '
            'mas o arquivo baixado não contém nenhum dia completo com as horas sinóticas requeridas '
            f'({_hours_label(required_hours_utc)}).'
        )

    if last_complete_adj.day >= requested_end_day:
        return

    hours_last_any = summary.get('hours_present_last_any_day', []) or []
    hours_last_any_str = ', '.join(f'{h:02d}Z' for h in hours_last_any) if hours_last_any else 'nenhuma'

    if last_any is None:
        extra = 'O arquivo não possui registros de tempo válidos.'
    elif last_any == last_complete_raw:
        extra = (
            f'Observação: o arquivo/API contém registros até {last_any.strftime("%d/%m/%Y")} '
            f'com horas {hours_last_any_str}, porém no modo conservador alinhado à UI do CDS '
            f'esse dia ainda não é considerado liberado.'
        )
    else:
        extra = (
            f'O arquivo possui registros até {last_any.strftime("%d/%m/%Y")}, '
            f'mas no último dia presente há apenas horas {hours_last_any_str} '
            f'(requeridas: {_hours_label(required_hours_utc)}).'
        )

    raise RuntimeError(
        f'[ERA5_ALTURA_GEOPOTENCIAL_250_GLOBAL_GRIB] Você solicitou até {requested_end_day:02d}/{month:02d}/{year:04d}, '
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
    target_grib: Path,
    *,
    year: int,
    month: int,
    requested_end_day: Optional[int],
    required_hours_utc: Sequence[int],
    allow_auto_trim_when_end_day_none: bool = True,
) -> Tuple[Path, List[str]]:
    days_requested = list(request.get('day', []))
    if not days_requested:
        raise RuntimeError("[ERA5_ALTURA_GEOPOTENCIAL_250_GLOBAL_GRIB] Request sem campo 'day'.")

    tmp_path = target_grib.with_suffix(target_grib.suffix + '.part')
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

            last_complete = _last_complete_day_from_latest_dt(
                latest_dt,
                required_hours_utc=required_hours_utc,
            )
            last_complete = _apply_cds_ui_conservative_cutoff_if_needed(
                last_complete_day=last_complete,
                year=year,
                month=month,
                requested_end_day=requested_end_day,
            )

            if last_complete is None or last_complete <= 0:
                raise RuntimeError(
                    f'[ERA5_ALTURA_GEOPOTENCIAL_250_GLOBAL_GRIB] O CDS indicou disponibilidade parcial para {month:02d}/{year:04d}, '
                    'mas não há dia completo disponível para as horas solicitadas.'
                ) from e

            if requested_end_day is not None:
                raise RuntimeError(
                    _format_incomplete_period_message(
                        year=year,
                        month=month,
                        requested_end_day=requested_end_day,
                        last_complete_day=last_complete,
                    )
                ) from e

            if allow_auto_trim_when_end_day_none:
                trimmed_days = [d for d in days_requested if int(d) <= last_complete]
                if not trimmed_days:
                    raise RuntimeError(
                        _format_incomplete_period_message(
                            year=year,
                            month=month,
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
                        year=year,
                        month=month,
                        requested_end_day=None,
                        last_complete_day=last_complete,
                    )
                ) from e
        else:
            raise

    if not _file_ok(tmp_path, MIN_BYTES_GRIB):
        _safe_unlink(tmp_path)
        raise RuntimeError(
            f'[ERA5_ALTURA_GEOPOTENCIAL_250_GLOBAL_GRIB] Arquivo temporário inválido após download: {tmp_path}'
        )

    _safe_unlink(target_grib)
    tmp_path.rename(target_grib)

    _validate_downloaded_file_coverage(
        target_grib,
        year=year,
        month=month,
        requested_end_day=requested_end_day,
        required_hours_utc=required_hours_utc,
    )

    return target_grib, days_requested


# -----------------------------------------------------------------------------
# Downloader principal: geopotential 250 hPa global em GRIB
# -----------------------------------------------------------------------------
def download_era5_altura_geopotencial_250_global_hourly_grib(
    year: int,
    month: int,
    end_day: Optional[int] = None,
    hours_utc: Sequence[int] | None = None,
    force_redownload: bool = False,
) -> Path:
    """
    Baixa ERA5 pressure-levels global em GRIB com geopotential (z) em 250 hPa.

    Observação:
    - O arquivo salvo contém geopotential (z).
    - A conversão para altura geopotencial deve ser feita depois.
    """
    norm_hours = _normalize_hours(hours_utc)
    time_list = _time_list_from_hours(norm_hours)

    _ensure_dir(DIR_ERA5_Z250)

    hours_tag = ''.join(f'{h:02d}' for h in norm_hours)
    fname_grib = f'era5_z250_global_hourly_{year:04d}{month:02d}_h{hours_tag}.grib'
    target_grib = DIR_ERA5_Z250 / fname_grib

    if not force_redownload:
        if _existing_file_satisfies_period(
            target_grib,
            year=year,
            month=month,
            end_day=end_day,
            required_hours_utc=norm_hours,
        ):
            return target_grib

    month_last = calendar.monthrange(year, month)[1]
    if end_day is not None and (end_day < 1 or end_day > month_last):
        raise ValueError(
            f'[ERA5_ALTURA_GEOPOTENCIAL_250_GLOBAL_GRIB] end_day inválido ({end_day}) para {month:02d}/{year:04d}.'
        )

    client = _get_cds_client()

    days = (
        [f'{d:02d}' for d in range(1, end_day + 1)]
        if end_day is not None
        else _day_list_for_month(year, month)
    )

    request = {
        'product_type': ['reanalysis'],
        'variable': VARIABLES_Z250,
        'pressure_level': [PRESSURE_LEVEL_Z250],
        'year': [f'{year:04d}'],
        'month': [f'{month:02d}'],
        'day': days,
        'time': time_list,
        'data_format': 'grib',
        'download_format': 'unarchived',
    }

    LOGGER.info(
        'Baixando %s -> %s (z250 global %04d-%02d, dias 01..%s, horas=%s, formato=GRIB)',
        DATASET_ERA5_PRESSURE_LEVELS,
        target_grib.name,
        year,
        month,
        days[-1],
        ','.join(time_list),
    )

    final_path, used_days = _safe_download_with_cds_interpretation(
        client=client,
        dataset=DATASET_ERA5_PRESSURE_LEVELS,
        request=request,
        target_grib=target_grib,
        year=year,
        month=month,
        requested_end_day=end_day,
        required_hours_utc=norm_hours,
        allow_auto_trim_when_end_day_none=True,
    )

    LOGGER.info(
        '[OK] z250 global %04d-%02d salvo (%s). Dias efetivos requisitados: 01..%s | Horas=%s | Formato=GRIB',
        year,
        month,
        final_path,
        used_days[-1],
        ','.join(time_list),
    )
    return final_path


# -----------------------------------------------------------------------------
# Conversão de geopotential (z) -> altura geopotencial (m) e salvamento em NetCDF
# -----------------------------------------------------------------------------
def converter_grib_z_para_altura_geopotencial_netcdf(
    arquivo_entrada_grib: Path | str,
    arquivo_saida_nc: Path | str | None = None,
    var_out: str = 'hgt',
) -> Path:
    """
    Converte geopotential (z, m²/s²) de um GRIB para altura geopotencial (m)
    e salva em NetCDF.

    Fórmula:
        hgt = z / 9.80665
    """
    arquivo_entrada_grib = Path(arquivo_entrada_grib)

    if arquivo_saida_nc is None:
        stem = arquivo_entrada_grib.stem.replace('_z250_', '_hgt250_')
        if stem == arquivo_entrada_grib.stem:
            stem = arquivo_entrada_grib.stem + '_hgt250'
        arquivo_saida_nc = arquivo_entrada_grib.with_name(stem + '.nc')

    arquivo_saida_nc = Path(arquivo_saida_nc)

    ds = _open_grib_dataset(arquivo_entrada_grib)
    try:
        if 'z' in ds.data_vars:
            z = ds['z']
        elif 'geopotential' in ds.data_vars:
            z = ds['geopotential']
        else:
            raise KeyError(f"Arquivo {arquivo_entrada_grib} não contém variável 'z' nem 'geopotential'.")

        hgt = z / 9.80665
        hgt.name = var_out
        hgt.attrs['long_name'] = 'geopotential height'
        hgt.attrs['units'] = 'm'

        ds_out = hgt.to_dataset(name=var_out)
        ds_out.attrs.update(ds.attrs)

        ds_out.to_netcdf(arquivo_saida_nc)
    finally:
        ds.close()

    LOGGER.info('[OK] Altura geopotencial convertida e salva em: %s', arquivo_saida_nc)
    return arquivo_saida_nc


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


def ensure_era5_altura_geopotencial_250_global_for_period_grib(
    start: datetime,
    end: datetime,
    hours_utc: Sequence[int] | None = None,
    force_redownload: bool = False,
    convert_to_height_netcdf: bool = False,
) -> List[Path]:
    """
    Garante arquivos mensais do período [start, end], ajustando end_day apenas no mês final.

    Se convert_to_height_netcdf=True, gera também os arquivos convertidos para altura geopotencial em NetCDF.
    """
    files: List[Path] = []
    for y, m in _iter_year_month_range(start, end):
        end_day = end.day if (y == end.year and m == end.month) else None
        arq = download_era5_altura_geopotencial_250_global_hourly_grib(
            y,
            m,
            end_day=end_day,
            hours_utc=hours_utc,
            force_redownload=force_redownload,
        )
        if convert_to_height_netcdf:
            arq = converter_grib_z_para_altura_geopotencial_netcdf(arq)
        files.append(arq)
    return files


# -----------------------------------------------------------------------------
# Exemplo de uso local
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    arquivo_grib = download_era5_altura_geopotencial_250_global_hourly_grib(
        year=2026,
        month=2,
        end_day=20,
        hours_utc=[0, 6, 12, 18],
        force_redownload=True,
    )
    print(f'Arquivo GRIB salvo em: {arquivo_grib}')

    arquivo_hgt_nc = converter_grib_z_para_altura_geopotencial_netcdf(arquivo_grib)
    print(f'Arquivo NetCDF de altura geopotencial salvo em: {arquivo_hgt_nc}')
