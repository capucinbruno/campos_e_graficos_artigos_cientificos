from __future__ import annotations

# Bibliotecas padrão
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Sequence

# Bibliotecas de terceiros
import httpx
import numpy as np
import xarray as xr

# Módulos locais
from app.shared.logger import get_logger
from app.shared.settings_factory import settings

logger = get_logger(__name__)

NOMADS_FILTER_URL = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_gdas_0p25.pl'
DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)
DEFAULT_LEVELS_HPA: List[int] = [100, 250, 300, 500, 850]

try:
    DIR_DADOS_BASE = Path(settings.DIR_DADOS)
except Exception:
    DIR_DADOS_BASE = Path('dados')

DIR_GDAS_OMEGA = DIR_DADOS_BASE / 'GDAS_OMEGA'


def _build_filter_params(date_str: str, hour: int, levels_hpa: List[int]) -> dict:
    hh = f'{hour:02d}'
    params = {
        'file': f'gdas.t{hh}z.pgrb2.0p25.f000',
        'var_VVEL': 'on',
        'dir': f'/gdas.{date_str}/{hh}/atmos',
    }
    for lev in levels_hpa:
        params[f'lev_{lev}_mb'] = 'on'
    return params


def _download_grb2(params: dict, target: Path, timeout: int = 120) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.part')

    for attempt in range(1, 4):
        try:
            with httpx.stream('GET', NOMADS_FILTER_URL, params=params, timeout=timeout, follow_redirects=True) as r:
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
                raise RuntimeError(f'Falha ao baixar GDAS omega após 3 tentativas: {target.name}') from exc


def _open_gdas_grb2_omega(path: Path, levels_hpa: List[int]) -> xr.Dataset:
    """Abre GRIB2 filtrado do GDAS e normaliza omega multi-nível para padrão do projeto."""
    # cfgrib abre todos os níveis isobáricos num único dataset quando há vários níveis
    try:
        ds = xr.open_dataset(
            path,
            engine='cfgrib',
            backend_kwargs={'indexpath': ''},
            filter_by_keys={'typeOfLevel': 'isobaricInhPa'},
        )
    except Exception as exc:
        raise RuntimeError(f'Erro ao abrir {path.name} com cfgrib: {exc}') from exc

    # Normaliza lat/lon
    rename = {}
    for name in list(ds.dims) + list(ds.coords):
        low = name.lower()
        if low == 'latitude' and 'lat' not in ds.dims:
            rename[name] = 'lat'
        elif low == 'longitude' and 'lon' not in ds.dims:
            rename[name] = 'lon'
    if rename:
        ds = ds.rename(rename)

    # Encontra variável de omega (shortName 'w' no eccodes/cfgrib para NCEP VVEL)
    omega_da = None
    for vname in ('w', 'wz', 'omega', 'vvel', 'VVEL'):
        if vname in ds.data_vars:
            omega_da = ds[vname]
            break
    if omega_da is None:
        for vname, da in ds.data_vars.items():
            if np.issubdtype(da.dtype, np.number):
                omega_da = da
                break
    if omega_da is None:
        raise RuntimeError(f'Variável omega não encontrada em {path.name}. Vars: {list(ds.data_vars)}')

    # isobaricInhPa → pressure_level
    for lname in ('isobaricInhPa', 'level'):
        if lname in omega_da.dims:
            omega_da = omega_da.rename({lname: 'pressure_level'})
            break
        if lname in omega_da.coords and 'pressure_level' not in omega_da.coords:
            omega_da = omega_da.rename({lname: 'pressure_level'})

    # time: valid_time → time; garante dim.
    # Arquivos GDAS recentes têm tanto 'time' (ref. análise) quanto 'valid_time'
    # simultaneamente; remove 'time' antes de renomear para evitar conflito.
    if 'time' not in omega_da.dims and 'valid_time' in omega_da.coords:
        if 'time' in omega_da.coords:
            omega_da = omega_da.drop_vars('time')
        omega_da = omega_da.rename({'valid_time': 'time'})
    if 'time' not in omega_da.dims and 'time' in omega_da.coords:
        omega_da = omega_da.expand_dims('time')

    # Remove step se existir como dim extra
    if 'step' in omega_da.dims:
        omega_da = omega_da.isel(step=0, drop=True)

    omega_da.attrs['units'] = 'Pa s-1'
    omega_da.attrs['long_name'] = 'vertical velocity (omega) at pressure levels'

    ds.close()
    return omega_da.rename('omega').to_dataset()


def download_gdas_omega_for_date(
    target_date: date,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS,
    levels_hpa: List[int] = DEFAULT_LEVELS_HPA,
    force_redownload: bool = False,
) -> Path:
    """Baixa omega GDAS (multi-nível) para um dia via NOMADS e salva como NetCDF diário."""
    date_str = target_date.strftime('%Y%m%d')
    levels_tag = ''.join(str(l) for l in sorted(levels_hpa))
    nc_path = DIR_GDAS_OMEGA / f'gdas_omega_{date_str}_p{levels_tag}.nc'

    if nc_path.exists() and not force_redownload:
        logger.info('GDAS omega {} já existe, pulando download.', date_str)
        return nc_path

    logger.info('Baixando GDAS omega {} — níveis={} ({} sinóticas)', date_str, levels_hpa, len(hours))
    DIR_GDAS_OMEGA.mkdir(parents=True, exist_ok=True)

    datasets = []
    for hour in hours:
        grb_path = DIR_GDAS_OMEGA / f'gdas_omega_{date_str}_{hour:02d}z_p{levels_tag}.grb2'
        params = _build_filter_params(date_str, hour, levels_hpa)

        if not grb_path.exists() or force_redownload:
            logger.info('  Baixando {}Z...', f'{hour:02d}')
            _download_grb2(params, grb_path)

        ds = _open_gdas_grb2_omega(grb_path, levels_hpa)
        datasets.append(ds)

    ds_day = xr.concat(datasets, dim='time', coords='minimal', compat='override').sortby('time')

    if nc_path.exists():
        nc_path.unlink()
    ds_day.to_netcdf(nc_path, engine='netcdf4')

    # Remove GRIBs intermediários
    for hour in hours:
        grb_path = DIR_GDAS_OMEGA / f'gdas_omega_{date_str}_{hour:02d}z_p{levels_tag}.grb2'
        if grb_path.exists():
            grb_path.unlink()

    logger.info('GDAS omega {} salvo: {}', date_str, nc_path)
    return nc_path


def ensure_gdas_omega_for_period(
    start: datetime,
    end: datetime,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS,
    levels_hpa: List[int] = DEFAULT_LEVELS_HPA,
    force_redownload: bool = False,
) -> List[Path]:
    """Garante NetCDFs de omega GDAS (multi-nível) para o período [start, end]."""
    files: List[Path] = []
    current = start.date()
    end_date = end.date()

    while current <= end_date:
        nc_path = download_gdas_omega_for_date(current, hours, levels_hpa, force_redownload)
        files.append(nc_path)
        current += timedelta(days=1)

    logger.info('GDAS omega: {} arquivos para {} → {}', len(files), start.date(), end.date())
    return files
