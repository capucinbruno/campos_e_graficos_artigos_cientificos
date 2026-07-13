from __future__ import annotations

# Bibliotecas padrão
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Sequence

# Bibliotecas de terceiros
import httpx
import xarray as xr

# Módulos locais
from app.shared.logger import get_logger
from app.shared.settings_factory import settings

logger = get_logger(__name__)

NOMADS_FILTER_URL = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_gdas_0p25.pl'
DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)

try:
    DIR_DADOS_BASE = Path(settings.DIR_DADOS)
except Exception:
    DIR_DADOS_BASE = Path('dados')

DIR_GDAS_PWAT = DIR_DADOS_BASE / 'GDAS_PWAT'


def _build_filter_params(date_str: str, hour: int) -> dict:
    hh = f'{hour:02d}'
    return {
        'file': f'gdas.t{hh}z.pgrb2.0p25.f000',
        'lev_entire_atmosphere_(considered_as_a_single_layer)': 'on',
        'var_PWAT': 'on',
        'dir': f'/gdas.{date_str}/{hh}/atmos',
    }


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
                raise RuntimeError(f'Falha ao baixar GDAS PWAT após 3 tentativas: {target.name}') from exc


def _open_gdas_grb2(path: Path) -> xr.Dataset:
    """Abre GRIB2 filtrado do GDAS (PWAT, camada única) e normaliza coordenadas/variável."""
    ds = None
    last = None
    for fbk in ({'typeOfLevel': 'atmosphereSingleLayer'}, {}):
        try:
            ds = xr.open_dataset(path, engine='cfgrib', backend_kwargs={'indexpath': ''}, filter_by_keys=fbk)
            if len(ds.data_vars):
                break
        except Exception as exc:
            last = exc
            ds = None
    if ds is None or not len(ds.data_vars):
        raise RuntimeError(f'cfgrib nao conseguiu abrir o PWAT do GDAS: {last}')

    rename = {}
    for name in list(ds.dims) + list(ds.coords):
        low = name.lower()
        if low == 'latitude' and 'lat' not in ds.dims:
            rename[name] = 'lat'
        elif low == 'longitude' and 'lon' not in ds.dims:
            rename[name] = 'lon'
    if rename:
        ds = ds.rename(rename)

    if 'time' not in ds.coords and 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
    if 'time' not in ds.dims and 'time' in ds.coords:
        ds = ds.expand_dims('time')

    var = next((v for v in ds.data_vars if 'pwat' in str(v).lower()), list(ds.data_vars)[0])
    if var != 'pwat':
        ds = ds.rename({var: 'pwat'})

    for dim in ('atmosphere', 'level', 'surface'):
        if dim in ds.dims:
            ds = ds.isel({dim: 0}, drop=True)
        elif dim in ds.coords:
            ds = ds.drop_vars(dim, errors='ignore')

    ds['pwat'].attrs['units'] = 'kg m-2'
    ds['pwat'].attrs['long_name'] = 'precipitable water (entire atmosphere)'

    return ds[['pwat']]


def download_gdas_pwat_for_date(
    target_date: date,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS,
    force_redownload: bool = False,
) -> Path:
    """Baixa agua precipitavel do GDAS para um dia via NOMADS Grib Filter e salva como NetCDF diário."""
    date_str = target_date.strftime('%Y%m%d')
    nc_path = DIR_GDAS_PWAT / f'gdas_pwat_{date_str}.nc'

    if nc_path.exists() and not force_redownload:
        logger.info('GDAS PWAT {} já existe, pulando download.', date_str)
        return nc_path

    logger.info('Baixando GDAS PWAT para {} ({} sinóticas)', date_str, len(hours))
    DIR_GDAS_PWAT.mkdir(parents=True, exist_ok=True)

    datasets = []
    for hour in hours:
        grb_path = DIR_GDAS_PWAT / f'gdas_pwat_{date_str}_{hour:02d}z.grb2'
        params = _build_filter_params(date_str, hour)

        if not grb_path.exists() or force_redownload:
            logger.info('  Baixando {}Z...', f'{hour:02d}')
            _download_grb2(params, grb_path)

        ds = _open_gdas_grb2(grb_path)
        datasets.append(ds)

    ds_day = xr.concat(datasets, dim='time', coords='minimal', compat='override').sortby('time')

    if nc_path.exists():
        nc_path.unlink()
    ds_day.to_netcdf(nc_path, engine='netcdf4')

    for hour in hours:
        grb_path = DIR_GDAS_PWAT / f'gdas_pwat_{date_str}_{hour:02d}z.grb2'
        if grb_path.exists():
            grb_path.unlink()

    logger.info('GDAS PWAT {} salvo: {}', date_str, nc_path)
    return nc_path


def ensure_gdas_pwat_for_period(
    start: datetime,
    end: datetime,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS,
    force_redownload: bool = False,
) -> List[Path]:
    """Garante NetCDFs de agua precipitavel do GDAS para o período [start, end]."""
    files: List[Path] = []
    current = start.date()
    end_date = end.date()

    while current <= end_date:
        nc_path = download_gdas_pwat_for_date(current, hours, force_redownload)
        files.append(nc_path)
        current += timedelta(days=1)

    logger.info(
        'GDAS PWAT: {} arquivos para período {} → {}',
        len(files), start.date(), end.date(),
    )
    return files
