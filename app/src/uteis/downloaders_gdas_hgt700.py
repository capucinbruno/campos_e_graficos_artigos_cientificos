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

DIR_GDAS_HGT700 = DIR_DADOS_BASE / 'GDAS_HGT700'


def _build_filter_params(date_str: str, hour: int) -> dict:
    hh = f'{hour:02d}'
    return {
        'file': f'gdas.t{hh}z.pgrb2.0p25.f000',
        'lev_700_mb': 'on',
        'var_HGT': 'on',
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
                raise RuntimeError(f'Falha ao baixar GDAS HGT700 após 3 tentativas: {target.name}') from exc


def _open_gdas_grb2(path: Path) -> xr.Dataset:
    """Abre GRIB2 filtrado do GDAS e normaliza coordenadas/variável para padrão do projeto."""
    ds = xr.open_dataset(
        path,
        engine='cfgrib',
        backend_kwargs={'indexpath': ''},
        filter_by_keys={'typeOfLevel': 'isobaricInhPa', 'shortName': 'gh'},
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

    if 'time' not in ds.coords and 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
    if 'time' not in ds.dims and 'time' in ds.coords:
        ds = ds.expand_dims('time')

    if 'gh' in ds.data_vars:
        ds = ds.rename({'gh': 'hgt'})

    # Remover dimensão de nível (único nível: 700mb)
    for dim in ('isobaricInhPa', 'level', 'pressure_level'):
        if dim in ds.dims:
            ds = ds.isel({dim: 0}, drop=True)
        elif dim in ds.coords:
            ds = ds.drop_vars(dim)

    ds['hgt'].attrs['units'] = 'm'
    ds['hgt'].attrs['long_name'] = 'geopotential height at 700 hPa'

    return ds[['hgt']]


def download_gdas_hgt700_for_date(
    target_date: date,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS,
    force_redownload: bool = False,
) -> Path:
    """
    Baixa HGT 700mb do GDAS para um dia via NOMADS Grib Filter e salva como NetCDF diário.

    Retorna path do NetCDF com as horas sinóticas concatenadas.
    """
    date_str = target_date.strftime('%Y%m%d')
    nc_path = DIR_GDAS_HGT700 / f'gdas_hgt700_{date_str}.nc'

    if nc_path.exists() and not force_redownload:
        logger.info('GDAS HGT700 {} já existe, pulando download.', date_str)
        return nc_path

    logger.info('Baixando GDAS HGT700 para {} ({} sinóticas)', date_str, len(hours))
    DIR_GDAS_HGT700.mkdir(parents=True, exist_ok=True)

    datasets = []
    for hour in hours:
        grb_path = DIR_GDAS_HGT700 / f'gdas_hgt700_{date_str}_{hour:02d}z.grb2'
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
        grb_path = DIR_GDAS_HGT700 / f'gdas_hgt700_{date_str}_{hour:02d}z.grb2'
        if grb_path.exists():
            grb_path.unlink()

    logger.info('GDAS HGT700 {} salvo: {}', date_str, nc_path)
    return nc_path


def ensure_gdas_hgt700_for_period(
    start: datetime,
    end: datetime,
    hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS,
    force_redownload: bool = False,
) -> List[Path]:
    """
    Garante NetCDFs de HGT 700mb do GDAS para o período [start, end].

    Retorna lista de paths (um NetCDF por dia).
    """
    files: List[Path] = []
    current = start.date()
    end_date = end.date()

    while current <= end_date:
        nc_path = download_gdas_hgt700_for_date(current, hours, force_redownload)
        files.append(nc_path)
        current += timedelta(days=1)

    logger.info(
        'GDAS HGT700: {} arquivos para período {} → {}',
        len(files), start.date(), end.date(),
    )
    return files
