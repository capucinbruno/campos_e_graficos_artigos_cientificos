# app/src/uteis/plot_geop250.py
# -*- coding: utf-8 -*-
"""
Download e processamento de altura geopotencial 250 hPa (ERA5) para anomalia.

Pipeline:
1. Baixa dados ERA5 de geopotencial (z) em 250 hPa via CDS (GRIB)
2. Converte geopotencial (m2/s2) para altura geopotencial (m)
3. Concatena arquivos mensais e calcula media diaria
4. Carrega climatologia 1991-2020 de geopotencial 250 hPa
5. Calcula anomalia = media_periodo - media_climatologica
6. Salva resultado em dados/geop250.nc (variavel 'hgt', 1 timestep)

Chamado por: scripts/s01_geop250_anom.py via `from app.src.uteis.plot_geop250 import main`
"""

from __future__ import annotations

# Bibliotecas padrão
import logging
from datetime import datetime
from pathlib import Path
from typing import Sequence

# Bibliotecas de terceiros
import numpy as np
import pandas as pd
import xarray as xr

# Módulos locais
from app.src.uteis.downloaders_hgt250_ERA5 import (
    ensure_era5_altura_geopotencial_250_global_for_period_grib,
)

# -----------------------------------------------------------------------------
# Integracao com settings
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
LOGGER = logging.getLogger('PLOT_GEOP250')
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)

# Horas sinoticas padrao
DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)


# -----------------------------------------------------------------------------
# Utilitarios
# -----------------------------------------------------------------------------
def _ensure_time_coord(obj):
    """Normaliza valid_time -> time."""
    if hasattr(obj, 'dims') and 'time' not in obj.dims and 'valid_time' in obj.dims:
        obj = obj.rename({'valid_time': 'time'})
    elif hasattr(obj, 'coords') and 'time' not in obj.coords and 'valid_time' in obj.coords:
        obj = obj.rename({'valid_time': 'time'})
    if 'time' not in obj.coords:
        raise KeyError("Nem 'time' nem 'valid_time' encontrados.")
    return obj


def _drop_or_collapse_expver(ds: xr.Dataset) -> xr.Dataset:
    """Colapsa dim expver/number se existirem (ERA5/ERA5T)."""
    rename_dims = {}
    for d in ds.dims:
        dl = d.lower()
        if dl == 'expver' and d != 'expver':
            rename_dims[d] = 'expver'
        elif dl == 'number' and d != 'number':
            rename_dims[d] = 'number'

    if rename_dims:
        ds = ds.rename(rename_dims)

    if 'expver' in ds.dims:
        ds = ds.bfill('expver').ffill('expver').isel(expver=0, drop=True)

    if 'number' in ds.dims:
        ds = ds.isel(number=0, drop=True)

    for c in ('expver', 'number'):
        if c in ds.coords and c not in ds.dims:
            try:
                ds = ds.drop_vars(c)
            except Exception:
                pass

    return ds


def _normalize_latlon_names(ds):
    """Renomeia latitude/longitude -> lat/lon se necessario."""
    rename = {}
    for name in ds.dims:
        low = name.lower()
        if low in {'latitude'} and 'lat' not in ds.dims:
            rename[name] = 'lat'
        elif low in {'longitude'} and 'lon' not in ds.dims:
            rename[name] = 'lon'
    if rename:
        ds = ds.rename(rename)
    return ds


def _normalize_lon(ds):
    """Converte longitude de 0..360 para -180..180 se necessario."""
    if 'lon' not in ds.coords:
        return ds
    lon_vals = ds['lon'].values
    if np.any(lon_vals > 180):
        ds = ds.assign_coords(lon=(ds['lon'].values + 180) % 360 - 180)
        ds = ds.sortby('lon')
    return ds


def _sort_and_dedup_time(ds: xr.Dataset) -> xr.Dataset:
    """Ordena e remove timestamps duplicados."""
    ds = ds.sortby('time')
    t = pd.DatetimeIndex(pd.to_datetime(ds['time'].values))
    _, idx = np.unique(t.values, return_index=True)
    idx = np.sort(idx)
    if len(idx) != ds.sizes.get('time', 0):
        LOGGER.warning('Removendo %d timestamps duplicados.', ds.sizes['time'] - len(idx))
        ds = ds.isel(time=idx)
    return ds


def _drop_feb29(ds: xr.Dataset) -> xr.Dataset:
    """Remove 29/02 para compatibilizar com climatologia 365d."""
    ds = _ensure_time_coord(ds)
    t = xr.DataArray(ds['time'].values, dims=['time'])
    mask = ~((t.dt.month == 2) & (t.dt.day == 29))
    n_before = ds.sizes['time']
    ds2 = ds.isel(time=mask.values)
    n_after = ds2.sizes['time']
    if n_after < n_before:
        LOGGER.warning(
            'Removendo %d registro(s) de 29/02 para compatibilizar com climatologia 365d.',
            n_before - n_after,
        )
    return ds2


# -----------------------------------------------------------------------------
# Abertura e concatenacao de arquivos convertidos (NetCDF hgt)
# -----------------------------------------------------------------------------
def _open_and_merge_hgt_files(files: Sequence[Path]) -> xr.Dataset:
    """Abre e concatena arquivos mensais de altura geopotencial em NetCDF."""
    if not files:
        raise ValueError('Lista de arquivos de altura geopotencial vazia.')

    dsets = []
    for fp in files:
        LOGGER.info('Abrindo: %s', fp)
        ds = xr.open_dataset(fp, engine='netcdf4')
        ds = _ensure_time_coord(ds)
        ds = _drop_or_collapse_expver(ds)
        ds = _normalize_latlon_names(ds)
        dsets.append(ds)

    if len(dsets) == 1:
        ds_all = dsets[0]
    else:
        ds_all = xr.concat(dsets, dim='time', data_vars='minimal', coords='minimal', compat='override')

    ds_all = _ensure_time_coord(ds_all)
    ds_all = _sort_and_dedup_time(ds_all)

    LOGGER.info(
        'Dataset combinado: %d timestamps | %s -> %s',
        ds_all.sizes.get('time', 0),
        str(pd.to_datetime(ds_all.time.values[0])) if ds_all.sizes.get('time', 0) else 'NA',
        str(pd.to_datetime(ds_all.time.values[-1])) if ds_all.sizes.get('time', 0) else 'NA',
    )
    return ds_all


# -----------------------------------------------------------------------------
# Media diaria a partir de horas sinoticas
# -----------------------------------------------------------------------------
def _compute_daily_mean(
    ds: xr.Dataset,
    required_hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS,
) -> xr.Dataset:
    """Calcula media diaria filtrando apenas horas sinoticas requeridas."""
    ds = _ensure_time_coord(ds)
    ds = _sort_and_dedup_time(ds)

    required_set = set(int(h) for h in required_hours)
    t_idx = pd.DatetimeIndex(pd.to_datetime(ds['time'].values))

    mask = np.array([h in required_set for h in t_idx.hour], dtype=bool)
    ds = ds.isel(time=mask)

    if ds.sizes.get('time', 0) == 0:
        raise ValueError('Dataset sem timestamps apos filtrar horas sinoticas.')

    ds_daily = ds.resample(time='1D').mean(keep_attrs=True)

    # Remove dias vazios criados pelo resample
    for vn in ds_daily.data_vars:
        da = ds_daily[vn]
        other_dims = [d for d in da.dims if d != 'time']
        if other_dims:
            valid = da.notnull().any(dim=other_dims)
        else:
            valid = da.notnull()
        ds_daily = ds_daily.isel(time=valid.values)
        break

    LOGGER.info(
        'Media diaria: %d dias | %s -> %s',
        ds_daily.sizes.get('time', 0),
        str(pd.to_datetime(ds_daily.time.values[0])) if ds_daily.sizes.get('time', 0) else 'NA',
        str(pd.to_datetime(ds_daily.time.values[-1])) if ds_daily.sizes.get('time', 0) else 'NA',
    )
    return ds_daily


# -----------------------------------------------------------------------------
# Climatologia
# -----------------------------------------------------------------------------
def _load_climatology_geop250(path_clim: Path) -> xr.DataArray:
    """Carrega climatologia de geopotencial 250 hPa e retorna DataArray de hgt."""
    if not path_clim.exists():
        raise FileNotFoundError(f'Climatologia geop250 nao encontrada: {path_clim}')

    LOGGER.info('Abrindo climatologia geop250: %s', path_clim)
    ds = xr.open_dataset(path_clim, engine='netcdf4')
    ds = _normalize_latlon_names(ds)

    LOGGER.info('Variaveis na climatologia: %s', list(ds.data_vars))
    LOGGER.info('Coords na climatologia: %s', list(ds.coords))

    # Procura variavel de altura geopotencial
    for vname in ('hgt', 'z', 'geopotential', 'gh', 'geopotential_height'):
        if vname in ds.data_vars:
            da = ds[vname]
            # Se for geopotential (m2/s2), converte para metros
            units = da.attrs.get('units', '')
            if 'm**2' in units or 'm2' in units or 'J' in units:
                LOGGER.info('Convertendo climatologia de geopotencial (m2/s2) para altura (m)')
                da = da / 9.80665
                da.attrs['units'] = 'm'
                da.attrs['long_name'] = 'geopotential height'
            return da

    # Fallback: primeira variavel numerica 3D
    for vname in ds.data_vars:
        if np.issubdtype(ds[vname].dtype, np.number) and len(ds[vname].dims) >= 2:
            LOGGER.warning('Usando variavel %s como altura geopotencial da climatologia.', vname)
            return ds[vname]

    raise KeyError(
        f'Nao encontrei variavel de altura geopotencial na climatologia: {list(ds.data_vars)}'
    )


def _select_climatology_same_days(
    clim: xr.DataArray,
    daily_dates: xr.DataArray,
) -> xr.DataArray:
    """Seleciona na climatologia os mesmos dias do periodo.

    Suporta 3 formatos de coordenada temporal na climatologia:
    - time com inteiros 1-365 (dayofyear como inteiro)
    - time com datetimes (selecao por month-day)
    - dayofyear como coordenada separada
    """
    target_dates = pd.DatetimeIndex(pd.to_datetime(daily_dates.values))

    # Caso 1: coord 'time' com inteiros 1-365 (dayofyear)
    if 'time' in clim.coords:
        time_vals = clim['time'].values
        if np.issubdtype(type(time_vals[0]), np.integer) or (
            hasattr(time_vals[0], 'dtype') and np.issubdtype(time_vals[0].dtype, np.integer)
        ):
            LOGGER.info('Climatologia com coord time inteira (dayofyear 1-365).')
            target_doy = target_dates.dayofyear.to_numpy()
            clim_sel = clim.sel(time=xr.DataArray(target_doy, dims=['time']))
            clim_sel = clim_sel.assign_coords(time=daily_dates.values)
            LOGGER.info('Climatologia selecionada por dayofyear (inteiro): %d dias.', len(target_doy))
            return clim_sel

        # Caso 2: coord 'time' com datetimes
        try:
            clim_dates = pd.DatetimeIndex(pd.to_datetime(time_vals))
        except Exception:
            raise KeyError(
                f"Coord 'time' da climatologia nao e datetime nem inteiro: {type(time_vals[0])}"
            )

        clim_md = pd.Index(
            [f'{t.month:02d}-{t.day:02d}' for t in clim_dates],
            name='monthday',
        )
        target_md = [f'{t.month:02d}-{t.day:02d}' for t in target_dates]

        md_to_idx = {}
        for i, md in enumerate(clim_md.tolist()):
            md_to_idx[md] = i

        missing = [md for md in target_md if md not in md_to_idx]
        if missing:
            LOGGER.warning(
                'Dias ausentes na climatologia (serao ignorados na anomalia): %s', missing[:10]
            )
            valid_pairs = [(md, i) for i, md in enumerate(target_md) if md in md_to_idx]
            if not valid_pairs:
                raise ValueError('Nenhum dia do periodo encontrado na climatologia.')
            valid_md, valid_target_idx = zip(*valid_pairs)
            idxs = [md_to_idx[md] for md in valid_md]
            clim_sel = clim.isel(time=idxs)
            target_times = daily_dates.values[list(valid_target_idx)]
            clim_sel = clim_sel.assign_coords(time=target_times)
        else:
            idxs = [md_to_idx[md] for md in target_md]
            clim_sel = clim.isel(time=idxs)
            clim_sel = clim_sel.assign_coords(time=daily_dates.values)

        LOGGER.info("Climatologia selecionada por month-day usando coord 'time'.")
        return clim_sel

    # Caso 3: coord 'dayofyear' separada
    if 'dayofyear' in clim.coords:
        target_doy = target_dates.dayofyear.to_numpy()
        clim_sel = clim.sel(dayofyear=xr.DataArray(target_doy, dims=['time']))
        clim_sel = clim_sel.assign_coords(time=daily_dates.values)
        LOGGER.info("Climatologia selecionada por coord 'dayofyear'.")
        return clim_sel

    raise KeyError("Climatologia precisa ter coord 'time' (datetime ou dayofyear int) ou 'dayofyear'.")


# -----------------------------------------------------------------------------
# Pipeline principal
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Download, processamento e calculo de anomalia de geopotencial 250 hPa.

    Resultado: dados/geop250.nc com variavel 'hgt' (anomalia em metros, 1 timestep).
    """
    dt_ini = datetime.strptime(settings.DATA_INICIAL, '%Y-%m-%d')
    dt_fim = datetime.strptime(settings.DATA_FINAL, '%Y-%m-%d')

    LOGGER.info('=' * 70)
    LOGGER.info('PLOT_GEOP250: Download e anomalia geopotencial 250 hPa')
    LOGGER.info('Periodo: %s a %s', settings.DATA_INICIAL, settings.DATA_FINAL)
    LOGGER.info('=' * 70)

    # 1. Download ERA5 GRIB + conversao para NetCDF (hgt em metros)
    LOGGER.info('Etapa 1: Download ERA5 geopotencial 250 hPa (NetCDF) e conversao z -> hgt')
    hgt_files = ensure_era5_altura_geopotencial_250_global_for_period_grib(
        start=dt_ini,
        end=dt_fim,
        hours_utc=list(DEFAULT_SYNOPTIC_HOURS),
        force_redownload=getattr(settings, 'FORCE_DOWNLOAD', False),
        convert_to_height_netcdf=True,
    )
    LOGGER.info('Arquivos de altura geopotencial convertidos: %d', len(hgt_files))
    for f in hgt_files:
        LOGGER.info('  - %s', f)

    # 2. Abrir e concatenar arquivos mensais
    LOGGER.info('Etapa 2: Concatenando arquivos mensais')
    ds_hgt = _open_and_merge_hgt_files(hgt_files)
    ds_hgt = _normalize_latlon_names(ds_hgt)
    ds_hgt = _normalize_lon(ds_hgt)

    # 3. Media diaria (00/06/12/18 UTC)
    LOGGER.info('Etapa 3: Calculando media diaria')
    ds_daily = _compute_daily_mean(ds_hgt)
    ds_daily = _drop_feb29(ds_daily)

    # Identificar variavel de altura
    hgt_var = None
    for vname in ('hgt', 'z', 'geopotential'):
        if vname in ds_daily.data_vars:
            hgt_var = vname
            break
    if hgt_var is None:
        hgt_var = list(ds_daily.data_vars)[0]
        LOGGER.warning('Usando variavel %s como altura geopotencial.', hgt_var)

    hgt_daily = ds_daily[hgt_var]

    # Dropar dimensao pressure_level/isobaricInhPa se existir (ERA5 250hPa unico nivel)
    for dim_name in ('pressure_level', 'isobaricInhPa', 'level'):
        if dim_name in hgt_daily.dims:
            hgt_daily = hgt_daily.isel({dim_name: 0}, drop=True)
            LOGGER.info('Dimensao %s removida do ERA5 (nivel unico).', dim_name)

    # 4. Carregar climatologia e calcular anomalia
    LOGGER.info('Etapa 4: Carregando climatologia e calculando anomalia')
    clim_path = Path(settings.FILE_CLIMATOLOGIA_GEOP250)
    clim_da = _load_climatology_geop250(clim_path)

    # Dropar dimensao 'level' se existir (climatologia pode ter level=250 como dim escalar)
    if 'level' in clim_da.dims:
        clim_da = clim_da.isel(level=0, drop=True)
    elif 'level' in clim_da.coords:
        clim_da = clim_da.drop_vars('level')

    ds_clim = _normalize_latlon_names(clim_da.to_dataset(name=clim_da.name or 'hgt_clim'))
    ds_clim = _normalize_lon(ds_clim)
    clim_da = list(ds_clim.data_vars.values())[0]

    clim_period = _select_climatology_same_days(
        clim=clim_da,
        daily_dates=hgt_daily['time'],
    )

    # Interpolar climatologia para o grid do ERA5 se necessario
    if (
        'lat' in hgt_daily.coords
        and 'lon' in hgt_daily.coords
        and 'lat' in clim_period.coords
        and 'lon' in clim_period.coords
    ):
        if (
            hgt_daily.sizes.get('lat') != clim_period.sizes.get('lat')
            or hgt_daily.sizes.get('lon') != clim_period.sizes.get('lon')
            or not np.array_equal(hgt_daily['lat'].values, clim_period['lat'].values)
            or not np.array_equal(hgt_daily['lon'].values, clim_period['lon'].values)
        ):
            LOGGER.warning('Grid da climatologia difere do ERA5. Interpolando climatologia.')
            clim_period = clim_period.interp_like(hgt_daily)

    # Media do periodo e media climatologica
    hgt_period_mean = hgt_daily.mean(dim='time')
    clim_mean = clim_period.mean(dim='time')

    # Anomalia = observado - climatologia
    anomaly = hgt_period_mean - clim_mean
    anomaly.name = 'hgt'
    anomaly.attrs['long_name'] = 'geopotential height anomaly at 250 hPa'
    anomaly.attrs['units'] = 'm'

    LOGGER.info(
        'Anomalia calculada: min=%.1f, max=%.1f, mean=%.1f mgp',
        float(anomaly.min()),
        float(anomaly.max()),
        float(anomaly.mean()),
    )

    # 5. Salvar como dados/geop250.nc (1 timestep para compatibilidade com s01)
    # O s01 espera ds['hgt'].isel(time=0), entao adicionamos dim time
    anomaly_ds = anomaly.expand_dims('time').to_dataset(name='hgt')
    anomaly_ds['time'] = [pd.Timestamp(settings.DATA_FINAL)]

    output_path = DIR_DADOS_BASE / 'geop250.nc'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    anomaly_ds.to_netcdf(output_path, engine='netcdf4')
    LOGGER.info('Anomalia salva em: %s', output_path)
    LOGGER.info('=' * 70)
    LOGGER.info('PLOT_GEOP250: Concluido com sucesso')
    LOGGER.info('=' * 70)


if __name__ == '__main__':
    main()
