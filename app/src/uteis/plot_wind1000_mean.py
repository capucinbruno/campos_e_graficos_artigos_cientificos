# app/src/uteis/plot_wind1000_mean.py
# -*- coding: utf-8 -*-
"""
Download e processamento de vento medio 1000 hPa (ERA5/GDAS) para s12.

Pipeline:
1. Seleciona fonte de dados por latencia do ERA5 (~7 dias):
   - Periodo recente (ultimos 7 dias): GDAS via NOMADS Grib Filter
   - Periodo mais antigo: ERA5 via Copernicus CDS (1000 hPa)
   - Hibrido: ERA5 [ini->cutoff-1] + GDAS [cutoff->fim]
2. Processa um arquivo por vez (streaming) — sem carregar tudo na RAM
3. Calcula media simples (nao e anomalia — sem climatologia PSL)
4. Salva resultado em dados/wind1000_mean.nc (variaveis u_mean, v_mean)

Chamado por: scripts/s12_sst_todas_areas.py
"""

from __future__ import annotations

# Bibliotecas padrão
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence, Tuple

# Bibliotecas de terceiros
import numpy as np
import pandas as pd
import xarray as xr

# Módulos locais
from app.src.uteis.downloaders_gdas_uv1000 import ensure_gdas_uv1000_for_period
from app.src.uteis.downloaders_wind1000_ERA5 import ensure_era5_uv1000_for_period

# -----------------------------------------------------------------------------
# Integração com settings
# -----------------------------------------------------------------------------
try:
    from app.shared.settings_factory import settings  # type: ignore

    DIR_DADOS_BASE = Path(settings.DIR_DADOS)
except Exception:
    DIR_DADOS_BASE = Path('dados')

# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
LOGGER = logging.getLogger('PLOT_WIND1000_MEAN')
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)

DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)
ERA5_LATENCY_DAYS = 7
WIND1000_FILE_NAME = 'wind1000_mean.nc'


# -----------------------------------------------------------------------------
# Utilitários (idênticos ao plot_olr_wind850_anom)
# -----------------------------------------------------------------------------
def _ensure_time_coord(obj):
    if hasattr(obj, 'dims') and 'time' not in obj.dims and 'valid_time' in obj.dims:
        obj = obj.rename({'valid_time': 'time'})
    elif hasattr(obj, 'coords') and 'time' not in obj.coords and 'valid_time' in obj.coords:
        obj = obj.rename({'valid_time': 'time'})
    if 'time' not in obj.coords:
        raise KeyError("Nem 'time' nem 'valid_time' encontrados.")
    return obj


def _drop_or_collapse_expver(ds: xr.Dataset) -> xr.Dataset:
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
    rename = {}
    for name in ds.dims:
        low = name.lower()
        if low == 'latitude' and 'lat' not in ds.dims:
            rename[name] = 'lat'
        elif low == 'longitude' and 'lon' not in ds.dims:
            rename[name] = 'lon'
    if rename:
        ds = ds.rename(rename)
    return ds


def _normalize_lon(ds):
    if 'lon' not in ds.coords:
        return ds
    lon_vals = ds['lon'].values
    if np.any(lon_vals > 180):
        ds = ds.assign_coords(lon=(ds['lon'].values + 180) % 360 - 180)
        ds = ds.sortby('lon')
    return ds


def _sort_and_dedup_time(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.sortby('time')
    t = pd.DatetimeIndex(pd.to_datetime(ds['time'].values))
    _, idx = np.unique(t.values, return_index=True)
    idx = np.sort(idx)
    if len(idx) != ds.sizes.get('time', 0):
        ds = ds.isel(time=idx)
    return ds


def _find_uv_vars(ds: xr.Dataset) -> tuple[str, str]:
    u_candidates = ('u', 'u_component_of_wind', 'U_GRD_L100', 'uwnd')
    v_candidates = ('v', 'v_component_of_wind', 'V_GRD_L100', 'vwnd')
    u_var = next((vn for vn in u_candidates if vn in ds.data_vars), None)
    v_var = next((vn for vn in v_candidates if vn in ds.data_vars), None)
    if u_var is None or v_var is None:
        raise KeyError(f'Não encontrei u/v no dataset. Disponíveis: {list(ds.data_vars)}')
    return u_var, v_var


# -----------------------------------------------------------------------------
# Seleção de fonte de dados (ERA5 vs GDAS)
# -----------------------------------------------------------------------------
def _get_data_sources(
    dt_ini: datetime,
    dt_fim: datetime,
) -> Tuple[Optional[Tuple[datetime, datetime]], Optional[Tuple[datetime, datetime]]]:
    cutoff = (datetime.now() - timedelta(days=ERA5_LATENCY_DAYS)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    if dt_fim < cutoff:
        return (dt_ini, dt_fim), None
    if dt_ini >= cutoff:
        return None, (dt_ini, dt_fim)
    return (dt_ini, cutoff - timedelta(days=1)), (cutoff, dt_fim)


# -----------------------------------------------------------------------------
# Streaming accumulator (um arquivo por vez, sem xr.concat na RAM)
# -----------------------------------------------------------------------------
def _compute_period_mean_streaming_uv(
    files: Sequence[Path],
    required_hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS,
    dt_ini: Optional[datetime] = None,
    dt_fim: Optional[datetime] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calcula médias do período para u e v 1000 hPa processando um arquivo por vez.

    Retorna (u_mean_2d, v_mean_2d, ref_lat, ref_lon).
    """
    required_set = set(int(h) for h in required_hours)
    t_ini = np.datetime64(dt_ini.date()) if dt_ini else None
    t_fim = np.datetime64(dt_fim.date()) if dt_fim else None

    sum_u: Optional[np.ndarray] = None
    sum_v: Optional[np.ndarray] = None
    count_u: Optional[np.ndarray] = None
    count_v: Optional[np.ndarray] = None
    ref_lat: Optional[np.ndarray] = None
    ref_lon: Optional[np.ndarray] = None
    total_days = 0

    for fp in files:
        LOGGER.info('Streaming: abrindo %s', fp.name)
        ds = xr.open_dataset(fp, engine='netcdf4')
        try:
            ds = _ensure_time_coord(ds)
            ds = _drop_or_collapse_expver(ds)
            ds = _normalize_latlon_names(ds)
            ds = _normalize_lon(ds)
            ds = _sort_and_dedup_time(ds)

            u_var, v_var = _find_uv_vars(ds)
            da_u = ds[u_var]
            da_v = ds[v_var]

            for dim_name in ('pressure_level', 'isobaricInhPa', 'level'):
                if dim_name in da_u.dims:
                    da_u = da_u.isel({dim_name: 0}, drop=True)
                    da_v = da_v.isel({dim_name: 0}, drop=True)

            t_idx = pd.DatetimeIndex(pd.to_datetime(da_u['time'].values))
            mask_h = np.array([h in required_set for h in t_idx.hour], dtype=bool)
            da_u = da_u.isel(time=mask_h)
            da_v = da_v.isel(time=mask_h)

            if da_u.sizes.get('time', 0) == 0:
                LOGGER.warning('Sem horas sinóticas válidas em: %s', fp.name)
                continue

            if t_ini is not None or t_fim is not None:
                da_u = da_u.sel(time=slice(t_ini, t_fim))
                da_v = da_v.sel(time=slice(t_ini, t_fim))
            if da_u.sizes.get('time', 0) == 0:
                LOGGER.warning('Fora do período solicitado: %s', fp.name)
                continue

            t = xr.DataArray(da_u['time'].values, dims=['time'])
            mask_feb = (~((t.dt.month == 2) & (t.dt.day == 29))).values
            da_u = da_u.isel(time=mask_feb)
            da_v = da_v.isel(time=mask_feb)

            if da_u.sizes.get('time', 0) == 0:
                continue

            da_u_daily = da_u.resample(time='1D').mean(keep_attrs=True)
            da_v_daily = da_v.resample(time='1D').mean(keep_attrs=True)

            valid = da_u_daily.notnull().any(dim=['lat', 'lon'])
            da_u_daily = da_u_daily.isel(time=valid.values)
            da_v_daily = da_v_daily.isel(time=valid.values)
            n_days_file = da_u_daily.sizes['time']
            if n_days_file == 0:
                continue

            if sum_u is None:
                sum_u = np.zeros(da_u_daily.shape[1:], dtype=np.float64)
                sum_v = np.zeros(da_v_daily.shape[1:], dtype=np.float64)
                count_u = np.zeros(da_u_daily.shape[1:], dtype=np.int64)
                count_v = np.zeros(da_v_daily.shape[1:], dtype=np.int64)
                ref_lat = da_u_daily['lat'].values.copy()
                ref_lon = da_u_daily['lon'].values.copy()
            elif da_u_daily.shape[1:] != sum_u.shape:
                ref_lat_da = xr.DataArray(ref_lat, dims=['lat'])
                ref_lon_da = xr.DataArray(ref_lon, dims=['lon'])
                da_u_daily = da_u_daily.interp(lat=ref_lat_da, lon=ref_lon_da, method='linear')
                da_v_daily = da_v_daily.interp(lat=ref_lat_da, lon=ref_lon_da, method='linear')
                LOGGER.info('Grade interpolada para referência (%dx%d).', len(ref_lat), len(ref_lon))

            vals_u = da_u_daily.values
            vals_v = da_v_daily.values

            sum_u += np.nansum(vals_u, axis=0)
            sum_v += np.nansum(vals_v, axis=0)
            count_u += (~np.isnan(vals_u)).sum(axis=0)
            count_v += (~np.isnan(vals_v)).sum(axis=0)
            total_days += n_days_file

            LOGGER.info('Streaming: %s → %d dias (acumulado: %d)', fp.name, n_days_file, total_days)
        finally:
            ds.close()

    if sum_u is None or total_days == 0:
        raise RuntimeError('Nenhum dado válido de vento 1000 hPa encontrado no período solicitado.')

    u_mean = np.where(count_u > 0, sum_u / count_u, np.nan).astype(np.float32)
    v_mean = np.where(count_v > 0, sum_v / count_v, np.nan).astype(np.float32)

    LOGGER.info(
        'Média do período: %d dias | u=[%.2f, %.2f] m/s | v=[%.2f, %.2f] m/s',
        total_days,
        float(np.nanmin(u_mean)), float(np.nanmax(u_mean)),
        float(np.nanmin(v_mean)), float(np.nanmax(v_mean)),
    )

    return u_mean, v_mean, ref_lat, ref_lon


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main() -> Path:
    """Calcula média de vento 1000 hPa do período e salva em dados/wind1000_mean.nc."""
    dt_ini = datetime.strptime(settings.DATA_INICIAL, '%Y-%m-%d')
    dt_fim = datetime.strptime(settings.DATA_FINAL, '%Y-%m-%d')

    LOGGER.info('Vento 1000 hPa média — período: %s → %s', dt_ini.date(), dt_fim.date())

    era5_range, gdas_range = _get_data_sources(dt_ini, dt_fim)

    all_files: list[Path] = []

    if era5_range:
        LOGGER.info('Baixando ERA5 UV1000: %s → %s', era5_range[0].date(), era5_range[1].date())
        files = ensure_era5_uv1000_for_period(era5_range[0], era5_range[1])
        all_files.extend(files)

    if gdas_range:
        LOGGER.info('Baixando GDAS UV1000: %s → %s', gdas_range[0].date(), gdas_range[1].date())
        files = ensure_gdas_uv1000_for_period(gdas_range[0], gdas_range[1])
        all_files.extend(files)

    if not all_files:
        raise RuntimeError('Nenhum arquivo de vento 1000 hPa disponível para o período solicitado.')

    u_mean, v_mean, ref_lat, ref_lon = _compute_period_mean_streaming_uv(
        all_files,
        required_hours=DEFAULT_SYNOPTIC_HOURS,
        dt_ini=dt_ini,
        dt_fim=dt_fim,
    )

    DIR_DADOS_BASE.mkdir(parents=True, exist_ok=True)
    out_path = DIR_DADOS_BASE / WIND1000_FILE_NAME
    ds_out = xr.Dataset(
        {
            'u_mean': xr.DataArray(u_mean, dims=['lat', 'lon'], attrs={'units': 'm s-1', 'long_name': 'U-component mean wind 1000 hPa'}),
            'v_mean': xr.DataArray(v_mean, dims=['lat', 'lon'], attrs={'units': 'm s-1', 'long_name': 'V-component mean wind 1000 hPa'}),
        },
        coords={'lat': ref_lat, 'lon': ref_lon},
    )
    if out_path.exists():
        out_path.unlink()
    ds_out.to_netcdf(str(out_path), engine='netcdf4')
    ds_out.close()

    LOGGER.info('Salvo: %s (grade %dx%d)', out_path.name, len(ref_lat), len(ref_lon))
    return out_path
