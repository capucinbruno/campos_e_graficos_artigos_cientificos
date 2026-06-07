# app/src/uteis/plot_z250_mean.py
"""
Altura geopotencial média do período em 250 hPa (ERA5).

Baixa geopotencial ERA5 250 hPa, calcula a média do período em
streaming (arquivo por arquivo) e converte para altura geopotencial
(Φ / g, em metros). Salva resultado em dados/z250_mean.nc.

Chamado por: scripts/s16_wnd250_zonal_anom_div.py
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from app.src.uteis.downloaders_z250_era5 import ensure_era5_z250_for_period

try:
    from app.shared.settings_factory import settings
    DIR_DADOS_BASE = Path(settings.DIR_DADOS)
except Exception:
    DIR_DADOS_BASE = Path('dados')

LOGGER = logging.getLogger('PLOT_Z250_MEAN')
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
    )
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)

G_STANDARD = 9.80665  # m/s² — constante gravidade padrão WMO
ERA5_LATENCY_DAYS = 7
DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)


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
    if np.any(ds['lon'].values > 180):
        ds = ds.assign_coords(lon=(ds['lon'].values + 180) % 360 - 180)
        ds = ds.sortby('lon')
    return ds


def _find_z_var(ds: xr.Dataset) -> str:
    for vn in ('z', 'geopotential', 'Z_GRD_L100', 'hgt'):
        if vn in ds.data_vars:
            return vn
    raise KeyError(f'Variável geopotencial não encontrada. Disponíveis: {list(ds.data_vars)}')


def _compute_period_mean_z_streaming(
    files: Sequence[Path],
    required_hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS,
    dt_ini: Optional[datetime] = None,
    dt_fim: Optional[datetime] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula média de geopotencial 250 hPa em streaming (arquivo por arquivo)."""
    required_set = set(int(h) for h in required_hours)
    t_ini = np.datetime64(dt_ini.date()) if dt_ini else None
    t_fim = np.datetime64(dt_fim.date()) if dt_fim else None

    sum_z: Optional[np.ndarray] = None
    count_z: Optional[np.ndarray] = None
    ref_lat: Optional[np.ndarray] = None
    ref_lon: Optional[np.ndarray] = None
    total_days = 0

    for fp in files:
        LOGGER.info('Z250 streaming: abrindo %s', fp.name)
        ds = xr.open_dataset(fp, engine='netcdf4')
        try:
            ds = _ensure_time_coord(ds)
            ds = _drop_or_collapse_expver(ds)
            ds = _normalize_latlon_names(ds)
            ds = _normalize_lon(ds)
            ds = ds.sortby('time')

            z_var = _find_z_var(ds)
            da_z = ds[z_var]

            for dim_name in ('pressure_level', 'isobaricInhPa', 'level'):
                if dim_name in da_z.dims:
                    da_z = da_z.isel({dim_name: 0}, drop=True)

            t_idx = pd.DatetimeIndex(pd.to_datetime(da_z['time'].values))
            mask_h = np.array([h in required_set for h in t_idx.hour], dtype=bool)
            da_z = da_z.isel(time=mask_h)

            if da_z.sizes.get('time', 0) == 0:
                LOGGER.warning('Sem horas sinóticas válidas em: %s', fp.name)
                continue

            if t_ini is not None or t_fim is not None:
                da_z = da_z.sel(time=slice(t_ini, t_fim))
            if da_z.sizes.get('time', 0) == 0:
                LOGGER.warning('Fora do período solicitado: %s', fp.name)
                continue

            # Remove 29-fev
            t = xr.DataArray(da_z['time'].values, dims=['time'])
            mask_feb = (~((t.dt.month == 2) & (t.dt.day == 29))).values
            da_z = da_z.isel(time=mask_feb)
            if da_z.sizes.get('time', 0) == 0:
                continue

            da_z_daily = da_z.resample(time='1D').mean(keep_attrs=True)
            valid = da_z_daily.notnull().any(dim=['lat', 'lon'])
            da_z_daily = da_z_daily.isel(time=valid.values)
            n_days = da_z_daily.sizes['time']
            if n_days == 0:
                continue

            if sum_z is None:
                sum_z = np.zeros(da_z_daily.shape[1:], dtype=np.float64)
                count_z = np.zeros(da_z_daily.shape[1:], dtype=np.int64)
                ref_lat = da_z_daily['lat'].values.copy()
                ref_lon = da_z_daily['lon'].values.copy()
            elif da_z_daily.shape[1:] != sum_z.shape:
                ref_lat_da = xr.DataArray(ref_lat, dims=['lat'])
                ref_lon_da = xr.DataArray(ref_lon, dims=['lon'])
                da_z_daily = da_z_daily.interp(lat=ref_lat_da, lon=ref_lon_da, method='linear')

            vals = da_z_daily.values
            sum_z += np.nansum(vals, axis=0)
            count_z += (~np.isnan(vals)).sum(axis=0)
            total_days += n_days
            LOGGER.info('Z250 streaming: %s → %d dias (acumulado: %d)', fp.name, n_days, total_days)
        finally:
            ds.close()

    if sum_z is None or total_days == 0:
        raise RuntimeError('Nenhum dado Z250 válido encontrado no período solicitado.')

    z_mean = np.where(count_z > 0, sum_z / count_z, np.nan).astype(np.float32)
    # Geopotencial (m²/s²) → altura geopotencial (m)
    z_mean_height = z_mean / G_STANDARD
    LOGGER.info(
        'Z250 média: %d dias | Z=[%.0f, %.0f] m',
        total_days, float(np.nanmin(z_mean_height)), float(np.nanmax(z_mean_height)),
    )
    return z_mean_height, ref_lat, ref_lon


def main() -> None:
    """Download e cálculo da altura geopotencial média 250 hPa (ERA5)."""
    def _to_datetime(val) -> datetime:
        if isinstance(val, datetime):
            return val
        if hasattr(val, 'year'):
            return datetime(val.year, val.month, val.day)
        return datetime.strptime(str(val), '%Y-%m-%d')

    dt_ini = _to_datetime(settings.DATA_INICIAL)
    dt_fim = _to_datetime(settings.DATA_FINAL)
    force = getattr(settings, 'FORCE_DOWNLOAD', False)

    cutoff = (datetime.now() - timedelta(days=ERA5_LATENCY_DAYS)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    if dt_fim > cutoff:
        LOGGER.warning(
            'Z250: DATA_FINAL (%s) está dentro do período de latência ERA5 (%d dias). '
            'Usando %s como data final.',
            dt_fim.strftime('%Y-%m-%d'), ERA5_LATENCY_DAYS, cutoff.strftime('%Y-%m-%d'),
        )
        dt_fim = cutoff

    LOGGER.info('=' * 70)
    LOGGER.info('PLOT_Z250_MEAN: Altura Geopotencial 250 hPa')
    LOGGER.info('Periodo: %s a %s', settings.DATA_INICIAL, dt_fim.strftime('%Y-%m-%d'))
    LOGGER.info('=' * 70)

    files = ensure_era5_z250_for_period(
        start=dt_ini,
        end=dt_fim,
        hours_utc=list(DEFAULT_SYNOPTIC_HOURS),
        force_redownload=force,
    )

    z_mean_height, ref_lat, ref_lon = _compute_period_mean_z_streaming(
        files,
        required_hours=DEFAULT_SYNOPTIC_HOURS,
        dt_ini=dt_ini,
        dt_fim=dt_fim,
    )

    # Lat descendente para consistência com outros campos
    if ref_lat[0] < ref_lat[-1]:
        da_tmp = xr.DataArray(z_mean_height, dims=['lat', 'lon'],
                              coords={'lat': ref_lat, 'lon': ref_lon})
        da_tmp = da_tmp.sortby('lat', ascending=False)
        z_mean_height = da_tmp.values
        ref_lat = da_tmp['lat'].values

    DIR_DADOS_BASE.mkdir(parents=True, exist_ok=True)
    ds_out = xr.Dataset({
        'z_mean': xr.DataArray(
            z_mean_height,
            dims=['lat', 'lon'],
            coords={'lat': ref_lat, 'lon': ref_lon},
            attrs={'long_name': 'mean geopotential height 250 hPa', 'units': 'm'},
        )
    })
    output_path = DIR_DADOS_BASE / 'z250_mean.nc'
    if output_path.exists():
        output_path.unlink()
    ds_out.to_netcdf(output_path)
    LOGGER.info('Salvo: %s', output_path)
