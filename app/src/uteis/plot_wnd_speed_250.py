# app/src/uteis/plot_wnd_speed_250.py
"""
Magnitude media do vento 250 hPa: speed = sqrt(u² + v²).

Reutiliza os mesmos arquivos ERA5/GDAS baixados por plot_wnd_zonal_250_anom
(sem novos downloads quando os dados ja existem no disco).
Salva resultado em dados/wnd_speed_250.nc.

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

from app.src.uteis.downloaders_gdas_uv250 import ensure_gdas_uv250_for_period
from app.src.uteis.downloaders_wind250 import ensure_era5_uv250_for_period

try:
    from app.shared.settings_factory import settings
    DIR_DADOS_BASE = Path(settings.DIR_DADOS)
except Exception:
    DIR_DADOS_BASE = Path('dados')

LOGGER = logging.getLogger('PLOT_WND_SPEED_250')
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
    )
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)

DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)
ERA5_LATENCY_DAYS = 7


# ---------------------------------------------------------------------------
# Utilitários internos (mesmo padrão de plot_wnd_zonal_250_anom)
# ---------------------------------------------------------------------------
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


def _sort_and_dedup_time(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.sortby('time')
    t = pd.DatetimeIndex(pd.to_datetime(ds['time'].values))
    _, idx = np.unique(t.values, return_index=True)
    idx = np.sort(idx)
    if len(idx) != ds.sizes.get('time', 0):
        ds = ds.isel(time=idx)
    return ds


def _find_u_var(ds: xr.Dataset) -> str:
    for vn in ('u', 'u_component_of_wind', 'U_GRD_L100', 'uwnd'):
        if vn in ds.data_vars:
            return vn
    raise KeyError(f'Variável u não encontrada. Disponíveis: {list(ds.data_vars)}')


def _find_v_var(ds: xr.Dataset) -> str:
    for vn in ('v', 'v_component_of_wind', 'V_GRD_L100', 'vwnd'):
        if vn in ds.data_vars:
            return vn
    raise KeyError(f'Variável v não encontrada. Disponíveis: {list(ds.data_vars)}')


def _get_data_sources(
    dt_ini: datetime, dt_fim: datetime,
) -> Tuple[Optional[Tuple[datetime, datetime]], Optional[Tuple[datetime, datetime]]]:
    cutoff = (datetime.now() - timedelta(days=ERA5_LATENCY_DAYS)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    if dt_fim < cutoff:
        return (dt_ini, dt_fim), None
    if dt_ini >= cutoff:
        return None, (dt_ini, dt_fim)
    return (dt_ini, cutoff - timedelta(days=1)), (cutoff, dt_fim)


# ---------------------------------------------------------------------------
# Streaming: u e v simultâneos → speed = sqrt(u²+v²)
# ---------------------------------------------------------------------------
def _compute_period_mean_speed_streaming(
    files: Sequence[Path],
    required_hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS,
    dt_ini: Optional[datetime] = None,
    dt_fim: Optional[datetime] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula sqrt(u²+v²) médio do período processando um arquivo por vez."""
    required_set = set(int(h) for h in required_hours)
    t_ini = np.datetime64(dt_ini.date()) if dt_ini else None
    t_fim = np.datetime64(dt_fim.date()) if dt_fim else None

    sum_spd: Optional[np.ndarray] = None
    count_spd: Optional[np.ndarray] = None
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

            u_var = _find_u_var(ds)
            v_var = _find_v_var(ds)
            da_u = ds[u_var]
            da_v = ds[v_var]

            for dim_name in ('pressure_level', 'isobaricInhPa', 'level'):
                if dim_name in da_u.dims:
                    da_u = da_u.isel({dim_name: 0}, drop=True)
                if dim_name in da_v.dims:
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

            # Remove 29-fev para compatibilidade com climatologias
            t = xr.DataArray(da_u['time'].values, dims=['time'])
            mask_feb = (~((t.dt.month == 2) & (t.dt.day == 29))).values
            da_u = da_u.isel(time=mask_feb)
            da_v = da_v.isel(time=mask_feb)
            if da_u.sizes.get('time', 0) == 0:
                continue

            spd_vals = np.sqrt(da_u.values ** 2 + da_v.values ** 2)
            da_spd = xr.DataArray(spd_vals, dims=da_u.dims, coords=da_u.coords)
            da_spd_daily = da_spd.resample(time='1D').mean(keep_attrs=True)
            valid = da_spd_daily.notnull().any(dim=['lat', 'lon'])
            da_spd_daily = da_spd_daily.isel(time=valid.values)
            n_days_file = da_spd_daily.sizes['time']
            if n_days_file == 0:
                continue

            if sum_spd is None:
                sum_spd = np.zeros(da_spd_daily.shape[1:], dtype=np.float64)
                count_spd = np.zeros(da_spd_daily.shape[1:], dtype=np.int64)
                ref_lat = da_spd_daily['lat'].values.copy()
                ref_lon = da_spd_daily['lon'].values.copy()
            elif da_spd_daily.shape[1:] != sum_spd.shape:
                ref_lat_da = xr.DataArray(ref_lat, dims=['lat'])
                ref_lon_da = xr.DataArray(ref_lon, dims=['lon'])
                da_spd_daily = da_spd_daily.interp(
                    lat=ref_lat_da, lon=ref_lon_da, method='linear'
                )
                LOGGER.info('Grade interpolada para referência (%dx%d).', len(ref_lat), len(ref_lon))

            vals = da_spd_daily.values
            sum_spd += np.nansum(vals, axis=0)
            count_spd += (~np.isnan(vals)).sum(axis=0)
            total_days += n_days_file
            LOGGER.info('Streaming: %s → %d dias (acumulado: %d)', fp.name, n_days_file, total_days)
        finally:
            ds.close()

    if sum_spd is None or total_days == 0:
        raise RuntimeError('Nenhum dado válido encontrado no período solicitado.')

    speed_mean = np.where(count_spd > 0, sum_spd / count_spd, np.nan).astype(np.float32)
    LOGGER.info(
        'Magnitude média: %d dias | speed=[%.1f, %.1f] m/s',
        total_days, float(np.nanmin(speed_mean)), float(np.nanmax(speed_mean)),
    )
    return speed_mean, ref_lat, ref_lon


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def main() -> None:
    """Download e calculo da magnitude media do vento 250 hPa."""
    def _to_datetime(val) -> datetime:
        if isinstance(val, datetime):
            return val
        if hasattr(val, 'year'):
            return datetime(val.year, val.month, val.day)
        return datetime.strptime(str(val), '%Y-%m-%d')

    dt_ini = _to_datetime(settings.DATA_INICIAL)
    dt_fim = _to_datetime(settings.DATA_FINAL)
    force = getattr(settings, 'FORCE_DOWNLOAD', False)

    ontem = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    if dt_fim >= datetime.now().replace(hour=0, minute=0, second=0, microsecond=0):
        LOGGER.warning(
            'DATA_FINAL (%s) é hoje ou futura. Última data considerada: %s',
            dt_fim.strftime('%Y-%m-%d'), ontem.strftime('%Y-%m-%d'),
        )
        dt_fim = ontem

    LOGGER.info('=' * 70)
    LOGGER.info('PLOT_WND_SPEED_250: Magnitude vento 250 hPa')
    LOGGER.info('Periodo: %s a %s', settings.DATA_INICIAL, dt_fim.strftime('%Y-%m-%d'))
    LOGGER.info('=' * 70)

    era5_period, gdas_period = _get_data_sources(dt_ini, dt_fim)

    all_files = []
    if era5_period:
        LOGGER.info('ERA5:  %s → %s', era5_period[0].date(), era5_period[1].date())
        all_files.extend(ensure_era5_uv250_for_period(
            start=era5_period[0],
            end=era5_period[1],
            hours_utc=list(DEFAULT_SYNOPTIC_HOURS),
            force_redownload=force,
        ))
    if gdas_period:
        LOGGER.info('GDAS:  %s → %s', gdas_period[0].date(), gdas_period[1].date())
        all_files.extend(ensure_gdas_uv250_for_period(
            start=gdas_period[0],
            end=gdas_period[1],
            force_redownload=force,
        ))

    LOGGER.info('Calculando magnitude média em streaming (u e v simultâneos)')
    speed_mean, ref_lat, ref_lon = _compute_period_mean_speed_streaming(
        all_files,
        required_hours=DEFAULT_SYNOPTIC_HOURS,
        dt_ini=dt_ini,
        dt_fim=dt_fim,
    )

    # Garante lat descendente (consistente com wnd_zonal_250_anom.nc)
    if ref_lat[0] < ref_lat[-1]:
        da_tmp = xr.DataArray(speed_mean, dims=['lat', 'lon'],
                              coords={'lat': ref_lat, 'lon': ref_lon})
        da_tmp = da_tmp.sortby('lat', ascending=False)
        speed_mean = da_tmp.values
        ref_lat = da_tmp['lat'].values

    DIR_DADOS_BASE.mkdir(parents=True, exist_ok=True)
    ds_out = xr.Dataset({
        'speed_mean': xr.DataArray(
            speed_mean,
            dims=['lat', 'lon'],
            coords={'lat': ref_lat, 'lon': ref_lon},
            attrs={'long_name': 'mean wind speed 250 hPa', 'units': 'm s-1'},
        )
    })
    output_path = DIR_DADOS_BASE / 'wnd_speed_250.nc'
    if output_path.exists():
        output_path.unlink()
    ds_out.to_netcdf(output_path)
    LOGGER.info('Salvo: %s', output_path)
