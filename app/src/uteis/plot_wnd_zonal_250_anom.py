# app/src/uteis/plot_wnd_zonal_250_anom.py
# -*- coding: utf-8 -*-
"""
Download e processamento de vento zonal 250 hPa (ERA5/GDAS) para anomalia de U250.

Pipeline:
1. Seleciona fonte por latência do ERA5 (~7 dias):
   - Período recente (últimos 7 dias): GDAS via NOMADS Grib Filter
   - Período mais antigo: ERA5 via Copernicus CDS (250 hPa)
   - Híbrido: ERA5 [ini→cutoff-1] + GDAS [cutoff→fim]
2. Processa um arquivo por vez (streaming) — sem carregar tudo na RAM
3. Carrega climatologia PSL/NOAA via Playwright (cache local por período MM-DD)
4. Calcula anomalia = u_médio_período - climatologia_u_PSL
5. Salva resultado em dados/wnd_zonal_250_anom.nc

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
from cartopy.util import add_cyclic_point as _acp

from app.src.uteis.clim_PSL_wnd_zonal_250 import get_clim_wnd_zonal_250_path
from app.src.uteis.downloaders_gdas_uv250 import ensure_gdas_uv250_for_period
from app.src.uteis.downloaders_wind250 import ensure_era5_uv250_for_period

try:
    from app.shared.settings_factory import settings
    DIR_DADOS_BASE = Path(settings.DIR_DADOS)
except Exception:
    DIR_DADOS_BASE = Path('dados')

LOGGER = logging.getLogger('PLOT_WND_ZONAL_250')
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)

DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)
ERA5_LATENCY_DAYS = 7
OUTPUT_FILE = DIR_DADOS_BASE / 'wnd_zonal_250_anom.nc'


# -----------------------------------------------------------------------------
# Utilitários (padrão do projeto)
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
        LOGGER.warning('Removendo %d timestamps duplicados.', ds.sizes['time'] - len(idx))
        ds = ds.isel(time=idx)
    return ds


def _find_u_var(ds: xr.Dataset) -> str:
    for vn in ('u', 'u_component_of_wind', 'U_GRD_L100', 'uwnd'):
        if vn in ds.data_vars:
            return vn
    raise KeyError(f'Variável u não encontrada. Disponíveis: {list(ds.data_vars)}')


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
# Streaming accumulator — apenas u-zonal
# -----------------------------------------------------------------------------
def _compute_period_mean_u_streaming(
    files: Sequence[Path],
    required_hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS,
    dt_ini: Optional[datetime] = None,
    dt_fim: Optional[datetime] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula média do período para u-zonal processando um arquivo por vez."""
    required_set = set(int(h) for h in required_hours)
    t_ini = np.datetime64(dt_ini.date()) if dt_ini else None
    t_fim = np.datetime64(dt_fim.date()) if dt_fim else None

    sum_u: Optional[np.ndarray] = None
    count_u: Optional[np.ndarray] = None
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
            da_u = ds[u_var]

            for dim_name in ('pressure_level', 'isobaricInhPa', 'level'):
                if dim_name in da_u.dims:
                    da_u = da_u.isel({dim_name: 0}, drop=True)

            t_idx = pd.DatetimeIndex(pd.to_datetime(da_u['time'].values))
            mask_h = np.array([h in required_set for h in t_idx.hour], dtype=bool)
            da_u = da_u.isel(time=mask_h)

            if da_u.sizes.get('time', 0) == 0:
                LOGGER.warning('Sem horas sinóticas válidas em: %s', fp.name)
                continue

            if t_ini is not None or t_fim is not None:
                da_u = da_u.sel(time=slice(t_ini, t_fim))
            if da_u.sizes.get('time', 0) == 0:
                LOGGER.warning('Fora do período solicitado: %s', fp.name)
                continue

            t = xr.DataArray(da_u['time'].values, dims=['time'])
            mask_feb = (~((t.dt.month == 2) & (t.dt.day == 29))).values
            da_u = da_u.isel(time=mask_feb)
            if da_u.sizes.get('time', 0) == 0:
                continue

            da_u_daily = da_u.resample(time='1D').mean(keep_attrs=True)
            valid = da_u_daily.notnull().any(dim=['lat', 'lon'])
            da_u_daily = da_u_daily.isel(time=valid.values)
            n_days_file = da_u_daily.sizes['time']
            if n_days_file == 0:
                continue

            if sum_u is None:
                sum_u = np.zeros(da_u_daily.shape[1:], dtype=np.float64)
                count_u = np.zeros(da_u_daily.shape[1:], dtype=np.int64)
                ref_lat = da_u_daily['lat'].values.copy()
                ref_lon = da_u_daily['lon'].values.copy()
            elif da_u_daily.shape[1:] != sum_u.shape:
                ref_lat_da = xr.DataArray(ref_lat, dims=['lat'])
                ref_lon_da = xr.DataArray(ref_lon, dims=['lon'])
                da_u_daily = da_u_daily.interp(lat=ref_lat_da, lon=ref_lon_da, method='linear')
                LOGGER.info('Grade interpolada para referência (%dx%d).', len(ref_lat), len(ref_lon))

            vals_u = da_u_daily.values
            sum_u += np.nansum(vals_u, axis=0)
            count_u += (~np.isnan(vals_u)).sum(axis=0)
            total_days += n_days_file
            LOGGER.info('Streaming: %s → %d dias (acumulado: %d)', fp.name, n_days_file, total_days)
        finally:
            ds.close()

    if sum_u is None or total_days == 0:
        raise RuntimeError('Nenhum dado válido encontrado no período solicitado.')

    u_mean = np.where(count_u > 0, sum_u / count_u, np.nan).astype(np.float32)
    LOGGER.info(
        'Média do período: %d dias | u=[%.2f, %.2f] m/s',
        total_days, float(np.nanmin(u_mean)), float(np.nanmax(u_mean)),
    )
    return u_mean, ref_lat, ref_lon


# -----------------------------------------------------------------------------
# Climatologia PSL — u-zonal 250mb
# -----------------------------------------------------------------------------
def _load_psl_clim_u(path: Path) -> xr.DataArray:
    if not path.exists():
        raise FileNotFoundError(f'Climatologia PSL u-zonal 250mb não encontrada: {path}')

    ds = xr.open_dataset(path, engine='netcdf4')
    ds = _normalize_latlon_names(ds)

    da = None
    for vname in ('uwnd', 'u', 'u_component_of_wind', 'uwnd.1'):
        if vname in ds.data_vars:
            da = ds[vname]
            break
    if da is None:
        da = next(iter(ds.data_vars.values()))
        LOGGER.warning('Usando variável %s como u-zonal da climatologia PSL.', da.name)

    if 'time' in da.dims:
        da = da.isel(time=0, drop=True)
    for dim_name in ('level', 'isobaricInhPa', 'pressure_level'):
        if dim_name in da.dims:
            da = da.isel({dim_name: 0}, drop=True)

    ds_tmp = _normalize_lon(da.to_dataset(name='u'))
    da = ds_tmp['u']
    LOGGER.info(
        'Climatologia PSL u-zonal 250mb: shape=%s | lon=[%.1f, %.1f]',
        da.shape, float(da.lon.min()), float(da.lon.max()),
    )
    return da


def _interp_psl_clim(clim_da: xr.DataArray, target_lat: np.ndarray, target_lon: np.ndarray) -> xr.DataArray:
    clim_vals_cyc, clim_lon_cyc = _acp(clim_da.values, coord=clim_da['lon'].values)
    clim_cyc = xr.DataArray(
        clim_vals_cyc,
        dims=clim_da.dims,
        coords={'lat': clim_da['lat'].values, 'lon': clim_lon_cyc},
    )
    return clim_cyc.interp(lat=target_lat, lon=target_lon, method='linear')


# -----------------------------------------------------------------------------
# Pipeline principal
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Download, processamento e cálculo de anomalia de vento zonal 250 hPa.

    Fontes:
    - ERA5 (CDS): períodos mais antigos que 7 dias, 250 hPa
    - GDAS (NOMADS): últimos 7 dias, 250mb
    - Climatologia: PSL via Playwright (cache local por período MM-DD)

    Resultado: dados/wnd_zonal_250_anom.nc com variável:
    - u_anom_mean: anomalia do vento zonal (m/s)
    """
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
            'DATA_FINAL (%s) é hoje ou futura. GDAS só tem dados completos até ontem. '
            'Última data considerada: %s',
            dt_fim.strftime('%Y-%m-%d'), ontem.strftime('%Y-%m-%d'),
        )
        dt_fim = ontem

    LOGGER.info('=' * 70)
    LOGGER.info('PLOT_WND_ZONAL_250: Anomalia vento zonal 250 hPa')
    LOGGER.info('Periodo: %s a %s', settings.DATA_INICIAL, dt_fim.strftime('%Y-%m-%d'))
    LOGGER.info('=' * 70)

    # 1. Fontes de dados
    era5_period, gdas_period = _get_data_sources(dt_ini, dt_fim)
    if era5_period:
        LOGGER.info('ERA5:  %s → %s', era5_period[0].date(), era5_period[1].date())
    if gdas_period:
        LOGGER.info('GDAS:  %s → %s', gdas_period[0].date(), gdas_period[1].date())

    # 2. Download
    all_files = []
    if era5_period:
        LOGGER.info('Etapa 2a: Download ERA5 u/v 250 hPa')
        era5_files = ensure_era5_uv250_for_period(
            start=era5_period[0],
            end=era5_period[1],
            hours_utc=list(DEFAULT_SYNOPTIC_HOURS),
            force_redownload=force,
        )
        all_files.extend(era5_files)

    if gdas_period:
        LOGGER.info('Etapa 2b: Download GDAS u/v 250mb (NOMADS)')
        gdas_files = ensure_gdas_uv250_for_period(
            start=gdas_period[0],
            end=gdas_period[1],
            force_redownload=force,
        )
        all_files.extend(gdas_files)

    # 3. Média streaming (u apenas)
    LOGGER.info('Etapa 3: Calculando média do período em streaming')
    u_mean, ref_lat, ref_lon = _compute_period_mean_u_streaming(
        all_files,
        required_hours=DEFAULT_SYNOPTIC_HOURS,
        dt_ini=dt_ini,
        dt_fim=dt_fim,
    )

    u_period_da = xr.DataArray(
        u_mean, dims=['lat', 'lon'],
        coords={'lat': ref_lat, 'lon': ref_lon},
    )

    # 4. Climatologia PSL
    LOGGER.info('Etapa 4: Climatologia PSL u-zonal 250mb')
    clim_u_path = get_clim_wnd_zonal_250_path(settings.DATA_INICIAL, settings.DATA_FINAL)
    clim_u_da = _load_psl_clim_u(clim_u_path)
    clim_u_regrid = _interp_psl_clim(clim_u_da, ref_lat, ref_lon)

    # 5. Anomalia
    LOGGER.info('Etapa 5: Calculando anomalia u-zonal')

    if ref_lat[0] < ref_lat[-1]:
        u_period_da = u_period_da.sortby('lat', ascending=False)
        ref_lat = u_period_da['lat'].values

    u_anom_da = u_period_da - clim_u_regrid
    LOGGER.info(
        'Anomalia u-zonal: min=%.2f, max=%.2f m/s',
        float(u_anom_da.min()), float(u_anom_da.max()),
    )

    # 6. Salvar
    LOGGER.info('Etapa 6: Salvando wnd_zonal_250_anom.nc')
    DIR_DADOS_BASE.mkdir(parents=True, exist_ok=True)

    ds_out = xr.Dataset({
        'u_anom_mean': xr.DataArray(
            u_anom_da.values.astype(np.float32),
            dims=['lat', 'lon'],
            coords={'lat': ref_lat, 'lon': ref_lon},
            attrs={'long_name': 'zonal wind anomaly 250 hPa', 'units': 'm s-1'},
        )
    })

    output_path = DIR_DADOS_BASE / 'wnd_zonal_250_anom.nc'
    if output_path.exists():
        output_path.unlink()
    ds_out.to_netcdf(output_path)
    LOGGER.info('Salvo: %s', output_path)
