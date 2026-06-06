# app/src/uteis/plot_tmp850_anom.py
# -*- coding: utf-8 -*-
"""
Download e processamento de temperatura 850 hPa (ERA5/GDAS) para anomalia.

Pipeline:
1. Seleciona fonte por latência do ERA5 (~7 dias): ERA5 (CDS) ou GDAS (NOMADS)
2. Calcula média do período em streaming (sem carregar tudo na RAM)
3. Baixa climatologia 1991-2020 de temperatura 850 hPa do PSL/NOAA (Playwright)
4. Calcula anomalia = media_periodo - climatologia
5. Salva resultado em dados/tmp850.nc (variável 'tmp', 1 timestep)

Chamado por: scripts/s10_fluxo_rossby_wave_tmp850.py
"""

from __future__ import annotations

# Bibliotecas padrão
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Bibliotecas de terceiros
import numpy as np
import pandas as pd
import xarray as xr

# Módulos locais
from app.src.uteis.clim_PSL_tmp850 import get_clim_tmp850_path
from app.src.uteis.downloaders_gdas_tmp850 import ensure_gdas_tmp850_for_period
from app.src.uteis.downloaders_tmp850_ERA5 import ensure_era5_t850_for_period

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
LOGGER = logging.getLogger('PLOT_TMP850')
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)

DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)
ERA5_LATENCY_DAYS = 7


# -----------------------------------------------------------------------------
# Utilitários
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
# Streaming accumulator
# -----------------------------------------------------------------------------
def _compute_period_mean_streaming(
    files: Sequence[Path],
    required_hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS,
    dt_ini: Optional[datetime] = None,
    dt_fim: Optional[datetime] = None,
) -> xr.DataArray:
    """Calcula média do período processando um arquivo por vez."""
    required_set = set(int(h) for h in required_hours)
    t_ini = np.datetime64(dt_ini.date()) if dt_ini else None
    t_fim = np.datetime64(dt_fim.date()) if dt_fim else None

    sum_2d: Optional[np.ndarray] = None
    count_2d: Optional[np.ndarray] = None
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

            tmp_var = next(
                (v for v in ('tmp', 't', 'temperature', 'air') if v in ds.data_vars),
                list(ds.data_vars)[0],
            )
            da = ds[tmp_var]

            for dim_name in ('pressure_level', 'isobaricInhPa', 'level'):
                if dim_name in da.dims:
                    da = da.isel({dim_name: 0}, drop=True)

            t_idx = pd.DatetimeIndex(pd.to_datetime(da['time'].values))
            mask_h = np.array([h in required_set for h in t_idx.hour], dtype=bool)
            da = da.isel(time=mask_h)

            if da.sizes.get('time', 0) == 0:
                LOGGER.warning('Sem horas sinóticas válidas em: %s', fp.name)
                continue

            if t_ini is not None or t_fim is not None:
                da = da.sel(time=slice(t_ini, t_fim))
            if da.sizes.get('time', 0) == 0:
                LOGGER.warning('Fora do período solicitado: %s', fp.name)
                continue

            t = xr.DataArray(da['time'].values, dims=['time'])
            da = da.isel(time=(~((t.dt.month == 2) & (t.dt.day == 29))).values)
            if da.sizes.get('time', 0) == 0:
                continue

            da_daily = da.resample(time='1D').mean(keep_attrs=True)
            valid = da_daily.notnull().any(dim=['lat', 'lon'])
            da_daily = da_daily.isel(time=valid.values)
            n_days_file = da_daily.sizes['time']
            if n_days_file == 0:
                continue

            vals = da_daily.values
            if sum_2d is None:
                sum_2d = np.zeros(vals.shape[1:], dtype=np.float64)
                count_2d = np.zeros(vals.shape[1:], dtype=np.int64)
                ref_lat = da_daily['lat'].values.copy()
                ref_lon = da_daily['lon'].values.copy()

            sum_2d += np.nansum(vals, axis=0)
            count_2d += (~np.isnan(vals)).sum(axis=0)
            total_days += n_days_file

            LOGGER.info('Streaming: %s → %d dias (acumulado: %d)', fp.name, n_days_file, total_days)
        finally:
            ds.close()

    if sum_2d is None or total_days == 0:
        raise RuntimeError('Nenhum dado válido de temperatura 850 hPa encontrado no período.')

    mean_2d = np.where(count_2d > 0, sum_2d / count_2d, np.nan).astype(np.float32)
    LOGGER.info(
        'Média do período: %d dias | min=%.2f max=%.2f K',
        total_days, float(np.nanmin(mean_2d)), float(np.nanmax(mean_2d)),
    )

    return xr.DataArray(
        mean_2d,
        dims=['lat', 'lon'],
        coords={'lat': ref_lat, 'lon': ref_lon},
        attrs={'long_name': 'air temperature at 850 hPa', 'units': 'K'},
        name='tmp',
    )


# -----------------------------------------------------------------------------
# Carrega climatologia PSL
# -----------------------------------------------------------------------------
def _load_climatology_psl(path: Path) -> xr.DataArray:
    if not path.exists():
        raise FileNotFoundError(f'Climatologia PSL tmp850 não encontrada: {path}')
    ds = xr.open_dataset(path, engine='netcdf4')
    ds = _normalize_latlon_names(ds)

    # PSL/NCEP reanalysis: temperatura = variável 'air'
    tmp_var = next(
        (v for v in ('air', 'tmp', 't', 'temperature') if v in ds.data_vars),
        list(ds.data_vars)[0],
    )
    da = ds[tmp_var]
    if 'time' in da.dims:
        da = da.isel(time=0, drop=True)
    if 'level' in da.dims:
        da = da.isel(level=0, drop=True)
    LOGGER.info(
        'Climatologia PSL tmp850 carregada: %s | shape=%s | lon=[%.1f, %.1f]',
        path.name, da.shape, float(da.lon.min()), float(da.lon.max()),
    )
    return da


# -----------------------------------------------------------------------------
# Pipeline principal
# -----------------------------------------------------------------------------
def main() -> None:
    """Download, processamento e cálculo de anomalia de temperatura 850 hPa.

    Resultado: dados/tmp850.nc com variável 'tmp' (anomalia em K/°C, 1 timestep).
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
            'DATA_FINAL (%s) é hoje ou futura. Última data considerada: %s',
            dt_fim.strftime('%Y-%m-%d'), ontem.strftime('%Y-%m-%d'),
        )
        dt_fim = ontem

    LOGGER.info('=' * 70)
    LOGGER.info('PLOT_TMP850: Download e anomalia temperatura 850 hPa')
    LOGGER.info('Periodo: %s a %s', settings.DATA_INICIAL, dt_fim.strftime('%Y-%m-%d'))
    LOGGER.info('=' * 70)

    era5_period, gdas_period = _get_data_sources(dt_ini, dt_fim)
    if era5_period:
        LOGGER.info('ERA5:  %s → %s', era5_period[0].date(), era5_period[1].date())
    if gdas_period:
        LOGGER.info('GDAS:  %s → %s', gdas_period[0].date(), gdas_period[1].date())

    all_files: List[Path] = []

    if era5_period:
        LOGGER.info('Etapa 2a: Download ERA5 temperatura 850 hPa')
        era5_files = ensure_era5_t850_for_period(
            start=era5_period[0],
            end=era5_period[1],
            hours_utc=list(DEFAULT_SYNOPTIC_HOURS),
            force_redownload=force,
        )
        all_files.extend(era5_files)

    if gdas_period:
        LOGGER.info('Etapa 2b: Download GDAS temperatura 850 hPa (NOMADS)')
        gdas_files = ensure_gdas_tmp850_for_period(
            start=gdas_period[0],
            end=gdas_period[1],
            force_redownload=force,
        )
        all_files.extend(gdas_files)

    LOGGER.info('Etapa 3: Calculando média do período em streaming')
    tmp_period_mean = _compute_period_mean_streaming(
        all_files,
        required_hours=DEFAULT_SYNOPTIC_HOURS,
        dt_ini=dt_ini,
        dt_fim=dt_fim,
    )

    LOGGER.info('Etapa 4: Climatologia PSL temperatura 850 hPa')
    clim_path = get_clim_tmp850_path(settings.DATA_INICIAL, settings.DATA_FINAL)
    clim_da = _load_climatology_psl(clim_path)
    clim_da = _normalize_lon(clim_da.to_dataset(name='tmp'))['tmp']

    from cartopy.util import add_cyclic_point as _acp
    clim_vals_cyc, clim_lon_cyc = _acp(clim_da.values, coord=clim_da['lon'].values)
    clim_da = xr.DataArray(
        clim_vals_cyc,
        dims=clim_da.dims,
        coords={'lat': clim_da['lat'].values, 'lon': clim_lon_cyc},
    )

    clim_regrid = clim_da.interp(
        lat=tmp_period_mean.lat, lon=tmp_period_mean.lon, method='linear'
    )

    LOGGER.info('Etapa 5: Calculando anomalia')
    anomaly = tmp_period_mean - clim_regrid
    anomaly.name = 'tmp'
    anomaly.attrs['long_name'] = 'air temperature anomaly at 850 hPa'
    anomaly.attrs['units'] = 'K'

    LOGGER.info(
        'Anomalia calculada: min=%.2f, max=%.2f, mean=%.2f K',
        float(anomaly.min()), float(anomaly.max()), float(anomaly.mean()),
    )

    anomaly_ds = anomaly.expand_dims('time').to_dataset(name='tmp')
    anomaly_ds['time'] = [pd.Timestamp(settings.DATA_FINAL)]

    output_path = DIR_DADOS_BASE / 'tmp850.nc'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    anomaly_ds.to_netcdf(output_path, engine='netcdf4')
    LOGGER.info('Anomalia salva em: %s', output_path)
    LOGGER.info('=' * 70)
    LOGGER.info('PLOT_TMP850: Concluido com sucesso')
    LOGGER.info('=' * 70)


if __name__ == '__main__':
    main()
