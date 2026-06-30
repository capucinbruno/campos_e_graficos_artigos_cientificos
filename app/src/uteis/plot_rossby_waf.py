# app/src/uteis/plot_rossby_waf.py
# -*- coding: utf-8 -*-
"""
Download e processamento de hgt/uv 250 hPa (ERA5/GDAS) para Rossby Wave Activity Flux.

Pipeline:
1. Seleciona fonte de dados por latência do ERA5 (~7 dias):
   - Período recente (últimos 7 dias): GDAS via NOMADS Grib Filter
   - Período mais antigo: ERA5 via Copernicus CDS (250 hPa)
   - Híbrido: ERA5 [ini→cutoff-1] + GDAS [cutoff→fim]
2. Processa hgt um arquivo por vez (streaming) — sem carregar tudo na RAM
3. Carrega climatologia PSL geopotencial 250mb (cache local por MM-DD)
4. Carrega climatologia PSL u/v 250mb (cache local por MM-DD)
5. Calcula anomalia hgt = período - climatologia PSL
6. Regrida tudo para 2.5° e calcula WAF via tnflux (Takaya & Nakamura 2001)
7. Salva resultado em dados/rossby_waf.nc

Entradas do tnflux.tnf2d:
    - u_clim, v_clim: vento climatológico médio do período (m/s) a 250 hPa
    - phi_clim: geopotencial climatológico (m2/s2) = hgt_clim * g
    - phi_obs: geopotencial observado médio do período (m2/s2) = hgt_mean * g
    - lat, lon, pressure_level (hPa)

Chamado por: scripts/s04_fluxo_rossby_wave.py
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
from cartopy.util import add_cyclic_point as _acp

import tnflux

# Módulos locais
from app.src.uteis.clim_PSL_geop250 import get_clim_geop250_path
from app.src.uteis.clim_PSL_wnd250 import get_clim_wnd250_paths
from app.src.uteis.downloaders_gdas_hgt250 import ensure_gdas_hgt250_for_period
from app.src.uteis.downloaders_hgt250_ERA5 import (
    ensure_era5_altura_geopotencial_250_global_for_period_grib,
)

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
LOGGER = logging.getLogger('PLOT_ROSSBY_WAF')
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Constantes
# -----------------------------------------------------------------------------
DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)
G = 9.80665
TROPICAL_MASK_LAT = 15.0   # máscara |lat| < 15° (singularidade equatorial TN2001)
POLAR_MASK_LAT = 75.0      # máscara |lat| > 75° (singularidade polar 1/cos²φ TN2001)
WAF_GRID_SPACING = 2.5

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
        ds = ds.isel(time=idx)
    return ds


# -----------------------------------------------------------------------------
# Seleção de fonte de dados (ERA5 vs GDAS)
# -----------------------------------------------------------------------------
def _get_data_sources(
    dt_ini: datetime,
    dt_fim: datetime,
) -> Tuple[Optional[Tuple[datetime, datetime]], Optional[Tuple[datetime, datetime]]]:
    """Retorna (periodo_era5, periodo_gdas) com base na latência do ERA5."""
    cutoff = (datetime.now() - timedelta(days=ERA5_LATENCY_DAYS)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    if dt_fim < cutoff:
        return (dt_ini, dt_fim), None
    if dt_ini >= cutoff:
        return None, (dt_ini, dt_fim)
    return (dt_ini, cutoff - timedelta(days=1)), (cutoff, dt_fim)


# -----------------------------------------------------------------------------
# Streaming accumulator para hgt (um arquivo por vez)
# -----------------------------------------------------------------------------
def _compute_period_mean_streaming_hgt(
    files: Sequence[Path],
    required_hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS,
    dt_ini: Optional[datetime] = None,
    dt_fim: Optional[datetime] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula média do período para hgt processando um arquivo por vez.

    Mantém no máximo 1 arquivo na RAM por vez.
    Retorna (hgt_mean_2d, ref_lat, ref_lon).
    """
    required_set = set(int(h) for h in required_hours)
    t_ini = np.datetime64(dt_ini.date()) if dt_ini else None
    t_fim = np.datetime64(dt_fim.date()) if dt_fim else None

    sum_2d: Optional[np.ndarray] = None
    count_2d: Optional[np.ndarray] = None
    ref_lat: Optional[np.ndarray] = None
    ref_lon: Optional[np.ndarray] = None
    total_days = 0

    for fp in files:
        LOGGER.info('Streaming hgt: abrindo %s', fp.name)
        ds = xr.open_dataset(fp, engine='netcdf4')
        try:
            ds = _ensure_time_coord(ds)
            ds = _drop_or_collapse_expver(ds)
            ds = _normalize_latlon_names(ds)
            ds = _normalize_lon(ds)
            ds = _sort_and_dedup_time(ds)

            hgt_var = next(
                (v for v in ('hgt', 'z', 'geopotential') if v in ds.data_vars),
                list(ds.data_vars)[0],
            )
            da = ds[hgt_var]

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

            if sum_2d is None:
                sum_2d = np.zeros(da_daily.shape[1:], dtype=np.float64)
                count_2d = np.zeros(da_daily.shape[1:], dtype=np.int64)
                ref_lat = da_daily['lat'].values.copy()
                ref_lon = da_daily['lon'].values.copy()
            elif da_daily.shape[1:] != sum_2d.shape:
                ref_lat_da = xr.DataArray(ref_lat, dims=['lat'])
                ref_lon_da = xr.DataArray(ref_lon, dims=['lon'])
                da_daily = da_daily.interp(lat=ref_lat_da, lon=ref_lon_da, method='linear')
                LOGGER.info('Grade hgt interpolada para referência (%dx%d).', len(ref_lat), len(ref_lon))

            vals = da_daily.values
            sum_2d += np.nansum(vals, axis=0)
            count_2d += (~np.isnan(vals)).sum(axis=0)
            total_days += n_days_file

            LOGGER.info('Streaming hgt: %s → %d dias (acumulado: %d)', fp.name, n_days_file, total_days)
        finally:
            ds.close()

    if sum_2d is None or total_days == 0:
        raise RuntimeError('Nenhum dado hgt válido encontrado no período solicitado.')

    hgt_mean = np.where(count_2d > 0, sum_2d / count_2d, np.nan).astype(np.float32)
    LOGGER.info(
        'hgt médio do período: %d dias | min=%.1f, max=%.1f m',
        total_days, float(np.nanmin(hgt_mean)), float(np.nanmax(hgt_mean)),
    )
    return hgt_mean, ref_lat, ref_lon


# -----------------------------------------------------------------------------
# Carregamento de climatologias PSL
# -----------------------------------------------------------------------------
def _load_psl_clim_hgt(path: Path) -> xr.DataArray:
    """Carrega campo 2D da climatologia PSL hgt 250mb."""
    if not path.exists():
        raise FileNotFoundError(f'Climatologia PSL hgt250 não encontrada: {path}')

    ds = xr.open_dataset(path, engine='netcdf4')
    ds = _normalize_latlon_names(ds)

    for vname in ('hgt', 'z', 'geopotential', 'gh', 'geopotential_height'):
        if vname in ds.data_vars:
            da = ds[vname]
            units = da.attrs.get('units', '')
            if 'm**2' in units or 'm2' in units or 'J' in units:
                da = da / G
                da.attrs['units'] = 'm'
            if 'time' in da.dims:
                da = da.isel(time=0, drop=True)
            ds_tmp = _normalize_lon(da.to_dataset(name='hgt'))
            return ds_tmp['hgt']

    da = next(iter(ds.data_vars.values()))
    if 'time' in da.dims:
        da = da.isel(time=0, drop=True)
    ds_tmp = _normalize_lon(da.to_dataset(name='hgt'))
    LOGGER.warning('Usando variável %s como hgt da climatologia PSL.', da.name)
    return ds_tmp['hgt']


def _load_psl_clim_wind_component(path: Path, component: str) -> xr.DataArray:
    """Carrega campo 2D da climatologia PSL u ou v 250mb."""
    if not path.exists():
        raise FileNotFoundError(f'Climatologia PSL {component}250 não encontrada: {path}')

    ds = xr.open_dataset(path, engine='netcdf4')
    ds = _normalize_latlon_names(ds)

    candidates = ('uwnd', 'u', 'u_component_of_wind') if component == 'u' else ('vwnd', 'v', 'v_component_of_wind')
    da = None
    for vname in candidates:
        if vname in ds.data_vars:
            da = ds[vname]
            break

    if da is None:
        da = next(iter(ds.data_vars.values()))
        LOGGER.warning('Usando variável %s como %s-wind da climatologia PSL.', da.name, component)

    if 'time' in da.dims:
        da = da.isel(time=0, drop=True)
    ds_tmp = _normalize_lon(da.to_dataset(name=component))
    return ds_tmp[component]


def _interp_psl_to_grid(clim_da: xr.DataArray, target_lat: np.ndarray, target_lon: np.ndarray) -> xr.DataArray:
    """Adiciona cyclic point e interpola climatologia PSL (2.5°) para o grid alvo."""
    clim_vals_cyc, clim_lon_cyc = _acp(clim_da.values, coord=clim_da['lon'].values)
    clim_cyc = xr.DataArray(
        clim_vals_cyc,
        dims=clim_da.dims,
        coords={'lat': clim_da['lat'].values, 'lon': clim_lon_cyc},
    )
    return clim_cyc.interp(lat=target_lat, lon=target_lon, method='linear')


def _to_180(da: xr.DataArray) -> xr.DataArray:
    """Converte coords de lon 0..360 para -180..180 ordenado."""
    if np.any(da['lon'].values > 180):
        da = da.assign_coords(lon=(((da['lon'].values + 180) % 360) - 180)).sortby('lon')
    return da


def waf_from_means(hgt_mean_da, hgt_clim_da, u_clim, v_clim, lat, lon, pressure: float = 250.0):
    """WAF (Takaya & Nakamura 2001) a partir de medias ja calculadas.

    Versao reutilizavel (usada pelo s16 no modo previsao por janela): recebe a media do periodo
    e a climatologia ja calculadas (mesma grade lat/lon), regrida para a grade WAF de 2.5°,
    aplica as mascaras equatorial/polar e chama `tnflux.tnf2d` no nivel `pressure` (hPa).

    Parametros:
        hgt_mean_da, hgt_clim_da: DataArray (lat, lon) de altura geopotencial (m) — periodo e clim.
        u_clim, v_clim: arrays (lat, lon) do vento climatologico (m/s) do nivel.
        lat, lon: eixos da grade de entrada.
        pressure: nivel em hPa (250 para o s16; 200 no s34).

    Retorna (hgt_anom_da[grade entrada], px, py, lat_waf, lon_waf[grade WAF 2.5°]).
    """
    hgt_anom = hgt_mean_da - hgt_clim_da
    lat_waf = np.arange(90, -90 - WAF_GRID_SPACING / 2, -WAF_GRID_SPACING)
    lon_waf = np.arange(-180, 180, WAF_GRID_SPACING)

    def _to_waf(arr2d):
        da = _to_180(xr.DataArray(arr2d, dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}))
        return da.interp(lat=lat_waf, lon=lon_waf, method='linear').values

    phi_obs = _to_waf(hgt_mean_da.values) * G
    phi_clim = _to_waf(hgt_clim_da.values) * G
    u_c, v_c = _to_waf(u_clim), _to_waf(v_clim)
    mask = (np.abs(lat_waf) < TROPICAL_MASK_LAT) | (np.abs(lat_waf) > POLAR_MASK_LAT)
    for a in (phi_obs, phi_clim, u_c, v_c):
        a[mask, :] = np.nan
    px, py = tnflux.tnf2d(u_c, v_c, phi_clim, phi_obs, lat_waf, lon_waf, float(pressure))
    px[mask, :] = np.nan
    py[mask, :] = np.nan
    return hgt_anom, px, py, lat_waf, lon_waf


# -----------------------------------------------------------------------------
# Pipeline principal
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Download, processamento e cálculo do Rossby WAF.

    Fontes de dados selecionadas automaticamente:
    - ERA5 (CDS): períodos mais antigos que 7 dias, hgt 250 hPa
    - GDAS (NOMADS): últimos 7 dias, hgt 250mb
    - Climatologias: PSL geopotencial + u/v 250mb (cache local por MM-DD)

    Resultado: dados/rossby_waf.nc com variáveis:
    - hgt_anom_mean: anomalia de altura geopotencial 250 hPa (m)
    - waf_x: componente zonal do WAF (m2/s2)
    - waf_y: componente meridional do WAF (m2/s2)
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
    LOGGER.info('PLOT_ROSSBY_WAF: Download e cálculo do Wave Activity Flux 250 hPa')
    LOGGER.info('Periodo: %s a %s', settings.DATA_INICIAL, dt_fim.strftime('%Y-%m-%d'))
    LOGGER.info('=' * 70)

    # 1. Determinar fontes de dados
    era5_period, gdas_period = _get_data_sources(dt_ini, dt_fim)
    if era5_period:
        LOGGER.info('ERA5:  %s → %s', era5_period[0].date(), era5_period[1].date())
    if gdas_period:
        LOGGER.info('GDAS:  %s → %s', gdas_period[0].date(), gdas_period[1].date())

    # 2. Download dos dados hgt 250 hPa
    all_files = []

    if era5_period:
        LOGGER.info('Etapa 2a: Download ERA5 hgt 250 hPa')
        era5_files = ensure_era5_altura_geopotencial_250_global_for_period_grib(
            start=era5_period[0],
            end=era5_period[1],
            hours_utc=list(DEFAULT_SYNOPTIC_HOURS),
            force_redownload=force,
            convert_to_height_netcdf=True,
        )
        all_files.extend(era5_files)

    if gdas_period:
        LOGGER.info('Etapa 2b: Download GDAS hgt 250mb (NOMADS)')
        gdas_files = ensure_gdas_hgt250_for_period(
            start=gdas_period[0],
            end=gdas_period[1],
            force_redownload=force,
        )
        all_files.extend(gdas_files)

    # 3. Média do período em streaming (sem carregar tudo na RAM)
    LOGGER.info('Etapa 3: Calculando média hgt do período em streaming')
    hgt_mean, ref_lat, ref_lon = _compute_period_mean_streaming_hgt(
        all_files,
        required_hours=DEFAULT_SYNOPTIC_HOURS,
        dt_ini=dt_ini,
        dt_fim=dt_fim,
    )

    # 4. Climatologia PSL hgt 250mb
    LOGGER.info('Etapa 4: Climatologia PSL hgt 250mb')
    clim_hgt_path = get_clim_geop250_path(settings.DATA_INICIAL, settings.DATA_FINAL)
    clim_hgt_da = _load_psl_clim_hgt(clim_hgt_path)
    clim_hgt_regrid = _interp_psl_to_grid(clim_hgt_da, ref_lat, ref_lon)

    # 5. Climatologia PSL u/v 250mb
    LOGGER.info('Etapa 5: Climatologia PSL u/v 250mb')
    clim_u_path, clim_v_path = get_clim_wnd250_paths(settings.DATA_INICIAL, settings.DATA_FINAL)
    clim_u_da = _load_psl_clim_wind_component(clim_u_path, 'u')
    clim_v_da = _load_psl_clim_wind_component(clim_v_path, 'v')
    clim_u_regrid = _interp_psl_to_grid(clim_u_da, ref_lat, ref_lon)
    clim_v_regrid = _interp_psl_to_grid(clim_v_da, ref_lat, ref_lon)

    # 6. Anomalia hgt
    LOGGER.info('Etapa 6: Calculando anomalia hgt')
    hgt_mean_da = xr.DataArray(hgt_mean, dims=['lat', 'lon'], coords={'lat': ref_lat, 'lon': ref_lon})
    hgt_anom_da = hgt_mean_da - clim_hgt_regrid

    LOGGER.info(
        'Anomalia hgt: min=%.1f, max=%.1f m',
        float(hgt_anom_da.min()), float(hgt_anom_da.max()),
    )

    # 7. Regrida para 2.5° e calcula WAF
    LOGGER.info('Etapa 7: Calculando TNFLUX (Takaya & Nakamura 2001)')
    lat_waf = np.arange(90, -90 - WAF_GRID_SPACING / 2, -WAF_GRID_SPACING)
    lon_waf = np.arange(-180, 180, WAF_GRID_SPACING)

    LOGGER.info(
        'Regridando para %.1f° (%d x %d) para cálculo do WAF',
        WAF_GRID_SPACING, len(lat_waf), len(lon_waf),
    )

    hgt_mean_waf = hgt_mean_da.interp(lat=lat_waf, lon=lon_waf, method='linear')
    clim_hgt_waf = clim_hgt_regrid.interp(lat=lat_waf, lon=lon_waf, method='linear')
    u_clim_waf = clim_u_regrid.interp(lat=lat_waf, lon=lon_waf, method='linear')
    v_clim_waf = clim_v_regrid.interp(lat=lat_waf, lon=lon_waf, method='linear')

    phi_obs = (hgt_mean_waf * G).values
    phi_clim = (clim_hgt_waf * G).values
    u_c = u_clim_waf.values
    v_c = v_clim_waf.values

    # Máscara faixa tropical |lat| < 15° e pólos |lat| > 75°
    # (TN2001 tem fator 1/cos²φ que explode próximo aos pólos)
    mask_eq = np.abs(lat_waf) < TROPICAL_MASK_LAT
    mask_poles = np.abs(lat_waf) > POLAR_MASK_LAT
    mask = mask_eq | mask_poles
    if np.any(mask):
        phi_obs[mask, :] = np.nan
        phi_clim[mask, :] = np.nan
        u_c[mask, :] = np.nan
        v_c[mask, :] = np.nan

    px, py = tnflux.tnf2d(u_c, v_c, phi_clim, phi_obs, lat_waf, lon_waf, 250.0)

    if np.any(mask):
        px[mask, :] = np.nan
        py[mask, :] = np.nan

    LOGGER.info(
        'WAF calculado: px min=%.2e max=%.2e | py min=%.2e max=%.2e',
        float(np.nanmin(px)), float(np.nanmax(px)),
        float(np.nanmin(py)), float(np.nanmax(py)),
    )

    # 8. Salvar rossby_waf.nc
    LOGGER.info('Etapa 8: Salvando rossby_waf.nc')

    ds_out = xr.Dataset({
        'hgt_anom_mean': xr.DataArray(
            hgt_anom_da.values,
            dims=['lat', 'lon'],
            coords={'lat': ref_lat, 'lon': ref_lon},
            attrs={'long_name': 'geopotential height anomaly 250 hPa', 'units': 'm'},
        ),
        'waf_x': xr.DataArray(
            px,
            dims=['lat_waf', 'lon_waf'],
            coords={'lat_waf': lat_waf, 'lon_waf': lon_waf},
            attrs={'long_name': 'Wave Activity Flux zonal (TN2001)', 'units': 'm2 s-2'},
        ),
        'waf_y': xr.DataArray(
            py,
            dims=['lat_waf', 'lon_waf'],
            coords={'lat_waf': lat_waf, 'lon_waf': lon_waf},
            attrs={'long_name': 'Wave Activity Flux meridional (TN2001)', 'units': 'm2 s-2'},
        ),
    })

    output_path = DIR_DADOS_BASE / 'rossby_waf.nc'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    ds_out.to_netcdf(output_path, engine='netcdf4')
    LOGGER.info('Rossby WAF salvo em: %s', output_path)
    LOGGER.info('=' * 70)
    LOGGER.info('PLOT_ROSSBY_WAF: Concluído com sucesso')
    LOGGER.info('=' * 70)


if __name__ == '__main__':
    main()
