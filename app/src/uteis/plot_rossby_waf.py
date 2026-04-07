# app/src/uteis/plot_rossby_waf.py
# -*- coding: utf-8 -*-
"""
Download e processamento de hgt 250 hPa (ERA5) para Rossby Wave Activity Flux.

Pipeline:
1. Baixa dados ERA5 de geopotencial (z) em 250 hPa via CDS
2. Converte geopotencial (m2/s2) para altura geopotencial (m)
3. Concatena arquivos mensais e calcula media diaria
4. Carrega climatologias de hgt/uwnd/vwnd 250 hPa
5. Calcula medias do periodo e climatologicas (mesmos dias)
6. Converte hgt (m) -> geopotencial (m2/s2) para entrada do tnflux
7. Mascara faixa tropical |lat| < 15 graus
8. Calcula WAF com tnflux.tnf2d (Takaya & Nakamura 2001)
9. Salva resultado em dados/rossby_waf.nc

Entradas do tnflux.tnf2d:
    - u_clim, v_clim: vento climatologico medio do periodo (m/s)
    - phi_clim: geopotencial climatologico (m2/s2) = hgt_clim * g
    - phi_obs: geopotencial observado medio do periodo (m2/s2) = hgt_mean * g
    - lat, lon, pressure_level (hPa)

Chamado por: scripts/s04_fluxo_rossby_wave.py
"""

from __future__ import annotations

# Bibliotecas padrao
import logging
from datetime import datetime
from pathlib import Path

# Bibliotecas de terceiros
import numpy as np
import xarray as xr

import tnflux

# Modulos locais — reutiliza funcoes do pipeline de geop250 e psi200
from app.src.uteis.downloaders_hgt250_ERA5 import (
    ensure_era5_altura_geopotencial_250_global_for_period_grib,
)
from app.src.uteis.plot_geop250 import (
    _compute_daily_mean as _compute_daily_mean_hgt,
    _drop_feb29 as _drop_feb29_hgt,
    _load_climatology_geop250,
    _normalize_latlon_names as _normalize_latlon_names_hgt,
    _normalize_lon as _normalize_lon_hgt,
    _open_and_merge_hgt_files,
    _select_climatology_same_days as _select_climatology_same_days_hgt,
)
from app.src.uteis.plot_psi200 import (
    _add_cyclic_and_interp_clim,
    _load_wind_climatology,
    _normalize_latlon_names,
    _normalize_lon,
    _select_climatology_same_days,
)

# -----------------------------------------------------------------------------
# Integracao com settings
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
G = 9.80665  # gravidade (m/s2)
TROPICAL_MASK_LAT = 15.0  # graus — mascara faixa |lat| < 15


# -----------------------------------------------------------------------------
# Pipeline principal
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Download, processamento e calculo do Rossby WAF.

    Resultado: dados/rossby_waf.nc com variaveis:
    - hgt_anom_mean: anomalia de altura geopotencial 250 hPa (m)
    - waf_x: componente zonal do WAF (m2/s2)
    - waf_y: componente meridional do WAF (m2/s2)
    """
    dt_ini = datetime.strptime(settings.DATA_INICIAL, '%Y-%m-%d')
    dt_fim = datetime.strptime(settings.DATA_FINAL, '%Y-%m-%d')

    LOGGER.info('=' * 70)
    LOGGER.info('PLOT_ROSSBY_WAF: Download e calculo do Wave Activity Flux 250 hPa')
    LOGGER.info('Periodo: %s a %s', settings.DATA_INICIAL, settings.DATA_FINAL)
    LOGGER.info('=' * 70)

    # =========================================================================
    # 1. Download ERA5 hgt 250 hPa (z -> hgt em metros)
    # =========================================================================
    LOGGER.info('Etapa 1: Download ERA5 geopotencial 250 hPa e conversao z -> hgt')
    hgt_files = ensure_era5_altura_geopotencial_250_global_for_period_grib(
        start=dt_ini,
        end=dt_fim,
        hours_utc=list(DEFAULT_SYNOPTIC_HOURS),
        force_redownload=getattr(settings, 'FORCE_DOWNLOAD', False),
        convert_to_height_netcdf=True,
    )
    LOGGER.info('Arquivos de altura geopotencial: %d', len(hgt_files))

    # =========================================================================
    # 2. Abrir, concatenar e media diaria
    # =========================================================================
    LOGGER.info('Etapa 2: Concatenando arquivos e calculando media diaria')
    ds_hgt = _open_and_merge_hgt_files(hgt_files)
    ds_hgt = _normalize_latlon_names_hgt(ds_hgt)
    ds_hgt = _normalize_lon_hgt(ds_hgt)

    ds_daily = _compute_daily_mean_hgt(ds_hgt)
    ds_daily = _drop_feb29_hgt(ds_daily)

    # Recortar ao periodo solicitado (arquivo mensal pode conter dias extras)
    ds_daily = ds_daily.sel(time=slice(np.datetime64(dt_ini.date()), np.datetime64(dt_fim.date())))
    LOGGER.info('Periodo recortado: %d dias', ds_daily.sizes.get('time', 0))

    # Identificar variavel de altura
    hgt_var = None
    for vname in ('hgt', 'z', 'geopotential'):
        if vname in ds_daily.data_vars:
            hgt_var = vname
            break
    if hgt_var is None:
        hgt_var = list(ds_daily.data_vars)[0]

    hgt_daily = ds_daily[hgt_var]

    for dim_name in ('pressure_level', 'isobaricInhPa', 'level'):
        if dim_name in hgt_daily.dims:
            hgt_daily = hgt_daily.isel({dim_name: 0}, drop=True)

    # =========================================================================
    # 3. Climatologia hgt 250 hPa
    # =========================================================================
    LOGGER.info('Etapa 3: Carregando climatologia hgt 250 hPa')
    clim_hgt_path = Path(settings.FILE_CLIMATOLOGIA_GEOP250)
    clim_hgt_da = _load_climatology_geop250(clim_hgt_path)

    if 'level' in clim_hgt_da.dims:
        clim_hgt_da = clim_hgt_da.isel(level=0, drop=True)
    elif 'level' in clim_hgt_da.coords:
        clim_hgt_da = clim_hgt_da.drop_vars('level')

    ds_clim_hgt = _normalize_latlon_names_hgt(clim_hgt_da.to_dataset(name=clim_hgt_da.name or 'hgt_clim'))
    ds_clim_hgt = _normalize_lon_hgt(ds_clim_hgt)
    clim_hgt_da = list(ds_clim_hgt.data_vars.values())[0]

    clim_hgt_period = _select_climatology_same_days_hgt(clim_hgt_da, hgt_daily['time'])

    # Cyclic point + interpolacao para grid ERA5
    if 'lon' in clim_hgt_period.coords:
        from cartopy.util import add_cyclic_point as _acp

        clim_vals, clim_lon = _acp(clim_hgt_period.values, coord=clim_hgt_period['lon'].values)
        clim_hgt_period = xr.DataArray(
            clim_vals,
            dims=clim_hgt_period.dims,
            coords={d: clim_hgt_period.coords[d] for d in clim_hgt_period.dims if d != 'lon'},
        )
        clim_hgt_period = clim_hgt_period.assign_coords(lon=('lon', clim_lon))

    if (
        hgt_daily.sizes.get('lat') != clim_hgt_period.sizes.get('lat')
        or hgt_daily.sizes.get('lon') != clim_hgt_period.sizes.get('lon')
        or not np.array_equal(hgt_daily['lat'].values, clim_hgt_period['lat'].values)
        or not np.array_equal(hgt_daily['lon'].values, clim_hgt_period['lon'].values)
    ):
        LOGGER.warning('Grid da climatologia hgt difere do ERA5. Interpolando.')
        clim_hgt_period = clim_hgt_period.interp_like(hgt_daily)

    # =========================================================================
    # 4. Climatologias uwnd/vwnd 250 hPa
    # =========================================================================
    LOGGER.info('Etapa 4: Carregando climatologias uwnd/vwnd 250 hPa')

    u_clim_path = Path(settings.FILE_CLIMATOLOGIA_UWND250)
    v_clim_path = Path(settings.FILE_CLIMATOLOGIA_VWND250)

    u_clim_da = _load_wind_climatology(u_clim_path, 'u')
    v_clim_da = _load_wind_climatology(v_clim_path, 'v')

    if 'level' in u_clim_da.dims:
        u_clim_da = u_clim_da.isel(level=0, drop=True)
    if 'level' in v_clim_da.dims:
        v_clim_da = v_clim_da.isel(level=0, drop=True)

    for name, clim in [('u_clim', u_clim_da), ('v_clim', v_clim_da)]:
        ds_tmp = _normalize_latlon_names(clim.to_dataset(name=clim.name or name))
        ds_tmp = _normalize_lon(ds_tmp)
        if name == 'u_clim':
            u_clim_da = list(ds_tmp.data_vars.values())[0]
        else:
            v_clim_da = list(ds_tmp.data_vars.values())[0]

    u_clim_period = _select_climatology_same_days(u_clim_da, hgt_daily['time'])
    v_clim_period = _select_climatology_same_days(v_clim_da, hgt_daily['time'])

    u_clim_period = _add_cyclic_and_interp_clim(u_clim_period, hgt_daily)
    v_clim_period = _add_cyclic_and_interp_clim(v_clim_period, hgt_daily)

    # =========================================================================
    # 5. Medias do periodo
    # =========================================================================
    LOGGER.info('Etapa 5: Calculando medias do periodo')

    hgt_mean = hgt_daily.mean(dim='time', keep_attrs=True)
    clim_hgt_mean = clim_hgt_period.mean(dim='time')
    u_clim_mean = u_clim_period.mean(dim='time')
    v_clim_mean = v_clim_period.mean(dim='time')

    hgt_anom_mean = hgt_mean - clim_hgt_mean

    LOGGER.info(
        'Anomalia hgt: min=%.1f, max=%.1f m',
        float(hgt_anom_mean.min()),
        float(hgt_anom_mean.max()),
    )

    # =========================================================================
    # 6. Preparar entradas do tnflux e calcular WAF
    # =========================================================================
    LOGGER.info('Etapa 6: Calculando TNFLUX (Takaya & Nakamura 2001)')

    # O calculo TN2001 opera em escala sinotica/planetaria.
    # Em resolucao alta (0.25°) as derivadas capturam ruido de mesoescala
    # que nao tem significado fisico para ondas de Rossby.
    # Regridar para ~2.5° (resolucao tipica para WAF) antes de calcular.
    WAF_GRID_SPACING = 2.5
    lat_waf = np.arange(90, -90 - WAF_GRID_SPACING / 2, -WAF_GRID_SPACING)
    lon_waf = np.arange(-180, 180, WAF_GRID_SPACING)

    LOGGER.info(
        'Regridando para %.1f° (%d x %d) para calculo do WAF',
        WAF_GRID_SPACING,
        len(lat_waf),
        len(lon_waf),
    )

    hgt_mean_waf = hgt_mean.interp(lat=lat_waf, lon=lon_waf, method='linear')
    clim_hgt_mean_waf = clim_hgt_mean.interp(lat=lat_waf, lon=lon_waf, method='linear')
    u_clim_mean_waf = u_clim_mean.interp(lat=lat_waf, lon=lon_waf, method='linear')
    v_clim_mean_waf = v_clim_mean.interp(lat=lat_waf, lon=lon_waf, method='linear')

    # Converter hgt (m) -> geopotencial (m2/s2) para o tnflux
    phi_obs = (hgt_mean_waf * G).values
    phi_clim = (clim_hgt_mean_waf * G).values
    u_c = u_clim_mean_waf.values
    v_c = v_clim_mean_waf.values

    # Mascara faixa tropical |lat| < 15 graus
    lat_abs = np.abs(lat_waf)
    mask_eq = lat_abs < TROPICAL_MASK_LAT
    if np.any(mask_eq):
        phi_obs[mask_eq, :] = np.nan
        phi_clim[mask_eq, :] = np.nan
        u_c[mask_eq, :] = np.nan
        v_c[mask_eq, :] = np.nan

    px, py = tnflux.tnf2d(
        u_c,
        v_c,
        phi_clim,
        phi_obs,
        lat_waf,
        lon_waf,
        250.0,
    )

    # Garante NaN na faixa tropical
    if np.any(mask_eq):
        px[mask_eq, :] = np.nan
        py[mask_eq, :] = np.nan

    LOGGER.info(
        'WAF calculado: px min=%.2e max=%.2e | py min=%.2e max=%.2e',
        float(np.nanmin(px)),
        float(np.nanmax(px)),
        float(np.nanmin(py)),
        float(np.nanmax(py)),
    )

    # =========================================================================
    # 7. Salvar rossby_waf.nc
    # =========================================================================
    LOGGER.info('Etapa 7: Salvando rossby_waf.nc')

    # Anomalia em resolucao original (0.25°) para contourf suave
    # WAF em resolucao grossa (~2.5°) para vetores corretos
    full_lat = hgt_anom_mean['lat'].values
    full_lon = hgt_anom_mean['lon'].values

    ds_out = xr.Dataset({
        'hgt_anom_mean': xr.DataArray(
            hgt_anom_mean.values,
            dims=['lat', 'lon'],
            coords={'lat': full_lat, 'lon': full_lon},
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
    LOGGER.info('PLOT_ROSSBY_WAF: Concluido com sucesso')
    LOGGER.info('=' * 70)


if __name__ == '__main__':
    main()
