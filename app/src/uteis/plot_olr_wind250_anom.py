# app/src/uteis/plot_olr_wind250_anom.py
# -*- coding: utf-8 -*-
"""
Download e processamento de vento 250 hPa (ERA5) para anomalia simples de u/v.

Pipeline:
1. Baixa dados ERA5 de u/v em 250 hPa via CDS (NetCDF)
2. Concatena arquivos mensais e calcula media diaria
3. Carrega climatologias de u/v 250 hPa (uwnd250_clim, vwnd250_clim)
4. Interpola climatologias para o grid do ERA5 se necessario
5. Calcula anomalia = media_periodo - media_climatologica
6. Salva resultado em dados/wind250_anom.nc

Chamado por: scripts/s06_olr_wind250_anom.py
"""

from __future__ import annotations

# Bibliotecas padrao
import logging
from datetime import datetime
from pathlib import Path

# Bibliotecas de terceiros
import numpy as np
import xarray as xr

# Modulos locais — reutiliza infraestrutura do plot_chi200
from app.src.uteis.downloaders_wind250 import ensure_era5_uv250_for_period
from app.src.uteis.plot_chi200 import (
    _add_cyclic_and_interp_clim,
    _compute_daily_mean,
    _drop_feb29,
    _find_uv_vars,
    _load_wind_climatology,
    _normalize_latlon_names,
    _normalize_lon,
    _open_and_merge_uv_files,
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
LOGGER = logging.getLogger('PLOT_OLR_WIND250')
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)

# Horas sinoticas padrao
DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)

WIND250_FILE_NAME = 'wind250_anom.nc'


def main() -> None:
    """
    Download, processamento e calculo de anomalia de vento 250 hPa.

    Resultado: dados/wind250_anom.nc com variaveis:
    - u_anom_mean: anomalia de u 250 hPa (m/s)
    - v_anom_mean: anomalia de v 250 hPa (m/s)
    """
    dt_ini = datetime.strptime(settings.DATA_INICIAL, '%Y-%m-%d')
    dt_fim = datetime.strptime(settings.DATA_FINAL, '%Y-%m-%d')

    LOGGER.info('=' * 70)
    LOGGER.info('PLOT_OLR_WIND250: Download e anomalia vento 250 hPa')
    LOGGER.info('Periodo: %s a %s', settings.DATA_INICIAL, settings.DATA_FINAL)
    LOGGER.info('=' * 70)

    # 1. Download ERA5 u/v 250 hPa (NetCDF)
    LOGGER.info('Etapa 1: Download ERA5 u/v 250 hPa (NetCDF)')
    uv_files = ensure_era5_uv250_for_period(
        start=dt_ini,
        end=dt_fim,
        hours_utc=list(DEFAULT_SYNOPTIC_HOURS),
        force_redownload=getattr(settings, 'FORCE_DOWNLOAD', False),
    )
    LOGGER.info('Arquivos de vento 250 hPa: %d', len(uv_files))
    for f in uv_files:
        LOGGER.info('  - %s', f)

    # 2. Abrir e concatenar arquivos mensais
    LOGGER.info('Etapa 2: Concatenando arquivos mensais')
    ds_uv = _open_and_merge_uv_files(uv_files)
    ds_uv = _normalize_latlon_names(ds_uv)
    ds_uv = _normalize_lon(ds_uv)

    # 3. Media diaria (00/06/12/18 UTC)
    LOGGER.info('Etapa 3: Calculando media diaria')
    ds_daily = _compute_daily_mean(ds_uv)
    ds_daily = _drop_feb29(ds_daily)

    # Recortar ao periodo solicitado
    ds_daily = ds_daily.sel(time=slice(np.datetime64(dt_ini.date()), np.datetime64(dt_fim.date())))
    LOGGER.info('Periodo recortado: %d dias', ds_daily.sizes.get('time', 0))

    # Identificar variaveis u/v
    u_var, v_var = _find_uv_vars(ds_daily)
    u_daily = ds_daily[u_var]
    v_daily = ds_daily[v_var]

    # Dropar dimensao pressure_level se existir (unico nivel 250 hPa)
    for dim_name in ('pressure_level', 'isobaricInhPa', 'level'):
        if dim_name in u_daily.dims:
            u_daily = u_daily.isel({dim_name: 0}, drop=True)
            v_daily = v_daily.isel({dim_name: 0}, drop=True)
            LOGGER.info('Dimensao %s removida do ERA5 (nivel unico).', dim_name)

    # 4. Carregar climatologias u/v e calcular anomalias
    LOGGER.info('Etapa 4: Carregando climatologias e calculando anomalias')

    u_clim_path = Path(settings.FILE_CLIMATOLOGIA_UWND250)
    v_clim_path = Path(settings.FILE_CLIMATOLOGIA_VWND250)

    u_clim_da = _load_wind_climatology(u_clim_path, 'u')
    v_clim_da = _load_wind_climatology(v_clim_path, 'v')

    # Dropar dimensao 'level' se existir
    if 'level' in u_clim_da.dims:
        u_clim_da = u_clim_da.isel(level=0, drop=True)
    if 'level' in v_clim_da.dims:
        v_clim_da = v_clim_da.isel(level=0, drop=True)

    # Normalizar lon das climatologias
    for name, clim in [('u_clim', u_clim_da), ('v_clim', v_clim_da)]:
        ds_tmp = _normalize_latlon_names(clim.to_dataset(name=clim.name or name))
        ds_tmp = _normalize_lon(ds_tmp)
        if name == 'u_clim':
            u_clim_da = list(ds_tmp.data_vars.values())[0]
        else:
            v_clim_da = list(ds_tmp.data_vars.values())[0]

    # Selecionar mesmos dias na climatologia
    u_clim_period = _select_climatology_same_days(u_clim_da, u_daily['time'])
    v_clim_period = _select_climatology_same_days(v_clim_da, v_daily['time'])

    # Cyclic point + interpolacao para grid ERA5
    u_clim_period = _add_cyclic_and_interp_clim(u_clim_period, u_daily)
    v_clim_period = _add_cyclic_and_interp_clim(v_clim_period, v_daily)

    # Medias do periodo e climatologica
    u_period_mean = u_daily.mean(dim='time')
    v_period_mean = v_daily.mean(dim='time')
    u_clim_mean = u_clim_period.mean(dim='time')
    v_clim_mean = v_clim_period.mean(dim='time')

    # Anomalias
    u_anom = u_period_mean - u_clim_mean
    v_anom = v_period_mean - v_clim_mean

    LOGGER.info(
        'Anomalia u: min=%.2f, max=%.2f m/s',
        float(u_anom.min()),
        float(u_anom.max()),
    )
    LOGGER.info(
        'Anomalia v: min=%.2f, max=%.2f m/s',
        float(v_anom.min()),
        float(v_anom.max()),
    )

    # 5. Salvar como dados/wind250_anom.nc
    LOGGER.info('Etapa 5: Salvando wind250_anom.nc')

    lat = u_anom['lat'].values
    lon = u_anom['lon'].values

    u_anom_da = xr.DataArray(
        u_anom.values,
        dims=['lat', 'lon'],
        coords={'lat': lat, 'lon': lon},
        name='u_anom_mean',
        attrs={'long_name': 'anomaly of u-wind 250 hPa', 'units': 'm s-1'},
    )

    v_anom_da = xr.DataArray(
        v_anom.values,
        dims=['lat', 'lon'],
        coords={'lat': lat, 'lon': lon},
        name='v_anom_mean',
        attrs={'long_name': 'anomaly of v-wind 250 hPa', 'units': 'm s-1'},
    )

    ds_out = xr.Dataset({
        'u_anom_mean': u_anom_da,
        'v_anom_mean': v_anom_da,
    })

    output_path = DIR_DADOS_BASE / WIND250_FILE_NAME
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    ds_out.to_netcdf(output_path, engine='netcdf4')
    LOGGER.info('Wind250 anom salvo em: %s', output_path)
    LOGGER.info('=' * 70)
    LOGGER.info('PLOT_OLR_WIND250: Concluido com sucesso')
    LOGGER.info('=' * 70)


if __name__ == '__main__':
    main()
