"""s34 - Guia de onda de Rossby: numero de onda estacionario (Ks) em 200 hPa.

Diagnostico de waveguide de Rossby seguindo Hoskins & Ambrizzi (1993). Calcula o
numero de onda estacionario Ks a partir do vento zonal medio basico em 200 hPa e
o sobrepoe a anomalia do vento meridional v'200 (a onda real), permitindo um
diagnostico CONCLUSIVO de propagacao/estacionariedade de ondas de Rossby.

Por que dois campos:
  - Ks (meio): diz ONDE uma onda estacionaria PODE existir (waveguide). Maximos de
    Ks = nucleos de jato (guias); regioes de leste/evanescentes = sem onda
    estacionaria (mascaradas/hachuradas).
  - v'200 (onda real): diz se a onda ESTA la. No mapa, o trem de ondas; no
    Hovmoller, bandas verticais (estacionaria) vs inclinadas (propagante).

No MAPA, a onda real e mostrada por: anomalia de altura geopotencial Z200
(isolinhas) + vetores de WAF (Takaya & Nakamura 2001, direcao da propagacao de
energia). Pareamento classico de waveguide (Hoskins & Ambrizzi; Takaya &
Nakamura). Nos HOVMOLLERS, a onda e a anomalia de vento meridional v'200.

Plotagem padronizada conforme o s07 (areas de plotagem, features, posicao das
barras de cor e tamanho dos ticks).

Pipeline:
  1. Download ERA5/GDAS u/v 200 hPa (hibrido: ERA5 historico, GDAS recente)
  2. Serie diaria u/v200 regridada para a grade da LTM NCEP (2.5°)
  3. Anomalia diaria v'200 = v200 - LTM_diaria(dia-do-ano)
  4. Ks (Hoskins & Ambrizzi 1993) do vento zonal medio do periodo
  5. WAF 200 hPa: media de hgt (ERA5/GDAS) - clim PSL, anomalia Z200 e tnflux.tnf2d
  6. Mapa (por area): Ks sombreado + branco (leste) + U=0 + Z200 anom + WAF
  7. Hovmollers: v'200 medio em 2 faixas de jato (subtropical + polar)

Saida:
    - PNG (mapas por area) + 2 PNG (Hovmollers) + NetCDF em Saida/s34_WAVEGUIDE_ROSSBY/

Criado em: 2026-06-14
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.dates as mdates
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.util import add_cyclic_point as _acp
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from PIL import Image

import tnflux

from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.common.dataset_utils import area_display_name
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.src.uteis.clim_diaria_uv200_ltm import clim_uv200_daily
from app.src.uteis.clim_PSL_geop200 import get_clim_geop200_path
from app.src.uteis.clim_PSL_wnd200 import get_clim_wnd200_paths
from app.src.uteis.downloaders_gdas_hgt200 import ensure_gdas_hgt200_for_period
from app.src.uteis.downloaders_gdas_uv200 import ensure_gdas_uv200_for_period
from app.src.uteis.downloaders_hgt200_ERA5 import ensure_era5_hgt200_for_period
from app.src.uteis.downloaders_wind200 import ensure_era5_uv200_for_period
from app.src.uteis.plot_rossby_waf import (
    G,
    POLAR_MASK_LAT,
    TROPICAL_MASK_LAT,
    WAF_GRID_SPACING,
    _compute_period_mean_streaming_hgt,
    _interp_psl_to_grid,
    _load_psl_clim_hgt,
    _load_psl_clim_wind_component,
)
from app.src.uteis.rossby_wave_source import rossby_wave_source
from app.src.uteis.stationary_wavenumber import stationary_wavenumber

# ---------------------------------------------------------------------------
# Identidade do script
# ---------------------------------------------------------------------------
SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's34'
SCRIPT_NAME = Path(__file__).stem
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
ERA5_LATENCY_DAYS = 7
DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)

# Areas de plotagem (mesmo padrao do s07/s04)
DEFAULT_AREAS = ['globo', 'psa', 'hemisferio_sul', 'hemisferio_norte', 'america_sul', 'globo_3d']

# Ks — niveis de isolinha/sombreado (numero de onda estacionario adimensional)
KS_LEVELS = [1, 2, 3, 4, 5, 6, 7, 8]

# Colormap do Ks: YlOrRd truncado em 0.78 para suavizar os tons de vinho do topo.
# O extend='max' usa a mesma cor do topo truncado (set_over), evitando reescurecer.
_ks_base = plt.get_cmap('YlOrRd')
KS_CMAP = ListedColormap(_ks_base(np.linspace(0.0, 0.78, 256)))
KS_CMAP.set_over(_ks_base(0.78))

# Z200 anomalia — isolinhas no mapa (m); solido=positivo, tracejado=negativo
Z200_LEVELS = np.array([-200, -160, -120, -80, -40, 40, 80, 120, 160, 200], dtype=float)

# Hovmoller v'200 — niveis/ticks fixos e arredondados (sem decimais), paleta padrao do projeto
HOV_LEVELS = np.arange(-30, 33, 3)        # m/s — -30..30 (extend='both' captura |v'|>30)
HOV_TICKS = np.arange(-30, 31, 10)        # -30,-20,-10,0,10,20,30

# RWS (fonte de onda de Rossby) — anomala, em 10^-11 s^-2. Paleta do s05 (BrBG):
#   verde (RWS>0) = fonte anticiclonica no HS (lanca o trem); marrom (RWS<0) = ciclonica.
RWS_SCALE = 1e11
RWS_LEVELS = np.arange(-40, 44, 4)        # x10^-11 s^-2
RWS_TICKS = np.arange(-40, 41, 10)
RWS_MASK_LAT = 10.0                       # mascara |lat|<10° (f->0 deixa a RWS ruidosa)
# Vento divergente: densidade media (campo ja suavizado evita o caos do bruto)
DIVWIND_QUIVER = {'step': 2, 'width': 0.0014, 'scale': 80.0, 'min_mag': 0.5}

# WAF (quiver) — normalizado pelo maximo, mascara fracos. Padrao do s07.
QUIVER_DEFAULTS = {
    'step': 2, 'width': 0.002, 'headwidth': 4.5, 'headlength': 6.0,
    'scale': None, 'scale_units': 'xy', 'min_amp_ratio': 0.05,
}
QUIVER_POR_AREA = {
    'globo': {'step': 2},
    'psa': {'step': 2},
    'hemisferio_sul': {'step': 2},
    'hemisferio_norte': {'step': 2},
    'america_sul': {'step': 1, 'width': 0.004, 'headwidth': 5.0, 'headlength': 7.0, 'scale': 0.12},
    'globo_3d': {'step': 2, 'width': 0.003, 'headwidth': 5.0, 'headlength': 7.0},
}


def _get_quiver_config(area: str) -> dict:
    cfg = dict(QUIVER_DEFAULTS)
    if area in QUIVER_POR_AREA:
        cfg.update(QUIVER_POR_AREA[area])
    return cfg


def _prep_cyclic_180(da: xr.DataArray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converte para -180..180, ordena, adiciona cyclic point. Retorna (vals, lon, lat)."""
    da = _to_180(da)
    vals, lon_c = _acp(da.values, coord=da['lon'].values)
    return vals, lon_c, da['lat'].values


def _compute_waf200(era5_period, gdas_period, ini_dt, fim_dt, ini_str, fim_str, logger):
    """Calcula o Wave Activity Flux (Takaya & Nakamura 2001) em 200 hPa.

    Reusa o pipeline do plot_rossby_waf: media do periodo de hgt (streaming),
    climatologia PSL hgt+u/v 200, anomalia, regrida p/ 2.5° e chama tnflux.tnf2d.
    Retorna (hgt_anom_da, px, py, lat_waf, lon_waf).
    """
    hgt_files = []
    if era5_period:
        logger.info(f'  ERA5 hgt 200 hPa ({era5_period[0].date()} a {era5_period[1].date()})...')
        hgt_files += list(ensure_era5_hgt200_for_period(
            start=era5_period[0], end=era5_period[1], hours_utc=list(DEFAULT_SYNOPTIC_HOURS),
        ))
    if gdas_period:
        logger.info(f'  GDAS hgt 200mb ({gdas_period[0].date()} a {gdas_period[1].date()})...')
        hgt_files += list(ensure_gdas_hgt200_for_period(start=gdas_period[0], end=gdas_period[1]))

    hgt_mean, ref_lat, ref_lon = _compute_period_mean_streaming_hgt(
        hgt_files, DEFAULT_SYNOPTIC_HOURS, ini_dt, fim_dt,
    )
    hgt_mean_da = xr.DataArray(hgt_mean, dims=['lat', 'lon'], coords={'lat': ref_lat, 'lon': ref_lon})

    clim_hgt_da = _load_psl_clim_hgt(get_clim_geop200_path(ini_str, fim_str))
    clim_hgt = _interp_psl_to_grid(clim_hgt_da, ref_lat, ref_lon)
    clim_u_path, clim_v_path = get_clim_wnd200_paths(ini_str, fim_str)
    clim_u = _interp_psl_to_grid(_load_psl_clim_wind_component(clim_u_path, 'u'), ref_lat, ref_lon)
    clim_v = _interp_psl_to_grid(_load_psl_clim_wind_component(clim_v_path, 'v'), ref_lat, ref_lon)

    hgt_anom = hgt_mean_da - clim_hgt

    lat_waf = np.arange(90, -90 - WAF_GRID_SPACING / 2, -WAF_GRID_SPACING)
    lon_waf = np.arange(-180, 180, WAF_GRID_SPACING)
    phi_obs = (hgt_mean_da.interp(lat=lat_waf, lon=lon_waf, method='linear') * G).values
    phi_clim = (clim_hgt.interp(lat=lat_waf, lon=lon_waf, method='linear') * G).values
    u_c = clim_u.interp(lat=lat_waf, lon=lon_waf, method='linear').values
    v_c = clim_v.interp(lat=lat_waf, lon=lon_waf, method='linear').values

    mask = (np.abs(lat_waf) < TROPICAL_MASK_LAT) | (np.abs(lat_waf) > POLAR_MASK_LAT)
    if np.any(mask):
        for arr in (phi_obs, phi_clim, u_c, v_c):
            arr[mask, :] = np.nan

    px, py = tnflux.tnf2d(u_c, v_c, phi_clim, phi_obs, lat_waf, lon_waf, 200.0)
    if np.any(mask):
        px[mask, :] = np.nan
        py[mask, :] = np.nan

    return hgt_anom, px, py, lat_waf, lon_waf


# ---------------------------------------------------------------------------
# Utilitarios de grade (padrao s31/s33)
# ---------------------------------------------------------------------------
def _ensure_time_coord(obj):
    if hasattr(obj, 'dims') and 'time' not in obj.dims and 'valid_time' in obj.dims:
        obj = obj.rename({'valid_time': 'time'})
    elif hasattr(obj, 'coords') and 'time' not in obj.coords and 'valid_time' in obj.coords:
        obj = obj.rename({'valid_time': 'time'})
    if 'time' not in obj.coords:
        raise KeyError("Nem 'time' nem 'valid_time' encontrados.")
    return obj


def _rename_std_latlon(obj):
    ren = {}
    for name in list(obj.dims) + list(obj.coords):
        low = name.lower()
        if low == 'latitude' and 'lat' not in obj.dims:
            ren[name] = 'lat'
        elif low == 'longitude' and 'lon' not in obj.dims:
            ren[name] = 'lon'
    return obj.rename(ren) if ren else obj


def _drop_expver(ds: xr.Dataset) -> xr.Dataset:
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


def _sort_dedup_time(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.sortby('time')
    t = pd.DatetimeIndex(pd.to_datetime(ds['time'].values))
    _, idx = np.unique(t.values, return_index=True)
    if len(idx) != ds.sizes.get('time', 0):
        ds = ds.isel(time=np.sort(idx))
    return ds


def _find_uv_vars(ds: xr.Dataset) -> Tuple[str, str]:
    u_name = v_name = None
    for name in ('u', 'u_component_of_wind', 'U_GRD_L100', 'uwnd'):
        if name in ds.data_vars:
            u_name = name
            break
    for name in ('v', 'v_component_of_wind', 'V_GRD_L100', 'vwnd'):
        if name in ds.data_vars:
            v_name = name
            break
    if u_name is None or v_name is None:
        raise KeyError(f'u/v nao encontrados. Disponiveis: {list(ds.data_vars)}')
    return u_name, v_name


def _to_180(da: xr.DataArray) -> xr.DataArray:
    """Converte coords de lon 0..360 para -180..180 ordenado (para plotagem)."""
    if np.any(da['lon'].values > 180):
        da = da.assign_coords(lon=(((da['lon'].values + 180) % 360) - 180)).sortby('lon')
    return da


# ---------------------------------------------------------------------------
# Selecao de fonte ERA5/GDAS
# ---------------------------------------------------------------------------
def _get_data_sources(
    dt_ini: datetime,
    dt_fim: datetime,
) -> Tuple[Optional[Tuple[datetime, datetime]], Optional[Tuple[datetime, datetime]]]:
    """Retorna (era5_period, gdas_period) — None quando nao ha dados dessa fonte."""
    cutoff = (datetime.now() - timedelta(days=ERA5_LATENCY_DAYS)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    if dt_fim < cutoff:
        return (dt_ini, dt_fim), None
    if dt_ini >= cutoff:
        return None, (dt_ini, dt_fim)
    return (dt_ini, cutoff - timedelta(days=1)), (cutoff, dt_fim)


def _daily_uv200_on_grid(
    files, dt_ini: datetime, dt_fim: datetime,
    target_lat: np.ndarray, target_lon: np.ndarray, logger,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Le ERA5/GDAS, filtra horas sinoticas, media diaria, regrida p/ a grade da LTM.

    Mantem lon 0..360 (igual a LTM NCEP) para a anomalia ser consistente. Retorna
    (u_da, v_da) em (time, lat, lon), lat ascendente.
    """
    t_ini = np.datetime64(dt_ini.date())
    t_fim = np.datetime64(dt_fim.date())
    req = set(DEFAULT_SYNOPTIC_HOURS)
    tgt_lat = xr.DataArray(target_lat, dims=['lat'])
    tgt_lon = xr.DataArray(target_lon, dims=['lon'])

    us, vs = [], []
    for fp in files:
        ds = xr.open_dataset(fp, engine='netcdf4')
        try:
            ds = _sort_dedup_time(_rename_std_latlon(_drop_expver(_ensure_time_coord(ds))))
            ds = ds.assign_coords(lon=(ds['lon'] % 360)).sortby('lon')
            u_var, v_var = _find_uv_vars(ds)
            da_u, da_v = ds[u_var], ds[v_var]
            for dim in ('pressure_level', 'isobaricInhPa', 'level'):
                if dim in da_u.dims:
                    da_u = da_u.isel({dim: 0}, drop=True)
                    da_v = da_v.isel({dim: 0}, drop=True)
            ti = pd.DatetimeIndex(pd.to_datetime(da_u['time'].values))
            mh = np.array([h in req for h in ti.hour], dtype=bool)
            da_u, da_v = da_u.isel(time=mh), da_v.isel(time=mh)
            da_u = da_u.sel(time=slice(t_ini, t_fim))
            da_v = da_v.sel(time=slice(t_ini, t_fim))
            if da_u.sizes.get('time', 0) == 0:
                continue
            da_u = da_u.resample(time='1D').mean()
            da_v = da_v.resample(time='1D').mean()
            keep = ~((da_u['time'].dt.month == 2) & (da_u['time'].dt.day == 29))
            da_u, da_v = da_u.isel(time=keep.values), da_v.isel(time=keep.values)
            da_u = da_u.interp(lat=tgt_lat, lon=tgt_lon, method='linear').reset_coords(drop=True)
            da_v = da_v.interp(lat=tgt_lat, lon=tgt_lon, method='linear').reset_coords(drop=True)
            us.append(da_u.load())
            vs.append(da_v.load())
            logger.info('Serie diaria: {} -> {} dias', fp.name, da_u.sizes['time'])
        finally:
            ds.close()

    if not us:
        raise RuntimeError('Nenhum dado u/v 200 valido no periodo.')

    u_da = xr.concat(us, dim='time', coords='minimal', compat='override').sortby('time')
    v_da = xr.concat(vs, dim='time', coords='minimal', compat='override').sortby('time')
    _, uniq = np.unique(u_da['time'].values, return_index=True)
    return u_da.isel(time=uniq), v_da.isel(time=uniq)


# ---------------------------------------------------------------------------
# Plotagem — utilitarios (padrao s07)
# ---------------------------------------------------------------------------
def _get_area_list():
    if hasattr(settings, 'LST_AREAS_S34'):
        return list(settings.LST_AREAS_S34)
    return list(DEFAULT_AREAS)


def _add_logo_to_map(ax, logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=500):
    logo = Image.open(logo_path).convert('RGBA')
    bbox = logo.getbbox()
    if bbox is not None:
        logo = logo.crop(bbox)
    imagebox = OffsetImage(np.array(logo), zoom=zoom)
    ab = AnnotationBbox(
        imagebox, (0, 0), xycoords=ax.transAxes,
        xybox=(xoffset, yoffset), boxcoords='offset points',
        box_alignment=(0, 0), frameon=False, pad=0, zorder=zorder, clip_on=False,
    )
    ax.add_artist(ab)


def _configure_gridlines(gl, area):
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 20, 'color': 'black'}
    gl.ylabel_style = {'size': 20, 'color': 'black'}

    if area in {'hemisferio_sul', 'hemisferio_norte', 'psa'}:
        gl.xlocator = MultipleLocator(40)
        gl.ylocator = MultipleLocator(20)
    elif area == 'america_sul':
        gl.xlocator = MultipleLocator(20)
        gl.ylocator = MultipleLocator(20)
    elif area == 'globo':
        gl.xlocator = MultipleLocator(40)
        gl.ylocator = MultipleLocator(20)
        gl.xlabel_style = {'size': 15, 'color': 'black'}
        gl.ylabel_style = {'size': 15, 'color': 'black'}


def _fmt_lon(v: int) -> str:
    if v == 0 or abs(v) == 180:
        return f'{abs(v)}°'
    return f'{v}°E' if v > 0 else f'{-v}°W'


def _fmt_lat(v: float) -> str:
    if v == 0:
        return '0°'
    return f'{abs(v):g}°N' if v > 0 else f'{abs(v):g}°S'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger = get_logger(SCRIPT_ID)

    logger.info('=' * 80)
    logger.info(f'SCRIPT {SCRIPT_ID.upper()}: {SCRIPT_DESC}')
    logger.info('=' * 80)

    # ---- Parametros do settings ----
    smooth_deg = float(settings.get('WGUIDE_SMOOTH_DEG', 5.0))

    # Dois Hovmollers: um por corrente de jato do HS (subtropical ~30S, polar ~50S).
    # Cada jato e um waveguide distinto, com trens de onda em geral em fases diferentes.
    hov_bands = [
        {
            'slug': 'subtropical',
            'nome': 'Jato Subtropical HS',
            'lat_min': float(settings.get('WGUIDE_HOV_SUBTROPICAL_LAT_MIN', -45.0)),
            'lat_max': float(settings.get('WGUIDE_HOV_SUBTROPICAL_LAT_MAX', -20.0)),
        },
        {
            'slug': 'polar',
            'nome': 'Jato Polar HS',
            'lat_min': float(settings.get('WGUIDE_HOV_POLAR_LAT_MIN', -60.0)),
            'lat_max': float(settings.get('WGUIDE_HOV_POLAR_LAT_MAX', -40.0)),
        },
    ]
    for b in hov_bands:
        if b['lat_min'] >= b['lat_max']:
            raise ValueError(
                f"Faixa {b['slug']}: LAT_MIN ({b['lat_min']}) deve ser menor que "
                f"LAT_MAX ({b['lat_max']})."
            )
        b['band'] = f"{_fmt_lat(b['lat_min'])}–{_fmt_lat(b['lat_max'])}"

    lst_areas = _get_area_list()
    logger.info(
        f'Areas: {lst_areas} | Hovmollers: '
        f"{', '.join(b['slug'] + ' ' + b['band'] for b in hov_bands)} | suavizacao {smooth_deg}°"
    )

    ini_str = str(settings.DATA_INICIAL)
    fim_str = str(settings.DATA_FINAL)
    ini_dt = datetime.fromisoformat(ini_str)
    fim_dt = datetime.fromisoformat(fim_str)

    total_days = (fim_dt.date() - ini_dt.date()).days + 1
    if total_days < 2:
        raise ValueError(
            f'O Hovmoller precisa de pelo menos 2 dias (periodo atual: {total_days} dia). '
            f'Ajuste DATA_INICIAL/DATA_FINAL no settings.local.toml.'
        )

    output_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_WAVEGUIDE_ROSSBY'
    dados_dir = Path(settings.DIR_DADOS)
    entrada_dir = Path(settings.DIR_INPUT)

    map_files = {area: output_dir / f'ks_waveguide_200hpa_{area}.png' for area in lst_areas}
    rws_files = {area: output_dir / f'fontes_rws_200hpa_{area}.png' for area in lst_areas}
    for b in hov_bands:
        b['png'] = output_dir / f"hovmoller_vprime200_{b['slug']}.png"
    nc_name = f'waveguide_ks_vprime200_{ini_str}_to_{fim_str}.nc'
    output_files = (
        [str(p) for p in map_files.values()]
        + [str(p) for p in rws_files.values()]
        + [str(b['png']) for b in hov_bands]
    )

    cache_params = {
        'DATA_INICIAL': ini_str,
        'DATA_FINAL': fim_str,
        'areas': lst_areas,
        'hov_bands': [(b['slug'], b['lat_min'], b['lat_max']) for b in hov_bands],
        'smooth_deg': smooth_deg,
        'script_version': '3.0',  # remove ∇·WAF (mal-condicionado); mapa de fontes limpo (RWS + v_div)
    }

    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('CACHE VALIDO! Execucao ja foi realizada com os mesmos parametros.')
        return

    start_time = time.time()
    logger.info(f'Periodo: {ini_str} a {fim_str} ({total_days} dias)')
    output_dir.mkdir(parents=True, exist_ok=True)
    dados_dir.mkdir(parents=True, exist_ok=True)

    # ---- Etapa 1: Download ERA5/GDAS u/v 200 ----
    era5_period, gdas_period = _get_data_sources(ini_dt, fim_dt)
    files = []
    if era5_period:
        logger.info(f'Etapa 1a: ERA5 u/v 200 hPa ({era5_period[0].date()} a {era5_period[1].date()})...')
        files += list(ensure_era5_uv200_for_period(
            start=era5_period[0], end=era5_period[1],
            hours_utc=list(DEFAULT_SYNOPTIC_HOURS),
        ))
    if gdas_period:
        logger.info(f'Etapa 1b: GDAS u/v 200mb ({gdas_period[0].date()} a {gdas_period[1].date()})...')
        files += list(ensure_gdas_uv200_for_period(
            start=gdas_period[0], end=gdas_period[1],
        ))

    # ---- Etapa 2: Serie diaria u/v200 na grade da LTM (2.5°) ----
    logger.info('Etapa 2: Serie diaria u/v 200 (grade LTM NCEP 2.5°)...')
    dates_probe = np.array([np.datetime64(fim_dt.date())])
    _, _, ltm_lat, ltm_lon = clim_uv200_daily(dates_probe)
    order = np.argsort(ltm_lat)
    lat, lon = ltm_lat[order], ltm_lon  # lat ascendente, lon 0..360

    u_da, v_da = _daily_uv200_on_grid(files, ini_dt, fim_dt, lat, lon, logger)
    dates = np.array([np.datetime64(pd.Timestamp(t).date()) for t in u_da['time'].values])

    # ---- Etapa 3: Anomalia diaria v'200 = v - LTM(dia-do-ano) ----
    logger.info('Etapa 3: Anomalia diaria v\'200 (- LTM diaria NCEP)...')
    u_clim_d, v_clim_d, _, _ = clim_uv200_daily(dates)
    u_clim_d = u_clim_d[:, order, :]  # alinha lat ascendente
    v_clim_d = v_clim_d[:, order, :]
    v_anom = xr.DataArray(
        v_da.values - v_clim_d, dims=('time', 'lat', 'lon'),
        coords={'time': u_da['time'].values, 'lat': lat, 'lon': lon},
    )

    # ---- Etapa 4: Ks do vento zonal medio do periodo (escoamento basico) ----
    logger.info('Etapa 4: Numero de onda estacionario Ks (Hoskins & Ambrizzi 1993)...')
    u_mean = u_da.mean(dim='time', skipna=True)            # vento total medio
    v_mean = v_da.mean(dim='time', skipna=True)
    v_anom_mean = v_anom.mean(dim='time', skipna=True)     # onda real media
    sw = stationary_wavenumber(u_mean.values, lat, lon, smooth_deg=smooth_deg)

    # ---- Etapa 4b: Fonte de onda de Rossby (RWS) anomala = periodo - climatologia ----
    logger.info('Etapa 4b: Fonte de onda de Rossby (Sardeshmukh & Hoskins 1988)...')
    rws_p, uchi_p, vchi_p = rossby_wave_source(u_mean.values, v_mean.values, lat, lon)
    rws_c, uchi_c, vchi_c = rossby_wave_source(u_clim_d.mean(axis=0), v_clim_d.mean(axis=0), lat, lon)
    trop = np.abs(lat) < RWS_MASK_LAT
    rws_anom = rws_p - rws_c
    rws_anom[trop, :] = np.nan
    uchi_anom = uchi_p - uchi_c
    vchi_anom = vchi_p - vchi_c

    # ---- Etapa 5: WAF (Takaya & Nakamura 2001) + anomalia Z200 ----
    logger.info('Etapa 5: Wave Activity Flux 200 hPa (Takaya & Nakamura 2001)...')
    hgt_anom, px, py, lat_waf, lon_waf = _compute_waf200(
        era5_period, gdas_period, ini_dt, fim_dt, ini_str, fim_str, logger,
    )

    # ---- Etapa 6: Mapas por area (padrao s07): Ks + Z200 anom + WAF + U=0 ----
    logger.info('Etapa 6: Gerando mapas Ks + Z200 anomalia + WAF por area...')
    # Prepara campos em -180..180 + cyclic point (uma vez para todas as areas).
    # Ks fica NaN onde a onda e evanescente -> regiao em branco (sem mascara, padrao literatura).
    ks_da = _to_180(xr.DataArray(sw.ks, dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}))
    u180 = _to_180(u_mean)
    plon = ks_da['lon'].values
    ks_cyc, lon_cyc = _acp(ks_da.values, coord=plon)
    u_cyc, _ = _acp(u180.values, coord=plon)
    # Z200 anomalia (grade ref ERA5/GDAS) e WAF (grade 2.5°) em -180..180 + cyclic
    hgt_cyc, lon_hgt, lat_hgt = _prep_cyclic_180(hgt_anom)
    px_cyc, lon_waf_cyc, lat_waf_c = _prep_cyclic_180(
        xr.DataArray(px, dims=('lat', 'lon'), coords={'lat': lat_waf, 'lon': lon_waf}))
    py_cyc, _, _ = _prep_cyclic_180(
        xr.DataArray(py, dims=('lat', 'lon'), coords={'lat': lat_waf, 'lon': lon_waf}))

    # RWS anomala (grade LTM 2.5°) + vento divergente anomalo em -180..180 + cyclic
    rws_cyc, lon_rws, lat_rws = _prep_cyclic_180(
        xr.DataArray(rws_anom * RWS_SCALE, dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}))
    uchi_cyc, _, _ = _prep_cyclic_180(
        xr.DataArray(uchi_anom, dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}))
    vchi_cyc, _, _ = _prep_cyclic_180(
        xr.DataArray(vchi_anom, dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}))

    info_plot = settings['areas_plotagem']
    for area in lst_areas:
        logger.info(f'Gerando mapa para area: {area_display_name(area)}')
        _plot_ks_area(
            area=area, info_plot=info_plot, lon_cyc=lon_cyc, lat=lat,
            ks_cyc=ks_cyc, u_cyc=u_cyc,
            hgt_cyc=hgt_cyc, lon_hgt=lon_hgt, lat_hgt=lat_hgt,
            px_cyc=px_cyc, py_cyc=py_cyc, lon_waf=lon_waf_cyc, lat_waf=lat_waf_c,
            ini_dt=ini_dt, fim_dt=fim_dt, entrada_dir=entrada_dir,
            out_path=map_files[area], logger=logger,
        )
        _plot_rws_area(
            area=area, info_plot=info_plot, lon_rws=lon_rws, lat_rws=lat_rws,
            rws_cyc=rws_cyc, uchi_cyc=uchi_cyc, vchi_cyc=vchi_cyc,
            ini_dt=ini_dt, fim_dt=fim_dt, entrada_dir=entrada_dir,
            out_path=rws_files[area], logger=logger,
        )

    # ---- Etapa 7: Hovmollers v'200, um por corrente de jato (subtropical + polar) ----
    # Mantem lon 0..360 (Pacifico inteiro no centro, igual aos mapas) — foco no HS.
    logger.info('Etapa 7: Gerando Hovmollers de v\'200 (jato subtropical + polar)...')
    for b in hov_bands:
        logger.info(f"  Hovmoller {b['nome']} ({b['band']})...")
        b['hov'] = (
            v_anom.sel(lat=slice(b['lat_min'], b['lat_max'])).mean(dim='lat', skipna=True)
        )
        _plot_hovmoller(
            hov_v=b['hov'], hov_band=b['band'], nome=b['nome'], ini_dt=ini_dt, fim_dt=fim_dt,
            total_days=total_days, entrada_dir=entrada_dir,
            out_path=b['png'], logger=logger,
        )

    # ---- NetCDF ----
    ks_out = xr.DataArray(
        sw.ks, dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon},
        attrs={'long_name': 'Numero de onda estacionario de Rossby (Hoskins & Ambrizzi 1993)',
               'units': '1'},
    )
    ds_out = xr.Dataset({
        'ks': ks_out,
        'u_mean_200': u_mean.assign_attrs(long_name='Vento zonal medio 200 hPa', units='m s-1'),
        'vprime_mean_200': v_anom_mean.assign_attrs(long_name='Anomalia media v 200 hPa', units='m s-1'),
        'hgt_anom_200': hgt_anom.assign_attrs(long_name='Anomalia hgt 200 hPa', units='m'),
        'rws_anom_200': xr.DataArray(
            rws_anom, dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon},
            attrs={'long_name': 'Fonte de onda de Rossby anomala (Sardeshmukh-Hoskins)', 'units': 's-2'},
        ),
        'waf_x': xr.DataArray(
            px, dims=('lat_waf', 'lon_waf'), coords={'lat_waf': lat_waf, 'lon_waf': lon_waf},
            attrs={'long_name': 'Wave Activity Flux zonal (TN2001)', 'units': 'm2 s-2'},
        ),
        'waf_y': xr.DataArray(
            py, dims=('lat_waf', 'lon_waf'), coords={'lat_waf': lat_waf, 'lon_waf': lon_waf},
            attrs={'long_name': 'Wave Activity Flux meridional (TN2001)', 'units': 'm2 s-2'},
        ),
    })
    for b in hov_bands:
        ds_out[f"vprime_hov_{b['slug']}"] = b['hov'].assign_attrs(
            long_name=f"Hovmoller v'200 {b['nome']} ({b['band']})", units='m s-1',
        )
    nc_path = output_dir / nc_name
    logger.info(f'Salvando NetCDF: {nc_path}')
    ds_out.to_netcdf(str(nc_path))

    execution_time = time.time() - start_time
    save_cache_metadata(SCRIPT_ID, cache_params, output_files, execution_time)
    logger.info('=' * 80)
    logger.info(f'Script {SCRIPT_ID.upper()} concluido com sucesso!')
    logger.info(f'Tempo de execucao: {execution_time:.1f}s ({execution_time / 60:.1f} min)')
    logger.info(f'{len(lst_areas)} mapas + {len(hov_bands)} Hovmollers gerados em: {output_dir}')
    logger.info('=' * 80)


def _plot_ks_area(
    area, info_plot, lon_cyc, lat, ks_cyc, u_cyc,
    hgt_cyc, lon_hgt, lat_hgt, px_cyc, py_cyc, lon_waf, lat_waf,
    ini_dt, fim_dt, entrada_dir, out_path, logger,
):
    """Mapa de UMA area (padrao s07): Ks sombreado + Z200 anomalia + WAF + U=0."""
    cfg = info_plot[area]
    is_polar = cfg.get('projection', '') == 'orthographic_south'

    if is_polar:
        proj = ccrs.Orthographic(
            central_longitude=settings.get(
                'ORTHO_CENTRAL_LONGITUDE', cfg.get('ortho_central_longitude', -71)),
            central_latitude=settings.get(
                'ORTHO_CENTRAL_LATITUDE', cfg.get('ortho_central_latitude', -84)),
        )
    else:
        proj = ccrs.PlateCarree(central_longitude=cfg['central_longitude_mapa'])

    transform = ccrs.PlateCarree(central_longitude=cfg['central_longitude_plot'])

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    if is_polar:
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

    # Grade
    if is_polar:
        gl = ax.gridlines(draw_labels=False, linestyle='--', alpha=0.5)
        gl.xlocator = MultipleLocator(30)
        gl.ylocator = MultipleLocator(20)
    else:
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.0)
        _configure_gridlines(gl, area)

    # Limites
    if not is_polar:
        ax.set_xlim([cfg['lon_esq'], cfg['lon_dir']])
        ax.set_ylim([cfg['lat_inf'], cfg['lat_sup']])

    # Features (mesma pilha do s07)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2)
    ax.add_feature(cfeature.LAND.with_scale('50m'), linewidth=0.5, facecolor='whitesmoke')
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1.2, zorder=100)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, zorder=100)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, zorder=100)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), linewidth=0.5, facecolor='white')

    # Ks sombreado (apenas onde a onda estacionaria E possivel)
    im = ax.contourf(
        lon_cyc, lat, ks_cyc, levels=KS_LEVELS, cmap=KS_CMAP, extend='max',
        transform=transform, zorder=10,
    )
    # Regioes evanescentes (leste / beta_M<0) ficam em branco (Ks=NaN, sem mascara).
    # Linha critica U=0 (fronteira oeste/leste) — delimita o "duto" do waveguide
    ax.contour(
        lon_cyc, lat, u_cyc, levels=[0.0], colors='black',
        linewidths=4.5, transform=transform, zorder=40,
    )
    # Onda real: isolinhas de anomalia de Z200 (preto solido=positivo, tracejado=negativo)
    zpos = Z200_LEVELS[Z200_LEVELS > 0]
    zneg = Z200_LEVELS[Z200_LEVELS < 0]
    ax.contour(
        lon_hgt, lat_hgt, hgt_cyc, levels=zpos, colors='black',
        linewidths=1.1, transform=transform, zorder=50,
    )
    ax.contour(
        lon_hgt, lat_hgt, hgt_cyc, levels=zneg, colors='black', linestyles='dashed',
        linewidths=1.1, transform=transform, zorder=50,
    )
    # WAF (Takaya & Nakamura 2001): direcao da propagacao de energia da onda
    qcfg = _get_quiver_config(area)
    step = int(qcfg['step'])
    lon_q = lon_waf[::step]
    lat_q = lat_waf[::step]
    px_q = px_cyc[::step, ::step]
    py_q = py_cyc[::step, ::step]
    amp = np.sqrt(px_q ** 2 + py_q ** 2)
    max_amp = np.nanmax(amp)
    if max_amp and max_amp > 0:
        weak = amp < float(qcfg['min_amp_ratio']) * max_amp
        px_q = np.where(weak, np.nan, px_q / max_amp)
        py_q = np.where(weak, np.nan, py_q / max_amp)
    ax.quiver(
        lon_q, lat_q, px_q, py_q, transform=transform,
        scale=qcfg['scale'], scale_units=qcfg['scale_units'], width=float(qcfg['width']),
        headwidth=float(qcfg['headwidth']), headlength=float(qcfg['headlength']),
        color='black', zorder=60,
    )

    # Colorbar (posicao/tamanho de ticks conforme o s07)
    if is_polar and area != 'globo_3d':
        cbar = plt.colorbar(im, ax=ax, pad=0.05, fraction=0.04, ticks=KS_LEVELS)
        cbar.set_label(label='Ks', size=10)
        cbar.ax.tick_params(labelsize=10)
    else:
        divider = make_axes_locatable(ax)
        if area in {'america_sul', 'globo_3d'}:
            cax = divider.append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
            cbar = plt.colorbar(
                im, cax=cax, pad=0.02, fraction=0.02375, extend='max', ticks=KS_LEVELS,
            )
        else:
            cax = divider.append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)
            cbar = plt.colorbar(
                im, cax=cax, pad=0.02, fraction=0.02375, location='bottom',
                extend='max', orientation='horizontal', ticks=KS_LEVELS,
            )
        cbar.set_label(label='Numero de onda estacionario de Rossby (Ks)', size=18)
        cbar.ax.tick_params(labelsize=20)

    # Titulo
    dt_ini = ini_dt.strftime('%d-%m-%y')
    dt_fim = fim_dt.strftime('%d-%m-%y')
    titulo = (
        f'Guia de Onda de Rossby — Ks 200 hPa + anomalia Z200 + WAF\n'
        f'(De {dt_ini} a {dt_fim}) — branco: sem onda estacionaria (leste); linha grossa: U=0'
    )
    ax.set_title(titulo, fontsize=12 if is_polar else 16, loc='left')

    # Logo
    logo_path = (
        None if settings.get('SEM_LOGO', False)
        else entrada_dir / ('logo_grec.png' if settings.get('LOGO_GREC', False) else 'novo_logo.png')
    )
    if logo_path is not None and logo_path.exists():
        _add_logo_to_map(ax=ax, logo_path=logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=500)

    logger.info(f'Salvando a figura {out_path}')
    plt.savefig(str(out_path), dpi=fig.dpi, bbox_inches='tight')
    plt.close('all')


def _plot_rws_area(
    area, info_plot, lon_rws, lat_rws, rws_cyc, uchi_cyc, vchi_cyc,
    ini_dt, fim_dt, entrada_dir, out_path, logger,
):
    """Mapa companheiro (padrao s07): fonte de onda de Rossby anomala + vento divergente.

    Paleta BrBG (s05): verde (RWS>0) = fonte anticiclonica no HS (lanca o trem);
    marrom (RWS<0) = fonte ciclonica.
    """
    cfg = info_plot[area]
    is_polar = cfg.get('projection', '') == 'orthographic_south'

    if is_polar:
        proj = ccrs.Orthographic(
            central_longitude=settings.get(
                'ORTHO_CENTRAL_LONGITUDE', cfg.get('ortho_central_longitude', -71)),
            central_latitude=settings.get(
                'ORTHO_CENTRAL_LATITUDE', cfg.get('ortho_central_latitude', -84)),
        )
    else:
        proj = ccrs.PlateCarree(central_longitude=cfg['central_longitude_mapa'])
    transform = ccrs.PlateCarree(central_longitude=cfg['central_longitude_plot'])

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    if is_polar:
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        ax.set_boundary(mpath.Path(verts * radius + center), transform=ax.transAxes)

    if is_polar:
        gl = ax.gridlines(draw_labels=False, linestyle='--', alpha=0.5)
        gl.xlocator = MultipleLocator(30)
        gl.ylocator = MultipleLocator(20)
    else:
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.0)
        _configure_gridlines(gl, area)

    if not is_polar:
        ax.set_xlim([cfg['lon_esq'], cfg['lon_dir']])
        ax.set_ylim([cfg['lat_inf'], cfg['lat_sup']])

    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2)
    ax.add_feature(cfeature.LAND.with_scale('50m'), linewidth=0.5, facecolor='whitesmoke')
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1.2, zorder=100)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, zorder=100)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, zorder=100)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), linewidth=0.5, facecolor='white')

    # RWS anomala sombreada (BrBG: verde=positivo/anticiclonico HS, marrom=negativo/ciclonico)
    cmap = plt.get_cmap('BrBG')
    im = ax.contourf(
        lon_rws, lat_rws, rws_cyc, levels=RWS_LEVELS, cmap=cmap, extend='both',
        transform=transform, zorder=10,
    )

    # Vento divergente anomalo (o "sopro" que gera a fonte)
    step = DIVWIND_QUIVER['step']
    lon_q, lat_q = lon_rws[::step], lat_rws[::step]
    u_q = uchi_cyc[::step, ::step].copy()
    v_q = vchi_cyc[::step, ::step].copy()
    mag = np.sqrt(u_q ** 2 + v_q ** 2)
    weak = mag < DIVWIND_QUIVER['min_mag']
    u_q[weak] = np.nan
    v_q[weak] = np.nan
    ax.quiver(
        lon_q, lat_q, u_q, v_q, transform=transform,
        scale=DIVWIND_QUIVER['scale'], width=DIVWIND_QUIVER['width'],
        color='black', zorder=60,
    )

    # Colorbar (posicao/tamanho de ticks conforme o s07)
    if is_polar and area != 'globo_3d':
        cbar = plt.colorbar(im, ax=ax, pad=0.05, fraction=0.04, ticks=RWS_TICKS)
        cbar.set_label(label='RWS (x10^-11 s^-2)', size=10)
        cbar.ax.tick_params(labelsize=10)
    else:
        divider = make_axes_locatable(ax)
        if area in {'america_sul', 'globo_3d'}:
            cax = divider.append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
            cbar = plt.colorbar(im, cax=cax, pad=0.02, fraction=0.02375, extend='both', ticks=RWS_TICKS)
        else:
            cax = divider.append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)
            cbar = plt.colorbar(
                im, cax=cax, pad=0.02, fraction=0.02375, location='bottom',
                extend='both', orientation='horizontal', ticks=RWS_TICKS,
            )
        cbar.set_label(label='Fonte de onda de Rossby anomala (x10^-11 s^-2)', size=18)
        cbar.ax.tick_params(labelsize=20)

    dt_ini = ini_dt.strftime('%d-%m-%y')
    dt_fim = fim_dt.strftime('%d-%m-%y')
    titulo = (
        f'Fontes de Onda de Rossby (anomalia) — 200 hPa + vento divergente\n'
        f'(De {dt_ini} a {dt_fim}) — verde: fonte anticiclonica (HS); marrom: ciclonica'
    )
    ax.set_title(titulo, fontsize=12 if is_polar else 16, loc='left')

    logo_path = (
        None if settings.get('SEM_LOGO', False)
        else entrada_dir / ('logo_grec.png' if settings.get('LOGO_GREC', False) else 'novo_logo.png')
    )
    if logo_path is not None and logo_path.exists():
        _add_logo_to_map(ax=ax, logo_path=logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=500)

    logger.info(f'Salvando a figura {out_path}')
    plt.savefig(str(out_path), dpi=fig.dpi, bbox_inches='tight')
    plt.close('all')


def _plot_hovmoller(
    hov_v, hov_band, nome, ini_dt, fim_dt, total_days, entrada_dir, out_path, logger,
):
    """Hovmoller de v'200 na faixa do jato: bandas verticais=estacionaria, inclinadas=propagante.

    Eixo X em 0..360 (Pacifico inteiro no centro, data line em 180°), com rotulos W/E.
    """
    # cyclic point: fecha a costura repetindo lon=0 em lon=360 (faixa global continua)
    hov_v = xr.concat(
        [hov_v, hov_v.isel(lon=0).assign_coords(lon=360.0)], dim='lon',
    )
    lons_plot = hov_v['lon'].values  # 0..360
    times = hov_v['time'].values
    ytick_interval = 5 if total_days > 31 else 1

    cmap = LinearSegmentedColormap.from_list('anom', settings.LST_ANOM_CORRETA)

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.contourf(
        lons_plot, mdates.date2num(times), hov_v.values,
        levels=HOV_LEVELS, cmap=cmap, extend='both',
    )

    ax.yaxis.set_major_locator(mdates.DayLocator(interval=ytick_interval))
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    ax.invert_yaxis()

    # Ticks absolutos 0..360, rotulados em W/E (180 = data line, no centro)
    raw_ticks = np.arange(0, 361, 40)
    labels = [_fmt_lon(int(((t + 180) % 360) - 180)) for t in raw_ticks]
    ax.set_xticks(raw_ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(0, 360)
    ax.tick_params(axis='both', labelsize=15)

    ax.set_xlabel('Longitude', fontsize=16, labelpad=6)
    ax.set_ylabel('Data', fontsize=16, labelpad=6)

    cax = make_axes_locatable(ax).append_axes('right', size='3%', pad=0.15)
    cbar = fig.colorbar(im, cax=cax, ticks=HOV_TICKS)
    cbar.set_label("v'200 (m/s)", fontsize=14)
    cbar.ax.tick_params(labelsize=13)

    ini_fmt = ini_dt.strftime('%d/%m/%Y')
    fim_fmt = fim_dt.strftime('%d/%m/%Y')
    ax.set_title(
        f'Hovmöller — Anomalia de Vento Meridional 200 hPa ({hov_band})\n'
        f'{nome} — De {ini_fmt} a {fim_fmt} '
        f'(bandas verticais = onda estacionaria; inclinadas = propagante)',
        fontsize=12, loc='left',
    )

    logo_path = (
        None if settings.get('SEM_LOGO', False)
        else entrada_dir / ('logo_grec.png' if settings.get('LOGO_GREC', False) else 'novo_logo.png')
    )
    if logo_path is not None and logo_path.exists():
        _add_logo_to_map(ax=ax, logo_path=logo_path, zoom=0.55)

    logger.info(f'Salvando Hovmoller: {out_path}')
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
