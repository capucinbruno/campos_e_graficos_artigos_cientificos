"""s27 - Hovmöller: Anomalia de TSM (OISSTv2) e Vento Zonal 850 hPa (ERA5/GDAS).

Pipeline:
  1. Download OISST por ano (sst.day.anom.{year}.nc) — compartilhado com s11/s24
  2. Download ERA5/GDAS u/v 850 hPa (hibrido: ERA5 p/ historico, GDAS p/ recente)
  3. Climatologia u-zonal 850 mb via PSL composites (cache local por MM-DD)
  4. Anomalia diaria U850 = media_diaria_ERA5_GDAS - climatologia_PSL
  5. Hovmoller (media 5S-5N): SST shaded + isolinhas U850

Dominio: 5S-5N, 160E-80W (cruza o antimeridiano)

Saida:
    - PNG + NetCDF em Saida/s27_MONITORAMENTO_ENSO_HOVMOLLER/

Criado em: 2026-06-06
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.dates as mdates
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.util import add_cyclic_point as _acp
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from PIL import Image

from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.common.dataset_utils import arquivo_cobre_periodo
from app.common.download_helper import DownloadEngine, download_with_progress
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.common.logo_helper import resolve_logo_path
from app.common.logo_helper import proportional_logo_zoom
from app.src.uteis.chi200_intrasazonal import chi_from_wind
from app.src.uteis.clim_diaria_uv200_ltm import clim_uv200_daily
from app.src.uteis.clim_PSL_wnd_zonal_850 import get_clim_wnd_zonal_850_path
from app.src.uteis.clim_PSL_wnd_zonal_925 import get_clim_wnd_zonal_925_path
from app.src.uteis.downloaders_gdas_uv200 import ensure_gdas_uv200_for_period
from app.src.uteis.downloaders_gdas_uv850 import ensure_gdas_uv850_for_period
from app.src.uteis.downloaders_gdas_uv925 import ensure_gdas_uv925_for_period
from app.src.uteis.downloaders_wind200 import ensure_era5_uv200_for_period
from app.src.uteis.downloaders_wind850 import ensure_era5_uv850_for_period
from app.src.uteis.downloaders_wind925 import ensure_era5_uv925_for_period
from app.src.uteis.plot_olr_wind850_anom import main as plot_wind850_anom
from app.src.uteis.plot_olr_wind925_anom import main as plot_wind925_anom

# ---------------------------------------------------------------------------
# Identidade do script
# ---------------------------------------------------------------------------
SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's27'
SCRIPT_NAME = Path(__file__).stem
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME

# ---------------------------------------------------------------------------
# Constantes — OISST
# ---------------------------------------------------------------------------
OISST_URL_TPL = (
    'https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.day.anom.{year}.nc'
)

# ---------------------------------------------------------------------------
# Constantes — ERA5/GDAS hibrido
# ---------------------------------------------------------------------------
ERA5_LATENCY_DAYS = 7
DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)

# ---------------------------------------------------------------------------
# Constantes — dominio e plot
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = -5.0, 5.0
LON_MIN, LON_MAX = 160.0, -80.0  # 160E a 80W; cruza o antimeridiano

THRESH_U = 3.0  # m/s — modulo minimo da anomalia de vento a contornar
WIND_BASE = np.arange(2, 16, 2, dtype=float)  # 2, 4, ..., 14
# CHI200 (potencial de velocidade 200 hPa) — anomalia, escala 10^6 m²/s
CHI_SCALE = 1e6
CHI_LEVELS = np.arange(-10, 10.5, 1.0)  # ×10^6 m²/s (-10 a 10)
# Paleta verde/terra do CHI200 (a mesma do s31): verde/teal = ascensão (chi<0); marrom = subsidência (chi>0)
CHI200_COLORS = [
    '#005a45', '#0f7a6c', '#2e9b96', '#62bdb7', '#9dd8d2', '#dff3f1',
    '#f7f4eb', '#e7d9a9', '#d6b566', '#bd8a35', '#9a6313', '#6f4300',
]
CHI_CMAP = LinearSegmentedColormap.from_list('chi200', CHI200_COLORS, N=len(CHI_LEVELS) + 1)
# OLR (CPC Blended 2.5°, PSL) — anomalia (W/m²). BrBG_r: marrom = suprimido (OLR>0); verde = convecção (OLR<0)
OLR_URL = 'https://downloads.psl.noaa.gov/Datasets/cpc_blended_olr-2.5deg/olr.day.anom.nc'
OLR_FILE_NAME = 'olr.day.anom.nc'
OLR_LEVELS = list(np.arange(-60, 66, 6))
OLR_CMAP = 'BrBG_r'

# ---------------------------------------------------------------------------
# Constantes — mapa SSTA + vento 850 hPa
# ---------------------------------------------------------------------------
WIND850_FILE_NAME = 'wind850_anom.nc'
WIND925_FILE_NAME = 'wind925_anom.nc'
QUIVER_ENSO_STEP = 4      # pula N pontos do grid (maior = menos setas)
QUIVER_ENSO_SCALE = 200    # aumentar = setas menores; diminuir = setas maiores
QUIVER_ENSO_WIDTH = 0.002
QUIVER_ENSO_MIN_MAG = 0.5  # m/s — oculta vetores abaixo deste valor

# Boxes ENSO (lon/lat reais, -180..180) — regiões canônicas NOAA/CPC
ENSO_BOXES = {
    'Nino 1+2': {'lon_min': -90,  'lon_max': -80,  'lat_min': -10, 'lat_max': 0, 'wrap': False},
    'Nino 3':   {'lon_min': -150, 'lon_max': -90,  'lat_min': -5,  'lat_max': 5, 'wrap': False},
    'Nino 3.4': {'lon_min': -170, 'lon_max': -120, 'lat_min': -5,  'lat_max': 5, 'wrap': False},
    'Nino 4':   {'lon_min': 160,  'lon_max': -150, 'lat_min': -5,  'lat_max': 5, 'wrap': True},
}

# Overrides de cor por índice em lst_boxes — local neste script, sem tocar em settings.json
# idx 0 = Niño 1+2 (settings: 'r'), idx 2 = Niño 4 (settings: 'm')
_BOX_COLOR_OVERRIDE = {0: 'limegreen', 2: 'magenta'}
_NINO4_COLOR = 'magenta'


# ---------------------------------------------------------------------------
# Utilitarios de grade (adaptados de plot_olr_wind850_anom.py)
# ---------------------------------------------------------------------------
def _ensure_lon180(da: xr.DataArray) -> xr.DataArray:
    if 'lon' in da.coords:
        lon = da['lon'].values
        if np.any(lon > 180):
            da = da.assign_coords(lon=((lon + 180) % 360) - 180).sortby('lon')
    return da


def _rename_std_latlon(obj):
    ren = {}
    for name in list(obj.dims) + list(obj.coords):
        low = name.lower()
        if low == 'latitude' and 'lat' not in obj.dims:
            ren[name] = 'lat'
        elif low == 'longitude' and 'lon' not in obj.dims:
            ren[name] = 'lon'
    return obj.rename(ren) if ren else obj


def _ensure_time_coord(obj):
    if hasattr(obj, 'dims') and 'time' not in obj.dims and 'valid_time' in obj.dims:
        obj = obj.rename({'valid_time': 'time'})
    elif hasattr(obj, 'coords') and 'time' not in obj.coords and 'valid_time' in obj.coords:
        obj = obj.rename({'valid_time': 'time'})
    if 'time' not in obj.coords:
        raise KeyError("Nem 'time' nem 'valid_time' encontrados.")
    return obj


def _drop_expver(ds: xr.Dataset) -> xr.Dataset:
    ren = {}
    for d in ds.dims:
        if d.lower() == 'expver' and d != 'expver':
            ren[d] = 'expver'
        elif d.lower() == 'number' and d != 'number':
            ren[d] = 'number'
    if ren:
        ds = ds.rename(ren)
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


def _find_u_var(ds: xr.Dataset) -> str:
    for name in ('u', 'u_component_of_wind', 'U_GRD_L100', 'uwnd'):
        if name in ds.data_vars:
            return name
    raise KeyError(f"Variavel u nao encontrada em dataset. Disponiveis: {list(ds.data_vars)}")


def _sel_lon_wrap(da: xr.DataArray, lon_min: float, lon_max: float) -> xr.DataArray:
    """Seleciona longitudes cruzando o antimeridiano; retorna coords monotônicas (ex: 160..280).

    Sem isso, o concat produz [160..180, -180..-80] — não-monotônico — e o matplotlib
    interrompe as isolinhas na junção dos dois pedaços.
    """
    if lon_min <= lon_max:
        return da.sel(lon=slice(lon_min, lon_max))
    da1 = da.sel(lon=slice(lon_min, 180.0))
    da2 = da.sel(lon=slice(-180.0, lon_max))
    # Desloca da2 para [180..360] tornando as coords monotônicas
    da2 = da2.assign_coords(lon=da2['lon'].values + 360.0)
    combined = xr.concat([da1, da2], dim='lon')
    # Remove possível duplicata em lon=180 (fim de da1 e início de da2 após +360)
    lons = combined['lon'].values
    _, uniq = np.unique(lons, return_index=True)
    if len(uniq) < len(lons):
        combined = combined.isel(lon=uniq)
    return combined


def _wrap_lon_for_plot(lon_vals: np.ndarray, anchor: float) -> np.ndarray:
    return np.where(lon_vals < anchor, lon_vals + 360.0, lon_vals)


def _fmt_lon(v: int) -> str:
    if v == 0 or abs(v) == 180:
        return f'{abs(v)}°'
    return f'{v}°E' if v > 0 else f'{-v}°W'


def _xticks_wrap(lon_min: float, lon_max: float, step: int = 10):
    ticks = np.arange(lon_min, lon_max + 360 + 1, step)
    raw = [int(((t + 180) % 360) - 180) for t in ticks]
    labels = [_fmt_lon(v) for v in raw]
    return ticks, labels


_LOGO_CORNERS = {  # canto -> (xy em transAxes, box_alignment, sinal do offset p/ empurrar p/ DENTRO)
    'lower-left': ((0, 0), (0, 0), (1, 1)),
    'upper-right': ((1, 1), (1, 1), (-1, -1)),
}


def _add_logo_to_map(ax, logo_path, zoom=0.55, xoffset=10, yoffset=10, zorder=3000,
                     corner='lower-left'):
    logo = Image.open(logo_path).convert('RGBA')
    bbox = logo.getbbox()
    if bbox is not None:
        logo = logo.crop(bbox)
    imagebox = OffsetImage(np.array(logo), zoom=proportional_logo_zoom(ax, np.array(logo).shape[1]))
    xy, box_align, (sx, sy) = _LOGO_CORNERS.get(corner, _LOGO_CORNERS['lower-left'])
    ab = AnnotationBbox(
        imagebox, xy, xycoords=ax.transAxes,
        xybox=(sx * xoffset, sy * yoffset), boxcoords='offset points',
        box_alignment=box_align, frameon=False, pad=0, zorder=zorder, clip_on=False,
    )
    ax.add_artist(ab)


def _box_mean(
    arr: np.ndarray, lon: np.ndarray, lat: np.ndarray,
    lon_min: float, lon_max: float, lat_min: float, lat_max: float,
    wrap: bool = False,
) -> float:
    """Média de `arr` (lat × lon, coords em -180..180) dentro de um box.

    `wrap=True` para boxes que cruzam a linha de data (ex: Niño 4).
    """
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    if wrap:
        lon_sel = (lon >= lon_min) | (lon <= lon_max)
    else:
        lon_sel = (lon >= lon_min) & (lon <= lon_max)
    lat_sel = (lat >= lat_min) & (lat <= lat_max)
    return float(np.nanmean(arr[np.ix_(lat_sel, lon_sel)]))


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


# ---------------------------------------------------------------------------
# Download OISST — identico ao s11
# ---------------------------------------------------------------------------
def _download_sst_anos(
    dados_dir: Path,
    start_date: np.datetime64,
    end_date: np.datetime64,
    logger,
) -> list[Path]:
    current_year = datetime.now().year
    start_dt = pd.Timestamp(start_date).to_pydatetime()
    end_dt = pd.Timestamp(end_date).to_pydatetime()
    anos = list(range(start_dt.year, end_dt.year + 1))
    paths = []
    for year in anos:
        url = OISST_URL_TPL.format(year=year)
        sst_path = dados_dir / f'sst.day.anom.{year}.nc'

        if year < current_year and sst_path.exists():
            logger.info(f'Arquivo SST {year} ja existe localmente — pulando download')
            paths.append(sst_path)
            continue

        year_start_needed = np.datetime64(f'{year}-01-01')
        year_end_needed = (
            end_date if year == end_dt.year
            else np.datetime64(f'{year}-12-31')
        )

        if arquivo_cobre_periodo(sst_path, year_start_needed, year_end_needed):
            logger.info(f'Arquivo SST {year} ja cobre o periodo ate {year_end_needed} — pulando download')
            paths.append(sst_path)
            continue

        if sst_path.exists():
            logger.info(f'Arquivo SST {year} desatualizado — rebaixando')
        download_with_progress(
            url=url,
            output_path=str(sst_path),
            description=f'SST anomalia {year}',
            max_retries=5,
            force=sst_path.exists(),
            engine=DownloadEngine.AUTO,
        )
        paths.append(sst_path)
    return paths


# ---------------------------------------------------------------------------
# Processamento diario u850 com ERA5/GDAS
# ---------------------------------------------------------------------------
def _load_daily_u850(
    files: list[Path],
    dt_ini: datetime,
    dt_fim: datetime,
) -> xr.DataArray:
    """
    Carrega u850 de arquivos ERA5/GDAS, calcula media diaria e retorna
    DataArray com dims (time, lat, lon) em -180..180.
    """
    parts = []
    for fp in files:
        ds = xr.open_dataset(str(fp), engine='netcdf4')
        ds = _ensure_time_coord(ds)
        ds = _drop_expver(ds)
        ds = _rename_std_latlon(ds)
        ds = _ensure_lon180(ds)
        ds = _sort_dedup_time(ds)

        u_var = _find_u_var(ds)
        da_u = ds[u_var]

        # Remove dimensao de nivel se presente
        for dim in ('pressure_level', 'isobaricInhPa', 'level'):
            if dim in da_u.dims:
                da_u = da_u.isel({dim: 0}, drop=True)

        # Remove coordenadas não-dimensionais que conflitam entre ERA5 e GDAS
        for coord in ('valid_time', 'step', 'expver', 'number'):
            if coord in da_u.coords and coord not in da_u.dims:
                da_u = da_u.drop_vars(coord)

        # Seleciona apenas horas sinoticas
        da_u = da_u.sel(time=da_u['time'].dt.hour.isin(list(DEFAULT_SYNOPTIC_HOURS)))
        parts.append(da_u)
        ds.close()

    # ERA5 (1°, 181 lat) e GDAS (0.25°, 721 lat) têm grades diferentes:
    # interpola todos para a grade de referencia (primeiro arquivo = ERA5)
    if len(parts) > 1:
        ref_lat = parts[0]['lat'].values
        ref_lon = parts[0]['lon'].values
        parts = [
            p.interp(lat=ref_lat, lon=ref_lon, method='linear')
            if p['lat'].size != ref_lat.size
            else p
            for p in parts
        ]

    da_all = xr.concat(parts, dim='time', join='override').sortby('time')

    # Seleciona o periodo exato
    t0 = np.datetime64(dt_ini.date())
    t1 = np.datetime64(dt_fim.date())
    da_all = da_all.sel(time=slice(t0, t1))

    # Media diaria; garante lat ascendente para sel(lat=slice(min, max)) funcionar
    da_daily = da_all.resample(time='1D').mean(skipna=True)
    da_daily = da_daily.sortby('lat')
    return da_daily


def _load_psl_clim_u(path: Path, ref_lat: np.ndarray, ref_lon: np.ndarray) -> xr.DataArray:
    """Carrega climatologia PSL u-zonal 850mb e interpola para o grid ERA5/GDAS."""
    ds = xr.open_dataset(str(path), engine='netcdf4')
    ds = _rename_std_latlon(ds)

    da = None
    for vname in ('uwnd', 'u', 'u_component_of_wind'):
        if vname in ds.data_vars:
            da = ds[vname]
            break
    if da is None:
        da = next(iter(ds.data_vars.values()))

    # Remove dimensoes extras
    for dim in ('time', 'level', 'isobaricInhPa', 'pressure_level'):
        if dim in da.dims:
            da = da.isel({dim: 0}, drop=True)

    # Converte para -180..180
    ds_tmp = _ensure_lon180(da.to_dataset(name='u'))
    da = ds_tmp['u']
    ds.close()

    # Interpola PSL (2.5°) para o grid ERA5/GDAS (1.0°) via add_cyclic_point
    clim_vals_cyc, clim_lon_cyc = _acp(da.values, coord=da['lon'].values)
    clim_cyc = xr.DataArray(
        clim_vals_cyc,
        dims=['lat', 'lon'],
        coords={'lat': da['lat'].values, 'lon': clim_lon_cyc},
    )
    return clim_cyc.interp(lat=ref_lat, lon=ref_lon, method='linear')


def _find_v_var(ds: xr.Dataset) -> str:
    for name in ('v', 'v_component_of_wind', 'V_GRD_L100', 'vwnd'):
        if name in ds.data_vars:
            return name
    raise KeyError(f'v não encontrado. Disponíveis: {list(ds.data_vars)}')


def _pentad_smooth(hov: xr.DataArray, dias: int) -> xr.DataArray:
    """Média móvel de `dias` (pêntada=5) ao longo do tempo. dias<=1 -> sem suavização."""
    return hov.rolling(time=int(dias), center=True, min_periods=1).mean() if dias and dias > 1 else hov


def _hov_domain_mean(da_anom: xr.DataArray, ref_hov: xr.DataArray) -> xr.DataArray:
    """Recorta o domínio ENSO (5S-5N, 160E-80W), média na latitude e alinha lon/time ao ref_hov."""
    box = _sel_lon_wrap(da_anom, LON_MIN, LON_MAX).sel(lat=slice(LAT_MIN, LAT_MAX))
    hov = box.mean(dim='lat', skipna=True)
    return hov.interp(lon=ref_hov['lon']).interp(time=ref_hov['time'])


def _load_daily_uv200_on_grid(files, dt_ini, dt_fim, tgt_lat, tgt_lon):
    """u/v 200 (ERA5/GDAS) -> média diária -> interpola p/ (tgt_lat, tgt_lon), lon 0..360.
    Retorna (u_da, v_da) em (time, lat, lon)."""
    tgt_lat_da = xr.DataArray(np.asarray(tgt_lat), dims=['lat'])
    tgt_lon_da = xr.DataArray(np.asarray(tgt_lon), dims=['lon'])
    t0, t1 = np.datetime64(dt_ini.date()), np.datetime64(dt_fim.date())
    us, vs = [], []
    for fp in files:
        ds = xr.open_dataset(str(fp), engine='netcdf4')
        ds = _sort_dedup_time(_rename_std_latlon(_drop_expver(_ensure_time_coord(ds))))
        ds = ds.assign_coords(lon=(ds['lon'] % 360)).sortby('lon').sortby('lat')
        da_u, da_v = ds[_find_u_var(ds)], ds[_find_v_var(ds)]
        for dim in ('pressure_level', 'isobaricInhPa', 'level'):
            if dim in da_u.dims:
                da_u = da_u.isel({dim: 0}, drop=True)
                da_v = da_v.isel({dim: 0}, drop=True)
        for coord in ('valid_time', 'step', 'expver', 'number'):
            if coord in da_u.coords and coord not in da_u.dims:
                da_u = da_u.drop_vars(coord)
            if coord in da_v.coords and coord not in da_v.dims:
                da_v = da_v.drop_vars(coord)
        m = da_u['time'].dt.hour.isin(list(DEFAULT_SYNOPTIC_HOURS))
        da_u, da_v = da_u.sel(time=m).sel(time=slice(t0, t1)), da_v.sel(time=m).sel(time=slice(t0, t1))
        if da_u.sizes.get('time', 0) == 0:
            ds.close()
            continue
        da_u = da_u.resample(time='1D').mean(skipna=True).interp(
            lat=tgt_lat_da, lon=tgt_lon_da, method='linear').reset_coords(drop=True)
        da_v = da_v.resample(time='1D').mean(skipna=True).interp(
            lat=tgt_lat_da, lon=tgt_lon_da, method='linear').reset_coords(drop=True)
        us.append(da_u.load())
        vs.append(da_v.load())
        ds.close()
    if not us:
        raise RuntimeError('Nenhum dado u/v 200 válido no período.')
    u_da = xr.concat(us, dim='time', join='override').sortby('time')
    v_da = xr.concat(vs, dim='time', join='override').sortby('time')
    _, uniq = np.unique(u_da['time'].values, return_index=True)
    return u_da.isel(time=uniq), v_da.isel(time=uniq)


def _plot_hov_panel(hov_shaded, hov_contour, lons, times, ytick_interval, *, cmap, levels,
                    cbar_label, titulo, out_png, entrada_dir, logger):
    """Plota um Hovmöller no estilo do s27: campo `hov_shaded` (contourf) + isolinhas de vento."""
    crosses = LON_MIN > LON_MAX
    x_min, x_max = (LON_MIN, LON_MAX + 360) if crosses else (LON_MIN, LON_MAX)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.contourf(lons, mdates.date2num(times), hov_shaded, levels=levels, cmap=cmap, extend='both')
    wind_pos = WIND_BASE[WIND_BASE >= THRESH_U]
    wind_neg = -wind_pos[::-1]
    if wind_neg.size:
        ax.contour(lons, mdates.date2num(times), hov_contour, levels=wind_neg,
                   colors='blue', linestyles='dashed', linewidths=1.5)
    if wind_pos.size:
        ax.contour(lons, mdates.date2num(times), hov_contour, levels=wind_pos,
                   colors='darkred', linestyles='solid', linewidths=1.5)
    ax.yaxis.set_major_locator(mdates.DayLocator(interval=ytick_interval))
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    ax.invert_yaxis()
    ax.set_xlim(x_min, x_max)
    if crosses:
        ticks, labels = _xticks_wrap(LON_MIN, LON_MAX, step=10)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    else:
        raw_ticks = np.arange(LON_MIN, LON_MAX + 1, 10)
        ax.set_xticks(raw_ticks)
        ax.set_xticklabels([_fmt_lon(int(v)) for v in raw_ticks])
    ax.set_xlabel('Longitude', fontsize=14, labelpad=6)
    ax.set_ylabel('Data', fontsize=14, labelpad=6)
    cax = make_axes_locatable(ax).append_axes('right', size='3%', pad=0.15)
    cbar = fig.colorbar(im, cax=cax, ticks=levels[::2])
    cbar.set_label(cbar_label, fontsize=12)
    ax.set_title(titulo, fontsize=14, loc='left')
    logo_path = resolve_logo_path(entrada_dir)
    if logo_path is not None and logo_path.exists():
        _add_logo_to_map(ax=ax, logo_path=logo_path, corner='upper-right')
    logger.info(f'Salvando figura: {out_png}')
    plt.savefig(str(out_png), dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_enso_map(shaded_cyc, shaded_lon_cyc, shaded_lat, *, levels, cmap, cbar_label,
                   box_means, box_unit, box_fmt, u_cyc, v_cyc, lon_wind_cyc, lat_wind,
                   titulo, out_png, area_cfg, entrada_dir, logo_path, logger,
                   show_box_value: bool = True):
    """Mapa da área ENSO: campo `shaded` (contourf) + boxes Niño + quiver de vento anômalo + labels.

    show_box_value=False -> rotulo so com o NOME do box (sem o valor); usado nos mapas de OLR."""
    central_lon_mapa = int(area_cfg['central_longitude_mapa'])
    central_lon_plot = int(area_cfg.get('central_longitude_plot', 0))
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=central_lon_mapa))
    ax.set_xlim([area_cfg['lon_esq'], area_cfg['lon_dir']])
    ax.set_ylim([area_cfg['lat_inf'], area_cfg['lat_sup']])
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='whitesmoke')
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1.0, zorder=100)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, zorder=100)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, zorder=100)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='white')
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.0)
    gl.top_labels = gl.right_labels = False
    gl.xlocator = mticker.MultipleLocator(20)
    gl.xlabel_style = {'size': 14, 'color': 'black'}
    gl.ylabel_style = {'size': 14, 'color': 'black'}
    im = ax.contourf(shaded_lon_cyc, shaded_lat, shaded_cyc, levels=levels, cmap=cmap, extend='both',
                     transform=ccrs.PlateCarree(central_longitude=central_lon_plot))
    for i, box in enumerate(area_cfg.get('lst_boxes', [])):
        edgecolor = _BOX_COLOR_OVERRIDE.get(i, box['edgecolor'])
        ax.add_patch(patches.Rectangle(
            (box['x_anc'], box['y_anc']), box['x_larg'], box['y_larg'],
            linewidth=box['linewidth'], edgecolor=edgecolor, facecolor='none', zorder=300))
    step = QUIVER_ENSO_STEP
    lon_q, lat_q = lon_wind_cyc[::step], lat_wind[::step]
    u_q, v_q = u_cyc[::step, ::step].copy(), v_cyc[::step, ::step].copy()
    mag = np.sqrt(u_q**2 + v_q**2)
    u_q = np.where(mag < QUIVER_ENSO_MIN_MAG, np.nan, u_q)
    v_q = np.where(mag < QUIVER_ENSO_MIN_MAG, np.nan, v_q)
    ax.quiver(lon_q, lat_q, u_q, v_q, transform=ccrs.PlateCarree(central_longitude=central_lon_plot),
              scale=QUIVER_ENSO_SCALE, scale_units='width', width=QUIVER_ENSO_WIDTH,
              color='black', zorder=200)
    boxes_cfg = area_cfg.get('lst_boxes', [])
    for txt, box_idx, y, cor in [('Nino 1+2', 0, -13.64, 'limegreen'), ('Nino 3', 1, 8.45, 'blue'),
                                 ('Nino 3.4', 3, -9.45, 'black'), ('Nino 4', 2, 8.45, _NINO4_COLOR)]:
        if box_idx >= len(boxes_cfg):
            continue
        box = boxes_cfg[box_idx]
        cx = box['x_anc'] + box['x_larg'] / 2     # centro em x do box
        val = box_means.get(txt)
        # show_box_value=False (OLR) -> so o NOME; senao "Nome = valor unidade". Sempre centralizado.
        label = (f'{txt} = {val:{box_fmt}}{box_unit}'
                 if show_box_value and val is not None and np.isfinite(val) else txt)
        t = ax.text(cx, y, label, fontsize=14, color=cor, weight='bold', ha='center', zorder=400)
        fg = 'black' if cor in {'limegreen', 'magenta'} else 'white'
        t.set_path_effects([path_effects.Stroke(linewidth=3, foreground=fg), path_effects.Normal()])
    cax = make_axes_locatable(ax).append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal', extend='both', location='bottom',
                        ticks=levels[::2])
    cbar.set_label(cbar_label, fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    ax.set_title(titulo, fontsize=14, loc='left')
    if logo_path is not None and logo_path.exists():
        _add_logo_to_map(ax=ax, logo_path=logo_path)
    logger.info(f'Salvando figura: {out_png}')
    plt.savefig(str(out_png), dpi=300, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger = get_logger(SCRIPT_ID)

    logger.info('=' * 80)
    logger.info(f'SCRIPT {SCRIPT_ID.upper()}: {SCRIPT_DESC}')
    logger.info('=' * 80)

    ini_str = str(settings.DATA_INICIAL)
    fim_str = str(settings.DATA_FINAL)
    ini_dt = datetime.fromisoformat(ini_str)
    fim_dt = datetime.fromisoformat(fim_str)

    output_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_MONITORAMENTO_ENSO_HOVMOLLER'
    dados_dir = Path(settings.DIR_DADOS)
    entrada_dir = Path(settings.DIR_INPUT)

    suaviz = int(settings.get('HOVMOLLER_SUAVIZACAO_DIAS', 5))  # pêntada (5d); 0 = sem suavização
    png_name = f'hovmoller_sst_u850_ENSO_{ini_str}_to_{fim_str}.png'
    nc_name = f'hovmoller_sst_u850_ENSO_{ini_str}_to_{fim_str}.nc'
    enso_map_name = f'ssta_vento850_ENSO_{ini_str}_to_{fim_str}.png'
    png_sst_u925_name = f'hovmoller_sst_u925_ENSO_{ini_str}_to_{fim_str}.png'
    png_chi_u925_name = f'hovmoller_chi200_u925_ENSO_{ini_str}_to_{fim_str}.png'
    png_olr_u925_name = f'hovmoller_olr_u925_ENSO_{ini_str}_to_{fim_str}.png'
    png_olr_u850_name = f'hovmoller_olr_u850_ENSO_{ini_str}_to_{fim_str}.png'
    png_chi_u850_name = f'hovmoller_chi200_u850_ENSO_{ini_str}_to_{fim_str}.png'
    png_sst_u925_raw_name = f'hovmoller_sst_u925_sem_mm_ENSO_{ini_str}_to_{fim_str}.png'
    enso_map_sst925_name = f'ssta_vento925_ENSO_{ini_str}_to_{fim_str}.png'
    enso_map_olr925_name = f'olr_vento925_ENSO_{ini_str}_to_{fim_str}.png'
    enso_map_olr850_name = f'olr_vento850_ENSO_{ini_str}_to_{fim_str}.png'
    output_files = [
        str(output_dir / png_name),
        str(output_dir / enso_map_name),
        str(output_dir / png_sst_u925_name),
        str(output_dir / png_chi_u925_name),
        str(output_dir / png_olr_u925_name),
        str(output_dir / png_olr_u850_name),
        str(output_dir / png_chi_u850_name),
        str(output_dir / png_sst_u925_raw_name),
        str(output_dir / enso_map_sst925_name),
        str(output_dir / enso_map_olr925_name),
        str(output_dir / enso_map_olr850_name),
    ]

    cache_params = {
        'DATA_INICIAL': ini_str,
        'DATA_FINAL': fim_str,
        'lat_min': LAT_MIN,
        'lat_max': LAT_MAX,
        'lon_min': LON_MIN,
        'lon_max': LON_MAX,
        'thresh_u': THRESH_U,
        'suavizacao_dias': suaviz,
        'script_version': '1.13',  # mapas OLR: rotulo do box so com o nome (sem W/m2), centralizado
    }

    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('CACHE VALIDO! Execucao ja foi realizada com os mesmos parametros.')
        logger.info(f'   Periodo: {ini_str} a {fim_str}')
        logger.info('   Pulando execucao')
        return

    start_time = time.time()
    total_days = (fim_dt.date() - ini_dt.date()).days + 1
    ytick_interval = 5 if total_days > 31 else 1
    logger.info(f'Periodo: {ini_str} a {fim_str} ({total_days} dias)')

    output_dir.mkdir(parents=True, exist_ok=True)
    dados_dir.mkdir(parents=True, exist_ok=True)

    start_date = np.datetime64(ini_str)
    end_date = np.datetime64(fim_str)

    # ---- Etapa 1: OISST anomalia diaria ----
    logger.info(f'Etapa 1: OISST anomalia diaria...')
    sst_paths = _download_sst_anos(dados_dir, start_date, end_date, logger)

    sst_datasets = [xr.open_dataset(str(p)) for p in sst_paths]
    ds_sst = (
        sst_datasets[0] if len(sst_datasets) == 1
        else xr.concat(sst_datasets, dim='time').sortby('time')
    )
    da_sst = ds_sst['anom'].sel(time=slice(start_date, end_date))
    da_sst = _ensure_lon180(da_sst).sortby('lat')
    da_sst_box = _sel_lon_wrap(da_sst, LON_MIN, LON_MAX).sel(lat=slice(LAT_MIN, LAT_MAX))
    hov_sst = da_sst_box.mean(dim='lat', skipna=True)
    if hov_sst.size == 0 or np.all(np.isnan(hov_sst.values)):
        raise ValueError('SST Hovmoller vazio/NaN. Verifique datas e recortes.')

    # ---- Etapa 2: ERA5/GDAS u/v 850 hPa ----
    era5_period, gdas_period = _get_data_sources(ini_dt, fim_dt)
    if era5_period:
        logger.info(f'Etapa 2a: ERA5 u/v 850 hPa ({era5_period[0].date()} a {era5_period[1].date()})...')
    if gdas_period:
        logger.info(f'Etapa 2b: GDAS u/v 850mb ({gdas_period[0].date()} a {gdas_period[1].date()})...')

    all_files: list[Path] = []
    if era5_period:
        all_files.extend(ensure_era5_uv850_for_period(
            start=era5_period[0], end=era5_period[1],
            hours_utc=list(DEFAULT_SYNOPTIC_HOURS),
        ))
    if gdas_period:
        all_files.extend(ensure_gdas_uv850_for_period(
            start=gdas_period[0], end=gdas_period[1],
        ))

    logger.info(f'Etapa 2: Calculando media diaria u850...')
    da_u_daily = _load_daily_u850(all_files, ini_dt, fim_dt)

    # ---- Etapa 3: Climatologia PSL u-zonal 850mb ----
    logger.info('Etapa 3: Climatologia PSL u-zonal 850mb...')
    clim_path = get_clim_wnd_zonal_850_path(ini_str, fim_str)
    ref_lat = da_u_daily['lat'].values
    ref_lon = da_u_daily['lon'].values
    clim_u = _load_psl_clim_u(clim_path, ref_lat, ref_lon)

    # ---- Etapa 4: Anomalia diaria u850 ----
    logger.info('Etapa 4: Anomalia diaria u850 = ERA5/GDAS - climatologia PSL...')
    da_u_anom = da_u_daily - clim_u  # broadcasting (time, lat, lon) - (lat, lon)

    # ---- Hovmoller u850 ----
    da_u_box = _sel_lon_wrap(da_u_anom, LON_MIN, LON_MAX).sel(lat=slice(LAT_MIN, LAT_MAX))
    hov_u = da_u_box.mean(dim='lat', skipna=True)

    # Alinha longitude com SST
    hov_u = hov_u.interp(lon=hov_sst['lon'])
    # Alinha tempo (interpola se necessario)
    hov_u = hov_u.interp(time=hov_sst['time'])

    logger.info(
        f'Hovmoller SST: {hov_sst.shape} | Hovmoller U850: {hov_u.shape}'
    )

    # ---- Salvar NetCDF ----
    ds_out = xr.Dataset({'sst_anom_hov': hov_sst, 'u850_anom_hov': hov_u})
    ds_out['sst_anom_hov'].attrs.update({'long_name': 'Anomalia TSM Hovmoller (5S-5N)', 'units': 'degC'})
    ds_out['u850_anom_hov'].attrs.update({'long_name': 'Anomalia U850 Hovmoller (5S-5N)', 'units': 'm s-1'})
    nc_path = output_dir / nc_name
    logger.info(f'Salvando NetCDF: {nc_path}')
    ds_out.to_netcdf(str(nc_path))

    # ---- Plot ----
    lons = hov_sst['lon'].values
    times = hov_sst['time'].values

    # _sel_lon_wrap já retorna coords monotônicas (160..280 para ENSO)
    lons_plot = lons
    crosses = LON_MIN > LON_MAX
    x_min, x_max = (LON_MIN, LON_MAX + 360) if crosses else (LON_MIN, LON_MAX)

    cmap = LinearSegmentedColormap.from_list('sst_anom', settings.LST_ANOM_CORRETA)
    levels = list(settings.LST_SSTA_NEW_GREC)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Camada 1: SST shaded
    im = ax.contourf(
        lons_plot,
        mdates.date2num(times),
        hov_sst.values,
        levels=levels,
        cmap=cmap,
        extend='both',
    )

    # Camada 2: isolinhas U850 (azul=negativo, vermelho=positivo)
    wind_pos = WIND_BASE[WIND_BASE >= THRESH_U]
    wind_neg = -wind_pos[::-1]

    if wind_neg.size:
        ax.contour(
            lons_plot, mdates.date2num(times), hov_u.values,
            levels=wind_neg, colors='blue', linestyles='dashed', linewidths=1.5,
        )
    if wind_pos.size:
        ax.contour(
            lons_plot, mdates.date2num(times), hov_u.values,
            levels=wind_pos, colors='darkred', linestyles='solid', linewidths=1.5,
        )

    # Eixo Y — datas com passo dinamico
    ax.yaxis.set_major_locator(mdates.DayLocator(interval=ytick_interval))
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    ax.invert_yaxis()

    # Eixo X — longitudes (com wrap se necessario)
    ax.set_xlim(x_min, x_max)
    if crosses:
        ticks, labels = _xticks_wrap(LON_MIN, LON_MAX, step=10)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    else:
        raw_ticks = np.arange(LON_MIN, LON_MAX + 1, 10)
        ax.set_xticks(raw_ticks)
        ax.set_xticklabels([_fmt_lon(int(v)) for v in raw_ticks])

    ax.set_xlabel('Longitude', fontsize=14, labelpad=6)
    ax.set_ylabel('Data', fontsize=14, labelpad=6)

    # Colorbar SST (vertical, direita)
    cax = make_axes_locatable(ax).append_axes('right', size='3%', pad=0.15)
    cbar = fig.colorbar(im, cax=cax, ticks=levels[::2])
    cbar.set_label('°C', fontsize=12)

    # Titulo
    ini_fmt = ini_dt.strftime('%d/%m/%Y')
    fim_fmt = fim_dt.strftime('%d/%m/%Y')
    ax.set_title(
        f'Hovmöller — Anomalia de TSM e Vento Zonal 850 hPa (5°S–5°N)\n'
        f'De {ini_fmt} a {fim_fmt}',
        fontsize=14, loc='left',
    )

    # Logo
    logo_path = resolve_logo_path(entrada_dir)
    if logo_path is not None and logo_path.exists():
        _add_logo_to_map(ax=ax, logo_path=logo_path, corner='upper-right')

    # Salvar figura
    fig_path = output_dir / png_name
    logger.info(f'Salvando figura: {fig_path}')
    plt.savefig(str(fig_path), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ---- Hovmollers NOVOS (U925, pentada): SST+U925 e CHI200+U925 ----
    # U925 anomalia (ERA5/GDAS - climatologia PSL 925), mesmo dominio/estilo do U850
    logger.info('Etapa 4b: u 925 hPa (ERA5/GDAS) + anomalia (clim PSL 925)...')
    files925: list[Path] = []
    if era5_period:
        files925.extend(ensure_era5_uv925_for_period(
            start=era5_period[0], end=era5_period[1], hours_utc=list(DEFAULT_SYNOPTIC_HOURS)))
    if gdas_period:
        files925.extend(ensure_gdas_uv925_for_period(start=gdas_period[0], end=gdas_period[1]))
    da_u925_daily = _load_daily_u850(files925, ini_dt, fim_dt)  # loader e agnostico de nivel
    clim925_path = get_clim_wnd_zonal_925_path(ini_str, fim_str)
    clim_u925 = _load_psl_clim_u(clim925_path, da_u925_daily['lat'].values, da_u925_daily['lon'].values)
    hov_u925_raw = _hov_domain_mean(da_u925_daily - clim_u925, hov_sst)
    hov_u925_p = _pentad_smooth(hov_u925_raw, suaviz)

    logger.info('Hovmoller 2/3: SST + U925 (media movel {}d)...', suaviz)
    _plot_hov_panel(
        _pentad_smooth(hov_sst, suaviz).values, hov_u925_p.values, lons_plot, times, ytick_interval,
        cmap=cmap, levels=levels, cbar_label='°C',
        titulo=(f'Hovmöller — Anomalia de TSM e Vento Zonal 925 hPa (5°S–5°N) — média móvel {suaviz}d\n'
                f'De {ini_fmt} a {fim_fmt}'),
        out_png=output_dir / png_sst_u925_name, entrada_dir=entrada_dir, logger=logger)

    # CHI200 anomalia (u/v200 -> anomalia vs LTM NCEP -> Poisson) no dominio ENSO
    logger.info('Etapa 4c: CHI200 (u/v200 -> anomalia vs LTM -> Poisson)...')
    files200: list[Path] = []
    if era5_period:
        files200.extend(ensure_era5_uv200_for_period(
            start=era5_period[0], end=era5_period[1], hours_utc=list(DEFAULT_SYNOPTIC_HOURS)))
    if gdas_period:
        files200.extend(ensure_gdas_uv200_for_period(start=gdas_period[0], end=gdas_period[1]))
    _, _, ltm_lat0, ltm_lon = clim_uv200_daily(np.array([np.datetime64(fim_dt.date())]))
    order = np.argsort(ltm_lat0)
    ltm_lat = ltm_lat0[order]
    u200, v200 = _load_daily_uv200_on_grid(files200, ini_dt, fim_dt, ltm_lat, ltm_lon)
    dts200 = np.array([np.datetime64(pd.Timestamp(t).date()) for t in u200['time'].values])
    u_clim, v_clim, _, _ = clim_uv200_daily(dts200)
    u_anom = u200.values - u_clim[:, order, :]
    v_anom = v200.values - v_clim[:, order, :]
    chi = np.stack([chi_from_wind(u_anom[k], v_anom[k], ltm_lat, ltm_lon)
                    for k in range(u_anom.shape[0])]) / CHI_SCALE
    chi_da = _ensure_lon180(xr.DataArray(
        chi, dims=('time', 'lat', 'lon'),
        coords={'time': u200['time'].values, 'lat': ltm_lat, 'lon': ltm_lon})).sortby('lat')
    hov_chi_p = _pentad_smooth(_hov_domain_mean(chi_da, hov_sst), suaviz)

    logger.info('Hovmoller 3/3: CHI200 + U925 (media movel {}d)...', suaviz)
    _plot_hov_panel(
        hov_chi_p.values, hov_u925_p.values, lons_plot, times, ytick_interval,
        cmap=CHI_CMAP, levels=list(CHI_LEVELS), cbar_label='CHI200 anomalia (×10⁶ m²/s)',
        titulo=(f'Hovmöller — Anomalia de CHI200 (200 hPa) e Vento Zonal 925 hPa (5°S–5°N) — média móvel {suaviz}d\n'
                f'De {ini_fmt} a {fim_fmt}'),
        out_png=output_dir / png_chi_u925_name, entrada_dir=entrada_dir, logger=logger)

    # OLR anomalia (CPC Blended 2.5°, PSL) + U925
    logger.info('Etapa 4d: OLR (CPC Blended PSL) anomalia + U925...')
    olr_path = dados_dir / OLR_FILE_NAME
    # OLR tem latencia de alguns dias; exige cobertura so ate end-7 p/ nao re-baixar toda rodada
    if not arquivo_cobre_periodo(olr_path, start_date, end_date - np.timedelta64(7, 'D')):
        download_with_progress(url=OLR_URL, output_path=str(olr_path), description=OLR_FILE_NAME,
                               max_retries=5, force=olr_path.exists(),
                               engine=DownloadEngine.ARIA2, timeout=300)
    ds_olr = xr.open_dataset(str(olr_path))
    da_olr = _ensure_lon180(
        _rename_std_latlon(ds_olr)['olr'].sel(time=slice(start_date, end_date))).sortby('lat').load()
    hov_olr_p = _pentad_smooth(_hov_domain_mean(da_olr, hov_sst), suaviz)
    ds_olr.close()

    logger.info('Hovmoller 4/4: OLR + U925 (media movel {}d)...', suaviz)
    _plot_hov_panel(
        hov_olr_p.values, hov_u925_p.values, lons_plot, times, ytick_interval,
        cmap=OLR_CMAP, levels=OLR_LEVELS, cbar_label='Anomalia OLR (W/m²)',
        titulo=(f'Hovmöller — Anomalia de OLR e Vento Zonal 925 hPa (5°S–5°N) — média móvel {suaviz}d\n'
                f'De {ini_fmt} a {fim_fmt}'),
        out_png=output_dir / png_olr_u925_name, entrada_dir=entrada_dir, logger=logger)

    # OLR anomalia + U850 anomalia — SEM media movel (dado diario cru)
    logger.info('Hovmoller OLR + U850 (sem media movel)...')
    hov_olr_raw = _hov_domain_mean(da_olr, hov_sst)
    _plot_hov_panel(
        hov_olr_raw.values, hov_u.values, lons_plot, times, ytick_interval,
        cmap=OLR_CMAP, levels=OLR_LEVELS, cbar_label='Anomalia OLR (W/m²)',
        titulo=(f'Hovmöller — Anomalia de OLR e Vento Zonal 850 hPa (5°S–5°N)\n'
                f'De {ini_fmt} a {fim_fmt}'),
        out_png=output_dir / png_olr_u850_name, entrada_dir=entrada_dir, logger=logger)

    # CHI200 anomalia + U850 anomalia — SEM media movel (dado diario cru)
    logger.info('Hovmoller CHI200 + U850 (sem media movel)...')
    hov_chi_raw = _hov_domain_mean(chi_da, hov_sst)
    _plot_hov_panel(
        hov_chi_raw.values, hov_u.values, lons_plot, times, ytick_interval,
        cmap=CHI_CMAP, levels=list(CHI_LEVELS), cbar_label='CHI200 anomalia (×10⁶ m²/s)',
        titulo=(f'Hovmöller — Anomalia de CHI200 (200 hPa) e Vento Zonal 850 hPa (5°S–5°N)\n'
                f'De {ini_fmt} a {fim_fmt}'),
        out_png=output_dir / png_chi_u850_name, entrada_dir=entrada_dir, logger=logger)

    # TSM anomalia + U925 anomalia — SEM media movel (dado diario cru)
    logger.info('Hovmoller TSM + U925 (sem media movel)...')
    _plot_hov_panel(
        hov_sst.values, hov_u925_raw.values, lons_plot, times, ytick_interval,
        cmap=cmap, levels=levels, cbar_label='°C',
        titulo=(f'Hovmöller — Anomalia de TSM e Vento Zonal 925 hPa (5°S–5°N)\n'
                f'De {ini_fmt} a {fim_fmt}'),
        out_png=output_dir / png_sst_u925_raw_name, entrada_dir=entrada_dir, logger=logger)

    # ---- Etapa 5: Anomalia vento 850 hPa (u + v) ----
    logger.info('Etapa 5: Processando anomalia vento 850 hPa (u+v)...')
    plot_wind850_anom()

    wind_file = dados_dir / WIND850_FILE_NAME
    if not wind_file.exists():
        raise FileNotFoundError(
            f'Arquivo esperado nao encontrado: {wind_file}. '
            'plot_wind850_anom() precisa salvar esse arquivo antes da plotagem.'
        )
    ds_wind = xr.open_dataset(str(wind_file))
    u_anom_da = ds_wind['u_anom_mean']
    v_anom_da = ds_wind['v_anom_mean']
    lat_wind = u_anom_da['lat'].values
    lon_wind = u_anom_da['lon'].values
    u_cyc_wind, lon_wind_cyc = _acp(u_anom_da.values, coord=lon_wind)
    v_cyc_wind, _ = _acp(v_anom_da.values, coord=lon_wind)
    ds_wind.close()

    # ---- Etapa 6: Mapa SSTA + Vento Anomalo 850 hPa — area ENSO ----
    logger.info('Etapa 6: Gerando mapa SSTA + Vento 850 hPa (area ENSO)...')

    da_sst_mean = da_sst.mean(dim='time', skipna=True)
    sst_lat = da_sst_mean['lat'].values
    sst_lon = da_sst_mean['lon'].values
    sst_cyc, sst_lon_cyc = _acp(da_sst_mean.values, coord=sst_lon)

    # Anomalia média de TSM por box ENSO (para os labels do mapa)
    enso_box_means = {
        name: _box_mean(
            da_sst_mean.values, sst_lon, sst_lat,
            b['lon_min'], b['lon_max'], b['lat_min'], b['lat_max'], b['wrap'],
        )
        for name, b in ENSO_BOXES.items()
    }

    info_plot = settings['areas_plotagem']
    area_cfg = info_plot['enso']
    central_lon_mapa = int(area_cfg['central_longitude_mapa'])
    central_lon_plot = int(area_cfg.get('central_longitude_plot', 0))

    fig2 = plt.figure(figsize=(16, 8))
    ax2 = fig2.add_subplot(
        1, 1, 1,
        projection=ccrs.PlateCarree(central_longitude=central_lon_mapa),
    )
    ax2.set_xlim([area_cfg['lon_esq'], area_cfg['lon_dir']])
    ax2.set_ylim([area_cfg['lat_inf'], area_cfg['lat_sup']])

    ax2.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2)
    ax2.add_feature(cfeature.LAND.with_scale('50m'), facecolor='whitesmoke')
    ax2.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1.0, zorder=100)
    ax2.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, zorder=100)
    ax2.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, zorder=100)
    ax2.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='white')

    gl2 = ax2.gridlines(draw_labels=True, linestyle='--', alpha=0.0)
    gl2.top_labels = False
    gl2.right_labels = False
    gl2.xlocator = mticker.MultipleLocator(20)
    gl2.xlabel_style = {'size': 14, 'color': 'black'}
    gl2.ylabel_style = {'size': 14, 'color': 'black'}

    sst_levels_map = list(settings.LST_SSTA_NEW_GREC)
    cmap_sst = LinearSegmentedColormap.from_list('sst_anom', settings.LST_ANOM_CORRETA)
    im2 = ax2.contourf(
        sst_lon_cyc, sst_lat, sst_cyc,
        levels=sst_levels_map,
        cmap=cmap_sst,
        extend='both',
        transform=ccrs.PlateCarree(central_longitude=central_lon_plot),
    )

    # Boxes Niño — zorder=300 (acima do quiver=200) para bordas não serem cortadas pelos vetores
    for i, box in enumerate(area_cfg.get('lst_boxes', [])):
        edgecolor = _BOX_COLOR_OVERRIDE.get(i, box['edgecolor'])
        rect = patches.Rectangle(
            (box['x_anc'], box['y_anc']),
            box['x_larg'], box['y_larg'],
            linewidth=box['linewidth'],
            edgecolor=edgecolor,
            facecolor='none',
            zorder=300,
        )
        ax2.add_patch(rect)

    # Quiver vento 850 hPa anomalo
    step = QUIVER_ENSO_STEP
    lon_q = lon_wind_cyc[::step]
    lat_q = lat_wind[::step]
    u_q = u_cyc_wind[::step, ::step].copy()
    v_q = v_cyc_wind[::step, ::step].copy()
    mag = np.sqrt(u_q**2 + v_q**2)
    u_q = np.where(mag < QUIVER_ENSO_MIN_MAG, np.nan, u_q)
    v_q = np.where(mag < QUIVER_ENSO_MIN_MAG, np.nan, v_q)

    ax2.quiver(
        lon_q, lat_q, u_q, v_q,
        transform=ccrs.PlateCarree(central_longitude=central_lon_plot),
        scale=QUIVER_ENSO_SCALE,
        scale_units='width',
        width=QUIVER_ENSO_WIDTH,
        color='black',
        zorder=200,
    )

    # Labels ENSO — nome do box + anomalia média de TSM (ex: "Niño 3.4 = 1.2°C")
    # (txt, índice em lst_boxes, y em coords de dados, cor do texto)
    boxes_cfg = area_cfg.get('lst_boxes', [])
    for txt, box_idx, y, cor in [
        ('Nino 1+2', 0, -13.64, 'limegreen'),
        ('Nino 3',   1,  8.45,  'blue'),
        ('Nino 3.4', 3, -9.45,  'black'),
        ('Nino 4',   2,  8.45,  _NINO4_COLOR),
    ]:
        if box_idx >= len(boxes_cfg):
            continue
        box = boxes_cfg[box_idx]
        cx = box['x_anc'] + box['x_larg'] / 2
        val = enso_box_means.get(txt)
        label = f'{txt} = {val:.2f}°C' if val is not None and np.isfinite(val) else txt
        t = ax2.text(cx, y, label, fontsize=14, color=cor, weight='bold', ha='center', zorder=400)
        fg = 'black' if cor in {'limegreen', 'magenta'} else 'white'
        t.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=fg),
            path_effects.Normal(),
        ])

    # Colorbar
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)
    cbar2 = plt.colorbar(
        im2, cax=cax2, orientation='horizontal', extend='both',
        location='bottom', ticks=sst_levels_map[::2],
    )
    cbar2.set_label('°C', fontsize=14)
    cbar2.ax.tick_params(labelsize=14)

    # Titulo
    ax2.set_title(
        f'Anomalia TSM + Vento Anômalo 850 hPa\n'
        f'De {ini_fmt} a {fim_fmt}',
        fontsize=14, loc='left',
    )

    if logo_path is not None and logo_path.exists():
        _add_logo_to_map(ax=ax2, logo_path=logo_path)

    enso_fig_path = output_dir / enso_map_name
    logger.info(f'Salvando figura SSTA+Vento850: {enso_fig_path}')
    plt.savefig(str(enso_fig_path), dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # ---- Etapa 7: Mapas ENSO NOVOS — vento ANÔMALO 925 hPa (u+v): SST+925 e OLR+925 ----
    logger.info('Etapa 7: vento 925 hPa anomalo (u+v) p/ os mapas ENSO...')
    plot_wind925_anom()
    ds_w925 = xr.open_dataset(str(dados_dir / WIND925_FILE_NAME))
    u925a, v925a = ds_w925['u_anom_mean'], ds_w925['v_anom_mean']
    lat_w925 = u925a['lat'].values
    u925_cyc, lon_w925_cyc = _acp(u925a.values, coord=u925a['lon'].values)
    v925_cyc, _ = _acp(v925a.values, coord=u925a['lon'].values)
    ds_w925.close()

    logger.info('Mapa ENSO: SST + Vento 925 hPa...')
    _plot_enso_map(
        sst_cyc, sst_lon_cyc, sst_lat, levels=sst_levels_map, cmap=cmap_sst, cbar_label='°C',
        box_means=enso_box_means, box_unit='°C', box_fmt='.2f',
        u_cyc=u925_cyc, v_cyc=v925_cyc, lon_wind_cyc=lon_w925_cyc, lat_wind=lat_w925,
        titulo=f'Anomalia TSM + Vento Anômalo 925 hPa\nDe {ini_fmt} a {fim_fmt}',
        out_png=output_dir / enso_map_sst925_name, area_cfg=area_cfg, entrada_dir=entrada_dir,
        logo_path=logo_path, logger=logger)

    logger.info('Mapa ENSO: OLR + Vento 925 hPa...')
    olr_mean = da_olr.mean(dim='time', skipna=True)
    olr_cyc, olr_lon_cyc = _acp(olr_mean.values, coord=olr_mean['lon'].values)
    olr_box_means = {
        name: _box_mean(olr_mean.values, olr_mean['lon'].values, olr_mean['lat'].values,
                        b['lon_min'], b['lon_max'], b['lat_min'], b['lat_max'], b['wrap'])
        for name, b in ENSO_BOXES.items()
    }
    _plot_enso_map(
        olr_cyc, olr_lon_cyc, olr_mean['lat'].values, levels=OLR_LEVELS, cmap=OLR_CMAP,
        cbar_label='Anomalia OLR (W/m²)', box_means=olr_box_means, box_unit=' W/m²', box_fmt='.0f',
        u_cyc=u925_cyc, v_cyc=v925_cyc, lon_wind_cyc=lon_w925_cyc, lat_wind=lat_w925,
        titulo=f'Anomalia OLR + Vento Anômalo 925 hPa\nDe {ini_fmt} a {fim_fmt}',
        out_png=output_dir / enso_map_olr925_name, area_cfg=area_cfg, entrada_dir=entrada_dir,
        logo_path=logo_path, logger=logger, show_box_value=False)

    logger.info('Mapa ENSO: OLR + Vento 850 hPa...')
    _plot_enso_map(
        olr_cyc, olr_lon_cyc, olr_mean['lat'].values, levels=OLR_LEVELS, cmap=OLR_CMAP,
        cbar_label='Anomalia OLR (W/m²)', box_means=olr_box_means, box_unit=' W/m²', box_fmt='.0f',
        u_cyc=u_cyc_wind, v_cyc=v_cyc_wind, lon_wind_cyc=lon_wind_cyc, lat_wind=lat_wind,
        titulo=f'Anomalia OLR + Vento Anômalo 850 hPa\nDe {ini_fmt} a {fim_fmt}',
        out_png=output_dir / enso_map_olr850_name, area_cfg=area_cfg, entrada_dir=entrada_dir,
        logo_path=logo_path, logger=logger, show_box_value=False)

    execution_time = time.time() - start_time
    save_cache_metadata(SCRIPT_ID, cache_params, output_files, execution_time)
    logger.info('=' * 80)
    logger.info(f'Script {SCRIPT_ID.upper()} concluido com sucesso!')
    logger.info(f'Tempo de execucao: {execution_time:.1f}s ({execution_time / 60:.1f} min)')
    logger.info(f'Mapa gerado em: {output_dir}')
    logger.info('=' * 80)


if __name__ == '__main__':
    main()
