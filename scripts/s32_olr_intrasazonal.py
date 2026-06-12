# -*- coding: utf-8 -*-
"""
s32 - OLR intrasazonal (CPC Blended OLR 2.5°, banda MJO/intrasazonal).

Baixa a anomalia diaria de OLR do PSL/NOAA (CPC Blended OLR 2.5°, ja em relacao
a LTM 1991-2020) e isola o sinal intrasazonal em dois passos:

    1. Remove variabilidade interanual (> JANELA_MEDIA_MOVEL dias):
           olr_intra(t) = olr_anom(t) - media_movel(olr_anom, N=120d)

    2. Filtro passa-banda de Lanczos (20-90 dias) sobre olr_intra:
           olr_mjo(t) = Lanczos_bandpass(olr_intra, Tmin=20d, Tmax=90d, n=60)
       Remove o ruido sinotico (< 20 dias) e sinal lento remanescente (> 90 dias).
       Os pesos do filtro sao calculados como:
           w_k = [sin(2pi*f2*k) - sin(2pi*f1*k)] / (pi*k) * sinc(k/n)
       com f1=1/Tmax, f2=1/Tmin e janela de Lanczos sigma_k = sinc(k/n).

Produtos (todos da mesma serie diaria de OLR intrasazonal):
    1. Mapas de pentada (ultimas N_PENTADAS pentadas de 5 dias)
    2. Hovmoller (lon x tempo, media numa faixa equatorial)
    3. Mapa do periodo (media de olr_intra em [DATA_INICIAL, DATA_FINAL])

Dados:
    - PSL/NOAA: olr.cbo-2.5deg.day.anom.nc (CPC Blended OLR 2.5 graus)

Saida:
    - PNGs em {settings.DIR_OUTPUT}/s32_OLR_INTRASAZONAL/

Criado em: 2026-06-11
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')

import time
from datetime import datetime, timedelta
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.util import add_cyclic_point
from matplotlib.colors import BoundaryNorm
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from PIL import Image

from scipy.ndimage import convolve1d

from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.common.dataset_utils import arquivo_cobre_periodo, load_dataset, validar_cobertura_temporal
from app.common.download_helper import DownloadEngine, download_with_progress
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.src.uteis.downloaders_gdas_uv850 import ensure_gdas_uv850_for_period
from app.src.uteis.downloaders_wind850 import ensure_era5_uv850_for_period

SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's32'
SCRIPT_NAME = Path(__file__).stem
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME

OLR_URL = 'https://downloads.psl.noaa.gov/Datasets/cpc_blended_olr-2.5deg/olr.day.anom.nc'
OLR_FILE_NAME = 'olr.day.anom.nc'

ERA5_LATENCY_DAYS = 5
DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)

# Niveis de OLR intrasazonal (W/m²): amplitude tipica MJO ~15-30 W/m²
LEVELS = np.arange(-40, 44, 4)
# Niveis de u850 intrasazonal (m/s) para isolinhas no Hovmoller
LEVELS_U850 = [-6, -4, -2, 2, 4, 6]
# BrBG_r: negativo (ativo/mais chuva) → castanho, positivo (suprimido) → azul/verde
# Consistente com s05_olr_anom
CMAP_NAME = 'BrBG_r'


def _cfg(name: str, default):
    return settings.get(name, default)


# ---------------------------------------------------------------------------
# Série diária de u/v 850 (ERA5 + GDAS → grade 2.5° do OLR)
# ---------------------------------------------------------------------------
def _daily_series_uv850(
    files, start: datetime, end: datetime,
    target_lat: np.ndarray, target_lon: np.ndarray, logger,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Lê arquivos ERA5/GDAS 850 hPa, filtra horas sinóticas, faz média diária e
    interpola para a grade alvo (2.5° do OLR). Retorna (u_da, v_da).
    """
    from app.src.uteis.plot_chi200 import (
        _drop_or_collapse_expver, _ensure_time_coord,
        _find_uv_vars, _normalize_latlon_names, _sort_and_dedup_time,
    )
    t_ini = np.datetime64(start.date())
    t_fim = np.datetime64(end.date())
    req = set(DEFAULT_SYNOPTIC_HOURS)
    tgt_lat = xr.DataArray(target_lat, dims=['lat'])
    tgt_lon = xr.DataArray(target_lon, dims=['lon'])

    us, vs = [], []
    for fp in files:
        ds = xr.open_dataset(fp, engine='netcdf4')
        try:
            ds = _sort_and_dedup_time(_normalize_latlon_names(
                _drop_or_collapse_expver(_ensure_time_coord(ds))))
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
        finally:
            ds.close()

    if not us:
        raise RuntimeError('Nenhum dado u/v 850 válido no período de download.')

    u_da = xr.concat(us, dim='time', coords='minimal', compat='override').sortby('time')
    v_da = xr.concat(vs, dim='time', coords='minimal', compat='override').sortby('time')
    _, uniq = np.unique(u_da['time'].values, return_index=True)
    return u_da.isel(time=uniq), v_da.isel(time=uniq)


def _reindex_daily(da: xr.DataArray) -> xr.DataArray:
    t0 = pd.Timestamp(da['time'].values[0]).normalize()
    t1 = pd.Timestamp(da['time'].values[-1]).normalize()
    full = pd.date_range(t0, t1, freq='1D')
    return da.reindex(time=full).interpolate_na(dim='time', method='linear', fill_value='extrapolate')


def _wind850_filtered(
    u_da: xr.DataArray, v_da: xr.DataArray,
    janela: int, lanczos_n: int, period_min: float, period_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove media movel e aplica Lanczos bandpass ao vento 850 hPa bruto.
    Retorna (u_com, v_com, u_sem, v_sem) — com e sem filtro Lanczos.
    """
    from app.src.uteis.chi200_intrasazonal import lanczos_bandpass

    u = u_da.values  # (T, lat, lon)
    v = v_da.values

    def _remove_mm(arr):
        T = arr.shape[0]
        out = np.empty((T - janela,) + arr.shape[1:], dtype=np.float64)
        for k, t in enumerate(range(janela, T)):
            out[k] = arr[t] - arr[t - janela:t].mean(axis=0)
        return out

    u_sem = _remove_mm(u)
    v_sem = _remove_mm(v)
    u_com = np.nan_to_num(lanczos_bandpass(u_sem, period_min, period_max, lanczos_n))
    v_com = np.nan_to_num(lanczos_bandpass(v_sem, period_min, period_max, lanczos_n))
    return u_com, v_com, u_sem, v_sem


def _add_logo_to_map(ax, logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=30):
    logo = Image.open(logo_path).convert('RGBA')
    bbox = logo.getbbox()
    if bbox is not None:
        logo = logo.crop(bbox)
    img = np.array(logo)
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(
        imagebox, (0, 0),
        xycoords=ax.transAxes, xybox=(xoffset, yoffset),
        boxcoords='offset points', box_alignment=(0, 0),
        frameon=False, pad=0, zorder=zorder, clip_on=False,
    )
    ax.add_artist(ab)


def _normalize_ds(ds: xr.Dataset) -> xr.Dataset:
    """Normaliza coordenadas: lat ascendente, lon em 0-360."""
    rename = {}
    if 'latitude' in ds.coords and 'lat' not in ds.coords:
        rename['latitude'] = 'lat'
    if 'longitude' in ds.coords and 'lon' not in ds.coords:
        rename['longitude'] = 'lon'
    if rename:
        ds = ds.rename(rename)
    if float(ds['lat'].values[0]) > float(ds['lat'].values[-1]):
        ds = ds.isel(lat=slice(None, None, -1))
    if float(ds['lon'].values.min()) < 0:
        ds = ds.assign_coords(lon=(ds['lon'] % 360)).sortby('lon')
    return ds


def _lanczos_weights(period_min: float, period_max: float, n: int) -> np.ndarray:
    """
    Pesos do filtro passa-banda de Lanczos para a banda [period_min, period_max] dias.

    w_k = ideal_k * sigma_k
      ideal_k = [sin(2pi*f2*k) - sin(2pi*f1*k)] / (pi*k)   (k != 0)
      ideal_0 = 2*(f2 - f1)
      sigma_k = sinc(k/n)   (janela de Lanczos)

    onde f1 = 1/period_max e f2 = 1/period_min (ciclos/dia).
    Retorna array de comprimento 2n+1.
    """
    f1 = 1.0 / period_max
    f2 = 1.0 / period_min
    k = np.arange(-n, n + 1, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        ideal = np.where(
            k == 0,
            2.0 * (f2 - f1),
            (np.sin(2 * np.pi * f2 * k) - np.sin(2 * np.pi * f1 * k)) / (np.pi * k),
        )
    return ideal * np.sinc(k / n)  # np.sinc(x) = sin(pi*x)/(pi*x)


def _olr_intrasazonal(
    olr_anom: xr.DataArray,
    janela: int,
    lanczos_n: int,
    period_min: float,
    period_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Isola a banda intrasazonal em dois passos:
      1. Remove interanual: olr_sem = olr_anom - media_movel(janela)
      2. Aplica filtro passa-banda de Lanczos [period_min, period_max]: olr_com

    Retorna (olr_com, olr_sem, dates):
      olr_com  — bandpass Lanczos 20-90d (banda MJO)
      olr_sem  — apenas running mean removido (< janela dias, sem remocao de sinotico)
    """
    # Preenche dias completamente ausentes no arquivo PSL via interpolacao linear
    # (ex: 2026-02-22 inteiramente NaN — sem isso propaga NaN por `janela` dias)
    olr_anom = olr_anom.interpolate_na(dim='time', method='linear')

    # Passo 1: remove variabilidade interanual (> janela dias)
    olr_roll = olr_anom.rolling(time=janela, min_periods=janela).mean()
    olr_intra_da = (olr_anom - olr_roll).isel(time=slice(janela, None))
    olr_sem = olr_intra_da.values  # sem filtro Lanczos

    # Passo 2: filtro passa-banda de Lanczos (remove < period_min e > period_max)
    weights = _lanczos_weights(period_min, period_max, lanczos_n)
    # convolve1d ao longo do eixo temporal; mode='nearest' evita NaN nas bordas
    olr_com = convolve1d(olr_sem, weights, axis=0, mode='nearest')

    dates = np.array([
        np.datetime64(pd.Timestamp(t).date())
        for t in olr_intra_da['time'].values
    ])
    return olr_com, olr_sem, dates


def _agrupa_pentadas(
    olr_intra: np.ndarray,
    dates: np.ndarray,
    n_pentadas: int,
) -> list[tuple]:
    """Retorna [(d_ini, d_fim, campo_medio), ...] para as ultimas n_pentadas pentadas."""
    out = []
    last_day = dates[-1]
    for i in range(n_pentadas - 1, -1, -1):
        d_fim = last_day - np.timedelta64(i * 5, 'D')
        d_ini = d_fim - np.timedelta64(4, 'D')
        mask = (dates >= d_ini) & (dates <= d_fim)
        if not mask.any():
            continue
        out.append((d_ini, d_fim, olr_intra[mask].mean(axis=0)))
    return out


def _media_faixa_latitude(
    data: np.ndarray, lat: np.ndarray, lat_min: float, lat_max: float,
) -> np.ndarray:
    """Media latitudinal na faixa [lat_min, lat_max]. Retorna (time, lon)."""
    mask_lat = (lat >= lat_min) & (lat <= lat_max)
    return data[:, mask_lat, :].mean(axis=1)


def _cmap_norm():
    cmap = plt.get_cmap(CMAP_NAME, len(LEVELS) + 1)
    norm = BoundaryNorm(LEVELS, cmap.N, extend='both')
    return cmap, norm


def _plot_mapa(olr2d, lat, lon, titulo, out_png, input_dir, cbar_label='OLR intrasazonal (W/m²)'):
    cmap, norm = _cmap_norm()
    arr, lonc = add_cyclic_point(olr2d, coord=lon)
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_xlim([-180, 180])
    ax.set_ylim([-60, 60])
    im = ax.contourf(
        lonc, lat, arr, levels=LEVELS, cmap=cmap, norm=norm,
        extend='both', transform=ccrs.PlateCarree(central_longitude=0),
    )
    ax.add_feature(
        NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '50m',
                            facecolor='none', edgecolor='black'),
        linewidth=0.6, zorder=100,
    )
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.0, zorder=100)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.0, zorder=100)
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.3)
    gl.top_labels = gl.right_labels = False
    ax.set_title(titulo, fontsize=15, loc='left')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2.5%', pad=0.08, axes_class=plt.Axes)
    cbar = plt.colorbar(im, cax=cax, ticks=LEVELS[::2], extend='both')
    cbar.set_label(cbar_label, size=12)
    logo_path = (
        None if settings.get('SEM_LOGO', False)
        else input_dir / ('logo_grec.png' if settings.get('LOGO_GREC', False) else 'novo_logo.png')
    )
    if logo_path is not None and logo_path.exists():
        _add_logo_to_map(ax=ax, logo_path=logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=500)
    fig.savefig(str(out_png), dpi=fig.dpi, bbox_inches='tight')
    plt.close('all')


def _lon_we_formatter(x, pos):
    x = x % 360
    if x == 0 or x == 360:
        return '0°'
    if x == 180:
        return '180°'
    return f'{int(x)}°E' if x < 180 else f'{int(360 - x)}°W'


def _plot_hovmoller(hov, lon, dates, titulo, out_png, input_dir, cbar_label='OLR intrasazonal (W/m²)'):
    cmap, norm = _cmap_norm()
    hov_s, lonc = add_cyclic_point(hov, coord=lon)
    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.contourf(lonc, dates, hov_s, levels=LEVELS, cmap=cmap, norm=norm, extend='both')
    ax.invert_yaxis()  # tempo cresce para baixo (propagacao leste fica inclinada)
    ax.xaxis.set_major_locator(plt.MultipleLocator(60))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_lon_we_formatter))
    ax.set_xlabel('Longitude', fontsize=13)
    ax.set_ylabel('Data', fontsize=13)
    ax.set_title(titulo, fontsize=14, loc='left')
    cbar = fig.colorbar(im, ax=ax, ticks=LEVELS[::2], extend='both', pad=0.02)
    cbar.set_label(cbar_label, size=12)
    logo_path = (
        None if settings.get('SEM_LOGO', False)
        else input_dir / ('logo_grec.png' if settings.get('LOGO_GREC', False) else 'novo_logo.png')
    )
    if logo_path is not None and logo_path.exists():
        _add_logo_to_map(ax=ax, logo_path=logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=500)
    fig.savefig(str(out_png), dpi=fig.dpi, bbox_inches='tight')
    plt.close('all')


def _plot_mapa_olr_wind(olr2d, u2d, v2d, lat, lon, titulo, out_png, input_dir,
                        cbar_label='OLR intrasazonal (W/m²)'):
    """Mapa de OLR shaded + vetores de vento 850 hPa."""
    cmap, norm = _cmap_norm()
    arr, lonc = add_cyclic_point(olr2d, coord=lon)
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_xlim([-180, 180])
    ax.set_ylim([-60, 60])
    im = ax.contourf(
        lonc, lat, arr, levels=LEVELS, cmap=cmap, norm=norm,
        extend='both', transform=ccrs.PlateCarree(central_longitude=0),
    )
    # Quiver do vento 850 hPa (subamostrado a cada 5° ≈ passo 2 na grade 2.5°)
    s = 2
    lon_q, lat_q = lon[::s], lat[::s]
    u_q, v_q = u2d[::s, ::s].copy(), v2d[::s, ::s].copy()
    mag = np.hypot(u_q, v_q)
    # Clip ao P85 global: os 15% mais intensos (tipicamente extratropicais) são escalados
    valid_mag = mag[mag > 0.3]
    p_clip = float(np.nanpercentile(valid_mag, 50)) if valid_mag.size > 0 else np.inf
    too_big = mag > p_clip
    if too_big.any():
        ratio = np.where(too_big, p_clip / np.maximum(mag, 1e-10), 1.0)
        u_q *= ratio
        v_q *= ratio
        mag = np.hypot(u_q, v_q)
    u_q = np.ma.array(u_q, mask=mag < 0.3)
    v_q = np.ma.array(v_q, mask=mag < 0.3)
    ax.quiver(
        lon_q, lat_q, u_q, v_q,
        transform=ccrs.PlateCarree(),
        color='black', pivot='mid',
        scale=120.0, width=0.0012,
        headwidth=3.2, headlength=4.2, headaxislength=3.8, zorder=5,
    )
    ax.add_feature(
        NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '50m',
                            facecolor='none', edgecolor='black'),
        linewidth=0.6, zorder=100,
    )
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.0, zorder=100)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.0, zorder=100)
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.3)
    gl.top_labels = gl.right_labels = False
    ax.set_title(titulo, fontsize=15, loc='left')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2.5%', pad=0.08, axes_class=plt.Axes)
    cbar = plt.colorbar(im, cax=cax, ticks=LEVELS[::2], extend='both')
    cbar.set_label(cbar_label, size=12)
    logo_path = (
        None if settings.get('SEM_LOGO', False)
        else input_dir / ('logo_grec.png' if settings.get('LOGO_GREC', False) else 'novo_logo.png')
    )
    if logo_path is not None and logo_path.exists():
        _add_logo_to_map(ax=ax, logo_path=logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=500)
    fig.savefig(str(out_png), dpi=fig.dpi, bbox_inches='tight')
    plt.close('all')


def _plot_hovmoller_olr_wind(olr_hov, u_hov, lon, dates, titulo, out_png, input_dir):
    """Hovmöller OLR shaded + isolinhas de u850 (vermelho=positivo, azul=negativo)."""
    cmap, norm = _cmap_norm()
    olr_s, lonc = add_cyclic_point(olr_hov, coord=lon)
    u_s, _ = add_cyclic_point(u_hov, coord=lon)
    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.contourf(lonc, dates, olr_s, levels=LEVELS, cmap=cmap, norm=norm, extend='both')
    # Isolinhas u850: positivo=vermelho, negativo=azul
    pos = [lv for lv in LEVELS_U850 if lv > 0]
    neg = [lv for lv in LEVELS_U850 if lv < 0]
    if neg:
        ax.contour(lonc, dates, u_s, levels=neg, colors='blue', linewidths=1.2, linestyles='solid')
    if pos:
        ax.contour(lonc, dates, u_s, levels=pos, colors='red',  linewidths=1.2, linestyles='solid')
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(plt.MultipleLocator(60))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_lon_we_formatter))
    ax.set_xlabel('Longitude', fontsize=13)
    ax.set_ylabel('Data', fontsize=13)
    ax.set_title(titulo, fontsize=14, loc='left')
    cbar = fig.colorbar(im, ax=ax, ticks=LEVELS[::2], extend='both', pad=0.02)
    cbar.set_label('OLR intrasazonal (W/m²)', size=12)
    logo_path = (
        None if settings.get('SEM_LOGO', False)
        else input_dir / ('logo_grec.png' if settings.get('LOGO_GREC', False) else 'novo_logo.png')
    )
    if logo_path is not None and logo_path.exists():
        _add_logo_to_map(ax=ax, logo_path=logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=500)
    fig.savefig(str(out_png), dpi=fig.dpi, bbox_inches='tight')
    plt.close('all')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger = get_logger(SCRIPT_ID)
    logger.info('=' * 80)
    logger.info(f'SCRIPT {SCRIPT_ID.upper()}: {SCRIPT_DESC}')
    logger.info('=' * 80)

    janela = int(_cfg('JANELA_MEDIA_MOVEL', 120))
    n_pentadas = int(_cfg('N_PENTADAS', 6))
    hov_dias = int(_cfg('HOVMOLLER_DIAS', 120))
    faixa = list(_cfg('FAIXA_HOVMOLLER', [-5, 5]))
    lanczos_n = int(_cfg('LANCZOS_N', 60))
    period_min = float(_cfg('LANCZOS_PERIOD_MIN', 20.0))
    period_max = float(_cfg('LANCZOS_PERIOD_MAX', 90.0))

    output_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_OLR_INTRASAZONAL'
    input_dir = Path(settings.DIR_INPUT)
    dados_dir = Path(settings.DIR_DADOS)

    dt_ini = datetime.strptime(settings.DATA_INICIAL, '%Y-%m-%d')
    dt_fim = datetime.strptime(settings.DATA_FINAL, '%Y-%m-%d')

    hov_com_png = output_dir / 'olr_hovmoller_com_filtro.png'
    hov_sem_png = output_dir / 'olr_hovmoller_sem_filtro.png'
    hov_wind_png = output_dir / 'olr_u850_hovmoller_com_filtro.png'
    periodo_com_png = output_dir / 'olr_periodo_com_filtro.png'
    periodo_sem_png = output_dir / 'olr_periodo_sem_filtro.png'
    periodo_wind_png = output_dir / 'olr_u850_periodo_com_filtro.png'
    output_files = [
        str(hov_com_png), str(hov_sem_png), str(hov_wind_png),
        str(periodo_com_png), str(periodo_sem_png), str(periodo_wind_png),
    ]

    cache_params = {
        'DATA_INICIAL': settings.DATA_INICIAL,
        'DATA_FINAL': settings.DATA_FINAL,
        'janela': janela,
        'n_pentadas': n_pentadas,
        'hov_dias': hov_dias,
        'faixa': faixa,
        'lanczos_n': lanczos_n,
        'period_min': period_min,
        'period_max': period_max,
        'metodo': 'CPC running-mean + Lanczos bandpass (20-90d) + u850',
        'script_version': '2.0',
    }
    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('CACHE VALIDO! Pulando execucao.')
        return

    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    dados_dir.mkdir(parents=True, exist_ok=True)

    # Janela de download: precisa de `janela` dias antes do inicio da serie de interesse
    inicio_interesse = min(
        dt_ini,
        dt_fim - timedelta(days=max(hov_dias, n_pentadas * 5) - 1),
    )
    # janela: running mean | lanczos_n: metade da janela do filtro passa-banda
    start_dl = inicio_interesse - timedelta(days=janela + lanczos_n + 2)
    logger.info(f'Periodo de interesse: {dt_ini.date()} a {dt_fim.date()}')
    logger.info(f'Janela de download (running mean + Lanczos): {start_dl.date()} a {dt_fim.date()}')

    # ---- Download OLR (PSL/NOAA) ----
    olr_path = dados_dir / OLR_FILE_NAME
    if arquivo_cobre_periodo(olr_path, np.datetime64(start_dl.date()), np.datetime64(dt_fim.date())):
        logger.info('Arquivo OLR local ja cobre o periodo solicitado — pulando download')
    else:
        if olr_path.exists():
            logger.info('Arquivo OLR local nao cobre o periodo solicitado — re-baixando')
        logger.info('Etapa 1: Download OLR (CPC Blended 2.5°, PSL/NOAA)...')
        download_with_progress(
            url=OLR_URL,
            output_path=str(olr_path),
            description=OLR_FILE_NAME,
            max_retries=5,
            force=True,
            engine=DownloadEngine.ARIA2,
            timeout=300,
        )

    # ---- Carrega e valida cobertura temporal ----
    ds = load_dataset(str(olr_path))
    ds = _normalize_ds(ds)
    validar_cobertura_temporal(
        ds,
        np.datetime64(start_dl.date()),
        np.datetime64(dt_fim.date()),
        nome='arquivo OLR CPC Blended',
    )

    # ---- Seleciona janela e carrega na RAM ----
    logger.info('Etapa 2: Carregando serie temporal OLR...')
    subset = ds.sel(time=slice(np.datetime64(start_dl.date()), np.datetime64(dt_fim.date())))
    olr_anom_da = subset['olr'].load()  # (time, lat, lon)
    lat = olr_anom_da['lat'].values.astype(float)
    lon = olr_anom_da['lon'].values.astype(float)
    logger.info(
        'Serie carregada: {} dias, grade {}x{} (2.5°)',
        olr_anom_da.sizes['time'], len(lat), len(lon),
    )

    # ---- Filtro intrasazonal ----
    logger.info(
        'Etapa 3: Filtro intrasazonal (running mean {}d + Lanczos {}-{}d, n={})...',
        janela, int(period_min), int(period_max), lanczos_n,
    )
    olr_com, olr_sem, dates_intra = _olr_intrasazonal(
        olr_anom_da, janela, lanczos_n, period_min, period_max,
    )
    logger.info(
        'Serie filtrada: {} dias ({} a {})',
        len(dates_intra), dates_intra[0], dates_intra[-1],
    )

    # versoes: (sufixo, dados, rotulo_titulo, cbar_label, hov_png, periodo_png)
    versoes = [
        ('com_filtro', olr_com, 'OLR intrasazonal', 'OLR intrasazonal (W/m²)', hov_com_png, periodo_com_png),
        ('sem_filtro', olr_sem, 'OLR',              'OLR (W/m²)',               hov_sem_png, periodo_sem_png),
    ]

    # ---- Produto 1: mapas de pentada (com e sem filtro) ----
    logger.info('Etapa 4: Mapas de pentada (com e sem filtro Lanczos)...')
    for antigo in output_dir.glob('olr_*pentada*.png'):
        antigo.unlink()
    for sufixo, dados, rotulo, cbar_label, _, _ in versoes:
        pentadas = _agrupa_pentadas(dados, dates_intra, n_pentadas)
        for d_ini_p, d_fim_p, campo in pentadas:
            nome = f'olr_pentada_{sufixo}_{d_ini_p}_a_{d_fim_p}.png'
            _plot_mapa(
                campo, lat, lon,
                f'{rotulo} — pentada {d_ini_p} a {d_fim_p}',
                output_dir / nome, input_dir, cbar_label=cbar_label,
            )

    # ---- Produto 2: Hovmoller (com e sem filtro) ----
    logger.info('Etapa 5: Hovmoller (com e sem filtro Lanczos)...')
    m_hov = dates_intra >= (np.datetime64(dt_fim.date()) - np.timedelta64(hov_dias - 1, 'D'))
    for sufixo, dados, rotulo, cbar_label, hov_png_v, _ in versoes:
        hov = _media_faixa_latitude(dados[m_hov], lat, faixa[0], faixa[1])
        _plot_hovmoller(
            hov, lon, dates_intra[m_hov],
            f'{rotulo} — Hovmöller ({faixa[0]}° a {faixa[1]}°)',
            hov_png_v, input_dir, cbar_label=cbar_label,
        )

    # ---- Produto 3: mapa do periodo (com e sem filtro) ----
    logger.info('Etapa 6: Mapa do periodo (com e sem filtro Lanczos)...')
    m_per = (dates_intra >= np.datetime64(dt_ini.date())) & (dates_intra <= np.datetime64(dt_fim.date()))
    if m_per.any():
        for sufixo, dados, rotulo, cbar_label, _, periodo_png_v in versoes:
            campo_per = dados[m_per].mean(axis=0)
            _plot_mapa(
                campo_per, lat, lon,
                f'{rotulo} — media {dt_ini.date()} a {dt_fim.date()}',
                periodo_png_v, input_dir, cbar_label=cbar_label,
            )

    # ---- Download e processamento do vento 850 hPa (ERA5 + GDAS) ----
    logger.info('Etapa 7: Download u/v 850 hPa (ERA5 + GDAS)...')
    cutoff = (datetime.now() - timedelta(days=ERA5_LATENCY_DAYS)).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    wind_files = []
    if start_dl < cutoff:
        logger.info('Etapa 7a: ERA5 u/v 850 hPa...')
        wind_files += list(ensure_era5_uv850_for_period(
            start=start_dl, end=min(dt_fim, cutoff - timedelta(days=1)),
            hours_utc=list(DEFAULT_SYNOPTIC_HOURS), force_redownload=False,
        ))
    if dt_fim >= cutoff:
        logger.info('Etapa 7b: GDAS u/v 850 hPa...')
        wind_files += list(ensure_gdas_uv850_for_period(
            start=max(start_dl, cutoff), end=dt_fim, force_redownload=False,
        ))

    logger.info('Etapa 8: Serie diaria u/v 850 hPa (grade OLR 2.5°)...')
    u_da, v_da = _daily_series_uv850(wind_files, start_dl, dt_fim, lat, lon, logger)
    u_da, v_da = _reindex_daily(u_da), _reindex_daily(v_da)
    logger.info(
        'Serie vento: {} dias ({} a {})',
        u_da.sizes['time'],
        pd.Timestamp(u_da['time'].values[0]).date(),
        pd.Timestamp(u_da['time'].values[-1]).date(),
    )

    logger.info('Etapa 9: Filtro intrasazonal u/v 850 hPa...')
    u_com, v_com, u_sem, v_sem = _wind850_filtered(
        u_da, v_da, janela, lanczos_n, period_min, period_max,
    )
    # Datas do vento filtrado (alinhadas com OLR apos running mean)
    dates_wind = np.array([
        np.datetime64(pd.Timestamp(u_da['time'].values[i]).date())
        for i in range(janela, u_da.sizes['time'])
    ])

    # ---- Produto: Hovmoller OLR + u850 ----
    logger.info('Etapa 10: Hovmoller OLR + u850 (com filtro)...')
    # Alinha datas entre OLR e vento (usa intersecao)
    mask_hov_olr = dates_intra >= (np.datetime64(dt_fim.date()) - np.timedelta64(hov_dias - 1, 'D'))
    mask_hov_wnd = dates_wind >= (np.datetime64(dt_fim.date()) - np.timedelta64(hov_dias - 1, 'D'))
    # Garante mesmas datas (pode haver pequena diferenca por dias faltantes)
    d_hov = np.intersect1d(dates_intra[mask_hov_olr], dates_wind[mask_hov_wnd])
    idx_o = np.isin(dates_intra, d_hov)
    idx_w = np.isin(dates_wind, d_hov)
    olr_hov = _media_faixa_latitude(olr_com[idx_o], lat, faixa[0], faixa[1])
    u_hov = _media_faixa_latitude(u_com[idx_w], lat, faixa[0], faixa[1])
    _plot_hovmoller_olr_wind(
        olr_hov, u_hov, lon, d_hov,
        f'OLR intrasazonal + u850 — Hovmöller ({faixa[0]}° a {faixa[1]}°)',
        hov_wind_png, input_dir,
    )

    # ---- Produto: mapas de pentada OLR + vento 850 ----
    logger.info('Etapa 11: Mapas de pentada OLR + u850 (com filtro)...')
    for antigo in output_dir.glob('olr_u850_pentada_*.png'):
        antigo.unlink()
    for d_ini_p, d_fim_p, olr_campo in _agrupa_pentadas(olr_com, dates_intra, n_pentadas):
        # Media do vento na mesma pentada
        mw = (dates_wind >= d_ini_p) & (dates_wind <= d_fim_p)
        if not mw.any():
            continue
        u_campo = u_com[mw].mean(axis=0)
        v_campo = v_com[mw].mean(axis=0)
        nome = f'olr_u850_pentada_com_filtro_{d_ini_p}_a_{d_fim_p}.png'
        _plot_mapa_olr_wind(
            olr_campo, u_campo, v_campo, lat, lon,
            f'OLR intrasazonal + vento 850 hPa — pentada {d_ini_p} a {d_fim_p}',
            output_dir / nome, input_dir,
        )

    # ---- Produto: mapa do periodo OLR + vento 850 ----
    logger.info('Etapa 12: Mapa do periodo OLR + u850 (com filtro)...')
    d_per = np.intersect1d(
        dates_intra[(dates_intra >= np.datetime64(dt_ini.date())) & (dates_intra <= np.datetime64(dt_fim.date()))],
        dates_wind[(dates_wind >= np.datetime64(dt_ini.date())) & (dates_wind <= np.datetime64(dt_fim.date()))],
    )
    if len(d_per) > 0:
        idx_op = np.isin(dates_intra, d_per)
        idx_wp = np.isin(dates_wind, d_per)
        _plot_mapa_olr_wind(
            olr_com[idx_op].mean(axis=0),
            u_com[idx_wp].mean(axis=0),
            v_com[idx_wp].mean(axis=0),
            lat, lon,
            f'OLR intrasazonal + vento 850 hPa — media {dt_ini.date()} a {dt_fim.date()}',
            periodo_wind_png, input_dir,
        )

    execution_time = time.time() - start_time
    save_cache_metadata(SCRIPT_ID, cache_params, output_files, execution_time)
    logger.info('=' * 80)
    logger.info(f'Script {SCRIPT_ID.upper()} concluido em {execution_time:.1f}s')
    logger.info(f'Saida: {output_dir}')
    logger.info('=' * 80)


if __name__ == '__main__':
    main()
