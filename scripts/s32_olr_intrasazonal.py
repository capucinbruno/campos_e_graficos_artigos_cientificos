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

Modos (MODE):
    - 'reanalysis' (default): OLR PSL (CPC Blended anom) por DATA_INICIAL..DATA_FINAL, com Lanczos.
    - 'forecast': emenda anomalia PSL [init-~180d, init] + previsao GEFS(35d)/CFS(45d). Previsao
      = OLR bruto -> anomalia "igual com igual" (olr - clim_olr_daily, mesma clim CPC). Causal
      (sem Lanczos), a la NCICS. GEFS e CFS sempre rodam. CFS = pseudo-ensemble lagged (ulwtoa).

Saida:
    - reanalysis: {DIR_OUTPUT}/s32_OLR_INTRASAZONAL/REANALISE/
    - forecast:   {DIR_OUTPUT}/s32_OLR_INTRASAZONAL/FORECAST/<MODELO>/<N>_DAY|HOVMOLLER|MEDIA_PERIODO_TOTAL/

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
import matplotlib.patheffects as path_effects
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
# Climatologia/observado de OLR (CPC Blended ABSOLUTO) — p/ anomalia "igual com igual" no forecast.
from app.src.uteis.clim_diaria_olr import clim_olr_daily, olr_obs_daily
# Downloaders de previsao (forecast): GEFS e CFS (pseudo-ensemble lagged) de OLR e u/v 850.
from app.src.uteis.downloaders_cfs_ensemble import (
    CFS_LEAD_DAYS,
    ensure_cfs_olr_for_period,
    ensure_cfs_uv850_for_period,
)
from app.src.uteis.downloaders_gdas_uv850 import ensure_gdas_uv850_for_period
from app.src.uteis.downloaders_gefs_olr import ensure_gefs_olr_fcst_for_period
from app.src.uteis.downloaders_gefs_uv850 import ensure_gefs_uv850_fcst_for_period
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
# Forecast (MODE='forecast') — GEFS (35d) e CFS (45d), espelhando o s31
# ---------------------------------------------------------------------------
FORECAST_MAP_WINDOWS = (1, 2, 3, 5, 7, 10)  # janelas (dias) dos mapas espaciais (estilo NCICS)
_FCST_DL_OLR = {'gefs': ensure_gefs_olr_fcst_for_period, 'cfs': ensure_cfs_olr_for_period}
_FCST_DL_850 = {'gefs': ensure_gefs_uv850_fcst_for_period, 'cfs': ensure_cfs_uv850_for_period}


def _base_output_dir() -> Path:
    """Base do s32: {DIR_OUTPUT}/s32_OLR_INTRASAZONAL (com REANALISE/ e FORECAST/ dentro)."""
    return Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_OLR_INTRASAZONAL'


def _enabled_forecast_models() -> list:
    """GEFS (35d) e CFS (45d) — ambos SEMPRE rodam (não acopla com RUN_* do s34)."""
    return ['gefs', 'cfs']


def _resolve_forecast_init(spec, rodada: int, default_offset_days: int = 0) -> datetime:
    """Init de forecast (vazio/'latest' = hoje+offset na hora RODADA; ISO; ou YYYYMMDDHH)."""
    s = str(spec).strip()
    if s == '' or s.lower() == 'latest':
        base = datetime.now() + timedelta(days=default_offset_days)
        return base.replace(hour=rodada, minute=0, second=0, microsecond=0)
    if len(s) == 10 and s.isdigit():
        return datetime.strptime(s, '%Y%m%d%H')
    return datetime.strptime(s[:10], '%Y-%m-%d').replace(hour=rodada, minute=0, second=0, microsecond=0)


def _forecast_windows(campo, dates, init_date, dias: int, *extras):
    """Tila a PREVISÃO ([init+1 .. fim]) em blocos consecutivos de `dias`, a partir de init
    (só blocos completos; ancorado p/ frente). Retorna [(d_ini, d_fim, médio, *extras_médios), ...]."""
    d0 = np.datetime64(pd.Timestamp(init_date).date())
    fcst = np.where(dates > d0)[0]
    if len(fcst) == 0:
        return []
    out = []
    i, n = int(fcst[0]), len(dates)
    while i + dias <= n:
        sl = slice(i, i + dias)
        out.append((dates[i], dates[i + dias - 1], campo[sl].mean(axis=0))
                   + tuple(e[sl].mean(axis=0) for e in extras))
        i += dias
    return out


def _daily_olr_forecast_on_grid(files, start, end, target_lat, target_lon, logger):
    """Lê NetCDFs de OLR previsto (GEFS/CFS, var 'olr'), filtra horas/ média diária e interpola
    para a grade alvo (2.5° do PSL). Retorna DataArray (time, lat, lon) de OLR BRUTO (W/m²)."""
    t_ini, t_fim = np.datetime64(start.date()), np.datetime64(end.date())
    tgt_lat = xr.DataArray(target_lat, dims=['lat'])
    tgt_lon = xr.DataArray(target_lon, dims=['lon'])
    parts = []
    for fp in files:
        ds = xr.open_dataset(fp, engine='netcdf4')
        try:
            if 'valid_time' in ds.coords and 'time' not in ds.dims:
                ds = ds.rename({'valid_time': 'time'})
            ren = {}
            if 'latitude' in ds.coords and 'lat' not in ds.coords:
                ren['latitude'] = 'lat'
            if 'longitude' in ds.coords and 'lon' not in ds.coords:
                ren['longitude'] = 'lon'
            if ren:
                ds = ds.rename(ren)
            ds = ds.assign_coords(lon=(ds['lon'] % 360)).sortby('lon').sortby('lat')
            da = ds['olr']
            da = da.sel(time=slice(t_ini, t_fim))
            if da.sizes.get('time', 0) == 0:
                continue
            da = da.resample(time='1D').mean()
            da = da.interp(lat=tgt_lat, lon=tgt_lon, method='linear').reset_coords(drop=True)
            parts.append(da.load())
        finally:
            ds.close()
    if not parts:
        raise RuntimeError('Nenhum dado de OLR previsto válido no período.')
    out = xr.concat(parts, dim='time', coords='minimal', compat='override').sortby('time')
    _, uniq = np.unique(out['time'].values, return_index=True)
    return out.isel(time=uniq)


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


_LOGO_CORNERS = {
    'lower-left': ((0, 0), (0, 0)),
    'upper-right': ((1, 1), (1, 1)),
    'upper-left': ((0, 1), (0, 1)),
    'lower-right': ((1, 0), (1, 0)),
}


def _add_logo_to_map(ax, logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=30, corner='lower-left'):
    logo = Image.open(logo_path).convert('RGBA')
    bbox = logo.getbbox()
    if bbox is not None:
        logo = logo.crop(bbox)
    img = np.array(logo)
    imagebox = OffsetImage(img, zoom=zoom)
    xy, box_align = _LOGO_CORNERS.get(corner, _LOGO_CORNERS['lower-left'])
    ab = AnnotationBbox(
        imagebox, xy,
        xycoords=ax.transAxes, xybox=(xoffset, yoffset),
        boxcoords='offset points', box_alignment=box_align,
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
        _add_logo_to_map(ax=ax, logo_path=logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=500,
                         corner='lower-left')
    fig.savefig(str(out_png), dpi=fig.dpi, bbox_inches='tight')
    plt.close('all')


def _lon_we_formatter(x, pos):
    x = x % 360
    if x == 0 or x == 360:
        return '0°'
    if x == 180:
        return '180°'
    return f'{int(x)}°E' if x < 180 else f'{int(360 - x)}°W'


def _mark_forecast_start(ax, dates, init_date):
    """Linha tracejada grossa no init + rotulo 'Previsão' (branco/negrito, contorno preto)."""
    if init_date is None:
        return
    d0 = np.datetime64(pd.Timestamp(init_date).date())
    if not (dates.min() <= d0 <= dates.max()):
        return
    ax.axhline(d0, color='black', linewidth=4.5, linestyle='--', zorder=400)
    ax.annotate('Previsão', xy=(0.5, d0), xycoords=ax.get_yaxis_transform(),
                xytext=(0, 14), textcoords='offset points', ha='center', va='bottom',
                fontsize=20, fontweight='bold', color='white',
                path_effects=[path_effects.withStroke(linewidth=3.5, foreground='black')],
                zorder=401, clip_on=False, annotation_clip=False)


def _plot_hovmoller(hov, lon, dates, titulo, out_png, input_dir, cbar_label='OLR intrasazonal (W/m²)',
                    init_date=None):
    cmap, norm = _cmap_norm()
    hov_s, lonc = add_cyclic_point(hov, coord=lon)
    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.contourf(lonc, dates, hov_s, levels=LEVELS, cmap=cmap, norm=norm, extend='both')
    _mark_forecast_start(ax, dates, init_date)
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
        _add_logo_to_map(ax=ax, logo_path=logo_path, zoom=0.65, xoffset=-6, yoffset=-6, zorder=500,
                         corner='upper-right')
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
        _add_logo_to_map(ax=ax, logo_path=logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=500,
                         corner='lower-left')
    fig.savefig(str(out_png), dpi=fig.dpi, bbox_inches='tight')
    plt.close('all')


def _plot_hovmoller_olr_wind(olr_hov, u_hov, lon, dates, titulo, out_png, input_dir, init_date=None):
    """Hovmöller OLR shaded + isolinhas de u850 (vermelho=positivo, azul=negativo)."""
    cmap, norm = _cmap_norm()
    olr_s, lonc = add_cyclic_point(olr_hov, coord=lon)
    u_s, _ = add_cyclic_point(u_hov, coord=lon)
    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.contourf(lonc, dates, olr_s, levels=LEVELS, cmap=cmap, norm=norm, extend='both')
    _mark_forecast_start(ax, dates, init_date)
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
        _add_logo_to_map(ax=ax, logo_path=logo_path, zoom=0.65, xoffset=-6, yoffset=-6, zorder=500,
                         corner='upper-right')
    fig.savefig(str(out_png), dpi=fig.dpi, bbox_inches='tight')
    plt.close('all')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _run_once(mode: str, fcst_model, logger):
    """Núcleo do s32 para um (modo, modelo). Reanálise: fcst_model=None. Forecast: 'gefs'/'cfs'."""
    is_forecast = mode.startswith('forecast')
    logger.info('=' * 80)
    logger.info(f'SCRIPT {SCRIPT_ID.upper()}: {SCRIPT_DESC}'
                + (f' — FORECAST {fcst_model.upper()}' if fcst_model else ''))
    logger.info('=' * 80)

    janela = int(_cfg('JANELA_MEDIA_MOVEL', 120))
    n_pentadas = int(_cfg('N_PENTADAS', 6))
    hov_dias = int(_cfg('HOVMOLLER_DIAS', 120))
    faixa = list(_cfg('FAIXA_HOVMOLLER', [-5, 5]))
    lanczos_n = int(_cfg('LANCZOS_N', 60))
    period_min = float(_cfg('LANCZOS_PERIOD_MIN', 20.0))
    period_max = float(_cfg('LANCZOS_PERIOD_MAX', 90.0))
    input_dir = Path(settings.DIR_INPUT)
    dados_dir = Path(settings.DIR_DADOS)

    if is_forecast:
        if fcst_model == 'cfs':
            lead_days = int(_cfg('CFS_LEAD_DAYS', CFS_LEAD_DAYS))      # pseudo-ensemble lagged -> ontem
            init = _resolve_forecast_init(_cfg('FORECAST_INIT', ''), 0, default_offset_days=-1)
        else:  # gefs (geavg, 35d)
            rodada = int(_cfg('RODADA', 0))
            if rodada not in (0, 6, 12, 18):
                raise ValueError(f'RODADA deve ser 00/06/12/18 (UTC). Recebido: {rodada:02d}')
            lead_days = int(_cfg('FORECAST_LEAD_DAYS', 35))
            init = _resolve_forecast_init(_cfg('FORECAST_INIT', ''), rodada)
        dt_ini = init
        dt_fim = init + timedelta(days=lead_days)
        n_pentadas = max(1, lead_days // 5)
        output_dir = _base_output_dir() / 'FORECAST' / fcst_model.upper()
    else:
        init = None
        lead_days = 0
        dt_ini = datetime.strptime(settings.DATA_INICIAL, '%Y-%m-%d')
        dt_fim = datetime.strptime(settings.DATA_FINAL, '%Y-%m-%d')
        output_dir = _base_output_dir() / 'REANALISE'

    # ---- caminhos de saida + cache ----
    if is_forecast:
        hov_dir = output_dir / 'HOVMOLLER'
        periodo_dir = output_dir / 'MEDIA_PERIODO_TOTAL'
        hov_png = hov_dir / 'olr_hovmoller_forecast.png'
        hov_wind_png = hov_dir / 'olr_u850_hovmoller_forecast.png'
        periodo_wind_png = periodo_dir / 'olr_u850_periodo_forecast.png'
        output_files = [str(hov_png), str(hov_wind_png), str(periodo_wind_png)]
    else:
        hov_com_png = output_dir / 'olr_hovmoller_com_filtro.png'
        hov_sem_png = output_dir / 'olr_hovmoller_sem_filtro.png'
        hov_wind_png = output_dir / 'olr_u850_hovmoller_com_filtro.png'
        periodo_com_png = output_dir / 'olr_periodo_com_filtro.png'
        periodo_sem_png = output_dir / 'olr_periodo_sem_filtro.png'
        periodo_wind_png = output_dir / 'olr_u850_periodo_com_filtro.png'
        output_files = [str(hov_com_png), str(hov_sem_png), str(hov_wind_png),
                        str(periodo_com_png), str(periodo_sem_png), str(periodo_wind_png)]

    cache_params = {
        'MODE': mode, 'forecast_model': fcst_model, 'lead_days': lead_days,
        'DATA_INICIAL': settings.DATA_INICIAL if not is_forecast else init.strftime('%Y-%m-%d %H'),
        'DATA_FINAL': settings.DATA_FINAL if not is_forecast else dt_fim.strftime('%Y-%m-%d'),
        'janela': janela, 'n_pentadas': n_pentadas, 'hov_dias': hov_dias, 'faixa': faixa,
        'lanczos_n': lanczos_n, 'period_min': period_min, 'period_max': period_max,
        'metodo': ('CPC running-mean causal (forecast a la NCICS, igual-com-igual via clim_olr_daily)'
                   if is_forecast else 'CPC running-mean + Lanczos bandpass (20-90d) + u850'),
        'script_version': '3.0',
    }
    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('CACHE VALIDO! Pulando execucao.')
        return

    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    dados_dir.mkdir(parents=True, exist_ok=True)
    if is_forecast:
        hov_dir.mkdir(parents=True, exist_ok=True)
        periodo_dir.mkdir(parents=True, exist_ok=True)

    inicio_interesse = min(dt_ini, dt_fim - timedelta(days=max(hov_dias, n_pentadas * 5) - 1))
    start_dl = inicio_interesse - timedelta(days=janela + lanczos_n + 2)
    hist_end = init if is_forecast else dt_fim  # em forecast a reanalise vai so ate init
    logger.info(f'Periodo de interesse: {dt_ini.date()} a {dt_fim.date()}')
    logger.info(f'Janela de download: {start_dl.date()} a {dt_fim.date()}')

    # ---- Etapa 1: OLR (anomalia CPC Blended; em forecast + previsao GEFS/CFS) ----
    olr_path = dados_dir / OLR_FILE_NAME
    # Em forecast a obs (PSL) tem latencia e nao chega ate `init`; exige/valida so o que houver.
    chk_end = hist_end if not is_forecast else (datetime.now() - timedelta(days=10))
    if not arquivo_cobre_periodo(olr_path, np.datetime64(start_dl.date()), np.datetime64(chk_end.date())):
        logger.info('Etapa 1: Download OLR (CPC Blended 2.5°, PSL/NOAA)...')
        download_with_progress(url=OLR_URL, output_path=str(olr_path), description=OLR_FILE_NAME,
                               max_retries=5, force=True, engine=DownloadEngine.ARIA2, timeout=300)
    ds = _normalize_ds(load_dataset(str(olr_path)))
    psl_last = pd.Timestamp(ds['time'].values.max()).to_pydatetime()
    eff_end = hist_end if not is_forecast else min(hist_end, psl_last)
    validar_cobertura_temporal(ds, np.datetime64(start_dl.date()), np.datetime64(eff_end.date()),
                               nome='arquivo OLR CPC Blended')
    logger.info('Etapa 2: Carregando serie temporal OLR (anomalia historica ate {})...', eff_end.date())
    hist = ds.sel(time=slice(np.datetime64(start_dl.date()), np.datetime64(eff_end.date())))['olr'].load()
    lat = hist['lat'].values.astype(float)
    lon = hist['lon'].values.astype(float)

    if is_forecast:
        logger.info('Etapa 2b: {} OLR previsto (anomalia = olr - clim_olr_daily)...', fcst_model.upper())
        fcst_files = list(_FCST_DL_OLR[fcst_model](init=init, lead_hours=lead_days * 24,
                                                   force_redownload=False))
        olr_fcst = _daily_olr_forecast_on_grid(fcst_files, init, dt_fim, lat, lon, logger)
        fdates = np.array([np.datetime64(pd.Timestamp(t).date()) for t in olr_fcst['time'].values])
        fcst_anom = olr_fcst.values - clim_olr_daily(fdates, lat, lon)
        fcst_da = xr.DataArray(fcst_anom, dims=('time', 'lat', 'lon'),
                               coords={'time': olr_fcst['time'].values, 'lat': lat, 'lon': lon})
        # emenda: historico (PSL anom) ate init + previsao (anomalia) depois de init
        fcst_da = fcst_da.sel(time=slice(np.datetime64((init + timedelta(days=1)).date()), None))
        olr_anom_da = xr.concat([hist, fcst_da], dim='time').sortby('time')
        # reindex p/ diario CONTINUO ate o ULTIMO dia COM DADO (nao ate dt_fim): se a previsao
        # truncou (ex.: estendido do 00Z de hoje ainda nao publicado), NAO estende o eixo com NaN.
        # O gap interior (latencia da obs -> inicio da previsao) e interpolado no filtro.
        last_valid = pd.Timestamp(olr_anom_da['time'].values.max()).date()
        olr_anom_da = olr_anom_da.reindex(time=pd.date_range(start_dl.date(), last_valid, freq='1D'))
    else:
        olr_anom_da = hist
    logger.info('Serie OLR: {} dias, grade {}x{}', olr_anom_da.sizes['time'], len(lat), len(lon))

    # ---- Etapa 3: filtro intrasazonal ----
    logger.info('Etapa 3: Filtro intrasazonal (running mean {}d{})...',
                janela, '' if is_forecast else f' + Lanczos {int(period_min)}-{int(period_max)}d')
    olr_com, olr_sem, dates_intra = _olr_intrasazonal(olr_anom_da, janela, lanczos_n, period_min, period_max)
    logger.info('Serie filtrada: {} dias ({} a {})', len(dates_intra), dates_intra[0], dates_intra[-1])

    # ---- Etapa 4-9: vento 850 (reanalise + previsao no forecast) ----
    logger.info('Etapa 4: Download/serie u/v 850 hPa...')
    cutoff = (datetime.now() - timedelta(days=ERA5_LATENCY_DAYS)).replace(hour=0, minute=0, second=0, microsecond=0)
    wind_files = []
    if start_dl < cutoff:
        wind_files += list(ensure_era5_uv850_for_period(
            start=start_dl, end=min(hist_end, cutoff - timedelta(days=1)),
            hours_utc=list(DEFAULT_SYNOPTIC_HOURS), force_redownload=False))
    if hist_end >= cutoff:
        wind_files += list(ensure_gdas_uv850_for_period(
            start=max(start_dl, cutoff), end=hist_end, force_redownload=False))
    if is_forecast:
        wind_files += list(_FCST_DL_850[fcst_model](init=init, lead_hours=lead_days * 24, force_redownload=False))
    u_da, v_da = _daily_series_uv850(wind_files, start_dl, dt_fim, lat, lon, logger)
    u_da, v_da = _reindex_daily(u_da), _reindex_daily(v_da)
    u_com, v_com, u_sem, v_sem = _wind850_filtered(u_da, v_da, janela, lanczos_n, period_min, period_max)
    dates_wind = np.array([np.datetime64(pd.Timestamp(u_da['time'].values[i]).date())
                           for i in range(janela, u_da.sizes['time'])])

    # No forecast usa-se o sinal CAUSAL (sem Lanczos), igual ao s31; na reanalise, ambos.
    if is_forecast:
        _forecast_products(
            output_dir, init, dates_intra, dates_wind, olr_sem, u_sem, v_sem, lat, lon,
            faixa, hov_dias, hov_png, hov_wind_png, periodo_wind_png, input_dir, fcst_model, logger)
    else:
        _reanalysis_products(
            output_dir, dt_ini, dt_fim, n_pentadas, dates_intra, dates_wind,
            olr_com, olr_sem, u_com, v_com, lat, lon, faixa, hov_dias,
            hov_com_png, hov_sem_png, hov_wind_png, periodo_com_png, periodo_sem_png, periodo_wind_png,
            input_dir, logger)

    execution_time = time.time() - start_time
    save_cache_metadata(SCRIPT_ID, cache_params, output_files, execution_time)
    logger.info('=' * 80)
    logger.info(f'Script {SCRIPT_ID.upper()} concluido em {execution_time:.1f}s | Saida: {output_dir}')
    logger.info('=' * 80)


def _reanalysis_products(output_dir, dt_ini, dt_fim, n_pentadas, dates_intra, dates_wind,
                         olr_com, olr_sem, u_com, v_com, lat, lon, faixa, hov_dias,
                         hov_com_png, hov_sem_png, hov_wind_png, periodo_com_png, periodo_sem_png,
                         periodo_wind_png, input_dir, logger):
    """Produtos da REANALISE (inalterados): pentadas + Hovmoller + periodo (com/sem Lanczos) + OLR/u850."""
    versoes = [
        ('com_filtro', olr_com, 'OLR intrasazonal', 'OLR intrasazonal (W/m²)', hov_com_png, periodo_com_png),
        ('sem_filtro', olr_sem, 'OLR', 'OLR (W/m²)', hov_sem_png, periodo_sem_png),
    ]
    logger.info('Etapa 5: Mapas de pentada (com e sem filtro Lanczos)...')
    for antigo in output_dir.glob('olr_*pentada*.png'):
        antigo.unlink()
    for sufixo, dados, rotulo, cbar_label, _, _ in versoes:
        for d_ini_p, d_fim_p, campo in _agrupa_pentadas(dados, dates_intra, n_pentadas):
            _plot_mapa(campo, lat, lon, f'{rotulo} — pentada {d_ini_p} a {d_fim_p}',
                       output_dir / f'olr_pentada_{sufixo}_{d_ini_p}_a_{d_fim_p}.png', input_dir,
                       cbar_label=cbar_label)
    logger.info('Etapa 6: Hovmoller (com e sem filtro)...')
    m_hov = dates_intra >= (np.datetime64(dt_fim.date()) - np.timedelta64(hov_dias - 1, 'D'))
    for sufixo, dados, rotulo, cbar_label, hov_png_v, _ in versoes:
        hov = _media_faixa_latitude(dados[m_hov], lat, faixa[0], faixa[1])
        _plot_hovmoller(hov, lon, dates_intra[m_hov], f'{rotulo} — Hovmöller ({faixa[0]}° a {faixa[1]}°)',
                        hov_png_v, input_dir, cbar_label=cbar_label)
    logger.info('Etapa 7: Mapa do periodo (com e sem filtro)...')
    m_per = (dates_intra >= np.datetime64(dt_ini.date())) & (dates_intra <= np.datetime64(dt_fim.date()))
    if m_per.any():
        for sufixo, dados, rotulo, cbar_label, _, periodo_png_v in versoes:
            _plot_mapa(dados[m_per].mean(axis=0), lat, lon,
                       f'{rotulo} — media {dt_ini.date()} a {dt_fim.date()}',
                       periodo_png_v, input_dir, cbar_label=cbar_label)
    # OLR + u850
    logger.info('Etapa 8: Hovmoller/pentadas/periodo OLR + u850...')
    d_hov = np.intersect1d(dates_intra[m_hov], dates_wind[dates_wind >= (np.datetime64(dt_fim.date()) - np.timedelta64(hov_dias - 1, 'D'))])
    if len(d_hov):
        olr_h = _media_faixa_latitude(olr_com[np.isin(dates_intra, d_hov)], lat, faixa[0], faixa[1])
        u_h = _media_faixa_latitude(u_com[np.isin(dates_wind, d_hov)], lat, faixa[0], faixa[1])
        _plot_hovmoller_olr_wind(olr_h, u_h, lon, d_hov,
                                 f'OLR intrasazonal + u850 — Hovmöller ({faixa[0]}° a {faixa[1]}°)',
                                 hov_wind_png, input_dir)
    for antigo in output_dir.glob('olr_u850_pentada_*.png'):
        antigo.unlink()
    for d_ini_p, d_fim_p, olr_campo in _agrupa_pentadas(olr_com, dates_intra, n_pentadas):
        mw = (dates_wind >= d_ini_p) & (dates_wind <= d_fim_p)
        if not mw.any():
            continue
        _plot_mapa_olr_wind(olr_campo, u_com[mw].mean(axis=0), v_com[mw].mean(axis=0), lat, lon,
                            f'OLR intrasazonal + vento 850 hPa — pentada {d_ini_p} a {d_fim_p}',
                            output_dir / f'olr_u850_pentada_com_filtro_{d_ini_p}_a_{d_fim_p}.png', input_dir)
    mper_o = (dates_intra >= np.datetime64(dt_ini.date())) & (dates_intra <= np.datetime64(dt_fim.date()))
    d_per = np.intersect1d(dates_intra[mper_o], dates_wind[(dates_wind >= np.datetime64(dt_ini.date())) & (dates_wind <= np.datetime64(dt_fim.date()))])
    if len(d_per):
        _plot_mapa_olr_wind(olr_com[np.isin(dates_intra, d_per)].mean(axis=0),
                            u_com[np.isin(dates_wind, d_per)].mean(axis=0),
                            v_com[np.isin(dates_wind, d_per)].mean(axis=0), lat, lon,
                            f'OLR intrasazonal + vento 850 hPa — media {dt_ini.date()} a {dt_fim.date()}',
                            periodo_wind_png, input_dir)


def _forecast_products(output_dir, init, dates_intra, dates_wind, olr_sem, u_sem, v_sem, lat, lon,
                       faixa, hov_dias, hov_png, hov_wind_png, periodo_wind_png, input_dir, fcst_model, logger):
    """Produtos da PREVISAO (a la s31): mapas OLR+u850 por janela 1/2/3/5/7/10d em <N>_DAY/,
    Hovmollers em HOVMOLLER/, media do periodo em MEDIA_PERIODO_TOTAL/. Só causal (sem Lanczos)."""
    rotulo = f'OLR intrasazonal (previsão {fcst_model.upper()})'
    # alinha OLR e vento nas datas comuns
    d_common = np.intersect1d(dates_intra, dates_wind)
    olr_a = olr_sem[np.isin(dates_intra, d_common)]
    u_a = u_sem[np.isin(dates_wind, d_common)]
    v_a = v_sem[np.isin(dates_wind, d_common)]

    # Mapas espaciais OLR+u850 por janela (estilo NCICS "Select Days")
    windows = [int(w) for w in _cfg('FORECAST_MAP_WINDOWS', list(FORECAST_MAP_WINDOWS))]
    logger.info('Mapas espaciais OLR+u850 por janela: {} dias...', windows)
    for w in windows:
        wdir = output_dir / f'{w}_DAY'
        wdir.mkdir(parents=True, exist_ok=True)
        for antigo in wdir.glob('olr_*.png'):
            antigo.unlink()
        blocos = _forecast_windows(olr_a, d_common, init, w, u_a, v_a)
        for d_ini, d_fim, olr_c, u_c, v_c in blocos:
            _plot_mapa_olr_wind(olr_c, u_c, v_c, lat, lon,
                                f'{rotulo} + vento 850 — média {w}d: {d_ini} a {d_fim}',
                                wdir / f'olr_u850_{w}day_{d_ini}_a_{d_fim}.png', input_dir)
        logger.info('  {}_DAY: {} mapas', w, len(blocos))

    # Hovmollers (OLR e OLR+u850), marcando o inicio da previsao
    m_hov = dates_intra >= (np.datetime64(pd.Timestamp(dates_intra[-1])) - np.timedelta64(hov_dias - 1, 'D'))
    _plot_hovmoller(_media_faixa_latitude(olr_sem[m_hov], lat, faixa[0], faixa[1]), lon, dates_intra[m_hov],
                    f'{rotulo} — Hovmöller ({faixa[0]}° a {faixa[1]}°)', hov_png, input_dir, init_date=init)
    mhw = np.isin(d_common, dates_intra[m_hov])
    _plot_hovmoller_olr_wind(_media_faixa_latitude(olr_a[mhw], lat, faixa[0], faixa[1]),
                             _media_faixa_latitude(u_a[mhw], lat, faixa[0], faixa[1]), lon, d_common[mhw],
                             f'{rotulo} + u850 — Hovmöller ({faixa[0]}° a {faixa[1]}°)',
                             hov_wind_png, input_dir, init_date=init)

    # Media do periodo total (toda a previsao) OLR+u850
    mfc = d_common > np.datetime64(pd.Timestamp(init).date())
    if mfc.any():
        _plot_mapa_olr_wind(olr_a[mfc].mean(axis=0), u_a[mfc].mean(axis=0), v_a[mfc].mean(axis=0), lat, lon,
                            f'{rotulo} + vento 850 — média {d_common[mfc][0]} a {d_common[mfc][-1]}',
                            periodo_wind_png, input_dir)


def main():
    """Entry point. Reanálise: 1 execução. Forecast: laço pelos modelos (GEFS + CFS sempre rodam)."""
    logger = get_logger(SCRIPT_ID)
    mode = str(_cfg('MODE', 'reanalysis')).strip().lower()
    if mode.startswith('forecast'):
        models = _enabled_forecast_models()
        logger.info('s32 FORECAST — modelos habilitados: {}', [m.upper() for m in models])
        for model in models:
            _run_once('forecast', model, logger)
    else:
        _run_once('reanalysis', None, logger)


if __name__ == '__main__':
    main()
