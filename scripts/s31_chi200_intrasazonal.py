# -*- coding: utf-8 -*-
"""
s31 - CHI200 intrasazonal (potencial de velocidade 200 hPa, banda intrasazonal/MJO).

Em vez da anomalia media do periodo (s03), isola o sinal INTRASAZONAL pelo metodo
operacional do CPC (remocao de media movel):

    anom(t)  = u/v200(t) - LTM_diaria(dia-do-ano)        # remove o ciclo sazonal
    intra(t) = anom(t) - media(anom em [t-N, t-1])        # remove o interanual (N=120d)
    -> Poisson (vento divergente) -> CHI200 intrasazonal de cada dia

Produtos (todos da mesma serie diaria de chi intrasazonal):
    1. Mapas de pentada (ultimas N_PENTADAS pentadas de 5 dias)
    2. Hovmoller (lon x tempo, media numa faixa equatorial): com filtro, sem filtro
       e um terceiro com chi filtrado (shaded) + vento zonal 850 hPa filtrado (isolinhas)
    3. Mapa do periodo (media de intra em [DATA_INICIAL, DATA_FINAL])

Modos (MODE):
    - 'reanalysis' (default): ERA5/GDAS por DATA_INICIAL..DATA_FINAL, com Lanczos + WW.
    - 'forecast': emenda reanalise [init-~266d, init] + GEFS geavg [init, init+lead] numa
      serie diaria unica e roda a MESMA cadeia. Como `remove_media_movel` e CAUSAL (so usa
      os 120 dias anteriores), o intrasazonal e valido em toda a previsao. NAO usa Lanczos
      (filtro centrado n>lead degradaria a previsao inteira): a la NCICS/Schreck, usa o sinal
      causal + Wheeler-Weickmann com a previsao servindo de padding de borda. Pentadas para
      frente a partir de init; Hovmoller marca o inicio da previsao. So GEFS por enquanto.

Dados:
    - ERA5/GDAS u/v 200 e 850 hPa diario (downloaders existentes)
    - GEFS geavg u/v 200 e 850 hPa (forecast) — downloaders_gefs_fcst200 / _uv850
    - LTM diaria NCEP u/v 200 (app/src/uteis/clim_diaria_uv200_ltm)

Saida:
    - reanalysis: {DIR_OUTPUT}/s31_CHI200_INTRASAZONAL/REANALISE/
    - forecast:   {DIR_OUTPUT}/s31_CHI200_INTRASAZONAL/FORECAST/<MODELO>/<N>_DAY|HOVMOLLER/

Criado em: 2026-06-09
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
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from PIL import Image

from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.common.logo_helper import resolve_logo_path
from app.common.logo_helper import proportional_logo_zoom
from app.src.uteis.chi200_intrasazonal import (
    agrupa_pentadas,
    chi200_intrasazonal_series,
    lanczos_bandpass,
    media_faixa_latitude,
    remove_media_movel,
    ww_filter_chi_modes,
)
from app.src.uteis.clim_diaria_uv200_ltm import clim_u850_daily, clim_uv200_daily
from app.src.uteis.downloaders_gdas_uv200 import ensure_gdas_uv200_for_period
from app.src.uteis.downloaders_gdas_uv850 import ensure_gdas_uv850_for_period
# Modo forecast (MODE='forecast'): GEFS media do ensemble (geavg) — u/v 200 e 850 hPa.
from app.src.uteis.downloaders_cfs_ensemble import (
    CFS_LEAD_DAYS,
    ensure_cfs_fcst200_for_period,
    ensure_cfs_uv850_for_period,
)
from app.src.uteis.downloaders_gefs_fcst200 import ensure_gefs_fcst200_for_period
from app.src.uteis.downloaders_gefs_uv850 import ensure_gefs_uv850_fcst_for_period
from app.src.uteis.downloaders_wind200 import ensure_era5_uv200_for_period
from app.src.uteis.downloaders_wind850 import ensure_era5_uv850_for_period
from app.src.uteis.plot_chi200 import (
    _drop_or_collapse_expver,
    _ensure_time_coord,
    _find_uv_vars,
    _normalize_latlon_names,
    _sort_and_dedup_time,
)

SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's31'
SCRIPT_NAME = Path(__file__).stem
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
ERA5_LATENCY_DAYS = 5
DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)
CHI_SCALE = 1e5  # chi plotado em unidades de 1e5 m2/s (igual ao s03)
LEVELS = np.arange(-90, 100, 10)  # -90..90 de 10 em 10 (×10⁵ m²/s)
# Niveis de u850 intrasazonal (m/s) para isolinhas no 3o Hovmoller (igual ao s32)
LEVELS_U850 = [-6, -4, -2, 2, 4, 6]
QUIVER_STEP = 2        # subamostrar a cada 2 pontos de grade (~5°)
QUIVER_SCALE = 300.0   # escala das setas (maior = setas menores)
QUIVER_WIDTH = 0.0012
QUIVER_MIN_MAG = 0.3   # m/s: filtra vetores muito fracos
QUIVER_HEADWIDTH = 3.2
QUIVER_HEADLENGTH = 4.2
QUIVER_HEADAXISLENGTH = 3.8
# Paleta verde -> bege -> marrom (igual s03): divergencia (verde) x convergencia (marrom)
CHI200_COLORS = [
    '#005a45', '#0f7a6c', '#2e9b96', '#62bdb7', '#9dd8d2', '#dff3f1',
    '#f7f4eb', '#e7d9a9', '#d6b566', '#bd8a35', '#9a6313', '#6f4300',
]

# Isolinhas Wheeler-Weickmann sobrepostas aos mapas (linhas, nao shading)
# Magnitudes calibradas com filtro k=1-2 tropical-mean + Gauss 20°, T~120d:
#   MJO: max≈76, Kelvin k=1-2: max esperado ~30-50 (×10^5 m^2/s)
WW_LEVELS = [-30, -15, 15, 30]   # x10^5 m^2/s
WW_STYLE: dict[str, dict] = {
    'mjo':    {'colors': 'black', 'linewidths': 1.5},
    # 'kelvin': {'colors': 'blue',  'linewidths': 1.5},
}


def _cfg(name: str, default):
    return settings.get(name, default)


# Cantos suportados p/ o logo (ancora em transAxes, alinhamento da caixa do logo).
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
    imagebox = OffsetImage(img, zoom=proportional_logo_zoom(ax, img.shape[1]))
    xy, box_align = _LOGO_CORNERS.get(corner, _LOGO_CORNERS['lower-left'])
    ab = AnnotationBbox(
        imagebox,
        xy,
        xycoords=ax.transAxes,
        xybox=(xoffset, yoffset),
        boxcoords='offset points',
        box_alignment=box_align,
        frameon=False,
        pad=0,
        zorder=zorder,
        clip_on=False,
    )
    ax.add_artist(ab)


# ---------------------------------------------------------------------------
# Serie diaria de u/v 200 regridada para a grade da LTM (2.5°)
# ---------------------------------------------------------------------------
def _daily_series_uv200(
    files, start: datetime, end: datetime, target_lat: np.ndarray, target_lon: np.ndarray, logger,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Le os arquivos ERA5/GDAS, filtra horas sinoticas, faz media diaria e interpola
    para a grade alvo (2.5°). Retorna (u_da, v_da) em (time, lat, lon), lat ascendente.
    """
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
            # Longitude em 0-360 (igual a grade da LTM). NAO usar -180..180 aqui: senao a
            # interpolacao para lon 180-360 cai fora do dominio e gera NaN em metade do globo.
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
            # media diaria + remove 29/02
            da_u = da_u.resample(time='1D').mean()
            da_v = da_v.resample(time='1D').mean()
            keep = ~((da_u['time'].dt.month == 2) & (da_u['time'].dt.day == 29))
            da_u, da_v = da_u.isel(time=keep.values), da_v.isel(time=keep.values)
            # interpola para a grade alvo e descarta coords auxiliares (valid_time, number,
            # expver...) que conflitam no concat entre ERA5 e GDAS
            da_u = da_u.interp(lat=tgt_lat, lon=tgt_lon, method='linear').reset_coords(drop=True)
            da_v = da_v.interp(lat=tgt_lat, lon=tgt_lon, method='linear').reset_coords(drop=True)
            us.append(da_u.load())
            vs.append(da_v.load())
            logger.info('Serie diaria: {} -> {} dias', fp.name, da_u.sizes['time'])
        finally:
            ds.close()

    if not us:
        raise RuntimeError('Nenhum dado u/v 200 valido no periodo de download.')

    u_da = xr.concat(us, dim='time', coords='minimal', compat='override').sortby('time')
    v_da = xr.concat(vs, dim='time', coords='minimal', compat='override').sortby('time')
    # dedup tempo (sobreposicao entre arquivos)
    _, uniq = np.unique(u_da['time'].values, return_index=True)
    u_da = u_da.isel(time=uniq)
    v_da = v_da.isel(time=uniq)
    return u_da, v_da


def _reindex_daily(da: xr.DataArray) -> xr.DataArray:
    """Reindexa para eixo diario contiguo e interpola buracos pequenos (filtro index=calendario)."""
    t0 = pd.Timestamp(da['time'].values[0]).normalize()
    t1 = pd.Timestamp(da['time'].values[-1]).normalize()
    full = pd.date_range(t0, t1, freq='1D')
    da = da.reindex(time=full)
    return da.interpolate_na(dim='time', method='linear', fill_value='extrapolate')


# ---------------------------------------------------------------------------
# Plotagem
# ---------------------------------------------------------------------------
def _cmap_norm():
    # extend='both' exige 2 cores extras (uma por extensao) alem das len(LEVELS)-1 faixas
    cmap = LinearSegmentedColormap.from_list('chi200', CHI200_COLORS, N=len(LEVELS) + 1)
    norm = BoundaryNorm(LEVELS, cmap.N, extend='both')
    return cmap, norm


def _plot_mapa(chi2d, lat, lon, titulo, out_png, input_dir, u_div=None, v_div=None, ww_chi=None, cbar_label='CHI200 intrasazonal (×10⁵ m²/s)'):
    cmap, norm = _cmap_norm()
    arr, lonc = add_cyclic_point(chi2d / CHI_SCALE, coord=lon)
    LON2D, LAT2D = np.meshgrid(lonc, lat)
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_xlim([-180, 180])
    ax.set_ylim([-60, 60])
    im = ax.contourf(LON2D, LAT2D, arr, levels=LEVELS, cmap=cmap, norm=norm,
                     extend='both', transform=ccrs.PlateCarree(central_longitude=0),
                     transform_first=True)

    # Isolinhas Wheeler-Weickmann (MJO=laranja dashed, Kelvin=azul dotted)
    if ww_chi:
        ls_ww = ['solid' if lv < 0 else 'dashed' for lv in WW_LEVELS]
        for mode, chi_mode in ww_chi.items():
            style = WW_STYLE.get(mode)
            if style is None:
                continue
            arr_ww, _ = add_cyclic_point(chi_mode / CHI_SCALE, coord=lon)
            ax.contour(LON2D, LAT2D, arr_ww, levels=WW_LEVELS,
                       transform=ccrs.PlateCarree(central_longitude=0),
                       linestyles=ls_ww, zorder=6, transform_first=True, **style)

    if u_div is not None and v_div is not None:
        s = QUIVER_STEP
        lon_q, lat_q = lon[::s], lat[::s]
        u_q, v_q = u_div[::s, ::s], v_div[::s, ::s]
        mag = np.hypot(u_q, v_q)
        u_q = np.ma.array(u_q, mask=mag < QUIVER_MIN_MAG)
        v_q = np.ma.array(v_q, mask=mag < QUIVER_MIN_MAG)
        ax.quiver(
            lon_q, lat_q, u_q, v_q,
            transform=ccrs.PlateCarree(),
            color='black', pivot='mid',
            scale=QUIVER_SCALE, width=QUIVER_WIDTH,
            headwidth=QUIVER_HEADWIDTH, headlength=QUIVER_HEADLENGTH,
            headaxislength=QUIVER_HEADAXISLENGTH, zorder=5,
        )
    ax.add_feature(NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '50m',
                                       facecolor='none', edgecolor='black'), linewidth=0.6, zorder=100)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.0, zorder=100)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.0, zorder=100)
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.3)
    gl.top_labels = gl.right_labels = False
    ax.set_title(titulo, fontsize=15, loc='left')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2.5%', pad=0.08, axes_class=plt.Axes)
    cbar = plt.colorbar(im, cax=cax, ticks=LEVELS[::2], extend='both')
    cbar.set_label(cbar_label, size=12)
    logo_path = resolve_logo_path(input_dir)
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
    """Linha tracejada grossa marcando o inicio da previsao + rotulo 'Previsão' centrado.

    'Previsão' fica no meio horizontal da area de plotagem, logo acima da linha, em branco
    com negrito e contorno preto.
    """
    if init_date is None:
        return
    d0 = np.datetime64(pd.Timestamp(init_date).date())
    if not (dates.min() <= d0 <= dates.max()):
        return
    ax.axhline(d0, color='black', linewidth=4.5, linestyle='--', zorder=400)
    # get_yaxis_transform: x em fracao do eixo (0.5 = meio horizontal), y em coords de DADOS (a data d0).
    # xytext em offset points sobe o rotulo ~14 pt acima da linha (independente da inversao do eixo).
    ax.annotate(
        'Previsão', xy=(0.5, d0), xycoords=ax.get_yaxis_transform(),
        xytext=(0, 14), textcoords='offset points',
        ha='center', va='bottom', fontsize=20, fontweight='bold', color='white',
        path_effects=[path_effects.withStroke(linewidth=3.5, foreground='black')],
        zorder=401, clip_on=False, annotation_clip=False,
    )


def _plot_hovmoller(hov, lon, dates, titulo, out_png, input_dir, cbar_label='CHI200 intrasazonal (×10⁵ m²/s)',
                    init_date=None):
    cmap, norm = _cmap_norm()
    hov_s, lonc = add_cyclic_point(hov / CHI_SCALE, coord=lon)
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
    logo_path = resolve_logo_path(input_dir)
    if logo_path is not None and logo_path.exists():
        _add_logo_to_map(ax=ax, logo_path=logo_path, zoom=0.65, xoffset=-6, yoffset=-6, zorder=500,
                         corner='upper-right')
    fig.savefig(str(out_png), dpi=fig.dpi, bbox_inches='tight')
    plt.close('all')


def _u850_intrasazonal(
    u_anom: np.ndarray, janela: int, lanczos_n: int, period_min: float, period_max: float,
    aplicar_lanczos: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Vento zonal 850 hPa intrasazonal a partir da anomalia diaria (u850 - LTM).

    Mesma cadeia do chi200: anomalia vs LTM diaria -> remove media movel (janela) ->
    Lanczos bandpass. Retorna (u_com, idx) com idx = indices no eixo de tempo original.

    `aplicar_lanczos=False` (modo forecast): devolve so o sinal causal (trailing-mean), pois
    o Lanczos centrado (n>lead) degradaria toda a previsao.
    """
    u_sem, idx = remove_media_movel(u_anom, janela)
    if not aplicar_lanczos:
        return np.nan_to_num(u_sem), idx
    u_com = np.nan_to_num(lanczos_bandpass(u_sem, period_min, period_max, lanczos_n))
    return u_com, idx


def _plot_hovmoller_chi_wind(
    chi_hov, u_hov, lon, dates, titulo, out_png, input_dir,
    cbar_label='CHI200 intrasazonal (×10⁵ m²/s)', init_date=None,
):
    """Hovmoller CHI200 (shaded) + isolinhas de u850 intrasazonal (azul=negativo, vermelho=positivo)."""
    cmap, norm = _cmap_norm()
    chi_s, lonc = add_cyclic_point(chi_hov / CHI_SCALE, coord=lon)
    u_s, _ = add_cyclic_point(u_hov, coord=lon)
    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.contourf(lonc, dates, chi_s, levels=LEVELS, cmap=cmap, norm=norm, extend='both')
    _mark_forecast_start(ax, dates, init_date)
    # Isolinhas u850: negativo=azul (leste anomalo), positivo=vermelho (oeste anomalo)
    neg = [lv for lv in LEVELS_U850 if lv < 0]
    pos = [lv for lv in LEVELS_U850 if lv > 0]
    if neg:
        ax.contour(lonc, dates, u_s, levels=neg, colors='blue', linewidths=1.2, linestyles='solid')
    if pos:
        ax.contour(lonc, dates, u_s, levels=pos, colors='red', linewidths=1.2, linestyles='solid')
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(plt.MultipleLocator(60))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_lon_we_formatter))
    ax.set_xlabel('Longitude', fontsize=13)
    ax.set_ylabel('Data', fontsize=13)
    ax.set_title(titulo, fontsize=14, loc='left')
    cbar = fig.colorbar(im, ax=ax, ticks=LEVELS[::2], extend='both', pad=0.02)
    cbar.set_label(cbar_label, size=12)
    logo_path = resolve_logo_path(input_dir)
    if logo_path is not None and logo_path.exists():
        _add_logo_to_map(ax=ax, logo_path=logo_path, zoom=0.65, xoffset=-6, yoffset=-6, zorder=500,
                         corner='upper-right')
    fig.savefig(str(out_png), dpi=fig.dpi, bbox_inches='tight')
    plt.close('all')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _resolve_forecast_init(spec, rodada: int, default_offset_days: int = 0) -> datetime:
    """Init da rodada de forecast a partir de FORECAST_INIT (vazio/'latest' = hoje) na hora RODADA.

    `default_offset_days`: deslocamento aplicado quando FORECAST_INIT é vazio (ex.: -1 = ontem,
    usado pelo CFS, cujo pseudo-ensemble lagged precisa do dia com os 16 ciclos já publicados).
    Aceita ISO 'YYYY-MM-DD' (hora vem da RODADA) ou timestamp 'YYYYMMDDHH' (compat. s34).
    """
    s = str(spec).strip()
    if s == '' or s.lower() == 'latest':
        base = datetime.now() + timedelta(days=default_offset_days)
        return base.replace(hour=rodada, minute=0, second=0, microsecond=0)
    if len(s) == 10 and s.isdigit():  # YYYYMMDDHH
        return datetime.strptime(s, '%Y%m%d%H')
    return datetime.strptime(s[:10], '%Y-%m-%d').replace(hour=rodada, minute=0, second=0, microsecond=0)


# Despacho de downloaders de previsão por modelo (200 hPa e 850 hPa).
_FCST_DL_200 = {
    'gefs': ensure_gefs_fcst200_for_period,
    'cfs': ensure_cfs_fcst200_for_period,
}
_FCST_DL_850 = {
    'gefs': ensure_gefs_uv850_fcst_for_period,
    'cfs': ensure_cfs_uv850_for_period,
}


def _base_output_dir() -> Path:
    """Pasta base do s31: {DIR_OUTPUT}/s31_CHI200_INTRASAZONAL (com REANALISE/ e FORECAST/ dentro)."""
    return Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_CHI200_INTRASAZONAL'


def _enabled_forecast_models() -> list:
    """Modelos de previsão do s31: GEFS (35d) e CFS (45d) — ambos SEMPRE rodam, cada um com seu
    horizonte. (Não reusa RUN_GEFS/RUN_CFS do s34 para não acoplar os dois scripts.)"""
    return ['gefs', 'cfs']


# Tamanhos de janela (dias) dos mapas espaciais no forecast — espelha o "Select Days" do NCICS.
FORECAST_MAP_WINDOWS = (1, 2, 3, 5, 7, 10)


def _forecast_windows(chi, dates, init_date, dias: int, *extras):
    """Tila a PREVISÃO ([init+1 .. fim]) em blocos consecutivos de `dias` dias, a partir de init.

    Só blocos COMPLETOS (descarta o resto final). Anchorado para frente em init — não recua na
    reanálise mesmo se a previsão truncar. Retorna [(d_ini, d_fim, campo_médio, *extras_médios), ...].
    """
    d0 = np.datetime64(pd.Timestamp(init_date).date())
    fcst = np.where(dates > d0)[0]
    if len(fcst) == 0:
        return []
    out = []
    i, n = int(fcst[0]), len(dates)
    while i + dias <= n:
        sl = slice(i, i + dias)
        out.append((dates[i], dates[i + dias - 1], chi[sl].mean(axis=0))
                   + tuple(e[sl].mean(axis=0) for e in extras))
        i += dias
    return out


def _run_once(mode: str, fcst_model, logger):
    """Núcleo do s31 para um (modo, modelo). Reanálise: fcst_model=None. Forecast: 'gefs'/'cfs'."""
    logger.info('=' * 80)
    logger.info(f'SCRIPT {SCRIPT_ID.upper()}: {SCRIPT_DESC}'
                + (f' — FORECAST {fcst_model.upper()}' if fcst_model else ''))
    logger.info('=' * 80)

    janela = int(_cfg('JANELA_MEDIA_MOVEL', 120))
    n_pentadas = int(_cfg('N_PENTADAS', 6))
    hov_dias = int(_cfg('HOVMOLLER_DIAS', 120))
    faixa = list(_cfg('FAIXA_HOVMOLLER', [-5, 5]))
    ww_extra = int(_cfg('WW_EXTRA_JANELA', 0))
    lanczos_n = int(_cfg('LANCZOS_N', 60))
    period_min = float(_cfg('LANCZOS_PERIOD_MIN', 20.0))
    period_max = float(_cfg('LANCZOS_PERIOD_MAX', 90.0))

    input_dir = Path(settings.DIR_INPUT)
    is_forecast = mode.startswith('forecast')

    if is_forecast:
        if fcst_model == 'cfs':
            # CFS: pseudo-ensemble lagged (16 = 4 ciclos × 4 membros) -> dia D = ONTEM por padrao
            # (garante os 16 ciclos publicados); usa todos os ciclos, RODADA nao se aplica. 45 dias.
            lead_days = int(_cfg('CFS_LEAD_DAYS', CFS_LEAD_DAYS))
            init = _resolve_forecast_init(_cfg('FORECAST_INIT', ''), 0, default_offset_days=-1)
        else:  # gefs (geavg, 35 dias)
            rodada = int(_cfg('RODADA', 0))
            if rodada not in (0, 6, 12, 18):
                raise ValueError(f'RODADA deve ser 00/06/12/18 (UTC). Recebido: {rodada:02d}')
            lead_days = int(_cfg('FORECAST_LEAD_DAYS', 35))
            init = _resolve_forecast_init(_cfg('FORECAST_INIT', ''), rodada)
        dt_ini = init                                   # periodo de interesse = a previsao
        dt_fim = init + timedelta(days=lead_days)
        # pentadas para frente cobrindo a previsao (P1=init+1..+5 ... ); teto = lead/5
        n_pentadas = max(1, lead_days // 5)
        # base/MODO/MODELO: s31_CHI200_INTRASAZONAL/FORECAST/<GEFS|CFS>/
        output_dir = _base_output_dir() / 'FORECAST' / fcst_model.upper()
    else:
        init = None
        lead_days = 0
        dt_ini = datetime.strptime(settings.DATA_INICIAL, '%Y-%m-%d')
        dt_fim = datetime.strptime(settings.DATA_FINAL, '%Y-%m-%d')
        # base/MODO: s31_CHI200_INTRASAZONAL/REANALISE/
        output_dir = _base_output_dir() / 'REANALISE'

    cache_params = {
        'MODE': mode,
        'DATA_INICIAL': settings.DATA_INICIAL if not is_forecast else init.strftime('%Y-%m-%d %H'),
        'DATA_FINAL': settings.DATA_FINAL if not is_forecast else dt_fim.strftime('%Y-%m-%d'),
        'forecast_model': fcst_model, 'lead_days': lead_days,
        'janela': janela, 'n_pentadas': n_pentadas, 'hov_dias': hov_dias, 'faixa': faixa,
        'lanczos_n': lanczos_n, 'period_min': period_min, 'period_max': period_max,
        'metodo': ('CPC running-mean causal + Wheeler-Weickmann (forecast a la NCICS)' if is_forecast
                   else 'CPC running-mean + Lanczos bandpass (20-90d) + u850 anom (LTM diaria)'),
        'script_version': '2.12',
    }
    if is_forecast:
        # Forecast (a la NCICS): so o sinal causal (trailing-mean) + Wheeler-Weickmann. Sem Lanczos.
        # Hovmollers numa subpasta HOVMOLLER; mapas espaciais por janela em <N>_DAY/ (Produto 1).
        hov_dir = output_dir / 'HOVMOLLER'
        periodo_dir = output_dir / 'MEDIA_PERIODO_TOTAL'
        hov_fcst_png = hov_dir / 'chi200_hovmoller_forecast.png'
        hov_wind_png = hov_dir / 'chi200_u850_hovmoller_forecast.png'
        periodo_fcst_png = periodo_dir / 'chi200_periodo_forecast.png'
        output_files = [str(hov_fcst_png), str(hov_wind_png), str(periodo_fcst_png)]
    else:
        hov_com_png = output_dir / 'chi200_hovmoller_com_filtro.png'
        hov_sem_png = output_dir / 'chi200_hovmoller_sem_filtro.png'
        hov_wind_png = output_dir / 'chi200_u850_hovmoller_com_filtro.png'
        periodo_com_png = output_dir / 'chi200_periodo_com_filtro.png'
        periodo_sem_png = output_dir / 'chi200_periodo_sem_filtro.png'
        output_files = [
            str(hov_com_png), str(hov_sem_png), str(hov_wind_png),
            str(periodo_com_png), str(periodo_sem_png),
        ]
    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('CACHE VALIDO! Pulando execucao.')
        return

    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    if is_forecast:
        hov_dir.mkdir(parents=True, exist_ok=True)
        periodo_dir.mkdir(parents=True, exist_ok=True)

    # Janela de download: precisa de `janela` dias antes do inicio da serie de interesse
    inicio_interesse = min(
        dt_ini,
        dt_fim - timedelta(days=max(hov_dias, n_pentadas * 5) - 1),
    )
    start_dl = inicio_interesse - timedelta(days=janela + 2 + ww_extra + lanczos_n)
    logger.info(f'Periodo de interesse: {dt_ini.date()} a {dt_fim.date()}')
    logger.info(
        f'Download (janela={janela}d + ww_extra={ww_extra}d + lanczos_n={lanczos_n}d): {start_dl.date()} a {dt_fim.date()}'
    )

    # ---- Download ERA5/GDAS u/v 200 ----
    # Em forecast a reanalise vai SO ate `init`; o trecho [init, init+lead] vem do GEFS.
    cutoff = (datetime.now() - timedelta(days=ERA5_LATENCY_DAYS)).replace(hour=0, minute=0, second=0, microsecond=0)
    hist_end = init if is_forecast else dt_fim
    files = []
    if start_dl < cutoff:
        logger.info('Etapa 1a: ERA5 u/v 200 hPa...')
        files += list(ensure_era5_uv200_for_period(
            start=start_dl, end=min(hist_end, cutoff - timedelta(days=1)),
            hours_utc=list(DEFAULT_SYNOPTIC_HOURS), force_redownload=False,
        ))
    if hist_end >= cutoff:
        logger.info('Etapa 1b: GDAS u/v 200mb...')
        files += list(ensure_gdas_uv200_for_period(
            start=max(start_dl, cutoff), end=hist_end, force_redownload=False,
        ))
    if is_forecast:
        logger.info('Etapa 1c: {} u/v 200 hPa (previsao, init {:%Y-%m-%d} + {}d)...',
                    fcst_model.upper(), init, lead_days)
        files += list(_FCST_DL_200[fcst_model](
            init=init, lead_hours=lead_days * 24, force_redownload=False,
        ))

    # ---- Grade alvo (LTM 2.5°, lat ascendente) ----
    dates_probe = np.array([np.datetime64(dt_fim.date())])
    _, _, ltm_lat, ltm_lon = clim_uv200_daily(dates_probe)
    order = np.argsort(ltm_lat)
    lat, lon = ltm_lat[order], ltm_lon

    # ---- Serie diaria de vento (regridada) ----
    logger.info('Etapa 2: Montando serie diaria de u/v 200 (2.5°)...')
    u_da, v_da = _daily_series_uv200(files, start_dl, dt_fim, lat, lon, logger)
    u_da, v_da = _reindex_daily(u_da), _reindex_daily(v_da)
    dates = np.array([np.datetime64(pd.Timestamp(t).date()) for t in u_da['time'].values])
    logger.info(f'Serie diaria contigua: {len(dates)} dias ({dates[0]} a {dates[-1]})')

    # ---- Anomalia diaria = vento - LTM(dia-do-ano) ----
    logger.info('Etapa 3: Anomalia diaria (- LTM diaria)...')
    u_clim, v_clim, _, _ = clim_uv200_daily(dates)
    u_clim, v_clim = u_clim[:, order, :], v_clim[:, order, :]
    u_anom = u_da.values - u_clim
    v_anom = v_da.values - v_clim

    # ---- Intrasazonal + chi200 por dia ----
    logger.info(f'Etapa 4: Filtro intrasazonal (media movel {janela}d) + Poisson por dia...')
    chi_intra, u_div_series, v_div_series, idx = chi200_intrasazonal_series(u_anom, v_anom, lat, lon, janela=janela)
    dates_intra = dates[idx]
    logger.info(f'Serie chi intrasazonal: {chi_intra.shape[0]} dias ({dates_intra[0]} a {dates_intra[-1]})')

    # ---- Filtro Wheeler-Weickmann direto no chi intrasazonal ----
    logger.info('Etapa 4b: Filtro espectral Wheeler-Weickmann (MJO + Kelvin) sobre chi...')
    ww_chi_sem = ww_filter_chi_modes(chi_intra, lat)
    logger.info('  WW: MJO k=1-9 (30-90d) + Kelvin k=1-3 tropical-mean (2.5-30d, Gauss 20°) OK')
    logger.info(
        '  WW chi max — MJO: {:.1f} | Kelvin: {:.1f} (x10^5 m^2/s) | levels: {}',
        np.nanmax(np.abs(ww_chi_sem['mjo'])) / CHI_SCALE,
        np.nanmax(np.abs(ww_chi_sem['kelvin'])) / CHI_SCALE,
        WW_LEVELS,
    )

    # versoes: (sufixo, chi, u_div, v_div, ww_chi, rotulo, cbar_label, hov_png, periodo_png)
    if is_forecast:
        # A la NCICS/Schreck: so o causal (trailing-mean) + WW com a previsao como padding.
        # NADA de Lanczos — o filtro centrado (n>lead) degradaria toda a previsao.
        versoes = [
            ('forecast', chi_intra, u_div_series, v_div_series, ww_chi_sem,
             f'CHI200 intrasazonal (previsão {fcst_model.upper()})', 'CHI200 intrasazonal (×10⁵ m²/s)',
             hov_fcst_png, periodo_fcst_png),
        ]
    else:
        # ---- Lanczos bandpass sobre chi intrasazonal ----
        logger.info(
            'Etapa 4c: Lanczos bandpass {}-{}d (n={}) sobre chi, u_div, v_div...',
            int(period_min), int(period_max), lanczos_n,
        )
        chi_com = lanczos_bandpass(chi_intra, period_min, period_max, lanczos_n)
        u_div_com = lanczos_bandpass(u_div_series, period_min, period_max, lanczos_n)
        v_div_com = lanczos_bandpass(v_div_series, period_min, period_max, lanczos_n)
        ww_chi_com = ww_filter_chi_modes(chi_com, lat)
        versoes = [
            ('com_filtro', chi_com,   u_div_com,    v_div_com,    ww_chi_com,
             'CHI200 intrasazonal', 'CHI200 intrasazonal (×10⁵ m²/s)', hov_com_png, periodo_com_png),
            ('sem_filtro', chi_intra, u_div_series, v_div_series, ww_chi_sem,
             'CHI200',               'CHI200 (×10⁵ m²/s)',               hov_sem_png, periodo_sem_png),
        ]

    # ---- Produto 1: mapas espaciais ----
    if is_forecast:
        # Estilo NCICS "Select Days": médias de 1/2/3/5/7/10 dias, janelas consecutivas a partir
        # de init, cada tamanho em sua subpasta <N>_DAY/. (Só o causal — versoes tem 1 entrada.)
        windows = [int(w) for w in _cfg('FORECAST_MAP_WINDOWS', list(FORECAST_MAP_WINDOWS))]
        logger.info('Etapa 5: Mapas espaciais por janela (forecast): {} dias...', windows)
        for antigo in output_dir.glob('chi200_*.png'):  # limpa mapas antigos na raiz do modelo
            antigo.unlink()
        sufixo, chi_v, ud_v, vd_v, ww_v, rotulo, cbar_label = versoes[0][:7]
        for w in windows:
            wdir = output_dir / f'{w}_DAY'
            wdir.mkdir(parents=True, exist_ok=True)
            for antigo in wdir.glob('chi200_*.png'):
                antigo.unlink()
            blocos = _forecast_windows(chi_v, dates_intra, init, w, ud_v, vd_v, ww_v['mjo'], ww_v['kelvin'])
            for d_ini, d_fim, campo, ud, vd, chi_mjo_b, chi_kel_b in blocos:
                chi_ww = {'mjo': chi_mjo_b, 'kelvin': chi_kel_b}
                _plot_mapa(
                    campo, lat, lon, f'{rotulo} — média {w}d: {d_ini} a {d_fim}',
                    wdir / f'chi200_{w}day_{d_ini}_a_{d_fim}.png', input_dir,
                    u_div=ud, v_div=vd, ww_chi=chi_ww, cbar_label=cbar_label,
                )
            logger.info('  {}_DAY: {} mapas', w, len(blocos))
    else:
        logger.info('Etapa 5: Mapas de pentada (com e sem filtro Lanczos)...')
        for antigo in output_dir.glob('chi200_*pentada*.png'):
            antigo.unlink()
        for sufixo, chi_v, ud_v, vd_v, ww_v, rotulo, cbar_label, _, _ in versoes:
            pentadas = agrupa_pentadas(chi_v, dates_intra, n_pentadas, ud_v, vd_v, ww_v['mjo'], ww_v['kelvin'])
            for d_ini, d_fim, campo, ud, vd, chi_mjo_pent, chi_kel_pent in pentadas:
                chi_ww = {'mjo': chi_mjo_pent, 'kelvin': chi_kel_pent}
                nome = f'chi200_pentada_{sufixo}_{d_ini}_a_{d_fim}.png'
                _plot_mapa(
                    campo, lat, lon, f'{rotulo} — pentada {d_ini} a {d_fim}',
                    output_dir / nome, input_dir, u_div=ud, v_div=vd, ww_chi=chi_ww,
                    cbar_label=cbar_label,
                )

    # ---- Produto 2: Hovmoller (com e sem filtro) ----
    logger.info('Etapa 6: Hovmoller (com e sem filtro Lanczos)...')
    m_hov = dates_intra >= (np.datetime64(dt_fim.date()) - np.timedelta64(hov_dias - 1, 'D'))
    for sufixo, chi_v, _, _, _, rotulo, cbar_label, hov_png_v, _ in versoes:
        hov = media_faixa_latitude(chi_v[m_hov], lat, faixa[0], faixa[1])
        _plot_hovmoller(
            hov, lon, dates_intra[m_hov],
            f'{rotulo} — Hovmöller ({faixa[0]}° a {faixa[1]}°)',
            hov_png_v, input_dir, cbar_label=cbar_label, init_date=init,
        )

    # ---- Produto 3: mapa do periodo (com e sem filtro) ----
    logger.info('Etapa 7: Mapa do periodo (com e sem filtro Lanczos)...')
    m_per = (dates_intra >= np.datetime64(dt_ini.date())) & (dates_intra <= np.datetime64(dt_fim.date()))
    if m_per.any():
        for sufixo, chi_v, ud_v, vd_v, ww_v, rotulo, cbar_label, _, periodo_png_v in versoes:
            chi_ww_per = {
                'mjo':    ww_v['mjo'][m_per].mean(axis=0),
                'kelvin': ww_v['kelvin'][m_per].mean(axis=0),
            }
            _plot_mapa(
                chi_v[m_per].mean(axis=0), lat, lon,
                f'{rotulo} — media {dt_ini.date()} a {dt_fim.date()}',
                periodo_png_v, input_dir,
                u_div=ud_v[m_per].mean(axis=0), v_div=vd_v[m_per].mean(axis=0),
                ww_chi=chi_ww_per, cbar_label=cbar_label,
            )

    # ---- Etapa 8: vento zonal 850 hPa intrasazonal (ERA5 + GDAS [+ GEFS no forecast]) ----
    logger.info('Etapa 8: Download u/v 850 hPa (ERA5 + GDAS)...')
    files850 = []
    if start_dl < cutoff:
        logger.info('Etapa 8a: ERA5 u/v 850 hPa...')
        files850 += list(ensure_era5_uv850_for_period(
            start=start_dl, end=min(hist_end, cutoff - timedelta(days=1)),
            hours_utc=list(DEFAULT_SYNOPTIC_HOURS), force_redownload=False,
        ))
    if hist_end >= cutoff:
        logger.info('Etapa 8b: GDAS u/v 850mb...')
        files850 += list(ensure_gdas_uv850_for_period(
            start=max(start_dl, cutoff), end=hist_end, force_redownload=False,
        ))
    if is_forecast:
        logger.info('Etapa 8c: {} u/v 850 hPa (previsao)...', fcst_model.upper())
        files850 += list(_FCST_DL_850[fcst_model](
            init=init, lead_hours=lead_days * 24, force_redownload=False,
        ))

    logger.info('Etapa 8d: Serie diaria u850 (grade 2.5°) + anomalia (LTM diaria NCEP)...')
    u850_da, _ = _daily_series_uv200(files850, start_dl, dt_fim, lat, lon, logger)
    u850_da = _reindex_daily(u850_da)
    dates_w = np.array([np.datetime64(pd.Timestamp(t).date()) for t in u850_da['time'].values])

    # Anomalia diaria u850 = u850 - LTM_diaria(dia-do-ano), MESMA base/fonte/grade da
    # LTM de u200 (NCEP 1991-2020, 2.5°) -> anomalia consistente com a do chi200.
    u850_clim, clat850, _ = clim_u850_daily(dates_w)
    u850_clim = u850_clim[:, np.argsort(clat850), :]  # lat ascendente, alinhada com `lat`
    u850_anom = u850_da.values - u850_clim

    # Forecast: so o sinal causal (sem Lanczos) — coerente com o chi (a la NCICS).
    logger.info('Etapa 8e: Filtro intrasazonal u850 (running mean{})...',
                '' if is_forecast else ' + Lanczos')
    u850_com, idx_w = _u850_intrasazonal(
        u850_anom, janela, lanczos_n, period_min, period_max, aplicar_lanczos=not is_forecast)
    dates_u850 = dates_w[idx_w]
    logger.info(f'Serie u850 intrasazonal: {u850_com.shape[0]} dias ({dates_u850[0]} a {dates_u850[-1]})')

    # ---- Etapa 9: Hovmoller CHI200 + vento zonal 850 hPa intrasazonal ----
    logger.info('Etapa 9: Hovmoller CHI200 + vento zonal 850 hPa intrasazonal...')
    chi_for_hov = chi_intra if is_forecast else chi_com  # forecast nao tem versao Lanczos
    janela_hov = np.datetime64(dt_fim.date()) - np.timedelta64(hov_dias - 1, 'D')
    m_hov_chi = dates_intra >= janela_hov
    m_hov_w = dates_u850 >= janela_hov
    # Alinha datas entre chi (u200) e u850 (intersecao — podem diferir por dias faltantes)
    d_hov = np.intersect1d(dates_intra[m_hov_chi], dates_u850[m_hov_w])
    if d_hov.size == 0:
        raise ValueError('Sem datas em comum entre CHI200 e u850 para o Hovmoller com vento.')
    idx_c = np.isin(dates_intra, d_hov)
    idx_u = np.isin(dates_u850, d_hov)
    chi_hov = media_faixa_latitude(chi_for_hov[idx_c], lat, faixa[0], faixa[1])
    u_hov = media_faixa_latitude(u850_com[idx_u], lat, faixa[0], faixa[1])
    _titulo_hw = (f'CHI200 intrasazonal + vento zonal 850 hPa (previsão {fcst_model.upper()})' if is_forecast
                  else 'CHI200 intrasazonal + vento zonal 850 hPa')
    _plot_hovmoller_chi_wind(
        chi_hov, u_hov, lon, d_hov,
        f'{_titulo_hw} — Hovmöller ({faixa[0]}° a {faixa[1]}°)',
        hov_wind_png, input_dir, init_date=init,
    )

    execution_time = time.time() - start_time
    save_cache_metadata(SCRIPT_ID, cache_params, output_files, execution_time)
    logger.info('=' * 80)
    logger.info(f'Script {SCRIPT_ID.upper()} concluido em {execution_time:.1f}s')
    logger.info(f'Saida: {output_dir}')
    logger.info('=' * 80)


def main():
    """Entry point. Reanálise: 1 execução. Forecast: laço pelos modelos habilitados (GEFS + CFS)."""
    logger = get_logger(SCRIPT_ID)
    mode = str(_cfg('MODE', 'reanalysis')).strip().lower()
    if mode.startswith('forecast'):
        models = _enabled_forecast_models()
        logger.info('s31 FORECAST — modelos habilitados: {}', [m.upper() for m in models])
        for model in models:
            _run_once('forecast', model, logger)
    else:
        _run_once('reanalysis', None, logger)


if __name__ == '__main__':
    main()
