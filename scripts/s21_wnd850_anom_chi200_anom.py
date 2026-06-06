# -*- coding: utf-8 -*-
"""
s21 - Anomalia CHI200 (shaded) + Linhas de corrente vento 850 hPa + Vento divergente 200 hPa.

Ordem de sobreposição:
  1. CHI200 shaded (anomalia de função de velocidade)
  2. Linhas de corrente do vento anômalo 850 hPa (dimgray)
  3. Vento divergente 200 hPa (chi200 < 0, faixa -20° a 20°)

Pipeline de dados:
  - CHI200: ERA5/GDAS + climatologia PSL/NOAA (via plot_chi200)
  - Vento 850 hPa: ERA5/GDAS + climatologia PSL/NOAA (via plot_olr_wind850_anom)

Saida:
    - Mapas PNG em {settings.DIR_OUTPUT}/s21_WND850_ANOM_CHI200_ANOM/

Criado em: 2026-06-06
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')

import gc
import time
from datetime import datetime
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.util import add_cyclic_point
from matplotlib import patches
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import FixedLocator, MultipleLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from PIL import Image

from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.common.dataset_utils import area_display_name, load_dataset
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.src.uteis.plot_chi200 import main as plot_chi200
from app.src.uteis.plot_olr_wind850_anom import main as plot_wind850_anom

# ---------------------------------------------------------------------------
# Workaround: cartopy 0.25 + matplotlib 3.10 — GeometryCollection não subscritável.
# ---------------------------------------------------------------------------
import matplotlib.path as _mpath
import numpy as _np
from cartopy.mpl.geoaxes import InterProjectionTransform as _IPT

_orig_transform_path = _IPT.transform_path_non_affine


def _safe_transform_path(self, path):
    try:
        return _orig_transform_path(self, path)
    except TypeError:
        return _mpath.Path(_np.empty((0, 2)))


_IPT.transform_path_non_affine = _safe_transform_path

# ---------------------------------------------------------------------------
# Identidade do script
# ---------------------------------------------------------------------------
SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's21'
SCRIPT_NAME = Path(__file__).stem
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME

# ---------------------------------------------------------------------------
# Constantes — CHI200
# ---------------------------------------------------------------------------
CHI_SCALE = 1e5
CHI_FILE_NAME = 'chi200.nc'

LEVELS = np.arange(-60, 65, 5)

CHI_CANDIDATES = (
    'chi_anom_mean_scaled',
    'chi_anom_mean',
    'chi',
    'chi_anom',
    'velocity_potential_anomaly',
)
UCHI_CANDIDATES = ('uchi_anom_mean', 'uchi', 'uchi_anom', 'udiv', 'uchi_irrot')
VCHI_CANDIDATES = ('vchi_anom_mean', 'vchi', 'vchi_anom', 'vdiv', 'vchi_irrot')

CHI200_COLORS = [
    '#005a45', '#0f7a6c', '#2e9b96', '#62bdb7', '#9dd8d2', '#dff3f1',
    '#f7f4eb', '#e7d9a9', '#d6b566', '#bd8a35', '#9a6313', '#6f4300',
]

# ---------------------------------------------------------------------------
# Constantes — vento divergente (quiver tropical, chi200 < 0)
# ---------------------------------------------------------------------------
QUIVER_STEP = 4
QUIVER_SCALE = 80
QUIVER_WIDTH = 0.002
QUIVER_MIN_MAG = 0.3
LAT_DIV_MIN = -20.0
LAT_DIV_MAX = 20.0

# ---------------------------------------------------------------------------
# Constantes — linhas de corrente vento 850 hPa
# ---------------------------------------------------------------------------
WIND850_FILE_NAME = 'wind850_anom.nc'
STREAMPLOT_ARROWSIZE = 0.8

_STREAMPLOT_DENSITY: dict[str, float] = {
    'globo': 2.5,
    'globo_3d': 2.5,
    'tropico': 2.5,
    'hemisferio_sul': 2.5,
    'psa': 2.5,
    'mjo': 2.5,
    'enso': 2.5,
    'america_sul_zom_out': 2.5,
    'pacifico_leste_america_sul': 2.5,
    'pacific_chile': 2.0,
    'america_sul': 2.0,
    'africa': 2.0,
    'africa_monsoon': 2.0,
    'atlantico_tropical': 3.0,
    'amo': 3.0,
    'sad': 2.0,
    'iod': 2.0,
    'pdo': 2.0,
    'tna': 2.0,
    'tsa': 2.0,
    'MDR': 2.0,
    'china': 2.0,
    'estados_unidos': 2.0,
    'brasil': 2.5,
    'estados_unidos_zoom': 2.5,
    'costa_brasil': 3.0,
    'argentina': 3.0,
    'zona_zcit_atlantico': 3.0,
}
_STREAMPLOT_DENSITY_DEFAULT = 2.0

_STREAMPLOT_LINEWIDTH: dict[str, float] = {
    'globo': 0.5,
    'globo_3d': 0.5,
    'tropico': 0.5,
    'hemisferio_sul': 0.5,
    'psa': 0.5,
    'mjo': 0.5,
    'enso': 0.5,
    'america_sul_zom_out': 0.5,
    'pacifico_leste_america_sul': 0.5,
    'pacific_chile': 0.5,
    'america_sul': 0.5,
    'africa': 0.5,
    'africa_monsoon': 0.5,
    'atlantico_tropical': 0.5,
    'amo': 0.5,
    'sad': 0.5,
    'iod': 0.5,
    'pdo': 0.5,
    'tna': 0.5,
    'tsa': 0.5,
    'MDR': 0.5,
    'china': 0.5,
    'estados_unidos': 0.5,
    'brasil': 0.5,
    'estados_unidos_zoom': 0.5,
    'costa_brasil': 0.5,
    'argentina': 0.5,
    'zona_zcit_atlantico': 0.5,
}
_STREAMPLOT_LINEWIDTH_DEFAULT = 0.5

# ---------------------------------------------------------------------------
# Áreas padrão (mesmas do s20)
# ---------------------------------------------------------------------------
DEFAULT_AREAS = [
    'globo_3d', 'pacific_chile', 'china', 'pacifico_leste_america_sul',
    'america_sul_zom_out', 'MDR', 'tropico', 'zona_zcit_atlantico',
    'brasil', 'america_sul', 'africa_monsoon', 'africa', 'mjo', 'amo',
    'sad', 'iod', 'pdo', 'tna', 'tsa', 'atlantico_tropical', 'enso',
    'globo', 'costa_brasil', 'psa', 'argentina', 'estados_unidos_zoom',
    'estados_unidos', 'hemisferio_sul',
]


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------
def _pick_first_var(ds, candidates, *, required=True):
    for name in candidates:
        if name in ds.data_vars:
            return ds[name], name
    if required:
        raise KeyError(
            f'Nenhuma das variáveis {candidates} encontrada. '
            f'Disponíveis: {list(ds.data_vars)}'
        )
    return None, None


def _standardize_coords(da):
    ren = {}
    if 'latitude' in da.coords:
        ren['latitude'] = 'lat'
    if 'longitude' in da.coords:
        ren['longitude'] = 'lon'
    if ren:
        da = da.rename(ren)
    if 'lat' not in da.coords or 'lon' not in da.coords:
        raise ValueError("Campo sem coordenadas 'lat' e 'lon'.")
    for dim in list(da.dims):
        if dim not in {'lat', 'lon'} and da.sizes[dim] == 1:
            da = da.isel({dim: 0}, drop=True)
    extra_dims = [d for d in da.dims if d not in {'lat', 'lon'}]
    for dim in extra_dims:
        da = da.isel({dim: 0}, drop=True)
    if da.dims != ('lat', 'lon'):
        da = da.transpose('lat', 'lon')
    if float(da['lon'].min()) < 0:
        da = da.assign_coords(lon=((da['lon'] + 360) % 360))
        da = da.sortby('lon')
    if da['lat'][0] < da['lat'][-1]:
        da = da.sortby('lat', ascending=False)
    return da


def _add_cyclic_2d(da2d):
    data_cyc, lon_cyc = add_cyclic_point(da2d.values, coord=da2d['lon'].values)
    return data_cyc, lon_cyc


def _add_cyclic_uv(u2d, v2d):
    u_cyc, lon_cyc = add_cyclic_point(u2d.values, coord=u2d['lon'].values)
    v_cyc, _ = add_cyclic_point(v2d.values, coord=v2d['lon'].values)
    return u_cyc, v_cyc, lon_cyc


def _build_chi_levels_norm():
    n_bins = len(LEVELS) - 1
    cmap = LinearSegmentedColormap.from_list('chi200_green_brown', CHI200_COLORS, N=n_bins)
    norm = BoundaryNorm(LEVELS, ncolors=n_bins, clip=False)
    ticks = np.arange(-60, 65, 10)
    return LEVELS, ticks, cmap, norm


def _get_area_list() -> list[str]:
    for attr in ('LST_AREAS_S21', 'LST_AREAS_S20'):
        if hasattr(settings, attr):
            return list(getattr(settings, attr))
    return list(DEFAULT_AREAS)


def _to_str_date(val) -> str:
    return val.strftime('%Y-%m-%d') if hasattr(val, 'strftime') else str(val)


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

    if area in {'hemisferio_sul', 'psa', 'globo', 'mjo'}:
        gl.xlocator = MultipleLocator(40)
        gl.ylocator = MultipleLocator(20)
        if area in {'globo', 'mjo'}:
            gl.xlabel_style = {'size': 15, 'color': 'black'}
            gl.ylabel_style = {'size': 15, 'color': 'black'}
    elif area == 'enso':
        gl.xlocator = FixedLocator([
            -160, -140, -120, -100, -80, -60, 0, 20, 40, 60, 80,
            100, 120, 140, 150, 160, 170, 180,
        ])
        gl.ylocator = MultipleLocator(10)
        gl.xlabel_style = {'size': 15, 'color': 'black'}
        gl.ylabel_style = {'size': 15, 'color': 'black'}
    elif area == 'tropico':
        gl.xlocator = FixedLocator([-160, -120, -80, -40, 0, 40, 80, 120, 160])
        gl.ylocator = MultipleLocator(20)
        gl.xlabel_style = {'size': 15, 'color': 'black'}
        gl.ylabel_style = {'size': 15, 'color': 'black'}
    elif area == 'estados_unidos':
        gl.xlocator = MultipleLocator(10)
        gl.ylocator = MultipleLocator(10)
    elif area == 'estados_unidos_zoom':
        gl.xlocator = MultipleLocator(10)
        gl.ylocator = MultipleLocator(5)
    elif area in {'argentina', 'costa_brasil'}:
        gl.xlocator = MultipleLocator(5)
        gl.ylocator = MultipleLocator(5)
    elif area == 'brasil':
        gl.xlocator = MultipleLocator(10)
        gl.ylocator = MultipleLocator(5)
    elif area in {'america_sul', 'africa'}:
        gl.xlocator = MultipleLocator(20)
        gl.ylocator = MultipleLocator(20)
    elif area == 'zona_zcit_atlantico':
        gl.xlocator = MultipleLocator(10)
        gl.ylocator = MultipleLocator(5)
    elif area in {'tsa', 'tna'}:
        gl.xlocator = MultipleLocator(10)
        gl.ylocator = MultipleLocator(5)
    elif area in {'atlantico_tropical', 'pdo', 'iod', 'sad', 'amo', 'africa_monsoon'}:
        gl.xlocator = MultipleLocator(20)
        gl.ylocator = MultipleLocator(10)
    elif area in {
        'pacific_chile', 'pacifico_leste_america_sul', 'america_sul_zom_out',
        'china', 'MDR',
    }:
        gl.xlocator = MultipleLocator(20)
        gl.ylocator = MultipleLocator(10)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger = get_logger(SCRIPT_ID)

    logger.info('=' * 80)
    logger.info(f'SCRIPT {SCRIPT_ID.upper()}: {SCRIPT_DESC}')
    logger.info('=' * 80)

    lst_areas = _get_area_list()
    output_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_WND850_ANOM_CHI200_ANOM'
    input_dir = Path(settings.DIR_INPUT)
    dados_dir = Path(settings.DIR_DADOS)

    cache_params = {
        'DATA_INICIAL': settings.DATA_INICIAL,
        'DATA_FINAL': settings.DATA_FINAL,
        'areas': lst_areas,
        'script_version': '1.5',
        'chi_file': CHI_FILE_NAME,
        'wind_file': WIND850_FILE_NAME,
    }
    output_files = [
        str(output_dir / f'wnd850_anom_chi200_anom_{area}.png') for area in lst_areas
    ]

    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('CACHE VALIDO! Execucao ja foi realizada com os mesmos parametros.')
        logger.info(f'   Periodo: {settings.DATA_INICIAL} a {settings.DATA_FINAL}')
        logger.info(f'   {len(output_files)} mapas ja existem')
        logger.info(f'   Diretorio: {output_dir}')
        logger.info('   Pulando execucao')
        return

    start_time = time.time()
    logger.info(f'Periodo de analise: {settings.DATA_INICIAL} a {settings.DATA_FINAL}')
    logger.info(f'Gerando {len(lst_areas)} mapas — CHI200 + streamlines 850 hPa + vento divergente')
    logger.info('=' * 80)

    # ---- Etapa 1: CHI200 ----
    logger.info('Etapa 1: Calculando CHI200 (ERA5/GDAS + PSL)...')
    plot_chi200()
    gc.collect()

    # ---- Etapa 2: vento 850 hPa ----
    logger.info('Etapa 2: Calculando anomalia de vento 850 hPa (ERA5/GDAS + PSL)...')
    plot_wind850_anom()
    gc.collect()

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Carregar chi200.nc ----
    chi_file = dados_dir / CHI_FILE_NAME
    if not chi_file.exists():
        raise FileNotFoundError(f'Arquivo nao encontrado: {chi_file}')
    ds_chi = load_dataset(str(chi_file))

    da_chi, chi_varname = _pick_first_var(ds_chi, CHI_CANDIDATES)
    da_chi = _standardize_coords(da_chi)
    if chi_varname != 'chi_anom_mean_scaled':
        da_chi = da_chi / CHI_SCALE

    da_uchi = _standardize_coords(_pick_first_var(ds_chi, UCHI_CANDIDATES)[0])
    da_vchi = _standardize_coords(_pick_first_var(ds_chi, VCHI_CANDIDATES)[0])

    lat_chi = da_chi['lat'].values
    chi_cyc, lon_cyc = _add_cyclic_2d(da_chi)
    uchi_cyc, vchi_cyc, _ = _add_cyclic_uv(da_uchi, da_vchi)

    # ---- Carregar wind850_anom.nc ----
    wind_file = dados_dir / WIND850_FILE_NAME
    if not wind_file.exists():
        raise FileNotFoundError(f'Arquivo nao encontrado: {wind_file}')
    ds_wind = xr.open_dataset(wind_file)
    u_wind = ds_wind['u_anom_mean'].values
    v_wind = ds_wind['v_anom_mean'].values
    lat_wind = ds_wind['lat'].values
    lon_wind = ds_wind['lon'].values
    ds_wind.close()

    # streamplot requer lat ascendente
    if lat_wind[0] > lat_wind[-1]:
        lat_wind = lat_wind[::-1]
        u_wind = u_wind[::-1, :]
        v_wind = v_wind[::-1, :]

    # ponto cíclico evita descontinuidade do streamplot em 180°
    u_wind, lon_wind = add_cyclic_point(u_wind, coord=lon_wind)
    v_wind = add_cyclic_point(v_wind)

    levels, ticks, cmap, norm = _build_chi_levels_norm()
    info_plot = settings['areas_plotagem']

    dt_ini = datetime.strptime(_to_str_date(settings.DATA_INICIAL), '%Y-%m-%d').strftime('%d-%m-%y')
    dt_fim = datetime.strptime(_to_str_date(settings.DATA_FINAL), '%Y-%m-%d').strftime('%d-%m-%y')

    logo_path = (
        None if settings.get('SEM_LOGO', False)
        else input_dir / ('logo_grec.png' if settings.get('LOGO_GREC', False) else 'novo_logo.png')
    )

    for area in lst_areas:
        logger.info(f'Gerando mapa CHI200+streamlines 850+div para area: {area_display_name(area)}')

        is_polar = info_plot[area].get('projection', '') == 'orthographic_south'
        if is_polar:
            proj = ccrs.Orthographic(
                central_longitude=settings.get('ORTHO_CENTRAL_LONGITUDE', info_plot[area].get('ortho_central_longitude', -71)),
                central_latitude=settings.get('ORTHO_CENTRAL_LATITUDE', info_plot[area].get('ortho_central_latitude', -84)),
            )
        else:
            proj = ccrs.PlateCarree(central_longitude=info_plot[area]['central_longitude_mapa'])

        data_transform = ccrs.PlateCarree(
            central_longitude=info_plot[area]['central_longitude_plot']
        )

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(1, 1, 1, projection=proj)

        if is_polar:
            theta = np.linspace(0, 2 * np.pi, 100)
            center, radius = [0.5, 0.5], 0.5
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * radius + center)
            ax.set_boundary(circle, transform=ax.transAxes)

        if info_plot[area].get('plot_box', False):
            for box in info_plot[area]['lst_boxes']:
                ax.add_patch(patches.Rectangle(
                    (box['x_anc'], box['y_anc']),
                    box['x_larg'], box['y_larg'],
                    linewidth=box['linewidth'], edgecolor=box['edgecolor'],
                    facecolor='none', zorder=300,
                ))

        if area == 'MDR':
            ax.plot(
                [-86, -20, -20, -86, -86], [10, 10, 20, 20, 10],
                color='black', linewidth=3, linestyle='-', zorder=500,
                transform=ccrs.PlateCarree(),
            )

        if area == 'atlantico_tropical':
            legenda_atl = input_dir / 'legenda_atlantic.png'
            if legenda_atl.exists():
                fig.figimage(plt.imread(str(legenda_atl)), 125, 614, zorder=3, alpha=1)
            ax.add_patch(patches.Rectangle(
                (10, -20), -40, 20, linewidth=3, edgecolor='black', facecolor='none', zorder=300,
            ))
            ax.add_patch(patches.Rectangle(
                (-15, 5), -40, 20, linewidth=3, edgecolor='blue', facecolor='none', zorder=300,
            ))

        if area == 'iod':
            ax.add_patch(patches.Rectangle(
                (50, -10), 20, 20, linewidth=3, edgecolor='black', facecolor='none', zorder=300,
            ))
            ax.add_patch(patches.Rectangle(
                (90, -10), 20, 10, linewidth=3, edgecolor='black', facecolor='none', zorder=300,
            ))

        if area == 'enso':
            for txt, x, y, cor in [
                ('Niño 1+2', 66.25, -13.64, 'red'),
                ('Niño 3', 34.1, 8.45, 'blue'),
                ('Niño 3.4', 8.6, -9.45, 'black'),
                ('Niño 4', -22.5, 8.45, 'm'),
            ]:
                t = plt.text(x, y, txt, fontsize=14, color=cor, weight='bold', zorder=500)
                fg = 'black' if cor in {'red', 'm'} else 'white'
                t.set_path_effects([
                    path_effects.Stroke(linewidth=3, foreground=fg),
                    path_effects.Normal(),
                ])

        if area == 'mjo':
            for x, txt in [(-135.25, '1'), (-115.25, '2'), (-95.25, '3'), (-75.25, '4'),
                            (-55.25, '5'), (-35.25, '6'), (-15.25, '7'), (4.75, '8')]:
                t = plt.text(x, -4.64, txt, fontsize=50, color='white', weight='bold', zorder=400)
                t.set_path_effects([
                    path_effects.Stroke(linewidth=3, foreground='black'),
                    path_effects.Normal(),
                ])

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
            ax.set_xlim([info_plot[area]['lon_esq'], info_plot[area]['lon_dir']])
            ax.set_ylim([info_plot[area]['lat_inf'], info_plot[area]['lat_sup']])

        # Features cartográficas
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, edgecolor='black')
        ax.add_feature(cfeature.LAND.with_scale('50m'), linewidth=0.5, facecolor='whitesmoke')
        if area != 'china':
            ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1.2, edgecolor='black', zorder=100)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, edgecolor='black', zorder=100)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, edgecolor='black', zorder=100)
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), linewidth=0.5, facecolor='white')

        # --- Camada 1: CHI200 shaded (zorder=2) ---
        cf = ax.contourf(
            lon_cyc, lat_chi, chi_cyc,
            levels=levels, cmap=cmap, norm=norm, extend='both',
            transform=data_transform, zorder=2,
        )

        # --- Camada 2: linhas de corrente vento 850 hPa (zorder=3) ---
        density = _STREAMPLOT_DENSITY.get(area, _STREAMPLOT_DENSITY_DEFAULT)
        lw = _STREAMPLOT_LINEWIDTH.get(area, _STREAMPLOT_LINEWIDTH_DEFAULT)
        ax.streamplot(
            lon_wind, lat_wind, u_wind, v_wind,
            color='black',
            linewidth=lw,
            density=density,
            arrowsize=STREAMPLOT_ARROWSIZE,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

        # --- Camada 3: vento divergente — chi200 < 0, faixa -20° a 20° (zorder=5) ---
        lon_q = lon_cyc[::QUIVER_STEP]
        lat_q = lat_chi[::QUIVER_STEP]
        u_q = uchi_cyc[::QUIVER_STEP, ::QUIVER_STEP]
        v_q = vchi_cyc[::QUIVER_STEP, ::QUIVER_STEP]
        chi_q = chi_cyc[::QUIVER_STEP, ::QUIVER_STEP]
        lat_outside = (lat_q < LAT_DIV_MIN) | (lat_q > LAT_DIV_MAX)
        combined_mask = (
            np.broadcast_to(lat_outside[:, None], u_q.shape)
            | (chi_q >= 0)
            | (np.sqrt(u_q**2 + v_q**2) < QUIVER_MIN_MAG)
        )
        q = ax.quiver(
            lon_q, lat_q,
            np.ma.masked_where(combined_mask, u_q),
            np.ma.masked_where(combined_mask, v_q),
            transform=ccrs.PlateCarree(),
            color='white', pivot='mid',
            scale=QUIVER_SCALE, width=QUIVER_WIDTH,
            headwidth=3.2, headlength=4.2, headaxislength=3.8,
            zorder=200,
        )
        q.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='black')])

        # Colorbar CHI200
        if is_polar and area != 'globo_3d':
            cbar = plt.colorbar(cf, ax=ax, pad=0.05, fraction=0.04, ticks=ticks)
            cbar.set_label(label=r'10$^5$ m$^2$ s$^{-1}$', size=10)
            cbar.ax.tick_params(labelsize=10)
        elif area in {'enso', 'tropico', 'MDR', 'hemisferio_sul', 'psa'}:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)
            cbar = plt.colorbar(
                cf, cax=cax, pad=0.02, fraction=0.02375,
                location='bottom', extend='both', orientation='horizontal',
                ticks=ticks, boundaries=levels, spacing='proportional',
            )
            cbar.set_label(label=r'10$^5$ m$^2$ s$^{-1}$', size=18)
            cbar.ax.tick_params(labelsize=20)
        else:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
            cbar = plt.colorbar(
                cf, cax=cax, pad=0.02, fraction=0.02375,
                extend='both', ticks=ticks, boundaries=levels, spacing='proportional',
            )
            cbar.set_label(label=r'10$^5$ m$^2$ s$^{-1}$', size=18)
            cbar.ax.tick_params(labelsize=20)

        # Título
        titulo = f'Anom. CHI200 | Streamlines 850hPa | Vento Div. (chi<0, ±20°) (De {dt_ini} a {dt_fim})'
        ax.set_title(titulo, fontsize=13 if is_polar else 16, loc='left')

        if area != 'globo_3d':
            ax.add_patch(patches.Rectangle(
                (0, 0), 1, 1,
                linewidth=0.5, edgecolor='black', facecolor='none',
                transform=ax.transAxes, zorder=1000, clip_on=False,
            ))

        if logo_path is not None and logo_path.exists():
            _add_logo_to_map(ax=ax, logo_path=logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=500)

        filename_fig = output_dir / f'wnd850_anom_chi200_anom_{area}.png'
        logger.info(f'Salvando a figura {filename_fig}')
        plt.savefig(str(filename_fig), dpi=fig.dpi, bbox_inches='tight')
        plt.close('all')

    execution_time = time.time() - start_time
    save_cache_metadata(SCRIPT_ID, cache_params, output_files, execution_time)
    logger.info('=' * 80)
    logger.info(f'Script {SCRIPT_ID.upper()} concluido com sucesso!')
    logger.info(f'Tempo de execucao: {execution_time:.1f}s ({execution_time / 60:.1f} min)')
    logger.info(f'{len(output_files)} mapas gerados em: {output_dir}')
    logger.info('=' * 80)


if __name__ == '__main__':
    main()
