# -*- coding: utf-8 -*-
"""
s23 - Media de TSM + Correntes Marinhas de Superficie.

Baixa dados de TSM media diaria do PSL/NOAA (OISSTv2 High-Res) e correntes
marinhas de superficie do CMEMS (GLOBAL_ANALYSISFORECAST_PHY_001_024),
calcula a media do periodo e gera mapas com TSM em shaded e correntes
como linhas de corrente (streamlines).

Dados de entrada:
    - PSL/NOAA: sst.day.mean.{ano}.nc (OISSTv2 0.25 grau, um arquivo por ano)
    - CMEMS: cmems_mod_glo_phy_cur_anfc_0.083deg_P1D-m (correntes diarias 1/12 grau)

Saida:
    - Mapas PNG em {settings.DIR_OUTPUT}/s23_SST_CORRENTES/

Criado em: 2026-06-06
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')

import gc
import json
import time
from datetime import datetime
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import copernicusmarine
import matplotlib.path as mpath
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.util import add_cyclic_point
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import FixedLocator, MultipleLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from PIL import Image

from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.common.dataset_utils import area_display_name, arquivo_cobre_periodo, load_dataset, validar_cobertura_temporal
from app.common.download_helper import DownloadEngine, download_with_progress
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.common.logo_helper import resolve_logo_path
from app.common.logo_helper import proportional_logo_zoom

# ---------------------------------------------------------------------------
# Identidade do script
# ---------------------------------------------------------------------------
SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's23'
SCRIPT_NAME = Path(__file__).stem
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME

# ---------------------------------------------------------------------------
# Constantes — SST
# ---------------------------------------------------------------------------
SST_URL_TEMPLATE = (
    'https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.day.mean.{year}.nc'
)
SST_FILE_TEMPLATE = 'sst.day.mean.{year}.nc'

SST_MEAN_LEVELS = [round(i * 0.5, 1) for i in range(65)]  # 0–32°C em 0.5°C
SST_MEAN_TICKS = list(range(0, 33, 2))
SST_MEAN_COLORS = [
    'white', 'blueviolet', 'blue', 'cyan', 'limegreen', 'greenyellow',
    'yellow', 'gold', 'orange', 'darkorange', 'orangered', 'red',
    'darkred', 'crimson', 'magenta', 'white',
]

# ---------------------------------------------------------------------------
# Constantes — Correntes marinhas CMEMS
# ---------------------------------------------------------------------------
CMEMS_DATASET_ID = 'cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m'
CMEMS_CURRENTS_FILE = 'ocean_currents_surface.nc'
CMEMS_DEPTH_MIN = 0.0
CMEMS_DEPTH_MAX = 1.0  # superficie (~0.49 m)

_STREAMPLOT_DENSITY: dict[str, float] = {
    'globo': 4.0,
    'globo_3d': 4.0,
    'tropico': 4.0,
    'hemisferio_sul': 4.0,
    'psa': 4.0,
    'mjo': 4.0,
    'enso': 4.0,
    'america_sul_zom_out': 4.0,
    'pacifico_leste_america_sul': 4.0,
    'pacific_chile': 3.0,
    'america_sul': 3.0,
    'africa': 3.0,
    'africa_monsoon': 3.0,
    'atlantico_tropical': 4.0,
    'amo': 4.0,
    'sad': 3.0,
    'iod': 3.0,
    'pdo': 3.0,
    'tna': 3.0,
    'tsa': 3.0,
    'MDR': 3.0,
    'china': 3.0,
    'estados_unidos': 3.0,
    'brasil': 4.0,
    'estados_unidos_zoom': 4.0,
    'costa_brasil': 5.0,
    'argentina': 5.0,
    'zona_zcit_atlantico': 5.0,
}
_STREAMPLOT_DENSITY_DEFAULT = 3.0

_STREAMPLOT_LINEWIDTH: dict[str, float] = {area: 0.8 for area in _STREAMPLOT_DENSITY}
_STREAMPLOT_LINEWIDTH_DEFAULT = 0.8

# ---------------------------------------------------------------------------
# Contexto compartilhado com _plot_area_worker
# ---------------------------------------------------------------------------
_G: dict = {}

DEFAULT_AREAS = [
    'enso', 'pacifico_leste_america_sul', 'globo_3d', 'MDR', 'pacific_chile', 'china',
    'america_sul_zom_out', 'tropico', 'zona_zcit_atlantico', 'brasil',
    'america_sul', 'africa_monsoon', 'africa', 'mjo', 'amo', 'sad', 'iod',
    'pdo', 'tna', 'tsa', 'atlantico_tropical', 'globo', 'costa_brasil',
    'psa', 'argentina', 'estados_unidos_zoom', 'estados_unidos', 'hemisferio_sul',
]


# ---------------------------------------------------------------------------
# Utilitarios
# ---------------------------------------------------------------------------
def _get_area_list() -> list[str]:
    for attr in ('LST_AREAS_S23', 'LST_AREAS_S13'):
        if hasattr(settings, attr):
            return list(getattr(settings, attr))
    return list(DEFAULT_AREAS)


def _configure_gridlines(gl, area: str) -> None:
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 20, 'color': 'black'}
    gl.ylabel_style = {'size': 20, 'color': 'black'}

    if area in {'hemisferio_sul', 'psa'}:
        gl.xlocator = MultipleLocator(40)
        gl.ylocator = MultipleLocator(20)
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
    elif area in {'globo', 'mjo'}:
        gl.xlocator = MultipleLocator(40)
        gl.ylocator = MultipleLocator(20)
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
    elif area in {'pacific_chile', 'pacifico_leste_america_sul', 'america_sul_zom_out', 'china', 'MDR'}:
        gl.xlocator = MultipleLocator(20)
        gl.ylocator = MultipleLocator(10)


def _download_sst_anos(dados_dir: Path, start_date: np.datetime64, end_date: np.datetime64, logger) -> list[Path]:
    year_start = int(str(start_date)[:4])
    year_end = int(str(end_date)[:4])
    years = list(range(year_start, year_end + 1))
    current_year = datetime.now().year
    logger.info(f'Anos necessarios para o periodo: {years}')
    paths = []
    for year in years:
        url = SST_URL_TEMPLATE.format(year=year)
        sst_path = dados_dir / SST_FILE_TEMPLATE.format(year=year)
        if year < current_year and sst_path.exists():
            logger.info(f'Arquivo SST {year} ja existe localmente — pulando download')
            paths.append(sst_path)
            continue
        year_start_needed = np.datetime64(f'{year}-01-01', 'D')
        year_end_needed = np.datetime64(min(end_date, np.datetime64(f'{year}-12-31', 'D')), 'D')
        if arquivo_cobre_periodo(sst_path, year_start_needed, year_end_needed):
            logger.info(f'Arquivo SST {year} ja cobre o periodo — pulando download')
            paths.append(sst_path)
            continue
        if sst_path.exists():
            logger.info(f'Arquivo SST {year} nao cobre o periodo — re-baixando')
        download_with_progress(
            url=url, output_path=str(sst_path), description=f'SST media {year}',
            max_retries=5, force=sst_path.exists(), prefer_ftp=False,
            engine=DownloadEngine.AUTO, timeout=600,
        )
        paths.append(sst_path)
    return paths


def _download_cmems_currents(dados_dir: Path, start_date: np.datetime64, end_date: np.datetime64, logger) -> Path:
    """Baixa correntes de superficie do CMEMS via copernicusmarine.subset."""
    output_path = dados_dir / CMEMS_CURRENTS_FILE

    if arquivo_cobre_periodo(output_path, start_date, end_date):
        logger.info('Arquivo de correntes CMEMS ja cobre o periodo — pulando download')
        return output_path

    if output_path.exists():
        logger.info('Arquivo de correntes CMEMS nao cobre o periodo — re-baixando')
        output_path.unlink()

    logger.info(f'Baixando correntes CMEMS ({CMEMS_DATASET_ID})...')
    logger.info(f'  Periodo: {start_date} a {end_date} | Profundidade: superficie')

    copernicusmarine.subset(
        dataset_id=CMEMS_DATASET_ID,
        variables=['uo', 'vo'],
        minimum_longitude=-180.0,
        maximum_longitude=180.0,
        minimum_latitude=-90.0,
        maximum_latitude=90.0,
        start_datetime=str(start_date),
        end_datetime=str(end_date),
        minimum_depth=CMEMS_DEPTH_MIN,
        maximum_depth=CMEMS_DEPTH_MAX,
        output_filename=CMEMS_CURRENTS_FILE,
        output_directory=str(dados_dir),
        username=settings.get('CMEMS_USERNAME', ''),
        password=settings.get('CMEMS_PASSWORD', ''),
    )

    logger.info(f'Correntes CMEMS salvas em: {output_path}')
    return output_path


# ---------------------------------------------------------------------------
# Plotagem de uma area
# ---------------------------------------------------------------------------
def _plot_area_worker(area: str) -> str:
    average_data = _G['average_data']
    lon = _G['lon']
    lat = _G['lat']
    info_plot = _G['info_plot']
    output_dir = Path(_G['output_dir'])
    sst_levels = _G['sst_levels']
    sst_ticks = _G['sst_ticks']
    cmap_colors = _G['cmap_colors']
    blue_marble_arr = _G['blue_marble_arr']
    input_dir = Path(_G['input_dir'])
    dt_ini = _G['dt_ini']
    dt_fim = _G['dt_fim']
    lon_cur = _G['lon_cur']
    lat_cur = _G['lat_cur']
    u_cur = _G['u_cur']
    v_cur = _G['v_cur']

    cmap = LinearSegmentedColormap.from_list('sst_media', cmap_colors)

    is_polar = info_plot[area].get('projection', '') == 'orthographic_south'
    if is_polar:
        proj = ccrs.Orthographic(
            central_longitude=settings.get('ORTHO_CENTRAL_LONGITUDE', info_plot[area].get('ortho_central_longitude', -71)),
            central_latitude=settings.get('ORTHO_CENTRAL_LATITUDE', info_plot[area].get('ortho_central_latitude', -84)),
        )
    else:
        proj = ccrs.PlateCarree(central_longitude=info_plot[area]['central_longitude_mapa'])

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_frame_on(False)

    if is_polar:
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

    if blue_marble_arr is not None:
        ax.imshow(
            blue_marble_arr, origin='upper', extent=(-180, 180, -90, 90),
            transform=ccrs.PlateCarree(), interpolation='bilinear', zorder=0,
        )

    if info_plot[area].get('plot_box', False):
        for box in info_plot[area]['lst_boxes']:
            ax.add_patch(patches.Rectangle(
                (box['x_anc'], box['y_anc']), box['x_larg'], box['y_larg'],
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

    if area == 'pacifico_leste_america_sul':
        ax.plot(
            [-170, -20], [0, 0], color='black', linewidth=2.5, linestyle='-',
            transform=ccrs.PlateCarree(), zorder=900,
        )
        t = ax.text(
            -168, 1.5, 'Equador', fontsize=18, color='black', weight='bold',
            transform=ccrs.PlateCarree(), zorder=901,
        )
        t.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])

    if area == 'iod':
        ax.add_patch(patches.Rectangle(
            (50, -10), 20, 20, linewidth=3, edgecolor='black', facecolor='none', zorder=300,
        ))
        ax.add_patch(patches.Rectangle(
            (90, -10), 20, 10, linewidth=3, edgecolor='black', facecolor='none', zorder=300,
        ))

    if area == 'enso':
        for txt, x, y, cor in [
            ('Nino 1+2', 66.25, -13.64, 'red'), ('Nino 3', 34.1, 8.45, 'blue'),
            ('Nino 3.4', 8.6, -9.45, 'black'), ('Nino 4', -22.5, 8.45, 'm'),
        ]:
            t = plt.text(x, y, txt, fontsize=14, color=cor, weight='bold')
            fg = 'black' if cor in {'red', 'm'} else 'white'
            t.set_path_effects([path_effects.Stroke(linewidth=3, foreground=fg), path_effects.Normal()])

    if area == 'mjo':
        for x, txt in [(-135.25, '1'), (-115.25, '2'), (-95.25, '3'), (-75.25, '4'),
                       (-55.25, '5'), (-35.25, '6'), (-15.25, '7'), (4.75, '8')]:
            t = plt.text(x, -4.64, txt, fontsize=50, color='white', weight='bold', zorder=400)
            t.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

    if is_polar:
        gl = ax.gridlines(draw_labels=False, linestyle='--', alpha=0.5)
        gl.xlocator = MultipleLocator(30)
        gl.ylocator = MultipleLocator(20)
    else:
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.0)
        _configure_gridlines(gl, area)

    if not is_polar:
        ax.set_xlim([info_plot[area]['lon_esq'], info_plot[area]['lon_dir']])
        ax.set_ylim([info_plot[area]['lat_inf'], info_plot[area]['lat_sup']])

    if area != 'china':
        ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.8, edgecolor='black', zorder=100)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, edgecolor='black', zorder=100)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, edgecolor='black', zorder=100)

    # --- Camada 1: TSM shaded ---
    im = ax.contourf(
        lon, lat, np.ma.masked_invalid(average_data),
        levels=sst_levels, cmap=cmap, extend='both',
        transform=ccrs.PlateCarree(central_longitude=info_plot[area]['central_longitude_plot']),
        zorder=5,
    )

    # --- Camada 2: streamlines de correntes marinhas de superficie ---
    # Triplicar dados em longitude garante continuidade do streamplot em areas
    # que cruzam o antimeridiano (PSA, PDO, MJO, etc.) — cartopy renderiza
    # apenas o trecho visível no extent do mapa.
    lon_tiled = np.concatenate([lon_cur - 360, lon_cur, lon_cur + 360])
    u_tiled = np.tile(u_cur, (1, 3))
    v_tiled = np.tile(v_cur, (1, 3))

    density = _STREAMPLOT_DENSITY.get(area, _STREAMPLOT_DENSITY_DEFAULT)
    lw = _STREAMPLOT_LINEWIDTH.get(area, _STREAMPLOT_LINEWIDTH_DEFAULT)
    try:
        ax.streamplot(
            lon_tiled, lat_cur, u_tiled, v_tiled,
            color='white',
            linewidth=lw,
            density=density,
            arrowsize=1.0,
            transform=ccrs.PlateCarree(),
            zorder=20,
        )
    except Exception:
        pass  # streamplot pode falhar em areas muito pequenas

    # Colorbar TSM
    if is_polar and area != 'globo_3d':
        cbar = plt.colorbar(im, ax=ax, pad=0.05, fraction=0.04, ticks=sst_ticks)
        cbar.set_label(label='°C', size=10)
        cbar.ax.tick_params(labelsize=10)
    elif area == 'globo_3d':
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
        cbar = plt.colorbar(im, cax=cax, pad=0.02, fraction=0.02375, extend='both', ticks=sst_ticks)
        cbar.set_label(label='°C', size=18)
        cbar.ax.tick_params(labelsize=10)
    elif area in {'enso', 'tropico', 'MDR', 'hemisferio_sul', 'psa'}:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)
        cbar = plt.colorbar(
            im, cax=cax, pad=0.02, fraction=0.02375,
            location='bottom', extend='both', orientation='horizontal', ticks=sst_ticks,
        )
        cbar.set_label(label='°C', size=18)
        cbar.ax.tick_params(labelsize=10)
    else:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
        cbar = plt.colorbar(im, cax=cax, pad=0.02, fraction=0.02375, extend='both', ticks=sst_ticks)
        cbar.set_label(label='°C', size=18)
        cbar.ax.tick_params(labelsize=10)

    titulo = f'Média da TSM (°C) | Correntes Marinhas de Superfície (De {dt_ini} a {dt_fim})'
    ax.set_title(titulo, fontsize=14 if is_polar else 16, loc='left')

    logo_path = resolve_logo_path(input_dir)
    if logo_path is not None and logo_path.exists():
        logo = Image.open(logo_path).convert('RGBA')
        bbox = logo.getbbox()
        if bbox:
            logo = logo.crop(bbox)
        imagebox = OffsetImage(np.array(logo), zoom=proportional_logo_zoom(ax, np.array(logo).shape[1]))
        ab = AnnotationBbox(
            imagebox, (0, 0), xycoords=ax.transAxes,
            xybox=(0, 0), boxcoords='offset points',
            box_alignment=(0, 0), frameon=False, pad=0, zorder=500, clip_on=False,
        )
        ax.add_artist(ab)

    if area != 'globo_3d':
        ax.add_patch(patches.Rectangle(
            (0, 0), 1, 1, linewidth=0.5, edgecolor='black', facecolor='none',
            transform=ax.transAxes, zorder=1000, clip_on=False,
        ))

    filename_fig = output_dir / f'sst_correntes_{area}.png'
    plt.savefig(str(filename_fig), dpi=fig.dpi, bbox_inches='tight')
    plt.close('all')
    return str(filename_fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger = get_logger(SCRIPT_ID)

    logger.info('=' * 80)
    logger.info(f'SCRIPT {SCRIPT_ID.upper()}: {SCRIPT_DESC}')
    logger.info('=' * 80)

    lst_areas = _get_area_list()
    output_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_SST_CORRENTES'
    input_dir = Path(settings.DIR_INPUT)
    dados_dir = Path(settings.DIR_DADOS)

    cache_params = {
        'DATA_INICIAL': settings.DATA_INICIAL,
        'DATA_FINAL': settings.DATA_FINAL,
        'areas': lst_areas,
        'script_version': '1.1',
        'cmems_dataset': CMEMS_DATASET_ID,
    }
    output_files = [str(output_dir / f'sst_correntes_{area}.png') for area in lst_areas]

    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('CACHE VALIDO! Execucao ja foi realizada com os mesmos parametros.')
        logger.info(f'   Periodo: {settings.DATA_INICIAL} a {settings.DATA_FINAL}')
        logger.info(f'   {len(output_files)} mapas ja existem')
        logger.info(f'   Diretorio: {output_dir}')
        logger.info('   Pulando execucao')
        return

    start_time = time.time()
    logger.info(f'Periodo de analise: {settings.DATA_INICIAL} a {settings.DATA_FINAL}')
    logger.info(f'Gerando {len(lst_areas)} mapas — TSM + correntes marinhas')
    logger.info('=' * 80)

    dados_dir.mkdir(parents=True, exist_ok=True)
    start_date = np.datetime64(settings.DATA_INICIAL, 'D')
    end_date = np.datetime64(settings.DATA_FINAL, 'D')

    # ---- Etapa 1: Correntes marinhas CMEMS ----
    logger.info('Etapa 1: Download correntes marinhas de superficie (CMEMS)...')
    currents_path = _download_cmems_currents(dados_dir, start_date, end_date, logger)

    logger.info(f'Carregando {currents_path.name} e calculando media do periodo...')

    # Lê só metadados para detectar nomes de dimensão e calcular step ANTES de abrir tudo
    with xr.open_dataset(str(currents_path)) as _ds_meta:
        _da_meta = _ds_meta['uo']
        lat_dim = 'latitude' if 'latitude' in _da_meta.dims else 'lat'
        lon_dim = 'longitude' if 'longitude' in _da_meta.dims else 'lon'
        _lon_res = abs(float(_da_meta[lon_dim].values[1] - _da_meta[lon_dim].values[0]))

    # Step calculado antes: subsample para ~0.5° ANTES do compute (reduz 36x a memória em pico)
    step = max(1, round(0.5 / _lon_res))
    logger.info(f'CMEMS resolução {_lon_res:.4f}° → step={step} (efetivo ~{_lon_res*step:.2f}°)')

    # chunks={'time': 1} garante que apenas 1 time-step por variável entra na RAM por vez
    ds_cur = xr.open_dataset(str(currents_path), chunks={'time': 1})
    da_uo = ds_cur['uo'].isel(depth=0) if 'depth' in ds_cur['uo'].dims else ds_cur['uo']
    da_vo = ds_cur['vo'].isel(depth=0) if 'depth' in ds_cur['vo'].dims else ds_cur['vo']

    # Subsample espacial ANTES do mean + compute: grade 1/12° → ~0.5°
    da_uo = da_uo.isel(**{lat_dim: slice(None, None, step), lon_dim: slice(None, None, step)})
    da_vo = da_vo.isel(**{lat_dim: slice(None, None, step), lon_dim: slice(None, None, step)})

    da_uo = da_uo.sel(time=slice(str(start_date), str(end_date))).mean(dim='time').compute()
    da_vo = da_vo.sel(time=slice(str(start_date), str(end_date))).mean(dim='time').compute()
    ds_cur.close()

    lat_cur_raw = da_uo[lat_dim].values
    lon_cur_raw = da_uo[lon_dim].values
    u_cur_raw = da_uo.values
    v_cur_raw = da_vo.values
    del da_uo, da_vo

    # streamplot requer lat ascendente
    if lat_cur_raw[0] > lat_cur_raw[-1]:
        lat_cur_raw = lat_cur_raw[::-1]
        u_cur_raw = u_cur_raw[::-1, :]
        v_cur_raw = v_cur_raw[::-1, :]

    # reconstruir lon com linspace — evita imprecisão de float em add_cyclic_point
    lon_cur_raw = np.linspace(float(lon_cur_raw[0]), float(lon_cur_raw[-1]), len(lon_cur_raw))

    # ponto ciclico evita descontinuidade em 180°
    u_cur_cyc, lon_cur_cyc = add_cyclic_point(u_cur_raw, coord=lon_cur_raw)
    v_cur_cyc = add_cyclic_point(v_cur_raw)
    del u_cur_raw, v_cur_raw

    lon_cur = lon_cur_cyc
    lat_cur = lat_cur_raw
    u_cur = u_cur_cyc
    v_cur = v_cur_cyc
    logger.info(f'Correntes subamostradas: grade {len(lat_cur)}x{len(lon_cur)} (step={step})')

    # ---- Etapa 2: SST OISSTv2 ----
    logger.info('Etapa 2: Download SST OISSTv2 (media diaria)...')
    sst_paths = _download_sst_anos(dados_dir, start_date, end_date, logger)

    logger.info('Carregando e processando dados SST...')
    if len(sst_paths) == 1:
        ds = xr.open_dataset(str(sst_paths[0]), decode_times=True, chunks={'time': 20})
    else:
        ds = xr.open_mfdataset(
            [str(p) for p in sst_paths], combine='by_coords',
            decode_times=True, chunks={'time': 20}, coords='minimal',
        )

    ds = ds.sel(time=slice(str(start_date), str(end_date)))
    validar_cobertura_temporal(ds, start_date, end_date, nome='SST OISSTv2')
    ds_mean = ds['sst'].mean(dim='time').compute()
    ds_mean['lon'] = ((ds_mean['lon'] + 180) % 360) - 180
    da = ds_mean.sortby(ds_mean.lon)
    ds.close()

    lon_vals = da['lon'].values
    lat_vals = da['lat'].values
    average_data, lon_cyclic = add_cyclic_point(da.values, coord=lon_vals)

    # ---- Blue marble ----
    blue_marble_path = input_dir / 'blue_marble.png'
    if blue_marble_path.exists():
        _bm = Image.open(blue_marble_path)
        if _bm.width > 4096:
            _ratio = 4096 / _bm.width
            _bm = _bm.resize((4096, int(_bm.height * _ratio)), Image.LANCZOS)
            logger.info(f'blue_marble.png redimensionado para {_bm.size} (original acima de 4K)')
        blue_marble_arr: np.ndarray | None = np.array(_bm)
        del _bm
    else:
        blue_marble_arr = None
        logger.warning(f'blue_marble.png nao encontrado em {input_dir} — usando fundo branco')

    dt_ini = datetime.strptime(settings.DATA_INICIAL, '%Y-%m-%d').strftime('%d-%m-%y')
    dt_fim = datetime.strptime(settings.DATA_FINAL, '%Y-%m-%d').strftime('%d-%m-%y')
    info_plot = json.loads(json.dumps(dict(settings['areas_plotagem'])))

    output_dir.mkdir(parents=True, exist_ok=True)

    global _G
    _G = {
        'average_data': average_data,
        'lon': lon_cyclic,
        'lat': lat_vals,
        'info_plot': info_plot,
        'output_dir': str(output_dir),
        'sst_levels': SST_MEAN_LEVELS,
        'sst_ticks': SST_MEAN_TICKS,
        'cmap_colors': SST_MEAN_COLORS,
        'blue_marble_arr': blue_marble_arr,
        'input_dir': str(input_dir),
        'dt_ini': dt_ini,
        'dt_fim': dt_fim,
        'lon_cur': lon_cur,
        'lat_cur': lat_cur,
        'u_cur': u_cur,
        'v_cur': v_cur,
    }

    logger.info(f'Plotando {len(lst_areas)} areas em sequencial...')
    concluidos = 0
    falhas = []

    for i, area in enumerate(lst_areas):
        try:
            _plot_area_worker(area)
            concluidos += 1
            logger.info(f'[{concluidos}/{len(lst_areas)}] Mapa salvo: {area_display_name(area)}')
        except Exception as exc:
            falhas.append(area)
            logger.error(f'Falha ao gerar mapa para area {area}: {type(exc).__name__}: {exc}')

        # Cartopy acumula geometrias em cache — liberar a cada 5 mapas
        if (i + 1) % 5 == 0:
            gc.collect()

    if falhas:
        raise RuntimeError(f'Falha ao gerar {len(falhas)} mapa(s): {falhas}')

    execution_time = time.time() - start_time
    save_cache_metadata(SCRIPT_ID, cache_params, output_files, execution_time)

    logger.info('=' * 80)
    logger.info(f'Script {SCRIPT_ID.upper()} concluido com sucesso!')
    logger.info(f'Tempo de execucao: {execution_time:.1f}s ({execution_time / 60:.1f} min)')
    logger.info(f'{len(output_files)} mapas gerados em: {output_dir}')
    logger.info('=' * 80)


if __name__ == '__main__':
    main()
