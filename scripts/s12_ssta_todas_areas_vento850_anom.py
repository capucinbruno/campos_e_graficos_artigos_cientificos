# -*- coding: utf-8 -*-
"""
s07 - Anomalia de TSM + Vento Anomalo 850 hPa.

Combina a anomalia de TSM (OISSTv2/NOAA) com vetores de vento
anomalo em 850 hPa (ERA5/GDAS + climatologia PSL) para diversas
areas geograficas.

A anomalia de TSM e calculada a partir da SST absoluta (sst.day.mean) menos a
climatologia diaria OISST (LTM 1991-2020) recortada no mesmo periodo dia-a-dia.

Dados de entrada:
    - PSL/NOAA: sst.day.mean.{ano}.nc (OISSTv2 0.25 grau, um arquivo por ano)
    - Entrada/sst.day.mean.ltm.1991-2020.nc (climatologia diaria OISST p/ anomalia)
    - ERA5/GDAS: vento u/v 850 hPa (ERA5 para periodos antigos, GDAS para recentes)
    - PSL: climatologia u/v 850mb

Saida:
    - Mapas PNG em {settings.DIR_OUTPUT}/s07_SSTA_VENTO850/

Criado em: 2026-06-05
"""

from __future__ import annotations

# Forcar backend nao-interativo antes de qualquer import matplotlib
import matplotlib
matplotlib.use('Agg')

# ---------------------------------------------------------------------------
# Bibliotecas padrao
# ---------------------------------------------------------------------------
import json
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Bibliotecas de terceiros
# ---------------------------------------------------------------------------
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import xarray as xr
from cartopy.util import add_cyclic_point
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import FixedLocator, MultipleLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from PIL import Image

# ---------------------------------------------------------------------------
# Modulos locais
# ---------------------------------------------------------------------------
from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.common.dataset_utils import area_display_name, arquivo_cobre_periodo
from app.common.download_helper import DownloadEngine, download_with_progress
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.common.logo_helper import resolve_logo_path
from app.common.logo_helper import proportional_logo_zoom
from app.src.uteis.indices_climaticos_tsm import calcula_indice_pdo, desenha_boxes_indices
from app.src.uteis.ssta_climatologia import clim_mean_array

# ---------------------------------------------------------------------------
# Identidade do script
# ---------------------------------------------------------------------------
SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's07'
SCRIPT_NAME = Path(__file__).stem
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
SST_URL_TEMPLATE = (
    'https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.day.mean.{year}.nc'
)
SST_FILE_TEMPLATE = 'sst.day.mean.{year}.nc'
WIND850_FILE_NAME = 'wind850_anom.nc'

# Boxes do ENSO (lon/lat reais, -180..180) — regioes canonicas NOAA/CPC, iguais as
# usadas no s24/s30. Cada box tem sua propria regiao: a media respeita o box.
# 'wrap' indica box que cruza a linha de data (Nino 4: 160°E a 150°W).
ENSO_BOXES = {
    'Nino 1+2': {'lon_min': -90,  'lon_max': -80,  'lat_min': -10, 'lat_max': 0, 'wrap': False},
    'Nino 3':   {'lon_min': -150, 'lon_max': -90,  'lat_min': -5,  'lat_max': 5, 'wrap': False},
    'Nino 3.4': {'lon_min': -170, 'lon_max': -120, 'lat_min': -5,  'lat_max': 5, 'wrap': False},
    'Nino 4':   {'lon_min': 160,  'lon_max': -150, 'lat_min': -5,  'lat_max': 5, 'wrap': True},
}

# central_longitude do mapa da area `globo` igual ao s24 (mapa de indices centrado
# no Pacifico) — override local, sem alterar o settings.json compartilhado.
GLOBO_CENTRAL_LONGITUDE_S24 = 220

# Override local (s12) de cor da borda dos boxes por (area, indice) — sobrepoe o
# edgecolor do settings.json sem afetar os demais scripts. Chave = (area, indice em lst_boxes).
BOX_EDGECOLOR_OVERRIDE = {
    ('enso', 0): 'limegreen',  # Nino 1+2 (era 'r' no settings.json)
}

# Contexto compartilhado com a funcao de plotagem
_G: dict = {}

DEFAULT_AREAS = [
    'pacifico_leste_america_sul',
    'enso',
    'mjo',
    'pacific_chile',
    'globo_3d',      
    'america_sul_zom_out',
    'MDR',
    'tropico',
    'zona_zcit_atlantico',
    'brasil',
    'america_sul',
    'africa_monsoon',
    'africa',
    'amo',
    'sad',
    'iod',
    'pdo',
    'tna',
    'tsa',
    'atlantico_tropical',    
    'globo',
    'costa_brasil',
    'psa',
    'argentina',
    'estados_unidos_zoom',
    'estados_unidos',
    'hemisferio_sul',
    'china',
  
]

# Parametros de quiver por area (headwidth, scale, headlength, width)
_QUIVER_PARAMS: dict[str, dict] = {
    'africa_monsoon':             {'headwidth': 3, 'scale': 8,  'headlength': 5, 'width': 0.0022},
    'zona_zcit_atlantico':        {'headwidth': 3, 'scale': 8,  'headlength': 5, 'width': 0.0022},
    'MDR':                        {'headwidth': 3, 'scale': 10, 'headlength': 5, 'width': 0.002},
    'tropico':                    {'headwidth': 3, 'scale': 26, 'headlength': 5, 'width': 0.0008},
    'brasil':                     {'headwidth': 3, 'scale': 8,  'headlength': 5, 'width': 0.0032},
    'america_sul':                {'headwidth': 3, 'scale': 10, 'headlength': 5, 'width': 0.0034},
    'africa':                     {'headwidth': 3, 'scale': 15, 'headlength': 5, 'width': 0.002},
    'mjo':                        {'headwidth': 5, 'scale': 15, 'headlength': 5, 'width': 0.0009},
    'amo':                        {'headwidth': 5, 'scale': 10, 'headlength': 5, 'width': 0.0013},
    'pacifico_leste_america_sul': {'headwidth': 5, 'scale': 15, 'headlength': 5, 'width': 0.0014},
    'china':                      {'headwidth': 5, 'scale': 24, 'headlength': 5, 'width': 0.0006},
    'sad':                        {'headwidth': 5, 'scale': 12, 'headlength': 5, 'width': 0.0014},
    'iod':                        {'headwidth': 5, 'scale': 10, 'headlength': 5, 'width': 0.0013},
    'pdo':                        {'headwidth': 5, 'scale': 12, 'headlength': 5, 'width': 0.0012},
    'tna':                        {'headwidth': 5, 'scale': 10, 'headlength': 5, 'width': 0.0013},
    'tsa':                        {'headwidth': 5, 'scale': 12, 'headlength': 5, 'width': 0.0013},
    'atlantico_tropical':         {'headwidth': 5, 'scale': 12, 'headlength': 5, 'width': 0.0013},
    'enso':                       {'headwidth': 4, 'scale': 15, 'headlength': 5, 'width': 0.001},
    'america_sul_zom_out':        {'headwidth': 4, 'scale': 15, 'headlength': 5, 'width': 0.001},
    'globo':                      {'headwidth': 4, 'scale': 26, 'headlength': 5, 'width': 0.0006},
    'costa_brasil':               {'headwidth': 3, 'scale': 6,  'headlength': 5, 'width': 0.0038},
    'psa':                        {'headwidth': 5, 'scale': 26, 'headlength': 5, 'width': 0.0006},
    'hemisferio_sul':             {'headwidth': 5, 'scale': 26, 'headlength': 5, 'width': 0.0006},
    'argentina':                  {'headwidth': 3, 'scale': 5,  'headlength': 5, 'width': 0.005},
    'estados_unidos_zoom':        {'headwidth': 5, 'scale': 10, 'headlength': 5, 'width': 0.0017},
    'estados_unidos':             {'headwidth': 5, 'scale': 12, 'headlength': 5, 'width': 0.0015},
}
_QUIVER_DEFAULT = {'headwidth': 4, 'scale': 12, 'headlength': 5, 'width': 0.0012}

# Ajuste global de aparência dos vetores (1.0 = sem alteração dos params por área)
QUIVER_WIDTH_FACTOR = 1.0
QUIVER_SCALE_FACTOR = 1.0


# ---------------------------------------------------------------------------
# Funcoes utilitarias
# ---------------------------------------------------------------------------
def _get_area_list() -> list[str]:
    """Retorna lista de areas: prioriza LST_AREAS_S07, fallback para DEFAULT_AREAS."""
    if hasattr(settings, 'LST_AREAS_S07'):
        return list(settings.LST_AREAS_S07)
    return list(DEFAULT_AREAS)


def _box_mean(arr: np.ndarray, lon: np.ndarray, lat: np.ndarray,
              lon_min: float, lon_max: float, lat_min: float, lat_max: float,
              wrap: bool = False) -> float:
    """
    Media (np.nanmean) de `arr` (dims lat, lon) dentro de um box lon/lat (-180..180).

    Cada box e calculado isoladamente na sua propria regiao. `wrap=True` trata boxes
    que cruzam a linha de data (ex: Nino 4, 160°E a 150°W): lon >= lon_min OU lon <= lon_max.
    """
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    if wrap:
        lon_sel = (lon >= lon_min) | (lon <= lon_max)
    else:
        lon_sel = (lon >= lon_min) & (lon <= lon_max)
    lat_sel = (lat >= lat_min) & (lat <= lat_max)
    return float(np.nanmean(arr[np.ix_(lat_sel, lon_sel)]))


def _configure_gridlines(gl, area: str) -> None:
    """Configura gridlines do mapa conforme a area."""
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


# ---------------------------------------------------------------------------
# Plotagem de uma area (le dados do contexto global _G)
# ---------------------------------------------------------------------------
def _plot_area_worker(area: str) -> str:
    """Gera e salva o mapa SSTA + vento 850 hPa para uma area."""
    average_data = _G['average_data']
    lon = _G['lon']
    lat = _G['lat']
    lon_u = _G['lon_u']
    lat_u = _G['lat_u']
    zonal = _G['zonal']
    meridional = _G['meridional']
    info_plot = _G['info_plot']
    output_dir = Path(_G['output_dir'])
    sst_levels = _G['sst_levels']
    cmap_colors = _G['cmap_colors']
    input_dir = Path(_G['input_dir'])
    dt_ini = _G['dt_ini']
    dt_fim = _G['dt_fim']
    enso_box_means = _G['enso_box_means']
    index_pdo = _G['index_pdo']

    cmap = LinearSegmentedColormap.from_list('sst_anom', cmap_colors)

    is_polar = info_plot[area].get('projection', '') == 'orthographic_south'
    if is_polar:
        proj = ccrs.Orthographic(
            central_longitude=settings.get('ORTHO_CENTRAL_LONGITUDE', info_plot[area].get('ortho_central_longitude', -71)),
            central_latitude=settings.get('ORTHO_CENTRAL_LATITUDE', info_plot[area].get('ortho_central_latitude', -84)),
        )
    else:
        # A area `globo` usa a mesma projecao do s24 (centro no Pacifico, 220°) para
        # acomodar todos os boxes/indices sem corte — override local, sem mexer no settings.
        central_lon_mapa = (
            GLOBO_CENTRAL_LONGITUDE_S24 if area == 'globo'
            else info_plot[area]['central_longitude_mapa']
        )
        proj = ccrs.PlateCarree(central_longitude=central_lon_mapa)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_frame_on(False)

    if is_polar:
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

    # Boxes configurados no settings.json
    if info_plot[area].get('plot_box', False):
        for i, box in enumerate(info_plot[area]['lst_boxes']):
            # Forca cor no codigo quando ha override (sem alterar o settings.json)
            edgecolor = BOX_EDGECOLOR_OVERRIDE.get((area, i), box['edgecolor'])
            rect = patches.Rectangle(
                (box['x_anc'], box['y_anc']),
                box['x_larg'],
                box['y_larg'],
                linewidth=box['linewidth'],
                edgecolor=edgecolor,
                facecolor='none',
                zorder=100,
            )
            ax.add_patch(rect)

    # Box MDR
    if area == 'MDR':
        ax.plot(
            [-86, -20, -20, -86, -86],
            [10, 10, 20, 20, 10],
            color='black', linewidth=3, linestyle='-', zorder=500,
            transform=ccrs.PlateCarree(),
        )

    # Area global: replica os boxes e indices climaticos do s24
    # (IOD, Nino 1+2/3/3.4/4, AMO, TNA, TSA, SAD, PDO)
    if area == 'globo' and index_pdo is not None:
        da_avg = xr.DataArray(
            average_data, dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}
        )
        desenha_boxes_indices(ax, da_avg, index_pdo)

    # Linha do Equador + label para pacifico_leste_america_sul
    if area == 'pacifico_leste_america_sul':
        ax.plot(
            [-170, -20], [0, 0],
            color='black', linewidth=2.5, linestyle='-',
            transform=ccrs.PlateCarree(), zorder=900,
        )
        t = ax.text(
            -168, 1.5, 'Equador',
            fontsize=18, color='white', weight='bold',
            transform=ccrs.PlateCarree(), zorder=901,
        )
        t.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground='black'),
            path_effects.Normal(),
        ])

    # Legenda e boxes Atlantico tropical
    if area == 'atlantico_tropical':
        legenda_atl = input_dir / 'legenda_atlantic.png'
        if legenda_atl.exists():
            img_legenda_atlantic = plt.imread(str(legenda_atl))
            fig.figimage(img_legenda_atlantic, 125, 614, zorder=3, alpha=1)
        box_tsa = patches.Rectangle(
            (10, -20), -40, 20, linewidth=3, edgecolor='black', facecolor='none', zorder=100,
        )
        box_tna = patches.Rectangle(
            (-15, 5), -40, 20, linewidth=3, edgecolor='blue', facecolor='none', zorder=100,
        )
        ax.add_patch(box_tsa)
        ax.add_patch(box_tna)

    # Boxes IOD
    if area == 'iod':
        ax.add_patch(patches.Rectangle(
            (50, -10), 20, 20, linewidth=3, edgecolor='black', facecolor='none', zorder=100,
        ))
        ax.add_patch(patches.Rectangle(
            (90, -10), 20, 10, linewidth=3, edgecolor='black', facecolor='none', zorder=100,
        ))

    # Labels ENSO — centralizados no eixo x com o centro de cada box (lst_boxes do settings.json),
    # com a anomalia media de TSM do proprio box ao lado do nome (ex: "Nino 4 = 1.2°C").
    # (label, indice do box em lst_boxes, y do texto, cor do texto = cor do box)
    if area == 'enso':
        boxes = info_plot[area].get('lst_boxes', [])
        for txt, box_idx, y, cor in [
            ('Nino 1+2', 0, -13.64, 'limegreen'),
            ('Nino 3', 1, 8.45, 'blue'),
            ('Nino 3.4', 3, -9.45, 'black'),
            ('Nino 4', 2, 8.45, 'm'),
        ]:
            if box_idx >= len(boxes):
                continue
            box = boxes[box_idx]
            cx = box['x_anc'] + box['x_larg'] / 2  # centro horizontal do box
            val = enso_box_means.get(txt)
            label = f'{txt} = {val:.2f}°C' if val is not None and np.isfinite(val) else txt
            t = plt.text(cx, y, label, fontsize=14, color=cor, weight='bold', ha='center', zorder=400)
            fg = 'black' if cor in {'limegreen', 'm'} else 'white'
            t.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground=fg),
                path_effects.Normal(),
            ])

    # Fases MJO
    if area == 'mjo':
        for x, txt in [(-135.25, '1'), (-115.25, '2'), (-95.25, '3'), (-75.25, '4'),
                       (-55.25, '5'), (-35.25, '6'), (-15.25, '7'), (4.75, '8')]:
            t = plt.text(x, -4.64, txt, fontsize=50, color='white', weight='bold', zorder=400)
            t.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground='black'),
                path_effects.Normal(),
            ])

    # Gridlines
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

    # Fundo de terra (sem blue marble)
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='whitesmoke', zorder=3)

    # Features cartograficas
    if area != 'china':
        ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.8, edgecolor='black', zorder=100)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, edgecolor='black', zorder=100)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, edgecolor='black', zorder=100)

    # Contourf SSTA (NaN em terra e ignorado automaticamente)
    im = ax.contourf(
        lon,
        lat,
        np.ma.masked_invalid(average_data),
        levels=sst_levels,
        cmap=cmap,
        extend='both',
        transform=ccrs.PlateCarree(
            central_longitude=info_plot[area]['central_longitude_plot']
        ),
        zorder=5,
    )

    # Vetores de vento anomalo 850 hPa
    qp = _QUIVER_PARAMS.get(area, _QUIVER_DEFAULT)
    ax.quiver(
        lon_u,
        lat_u,
        zonal,
        meridional,
        scale_units='inches',
        color='k',
        headwidth=qp['headwidth'],
        scale=qp['scale'] * QUIVER_SCALE_FACTOR,
        headlength=qp['headlength'],
        width=qp['width'] * QUIVER_WIDTH_FACTOR,
        transform=ccrs.PlateCarree(
            central_longitude=info_plot[area]['central_longitude_plot']
        ),
        zorder=50,
    )

    # Colorbar
    if is_polar and area != 'globo_3d':
        cbar = plt.colorbar(im, ax=ax, pad=0.05, fraction=0.04, ticks=sst_levels)
        cbar.set_label(label='°C', size=10)
        cbar.ax.tick_params(labelsize=10)
    elif area in {'enso', 'tropico', 'MDR', 'hemisferio_sul', 'psa'}:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)
        cbar = plt.colorbar(
            im, cax=cax, pad=0.02, fraction=0.02375,
            location='bottom', extend='both', orientation='horizontal',
            ticks=sst_levels,
        )
        cbar.set_label(label='°C', size=18)
        cbar.ax.tick_params(labelsize=10)
    else:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
        cbar = plt.colorbar(
            im, cax=cax, pad=0.02, fraction=0.02375,
            extend='both', ticks=sst_levels,
        )
        cbar.set_label(label='°C', size=18)
        cbar.ax.tick_params(labelsize=10)

    # Titulo
    titulo = f'Anomalia de TSM + Vento 850 hPa (De {dt_ini} a {dt_fim})'
    ax.set_title(titulo, fontsize=14 if is_polar else 18, loc='left')

    # Logo
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

    # Contorno uniforme em todos os 4 lados da area de plotagem
    if area != 'globo_3d':
        ax.add_patch(patches.Rectangle(
            (0, 0), 1, 1,
            linewidth=0.5, edgecolor='black', facecolor='none',
            transform=ax.transAxes, zorder=1000, clip_on=False,
        ))

    # Salvar
    filename_fig = output_dir / f's07_ssta_vento850_{area}.png'
    plt.savefig(str(filename_fig), dpi=fig.dpi, bbox_inches='tight')
    plt.close('all')

    return str(filename_fig)


# ---------------------------------------------------------------------------
# Media SST via streaming (evita xr.concat de anos inteiros na RAM)
# ---------------------------------------------------------------------------
def _compute_sst_mean_streaming(
    sst_paths: list[Path],
    start_date: np.datetime64,
    end_date: np.datetime64,
    logger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula media SST processando um arquivo por vez para evitar estouro de RAM.

    Cada arquivo SST anual (0.25° global, 365 dias) ocupa ~1.5 GB se carregado
    por inteiro. Esta funcao processa em janelas de 30 dias (~120 MB/chunk) e
    acumula sum/count, mantendo apenas arrays 2D na RAM entre arquivos.
    """
    sum_2d: np.ndarray | None = None
    count_2d: np.ndarray | None = None
    lon_vals: np.ndarray | None = None
    lat_vals: np.ndarray | None = None
    total_days = 0

    for p in sst_paths:
        logger.info(f'Streaming SST: {p.name}...')
        with xr.open_dataset(str(p), decode_times=True) as ds:
            da = ds['sst'].sel(time=slice(str(start_date), str(end_date)))
            n = int(da.sizes.get('time', 0))
            if n == 0:
                logger.warning(f'Nenhum timestep no periodo em {p.name} — pulando')
                continue

            if lon_vals is None:
                lon_vals = ds['lon'].values.copy()
                lat_vals = ds['lat'].values.copy()
                nlat, nlon = len(lat_vals), len(lon_vals)
                sum_2d = np.zeros((nlat, nlon), dtype=np.float64)
                count_2d = np.zeros((nlat, nlon), dtype=np.int64)

            total_days += n
            chunk_size = 30  # ~120 MB por chunk; arquivo completo = ~1.5 GB
            for i in range(0, n, chunk_size):
                arr = da.isel(time=slice(i, i + chunk_size)).values
                valid = ~np.isnan(arr)
                sum_2d += np.where(valid, arr, 0.0).sum(axis=0)
                count_2d += valid.sum(axis=0).astype(np.int64)
                del arr, valid

    if sum_2d is None:
        raise ValueError(f'Nenhum dado SST encontrado no periodo {start_date} a {end_date}')

    logger.info(f'SST: {total_days} dias processados no periodo')
    return np.where(count_2d > 0, sum_2d / count_2d, np.nan), lon_vals, lat_vals


# ---------------------------------------------------------------------------
# Download SST
# ---------------------------------------------------------------------------
def _download_sst_anos(dados_dir: Path, start_date: np.datetime64, end_date: np.datetime64, logger) -> list[Path]:
    """Baixa arquivos OISSTv2 anuais cobrindo o periodo solicitado."""
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
        year_end_needed = np.datetime64(
            min(end_date, np.datetime64(f'{year}-12-31', 'D')), 'D'
        )

        if arquivo_cobre_periodo(sst_path, year_start_needed, year_end_needed):
            logger.info(f'Arquivo SST {year} ja cobre o periodo ate {year_end_needed} — pulando download')
            paths.append(sst_path)
            continue

        if sst_path.exists():
            logger.info(f'Arquivo SST {year} nao cobre {year_start_needed} a {year_end_needed} — re-baixando')

        download_with_progress(
            url=url,
            output_path=str(sst_path),
            description=f'SST media {year}',
            max_retries=5,
            force=sst_path.exists(),
            prefer_ftp=False,
            engine=DownloadEngine.AUTO,
            timeout=600,
        )
        paths.append(sst_path)

    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger = get_logger(SCRIPT_ID)

    logger.info('=' * 80)
    logger.info(f'SCRIPT {SCRIPT_ID.upper()}: {SCRIPT_DESC}')
    logger.info('=' * 80)

    lst_areas = _get_area_list()

    output_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_SSTA_VENTO850'
    input_dir = Path(settings.DIR_INPUT)
    dados_dir = Path(settings.DIR_DADOS)

    cache_params = {
        'DATA_INICIAL': settings.DATA_INICIAL,
        'DATA_FINAL': settings.DATA_FINAL,
        'areas': lst_areas,
        'anom_source': 'sst.day.mean - ltm.1991-2020',
        'script_version': '2.4',
    }
    output_files = [str(output_dir / f's07_ssta_vento850_{area}.png') for area in lst_areas]

    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('CACHE VALIDO! Execucao ja foi realizada com os mesmos parametros.')
        logger.info(f'   Periodo: {settings.DATA_INICIAL} a {settings.DATA_FINAL}')
        logger.info(f'   {len(output_files)} mapas ja existem')
        logger.info(f'   Diretorio: {output_dir}')
        logger.info('   Pulando execucao')
        return

    start_time = time.time()
    logger.info(f'Periodo de analise: {settings.DATA_INICIAL} a {settings.DATA_FINAL}')
    logger.info(f'Gerando {len(lst_areas)} mapas de SSTA + Vento 850 hPa')
    logger.info('=' * 80)

    start_date = np.datetime64(settings.DATA_INICIAL, 'D')
    end_date = np.datetime64(settings.DATA_FINAL, 'D')

    # ---- Anomalia de vento 850 hPa (ERA5/GDAS + PSL) ----
    logger.info('Etapa 1: Calculando anomalia de vento 850 hPa (ERA5/GDAS + PSL)...')
    from app.src.uteis.plot_olr_wind850_anom import main as _wind850_main
    _wind850_main()

    # ---- Carrega wind850_anom.nc ----
    wind850_path = dados_dir / WIND850_FILE_NAME
    logger.info(f'Carregando {wind850_path}...')
    ds_wind = xr.open_dataset(wind850_path)
    u_anom_raw = ds_wind['u_anom_mean'].values   # shape (lat, lon)
    v_anom_raw = ds_wind['v_anom_mean'].values
    lon_wind = ds_wind['lon'].values
    lat_wind = ds_wind['lat'].values
    ds_wind.close()

    # Subsample para quiver: ERA5 1°/1° → a cada 3 pontos (~3°), equivalente ao PSL 2.5°
    lon_u = lon_wind[::3]
    lat_u = lat_wind[::3]
    zonal = u_anom_raw[::3, ::3]
    meridional = v_anom_raw[::3, ::3]
    logger.info(f'Vento 850 subamostrado: grade {len(lat_u)}x{len(lon_u)}')

    # Mascara vetores fracos (equivalente ao s35)
    ws = (zonal**2 + meridional**2) ** 1.2
    zonal = ma.masked_where(ws < 1, zonal)
    meridional = ma.masked_where(ws < 1, meridional)

    # ---- Download dos arquivos SST anuais ----
    logger.info('Etapa 2: Download SST OISSTv2...')
    dados_dir.mkdir(parents=True, exist_ok=True)
    sst_paths = _download_sst_anos(dados_dir, start_date, end_date, logger)

    # ---- Carregamento e media SST absoluta (streaming — sem xr.concat de anos na RAM) ----
    logger.info('Etapa 3: Carregando e processando dados SST (streaming)...')
    average_sst_raw, lon_vals_raw, lat_vals = _compute_sst_mean_streaming(
        sst_paths, start_date, end_date, logger
    )

    # ---- Anomalia = media(SST) - media(climatologia) recortada no mesmo periodo ----
    logger.info('Etapa 3b: Calculando anomalia com a climatologia diaria OISST...')
    clim_mean_raw = clim_mean_array(start_date, end_date, lat_vals, lon_vals_raw, logger)
    average_data_raw = average_sst_raw - clim_mean_raw

    # Ajustar longitude de 0-360 para -180..180 e ordenar
    lon_centered = ((lon_vals_raw + 180) % 360) - 180
    sort_idx = np.argsort(lon_centered)
    lon_sorted = lon_centered[sort_idx]
    average_data_sorted = average_data_raw[:, sort_idx]

    average_data, lon_cyclic = add_cyclic_point(average_data_sorted, coord=lon_sorted)

    # ---- Anomalia media de TSM por box do ENSO (cada box na sua regiao real lon/lat) ----
    enso_box_means = {
        nome: _box_mean(average_data_sorted, lon_sorted, lat_vals,
                        b['lon_min'], b['lon_max'], b['lat_min'], b['lat_max'], b['wrap'])
        for nome, b in ENSO_BOXES.items()
    }
    logger.info(
        'Anomalia media de TSM por box ENSO (°C): '
        + ', '.join(f'{nome}={val:.2f}' for nome, val in enso_box_means.items())
    )

    # ---- Indice PDO (so quando a area global e plotada — depende da EOF1) ----
    index_pdo = None
    if 'globo' in lst_areas:
        ds_anom = xr.Dataset(
            {'anom': (('lat', 'lon'), average_data_sorted)},
            coords={'lat': lat_vals, 'lon': lon_sorted},
        )
        index_pdo = calcula_indice_pdo(ds_anom, 'anom')
        logger.info(f'PDO = {index_pdo:.2f}')

    # ---- Contexto compartilhado ----
    dt_ini = datetime.strptime(settings.DATA_INICIAL, '%Y-%m-%d').strftime('%d-%m-%y')
    dt_fim = datetime.strptime(settings.DATA_FINAL, '%Y-%m-%d').strftime('%d-%m-%y')
    info_plot = json.loads(json.dumps(dict(settings['areas_plotagem'])))
    sst_levels = [float(x) for x in settings.LST_SSTA_NEW_GREC]
    cmap_colors = [str(x) for x in settings.LST_ANOM_CORRETA]

    output_dir.mkdir(parents=True, exist_ok=True)

    global _G
    _G = {
        'average_data': average_data,
        'lon': lon_cyclic,
        'lat': lat_vals,
        'lon_u': lon_u,
        'lat_u': lat_u,
        'zonal': zonal,
        'meridional': meridional,
        'info_plot': info_plot,
        'output_dir': str(output_dir),
        'sst_levels': sst_levels,
        'cmap_colors': cmap_colors,
        'input_dir': str(input_dir),
        'dt_ini': dt_ini,
        'dt_fim': dt_fim,
        'enso_box_means': enso_box_means,
        'index_pdo': index_pdo,
    }

    # ---- Plotagem sequencial ----
    logger.info(f'Etapa 4: Plotando {len(lst_areas)} areas em sequencial...')

    concluidos = 0
    falhas = []

    for area in lst_areas:
        try:
            _plot_area_worker(area)
            concluidos += 1
            logger.info(f'[{concluidos}/{len(lst_areas)}] Mapa salvo: {area_display_name(area)}')
        except Exception as exc:
            falhas.append(area)
            logger.error(f'Falha ao gerar mapa para area {area}: {type(exc).__name__}: {exc}')

    if falhas:
        raise RuntimeError(f'Falha ao gerar {len(falhas)} mapa(s): {falhas}')

    # Salvar cache
    execution_time = time.time() - start_time
    save_cache_metadata(SCRIPT_ID, cache_params, output_files, execution_time)

    logger.info('=' * 80)
    logger.info(f'Script {SCRIPT_ID.upper()} concluido com sucesso!')
    logger.info(f'Tempo de execucao: {execution_time:.1f}s ({execution_time / 60:.1f} min)')
    logger.info(f'{len(output_files)} mapas gerados em: {output_dir}')
    logger.info('=' * 80)


if __name__ == '__main__':
    main()
