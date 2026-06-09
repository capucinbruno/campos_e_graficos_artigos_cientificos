# -*- coding: utf-8 -*-
"""
s12 - Media de TSM (Temperatura da Superficie do Mar).

Baixa dados de TSM media diaria do PSL/NOAA (OISSTv2 High-Res),
calcula a media do periodo selecionado e gera mapas de TSM media
para diversas areas geograficas.

Dados de entrada:
    - PSL/NOAA: sst.day.mean.{ano}.nc (OISSTv2 0.25 grau, um arquivo por ano)

Saida:
    - Mapas PNG em {settings.DIR_OUTPUT}/s12_SST_MEDIA/

Criado em: 2026-06-04
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
import metpy.calc as mpcalc
import numpy as np
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
from app.common.dataset_utils import area_display_name, arquivo_cobre_periodo, load_dataset, validar_cobertura_temporal
from app.common.download_helper import DownloadEngine, download_with_progress
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.src.uteis.plot_wind1000_mean import main as _wind1000_main

# ---------------------------------------------------------------------------
# Identidade do script
# ---------------------------------------------------------------------------
SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's12'
SCRIPT_NAME = Path(__file__).stem
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
SST_URL_TEMPLATE = (
    'https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.day.mean.{year}.nc'
)
SST_FILE_TEMPLATE = 'sst.day.mean.{year}.nc'
WIND1000_FILE_NAME = 'wind1000_mean.nc'
OLR_URL = 'https://downloads.psl.noaa.gov/Datasets/cpc_blended_olr-2.5deg/olr.day.anom.nc'
OLR_FILE_NAME = 'olr.day.anom.nc'
OLR_NEG_LEVELS = np.arange(-40, 0, 4)  # isolinhas negativas: -40, -36, ..., -4

# Levels: 0-32°C em passos de 0.5°C (65 niveis)
SST_MEAN_LEVELS = [round(i * 0.5, 1) for i in range(65)]
# Ticks da colorbar: de 2 em 2
SST_MEAN_TICKS = list(range(0, 33, 2))
# 16 cores interpoladas para TSM media
SST_MEAN_COLORS = [
    'white', 'blueviolet', 'blue', 'cyan', 'limegreen', 'greenyellow',
    'yellow', 'gold', 'orange', 'darkorange', 'orangered', 'red',
    'darkred', 'crimson', 'magenta', 'white',
]

# Contexto compartilhado com a funcao de plotagem (evita passar arrays grandes como argumento)
_G: dict = {}

# Cores dos boxes oceanicos por area — sobrepoe edgecolor do settings.json
# Indice = posicao do box em lst_boxes
BOX_COLORS: dict[str, list[str]] = {
    'amo':  ['black'],
    'tsa':  ['black'],
    'tna':  ['black'],
    'pdo':  ['black'],
    'iod':  ['black', 'black'],
    # enso: boxes desenhados diretamente no bloco if area=='enso'
    'sad':  ['black', 'black'],
}

DEFAULT_AREAS = [
    'enso',
    'pacifico_leste_america_sul',
    'globo_3d',
    'MDR',
    'pacific_chile',
    'china',  
    'america_sul_zom_out',   
    'tropico',
    'zona_zcit_atlantico',
    'brasil',
    'america_sul',
    'africa_monsoon',
    'africa',
    'mjo',
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
]


# ---------------------------------------------------------------------------
# Funcoes utilitarias
# ---------------------------------------------------------------------------
def _get_area_list() -> list[str]:
    """Retorna lista de areas: prioriza LST_AREAS_S12, fallback para DEFAULT_AREAS."""
    if hasattr(settings, 'LST_AREAS_S12'):
        return list(settings.LST_AREAS_S12)
    return list(DEFAULT_AREAS)


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
    """Gera e salva o mapa de media de TSM para uma area."""
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
    lon_w = _G['lon_w']
    lat_w = _G['lat_w']
    u_mean = _G['u_mean']
    v_mean = _G['v_mean']
    olr_data = _G['olr_data']
    lon_olr = _G['lon_olr']
    lat_olr = _G['lat_olr']

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
            blue_marble_arr,
            origin='upper',
            extent=(-180, 180, -90, 90),
            transform=ccrs.PlateCarree(),
            interpolation='bilinear',
            zorder=0,
        )

    # Boxes configurados no settings.json — cores sobrepostas por BOX_COLORS
    # enso e tratado separadamente abaixo com coordenadas e cores explicitas
    if info_plot[area].get('plot_box', False) and area != 'enso':
        area_colors = BOX_COLORS.get(area, [])
        for i, box in enumerate(info_plot[area]['lst_boxes']):
            color = area_colors[i] if i < len(area_colors) else box['edgecolor']
            rect = patches.Rectangle(
                (box['x_anc'], box['y_anc']),
                box['x_larg'],
                box['y_larg'],
                linewidth=box['linewidth'],
                edgecolor=color,
                facecolor='none',
                zorder=300,
            )
            ax.add_patch(rect)

    # Box MDR
    if area == 'MDR':
        ax.plot(
            [-86, -20, -20, -86, -86],
            [10, 10, 20, 20, 10],
            color='black', linewidth=3, linestyle='-', zorder=300,
            transform=ccrs.PlateCarree(),
        )

    # Legenda e boxes Atlantico tropical
    if area == 'atlantico_tropical':
        legenda_atl = input_dir / 'legenda_atlantic.png'
        if legenda_atl.exists():
            img_legenda_atlantic = plt.imread(str(legenda_atl))
            fig.figimage(img_legenda_atlantic, 125, 614, zorder=3, alpha=1)
        box_tsa = patches.Rectangle(
            (10, -20), -40, 20, linewidth=3, edgecolor='black', facecolor='none', zorder=300,
        )
        box_tna = patches.Rectangle(
            (-15, 5), -40, 20, linewidth=3, edgecolor='blue', facecolor='none', zorder=300,
        )
        ax.add_patch(box_tsa)
        ax.add_patch(box_tna)

    # Linha do Equador + label para pacifico_leste_america_sul
    if area == 'pacifico_leste_america_sul':
        ax.plot(
            [-170, -20], [0, 0],
            color='white', linewidth=2.5, linestyle='-',
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

    # Boxes IOD
    if area == 'iod':
        ax.add_patch(patches.Rectangle(
            (50, -10), 20, 20, linewidth=3, edgecolor='black', facecolor='none', zorder=300,
        ))
        ax.add_patch(patches.Rectangle(
            (90, -10), 20, 10, linewidth=3, edgecolor='black', facecolor='none', zorder=300,
        ))

    # Boxes e labels ENSO — coordenadas e cores definidas aqui, independente do settings.json
    # Coordenadas no sistema de PlateCarree(central_longitude=-160) do mapa ENSO
    if area == 'enso':
        for (x, y, w, h, cor, zo), (txt, lx, ly, lfg) in zip(
            [
                ( 70,   -9,  10,  10, 'black', 300),   # Nino 1+2
                ( 70, -4.2, -60,  10, 'cyan',  300),   # Nino 3
                ( 40, -4.2, -50,  10, 'lime',  350),   # Nino 3.4 — acima das demais
                (9.1, -4.2, -50,  10, 'yellow',300),   # Nino 4
            ],
            [
                ('Nino 1+2',  66.25, -13.64, 'white'),
                ('Nino 3',    34.1,    8.45, 'black'),
                ('Nino 3.4',   8.6,   -9.45, 'black'),
                ('Nino 4',   -22.5,    8.45, 'black'),
            ],
        ):
            ax.add_patch(patches.Rectangle(
                (x, y), w, h, linewidth=3, edgecolor=cor, facecolor='none', zorder=zo,
            ))
            t = plt.text(lx, ly, txt, fontsize=14, color=cor, weight='bold')
            t.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground=lfg),
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

    # Features cartograficas — sem LAND(whitesmoke): blue marble aparece em terra
    if area != 'china':
        ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.8, edgecolor='black', zorder=100)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, edgecolor='black', zorder=100)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, edgecolor='black', zorder=100)

    # Contourf TSM media (NaN em terra e ignorado automaticamente)
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

    # Isolinhas de anomalia negativa de OLR (preto) — conveccao intensa
    ax.contour(
        lon_olr,
        lat_olr,
        olr_data,
        levels=OLR_NEG_LEVELS,
        colors='black',
        linewidths=2.0,
        transform=ccrs.PlateCarree(
            central_longitude=info_plot[area]['central_longitude_plot']
        ),
        zorder=30,
    )

    # Streamlines de vento medio 1000 hPa (branco) — zorder acima das features cartograficas
    if not is_polar or area == 'globo_3d':
        try:
            ax.streamplot(
                lon_w, lat_w, u_mean, v_mean,
                color='white',
                linewidth=0.8,
                density=2,
                arrowsize=1.0,
                transform=ccrs.PlateCarree(),
                zorder=150,
            )
        except Exception:
            pass  # streamplot pode falhar em areas muito pequenas ou com dados insuficientes

    # Colorbar
    if is_polar and area != 'globo_3d':
        cbar = plt.colorbar(im, ax=ax, pad=0.05, fraction=0.04, ticks=sst_levels)
        cbar.set_label(label='°C', size=10)
        cbar.ax.tick_params(labelsize=10)
    elif area == 'globo_3d':
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
        cbar = plt.colorbar(
            im, cax=cax, pad=0.02, fraction=0.02375,
            extend='both', extendrect=False, ticks=sst_ticks,
        )
        cbar.set_label(label='°C', size=18)
        cbar.ax.tick_params(labelsize=10)
    elif area in {'enso', 'tropico', 'MDR', 'hemisferio_sul', 'psa'}:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)
        cbar = plt.colorbar(
            im, cax=cax, pad=0.02, fraction=0.02375,
            location='bottom', extend='both', orientation='horizontal',
            ticks=sst_ticks,
        )
        cbar.set_label(label='°C', size=18)
        cbar.ax.tick_params(labelsize=10)
    else:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
        cbar = plt.colorbar(
            im, cax=cax, pad=0.02, fraction=0.02375,
            extend='both', ticks=sst_ticks,
        )
        cbar.set_label(label='°C', size=18)
        cbar.ax.tick_params(labelsize=10)

    # Titulo
    titulo = f'Média da TSM (°C) | Vento 1000 hPa (m/s) | Anomalia de OLR < 0 W/m² (De {dt_ini} a {dt_fim})'
    ax.set_title(titulo, fontsize=14 if is_polar else 16, loc='left')

    # Logo
    logo_path = (
            None if settings.get('SEM_LOGO', False)
            else input_dir / ('logo_grec.png' if settings.get('LOGO_GREC', False) else 'novo_logo.png')
        )
    if logo_path is not None and logo_path.exists():
        logo = Image.open(logo_path).convert('RGBA')
        bbox = logo.getbbox()
        if bbox:
            logo = logo.crop(bbox)
        imagebox = OffsetImage(np.array(logo), zoom=0.65)
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
    filename_fig = output_dir / f'sst_media_{area}.png'
    plt.savefig(str(filename_fig), dpi=fig.dpi, bbox_inches='tight')
    plt.close('all')

    return str(filename_fig)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def _download_sst_anos(dados_dir: Path, start_date: np.datetime64, end_date: np.datetime64, logger) -> list[Path]:
    """
    Baixa arquivos OISSTv2 anuais cobrindo o periodo solicitado.

    Um arquivo por ano. Aria2 com 16 conexoes paralelas via HTTP (mais rapido que FTP
    single-connection). Anos passados so sao baixados se ausentes; o ano corrente
    e re-baixado quando o arquivo local nao cobre DATA_FINAL.
    """
    year_start = int(str(start_date)[:4])
    year_end = int(str(end_date)[:4])
    years = list(range(year_start, year_end + 1))
    current_year = datetime.now().year

    logger.info(f'Anos necessarios para o periodo: {years}')
    paths = []

    for year in years:
        url = SST_URL_TEMPLATE.format(year=year)
        sst_path = dados_dir / SST_FILE_TEMPLATE.format(year=year)

        # Anos passados: so baixar se arquivo nao existe
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
            prefer_ftp=False,  # manter HTTP para aria2 usar 16 conexoes paralelas
            engine=DownloadEngine.AUTO,  # AUTO: aria2 > pycurl > httpx > requests
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

    output_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_SST_MEDIA'
    input_dir = Path(settings.DIR_INPUT)
    dados_dir = Path(settings.DIR_DADOS)

    cache_params = {
        'DATA_INICIAL': settings.DATA_INICIAL,
        'DATA_FINAL': settings.DATA_FINAL,
        'areas': lst_areas,
        'script_version': '1.1',
    }
    output_files = [str(output_dir / f'sst_media_{area}.png') for area in lst_areas]

    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('CACHE VALIDO! Execucao ja foi realizada com os mesmos parametros.')
        logger.info(f'   Periodo: {settings.DATA_INICIAL} a {settings.DATA_FINAL}')
        logger.info(f'   {len(output_files)} mapas ja existem')
        logger.info(f'   Diretorio: {output_dir}')
        logger.info('   Pulando execucao')
        return

    start_time = time.time()
    logger.info(f'Periodo de analise: {settings.DATA_INICIAL} a {settings.DATA_FINAL}')
    logger.info(f'Gerando {len(lst_areas)} mapas de media de TSM')
    logger.info('=' * 80)

    start_date = np.datetime64(settings.DATA_INICIAL, 'D')
    end_date = np.datetime64(settings.DATA_FINAL, 'D')

    # ---- Vento medio 1000 hPa (ERA5/GDAS, sem anomalia) ----
    dados_dir.mkdir(parents=True, exist_ok=True)
    logger.info('Etapa 1: Calculando vento medio 1000 hPa (ERA5/GDAS)...')
    _wind1000_main()

    wind1000_path = dados_dir / WIND1000_FILE_NAME
    logger.info(f'Carregando {wind1000_path.name}...')
    ds_wind = xr.open_dataset(str(wind1000_path))
    u_raw = ds_wind['u_mean'].values   # (lat, lon)
    v_raw = ds_wind['v_mean'].values
    lon_wind = ds_wind['lon'].values
    lat_wind = ds_wind['lat'].values
    ds_wind.close()

    # Subamostrar para ~4° de resolucao — streamlines ficam mais limpas e rapidas
    step = max(1, round(4.0 / abs(float(lon_wind[1] - lon_wind[0]))))
    lon_w = lon_wind[::step]
    lat_w = lat_wind[::step]
    u_mean_w = u_raw[::step, ::step]
    v_mean_w = v_raw[::step, ::step]
    # Ponto ciclico evita corte de streamlines na longitude 180°
    u_mean_w, lon_w = add_cyclic_point(u_mean_w, coord=lon_w)
    v_mean_w = add_cyclic_point(v_mean_w)
    logger.info(f'Vento 1000 hPa subamostrado: grade {len(lat_w)}x{len(lon_w)} (~{step}° resolucao)')

    # ---- Download e processamento OLR (PSL/NOAA CPC Blended) ----
    logger.info('Etapa 2: Download e processamento de anomalia de OLR...')
    olr_path = dados_dir / OLR_FILE_NAME
    if arquivo_cobre_periodo(olr_path, start_date, end_date):
        logger.info('Arquivo OLR local ja cobre o periodo — pulando download')
    else:
        if olr_path.exists():
            logger.info('Arquivo OLR nao cobre o periodo solicitado — re-baixando')
        download_with_progress(
            url=OLR_URL,
            output_path=str(olr_path),
            description=OLR_FILE_NAME,
            max_retries=5,
            force=True,
            engine=DownloadEngine.ARIA2,
            timeout=300,
        )

    ds_olr = load_dataset(str(olr_path))
    validar_cobertura_temporal(ds_olr, start_date, end_date, nome='OLR PSL/NOAA')
    subset_olr = ds_olr.sel(time=slice(start_date, end_date))
    ds_olr_mean = subset_olr.mean(dim='time')
    ds_olr_mean['lon'] = ((ds_olr_mean['lon'] + 180) % 360) - 180
    da_olr = ds_olr_mean.sortby(ds_olr_mean.lon)['olr']
    da_olr = mpcalc.smooth_gaussian(da_olr, 5)
    lon_olr_1d = da_olr['lon'].values
    lat_olr_1d = da_olr['lat'].values
    olr_arr, lon_olr_cyc = add_cyclic_point(da_olr.values, coord=lon_olr_1d)
    logger.info(f'OLR processado: grade {olr_arr.shape[0]}x{olr_arr.shape[1]}')

    # ---- Download dos arquivos SST anuais ----
    logger.info('Etapa 3: Download SST OISSTv2 (media diaria)...')
    sst_paths = _download_sst_anos(dados_dir, start_date, end_date, logger)

    # ---- Carregamento e concatenacao ----
    logger.info('Carregando e processando dados SST...')
    if len(sst_paths) == 1:
        ds = xr.open_dataset(
            str(sst_paths[0]),
            decode_times=True,
            chunks={'time': 20},
        )
    else:
        ds = xr.open_mfdataset(
            [str(p) for p in sst_paths],
            combine='by_coords',
            decode_times=True,
            chunks={'time': 20},
            coords='minimal',
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
        blue_marble_arr = np.array(Image.open(blue_marble_path))
    else:
        blue_marble_arr = None
        logger.warning(f'blue_marble.png nao encontrado em {input_dir} — usando fundo branco')

    # ---- Contexto compartilhado para todos os workers ----
    dt_ini = datetime.strptime(settings.DATA_INICIAL, '%Y-%m-%d').strftime('%d-%m-%y')
    dt_fim = datetime.strptime(settings.DATA_FINAL, '%Y-%m-%d').strftime('%d-%m-%y')
    # DynaBox nao e picklavel pelo multiprocessing — JSON garante tipos Python puros
    info_plot = json.loads(json.dumps(dict(settings['areas_plotagem'])))
    sst_levels = SST_MEAN_LEVELS
    sst_ticks = SST_MEAN_TICKS
    cmap_colors = SST_MEAN_COLORS

    output_dir.mkdir(parents=True, exist_ok=True)

    # Popula _G antes do fork: workers herdam via copy-on-write sem pickle
    global _G
    _G = {
        'average_data': average_data,
        'lon': lon_cyclic,
        'lat': lat_vals,
        'info_plot': info_plot,
        'output_dir': str(output_dir),
        'sst_levels': sst_levels,
        'sst_ticks': sst_ticks,
        'cmap_colors': cmap_colors,
        'blue_marble_arr': blue_marble_arr,
        'input_dir': str(input_dir),
        'dt_ini': dt_ini,
        'dt_fim': dt_fim,
        'lon_w': lon_w,
        'lat_w': lat_w,
        'u_mean': u_mean_w,
        'v_mean': v_mean_w,
        'olr_data': olr_arr,
        'lon_olr': lon_olr_cyc,
        'lat_olr': lat_olr_1d,
    }

    # ---- Plotagem sequencial ----
    logger.info(f'Plotando {len(lst_areas)} areas em sequencial...')

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
