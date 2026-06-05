# -*- coding: utf-8 -*-
"""
s05 - Anomalia de OLR (Outgoing Longwave Radiation).

Baixa dados de anomalia de OLR do PSL/NOAA (CPC Blended OLR),
calcula a media do periodo selecionado e gera mapas de anomalia
para diversas areas geograficas.

Dados de entrada:
    - PSL/NOAA: olr.day.anom.nc (CPC Blended OLR 2.5 graus)

Saida:
    - Mapas PNG em {settings.DIR_OUTPUT}/s05_OLR_ANOM/

Criado em: 2026-04-07
"""

# ---------------------------------------------------------------------------
# Bibliotecas padrao
# ---------------------------------------------------------------------------
import time
from datetime import datetime
from pathlib import Path

# Bibliotecas de terceiros
# ---------------------------------------------------------------------------
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
from cartopy.util import add_cyclic_point
from matplotlib import patches
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import FixedLocator, MultipleLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from PIL import Image

# Modulos locais
# ---------------------------------------------------------------------------
from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.common.dataset_utils import (
    arquivo_cobre_periodo,
    load_dataset,
    validar_cobertura_temporal,
)
from app.common.download_helper import DownloadEngine, download_with_progress
from app.shared.logger import get_logger
from app.shared.settings_factory import settings

# ---------------------------------------------------------------------------
# Identidade do script
# ---------------------------------------------------------------------------
SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's05'
SCRIPT_NAME = Path(__file__).stem
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
OLR_URL = 'https://downloads.psl.noaa.gov/Datasets/cpc_blended_olr-2.5deg/olr.day.anom.nc'
OLR_FILE_NAME = 'olr.day.anom.nc'

DEFAULT_AREAS = [
    'inicio_SMAS',
    'china',
    'pacifico_leste_america_sul',
    'america_sul_zom_out',
    'MDR',
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
    'enso',
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
def _get_area_list():
    """Retorna lista de areas: prioriza LST_AREAS_S05, fallback para DEFAULT_AREAS."""
    if hasattr(settings, 'LST_AREAS_S05'):
        return list(settings.LST_AREAS_S05)
    return list(DEFAULT_AREAS)


def _add_logo_to_map(ax, logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=500):
    """Adiciona logo ao mapa usando AnchoredOffsetbox (sobrevive a bbox_inches='tight')."""
    logo = Image.open(logo_path).convert('RGBA')
    bbox = logo.getbbox()
    if bbox is not None:
        logo = logo.crop(bbox)

    img = np.array(logo)
    imagebox = OffsetImage(img, zoom=zoom)

    ab = AnnotationBbox(
        imagebox,
        (0, 0),
        xycoords=ax.transAxes,
        xybox=(xoffset, yoffset),
        boxcoords='offset points',
        box_alignment=(0, 0),
        frameon=False,
        pad=0,
        zorder=zorder,
        clip_on=False,
    )

    ax.add_artist(ab)


def _configure_gridlines(gl, area):
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
# Main
# ---------------------------------------------------------------------------
def main():
    logger = get_logger(SCRIPT_ID)

    logger.info('=' * 80)
    logger.info('SCRIPT %s: %s', SCRIPT_ID.upper(), SCRIPT_DESC)
    logger.info('=' * 80)

    lst_areas = _get_area_list()

    output_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_OLR_ANOM'
    input_dir = Path(settings.DIR_INPUT)
    dados_dir = Path(settings.DIR_DADOS)

    cache_params = {
        'DATA_INICIAL': settings.DATA_INICIAL,
        'DATA_FINAL': settings.DATA_FINAL,
        'areas': lst_areas,
        'script_version': '1.1',
    }
    output_files = [str(output_dir / f'olr_anom_{area}.png') for area in lst_areas]

    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('CACHE VALIDO! Execucao ja foi realizada com os mesmos parametros.')
        logger.info('   Periodo: %s a %s', settings.DATA_INICIAL, settings.DATA_FINAL)
        logger.info('   %d mapas ja existem', len(output_files))
        logger.info('   Diretorio: %s', output_dir)
        logger.info('   Pulando execucao')
        return

    start_time = time.time()
    logger.info('Periodo de analise: %s a %s', settings.DATA_INICIAL, settings.DATA_FINAL)
    logger.info('Gerando %d mapas de anomalia OLR', len(lst_areas))
    logger.info('=' * 80)

    # ---- Download do arquivo OLR (PSL/NOAA) ----
    dados_dir.mkdir(parents=True, exist_ok=True)
    olr_path = dados_dir / OLR_FILE_NAME

    start_date = np.datetime64(settings.DATA_INICIAL)
    end_date = np.datetime64(settings.DATA_FINAL)

    # Re-baixa apenas se o arquivo local nao cobrir o periodo solicitado
    if arquivo_cobre_periodo(olr_path, start_date, end_date):
        logger.info('Arquivo OLR local ja cobre o periodo solicitado — pulando download')
    else:
        if olr_path.exists():
            logger.info('Arquivo OLR local nao cobre %s a %s — re-baixando',
                        settings.DATA_INICIAL, settings.DATA_FINAL)
        download_with_progress(
            url=OLR_URL,
            output_path=str(olr_path),
            description=OLR_FILE_NAME,
            max_retries=5,
            force=True,
            engine=DownloadEngine.ARIA2,  # 16 conexões paralelas: 4-8x mais rápido
            timeout=300,
        )

    # ---- Processamento ----
    ds = load_dataset(str(olr_path))

    # Aborta com mensagem clara se servidor PSL/NOAA tambem nao tiver o periodo
    validar_cobertura_temporal(ds, start_date, end_date, nome='arquivo OLR')

    subset = ds.sel(time=slice(start_date, end_date))
    ds_mean = subset.mean(dim='time')
    ds_mean['lon'] = ((ds_mean['lon'] + 180) % 360) - 180
    da = ds_mean.sortby(ds_mean.lon)['olr']
    da = mpcalc.smooth_gaussian(da, 5)

    lon = da['lon']
    lat = da['lat']
    average_data, lon = add_cyclic_point(da, coord=lon)

    # ---- Plotagem por area ----
    output_dir.mkdir(parents=True, exist_ok=True)
    info_plot = settings['areas_plotagem']

    for area in lst_areas:
        logger.info('Gerando mapa OLR para area: %s', area)

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(
            1,
            1,
            1,
            projection=ccrs.PlateCarree(
                central_longitude=info_plot[area]['central_longitude_mapa']
            ),
        )

        # Boxes configurados no settings.json
        if info_plot[area].get('plot_box', False):
            for box in info_plot[area]['lst_boxes']:
                rect = patches.Rectangle(
                    (box['x_anc'], box['y_anc']),
                    box['x_larg'],
                    box['y_larg'],
                    linewidth=box['linewidth'],
                    edgecolor=box['edgecolor'],
                    facecolor='none',
                    zorder=100,
                )
                ax.add_patch(rect)

        # Box MDR
        if area == 'MDR':
            lon_min_mdr, lon_max_mdr = -86, -20
            lat_min_mdr, lat_max_mdr = 10, 20
            ax.plot(
                [lon_min_mdr, lon_max_mdr, lon_max_mdr, lon_min_mdr, lon_min_mdr],
                [lat_min_mdr, lat_min_mdr, lat_max_mdr, lat_max_mdr, lat_min_mdr],
                color='black',
                linewidth=3,
                linestyle='-',
                zorder=500,
                transform=ccrs.PlateCarree(),
            )

        # Legenda e boxes Atlantico tropical
        if area == 'atlantico_tropical':
            legenda_atl = input_dir / 'legenda_atlantic.png'
            if legenda_atl.exists():
                img_legenda_atlantic = plt.imread(str(legenda_atl))
                fig.figimage(img_legenda_atlantic, 125, 614, zorder=3, alpha=1)

            box_tsa = patches.Rectangle(
                (10, -20), -40, 20,
                linewidth=3, edgecolor='black',
                facecolor='none', zorder=100,
            )
            box_tna = patches.Rectangle(
                (-15, 5), -40, 20,
                linewidth=3, edgecolor='blue',
                facecolor='none', zorder=100,
            )
            ax.add_patch(box_tsa)
            ax.add_patch(box_tna)

        # Boxes IOD
        if area == 'iod':
            box_iod_w = patches.Rectangle(
                (50, -10), 20, 20,
                linewidth=3, edgecolor='black',
                facecolor='none', zorder=100,
            )
            box_iod_e = patches.Rectangle(
                (90, -10), 20, 10,
                linewidth=3, edgecolor='black',
                facecolor='none', zorder=100,
            )
            ax.add_patch(box_iod_w)
            ax.add_patch(box_iod_e)

        # Labels ENSO
        if area == 'enso':
            for txt, x, y, cor in [
                ('Nino 1+2', 66.25, -13.64, 'red'),
                ('Nino 3', 34.1, 8.45, 'blue'),
                ('Nino 3.4', 8.6, -9.45, 'black'),
                ('Nino 4', -22.5, 8.45, 'm'),
            ]:
                t = plt.text(x, y, txt, fontsize=14, color=cor, weight='bold')
                fg = 'black' if cor in {'red', 'm'} else 'white'
                t.set_path_effects([
                    path_effects.Stroke(linewidth=3, foreground=fg),
                    path_effects.Normal(),
                ])

        # Fases MJO
        if area == 'mjo':
            for x, y, txt in [
                (-135.25, -4.64, '1'),
                (-115.25, -4.64, '2'),
                (-95.25, -4.64, '3'),
                (-75.25, -4.64, '4'),
                (-55.25, -4.64, '5'),
                (-35.25, -4.64, '6'),
                (-15.25, -4.64, '7'),
                (4.75, -4.64, '8'),
            ]:
                t = plt.text(
                    x, y, txt,
                    fontsize=50,
                    color='white',
                    weight='bold',
                    zorder=400,
                )
                t.set_path_effects([
                    path_effects.Stroke(linewidth=3, foreground='black'),
                    path_effects.Normal(),
                ])

        # Gridlines
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.0)
        _configure_gridlines(gl, area)

        # Limites
        ax.set_xlim([info_plot[area]['lon_esq'], info_plot[area]['lon_dir']])
        ax.set_ylim([info_plot[area]['lat_inf'], info_plot[area]['lat_sup']])

        # Features cartograficas
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, edgecolor='black')
        ax.add_feature(cfeature.LAND.with_scale('50m'), linewidth=0.5, facecolor='whitesmoke')

        if area != 'china':
            ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1.2, edgecolor='black', zorder=100)

        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, edgecolor='black', zorder=100)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, edgecolor='black', zorder=100)
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), linewidth=0.5, facecolor='white')

        # Contourf OLR
        im = ax.contourf(
            lon,
            lat,
            average_data,
            levels=np.arange(-40, 44, 4),
            cmap='BrBG_r',
            extend='both',
            transform=ccrs.PlateCarree(
                central_longitude=info_plot[area]['central_longitude_plot']
            ),
        )

        # Colorbar
        if area in {'enso', 'tropico', 'MDR', 'hemisferio_sul', 'psa'}:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)
            cbar = plt.colorbar(
                im,
                cax=cax,
                pad=0.02,
                fraction=0.02375,
                location='bottom',
                extend='both',
                orientation='horizontal',
                ticks=np.arange(-40, 50, 10),
            )
        else:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
            cbar = plt.colorbar(
                im,
                cax=cax,
                pad=0.02,
                fraction=0.02375,
                extend='both',
                ticks=np.arange(-40, 50, 10),
            )

        cbar.set_label(label='W/m$^2$', size=18)
        cbar.ax.tick_params(labelsize=20)

        # Titulo em portugues
        dt_ini = datetime.strptime(settings.DATA_INICIAL, '%Y-%m-%d').strftime('%d-%m-%y')
        dt_fim = datetime.strptime(settings.DATA_FINAL, '%Y-%m-%d').strftime('%d-%m-%y')
        titulo = (
            f'Anomalia de OLR (De {dt_ini} a {dt_fim})\n'
            f'Fonte: PSL/NOAA'
        )
        ax.set_title(titulo, fontsize=18, loc='left')

        # Logo
        logo_path = input_dir / 'novo_logo.png'
        if logo_path.exists():
            _add_logo_to_map(
                ax=ax,
                logo_path=logo_path,
                zoom=0.65,
                xoffset=0,
                yoffset=0,
                zorder=500,
            )

        # Salvar
        filename_fig = output_dir / f'olr_anom_{area}.png'
        logger.info('Salvando a figura %s', filename_fig)

        plt.savefig(
            str(filename_fig),
            dpi=fig.dpi,
            bbox_inches='tight',
        )
        plt.close('all')

    # Salvar cache
    execution_time = time.time() - start_time
    save_cache_metadata(SCRIPT_ID, cache_params, output_files, execution_time)

    logger.info('=' * 80)
    logger.info('Script %s concluido com sucesso!', SCRIPT_ID.upper())
    logger.info('Tempo de execucao: %.1fs (%.1f min)', execution_time, execution_time / 60)
    logger.info('%d mapas gerados em: %s', len(output_files), output_dir)
    logger.info('=' * 80)


if __name__ == '__main__':
    main()
