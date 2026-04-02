"""s01 - Anomalia de Geopotencial 250 hPa (ERA5).

Baixa dados de altura geopotencial em 250 hPa da reanalise ERA5 via Copernicus CDS,
calcula a anomalia em relacao a media climatologica do periodo e gera mapas de anomalia
para diversas regioes geograficas configuradas em settings.json.

Dados de entrada:
    - ERA5 (CDS): geopotencial em 250 hPa
    - Legenda do mapa Atlantico (arquivo fixo em Entrada/)

Saida:
    - Mapas PNG em Saida/ (um por regiao)

Criado em:     2026-02-23
Atualizado em: 2026-03-18
"""

# Bibliotecas padrão
import time
from datetime import datetime
from pathlib import Path

# Bibliotecas de terceiros
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from cartopy.util import add_cyclic_point
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import FixedLocator, MultipleLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from PIL import Image

# Módulos locais
from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.common.dataset_utils import load_dataset
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.src.uteis.plot_geop250 import main as plot_geop250

# ---------------------------------------------------------------------------
# Identidade do script (derivada do nome do arquivo)
# ---------------------------------------------------------------------------
SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's01'
SCRIPT_NAME = Path(__file__).stem  # 's01_geop250_anom'
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME


def _add_logo_to_map(ax, logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=500):
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


def main():
    logger = get_logger(SCRIPT_ID)

    logger.info('=' * 80)
    logger.info('📊 SCRIPT %s: %s', SCRIPT_ID.upper(), SCRIPT_DESC)
    logger.info('=' * 80)

    # Lista de áreas vem do settings (LST_AREAS_S01 no settings.toml)
    lst_areas = list(settings.LST_AREAS_S01)

    # Diretórios via settings
    output_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_GEOP250'
    input_dir = Path(settings.DIR_INPUT)
    dados_dir = Path(settings.DIR_DADOS)

    # Verificação de cache
    cache_params = {
        'DATA_INICIAL': settings.DATA_INICIAL,
        'DATA_FINAL': settings.DATA_FINAL,
        'areas': lst_areas,
        'script_version': '2.2',  # Incrementado: fix cyclic_point (faixa branca)
    }
    output_files = [str(output_dir / f'geop250_{area}.png') for area in lst_areas]

    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('🎯 CACHE VÁLIDO! Execução já foi realizada com os mesmos parâmetros.')
        logger.info('   📅 Período: %s a %s', settings.DATA_INICIAL, settings.DATA_FINAL)
        logger.info('   📊 %d mapas já existem', len(output_files))
        logger.info('   📁 Diretório: %s', output_dir)
        logger.info('   ⏭️  Pulando execução')
        return

    start_time = time.time()
    logger.info('📅 Período de análise: %s a %s', settings.DATA_INICIAL, settings.DATA_FINAL)
    logger.info('📊 Gerando %d mapas de anomalia GEOP250', len(lst_areas))
    logger.info('=' * 80)
    try:
        plot_geop250()
    except Exception as err:
        logger.exception('Falha no download/processamento do geopotencial 250hPa')
        raise RuntimeError('Falha no download do geopotencial 250hPa') from err

    output_dir.mkdir(parents=True, exist_ok=True)

    ds1 = load_dataset(str(dados_dir / 'geop250.nc'))

    ds1 = ds1.assign_coords(lon=((ds1.lon + 360) % 360))
    da = ds1.sortby(ds1.lon)['hgt'].isel(time=0)
    lon = da['lon']
    lat = da['lat']
    hgt, lon = add_cyclic_point(da, coord=lon)


    if np.all((hgt >= -50) & (hgt <= 50)):
        levels = np.arange(-50, 53, 3)
        ticks = np.arange(-50, 55, 5)
    elif np.all((hgt >= -100) & (hgt <= 100)):
        levels = np.arange(-100, 120, 20)
        ticks = np.arange(-100, 150, 50)
    else:
        levels = np.arange(-200, 220, 20)
        ticks = np.arange(-200, 250, 50)

    # lst_areas já definido no início para cache
    for area in lst_areas:
        info_plot = settings['areas_plotagem']
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(
            1,
            1,
            1,
            projection=ccrs.PlateCarree(central_longitude=info_plot[area]['central_longitude_mapa']),
        )
        if info_plot[area]['plot_box']:
            lst_boxes = info_plot[area]['lst_boxes']
            for box in lst_boxes:
                box = patches.Rectangle(
                    (box['x_anc'], box['y_anc']),
                    box['x_larg'],
                    box['y_larg'],
                    linewidth=box['linewidth'],
                    edgecolor=box['edgecolor'],
                    facecolor='none',
                    zorder=100,
                )

                ax.add_patch(box)

        if area == 'MDR':
            lon_min_MDR, lon_max_MDR = -86, -20
            lat_min_MDR, lat_max_MDR = 10, 20

            ax.plot(
                [lon_min_MDR, lon_max_MDR, lon_max_MDR, lon_min_MDR, lon_min_MDR],
                [lat_min_MDR, lat_min_MDR, lat_max_MDR, lat_max_MDR, lat_min_MDR],
                color='black',
                linewidth=3,
                linestyle='-',
                zorder=500,
                transform=ccrs.PlateCarree(),
            )

        if area == 'atlantico_tropical':
            img_legenda_atlantic = plt.imread(str(input_dir / 'legenda_atlantic.png'))
            fig.figimage(
                img_legenda_atlantic,
                125,
                614,
                zorder=3,
                alpha=1,
            )

            # Box TSA
            box1 = patches.Rectangle(
                (10, -20),
                -40,
                20,
                linewidth=3,
                edgecolor='black',
                facecolor='none',
                zorder=100,
            )  # (posição x, posição y de ancoragem) largura

            # Box TNA
            box2 = patches.Rectangle(
                (-15, 5),
                -40,
                20,
                linewidth=3,
                edgecolor='blue',
                facecolor='none',
                zorder=100,
            )  # (posição x, posição y de ancoragem) largura

            # Add the patch to the Axes
            ax.add_patch(box1)
            ax.add_patch(box2)

        if area == 'iod':
            # Box Indico Ocidental
            box3 = patches.Rectangle(
                (50, -10),
                20,
                20,
                linewidth=3,
                edgecolor='black',
                facecolor='none',
                zorder=100,
            )  # (posição x, posição y de ancoragem) largura

            # Box Indico Oriental (Indonésia)
            box4 = patches.Rectangle(
                (90, -10),
                20,
                10,
                linewidth=3,
                edgecolor='black',
                facecolor='none',
                zorder=100,
            )  # (posição x, posição y de ancoragem) largura

            # Add the patch to the Axes
            ax.add_patch(box3)
            ax.add_patch(box4)

        if area == 'enso':
            text1 = plt.text(66.25, -13.64, 'Niño 1+2', fontsize=14, color='red', weight='bold', zorder=500)
            text1.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground='black'),
                path_effects.Normal(),
            ])

            text2 = plt.text(
                34.1,
                8.45,
                'Niño 3',
                fontsize=14,
                color='blue',
                weight='bold',
                zorder=500,
                path_effects=[path_effects.withSimplePatchShadow()],
            )
            text2.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground='white'),
                path_effects.Normal(),
            ])

            text3 = plt.text(
                8.6,
                -9.45,
                'Niño 3.4',
                fontsize=14,
                color='black',
                weight='bold',
                zorder=500,
                path_effects=[path_effects.withSimplePatchShadow()],
            )
            text3.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground='white'),
                path_effects.Normal(),
            ])

            text4 = plt.text(
                -22.5,
                8.45,
                'Niño 4',
                fontsize=14,
                color='m',
                weight='bold',
                zorder=500,
                path_effects=[path_effects.withSimplePatchShadow()],
            )
            text4.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground='black'),
                path_effects.Normal(),
            ])

        if area == 'mjo':
            text5 = plt.text(
                -135.25,
                -4.64,
                '1',
                fontsize=50,
                color='white',
                weight='bold',
                zorder=400,
            )
            text5.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground='black'),
                path_effects.Normal(),
            ])
            text6 = plt.text(
                -115.25,
                -4.64,
                '2',
                fontsize=50,
                color='white',
                weight='bold',
                zorder=400,
            )
            text6.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground='black'),
                path_effects.Normal(),
            ])
            text7 = plt.text(
                -95.25,
                -4.64,
                '3',
                fontsize=50,
                color='white',
                weight='bold',
                zorder=400,
            )
            text7.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground='black'),
                path_effects.Normal(),
            ])
            text8 = plt.text(
                -75.25,
                -4.64,
                '4',
                fontsize=50,
                color='white',
                weight='bold',
                zorder=400,
            )
            text8.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground='black'),
                path_effects.Normal(),
            ])
            text9 = plt.text(
                -55.25,
                -4.64,
                '5',
                fontsize=50,
                color='white',
                weight='bold',
                zorder=400,
            )
            text9.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground='black'),
                path_effects.Normal(),
            ])
            text10 = plt.text(
                -35.25,
                -4.64,
                '6',
                fontsize=50,
                color='white',
                weight='bold',
                zorder=400,
            )
            text10.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground='black'),
                path_effects.Normal(),
            ])
            text11 = plt.text(
                -15.25,
                -4.64,
                '7',
                fontsize=50,
                color='white',
                weight='bold',
                zorder=400,
            )
            text11.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground='black'),
                path_effects.Normal(),
            ])
            text12 = plt.text(
                4.75,
                -4.64,
                '8',
                fontsize=50,
                color='white',
                weight='bold',
                zorder=400,
            )
            text12.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground='black'),
                path_effects.Normal(),
            ])

        # Adicionar linhas de grade (latitude e longitude)
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.0)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 20, 'color': 'black'}
        gl.ylabel_style = {'size': 20, 'color': 'black'}

        # Configuração específica para "hemisferio_sul"
        if area == 'hemisferio_sul':
            gl.xlocator = MultipleLocator(40)
            gl.ylocator = MultipleLocator(20)

        elif area == 'psa':
            gl.xlocator = MultipleLocator(40)
            gl.ylocator = MultipleLocator(20)

        elif area == 'estados_unidos':
            gl.xlocator = MultipleLocator(10)
            gl.ylocator = MultipleLocator(10)

        elif area == 'estados_unidos_zoom':
            gl.xlocator = MultipleLocator(10)
            gl.ylocator = MultipleLocator(5)

        elif area == 'argentina':
            gl.xlocator = MultipleLocator(5)
            gl.ylocator = MultipleLocator(5)

        elif area == 'brasil':
            gl.xlocator = MultipleLocator(10)
            gl.ylocator = MultipleLocator(5)

        elif area == 'america_sul':
            gl.xlocator = MultipleLocator(20)
            gl.ylocator = MultipleLocator(20)

        elif area == 'zona_zcit_atlantico':
            gl.xlocator = MultipleLocator(10)
            gl.ylocator = MultipleLocator(5)

        elif area == 'costa_brasil':
            gl.xlocator = MultipleLocator(5)
            gl.ylocator = MultipleLocator(5)

        elif area == 'enso':
            # Definindo manualmente os valores de longitude para aparecer nos rótulos
            longitudes = [
                -160,
                -140,
                -120,
                -100,
                -80,
                -60,
                0,
                20,
                40,
                60,
                80,
                100,
                120,
                140,
                150,
                160,
                170,
                180,
            ]
            gl.xlocator = FixedLocator(longitudes)
            gl.ylocator = MultipleLocator(10)
            gl.xlabel_style = {'size': 15, 'color': 'black'}
            gl.ylabel_style = {'size': 15, 'color': 'black'}

        elif area == 'atlantico_tropical':
            gl.xlocator = MultipleLocator(20)
            gl.ylocator = MultipleLocator(10)

        elif area == 'tsa':
            gl.xlocator = MultipleLocator(10)
            gl.ylocator = MultipleLocator(5)

        elif area == 'tna':
            gl.xlocator = MultipleLocator(10)
            gl.ylocator = MultipleLocator(5)

        elif area == 'globo':
            gl.xlocator = MultipleLocator(40)
            gl.ylocator = MultipleLocator(20)
            gl.xlabel_style = {'size': 15, 'color': 'black'}
            gl.ylabel_style = {'size': 15, 'color': 'black'}

        elif area == 'pdo':
            gl.xlocator = MultipleLocator(20)
            gl.ylocator = MultipleLocator(10)

        elif area == 'iod':
            gl.xlocator = MultipleLocator(20)
            gl.ylocator = MultipleLocator(10)

        elif area == 'sad':
            gl.xlocator = MultipleLocator(20)
            gl.ylocator = MultipleLocator(10)

        elif area == 'amo':
            gl.xlocator = MultipleLocator(20)
            gl.ylocator = MultipleLocator(10)

        elif area == 'mjo':
            gl.xlocator = MultipleLocator(40)
            gl.ylocator = MultipleLocator(20)

        elif area == 'africa':
            gl.xlocator = MultipleLocator(20)
            gl.ylocator = MultipleLocator(20)

        elif area == 'africa_monsoon':
            gl.xlocator = MultipleLocator(20)
            gl.ylocator = MultipleLocator(10)

        elif area == 'tropico':
            longitudes = [-160, -120, -80, -40, 0, 40, 80, 120, 160]
            gl.xlocator = FixedLocator(longitudes)
            gl.ylocator = MultipleLocator(20)
            gl.xlabel_style = {'size': 15, 'color': 'black'}
            gl.ylabel_style = {'size': 15, 'color': 'black'}

        # Limites da area de plotagem
        ax.set_xlim([info_plot[area]['lon_esq'], info_plot[area]['lon_dir']])
        ax.set_ylim([info_plot[area]['lat_inf'], info_plot[area]['lat_sup']])

        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2)
        ax.add_feature(cfeature.LAND.with_scale('50m'), linewidth=0.5, facecolor='whitesmoke')

        if area != 'china':
            ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1.2, zorder=100)

        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, zorder=100)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, zorder=100)
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), linewidth=0.5, facecolor='white')

        cmap_name = settings.LST_ANOM_CORRETA
        cmap = LinearSegmentedColormap.from_list(cmap_name, settings.LST_ANOM_CORRETA)

        im = ax.contourf(
            lon,
            lat,
            hgt,
            levels=levels,
            # cmap="RdYlBu_r",
            cmap=cmap,
            extend='both',
            transform=ccrs.PlateCarree(central_longitude=info_plot[area]['central_longitude_plot']),
        )

        if (
            area == 'enso'
            or area == 'tropico'
            or area == 'MDR'
            or area == 'hemisferio_sul'
            or area == 'psa'
        ):
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
                # ticks=settings.LST_LEVELS_SST,
                ticks=ticks,
            )

        else:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)

            cbar = plt.colorbar(im, cax=cax, pad=0.02, fraction=0.02375, extend='both', ticks=ticks)

        cbar.set_label(label='mgp', size=18)
        cbar.ax.tick_params(labelsize=20)

        im = ax.contour(
            lon,
            lat,
            hgt,
            # levels=levels,
            levels=levels,
            colors='white',
            linewidths=0.6,
            transform=ccrs.PlateCarree(central_longitude=info_plot[area]['central_longitude_plot']),
        )
        

        # Adicionar um título com quebra de linha e variáveis
        dt_ini = datetime.strptime(settings.DATA_INICIAL, '%Y-%m-%d').strftime('%d-%m-%y')
        dt_fim = datetime.strptime(settings.DATA_FINAL, '%Y-%m-%d').strftime('%d-%m-%y')
        titulo = f'Anomalia de Alt. Geopotencial 250mb (De {dt_ini} a {dt_fim})'
        ax.set_title(titulo, fontsize=18, loc='left')

        # Logo — canto inferior esquerdo do frame do mapa
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

        filename_fig = output_dir / f'geop250_{area}.png'
        logger.info('Salvando a figura %s', filename_fig)

        plt.savefig(
            str(filename_fig),
            dpi=fig.dpi,
            bbox_inches='tight',
        )
        plt.close('all')

    # Salvar metadados do cache após execução bem-sucedida
    execution_time = time.time() - start_time
    save_cache_metadata(SCRIPT_ID, cache_params, output_files, execution_time)
    logger.info('=' * 80)
    logger.info('✅ Script %s concluído com sucesso!', SCRIPT_ID.upper())
    logger.info('⏱️  Tempo de execução: %.1fs (%.1f min)', execution_time, execution_time / 60)
    logger.info('📊 %d mapas gerados em: %s', len(output_files), output_dir)
    logger.info('=' * 80)


if __name__ == '__main__':
    main()
