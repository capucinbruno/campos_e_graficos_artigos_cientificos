# -*- coding: utf-8 -*-
"""
s06 - Anomalia de TSM (Temperatura da Superficie do Mar).

Baixa dados de anomalia de TSM do PSL/NOAA (OISSTv2 High-Res),
calcula a media do periodo selecionado e gera mapas de anomalia SSTA
para diversas areas geograficas.

Dados de entrada:
    - PSL/NOAA: sst.day.anom.{ano}.nc (OISSTv2 0.25 grau, um arquivo por ano)

Saida:
    - Mapas PNG em {settings.DIR_OUTPUT}/s06_SSTA/

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

# ---------------------------------------------------------------------------
# Modulos locais
# ---------------------------------------------------------------------------
from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.common.dataset_utils import arquivo_cobre_periodo, load_dataset, validar_cobertura_temporal
from app.common.download_helper import DownloadEngine, download_with_progress
from app.shared.logger import get_logger
from app.shared.settings_factory import settings

# ---------------------------------------------------------------------------
# Identidade do script
# ---------------------------------------------------------------------------
SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's06'
SCRIPT_NAME = Path(__file__).stem
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
SST_URL_TEMPLATE = (
    'https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.day.anom.{year}.nc'
)
SST_FILE_TEMPLATE = 'sst.day.anom.{year}.nc'

# Contexto compartilhado com a funcao de plotagem (evita passar arrays grandes como argumento)
_G: dict = {}

DEFAULT_AREAS = [
    'mjo',
    'china',
    'pacifico_leste_america_sul',
    'brasil',
    'MDR',
    'america_sul_zom_out',
    'tropico',
    'psa',
    'hemisferio_sul',
    'globo',
    'enso',
    'zona_zcit_atlantico',
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
    'costa_brasil',
    'argentina',
    'estados_unidos_zoom',
    'estados_unidos',
]


# ---------------------------------------------------------------------------
# Funcoes utilitarias
# ---------------------------------------------------------------------------
def _get_area_list() -> list[str]:
    """Retorna lista de areas: prioriza LST_AREAS_S06, fallback para DEFAULT_AREAS."""
    if hasattr(settings, 'LST_AREAS_S06'):
        return list(settings.LST_AREAS_S06)
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
    """Gera e salva o mapa SSTA para uma area."""
    average_data = _G['average_data']
    lon = _G['lon']
    lat = _G['lat']
    info_plot = _G['info_plot']
    output_dir = Path(_G['output_dir'])
    sst_levels = _G['sst_levels']
    cmap_colors = _G['cmap_colors']
    blue_marble_arr = _G['blue_marble_arr']
    input_dir = Path(_G['input_dir'])
    dt_ini = _G['dt_ini']
    dt_fim = _G['dt_fim']

    cmap = LinearSegmentedColormap.from_list('sst_anom', cmap_colors)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(
        1, 1, 1,
        projection=ccrs.PlateCarree(
            central_longitude=info_plot[area]['central_longitude_mapa']
        ),
    )
    ax.set_frame_on(False)

    if blue_marble_arr is not None:
        ax.imshow(
            blue_marble_arr,
            origin='upper',
            extent=(-180, 180, -90, 90),
            transform=ccrs.PlateCarree(),
            interpolation='bilinear',
            zorder=0,
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
        ax.plot(
            [-86, -20, -20, -86, -86],
            [10, 10, 20, 20, 10],
            color='black', linewidth=3, linestyle='-', zorder=500,
            transform=ccrs.PlateCarree(),
        )

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
        for x, txt in [(-135.25, '1'), (-115.25, '2'), (-95.25, '3'), (-75.25, '4'),
                       (-55.25, '5'), (-35.25, '6'), (-15.25, '7'), (4.75, '8')]:
            t = plt.text(x, -4.64, txt, fontsize=50, color='white', weight='bold', zorder=400)
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

    # Features cartograficas — sem LAND(whitesmoke): blue marble aparece em terra
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

    # Colorbar
    if area in {'enso', 'tropico', 'MDR', 'hemisferio_sul', 'psa'}:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)
        cbar = plt.colorbar(
            im, cax=cax, pad=0.02, fraction=0.02375,
            location='bottom', extend='both', orientation='horizontal',
            ticks=sst_levels[::2],
        )
    else:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
        cbar = plt.colorbar(
            im, cax=cax, pad=0.02, fraction=0.02375,
            extend='both', ticks=sst_levels[::2],
        )

    cbar.set_label(label='°C', size=18)
    cbar.ax.tick_params(labelsize=10)

    # Titulo
    titulo = f'Anomalia de TSM (De {dt_ini} a {dt_fim})\nFonte: OISSTv2/NOAA'
    ax.set_title(titulo, fontsize=18, loc='left')

    # Logo
    logo_path = input_dir / 'novo_logo.png'
    if logo_path.exists():
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
    ax.add_patch(patches.Rectangle(
        (0, 0), 1, 1,
        linewidth=0.5, edgecolor='black', facecolor='none',
        transform=ax.transAxes, zorder=1000, clip_on=False,
    ))

    # Salvar
    filename_fig = output_dir / f'ssta_{area}.png'
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

    logger.info('Anos necessarios para o periodo: %s', years)
    paths = []

    for year in years:
        url = SST_URL_TEMPLATE.format(year=year)
        sst_path = dados_dir / SST_FILE_TEMPLATE.format(year=year)

        # Anos passados: so baixar se arquivo nao existe
        if year < current_year and sst_path.exists():
            logger.info('Arquivo SST %d ja existe localmente — pulando download', year)
            paths.append(sst_path)
            continue

        year_start_needed = np.datetime64(f'{year}-01-01', 'D')
        year_end_needed = np.datetime64(
            min(end_date, np.datetime64(f'{year}-12-31', 'D')), 'D'
        )

        if arquivo_cobre_periodo(sst_path, year_start_needed, year_end_needed):
            logger.info('Arquivo SST %d ja cobre o periodo ate %s — pulando download', year, year_end_needed)
            paths.append(sst_path)
            continue

        if sst_path.exists():
            logger.info('Arquivo SST %d nao cobre %s a %s — re-baixando', year, year_start_needed, year_end_needed)

        download_with_progress(
            url=url,
            output_path=str(sst_path),
            description=f'SST anomalia {year}',
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
    logger.info('SCRIPT %s: %s', SCRIPT_ID.upper(), SCRIPT_DESC)
    logger.info('=' * 80)

    lst_areas = _get_area_list()

    output_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_SSTA'
    input_dir = Path(settings.DIR_INPUT)
    dados_dir = Path(settings.DIR_DADOS)

    cache_params = {
        'DATA_INICIAL': settings.DATA_INICIAL,
        'DATA_FINAL': settings.DATA_FINAL,
        'areas': lst_areas,
        'script_version': '1.1',
    }
    output_files = [str(output_dir / f'ssta_{area}.png') for area in lst_areas]

    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('CACHE VALIDO! Execucao ja foi realizada com os mesmos parametros.')
        logger.info('   Periodo: %s a %s', settings.DATA_INICIAL, settings.DATA_FINAL)
        logger.info('   %d mapas ja existem', len(output_files))
        logger.info('   Diretorio: %s', output_dir)
        logger.info('   Pulando execucao')
        return

    start_time = time.time()
    logger.info('Periodo de analise: %s a %s', settings.DATA_INICIAL, settings.DATA_FINAL)
    logger.info('Gerando %d mapas de anomalia SSTA', len(lst_areas))
    logger.info('=' * 80)

    start_date = np.datetime64(settings.DATA_INICIAL, 'D')
    end_date = np.datetime64(settings.DATA_FINAL, 'D')

    # ---- Download dos arquivos SST anuais ----
    dados_dir.mkdir(parents=True, exist_ok=True)
    sst_paths = _download_sst_anos(dados_dir, start_date, end_date, logger)

    # ---- Carregamento e concatenacao ----
    logger.info('Carregando e processando dados SST...')
    if len(sst_paths) == 1:
        ds = load_dataset(
            str(sst_paths[0]),
            adjust_lon=False,
            time_slice=slice(str(start_date), str(end_date)),
        )
    else:
        ds_list = [load_dataset(str(p), adjust_lon=False) for p in sst_paths]
        ds = xr.concat(ds_list, dim='time').sortby('time')

    validar_cobertura_temporal(ds, start_date, end_date, nome='SST OISSTv2')

    subset = ds.sel(time=slice(start_date, end_date))
    ds_mean = subset.mean(dim='time')
    ds_mean['lon'] = ((ds_mean['lon'] + 180) % 360) - 180
    da = ds_mean.sortby(ds_mean.lon)['anom']

    lon_vals = da['lon'].values
    lat_vals = da['lat'].values
    average_data, lon_cyclic = add_cyclic_point(da.values, coord=lon_vals)

    # ---- Blue marble ----
    blue_marble_path = input_dir / 'blue_marble.png'
    if blue_marble_path.exists():
        blue_marble_arr = np.array(Image.open(blue_marble_path))
    else:
        blue_marble_arr = None
        logger.warning('blue_marble.png nao encontrado em %s — usando fundo branco', input_dir)

    # ---- Contexto compartilhado para todos os workers ----
    dt_ini = datetime.strptime(settings.DATA_INICIAL, '%Y-%m-%d').strftime('%d-%m-%y')
    dt_fim = datetime.strptime(settings.DATA_FINAL, '%Y-%m-%d').strftime('%d-%m-%y')
    # DynaBox nao e picklavel pelo multiprocessing — JSON garante tipos Python puros
    info_plot = json.loads(json.dumps(dict(settings['areas_plotagem'])))
    sst_levels = [float(x) for x in settings.LST_SSTA_NEW_GREC]
    cmap_colors = [str(x) for x in settings.LST_ANOM_CORRETA]

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
        'cmap_colors': cmap_colors,
        'blue_marble_arr': blue_marble_arr,
        'input_dir': str(input_dir),
        'dt_ini': dt_ini,
        'dt_fim': dt_fim,
    }

    # ---- Plotagem sequencial ----
    logger.info(f'Plotando {len(lst_areas)} areas em sequencial...')

    concluidos = 0
    falhas = []

    for area in lst_areas:
        try:
            _plot_area_worker(area)
            concluidos += 1
            logger.info(f'[{concluidos}/{len(lst_areas)}] Mapa salvo: {area}')
        except Exception as exc:
            falhas.append(area)
            logger.error(f'Falha ao gerar mapa para area {area}: {type(exc).__name__}: {exc}')

    if falhas:
        raise RuntimeError(f'Falha ao gerar {len(falhas)} mapa(s): {falhas}')

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
