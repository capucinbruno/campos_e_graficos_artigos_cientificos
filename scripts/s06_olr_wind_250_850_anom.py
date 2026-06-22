# -*- coding: utf-8 -*-
"""
s06 - Anomalia de OLR + Linhas de Corrente do Vento Anomalo em 250 e 850 hPa.

Baixa dados de anomalia de OLR do PSL/NOAA e dados de vento 250/850 hPa via ERA5/GDAS,
calcula anomalias usando climatologia PSL, e gera mapas combinados de anomalia OLR
(contourf) com linhas de corrente da anomalia do vento (streamplot) para cada nível.

Dados de entrada:
    - PSL/NOAA: olr.day.anom.nc (CPC Blended OLR 2.5 graus)
    - ERA5/GDAS: u/v 250 hPa e 850 hPa (híbrido por latência)
    - Climatologias: PSL u/v 250mb e 850mb (download automático via Playwright)

Saida:
    - Mapas PNG em {settings.DIR_OUTPUT}/s06_OLR_WIND_250_850/
        olr_wind_anom_{area}_250hPa.png
        olr_wind_anom_{area}_850hPa.png

Criado em: 2026-04-07
Atualizado em: 2026-06-04
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
import matplotlib.path as mpath
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
    area_display_name,
    arquivo_cobre_periodo,
    load_dataset,
    validar_cobertura_temporal,
)
from app.common.download_helper import DownloadEngine, download_with_progress
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.common.logo_helper import resolve_logo_path
from app.common.logo_helper import proportional_logo_zoom
from app.src.uteis.plot_olr_wind250_anom import main as plot_wind250_anom
from app.src.uteis.plot_olr_wind850_anom import main as plot_wind850_anom

# ---------------------------------------------------------------------------
# Identidade do script
# ---------------------------------------------------------------------------
SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's06'
SCRIPT_NAME = Path(__file__).stem
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
OLR_URL = 'https://downloads.psl.noaa.gov/Datasets/cpc_blended_olr-2.5deg/olr.day.anom.nc'
OLR_FILE_NAME = 'olr.day.anom.nc'
WIND250_FILE_NAME = 'wind250_anom.nc'
WIND850_FILE_NAME = 'wind850_anom.nc'

DEFAULT_AREAS = [
    'pacific_chile',
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
    'globo_3d',
]

# Configuracao de streamlines por area (mesmo padrao do s03/PSI200)
STREAMLINE_DEFAULTS = {
    'density': 2,
    'linewidth': 0.8,
    'arrowsize': 1.0,
    'color': 'dimgray',
}

STREAMLINE_POR_AREA = {
    'brasil': {'density': 2, 'linewidth': 0.5, 'arrowsize': 1.2},
    'america_sul': {'density': 2, 'linewidth': 0.5},
    'america_sul_zom_out': {'density': 2.5, 'linewidth': 0.5},
    'argentina': {'density': 2, 'linewidth': 0.5, 'arrowsize': 1.2},
    'costa_brasil': {'density': 2, 'linewidth': 0.5},
    'hemisferio_sul': {'density': 3, 'linewidth': 0.5},
    'psa': {'density': 3, 'linewidth': 0.5},
    'globo': {'density': 3, 'linewidth': 0.5},
    'tropico': {'density': 3, 'linewidth': 0.5},
    'enso': {'density': 2.5, 'linewidth': 0.5},
    'mjo': {'density': 2, 'linewidth': 0.5},
    'atlantico_tropical': {'density': 3, 'linewidth': 0.5},
    'africa': {'density': 2, 'linewidth': 0.5},
    'africa_monsoon': {'density': 2, 'linewidth': 0.5},
    'china': {'density': 3, 'linewidth': 0.5},
    'estados_unidos': {'density': 2, 'linewidth': 0.5},
    'estados_unidos_zoom': {'density': 2, 'linewidth': 0.5},
    'tsa': {'density': 2, 'linewidth': 0.5},
    'tna': {'density': 2, 'linewidth': 0.5},
    'iod': {'density': 2, 'linewidth': 0.5},
    'pdo': {'density': 2.5, 'linewidth': 0.5},
    'sad': {'density': 2, 'linewidth': 0.5},
    'amo': {'density': 3, 'linewidth': 0.5},
    'MDR': {'density': 3, 'linewidth': 0.5},
    'pacifico_leste_america_sul': {'density': 2.5, 'linewidth': 0.5},
    'zona_zcit_atlantico': {'density': 2, 'linewidth': 0.5},
}


# ---------------------------------------------------------------------------
# Funcoes utilitarias
# ---------------------------------------------------------------------------
def _get_area_list():
    """Retorna lista de areas: prioriza LST_AREAS_S06, fallback para DEFAULT_AREAS."""
    if hasattr(settings, 'LST_AREAS_S06'):
        return list(settings.LST_AREAS_S06)
    return list(DEFAULT_AREAS)


def _get_streamline_config(area: str) -> dict:
    """Retorna config de streamlines para a area."""
    cfg = dict(STREAMLINE_DEFAULTS)
    if area in STREAMLINE_POR_AREA:
        cfg.update(STREAMLINE_POR_AREA[area])
    return cfg


def _add_logo_to_map(ax, logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=500):
    """Adiciona logo ao mapa usando AnchoredOffsetbox (sobrevive a bbox_inches='tight')."""
    logo = Image.open(logo_path).convert('RGBA')
    bbox = logo.getbbox()
    if bbox is not None:
        logo = logo.crop(bbox)

    img = np.array(logo)
    imagebox = OffsetImage(img, zoom=proportional_logo_zoom(ax, img.shape[1]))

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


def _prepare_streamline_lonuv(lon_cyc, u_cyc, v_cyc, central_lon_mapa):
    """Desloca lon/u/v para que a costura do array fique na borda do mapa.

    Mesmo padrao do s03 (PSI200).
    """
    shifted = lon_cyc - central_lon_mapa
    shifted = (shifted + 180) % 360 - 180
    order = np.argsort(shifted)
    lon_sorted = shifted[order]
    u_sorted = u_cyc[:, order]
    v_sorted = v_cyc[:, order]

    # Remove ponto duplicado na borda (vem do add_cyclic_point)
    if len(lon_sorted) > 1 and np.isclose(lon_sorted[0], lon_sorted[1]):
        lon_sorted = lon_sorted[1:]
        u_sorted = u_sorted[:, 1:]
        v_sorted = v_sorted[:, 1:]
    elif len(lon_sorted) > 1 and np.isclose(lon_sorted[-1], lon_sorted[-2]):
        lon_sorted = lon_sorted[:-1]
        u_sorted = u_sorted[:, :-1]
        v_sorted = v_sorted[:, :-1]

    return lon_sorted, u_sorted, v_sorted


# ---------------------------------------------------------------------------
# Plotagem por nível de vento (250 ou 850 hPa)
# ---------------------------------------------------------------------------
def _gerar_figuras_nivel(
    lst_areas, olr_data, lat_olr, lon_olr,
    u_cyc, v_cyc, lat_wind, lon_wind_cyc,
    nivel_hpa: int, output_dir, input_dir, logger,
):
    """Gera figuras OLR + streamlines de vento para um dado nível de pressão."""
    info_plot = settings['areas_plotagem']

    for area in lst_areas:
        logger.info(f'Gerando mapa OLR + vento {nivel_hpa} hPa para area: {area_display_name(area)}')

        is_polar = info_plot[area].get('projection', '') == 'orthographic_south'
        if is_polar:
            proj = ccrs.Orthographic(
                central_longitude=settings.get('ORTHO_CENTRAL_LONGITUDE', info_plot[area].get('ortho_central_longitude', -71)),
                central_latitude=settings.get('ORTHO_CENTRAL_LATITUDE', info_plot[area].get('ortho_central_latitude', -84)),
            )
        else:
            proj = ccrs.PlateCarree(
                central_longitude=info_plot[area]['central_longitude_mapa']
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
                from matplotlib import patches as _patches
                rect = _patches.Rectangle(
                    (box['x_anc'], box['y_anc']), box['x_larg'], box['y_larg'],
                    linewidth=box['linewidth'], edgecolor=box['edgecolor'],
                    facecolor='none', zorder=100,
                )
                ax.add_patch(rect)

        if area == 'MDR':
            ax.plot(
                [-86, -20, -20, -86, -86], [10, 10, 20, 20, 10],
                color='black', linewidth=3, linestyle='-', zorder=500,
                transform=ccrs.PlateCarree(),
            )

        if area == 'atlantico_tropical':
            legenda_atl = input_dir / 'legenda_atlantic.png'
            if legenda_atl.exists():
                img_legenda_atlantic = plt.imread(str(legenda_atl))
                fig.figimage(img_legenda_atlantic, 125, 614, zorder=3, alpha=1)
            from matplotlib import patches as _patches
            ax.add_patch(_patches.Rectangle((10, -20), -40, 20, linewidth=3, edgecolor='black', facecolor='none', zorder=100))
            ax.add_patch(_patches.Rectangle((-15, 5), -40, 20, linewidth=3, edgecolor='blue', facecolor='none', zorder=100))

        if area == 'iod':
            from matplotlib import patches as _patches
            ax.add_patch(_patches.Rectangle((50, -10), 20, 20, linewidth=3, edgecolor='black', facecolor='none', zorder=100))
            ax.add_patch(_patches.Rectangle((90, -10), 20, 10, linewidth=3, edgecolor='black', facecolor='none', zorder=100))

        if area == 'enso':
            for txt, x, y, cor in [
                ('Nino 1+2', 66.25, -13.64, 'red'), ('Nino 3', 34.1, 8.45, 'blue'),
                ('Nino 3.4', 8.6, -9.45, 'black'), ('Nino 4', -22.5, 8.45, 'm'),
            ]:
                t = plt.text(x, y, txt, fontsize=14, color=cor, weight='bold')
                fg = 'black' if cor in {'red', 'm'} else 'white'
                t.set_path_effects([path_effects.Stroke(linewidth=3, foreground=fg), path_effects.Normal()])

        if area == 'mjo':
            for x, y, txt in [
                (-135.25, -4.64, '1'), (-115.25, -4.64, '2'), (-95.25, -4.64, '3'),
                (-75.25, -4.64, '4'), (-55.25, -4.64, '5'), (-35.25, -4.64, '6'),
                (-15.25, -4.64, '7'), (4.75, -4.64, '8'),
            ]:
                t = plt.text(x, y, txt, fontsize=50, color='white', weight='bold', zorder=400)
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

        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, edgecolor='black')
        ax.add_feature(cfeature.LAND.with_scale('50m'), linewidth=0.5, facecolor='whitesmoke')
        if area != 'china':
            ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1.2, edgecolor='black', zorder=100)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, edgecolor='black', zorder=100)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, edgecolor='black', zorder=100)
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), linewidth=0.5, facecolor='white')

        im = ax.contourf(
            lon_olr, lat_olr, olr_data,
            levels=np.arange(-40, 44, 4), cmap='BrBG_r', extend='both',
            transform=ccrs.PlateCarree(central_longitude=info_plot[area]['central_longitude_plot']),
            zorder=2,
        )

        slcfg = _get_streamline_config(area)
        central_lon = info_plot[area]['central_longitude_mapa']
        lon_sl, u_sl, v_sl = _prepare_streamline_lonuv(lon_wind_cyc, u_cyc, v_cyc, central_lon)

        ax.streamplot(
            lon_sl, lat_wind, u_sl, v_sl,
            transform=ccrs.PlateCarree(central_longitude=central_lon),
            density=float(slcfg['density']),
            linewidth=float(slcfg['linewidth']),
            arrowsize=float(slcfg.get('arrowsize', 1.0)),
            color=slcfg['color'],
            zorder=5,
        )

        if is_polar and area != 'globo_3d':
            cbar = plt.colorbar(im, ax=ax, pad=0.05, fraction=0.04, ticks=np.arange(-40, 50, 10))
            cbar.set_label(label='W/m$^2$', size=10)
            cbar.ax.tick_params(labelsize=10)
        elif area in {'enso', 'tropico', 'MDR', 'hemisferio_sul', 'psa'}:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)
            cbar = plt.colorbar(im, cax=cax, pad=0.02, fraction=0.02375, location='bottom',
                                extend='both', orientation='horizontal', ticks=np.arange(-40, 50, 10))
            cbar.set_label(label='W/m$^2$', size=18)
            cbar.ax.tick_params(labelsize=20)
        else:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
            cbar = plt.colorbar(im, cax=cax, pad=0.02, fraction=0.02375, extend='both', ticks=np.arange(-40, 50, 10))
            cbar.set_label(label='W/m$^2$', size=18)
            cbar.ax.tick_params(labelsize=20)

        dt_ini = datetime.strptime(settings.DATA_INICIAL, '%Y-%m-%d').strftime('%d-%m-%y')
        dt_fim = datetime.strptime(settings.DATA_FINAL, '%Y-%m-%d').strftime('%d-%m-%y')
        titulo = f'Anomalia de OLR e do Vento em {nivel_hpa} hPa (De {dt_ini} a {dt_fim})'
        ax.set_title(titulo, fontsize=14 if is_polar else 18, loc='left')

        logo_path = resolve_logo_path(input_dir)
        if logo_path is not None and logo_path.exists():
            _add_logo_to_map(ax=ax, logo_path=logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=500)

        filename_fig = output_dir / f'olr_wind_anom_{area}_{nivel_hpa}hPa.png'
        logger.info(f'Salvando a figura {filename_fig}')
        plt.savefig(str(filename_fig), dpi=fig.dpi, bbox_inches='tight')
        plt.close('all')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger = get_logger(SCRIPT_ID)

    logger.info('=' * 80)
    logger.info(f'SCRIPT {SCRIPT_ID.upper()}: {SCRIPT_DESC}')
    logger.info('=' * 80)

    lst_areas = _get_area_list()

    output_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_OLR_WIND_250_850'
    input_dir = Path(settings.DIR_INPUT)
    dados_dir = Path(settings.DIR_DADOS)

    cache_params = {
        'DATA_INICIAL': settings.DATA_INICIAL,
        'DATA_FINAL': settings.DATA_FINAL,
        'areas': lst_areas,
        'script_version': '2.0',  # pipeline híbrido ERA5/GDAS + PSL clim + 850hPa adicionado
        'streamline_defaults': STREAMLINE_DEFAULTS,
    }
    output_files = (
        [str(output_dir / f'olr_wind_anom_{area}_250hPa.png') for area in lst_areas] +
        [str(output_dir / f'olr_wind_anom_{area}_850hPa.png') for area in lst_areas]
    )

    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('CACHE VALIDO! Execucao ja foi realizada com os mesmos parametros.')
        logger.info(f'   Periodo: {settings.DATA_INICIAL} a {settings.DATA_FINAL}')
        logger.info(f'   {len(output_files)} mapas ja existem')
        logger.info(f'   Diretorio: {output_dir}')
        logger.info('   Pulando execucao')
        return

    start_time = time.time()
    logger.info(f'Periodo de analise: {settings.DATA_INICIAL} a {settings.DATA_FINAL}')
    logger.info(f'Gerando {len(lst_areas)} mapas de anomalia OLR + vento 250 e 850 hPa ({len(output_files)} figuras total)')
    logger.info('=' * 80)

    # ---- 1) Download e processamento da anomalia de vento 250 hPa ----
    logger.info('Etapa 1: Download e processamento da anomalia de vento 250 hPa')
    plot_wind250_anom()

    # ---- 1b) Download e processamento da anomalia de vento 850 hPa ----
    logger.info('Etapa 1b: Download e processamento da anomalia de vento 850 hPa')
    plot_wind850_anom()

    # ---- 2) Download do OLR (PSL/NOAA) ----
    logger.info('Etapa 2: Download OLR (PSL/NOAA)')
    dados_dir.mkdir(parents=True, exist_ok=True)
    olr_path = dados_dir / OLR_FILE_NAME

    start_date = np.datetime64(settings.DATA_INICIAL)
    end_date = np.datetime64(settings.DATA_FINAL)

    # Re-baixa apenas se o arquivo local nao cobrir o periodo solicitado
    if arquivo_cobre_periodo(olr_path, start_date, end_date):
        logger.info('Arquivo OLR local ja cobre o periodo solicitado — pulando download')
    else:
        if olr_path.exists():
            logger.info(f'Arquivo OLR local nao cobre {settings.DATA_INICIAL} a {settings.DATA_FINAL} — re-baixando')
        download_with_progress(
            url=OLR_URL,
            output_path=str(olr_path),
            description=OLR_FILE_NAME,
            max_retries=5,
            force=True,
            engine=DownloadEngine.ARIA2,  # 16 conexões paralelas: 4-8x mais rápido
            timeout=300,
        )

    # ---- 3) Processamento OLR ----
    logger.info('Etapa 3: Processamento OLR')

    ds_olr = load_dataset(str(olr_path))

    # Aborta com mensagem clara se servidor PSL/NOAA tambem nao tiver o periodo
    validar_cobertura_temporal(ds_olr, start_date, end_date, nome='arquivo OLR')

    subset = ds_olr.sel(time=slice(start_date, end_date))
    ds_olr_mean = subset.mean(dim='time')
    ds_olr_mean['lon'] = ((ds_olr_mean['lon'] + 180) % 360) - 180
    da_olr = ds_olr_mean.sortby(ds_olr_mean.lon)['olr']
    da_olr = mpcalc.smooth_gaussian(da_olr, 5)

    lon_olr = da_olr['lon']
    lat_olr = da_olr['lat']
    olr_data, lon_olr = add_cyclic_point(da_olr, coord=lon_olr)

    # ---- 4) Carregar anomalias de vento 250 e 850 hPa ----
    logger.info('Etapa 4: Carregando anomalias de vento 250 e 850 hPa')

    def _load_wind_cyc(wind_file, label):
        if not wind_file.exists():
            raise FileNotFoundError(f'Arquivo esperado nao encontrado: {wind_file} ({label})')
        ds = load_dataset(str(wind_file))
        u = ds['u_anom_mean']
        v = ds['v_anom_mean']
        lat = u['lat'].values
        u_cyc, lon_cyc = add_cyclic_point(u.values, coord=u['lon'].values)
        v_cyc, _ = add_cyclic_point(v.values, coord=v['lon'].values)
        return u_cyc, v_cyc, lat, lon_cyc

    u_cyc_250, v_cyc_250, lat_wind_250, lon_wind_cyc_250 = _load_wind_cyc(
        dados_dir / WIND250_FILE_NAME, 'vento 250 hPa'
    )
    u_cyc_850, v_cyc_850, lat_wind_850, lon_wind_cyc_850 = _load_wind_cyc(
        dados_dir / WIND850_FILE_NAME, 'vento 850 hPa'
    )

    # ---- 5) Plotagem por area (250 e 850 hPa) ----
    logger.info('Etapa 5: Plotagem — 250 hPa')
    output_dir.mkdir(parents=True, exist_ok=True)

    _gerar_figuras_nivel(
        lst_areas, olr_data, lat_olr, lon_olr,
        u_cyc_250, v_cyc_250, lat_wind_250, lon_wind_cyc_250,
        250, output_dir, input_dir, logger,
    )

    logger.info('Etapa 5b: Plotagem — 850 hPa')
    _gerar_figuras_nivel(
        lst_areas, olr_data, lat_olr, lon_olr,
        u_cyc_850, v_cyc_850, lat_wind_850, lon_wind_cyc_850,
        850, output_dir, input_dir, logger,
    )


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
