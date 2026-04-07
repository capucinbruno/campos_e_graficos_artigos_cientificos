# -*- coding: utf-8 -*-
"""
s04 - Fluxo de Atividade de Onda de Rossby (WAF) em 250 hPa.

Baixa dados de altura geopotencial 250 hPa da reanalise ERA5 via Copernicus CDS,
carrega climatologias de hgt/uwnd/vwnd 250 hPa, calcula o Wave Activity Flux
(Takaya & Nakamura 2001) via pacote tnflux, e gera mapas de anomalia de hgt250
com vetores WAF sobrepostos para areas geograficas selecionadas.

Dados de entrada:
    - ERA5 (CDS): geopotencial em 250 hPa
    - Climatologias: hgt250, uwnd250, vwnd250 (arquivos fixos em Entrada/)
    - Legenda do mapa Atlantico (arquivo fixo em Entrada/)

Saida:
    - Mapas PNG em Saida/s04_ROSSBY_WAF/

Criado em: 2026-04-02
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
import matplotlib.pyplot as plt
import numpy as np
from cartopy.util import add_cyclic_point
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from PIL import Image

# Modulos locais
# ---------------------------------------------------------------------------
from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.common.dataset_utils import load_dataset
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.src.uteis.plot_rossby_waf import main as plot_rossby_waf

# ---------------------------------------------------------------------------
# Identidade do script
# ---------------------------------------------------------------------------
SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's04'
SCRIPT_NAME = Path(__file__).stem
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME

WAF_FILE_NAME = 'rossby_waf.nc'

# Areas de plotagem (apenas globo, psa, hemisferio_sul por enquanto)
DEFAULT_AREAS = ['globo', 'psa', 'hemisferio_sul', 'hemisferio_norte']

# Configuracao de quiver por area
# ┌──────────────┬──────────────────────────────────────────────────┐
# │ Parametro    │ O que controla                                   │
# ├──────────────┼──────────────────────────────────────────────────┤
# │ step         │ Densidade: pula N pontos do grid (maior = menos) │
# │ width        │ Grossura da haste (fracao da largura do eixo)     │
# │ headwidth    │ Largura da cabeca (multiplo de width)             │
# │ headlength   │ Comprimento da cabeca (multiplo de width)         │
# │ min_amp_ratio│ Limiar: oculta setas < ratio * max_amplitude     │
# └──────────────┴──────────────────────────────────────────────────┘
# Vetores normalizados pelo maximo (scale=None, scale_units='xy')
# para tamanho proporcional automatico — mesmo padrao do S85.
QUIVER_DEFAULTS = {
    'step': 2,
    'width': 0.002,
    'headwidth': 4.5,
    'headlength': 6.0,
    'scale': None,
    'scale_units': 'xy',
    'min_amp_ratio': 0.05,
}

QUIVER_POR_AREA = {
    'psa': {'step': 2},
    'hemisferio_sul': {'step': 2},
    'hemisferio_norte': {'step': 2},
    'globo': {'step': 2},
}


# ---------------------------------------------------------------------------
# Funcoes utilitarias
# ---------------------------------------------------------------------------
def _get_area_list():
    if hasattr(settings, 'LST_AREAS_S04'):
        return list(settings.LST_AREAS_S04)
    return list(DEFAULT_AREAS)


def _get_quiver_config(area: str) -> dict:
    cfg = dict(QUIVER_DEFAULTS)
    if area in QUIVER_POR_AREA:
        cfg.update(QUIVER_POR_AREA[area])
    return cfg


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


def _configure_gridlines(gl, area):
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 20, 'color': 'black'}
    gl.ylabel_style = {'size': 20, 'color': 'black'}

    if area in {'hemisferio_sul', 'hemisferio_norte', 'psa'}:
        gl.xlocator = MultipleLocator(40)
        gl.ylocator = MultipleLocator(20)

    elif area == 'globo':
        gl.xlocator = MultipleLocator(40)
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

    output_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_ROSSBY_WAF'
    input_dir = Path(settings.DIR_INPUT)
    dados_dir = Path(settings.DIR_DADOS)

    cache_params = {
        'DATA_INICIAL': settings.DATA_INICIAL,
        'DATA_FINAL': settings.DATA_FINAL,
        'areas': lst_areas,
        'script_version': '1.0',
        'waf_file': WAF_FILE_NAME,
        'quiver_defaults': QUIVER_DEFAULTS,
        'quiver_por_area': QUIVER_POR_AREA,
    }
    output_files = [str(output_dir / f'rossby_waf_{area}.png') for area in lst_areas]

    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('CACHE VALIDO! Execucao ja foi realizada com os mesmos parametros.')
        logger.info('   Periodo: %s a %s', settings.DATA_INICIAL, settings.DATA_FINAL)
        logger.info('   %d mapas ja existem', len(output_files))
        logger.info('   Diretorio: %s', output_dir)
        logger.info('   Pulando execucao')
        return

    start_time = time.time()
    logger.info('Periodo de analise: %s a %s', settings.DATA_INICIAL, settings.DATA_FINAL)
    logger.info('Gerando %d mapas de Rossby WAF', len(lst_areas))
    logger.info('=' * 80)

    # Etapa anterior: download + processamento -> rossby_waf.nc
    try:
        plot_rossby_waf()
    except Exception as err:
        logger.exception('Falha no download/processamento do Rossby WAF')
        raise RuntimeError('Falha no download/processamento do Rossby WAF') from err

    output_dir.mkdir(parents=True, exist_ok=True)

    waf_file = dados_dir / WAF_FILE_NAME
    if not waf_file.exists():
        raise FileNotFoundError(
            f'Arquivo esperado nao encontrado: {waf_file}. '
            'A rotina plot_rossby_waf() precisa salvar esse NetCDF antes da plotagem.'
        )

    ds = load_dataset(str(waf_file))

    hgt_anom = ds['hgt_anom_mean']
    waf_x = ds['waf_x']
    waf_y = ds['waf_y']

    # Grade alta resolucao (anomalia hgt para contourf)
    lat = hgt_anom['lat'].values
    lon = hgt_anom['lon'].values

    # Grade ~2.5° (WAF para quiver)
    lat_waf = waf_x['lat_waf'].values
    lon_waf = waf_x['lon_waf'].values

    # Ajuste de longitude para -180..180 e cyclic point — anomalia
    lon_shift = ((lon + 180) % 360) - 180
    sort_idx = np.argsort(lon_shift)
    lon_shift_sorted = lon_shift[sort_idx]

    hgt_sorted = hgt_anom.isel(lon=sort_idx).values
    hgt_cyc, lon_cyc = add_cyclic_point(hgt_sorted, coord=lon_shift_sorted)

    # Ajuste de longitude para -180..180 e cyclic point — WAF
    lon_waf_shift = ((lon_waf + 180) % 360) - 180
    sort_idx_waf = np.argsort(lon_waf_shift)
    lon_waf_sorted = lon_waf_shift[sort_idx_waf]

    px_sorted = waf_x.values[:, sort_idx_waf]
    py_sorted = waf_y.values[:, sort_idx_waf]

    px_cyc, lon_waf_cyc = add_cyclic_point(px_sorted, coord=lon_waf_sorted)
    py_cyc, _ = add_cyclic_point(py_sorted, coord=lon_waf_sorted)

    # Levels e paleta (mesmo padrao do s01)
    if np.all((hgt_cyc >= -50) & (hgt_cyc <= 50)):
        levels = np.arange(-50, 53, 3)
        ticks = np.arange(-50, 55, 5)
    elif np.all((hgt_cyc >= -100) & (hgt_cyc <= 100)):
        levels = np.arange(-100, 120, 20)
        ticks = np.arange(-100, 150, 50)
    else:
        levels = np.arange(-200, 220, 20)
        ticks = np.arange(-200, 250, 50)

    cmap_name = settings.LST_ANOM_CORRETA
    cmap = LinearSegmentedColormap.from_list(cmap_name, settings.LST_ANOM_CORRETA)

    info_plot = settings['areas_plotagem']

    for area in lst_areas:
        logger.info('Gerando mapa Rossby WAF para area: %s', area)

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(
            1,
            1,
            1,
            projection=ccrs.PlateCarree(
                central_longitude=info_plot[area]['central_longitude_mapa']
            ),
        )

        if info_plot[area].get('plot_box', False):
            from matplotlib import patches

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

        # Grade
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.0)
        _configure_gridlines(gl, area)

        # Limites
        ax.set_xlim([info_plot[area]['lon_esq'], info_plot[area]['lon_dir']])
        ax.set_ylim([info_plot[area]['lat_inf'], info_plot[area]['lat_sup']])

        # Features
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2)
        ax.add_feature(cfeature.LAND.with_scale('50m'), linewidth=0.5, facecolor='whitesmoke')
        ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1.2, zorder=100)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, zorder=100)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, zorder=100)
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), linewidth=0.5, facecolor='white')

        # Contourf — anomalia de hgt 250 hPa
        im = ax.contourf(
            lon_cyc,
            lat,
            hgt_cyc,
            levels=levels,
            cmap=cmap,
            extend='both',
            transform=ccrs.PlateCarree(
                central_longitude=info_plot[area]['central_longitude_plot']
            ),
        )

        # Contornos brancos
        ax.contour(
            lon_cyc,
            lat,
            hgt_cyc,
            levels=levels,
            colors='white',
            linewidths=0.6,
            transform=ccrs.PlateCarree(
                central_longitude=info_plot[area]['central_longitude_plot']
            ),
            zorder=110,
        )

        # Vetores WAF (quiver) — grade ~2.5°, normalizados pelo maximo
        qcfg = _get_quiver_config(area)
        step = int(qcfg['step'])

        lon_q = lon_waf_cyc[::step]
        lat_q = lat_waf[::step]
        px_q = px_cyc[::step, ::step]
        py_q = py_cyc[::step, ::step]

        # Normaliza pelo maximo e mascara fracos (padrao S85)
        amp = np.sqrt(px_q**2 + py_q**2)
        max_amp = np.nanmax(amp)

        if max_amp > 0:
            px_plot = px_q / max_amp
            py_plot = py_q / max_amp
            mask_weak = amp < float(qcfg['min_amp_ratio']) * max_amp
            px_plot = np.where(mask_weak, np.nan, px_plot)
            py_plot = np.where(mask_weak, np.nan, py_plot)
        else:
            px_plot = px_q
            py_plot = py_q

        ax.quiver(
            lon_q,
            lat_q,
            px_plot,
            py_plot,
            transform=ccrs.PlateCarree(
                central_longitude=info_plot[area]['central_longitude_plot']
            ),
            scale=qcfg['scale'],
            scale_units=qcfg['scale_units'],
            width=float(qcfg['width']),
            headwidth=float(qcfg['headwidth']),
            headlength=float(qcfg['headlength']),
            zorder=200,
            color='black',
        )

        # Colorbar
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
            ticks=ticks,
        )
        cbar.set_label(label='mgp', size=18)
        cbar.ax.tick_params(labelsize=20)

        # Titulo
        dt_ini = datetime.strptime(settings.DATA_INICIAL, '%Y-%m-%d').strftime('%d-%m-%y')
        dt_fim = datetime.strptime(settings.DATA_FINAL, '%Y-%m-%d').strftime('%d-%m-%y')
        titulo = (
            f'Anomalia Geopotencial 250hPa + Fluxo de Atividade de Onda de Rossby\n'
            f'(De {dt_ini} a {dt_fim})'
        )
        ax.set_title(titulo, fontsize=16, loc='left')

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

        filename_fig = output_dir / f'rossby_waf_{area}.png'
        logger.info('Salvando a figura %s', filename_fig)

        plt.savefig(
            str(filename_fig),
            dpi=fig.dpi,
            bbox_inches='tight',
        )
        plt.close('all')

    # Salvar metadados do cache
    execution_time = time.time() - start_time
    save_cache_metadata(SCRIPT_ID, cache_params, output_files, execution_time)
    logger.info('=' * 80)
    logger.info('Script %s concluido com sucesso!', SCRIPT_ID.upper())
    logger.info('Tempo de execucao: %.1fs (%.1f min)', execution_time, execution_time / 60)
    logger.info('%d mapas gerados em: %s', len(output_files), output_dir)
    logger.info('=' * 80)


if __name__ == '__main__':
    main()
