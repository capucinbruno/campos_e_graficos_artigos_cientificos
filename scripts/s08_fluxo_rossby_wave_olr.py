# -*- coding: utf-8 -*-
"""
s08 - Fluxo de Atividade de Onda de Rossby (WAF) + Anomalia OLR em 250 hPa.

Baixa dados de geopotencial 250 hPa (ERA5/GDAS) e OLR (PSL/NOAA), calcula
o Wave Activity Flux (Takaya & Nakamura 2001) via pacote tnflux, e gera mapas
com anomalia de OLR em shaded, contornos pretos de hgt250 e vetores WAF.

Dados de entrada:
    - ERA5/GDAS: geopotencial em 250 hPa (via plot_rossby_waf)
    - PSL/NOAA: olr.day.anom.nc (CPC Blended OLR 2.5 graus)
    - Climatologias: hgt250, uwnd250, vwnd250 (arquivos fixos em Entrada/)

Saida:
    - Mapas PNG em Saida/s08_ROSSBY_WAF_OLR/

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
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
from cartopy.util import add_cyclic_point
from matplotlib import patches
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import MultipleLocator
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
from app.src.uteis.plot_rossby_waf import main as plot_rossby_waf

# ---------------------------------------------------------------------------
# Identidade do script
# ---------------------------------------------------------------------------
SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's08'
SCRIPT_NAME = Path(__file__).stem
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME

WAF_FILE_NAME = 'rossby_waf.nc'
OLR_URL = 'https://downloads.psl.noaa.gov/Datasets/cpc_blended_olr-2.5deg/olr.day.anom.nc'
OLR_FILE_NAME = 'olr.day.anom.nc'

# Areas de plotagem
DEFAULT_AREAS = ['globo', 'psa', 'hemisferio_sul', 'hemisferio_norte', 'america_sul']

# Configuracao de quiver por area
# ┌──────────────┬──────────────────────────────────────────────────┐
# │ Parametro    │ O que controla                                   │
# ├──────────────┼──────────────────────────────────────────────────┤
# │ step         │ Densidade: pula N pontos do grid (maior = menos) │
# │ width        │ Grossura da haste (fracao da largura do eixo)     │
# │ headwidth    │ Largura da cabeca (multiplo de width)             │
# │ headlength   │ Comprimento da cabeca (multiplo de width)         │
# │ pct_weak     │ Remove vetores abaixo deste percentil             │
# │ pct_clip     │ Clipa vetores acima deste percentil               │
# └──────────────┴──────────────────────────────────────────────────┘
QUIVER_DEFAULTS = {
    'step': 2,
    'width': 0.002,
    'headwidth': 4.5,
    'headlength': 6.0,
    'scale': 0.5,
    'scale_units': 'xy',
    'pct_weak': 30,
    'pct_clip': 95,
}

QUIVER_POR_AREA = {
    'psa': {'step': 2},
    'hemisferio_sul': {'step': 2},
    'hemisferio_norte': {'step': 2},
    'globo': {'step': 2},
    'america_sul': {'step': 1, 'width': 0.004, 'headwidth': 5.0, 'headlength': 7.0},
    'globo_3d': {'step': 2, 'width': 0.003, 'headwidth': 5.0, 'headlength': 7.0, 'scale': 400, 'scale_units': 'width'},
}


# ---------------------------------------------------------------------------
# Funcoes utilitarias
# ---------------------------------------------------------------------------
def _get_area_list():
    if hasattr(settings, 'LST_AREAS_S08'):
        return list(settings.LST_AREAS_S08)
    return list(DEFAULT_AREAS)


def _get_quiver_config(area: str) -> dict:
    cfg = dict(QUIVER_DEFAULTS)
    if area in QUIVER_POR_AREA:
        cfg.update(QUIVER_POR_AREA[area])
    return cfg


def _quiver_scale_for_period(n_days: int) -> float:
    """Retorna scale do quiver baseado no tamanho do periodo.

    WAF de curto prazo tem magnitude muito maior que WAF de longo prazo:
    scale maior = setas menores. Ajustar os limiares conforme testes visuais.
    """
    if n_days > 60:
        return 0.5
    if n_days > 31:
        return 1.5
    if n_days > 15:
        return 3.0
    return 4.0


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
    elif area == 'america_sul':
        gl.xlocator = MultipleLocator(20)
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
    logger.info(f'SCRIPT {SCRIPT_ID.upper()}: {SCRIPT_DESC}')
    logger.info('=' * 80)

    lst_areas = _get_area_list()

    output_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_ROSSBY_WAF_OLR'
    input_dir = Path(settings.DIR_INPUT)
    dados_dir = Path(settings.DIR_DADOS)

    cache_params = {
        'DATA_INICIAL': settings.DATA_INICIAL,
        'DATA_FINAL': settings.DATA_FINAL,
        'areas': lst_areas,
        'script_version': '1.0',
        'waf_file': WAF_FILE_NAME,
        'olr_url': OLR_URL,
    }
    output_files = [str(output_dir / f'rossby_waf_olr_{area}.png') for area in lst_areas]

    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('CACHE VALIDO! Execucao ja foi realizada com os mesmos parametros.')
        logger.info(f'   Periodo: {settings.DATA_INICIAL} a {settings.DATA_FINAL}')
        logger.info(f'   {len(output_files)} mapas ja existem')
        logger.info(f'   Diretorio: {output_dir}')
        logger.info('   Pulando execucao')
        return

    start_time = time.time()
    logger.info(f'Periodo de analise: {settings.DATA_INICIAL} a {settings.DATA_FINAL}')
    logger.info(f'Gerando {len(lst_areas)} mapas de Rossby WAF + OLR')

    dt_ini = datetime.strptime(str(settings.DATA_INICIAL), '%Y-%m-%d')
    dt_fim = datetime.strptime(str(settings.DATA_FINAL), '%Y-%m-%d')
    n_days = (dt_fim - dt_ini).days + 1
    period_scale = _quiver_scale_for_period(n_days)
    logger.info(f'Periodo: {n_days} dias → scale do quiver: {period_scale:.1f}')
    logger.info('=' * 80)

    # Etapa 1: Download + processamento WAF → rossby_waf.nc
    plot_rossby_waf()

    # Etapa 2: Download do arquivo OLR (PSL/NOAA)
    dados_dir.mkdir(parents=True, exist_ok=True)
    olr_path = dados_dir / OLR_FILE_NAME

    start_date = np.datetime64(settings.DATA_INICIAL)
    end_date = np.datetime64(settings.DATA_FINAL)

    if arquivo_cobre_periodo(olr_path, start_date, end_date):
        logger.info('Arquivo OLR local ja cobre o periodo solicitado — pulando download')
    else:
        if olr_path.exists():
            logger.info(
                'Arquivo OLR local nao cobre %s a %s — re-baixando',
                settings.DATA_INICIAL,
                settings.DATA_FINAL,
            )
        download_with_progress(
            url=OLR_URL,
            output_path=str(olr_path),
            description=OLR_FILE_NAME,
            max_retries=5,
            force=True,
            engine=DownloadEngine.ARIA2,
            timeout=300,
        )

    # Etapa 3: Processamento OLR — media do periodo e cyclic point
    ds_olr = load_dataset(str(olr_path))
    validar_cobertura_temporal(ds_olr, start_date, end_date, nome='arquivo OLR')

    subset = ds_olr.sel(time=slice(start_date, end_date))
    ds_mean = subset.mean(dim='time')
    ds_mean['lon'] = ((ds_mean['lon'] + 180) % 360) - 180
    da_olr = ds_mean.sortby(ds_mean.lon)['olr']
    da_olr = mpcalc.smooth_gaussian(da_olr, 5)

    olr_lat = da_olr['lat']
    olr_lon = da_olr['lon']
    olr_cyc, olr_lon_cyc = add_cyclic_point(da_olr, coord=olr_lon)

    # Etapa 4: Carrega WAF (hgt_anom_mean + waf_x + waf_y)
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

    # Grade alta resolucao (hgt250 para contorno preto)
    lat = hgt_anom['lat'].values
    lon = hgt_anom['lon'].values

    # Grade ~2.5° (WAF para quiver)
    lat_waf = waf_x['lat_waf'].values
    lon_waf = waf_x['lon_waf'].values

    # Ajuste de longitude para -180..180 e cyclic point — hgt250
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

    # Levels de contorno para hgt250 (adaptativo ao intervalo dos dados)
    if np.all((hgt_cyc >= -50) & (hgt_cyc <= 50)):
        hgt_levels = np.arange(-50, 53, 3)
    elif np.all((hgt_cyc >= -100) & (hgt_cyc <= 100)):
        hgt_levels = np.arange(-100, 120, 20)
    else:
        hgt_levels = np.arange(-200, 220, 20)

    dt_ini_str = dt_ini.strftime('%d-%m-%y')
    dt_fim_str = dt_fim.strftime('%d-%m-%y')

    info_plot = settings['areas_plotagem']

    for area in lst_areas:
        logger.info(f'Gerando mapa Rossby WAF + OLR para area: {area_display_name(area)}')

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

        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2)
        ax.add_feature(cfeature.LAND.with_scale('50m'), linewidth=0.5, facecolor='whitesmoke')
        ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1.2, zorder=100)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, zorder=100)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, zorder=100)
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), linewidth=0.5, facecolor='white')

        data_transform = ccrs.PlateCarree(
            central_longitude=info_plot[area]['central_longitude_plot']
        )

        # Contourf — anomalia OLR (shaded, BrBG_r)
        im = ax.contourf(
            olr_lon_cyc,
            olr_lat,
            olr_cyc,
            levels=np.arange(-40, 44, 4),
            cmap='BrBG_r',
            extend='both',
            transform=data_transform,
        )

        # Contornos pretos — anomalia hgt 250 hPa (sem fill)
        ax.contour(
            lon_cyc,
            lat,
            hgt_cyc,
            levels=hgt_levels,
            colors='black',
            linewidths=1.0,
            transform=data_transform,
            zorder=110,
        )

        # Vetores WAF (quiver) — grade ~2.5°, sem normalizacao por maximo
        qcfg = _get_quiver_config(area)
        step = int(qcfg['step'])

        lon_q = lon_waf_cyc[::step]
        lat_q = lat_waf[::step]
        px_q = px_cyc[::step, ::step].copy()
        py_q = py_cyc[::step, ::step].copy()

        amp = np.hypot(px_q, py_q)

        # Remove vetores fracos (ruido)
        min_thr = np.nanpercentile(amp, float(qcfg['pct_weak']))
        px_q = np.where(amp < min_thr, np.nan, px_q)
        py_q = np.where(amp < min_thr, np.nan, py_q)

        # Clipa vetores extremos: escala para baixo sem remover
        amp = np.hypot(px_q, py_q)
        max_thr = np.nanpercentile(amp, float(qcfg['pct_clip']))
        factor = np.ones_like(amp)
        mask_big = amp > max_thr
        factor[mask_big] = max_thr / amp[mask_big]

        px_plot = px_q * factor
        py_plot = py_q * factor

        ax.quiver(
            lon_q,
            lat_q,
            px_plot,
            py_plot,
            transform=data_transform,
            scale=float(qcfg['scale']) if area in QUIVER_POR_AREA and 'scale' in QUIVER_POR_AREA[area] else period_scale,
            scale_units=qcfg['scale_units'],
            width=float(qcfg['width']),
            headwidth=float(qcfg['headwidth']),
            headlength=float(qcfg['headlength']),
            zorder=200,
            color='black',
        )

        # Colorbar — OLR (W/m²)
        if is_polar and area != 'globo_3d':
            cbar = plt.colorbar(im, ax=ax, pad=0.05, fraction=0.04, ticks=np.arange(-40, 50, 10))
            cbar.set_label(label='W/m$^2$', size=10)
            cbar.ax.tick_params(labelsize=10)
        else:
            divider = make_axes_locatable(ax)
            if area in {'america_sul', 'globo_3d'}:
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
            else:
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
                cbar.set_label(label='W/m$^2$', size=18)
                cbar.ax.tick_params(labelsize=20)

        # Titulo
        titulo = (
            f'Anomalia OLR (shaded) + Geopotencial 250hPa (linhas) +\n'
            f'Fluxo de Atividade de Onda de Rossby (De {dt_ini_str} a {dt_fim_str})'
        )
        ax.set_title(titulo, fontsize=12 if is_polar else 16, loc='left')

        # Logo
        logo_path = (
            None if settings.get('SEM_LOGO', False)
            else input_dir / ('logo_grec.png' if settings.get('LOGO_GREC', False) else 'novo_logo.png')
        )
        if logo_path is not None and logo_path.exists():
            _add_logo_to_map(
                ax=ax,
                logo_path=logo_path,
                zoom=0.65,
                xoffset=0,
                yoffset=0,
                zorder=500,
            )

        filename_fig = output_dir / f'rossby_waf_olr_{area}.png'
        logger.info(f'Salvando a figura {filename_fig}')

        plt.savefig(
            str(filename_fig),
            dpi=fig.dpi,
            bbox_inches='tight',
        )
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
