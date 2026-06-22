# -*- coding: utf-8 -*-
"""
s09 - Fluxo de Atividade de Onda de Rossby (WAF) + Anomalia TSM em 250 hPa.

Baixa dados de geopotencial 250 hPa (ERA5/GDAS) e anomalia de TSM (OISSTv2/NOAA),
calcula o Wave Activity Flux (Takaya & Nakamura 2001) via pacote tnflux, e gera
mapas com anomalia de TSM em shaded, contornos pretos de hgt250 e vetores WAF.

Dados de entrada:
    - ERA5/GDAS: geopotencial em 250 hPa (via plot_rossby_waf)
    - PSL/NOAA: sst.day.anom.{ano}.nc (OISSTv2 High-Res 0.25 grau)
    - Climatologias: hgt250, uwnd250, vwnd250 (arquivos fixos em Entrada/)

Saida:
    - Mapas PNG em Saida/s09_ROSSBY_WAF_SSTA/

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
import numpy as np
import xarray as xr
from cartopy.util import add_cyclic_point
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap
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
from app.common.logo_helper import resolve_logo_path
from app.common.logo_helper import proportional_logo_zoom
from app.src.uteis.plot_rossby_waf import main as plot_rossby_waf

# ---------------------------------------------------------------------------
# Identidade do script
# ---------------------------------------------------------------------------
SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's09'
SCRIPT_NAME = Path(__file__).stem
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME

WAF_FILE_NAME = 'rossby_waf.nc'
SST_URL_TEMPLATE = (
    'https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.day.anom.{year}.nc'
)
SST_FILE_TEMPLATE = 'sst.day.anom.{year}.nc'

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
    'scale': None,
    'scale_units': 'xy',
    'min_amp_ratio': 0.05,
}

QUIVER_POR_AREA = {
    'psa': {'step': 2},
    'hemisferio_sul': {'step': 2},
    'hemisferio_norte': {'step': 2},
    'globo': {'step': 2},
    'america_sul': {'step': 1, 'width': 0.004, 'headwidth': 5.0, 'headlength': 7.0, 'scale': 0.12},
    'globo_3d': {'step': 2, 'width': 0.003, 'headwidth': 5.0, 'headlength': 7.0},
}


# ---------------------------------------------------------------------------
# Funcoes utilitarias
# ---------------------------------------------------------------------------
def _get_area_list():
    if hasattr(settings, 'LST_AREAS_S09'):
        return list(settings.LST_AREAS_S09)
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


def _download_sst_anos(
    dados_dir: Path,
    start_date: np.datetime64,
    end_date: np.datetime64,
    logger,
) -> list:
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
            description=f'SST anomalia {year}',
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

    output_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_ROSSBY_WAF_SSTA'
    input_dir = Path(settings.DIR_INPUT)
    dados_dir = Path(settings.DIR_DADOS)

    cache_params = {
        'DATA_INICIAL': settings.DATA_INICIAL,
        'DATA_FINAL': settings.DATA_FINAL,
        'areas': lst_areas,
        'script_version': '1.0',
        'waf_file': WAF_FILE_NAME,
        'sst_url_template': SST_URL_TEMPLATE,
    }
    output_files = [str(output_dir / f'rossby_waf_ssta_{area}.png') for area in lst_areas]

    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('CACHE VALIDO! Execucao ja foi realizada com os mesmos parametros.')
        logger.info(f'   Periodo: {settings.DATA_INICIAL} a {settings.DATA_FINAL}')
        logger.info(f'   {len(output_files)} mapas ja existem')
        logger.info(f'   Diretorio: {output_dir}')
        logger.info('   Pulando execucao')
        return

    start_time = time.time()
    logger.info(f'Periodo de analise: {settings.DATA_INICIAL} a {settings.DATA_FINAL}')
    logger.info(f'Gerando {len(lst_areas)} mapas de Rossby WAF + SSTA')

    dt_ini = datetime.strptime(str(settings.DATA_INICIAL), '%Y-%m-%d')
    dt_fim = datetime.strptime(str(settings.DATA_FINAL), '%Y-%m-%d')

    logger.info('=' * 80)

    # Etapa 1: Download + processamento WAF → rossby_waf.nc
    plot_rossby_waf()

    # Etapa 2: Download dos arquivos SST anuais (OISSTv2)
    dados_dir.mkdir(parents=True, exist_ok=True)
    start_date = np.datetime64(settings.DATA_INICIAL, 'D')
    end_date = np.datetime64(settings.DATA_FINAL, 'D')
    sst_paths = _download_sst_anos(dados_dir, start_date, end_date, logger)

    # Etapa 3: Processamento SSTA — carregar sequencialmente e acumular média
    logger.info('Carregando e processando dados SST (modo eficiente em memória)...')

    # Estratégia: carregar cada arquivo SST sequencialmente, filtrar pelo período,
    # e acumular a soma + contagem para calcular a média sem manter múltiplos anos
    # em memória simultaneamente.
    sum_anom = None
    count_days = 0

    for path in sst_paths:
        with xr.open_dataset(str(path), decode_times=True) as ds_file:
            # Selecionar período desejado dentro do arquivo
            da_subset = ds_file['anom'].sel(time=slice(start_date, end_date))
            
            # Acumular soma e contagem
            if sum_anom is None:
                sum_anom = da_subset.sum(dim='time').load()
            else:
                sum_anom = sum_anom + da_subset.sum(dim='time').load()
            count_days += len(da_subset.time)
    
    # Calcular média final (resultado 2D em memória)
    da_mean = sum_anom / count_days
    
    # Validar que temos dados para o período
    if count_days == 0:
        raise ValueError(f'Nenhum dado SST encontrado para período {start_date} a {end_date}')

    # Ajuste de longitude e ordenação — agora sobre a DataArray 2D em memória
    da_mean['lon'] = ((da_mean['lon'] + 180) % 360) - 180
    da_mean = da_mean.sortby(da_mean.lon)

    da_sst = da_mean

    ssta_lat = da_sst['lat'].values
    ssta_lon = da_sst['lon'].values
    ssta_cyc, ssta_lon_cyc = add_cyclic_point(da_sst.values, coord=ssta_lon)

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

    # Colormap e levels SSTA (mesmos do s10)
    sst_levels = [float(x) for x in settings.LST_SSTA_NEW_GREC]
    cmap_colors = [str(x) for x in settings.LST_ANOM_CORRETA]
    cmap = LinearSegmentedColormap.from_list('sst_anom', cmap_colors)

    dt_ini_str = dt_ini.strftime('%d-%m-%y')
    dt_fim_str = dt_fim.strftime('%d-%m-%y')

    info_plot = settings['areas_plotagem']

    for area in lst_areas:
        logger.info(f'Gerando mapa Rossby WAF + SSTA para area: {area_display_name(area)}')

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

        # Fundo branco — cobre ocean onde o contourf nao tiver dado
        ax.set_facecolor('white')

        # LAND antes do contourf; SSTA e NaN em terra entao nao sobrepoe
        ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='whitesmoke', zorder=2)

        data_transform = ccrs.PlateCarree(
            central_longitude=info_plot[area]['central_longitude_plot']
        )

        # Contourf — anomalia SSTA (shaded, NaN em terra e ignorado)
        # Adicionar logs de diagnostico e tornar contourf mais robusto para
        # cenarios com dados faltantes ou coordenadas recortadas.
        try:
            ssta_arr = np.array(ssta_cyc, dtype=float)
            total_pts = ssta_arr.size
            valid_pts = int(np.isfinite(ssta_arr).sum())
            min_val = float(np.nanmin(ssta_arr)) if valid_pts > 0 else float('nan')
            max_val = float(np.nanmax(ssta_arr)) if valid_pts > 0 else float('nan')
            logger.debug(f'SSTA stats: min={min_val:.3f}, max={max_val:.3f}, valid={valid_pts}/{total_pts}')
        except Exception as _e:
            logger.debug(f'Erro ao calcular estatisticas SSTA: {_e}')
            ssta_arr = np.ma.masked_invalid(ssta_cyc).astype(float)

        if valid_pts == 0:
            logger.warning(f'Nenhum ponto valido de SSTA para o periodo/area — verificando entradas e limites de tempo')

        vmin = float(sst_levels[0]) if sst_levels else None
        vmax = float(sst_levels[-1]) if sst_levels else None

        # Usar pcolormesh é mais robusto a grades irregulares e mascaramento
        # e evita lacunas que por vezes ocorrem com contourf em grades globais.
        im = ax.pcolormesh(
            ssta_lon_cyc,
            ssta_lat,
            np.ma.masked_invalid(ssta_arr),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading='auto',
            transform=data_transform,
            zorder=5,
            rasterized=True,
        )

        # Features sobre o contourf
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, zorder=100)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, zorder=100)
        if area != 'china':
            ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1.2, zorder=100)

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

        # Vetores WAF (quiver) — grade ~2.5°, normalizados pelo maximo
        qcfg = _get_quiver_config(area)
        step = int(qcfg['step'])

        lon_q = lon_waf_cyc[::step]
        lat_q = lat_waf[::step]
        px_q = px_cyc[::step, ::step]
        py_q = py_cyc[::step, ::step]

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
            transform=data_transform,
            scale=qcfg['scale'],
            scale_units=qcfg['scale_units'],
            width=float(qcfg['width']),
            headwidth=float(qcfg['headwidth']),
            headlength=float(qcfg['headlength']),
            zorder=200,
            color='black',
        )

        # Colorbar — SSTA (°C)
        cbar_ticks = np.arange(-5, 6, 1)
        if is_polar and area != 'globo_3d':
            cbar = plt.colorbar(im, ax=ax, pad=0.05, fraction=0.04, ticks=cbar_ticks)
            cbar.set_label(label='°C', size=10)
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
                    extendrect=False,
                    ticks=cbar_ticks,
                )
                cbar.set_label(label='°C', size=18)
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
                    ticks=cbar_ticks,
                )
                cbar.set_label(label='°C', size=18)
                cbar.ax.tick_params(labelsize=20)

        # Titulo
        titulo = (
            f'Anomalia TSM (shaded) + Geopotencial 250hPa (linhas) +\n'
            f'Fluxo de Atividade de Onda de Rossby (De {dt_ini_str} a {dt_fim_str})'
        )
        ax.set_title(titulo, fontsize=12 if is_polar else 16, loc='left')

        # Logo
        logo_path = resolve_logo_path(input_dir)
        if logo_path is not None and logo_path.exists():
            _add_logo_to_map(
                ax=ax,
                logo_path=logo_path,
                zoom=0.65,
                xoffset=0,
                yoffset=0,
                zorder=500,
            )

        # Contorno uniforme da área de plotagem (não aplicar no globo_3d)
        if area != 'globo_3d':
            ax.add_patch(patches.Rectangle(
                (0, 0), 1, 1,
                linewidth=0.5,
                edgecolor='black',
                facecolor='none',
                transform=ax.transAxes,
                zorder=1000,
                clip_on=False,
            ))

        filename_fig = output_dir / f'rossby_waf_ssta_{area}.png'
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
