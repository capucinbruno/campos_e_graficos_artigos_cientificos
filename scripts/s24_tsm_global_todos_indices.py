"""s24 - Anomalia de TSM Global com todos os Indices Climaticos.

Baixa a SST absoluta do PSL/NOAA (OISSTv2 High-Res), calcula a media do periodo
e a anomalia subtraindo a climatologia diaria OISST (LTM 1991-2020) recortada no
mesmo periodo; gera mapa global com os indices ENSO (Nino 1+2, 3, 3.4, 4), IOD,
AMO, TNA, TSA, SAD e PDO sobrepostos.

Dados de entrada:
    - PSL/NOAA: sst.day.mean.{ano}.nc (OISSTv2 0.25 grau, um arquivo por ano)
    - Entrada/sst.day.mean.ltm.1991-2020.nc (climatologia diaria OISST p/ anomalia)
    - Entrada/pdo_eof/EOF1.csv (EOF1 para calculo do indice PDO)

Saida:
    - Mapas PNG em {settings.DIR_OUTPUT}/s24_SSTA_GLOBO_INDICES/

Criado em: 2026-06-06
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')

# Bibliotecas padrão
import time
from datetime import datetime
from pathlib import Path

# Bibliotecas de terceiros
import cartopy.crs as ccrs
import cartopy.feature as cfeature
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

# Módulos locais
from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.common.dataset_utils import arquivo_cobre_periodo, validar_cobertura_temporal
from app.common.download_helper import DownloadEngine, download_with_progress
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.src.uteis.indices_climaticos_tsm import calcula_indice_pdo, desenha_boxes_indices
from app.src.uteis.ssta_climatologia import clim_mean_array

# ---------------------------------------------------------------------------
# Identidade do script
# ---------------------------------------------------------------------------
SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's24'
SCRIPT_NAME = Path(__file__).stem
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
SST_URL_TEMPLATE = (
    'https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.day.mean.{year}.nc'
)
SST_FILE_TEMPLATE = 'sst.day.mean.{year}.nc'


# ---------------------------------------------------------------------------
# Download SST (idêntico ao s11)
# ---------------------------------------------------------------------------

def _download_sst_anos(
    dados_dir: Path,
    start_date: np.datetime64,
    end_date: np.datetime64,
    logger,
) -> list[Path]:
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

    lst_areas = ['globo_indices']
    output_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_SSTA_GLOBO_INDICES'
    input_dir = Path(settings.DIR_INPUT)
    dados_dir = Path(settings.DIR_DADOS)

    cache_params = {
        'DATA_INICIAL': settings.DATA_INICIAL,
        'DATA_FINAL': settings.DATA_FINAL,
        'areas': lst_areas,
        'anom_source': 'sst.day.mean - ltm.1991-2020',
        'script_version': '2.1',
    }
    output_files = [str(output_dir / f'ssta_globo_indices_{area}.png') for area in lst_areas]

    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('CACHE VALIDO! Execucao ja foi realizada com os mesmos parametros.')
        logger.info(f'   Periodo: {settings.DATA_INICIAL} a {settings.DATA_FINAL}')
        logger.info(f'   {len(output_files)} mapas ja existem')
        logger.info(f'   Diretorio: {output_dir}')
        logger.info('   Pulando execucao')
        return

    start_time = time.time()
    logger.info(f'Periodo de analise: {settings.DATA_INICIAL} a {settings.DATA_FINAL}')
    logger.info('=' * 80)

    start_date = np.datetime64(settings.DATA_INICIAL, 'D')
    end_date = np.datetime64(settings.DATA_FINAL, 'D')

    # ---- Download SST ----
    dados_dir.mkdir(parents=True, exist_ok=True)
    sst_paths = _download_sst_anos(dados_dir, start_date, end_date, logger)

    # ---- Carregamento e media temporal ----
    logger.info('Carregando e processando dados SST...')
    if len(sst_paths) == 1:
        ds_raw = xr.open_dataset(str(sst_paths[0]), decode_times=True, chunks={'time': 20})
    else:
        ds_raw = xr.open_mfdataset(
            [str(p) for p in sst_paths],
            combine='by_coords',
            decode_times=True,
            chunks={'time': 20},
            coords='minimal',
        )

    ds_raw = ds_raw.sel(time=slice(str(start_date), str(end_date)))
    validar_cobertura_temporal(ds_raw, start_date, end_date, nome='SST OISSTv2')
    ds_mean = ds_raw.mean(dim='time').compute()  # tem 'sst' (SST absoluta media)
    ds_raw.close()

    # ---- Anomalia = media(SST) - climatologia diaria recortada no mesmo periodo ----
    logger.info('Calculando anomalia com a climatologia diaria OISST...')
    clim_mean = clim_mean_array(
        start_date, end_date, ds_mean['lat'].values, ds_mean['lon'].values, logger
    )
    ds_mean['anom'] = (('lat', 'lon'), ds_mean['sst'].values - clim_mean)

    # ---- Indice PDO (antes de converter lon) ----
    logger.info('Calculando indice PDO...')
    index_pdo = round(calcula_indice_pdo(ds_mean, 'anom'), 2)
    logger.info(f'PDO = {index_pdo}')

    # ---- Preparar dados para plotagem (lon em -180/180, ponto ciclico) ----
    ds_mean['lon'] = ((ds_mean['lon'] + 180) % 360) - 180
    da = ds_mean.sortby(ds_mean.lon)['anom']
    lon_vals = da['lon'].values
    lat_vals = da['lat'].values
    average_data, lon_cyclic = add_cyclic_point(da.values, coord=lon_vals)
    da_average_data = xr.DataArray(
        average_data, dims=('lat', 'lon'), coords={'lat': lat_vals, 'lon': lon_cyclic}
    )

    sst_levels = [float(x) for x in settings.LST_SSTA_NEW_GREC]
    cmap = LinearSegmentedColormap.from_list('sst_anom', settings.LST_ANOM_CORRETA)

    dt_ini = datetime.strptime(settings.DATA_INICIAL, '%Y-%m-%d').strftime('%d-%m-%y')
    dt_fim = datetime.strptime(settings.DATA_FINAL, '%Y-%m-%d').strftime('%d-%m-%y')

    output_dir.mkdir(parents=True, exist_ok=True)

    for area in lst_areas:
        logger.info(f'Gerando mapa para area: {area}')
        info_plot = settings['areas_plotagem']

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(
            1, 1, 1,
            projection=ccrs.PlateCarree(
                central_longitude=info_plot[area]['central_longitude_mapa']
            ),
        )

        if info_plot[area].get('plot_box', False):
            for box in info_plot[area]['lst_boxes']:
                ax.add_patch(patches.Rectangle(
                    (box['x_anc'], box['y_anc']),
                    box['x_larg'],
                    box['y_larg'],
                    linewidth=box['linewidth'],
                    edgecolor=box['edgecolor'],
                    facecolor='none',
                    zorder=100,
                ))

        ax.set_xlim([info_plot[area]['lon_esq'], info_plot[area]['lon_dir']])
        ax.set_ylim([info_plot[area]['lat_inf'], info_plot[area]['lat_sup']])

        ax.add_feature(cfeature.LAND.with_scale('50m'), linewidth=1.2, facecolor='whitesmoke', zorder=2)
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), linewidth=1.2, facecolor='white')
        ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1.2, zorder=100)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, zorder=100)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, zorder=100)

        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.0)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 18, 'color': 'black'}
        gl.ylabel_style = {'size': 18, 'color': 'black'}
        gl.xlocator = MultipleLocator(40)
        gl.ylocator = MultipleLocator(20)

        # Boxes e rotulos de todos os indices climaticos de TSM (fonte unica
        # compartilhada com a area `globo` do s12)
        desenha_boxes_indices(ax, da_average_data, index_pdo)

        # ------------------------------------------------------------------ #
        # Contourf SSTA
        # ------------------------------------------------------------------ #
        im = ax.contourf(
            lon_cyclic, lat_vals, average_data,
            levels=sst_levels,
            cmap=cmap,
            extend='both',
            transform=ccrs.PlateCarree(
                central_longitude=info_plot[area]['central_longitude_plot']
            ),
        )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
        cbar = plt.colorbar(im, cax=cax, pad=0.02, fraction=0.02375, ticks=sst_levels)
        cbar.set_label(label='[°C]', size=18)
        cbar.ax.tick_params(labelsize=10)

        titulo = f'Anomalia de TSM (De {dt_ini} a {dt_fim})'
        ax.set_title(titulo, fontsize=18, loc='left', pad=4)

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

        filename_fig = output_dir / f'ssta_globo_indices_{area}.png'
        logger.info(f'Salvando figura {filename_fig}')
        plt.savefig(str(filename_fig), dpi=fig.dpi, bbox_inches='tight')
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
