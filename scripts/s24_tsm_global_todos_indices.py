"""s24 - Anomalia de TSM Global com todos os Indices Climaticos.

Baixa dados de anomalia de TSM do PSL/NOAA (OISSTv2 High-Res),
calcula a media do periodo selecionado e gera mapa global com os indices
ENSO (Nino 1+2, 3, 3.4, 4), IOD, AMO, TNA, TSA, SAD e PDO sobrepostos.

Dados de entrada:
    - PSL/NOAA: sst.day.anom.{ano}.nc (OISSTv2 0.25 grau, um arquivo por ano)
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
import warnings
from datetime import datetime
from pathlib import Path

# Bibliotecas de terceiros
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    'https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.day.anom.{year}.nc'
)
SST_FILE_TEMPLATE = 'sst.day.anom.{year}.nc'


# ---------------------------------------------------------------------------
# Funções auxiliares para cálculo do índice PDO
# ---------------------------------------------------------------------------

def interpolation_dataset(dataset, lst_lon: list, lst_lat: list):
    """Interpola o dataset para a grade especificada."""
    try:
        dataset = dataset.interp(lon=lst_lon, lat=lst_lat, method='linear')
        dataset = dataset.interpolate_na(dim='lon', method='linear', fill_value='extrapolate')
        dataset = dataset.interpolate_na(dim='lat', method='linear', fill_value='extrapolate')
    except ValueError:
        dataset = dataset.interp(longitude=lst_lon, latitude=lst_lat, method='linear')
        dataset = dataset.interpolate_na(dim='longitude', method='linear', fill_value='extrapolate')
    return dataset


def convert_dataset_to_df(ds_input: xr.Dataset, nome_variavel: str) -> pd.DataFrame:
    """Converte Dataset para DataFrame com colunas (lon, lat, variavel)."""
    df = ds_input.to_dataframe()
    df = (pd.DataFrame(df.index.to_list(), df[nome_variavel])).reset_index(drop=False)
    df = df[[0, 1, nome_variavel]]
    if min(df[0]) > min(df[1]):
        df.rename(columns={0: 'lat', 1: 'lon'}, inplace=True)
        df = df[['lon', 'lat', nome_variavel]]
    else:
        df.rename(columns={0: 'lon', 1: 'lat'}, inplace=True)
    return df.sort_values(by=['lon', 'lat']).reset_index(drop=True).round(2)


def calcula_indice_pdo(ds: xr.Dataset, name_var: str) -> float:
    """Calcula o indice PDO com base na projecao dos dados de SSTA na EOF1."""
    warnings.filterwarnings('ignore', category=UserWarning)

    file_eof1 = Path('Entrada/pdo_eof/EOF1.csv')
    df_eof1 = pd.read_csv(file_eof1)

    lst_lon, lst_lat, lst_values_eof = [], [], []
    for _, row in df_eof1.iterrows():
        parts = row.iloc[0].split(';')
        lst_lon.append(float(parts[0]))
        lst_lat.append(float(parts[1]))
        lst_values_eof.append(float(parts[2]))

    lst_lon_eof1 = np.array(list(set(lst_lon)))
    lst_lat_eof1 = np.array(list(set(lst_lat)))

    df_eof = pd.DataFrame({'lat': lst_lat, 'lon': lst_lon, 'var1': lst_values_eof})

    X = np.empty((0, 3739))
    try:
        ds = ds.assign_coords(longitude=(ds.longitude % 360)).sortby('longitude')
    except Exception:
        ds = ds.assign_coords(lon=(ds.lon % 360)).sortby('lon')

    ds = interpolation_dataset(ds, lst_lon_eof1, lst_lat_eof1)
    df = convert_dataset_to_df(ds, name_var)
    df = df.dropna(axis=0)
    df.rename(columns={'lat': 'lon', 'lon': 'lat', name_var: 'anom'}, inplace=True)
    df_concat = pd.merge(df_eof, df, on=['lat', 'lon'], how='inner')

    linha_unica = df_concat[['anom']].values.flatten()
    X = np.concatenate([X, linha_unica.reshape(1, -1)], axis=0)

    media = np.mean(X[0])
    desvio = np.std(X[0])
    dados_pad = (X - media) / desvio

    eof1_reshape = np.array(lst_values_eof).reshape(-1, 1)
    indice = np.dot(dados_pad[0].reshape(1, -1), eof1_reshape).item()

    desvio_hist = 1101.995133699693
    media_hist = 1.4019186154496004e-14
    return (indice - media_hist) / desvio_hist


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

    lst_areas = ['globo_indices']
    output_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_SSTA_GLOBO_INDICES'
    input_dir = Path(settings.DIR_INPUT)
    dados_dir = Path(settings.DIR_DADOS)

    cache_params = {
        'DATA_INICIAL': settings.DATA_INICIAL,
        'DATA_FINAL': settings.DATA_FINAL,
        'areas': lst_areas,
        'script_version': '1.0',
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
    ds_mean = ds_raw.mean(dim='time').compute()
    ds_raw.close()

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

        # ------------------------------------------------------------------ #
        # IOD
        # ------------------------------------------------------------------ #
        lon_min_iod_1, lon_max_iod_1, lat_min_iod_1, lat_max_iod_1 = 50, 70, -10, 10
        lon_min_iod_2, lon_max_iod_2, lat_min_iod_2, lat_max_iod_2 = 90, 110, -10, 0

        for lon_a, lon_b, lat_a, lat_b in [
            (lon_min_iod_1, lon_max_iod_1, lat_min_iod_1, lat_max_iod_1),
            (lon_min_iod_2, lon_max_iod_2, lat_min_iod_2, lat_max_iod_2),
        ]:
            ax.plot(
                [lon_a, lon_b, lon_b, lon_a, lon_a],
                [lat_a, lat_a, lat_b, lat_b, lat_a],
                color='black', linewidth=2, zorder=100, transform=ccrs.PlateCarree(),
            )

        sst_iod_oc = da_average_data.sel(
            lon=slice(lon_min_iod_1, lon_max_iod_1), lat=slice(lat_min_iod_1, lat_max_iod_1)
        ).mean(dim=('lon', 'lat'))
        sst_iod_or = da_average_data.sel(
            lon=slice(lon_min_iod_2, lon_max_iod_2), lat=slice(lat_min_iod_2, lat_max_iod_2)
        ).mean(dim=('lon', 'lat'))
        index_iod = round(float(sst_iod_oc) - float(sst_iod_or), 2)

        t = plt.text(-153.25, -18.64, f'IOD = {index_iod}', fontsize=10, color='white', weight='bold')
        t.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

        # ------------------------------------------------------------------ #
        # Niño 1+2
        # ------------------------------------------------------------------ #
        lon_min_12, lon_max_12, lat_min_12, lat_max_12 = -90, -80, -10, 0
        ax.plot(
            [lon_min_12, lon_max_12, lon_max_12, lon_min_12, lon_min_12],
            [lat_min_12, lat_min_12, lat_max_12, lat_max_12, lat_min_12],
            color='limegreen', linewidth=2, zorder=100, transform=ccrs.PlateCarree(),
        )
        val_12 = round(float(da_average_data.sel(
            lon=slice(lon_min_12, lon_max_12), lat=slice(lat_min_12, lat_max_12)
        ).mean(dim=('lon', 'lat'))), 2)
        t = plt.text(35.80, -17.64, f'Nino 1+2 = {val_12}', fontsize=10, color='limegreen', weight='bold', zorder=100)
        t.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

        # ------------------------------------------------------------------ #
        # Niño 3
        # ------------------------------------------------------------------ #
        lon_min_3, lon_max_3, lat_min_3, lat_max_3 = -149.5, -90, -5, 5
        ax.plot(
            [lon_min_3, lon_max_3, lon_max_3, lon_min_3, lon_min_3],
            [lat_min_3, lat_min_3, lat_max_3, lat_max_3, lat_min_3],
            color='blue', linewidth=2, zorder=100, transform=ccrs.PlateCarree(),
        )
        val_3 = round(float(da_average_data.sel(
            lon=slice(lon_min_3, lon_max_3), lat=slice(lat_min_3, lat_max_3)
        ).mean(dim=('lon', 'lat'))), 2)
        t = plt.text(4.01, 8.64, f'Nino 3 = {val_3}', fontsize=10, color='blue', weight='bold', zorder=100)
        t.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])

        # ------------------------------------------------------------------ #
        # Niño 3.4
        # ------------------------------------------------------------------ #
        lon_min_34, lon_max_34, lat_min_34, lat_max_34 = -170, -120, -5, 5
        ax.plot(
            [lon_min_34, lon_max_34, lon_max_34, lon_min_34, lon_min_34],
            [lat_min_34, lat_min_34, lat_max_34, lat_max_34, lat_min_34],
            color='black', linewidth=2, zorder=1000, transform=ccrs.PlateCarree(),
        )
        val_34 = round(float(da_average_data.sel(
            lon=slice(lon_min_34, lon_max_34), lat=slice(lat_min_34, lat_max_34)
        ).mean(dim=('lon', 'lat'))), 2)
        t = plt.text(-23.03, -11.64, f'Nino 3.4 = {val_34}', fontsize=10, color='white', weight='bold')
        t.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

        # ------------------------------------------------------------------ #
        # Niño 4
        # ------------------------------------------------------------------ #
        lon_min_4, lon_max_4, lat_min_4, lat_max_4 = 160, 209.5, -5, 5
        ax.plot(
            [lon_min_4, lon_max_4, lon_max_4, lon_min_4, lon_min_4],
            [lat_min_4, lat_min_4, lat_max_4, lat_max_4, lat_min_4],
            color='magenta', linewidth=2, transform=ccrs.PlateCarree(),
        )
        val_4 = round(float(da_average_data.sel(
            lon=slice(lon_min_4, lon_max_4), lat=slice(lat_min_4, lat_max_4)
        ).mean(dim=('lon', 'lat'))), 2)
        t = plt.text(-51.65, 8.64, f'Nino 4 = {val_4}', fontsize=10, color='magenta', weight='bold')
        t.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

        # ------------------------------------------------------------------ #
        # AMO
        # ------------------------------------------------------------------ #
        lon_min_amo, lon_max_amo, lat_min_amo, lat_max_amo = -70, -7, 1, 70
        ax.plot(
            [lon_min_amo, lon_max_amo, lon_max_amo, lon_min_amo, lon_min_amo],
            [lat_min_amo, lat_min_amo, lat_max_amo, lat_max_amo, lat_min_amo],
            color='black', linewidth=2, zorder=100, transform=ccrs.PlateCarree(),
        )
        val_amo = round(float(da_average_data.sel(
            lon=slice(lon_min_amo, lon_max_amo), lat=slice(lat_min_amo, lat_max_amo)
        ).mean(dim=('lon', 'lat'))), 2)
        t = plt.text(88.25, 32.15, f'AMO = {val_amo}', fontsize=10, color='black', weight='bold', zorder=100)
        t.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])

        # ------------------------------------------------------------------ #
        # TNA
        # ------------------------------------------------------------------ #
        lon_min_tna, lon_max_tna, lat_min_tna, lat_max_tna = -55, -15, 5, 25
        ax.plot(
            [lon_min_tna, lon_max_tna, lon_max_tna, lon_min_tna, lon_min_tna],
            [lat_min_tna, lat_min_tna, lat_max_tna, lat_max_tna, lat_min_tna],
            color='black', linewidth=2, zorder=100, transform=ccrs.PlateCarree(),
        )
        val_tna = round(float(da_average_data.sel(
            lon=slice(lon_min_tna, lon_max_tna), lat=slice(lat_min_tna, lat_max_tna)
        ).mean(dim=('lon', 'lat'))), 2)
        t = plt.text(91.25, 13.15, f'TNA = {val_tna}', fontsize=10, color='black', weight='bold', zorder=100)
        t.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])

        # ------------------------------------------------------------------ #
        # TSA
        # ------------------------------------------------------------------ #
        lon_min_tsa, lon_max_tsa, lat_min_tsa, lat_max_tsa = -30, 10, -20, 0
        ax.plot(
            [lon_min_tsa, lon_max_tsa, lon_max_tsa, lon_min_tsa, lon_min_tsa],
            [lat_min_tsa, lat_min_tsa, lat_max_tsa, lat_max_tsa, lat_min_tsa],
            color='blue', linewidth=2, zorder=100, transform=ccrs.PlateCarree(),
        )
        val_tsa = round(float(da_average_data.sel(
            lon=slice(lon_min_tsa, lon_max_tsa), lat=slice(lat_min_tsa, lat_max_tsa)
        ).mean(dim=('lon', 'lat'))), 2)
        t = plt.text(116.25, -12.15, f'TSA = {val_tsa}', fontsize=10, color='blue', weight='bold', zorder=1000)
        t.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])

        # ------------------------------------------------------------------ #
        # SAD
        # ------------------------------------------------------------------ #
        lon_min_sad_1, lon_max_sad_1, lat_min_sad_1, lat_max_sad_1 = -20, 9, -15, -1
        lon_min_sad_2, lon_max_sad_2, lat_min_sad_2, lat_max_sad_2 = -40, -10, -40, -25

        for lon_a, lon_b, lat_a, lat_b in [
            (lon_min_sad_1, lon_max_sad_1, lat_min_sad_1, lat_max_sad_1),
            (lon_min_sad_2, lon_max_sad_2, lat_min_sad_2, lat_max_sad_2),
        ]:
            ax.plot(
                [lon_a, lon_b, lon_b, lon_a, lon_a],
                [lat_a, lat_a, lat_b, lat_b, lat_a],
                color='black', linewidth=2, zorder=100, transform=ccrs.PlateCarree(),
            )

        val_sad_n = float(da_average_data.sel(
            lon=slice(lon_min_sad_1, lon_max_sad_1), lat=slice(lat_min_sad_1, lat_max_sad_1)
        ).mean(dim=('lon', 'lat')))
        val_sad_s = float(da_average_data.sel(
            lon=slice(lon_min_sad_2, lon_max_sad_2), lat=slice(lat_min_sad_2, lat_max_sad_2)
        ).mean(dim=('lon', 'lat')))
        index_sad = round(val_sad_n - val_sad_s, 2)

        t = plt.text(115.25, -22.15, f'SAD = {index_sad}', fontsize=10, color='white', weight='bold', zorder=100)
        t.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

        # ------------------------------------------------------------------ #
        # PDO
        # ------------------------------------------------------------------ #
        lon_min_pdo, lon_max_pdo, lat_min_pdo, lat_max_pdo = 140, 240, 20, 60
        ax.plot(
            [lon_min_pdo, lon_max_pdo, lon_max_pdo, lon_min_pdo, lon_min_pdo],
            [lat_min_pdo, lat_min_pdo, lat_max_pdo, lat_max_pdo, lat_min_pdo],
            color='black', linewidth=2, zorder=100, transform=ccrs.PlateCarree(),
        )
        t = plt.text(-44.03, 37.00, f'PDO = {index_pdo}', fontsize=10, color='black', weight='bold', zorder=100)
        t.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])

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
