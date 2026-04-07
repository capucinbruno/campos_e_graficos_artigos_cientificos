# -*- coding: utf-8 -*-
"""
s03 - Anomalia de PSI200 (ERA5).

Gera mapas de anomalia da função de corrente / streamfunction em 200 hPa (PSI200),
com linhas de corrente do componente rotacional do vento, a partir de um arquivo
NetCDF previamente preparado pelo pipeline de processamento.

Fluxo esperado:
    1) Um módulo anterior (plot_psi200) baixa/processa os dados ERA5 de u/v em 200 hPa
       e salva um NetCDF final em settings.DIR_DADOS / "psi200.nc"
    2) Este script lê o NetCDF final e gera os mapas PNG por área.

Arquivo esperado de entrada:
    - {settings.DIR_DADOS}/psi200.nc

Variáveis aceitas no NetCDF:
    - PSI: psi_anom_mean_scaled, psi_anom_mean, psi, psi_anom, streamfunction_anomaly
    - UPSI: upsi_anom_mean, upsi, upsi_anom, urot, upsi_rot
    - VPSI: vpsi_anom_mean, vpsi, vpsi_anom, vrot, vpsi_rot

Saída:
    - Mapas PNG em {settings.DIR_OUTPUT}/s03_PSI200/

Criado em: 2026-04-01
"""

# ---------------------------------------------------------------------------
# Bibliotecas padrão
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
import numpy as np
from cartopy.util import add_cyclic_point
from matplotlib import patches
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import FixedLocator, MultipleLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from PIL import Image

# Módulos locais
# ---------------------------------------------------------------------------
from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.common.dataset_utils import load_dataset
from app.shared.logger import get_logger
from app.shared.settings_factory import settings

# IMPORTANTE:
# Este módulo deve ser o responsável por preparar/salvar o arquivo psi200.nc
from app.src.uteis.plot_psi200 import main as plot_psi200

# ---------------------------------------------------------------------------
# Identidade do script
# ---------------------------------------------------------------------------
SCRIPT_ID = Path(__file__).stem.split('_')[0]  # ex.: 's03'
SCRIPT_NAME = Path(__file__).stem
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
PSI_SCALE = 1e6

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
    'pacific_chile': {'density': 2.5, 'linewidth': 0.5},
    'pacifico_leste_america_sul': {'density': 2.5, 'linewidth': 0.5},
    'zona_zcit_atlantico': {'density': 2, 'linewidth': 0.5},
}

PSI_FILE_NAME = 'psi200.nc'

PSI_CANDIDATES = (
    'psi_anom_mean_scaled',
    'psi_anom_mean',
    'psi',
    'psi_anom',
    'streamfunction_anomaly',
)

UPSI_CANDIDATES = (
    'upsi_anom_mean',
    'upsi',
    'upsi_anom',
    'urot',
    'upsi_rot',
)

VPSI_CANDIDATES = (
    'vpsi_anom_mean',
    'vpsi',
    'vpsi_anom',
    'vrot',
    'vpsi_rot',
)

LST_ANOM_CORRETA_INVERTIDA_CHUVA = [
    "#910e00",
    "#a71400",
    "#bd1c00",
    "#d32400",
    "#e92a00",
    "#ff4e18",
    "#ff6c30",
    "#ff8b48",
    "#ff9a55",
    "#ffb76c",
    "#ffe391",
    "#ffffa9",
    "#ffffff",
    "#ffffff",
    "#b4eff9",
    "#8adbf2",
    "#61c7eb",
    "#38b2e4",
    "#0faedf",
    "#00a0d2",
    "#0092c5",
    "#0084b8",
    "#0075ab",
    "#00679e",
    "#005890",
    "#004a83",
]

# ---------------------------------------------------------------------------
# Funções utilitárias
# ---------------------------------------------------------------------------
def _pick_first_var(ds, candidates, *, required=True):
    for name in candidates:
        if name in ds.data_vars:
            return ds[name], name

    if required:
        raise KeyError(
            f'Nenhuma das variáveis {candidates} foi encontrada no dataset. '
            f'Disponíveis: {list(ds.data_vars)}'
        )
    return None, None


def _standardize_coords(da):
    """
    Padroniza lat/lon e remove dimensões extras.
    """
    ren = {}
    if 'latitude' in da.coords:
        ren['latitude'] = 'lat'
    if 'longitude' in da.coords:
        ren['longitude'] = 'lon'
    if ren:
        da = da.rename(ren)

    if 'lat' not in da.coords or 'lon' not in da.coords:
        raise ValueError("O campo precisa conter coordenadas 'lat' e 'lon'.")

    for dim in list(da.dims):
        if dim not in {'lat', 'lon'} and da.sizes[dim] == 1:
            da = da.isel({dim: 0}, drop=True)

    extra_dims = [dim for dim in da.dims if dim not in {'lat', 'lon'}]
    for dim in extra_dims:
        da = da.isel({dim: 0}, drop=True)

    if da.dims != ('lat', 'lon'):
        da = da.transpose('lat', 'lon')

    if float(da['lon'].min()) < 0:
        da = da.assign_coords(lon=((da['lon'] + 360) % 360))
        da = da.sortby('lon')

    if da['lat'][0] < da['lat'][-1]:
        da = da.sortby('lat', ascending=False)

    return da


def _prepare_psi_field(ds):
    """
    Obtém o campo PSI e garante escala final em 10^6 m² s^-1.
    """
    da_psi, psi_name = _pick_first_var(ds, PSI_CANDIDATES, required=True)
    da_psi = _standardize_coords(da_psi)

    if psi_name == 'psi_anom_mean_scaled':
        return da_psi

    return da_psi / PSI_SCALE


def _prepare_vector_field(ds, candidates):
    da, _ = _pick_first_var(ds, candidates, required=True)
    return _standardize_coords(da)


def _add_cyclic_2d(da2d):
    data_cyc, lon_cyc = add_cyclic_point(da2d.values, coord=da2d['lon'].values)
    return data_cyc, lon_cyc


def _add_cyclic_uv(u2d, v2d):
    u_cyc, lon_cyc = add_cyclic_point(u2d.values, coord=u2d['lon'].values)
    v_cyc, _ = add_cyclic_point(v2d.values, coord=v2d['lon'].values)
    return u_cyc, v_cyc, lon_cyc


def _prepare_streamline_lonuv(lon_cyc, u_cyc, v_cyc, central_lon_mapa):
    """Desloca lon/u/v para que a costura do array fique na borda do mapa.

    O streamplot integra trajetórias no espaço nativo dos eixos e não
    consegue cruzar descontinuidades no array.  Ao deslocar as longitudes
    em relação ao ``central_longitude_mapa`` de cada área, a costura
    (±180° relativo) cai sempre na borda do domínio visível — nunca no
    centro — eliminando a faixa sem linhas de corrente.
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



def _build_levels_ticks_norm(psi_values):
    psi = np.asarray(psi_values)

    if np.all((psi >= -50) & (psi <= 50)):
        levels = np.arange(-50, 53, 3)
        ticks = np.arange(-50, 60, 10)
    elif np.all((psi >= -100) & (psi <= 100)):
        levels = np.arange(-100, 110, 10)
        ticks = np.arange(-100, 120, 20)
    else:
        levels = np.arange(-200, 220, 20)
        ticks = np.arange(-200, 250, 50)

    n_bins = len(levels) - 1
    cmap = LinearSegmentedColormap.from_list(
        'psi200_red_white_blue',
        LST_ANOM_CORRETA_INVERTIDA_CHUVA,
        N=n_bins,
    )
    norm = BoundaryNorm(levels, ncolors=n_bins, clip=False)

    return levels, ticks, cmap, norm


def _get_area_list():
    """
    Prioriza uma lista específica para PSI200, mas cai para S01 se necessário.
    """
    if hasattr(settings, 'LST_AREAS_S03'):
        return list(settings.LST_AREAS_S03)

    if hasattr(settings, 'LST_AREAS_PSI200'):
        return list(settings.LST_AREAS_PSI200)

    if hasattr(settings, 'LST_AREAS_S01'):
        return list(settings.LST_AREAS_S01)

    raise AttributeError(
        'Nenhuma lista de áreas encontrada em settings '
        '(LST_AREAS_S03, LST_AREAS_PSI200 ou LST_AREAS_S01).'
    )


def _get_streamline_config(area: str) -> dict:
    """
    Prioridade: STREAMLINE_POR_AREA (script) > settings.PSI200_STREAMLINE_POR_AREA > STREAMLINE_DEFAULTS.
    """
    cfg = dict(STREAMLINE_DEFAULTS)

    global_overrides = getattr(settings, 'PSI200_STREAMLINE_DEFAULTS', None)
    if global_overrides and isinstance(global_overrides, dict):
        cfg.update(global_overrides)

    per_area_settings = getattr(settings, 'PSI200_STREAMLINE_POR_AREA', None)
    if per_area_settings and isinstance(per_area_settings, dict) and area in per_area_settings:
        cfg.update(per_area_settings[area])

    if area in STREAMLINE_POR_AREA:
        cfg.update(STREAMLINE_POR_AREA[area])

    return cfg


def _add_logo_to_map(ax, logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=30):
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

    if area in {'hemisferio_sul', 'psa', 'globo', 'mjo'}:
        gl.xlocator = MultipleLocator(40)
        gl.ylocator = MultipleLocator(20)

    elif area == 'estados_unidos':
        gl.xlocator = MultipleLocator(10)
        gl.ylocator = MultipleLocator(10)

    elif area == 'estados_unidos_zoom':
        gl.xlocator = MultipleLocator(10)
        gl.ylocator = MultipleLocator(5)

    elif area in {'argentina', 'costa_brasil', 'tsa', 'tna'}:
        gl.xlocator = MultipleLocator(5 if area in {'argentina', 'costa_brasil'} else 10)
        gl.ylocator = MultipleLocator(5)

    elif area == 'brasil':
        gl.xlocator = MultipleLocator(10)
        gl.ylocator = MultipleLocator(5)

    elif area in {'america_sul', 'africa'}:
        gl.xlocator = MultipleLocator(20)
        gl.ylocator = MultipleLocator(20)

    elif area in {
        'zona_zcit_atlantico',
        'atlantico_tropical',
        'pdo',
        'iod',
        'sad',
        'amo',
        'africa_monsoon',
    }:
        gl.xlocator = MultipleLocator(
            20 if area in {'atlantico_tropical', 'pdo', 'iod', 'sad', 'amo', 'africa_monsoon'} else 10
        )
        gl.ylocator = MultipleLocator(
            10 if area in {'atlantico_tropical', 'pdo', 'iod', 'sad', 'amo', 'africa_monsoon'} else 5
        )

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
    logger.info('📊 SCRIPT %s: %s', SCRIPT_ID.upper(), SCRIPT_DESC)
    logger.info('=' * 80)

    lst_areas = _get_area_list()

    output_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_PSI200'
    input_dir = Path(settings.DIR_INPUT)
    dados_dir = Path(settings.DIR_DADOS)

    cache_params = {
        'DATA_INICIAL': settings.DATA_INICIAL,
        'DATA_FINAL': settings.DATA_FINAL,
        'areas': lst_areas,
        'script_version': '1.1',
        'psi_file': PSI_FILE_NAME,
        'streamline_defaults': STREAMLINE_DEFAULTS,
        'streamline_por_area': getattr(settings, 'PSI200_STREAMLINE_POR_AREA', {}),
        'palette': LST_ANOM_CORRETA_INVERTIDA_CHUVA,
        'palette_mode': 'dynamic_boundarynorm',
    }
    output_files = [str(output_dir / f'psi200_{area}.png') for area in lst_areas]

    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('🎯 CACHE VÁLIDO! Execução já foi realizada com os mesmos parâmetros.')
        logger.info('   📅 Período: %s a %s', settings.DATA_INICIAL, settings.DATA_FINAL)
        logger.info('   📊 %d mapas já existem', len(output_files))
        logger.info('   📁 Diretório: %s', output_dir)
        logger.info('   ⏭️  Pulando execução')
        return

    start_time = time.time()
    logger.info('📅 Período de análise: %s a %s', settings.DATA_INICIAL, settings.DATA_FINAL)
    logger.info('📊 Gerando %d mapas de anomalia PSI200', len(lst_areas))
    logger.info('📌 Paleta fixa vermelho -> branco -> azul (%d cores)', len(LST_ANOM_CORRETA_INVERTIDA_CHUVA))
    logger.info('=' * 80)

    try:
        plot_psi200()
    except Exception as err:
        logger.exception('Falha no download/processamento do PSI200')
        raise RuntimeError('Falha no download/processamento do PSI200') from err

    output_dir.mkdir(parents=True, exist_ok=True)

    psi_file = dados_dir / PSI_FILE_NAME
    if not psi_file.exists():
        raise FileNotFoundError(
            f'Arquivo esperado não encontrado: {psi_file}. '
            'A rotina plot_psi200() precisa salvar esse NetCDF antes da plotagem.'
        )

    ds = load_dataset(str(psi_file))

    da_psi = _prepare_psi_field(ds)
    da_upsi = _prepare_vector_field(ds, UPSI_CANDIDATES)
    da_vpsi = _prepare_vector_field(ds, VPSI_CANDIDATES)

    if not (
        np.array_equal(da_psi['lat'].values, da_upsi['lat'].values)
        and np.array_equal(da_psi['lon'].values, da_upsi['lon'].values)
        and np.array_equal(da_psi['lat'].values, da_vpsi['lat'].values)
        and np.array_equal(da_psi['lon'].values, da_vpsi['lon'].values)
    ):
        raise ValueError('PSI, UPSI e VPSI não estão na mesma grade lat/lon.')

    lat = da_psi['lat'].values

    psi_cyc, lon_cyc = _add_cyclic_2d(da_psi)
    upsi_cyc, vpsi_cyc, lon_cyc_uv = _add_cyclic_uv(da_upsi, da_vpsi)

    levels, ticks, cmap, norm = _build_levels_ticks_norm(da_psi.values)

    info_plot = settings['areas_plotagem']

    for area in lst_areas:
        logger.info('🖼️  Gerando mapa PSI200 para área: %s', area)

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

        if area == 'enso':
            for txt, x, y, cor in [
                ('Niño 1+2', 66.25, -13.64, 'red'),
                ('Niño 3', 34.1, 8.45, 'blue'),
                ('Niño 3.4', 8.6, -9.45, 'black'),
                ('Niño 4', -22.5, 8.45, 'm'),
            ]:
                t = plt.text(x, y, txt, fontsize=14, color=cor, weight='bold')
                fg = 'black' if cor in {'red', 'm'} else 'white'
                t.set_path_effects([
                    path_effects.Stroke(linewidth=3, foreground=fg),
                    path_effects.Normal(),
                ])

        if area == 'mjo':
            phase_positions = [
                (-135.25, -4.64, '1'),
                (-115.25, -4.64, '2'),
                (-95.25, -4.64, '3'),
                (-75.25, -4.64, '4'),
                (-55.25, -4.64, '5'),
                (-35.25, -4.64, '6'),
                (-15.25, -4.64, '7'),
                (4.75, -4.64, '8'),
            ]
            for x, y, txt in phase_positions:
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

        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.0)
        _configure_gridlines(gl, area)

        ax.set_xlim([info_plot[area]['lon_esq'], info_plot[area]['lon_dir']])
        ax.set_ylim([info_plot[area]['lat_inf'], info_plot[area]['lat_sup']])

        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, edgecolor='black')
        ax.add_feature(cfeature.LAND.with_scale('50m'), linewidth=0.5, facecolor='whitesmoke')

        if area != 'china':
            ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1.2, edgecolor='black', zorder=100)

        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, edgecolor='black', zorder=100)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, edgecolor='black', zorder=100)
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), linewidth=0.5, facecolor='white')

        cf = ax.contourf(
            lon_cyc,
            lat,
            psi_cyc,
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend='both',
            transform=ccrs.PlateCarree(
                central_longitude=info_plot[area]['central_longitude_plot']
            ),
            zorder=2,
        )

        ax.contour(
            lon_cyc,
            lat,
            psi_cyc,
            levels=levels,
            colors='white',
            linewidths=0.6,
            alpha=0.55,
            transform=ccrs.PlateCarree(
                central_longitude=info_plot[area]['central_longitude_plot']
            ),
            zorder=3,
        )

        slcfg = _get_streamline_config(area)

        central_lon = info_plot[area]['central_longitude_mapa']
        lon_sl, u_sl, v_sl = _prepare_streamline_lonuv(
            lon_cyc_uv, upsi_cyc, vpsi_cyc, central_lon,
        )

        ax.streamplot(
            lon_sl,
            lat,
            u_sl,
            v_sl,
            transform=ccrs.PlateCarree(central_longitude=central_lon),
            density=float(slcfg['density']),
            linewidth=float(slcfg['linewidth']),
            arrowsize=float(slcfg['arrowsize']),
            color=slcfg['color'],
            zorder=5,
        )

        if area in {'enso', 'tropico', 'MDR', 'hemisferio_sul', 'psa'}:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)

            cbar = plt.colorbar(
                cf,
                cax=cax,
                pad=0.02,
                fraction=0.02375,
                location='bottom',
                extend='both',
                orientation='horizontal',
                ticks=ticks,
                boundaries=levels,
                spacing='proportional',
            )
        else:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)

            cbar = plt.colorbar(
                cf,
                cax=cax,
                pad=0.02,
                fraction=0.02375,
                extend='both',
                ticks=ticks,
                boundaries=levels,
                spacing='proportional',
            )

        cbar.minorticks_off()
        cbar.set_label(
            label=r'10$^6$ m$^2$ s$^{-1}$',
            size=18,
        )
        cbar.ax.tick_params(labelsize=20)

        dt_ini = datetime.strptime(settings.DATA_INICIAL, '%Y-%m-%d').strftime('%d-%m-%y')
        dt_fim = datetime.strptime(settings.DATA_FINAL, '%Y-%m-%d').strftime('%d-%m-%y')
        titulo = f'Anomalia de PSI200 (De {dt_ini} a {dt_fim})'
        ax.set_title(titulo, fontsize=18, loc='left')

        filename_fig = output_dir / f'psi200_{area}.png'
        logger.info('Salvando a figura %s', filename_fig)

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

        plt.savefig(
            str(filename_fig),
            dpi=fig.dpi,
            bbox_inches='tight',
        )
        plt.close('all')

    execution_time = time.time() - start_time
    save_cache_metadata(SCRIPT_ID, cache_params, output_files, execution_time)

    logger.info('=' * 80)
    logger.info('✅ Script %s concluído com sucesso!', SCRIPT_ID.upper())
    logger.info('⏱️  Tempo de execução: %.1fs (%.1f min)', execution_time, execution_time / 60)
    logger.info('📊 %d mapas gerados em: %s', len(output_files), output_dir)
    logger.info('=' * 80)


if __name__ == '__main__':
    main()