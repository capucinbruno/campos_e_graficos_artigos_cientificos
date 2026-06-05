# -*- coding: utf-8 -*-
"""
s02 - Anomalia de CHI200 (ERA5).

Gera mapas de anomalia da função corrente / velocity potential em 200 hPa (CHI200),
com vetores do componente divergente/irrotacional do vento, a partir de um arquivo
NetCDF previamente preparado pelo pipeline de processamento.

Fluxo esperado:
    1) Um módulo anterior (plot_chi200) baixa/processa os dados ERA5 de u/v em 200 hPa
       e salva um NetCDF final em settings.DIR_DADOS / "chi200.nc"
    2) Este script lê o NetCDF final e gera os mapas PNG por área.

Arquivo esperado de entrada:
    - {settings.DIR_DADOS}/chi200.nc

Variáveis aceitas no NetCDF:
    - CHI: chi_anom_mean_scaled, chi_anom_mean, chi, chi_anom, velocity_potential_anomaly
    - UCHI: uchi_anom_mean, uchi, uchi_anom, udiv, uchi_irrot
    - VCHI: vchi_anom_mean, vchi, vchi_anom, vdiv, vchi_irrot

Saída:
    - Mapas PNG em {settings.DIR_OUTPUT}/s02_CHI200/

Criado em: 2026-03-31
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
import matplotlib.path as mpath
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from cartopy.util import add_cyclic_point
from matplotlib import patches
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import FixedLocator, MultipleLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image

# Módulos locais
# ---------------------------------------------------------------------------
from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.common.dataset_utils import area_display_name, load_dataset
from app.shared.logger import get_logger
from app.shared.settings_factory import settings

# IMPORTANTE:
# Este módulo deve ser o responsável por preparar/salvar o arquivo chi200.nc
# Se no seu projeto o nome for diferente, ajuste aqui.
from app.src.uteis.plot_chi200 import main as plot_chi200

# ---------------------------------------------------------------------------
# Identidade do script
# ---------------------------------------------------------------------------
SCRIPT_ID = Path(__file__).stem.split('_')[0]  # ex.: 's02'
SCRIPT_NAME = Path(__file__).stem
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
CHI_SCALE = 1e5

LEVELS = np.arange(-60, 65, 5)

QUIVER_DEFAULTS = {
    'scale': 100,
    'width': 0.0013,
    'step': 15,
    'min_mag': 0.5,
    'headwidth': 3.2,
    'headlength': 4.2,
    'headaxislength': 3.8,
    'color': 'black',
}

# Ajuste de densidade (step) e outros parâmetros de quiver por área.
# step menor = mais setas. Valores ausentes usam QUIVER_DEFAULTS.
QUIVER_POR_AREA = {
    'brasil': {'step': 2, 'scale': 30, 'width': 0.004},
    'america_sul': {'step': 3, 'scale': 50, 'width': 0.003},
    'america_sul_zom_out': {'step': 4, 'scale': 80, 'width': 0.002},
    'argentina': {'step': 1, 'scale': 20, 'width': 0.004},
    'costa_brasil': {'step': 1, 'scale': 20, 'width': 0.003},
    'hemisferio_sul': {'step': 5, 'scale': 100, 'width': 0.0013},
    'psa': {'step': 5, 'scale': 95, 'width': 0.0013},
    'globo': {'step': 5, 'scale': 95, 'width': 0.0017},
    'tropico': {'step': 4, 'scale': 100, 'width': 0.0013},
    'enso': {'step': 3, 'scale': 100, 'width': 0.0013},
    'mjo': {'step': 4, 'scale': 80, 'width': 0.002},
    'atlantico_tropical': {'step': 2, 'scale': 50, 'width': 0.0018},
    'africa': {'step': 3, 'scale': 50, 'width': 0.003},
    'africa_monsoon': {'step': 3, 'scale': 50, 'width': 0.003},
    'china': {'step': 2, 'scale': 100, 'width': 0.0016},
    'estados_unidos': {'step': 2, 'scale': 50, 'width': 0.003},
    'estados_unidos_zoom': {'step': 2, 'scale': 50, 'width': 0.002},
    'tsa': {'step': 2, 'scale': 50, 'width': 0.002},
    'tna': {'step': 2, 'scale': 50, 'width': 0.002},
    'iod': {'step': 2, 'scale': 50, 'width': 0.002},
    'pdo': {'step': 3, 'scale': 50, 'width': 0.002},
    'sad': {'step': 2, 'scale': 60, 'width': 0.002},
    'amo': {'step': 2, 'scale': 60, 'width': 0.002},
    'MDR': {'step': 2, 'scale': 50, 'width': 0.002},
    'pacific_chile': {'step': 3, 'scale': 50, 'width': 0.002},
    'pacifico_leste_america_sul': {'step': 3, 'scale': 100, 'width': 0.002},
    'zona_zcit_atlantico': {'step': 1, 'scale': 60, 'width': 0.002},
    'globo_3d': {'step': 4, 'scale': 70, 'width': 0.002},
}

CHI_FILE_NAME = 'chi200.nc'

CHI_CANDIDATES = (
    'chi_anom_mean_scaled',
    'chi_anom_mean',
    'chi',
    'chi_anom',
    'velocity_potential_anomaly',
)

UCHI_CANDIDATES = (
    'uchi_anom_mean',
    'uchi',
    'uchi_anom',
    'udiv',
    'uchi_irrot',
)

VCHI_CANDIDATES = (
    'vchi_anom_mean',
    'vchi',
    'vchi_anom',
    'vdiv',
    'vchi_irrot',
)

# Paleta fixa verde -> claro -> bege -> marrom
CHI200_COLORS = [
    '#005a45',
    '#0f7a6c',
    '#2e9b96',
    '#62bdb7',
    '#9dd8d2',
    '#dff3f1',
    '#f7f4eb',
    '#e7d9a9',
    '#d6b566',
    '#bd8a35',
    '#9a6313',
    '#6f4300',
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

    # Remove dimensões singleton extras, se existirem
    for dim in list(da.dims):
        if dim not in {'lat', 'lon'} and da.sizes[dim] == 1:
            da = da.isel({dim: 0}, drop=True)

    # Se ainda sobrou alguma dimensão além de lat/lon, tenta pegar o primeiro índice
    extra_dims = [dim for dim in da.dims if dim not in {'lat', 'lon'}]
    for dim in extra_dims:
        da = da.isel({dim: 0}, drop=True)

    if da.dims != ('lat', 'lon'):
        da = da.transpose('lat', 'lon')

    # longitude para 0..360
    if float(da['lon'].min()) < 0:
        da = da.assign_coords(lon=((da['lon'] + 360) % 360))
        da = da.sortby('lon')

    # latitude decrescente (N -> S)
    if da['lat'][0] < da['lat'][-1]:
        da = da.sortby('lat', ascending=False)

    return da


def _prepare_chi_field(ds):
    """
    Obtém o campo CHI e garante escala final em 10^5 m² s^-1.
    """
    da_chi, chi_name = _pick_first_var(ds, CHI_CANDIDATES, required=True)
    da_chi = _standardize_coords(da_chi)

    # Se já vier escalado, mantém.
    if chi_name == 'chi_anom_mean_scaled':
        return da_chi

    # Caso contrário, escala para 10^5 m² s^-1
    return da_chi / CHI_SCALE


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


def _prepare_quiver_masked(lon, lat, u, v, *, step, min_mag, lat_bounds=None, margin=3.0):
    """Amostra, filtra magnitude fraca e mascara vetores fora dos limites da área.

    Parameters
    ----------
    lat_bounds : tuple (lat_inf, lat_sup) ou None
        Se fornecido, mascara vetores fora desse intervalo.
    margin : float
        Recuo em graus para dentro dos limites da área, evitando setas
        cortadas/distorcidas nas bordas do mapa.
    """
    lon_q = lon[::step]
    lat_q = lat[::step]
    u_q = u[::step, ::step]
    v_q = v[::step, ::step]

    mag_q = np.sqrt(u_q**2 + v_q**2)
    mask = mag_q < min_mag

    # Mascara vetores fora dos limites lat da área (recuados pela margem)
    if lat_bounds is not None:
        lat_min, lat_max = sorted(lat_bounds)
        lat_grid = lat_q[:, None] if lat_q.ndim == 1 else lat_q
        outside_lat = (lat_grid < lat_min + margin) | (lat_grid > lat_max - margin)
        if lat_q.ndim == 1:
            outside_lat = np.broadcast_to(outside_lat, u_q.shape)
        mask = mask | outside_lat

    u_q_mask = np.ma.masked_where(mask, u_q)
    v_q_mask = np.ma.masked_where(mask, v_q)

    return lon_q, lat_q, u_q_mask, v_q_mask


def _build_levels_ticks_norm():
    """
    Monta levels, ticks, cmap e norm a partir da constante LEVELS.
    Interpola a paleta CHI200_COLORS para o número de faixas necessário.
    """
    n_bins = len(LEVELS) - 1
    cmap = LinearSegmentedColormap.from_list(
        'chi200_green_brown', CHI200_COLORS, N=n_bins,
    )
    norm = BoundaryNorm(LEVELS, ncolors=n_bins, clip=False)
    ticks = np.arange(-60, 65, 10)

    return LEVELS, ticks, cmap, norm


def _get_area_list():
    """
    Prioriza uma lista específica para CHI200, mas cai para S01 se necessário.
    """
    if hasattr(settings, 'LST_AREAS_S02'):
        return list(settings.LST_AREAS_S02)

    if hasattr(settings, 'LST_AREAS_CHI200'):
        return list(settings.LST_AREAS_CHI200)

    if hasattr(settings, 'LST_AREAS_S01'):
        return list(settings.LST_AREAS_S01)

    raise AttributeError(
        'Nenhuma lista de áreas encontrada em settings '
        '(LST_AREAS_S02, LST_AREAS_CHI200 ou LST_AREAS_S01).'
    )


def _get_quiver_config(area: str) -> dict:
    """Retorna configuracao de quiver para uma area especifica.

    Prioridade: QUIVER_POR_AREA (script) > settings.CHI200_QUIVER_POR_AREA > QUIVER_DEFAULTS.
    """
    cfg = dict(QUIVER_DEFAULTS)

    global_overrides = getattr(settings, 'CHI200_QUIVER_DEFAULTS', None)
    if global_overrides and isinstance(global_overrides, dict):
        cfg.update(global_overrides)

    per_area_settings = getattr(settings, 'CHI200_QUIVER_POR_AREA', None)
    if per_area_settings and isinstance(per_area_settings, dict) and area in per_area_settings:
        cfg.update(per_area_settings[area])

    if area in QUIVER_POR_AREA:
        cfg.update(QUIVER_POR_AREA[area])

    return cfg


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger = get_logger(SCRIPT_ID)

    logger.info('=' * 80)
    logger.info(f'📊 SCRIPT {SCRIPT_ID.upper()}: {SCRIPT_DESC}')
    logger.info('=' * 80)

    lst_areas = _get_area_list()

    output_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_CHI200'
    input_dir = Path(settings.DIR_INPUT)
    dados_dir = Path(settings.DIR_DADOS)

    cache_params = {
        'DATA_INICIAL': settings.DATA_INICIAL,
        'DATA_FINAL': settings.DATA_FINAL,
        'areas': lst_areas,
        'script_version': '2.0',  # pipeline híbrido ERA5/GDAS 200 hPa + PSL clim + streaming
        'chi_file': CHI_FILE_NAME,
        'quiver_defaults': QUIVER_DEFAULTS,
        'quiver_por_area': getattr(settings, 'CHI200_QUIVER_POR_AREA', {}),
        'palette': CHI200_COLORS,
        'palette_mode': 'discrete_listedcolormap_boundarynorm',
    }
    output_files = [str(output_dir / f'chi200_{area}.png') for area in lst_areas]

    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('🎯 CACHE VÁLIDO! Execução já foi realizada com os mesmos parâmetros.')
        logger.info(f'   📅 Período: {settings.DATA_INICIAL} a {settings.DATA_FINAL}')
        logger.info(f'   📊 {len(output_files)} mapas já existem')
        logger.info(f'   📁 Diretório: {output_dir}')
        logger.info('   ⏭️  Pulando execução')
        return

    start_time = time.time()
    logger.info(f'📅 Período de análise: {settings.DATA_INICIAL} a {settings.DATA_FINAL}')
    logger.info(f'📊 Gerando {len(lst_areas)} mapas de anomalia CHI200')
    logger.info(f'📌 Paleta fixa: verde -> bege -> marrom ({len(CHI200_COLORS)} cores)')
    logger.info('=' * 80)

    # Etapa anterior: deve gerar/salvar chi200.nc
    plot_chi200()

    output_dir.mkdir(parents=True, exist_ok=True)

    chi_file = dados_dir / CHI_FILE_NAME
    if not chi_file.exists():
        raise FileNotFoundError(
            f'Arquivo esperado não encontrado: {chi_file}. '
            'A rotina plot_chi200() precisa salvar esse NetCDF antes da plotagem.'
        )

    ds = load_dataset(str(chi_file))

    # Campos principais
    da_chi = _prepare_chi_field(ds)
    da_uchi = _prepare_vector_field(ds, UCHI_CANDIDATES)
    da_vchi = _prepare_vector_field(ds, VCHI_CANDIDATES)

    # Garante mesma grade
    if not (
        np.array_equal(da_chi['lat'].values, da_uchi['lat'].values)
        and np.array_equal(da_chi['lon'].values, da_uchi['lon'].values)
        and np.array_equal(da_chi['lat'].values, da_vchi['lat'].values)
        and np.array_equal(da_chi['lon'].values, da_vchi['lon'].values)
    ):
        raise ValueError('CHI, UCHI e VCHI não estão na mesma grade lat/lon.')

    lat = da_chi['lat'].values

    chi_cyc, lon_cyc = _add_cyclic_2d(da_chi)
    uchi_cyc, vchi_cyc, lon_cyc_uv = _add_cyclic_uv(da_uchi, da_vchi)

    levels, ticks, cmap, norm = _build_levels_ticks_norm()

    info_plot = settings['areas_plotagem']

    for area in lst_areas:
        logger.info(f'🖼️  Gerando mapa CHI200 para área: {area_display_name(area)}')

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

        # Boxes configuráveis
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

        # Alguns realces herdados
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
                t = plt.text(x, y, txt, fontsize=14, color=cor, weight='bold', zorder=500)
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

        # Grade
        if is_polar:
            gl = ax.gridlines(draw_labels=False, linestyle='--', alpha=0.5)
            gl.xlocator = MultipleLocator(30)
            gl.ylocator = MultipleLocator(20)
        else:
            gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.0)
            _configure_gridlines(gl, area)

        # Limites
        if not is_polar:
            ax.set_xlim([info_plot[area]['lon_esq'], info_plot[area]['lon_dir']])
            ax.set_ylim([info_plot[area]['lat_inf'], info_plot[area]['lat_sup']])

        # Features
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, edgecolor='black')
        ax.add_feature(cfeature.LAND.with_scale('50m'), linewidth=0.5, facecolor='whitesmoke')

        if area != 'china':
            ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1.2, edgecolor='black', zorder=100)

        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, edgecolor='black', zorder=100)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, edgecolor='black', zorder=100)
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), linewidth=0.5, facecolor='white')

        # Campo preenchido com paleta DISCRETA fixa
        cf = ax.contourf(
            lon_cyc,
            lat,
            chi_cyc,
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend='both',
            transform=ccrs.PlateCarree(
                central_longitude=info_plot[area]['central_longitude_plot']
            ),
            zorder=2,
        )

        # Contornos
        ax.contour(
            lon_cyc,
            lat,
            chi_cyc,
            levels=levels,
            colors='white',
            linewidths=0.6,
            alpha=0.55,
            transform=ccrs.PlateCarree(
                central_longitude=info_plot[area]['central_longitude_plot']
            ),
            zorder=3,
        )

        # Vetores divergentes/irrotacionais
        qcfg = _get_quiver_config(area)

        _AREAS_COM_CORTE_BORDA = {'globo', 'psa', 'hemisferio_sul'}
        lon_q, lat_q, u_q_mask, v_q_mask = _prepare_quiver_masked(
            lon=lon_cyc_uv,
            lat=lat,
            u=uchi_cyc,
            v=vchi_cyc,
            step=int(qcfg['step']),
            min_mag=float(qcfg['min_mag']),
            lat_bounds=(info_plot[area]['lat_inf'], info_plot[area]['lat_sup'])
            if area in _AREAS_COM_CORTE_BORDA
            else None,
        )

        ax.quiver(
            lon_q,
            lat_q,
            u_q_mask,
            v_q_mask,
            transform=ccrs.PlateCarree(),
            color=qcfg['color'],
            pivot='mid',
            scale=float(qcfg['scale']),
            width=float(qcfg['width']),
            headwidth=float(qcfg['headwidth']),
            headlength=float(qcfg['headlength']),
            headaxislength=float(qcfg['headaxislength']),
            zorder=5,
        )

        # Colorbar
        if is_polar and area != 'globo_3d':
            cbar = plt.colorbar(cf, ax=ax, pad=0.05, fraction=0.04, ticks=ticks)
            cbar.set_label(label=r'10$^5$ m$^2$ s$^{-1}$', size=10)
            cbar.ax.tick_params(labelsize=10)
        elif area in {'enso', 'tropico', 'MDR', 'hemisferio_sul', 'psa'}:
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
            cbar.set_label(
                label=r'10$^5$ m$^2$ s$^{-1}$',
                size=18,
            )
            cbar.ax.tick_params(labelsize=20)
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
            cbar.set_label(
                label=r'10$^5$ m$^2$ s$^{-1}$',
                size=18,
            )
            cbar.ax.tick_params(labelsize=20)

        # Título
        dt_ini = datetime.strptime(settings.DATA_INICIAL, '%Y-%m-%d').strftime('%d-%m-%y')
        dt_fim = datetime.strptime(settings.DATA_FINAL, '%Y-%m-%d').strftime('%d-%m-%y')
        titulo = f'Anomalia de CHI200 (De {dt_ini} a {dt_fim})'
        ax.set_title(titulo, fontsize=14 if is_polar else 18, loc='left')

        filename_fig = output_dir / f'chi200_{area}.png'
        logger.info(f'Salvando a figura {filename_fig}')

        # Logo — canto inferior esquerdo do frame do mapa
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

        plt.savefig(
            str(filename_fig),
            dpi=fig.dpi,
            bbox_inches='tight',
        )
        plt.close('all')

    execution_time = time.time() - start_time
    save_cache_metadata(SCRIPT_ID, cache_params, output_files, execution_time)

    logger.info('=' * 80)
    logger.info(f'✅ Script {SCRIPT_ID.upper()} concluído com sucesso!')
    logger.info(f'⏱️  Tempo de execução: {execution_time:.1f}s ({execution_time / 60:.1f} min)')
    logger.info(f'📊 {len(output_files)} mapas gerados em: {output_dir}')
    logger.info('=' * 80)


if __name__ == '__main__':
    main()