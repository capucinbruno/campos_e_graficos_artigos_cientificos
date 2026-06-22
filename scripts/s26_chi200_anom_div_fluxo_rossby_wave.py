"""s26 - chi200 (shaded) + Geopotencial 250 hPa (contornos) + WAF + Vento Divergente.

Pipeline:
  1. chi200 em shaded (verde -> bege -> marrom) — mesmo periodo das settings
  2. Contornos pretos de anomalia de geopotencial 250 hPa
  3. Vetores WAF (Rossby Wave Activity Flux, Takaya & Nakamura 2001)
  4. Vento divergente 200 hPa (chi200 < 0, faixa -20° a 20°, paleturquoise com contorno preto)

Dados de entrada:
    - ERA5/GDAS: geopotencial 250 hPa (via plot_rossby_waf)
    - ERA5/GDAS + PSL: chi200 (via plot_chi200) — shaded + mascara + vetores do vento divergente
    - Climatologias: hgt250, uwnd250, vwnd250 (arquivos fixos em Entrada/)

Saida:
    - Mapas PNG em Saida/s26_CHI200_ANOM_DIV_ROSSBY_WAF/

Criado em: 2026-06-06
"""

from __future__ import annotations

import gc
import time
from datetime import datetime
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from cartopy.util import add_cyclic_point
from matplotlib import patches
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from PIL import Image

from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.common.dataset_utils import load_dataset
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.common.logo_helper import resolve_logo_path
from app.common.logo_helper import proportional_logo_zoom
from app.src.uteis.plot_chi200 import main as plot_chi200
from app.src.uteis.plot_rossby_waf import main as plot_rossby_waf

# ---------------------------------------------------------------------------
# Identidade do script
# ---------------------------------------------------------------------------
SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's26'
SCRIPT_NAME = Path(__file__).stem
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME

# ---------------------------------------------------------------------------
# Constantes — chi200 shaded
# ---------------------------------------------------------------------------
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
CHI_SHADE_LEVELS = np.arange(-60, 65, 5)
CHI_SHADE_TICKS = np.arange(-60, 65, 10)

# ---------------------------------------------------------------------------
# Constantes — WAF (Rossby wave activity flux)
# ---------------------------------------------------------------------------
WAF_FILE_NAME = 'rossby_waf.nc'

QUIVER_WAF_DEFAULTS = {
    'step': 2,
    'width': 0.002,
    'headwidth': 4.5,
    'headlength': 6.0,
    'scale': None,
    'scale_units': 'xy',
    'min_amp_ratio': 0.05,
}
QUIVER_WAF_POR_AREA: dict[str, dict] = {
    'psa': {'step': 2},
    'hemisferio_sul': {'step': 2},
    'hemisferio_norte': {'step': 2},
    'globo': {'step': 2},
    'america_sul': {'step': 1, 'width': 0.004, 'headwidth': 5.0, 'headlength': 7.0, 'scale': 0.12},
    'globo_3d': {'step': 2, 'width': 0.003, 'headwidth': 5.0, 'headlength': 7.0},
}

# ---------------------------------------------------------------------------
# Constantes — vento divergente (quiver tropical, chi200 < 0)
# ---------------------------------------------------------------------------
CHI_FILE_NAME = 'chi200.nc'
CHI_SCALE = 1e5
CHI_CANDIDATES = ('chi_anom_mean_scaled', 'chi_anom_mean', 'chi', 'chi_anom', 'velocity_potential_anomaly')
UCHI_CANDIDATES = ('uchi_anom_mean', 'uchi', 'uchi_anom', 'udiv', 'uchi_irrot')
VCHI_CANDIDATES = ('vchi_anom_mean', 'vchi', 'vchi_anom', 'vdiv', 'vchi_irrot')

QUIVER_DIV_STEP = 4
QUIVER_DIV_SCALE = 80
QUIVER_DIV_WIDTH = 0.002
QUIVER_DIV_MIN_MAG = 0.3
LAT_DIV_MIN = -20.0
LAT_DIV_MAX = 20.0

# ---------------------------------------------------------------------------
# Areas e fallback
# ---------------------------------------------------------------------------
DEFAULT_AREAS = ['globo', 'psa', 'hemisferio_sul', 'hemisferio_norte', 'america_sul', 'globo_3d']


# ---------------------------------------------------------------------------
# Utilitarios
# ---------------------------------------------------------------------------
def _get_area_list() -> list[str]:
    if hasattr(settings, 'LST_AREAS_S26'):
        return list(settings.LST_AREAS_S26)
    return list(DEFAULT_AREAS)


def _get_waf_quiver_config(area: str) -> dict:
    cfg = dict(QUIVER_WAF_DEFAULTS)
    if area in QUIVER_WAF_POR_AREA:
        cfg.update(QUIVER_WAF_POR_AREA[area])
    return cfg


def _pick_first_var(ds, candidates, *, required=True):
    for name in candidates:
        if name in ds.data_vars:
            return ds[name], name
    if required:
        raise KeyError(
            f'Nenhuma das variaveis {candidates} encontrada. '
            f'Disponiveis: {list(ds.data_vars)}'
        )
    return None, None


def _standardize_coords(da):
    ren = {}
    if 'latitude' in da.coords:
        ren['latitude'] = 'lat'
    if 'longitude' in da.coords:
        ren['longitude'] = 'lon'
    if ren:
        da = da.rename(ren)
    if 'lat' not in da.coords or 'lon' not in da.coords:
        raise ValueError("Campo sem coordenadas 'lat' e 'lon'.")
    for dim in list(da.dims):
        if dim not in {'lat', 'lon'} and da.sizes[dim] == 1:
            da = da.isel({dim: 0}, drop=True)
    extra_dims = [d for d in da.dims if d not in {'lat', 'lon'}]
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


def _add_cyclic_2d(da2d):
    data_cyc, lon_cyc = add_cyclic_point(da2d.values, coord=da2d['lon'].values)
    return data_cyc, lon_cyc


def _add_cyclic_uv(u2d, v2d):
    u_cyc, lon_cyc = add_cyclic_point(u2d.values, coord=u2d['lon'].values)
    v_cyc, _ = add_cyclic_point(v2d.values, coord=v2d['lon'].values)
    return u_cyc, v_cyc, lon_cyc


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


def _add_logo_to_map(ax, logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=500):
    logo = Image.open(logo_path).convert('RGBA')
    bbox = logo.getbbox()
    if bbox is not None:
        logo = logo.crop(bbox)
    imagebox = OffsetImage(np.array(logo), zoom=proportional_logo_zoom(ax, np.array(logo).shape[1]))
    ab = AnnotationBbox(
        imagebox, (0, 0), xycoords=ax.transAxes,
        xybox=(xoffset, yoffset), boxcoords='offset points',
        box_alignment=(0, 0), frameon=False, pad=0, zorder=zorder, clip_on=False,
    )
    ax.add_artist(ab)


def _build_chi200_cmap():
    n_bins = len(CHI_SHADE_LEVELS) - 1
    cmap = LinearSegmentedColormap.from_list('chi200_green_brown', CHI200_COLORS, N=n_bins)
    norm = BoundaryNorm(CHI_SHADE_LEVELS, ncolors=n_bins, clip=False)
    return cmap, norm


# ---------------------------------------------------------------------------
# Interatividade — pergunta de defasagem
# ---------------------------------------------------------------------------

def _ask_lag(chi_ini: str, chi_fim: str) -> tuple[bool, str, str]:
    """Pergunta ao usuario se deseja defazar Geop250+WAF em relacao ao chi200+vento divergente.

    Retorna (usar_lag, lag_ini_str, lag_fim_str).
    Se o usuario nao quiser lag, lag_ini_str e lag_fim_str sao iguais ao periodo chi200.
    """
    print()
    print(f'  Periodo chi200 + Vento Divergente: {chi_ini} a {chi_fim}')
    resp = input(
        '  Deseja defazar chi200 + Vento Divergente com Geopotencial 250 hPa + WAF? [s/n]: '
    ).strip().lower()

    if resp not in ('s', 'sim', 'y', 'yes'):
        return False, chi_ini, chi_fim

    while True:
        lag_ini = input('    Data inicial para Geopotencial + WAF (YYYY-MM-DD): ').strip()
        lag_fim = input('    Data final   para Geopotencial + WAF (YYYY-MM-DD): ').strip()
        try:
            datetime.strptime(lag_ini, '%Y-%m-%d')
            datetime.strptime(lag_fim, '%Y-%m-%d')
            break
        except ValueError:
            print('    Formato invalido. Use YYYY-MM-DD e tente novamente.')

    print(f'  Defasagem ativada: Geop250 + WAF de {lag_ini} a {lag_fim}')
    print()
    return True, lag_ini, lag_fim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger = get_logger(SCRIPT_ID)

    logger.info('=' * 80)
    logger.info(f'SCRIPT {SCRIPT_ID.upper()}: {SCRIPT_DESC}')
    logger.info('=' * 80)

    lst_areas = _get_area_list()
    output_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_CHI200_ANOM_DIV_ROSSBY_WAF'
    input_dir = Path(settings.DIR_INPUT)
    dados_dir = Path(settings.DIR_DADOS)

    chi_ini_str = str(settings.DATA_INICIAL)
    chi_fim_str = str(settings.DATA_FINAL)

    # ---- Pergunta de defasagem (antes do cache para que o lag entre nos params) ----
    usar_lag, lag_ini_str, lag_fim_str = _ask_lag(chi_ini_str, chi_fim_str)

    cache_params = {
        'DATA_INICIAL': chi_ini_str,
        'DATA_FINAL': chi_fim_str,
        'areas': lst_areas,
        'script_version': '1.0',
        'waf_file': WAF_FILE_NAME,
        'chi_file': CHI_FILE_NAME,
        'lag_geop_waf_ini': lag_ini_str,
        'lag_geop_waf_fim': lag_fim_str,
    }
    output_files = [str(output_dir / f'chi200_anom_div_rossby_waf_{area}.png') for area in lst_areas]

    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('CACHE VALIDO! Execucao ja foi realizada com os mesmos parametros.')
        logger.info(f'   chi200 + Vento Div: {chi_ini_str} a {chi_fim_str}')
        if usar_lag:
            logger.info(f'   Geop250 + WAF:      {lag_ini_str} a {lag_fim_str}')
        logger.info(f'   {len(output_files)} mapas ja existem em: {output_dir}')
        logger.info('   Pulando execucao')
        return

    start_time = time.time()
    logger.info(f'chi200 + Vento Div: {chi_ini_str} a {chi_fim_str}')
    if usar_lag:
        logger.info(f'Geop250 + WAF:      {lag_ini_str} a {lag_fim_str}')
    logger.info(f'Gerando {len(lst_areas)} mapas — chi200 + WAF + vento divergente')

    start_date_waf = np.datetime64(lag_ini_str)
    end_date_waf = np.datetime64(lag_fim_str)

    dt_ini_waf = datetime.strptime(lag_ini_str, '%Y-%m-%d')
    dt_fim_waf = datetime.strptime(lag_fim_str, '%Y-%m-%d')
    logger.info('=' * 80)

    dados_dir.mkdir(parents=True, exist_ok=True)

    # ---- Etapa 1: WAF → rossby_waf.nc (usa periodo do lag se ativo) ----
    logger.info(f'Etapa 1: Calculando WAF ({lag_ini_str} a {lag_fim_str})...')
    if usar_lag:
        settings.DATA_INICIAL = lag_ini_str
        settings.DATA_FINAL = lag_fim_str
    plot_rossby_waf()
    if usar_lag:
        settings.DATA_INICIAL = chi_ini_str
        settings.DATA_FINAL = chi_fim_str
    gc.collect()

    # ---- Etapa 2: CHI200 (shaded + mascara + vetores do vento divergente) ----
    logger.info(f'Etapa 2: Calculando CHI200 ({chi_ini_str} a {chi_fim_str})...')
    plot_chi200()
    gc.collect()

    chi_file = dados_dir / CHI_FILE_NAME
    if not chi_file.exists():
        raise FileNotFoundError(f'Arquivo nao encontrado: {chi_file}')
    ds_chi = load_dataset(str(chi_file))

    da_chi, chi_varname = _pick_first_var(ds_chi, CHI_CANDIDATES)
    da_chi = _standardize_coords(da_chi)
    if chi_varname != 'chi_anom_mean_scaled':
        da_chi = da_chi / CHI_SCALE

    da_uchi = _standardize_coords(_pick_first_var(ds_chi, UCHI_CANDIDATES)[0])
    da_vchi = _standardize_coords(_pick_first_var(ds_chi, VCHI_CANDIDATES)[0])

    lat_chi = da_chi['lat'].values
    chi_cyc, lon_chi_cyc = _add_cyclic_2d(da_chi)
    uchi_cyc, vchi_cyc, _ = _add_cyclic_uv(da_uchi, da_vchi)

    # ---- Etapa 3: Carregar WAF (periodo do lag se ativo) ----
    waf_file = dados_dir / WAF_FILE_NAME
    if not waf_file.exists():
        raise FileNotFoundError(
            f'Arquivo esperado nao encontrado: {waf_file}. '
            'A rotina plot_rossby_waf() precisa salvar esse NetCDF antes da plotagem.'
        )
    ds_waf = load_dataset(str(waf_file))

    hgt_anom = ds_waf['hgt_anom_mean']
    waf_x = ds_waf['waf_x']
    waf_y = ds_waf['waf_y']

    lat_hgt = hgt_anom['lat'].values
    lon_hgt = hgt_anom['lon'].values
    lat_waf = waf_x['lat_waf'].values
    lon_waf = waf_x['lon_waf'].values

    # Ajuste de longitude -180/180 + cyclic — hgt250
    lon_shift = ((lon_hgt + 180) % 360) - 180
    sort_idx = np.argsort(lon_shift)
    hgt_cyc, lon_hgt_cyc = add_cyclic_point(
        hgt_anom.isel(lon=sort_idx).values, coord=lon_shift[sort_idx]
    )

    # Ajuste de longitude -180/180 + cyclic — WAF
    lon_waf_shift = ((lon_waf + 180) % 360) - 180
    sort_idx_waf = np.argsort(lon_waf_shift)
    lon_waf_sorted = lon_waf_shift[sort_idx_waf]
    px_sorted = waf_x.values[:, sort_idx_waf]
    py_sorted = waf_y.values[:, sort_idx_waf]
    px_cyc, lon_waf_cyc = add_cyclic_point(px_sorted, coord=lon_waf_sorted)
    py_cyc, _ = add_cyclic_point(py_sorted, coord=lon_waf_sorted)

    # Levels adaptativos de contorno hgt250
    if np.all((hgt_cyc >= -50) & (hgt_cyc <= 50)):
        hgt_levels = np.arange(-50, 53, 3)
    elif np.all((hgt_cyc >= -100) & (hgt_cyc <= 100)):
        hgt_levels = np.arange(-100, 120, 20)
    else:
        hgt_levels = np.arange(-200, 220, 20)

    # Colormap chi200
    chi_cmap, chi_norm = _build_chi200_cmap()

    # Strings formatadas para título
    chi_ini_fmt = datetime.strptime(chi_ini_str, '%Y-%m-%d').strftime('%d-%m-%y')
    chi_fim_fmt = datetime.strptime(chi_fim_str, '%Y-%m-%d').strftime('%d-%m-%y')
    lag_ini_fmt = datetime.strptime(lag_ini_str, '%Y-%m-%d').strftime('%d-%m-%y')
    lag_fim_fmt = datetime.strptime(lag_fim_str, '%Y-%m-%d').strftime('%d-%m-%y')

    info_plot = settings['areas_plotagem']
    output_dir.mkdir(parents=True, exist_ok=True)

    logo_path = resolve_logo_path(input_dir)

    for area in lst_areas:
        logger.info(f'Gerando mapa para area: {area}')

        is_polar = info_plot[area].get('projection', '') == 'orthographic_south'
        if is_polar:
            proj = ccrs.Orthographic(
                central_longitude=settings.get('ORTHO_CENTRAL_LONGITUDE', info_plot[area].get('ortho_central_longitude', -71)),
                central_latitude=settings.get('ORTHO_CENTRAL_LATITUDE', info_plot[area].get('ortho_central_latitude', -84)),
            )
        else:
            proj = ccrs.PlateCarree(central_longitude=info_plot[area]['central_longitude_mapa'])

        data_transform = ccrs.PlateCarree(central_longitude=info_plot[area]['central_longitude_plot'])

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
                ax.add_patch(patches.Rectangle(
                    (box['x_anc'], box['y_anc']),
                    box['x_larg'], box['y_larg'],
                    linewidth=box['linewidth'], edgecolor=box['edgecolor'],
                    facecolor='none', zorder=100,
                ))

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

        ax.add_feature(cfeature.LAND.with_scale('50m'), linewidth=0.5, facecolor='whitesmoke')
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), linewidth=0.5, facecolor='white')
        ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1.2, zorder=100)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, zorder=100)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, zorder=100)

        # --- Camada 1: chi200 shaded (verde -> bege -> marrom, sem contorno) ---
        im = ax.contourf(
            lon_chi_cyc, lat_chi, chi_cyc,
            levels=CHI_SHADE_LEVELS,
            cmap=chi_cmap,
            norm=chi_norm,
            extend='both',
            transform=data_transform,
            zorder=2,
        )

        # --- Camada 2: contornos pretos — anomalia geopotencial 250 hPa ---
        ax.contour(
            lon_hgt_cyc, lat_hgt, hgt_cyc,
            levels=hgt_levels,
            colors='black',
            linewidths=1.0,
            transform=data_transform,
            zorder=110,
        )

        # --- Camada 3: vetores WAF (Rossby wave activity flux) ---
        qcfg = _get_waf_quiver_config(area)
        step = int(qcfg['step'])
        lon_q_waf = lon_waf_cyc[::step]
        lat_q_waf = lat_waf[::step]
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
            lon_q_waf, lat_q_waf,
            px_plot, py_plot,
            transform=data_transform,
            scale=qcfg['scale'],
            scale_units=qcfg['scale_units'],
            width=float(qcfg['width']),
            headwidth=float(qcfg['headwidth']),
            headlength=float(qcfg['headlength']),
            color='black',
            zorder=150,
        )

        # --- Camada 4: vento divergente — chi200 < 0, faixa -20° a 20° ---
        lon_q_div = lon_chi_cyc[::QUIVER_DIV_STEP]
        lat_q_div = lat_chi[::QUIVER_DIV_STEP]
        u_q = uchi_cyc[::QUIVER_DIV_STEP, ::QUIVER_DIV_STEP]
        v_q = vchi_cyc[::QUIVER_DIV_STEP, ::QUIVER_DIV_STEP]
        chi_q = chi_cyc[::QUIVER_DIV_STEP, ::QUIVER_DIV_STEP]

        lat_outside = (lat_q_div < LAT_DIV_MIN) | (lat_q_div > LAT_DIV_MAX)
        combined_mask = (
            np.broadcast_to(lat_outside[:, None], u_q.shape)
            | (chi_q >= 0)
            | (np.sqrt(u_q**2 + v_q**2) < QUIVER_DIV_MIN_MAG)
        )
        q = ax.quiver(
            lon_q_div, lat_q_div,
            np.ma.masked_where(combined_mask, u_q),
            np.ma.masked_where(combined_mask, v_q),
            transform=ccrs.PlateCarree(),
            color='white', pivot='mid',
            scale=QUIVER_DIV_SCALE, width=QUIVER_DIV_WIDTH,
            headwidth=3.2, headlength=4.2, headaxislength=3.8,
            zorder=200,
        )
        q.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='black')])

        # --- Colorbar chi200 ---
        if is_polar and area != 'globo_3d':
            cbar = plt.colorbar(im, ax=ax, pad=0.05, fraction=0.04, ticks=CHI_SHADE_TICKS)
            cbar.set_label(label='10$^5$ m$^2$/s', size=10)
            cbar.ax.tick_params(labelsize=10)
        elif area in {'america_sul', 'globo_3d'}:
            cax = make_axes_locatable(ax).append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
            cbar = plt.colorbar(im, cax=cax, pad=0.02, fraction=0.02375, extend='both', ticks=CHI_SHADE_TICKS)
            cbar.set_label(label='10$^5$ m$^2$/s', size=18)
            cbar.ax.tick_params(labelsize=20)
        else:
            cax = make_axes_locatable(ax).append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)
            cbar = plt.colorbar(
                im, cax=cax, pad=0.02, fraction=0.02375,
                location='bottom', extend='both', orientation='horizontal', ticks=CHI_SHADE_TICKS,
            )
            cbar.set_label(label='10$^5$ m$^2$/s', size=18)
            cbar.ax.tick_params(labelsize=20)

        if usar_lag:
            titulo = (
                f'chi200 + Vento Div: {chi_ini_fmt} a {chi_fim_fmt} | '
                f'Geop250 + WAF: {lag_ini_fmt} a {lag_fim_fmt}\n'
                f'chi200 (shaded) + Geopotencial 250 hPa (linhas) + WAF + Vento Divergente chi200<0'
            )
        else:
            titulo = (
                f'CHI200 (shaded) + Geopotencial 250 hPa (linhas) +\n'
                f'WAF + Vento Divergente chi200<0 (De {chi_ini_fmt} a {chi_fim_fmt})'
            )
        ax.set_title(titulo, fontsize=11 if is_polar else 14, loc='left')

        if logo_path is not None and logo_path.exists():
            _add_logo_to_map(ax=ax, logo_path=logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=500)

        filename_fig = output_dir / f'chi200_anom_div_rossby_waf_{area}.png'
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
