# -*- coding: utf-8 -*-
"""
s15 - CHI200 (shaded) + PSI200 (contour).

Combina o campo de função de velocidade (CHI200) em shaded com as isolinhas
da função de corrente (PSI200) em azul escuro, a partir dos NetCDFs preparados
pelos pipelines plot_chi200 e plot_psi200.

Arquivos esperados de entrada:
    - {settings.DIR_DADOS}/chi200.nc
    - {settings.DIR_DADOS}/psi200.nc

Saída:
    - Mapas PNG em {settings.DIR_OUTPUT}/s15_CHI200_PSI200/

Criado em: 2026-06-05
"""

from __future__ import annotations

# Forcar backend nao-interativo antes de qualquer import matplotlib
import matplotlib
matplotlib.use('Agg')

# ---------------------------------------------------------------------------
# Bibliotecas padrão
# ---------------------------------------------------------------------------
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
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
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import FixedLocator, MultipleLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from PIL import Image

# ---------------------------------------------------------------------------
# Módulos locais
# ---------------------------------------------------------------------------
from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.common.dataset_utils import area_display_name, load_dataset
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.common.logo_helper import resolve_logo_path
from app.common.logo_helper import proportional_logo_zoom
from app.src.uteis.plot_chi200 import main as plot_chi200
from app.src.uteis.plot_psi200 import main as plot_psi200

# ---------------------------------------------------------------------------
# Identidade do script
# ---------------------------------------------------------------------------
SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's15'
SCRIPT_NAME = Path(__file__).stem
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME

# ---------------------------------------------------------------------------
# Constantes — CHI200
# ---------------------------------------------------------------------------
CHI_SCALE = 1e5
CHI_FILE_NAME = 'chi200.nc'

LEVELS = np.arange(-60, 65, 5)

CHI_CANDIDATES = (
    'chi_anom_mean_scaled',
    'chi_anom_mean',
    'chi',
    'chi_anom',
    'velocity_potential_anomaly',
)

UCHI_CANDIDATES = ('uchi_anom_mean', 'uchi', 'uchi_anom', 'udiv', 'uchi_irrot')
VCHI_CANDIDATES = ('vchi_anom_mean', 'vchi', 'vchi_anom', 'vdiv', 'vchi_irrot')

# Quiver — vento divergente restrito a chi200 < 0 na faixa tropical
QUIVER_STEP = 4
QUIVER_SCALE = 80
QUIVER_WIDTH = 0.002
QUIVER_MIN_MAG = 0.3
LAT_DIV_MIN = -20.0
LAT_DIV_MAX = 20.0

CHI200_COLORS = [
    '#005a45', '#0f7a6c', '#2e9b96', '#62bdb7', '#9dd8d2', '#dff3f1',
    '#f7f4eb', '#e7d9a9', '#d6b566', '#bd8a35', '#9a6313', '#6f4300',
]

# ---------------------------------------------------------------------------
# Constantes — PSI200
# ---------------------------------------------------------------------------
PSI_SCALE = 1e6
PSI_FILE_NAME = 'psi200.nc'
PSI_CONTOUR_LEVELS = np.arange(-100, 102, 2)  # unidades: 10^6 m² s^-1
PSI_CONTOUR_COLOR = 'darkblue'
PSI_CONTOUR_LINEWIDTH = 1.0

PSI_CANDIDATES = (
    'psi_anom_mean_scaled',
    'psi_anom_mean',
    'psi',
    'psi_anom',
    'streamfunction_anomaly',
)


# ---------------------------------------------------------------------------
# Funções utilitárias
# ---------------------------------------------------------------------------
def _pick_first_var(ds, candidates, *, required=True):
    for name in candidates:
        if name in ds.data_vars:
            return ds[name], name
    if required:
        raise KeyError(
            f'Nenhuma das variáveis {candidates} encontrada. '
            f'Disponíveis: {list(ds.data_vars)}'
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


def _build_chi_levels_norm():
    n_bins = len(LEVELS) - 1
    cmap = LinearSegmentedColormap.from_list('chi200_green_brown', CHI200_COLORS, N=n_bins)
    norm = BoundaryNorm(LEVELS, ncolors=n_bins, clip=False)
    ticks = np.arange(-60, 65, 10)
    return LEVELS, ticks, cmap, norm


def _get_area_list() -> list[str]:
    for attr in ('LST_AREAS_S15', 'LST_AREAS_S02', 'LST_AREAS_CHI200', 'LST_AREAS_S01'):
        if hasattr(settings, attr):
            return list(getattr(settings, attr))
    raise AttributeError(
        'Nenhuma lista de areas encontrada (LST_AREAS_S15, LST_AREAS_S02, LST_AREAS_CHI200 ou LST_AREAS_S01).'
    )


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
    elif area in {'argentina', 'costa_brasil'}:
        gl.xlocator = MultipleLocator(5)
        gl.ylocator = MultipleLocator(5)
    elif area in {'tsa', 'tna'}:
        gl.xlocator = MultipleLocator(10)
        gl.ylocator = MultipleLocator(5)
    elif area == 'brasil':
        gl.xlocator = MultipleLocator(10)
        gl.ylocator = MultipleLocator(5)
    elif area in {'america_sul', 'africa'}:
        gl.xlocator = MultipleLocator(20)
        gl.ylocator = MultipleLocator(20)
    elif area in {'atlantico_tropical', 'pdo', 'iod', 'sad', 'amo', 'africa_monsoon', 'zona_zcit_atlantico'}:
        gl.xlocator = MultipleLocator(20 if area != 'zona_zcit_atlantico' else 10)
        gl.ylocator = MultipleLocator(10 if area != 'zona_zcit_atlantico' else 5)
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
    imagebox = OffsetImage(np.array(logo), zoom=proportional_logo_zoom(ax, np.array(logo).shape[1]))
    ab = AnnotationBbox(
        imagebox, (0, 0), xycoords=ax.transAxes,
        xybox=(xoffset, yoffset), boxcoords='offset points',
        box_alignment=(0, 0), frameon=False, pad=0, zorder=zorder, clip_on=False,
    )
    ax.add_artist(ab)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger = get_logger(SCRIPT_ID)

    logger.info('=' * 80)
    logger.info(f'SCRIPT {SCRIPT_ID.upper()}: {SCRIPT_DESC}')
    logger.info('=' * 80)

    lst_areas = _get_area_list()

    output_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_CHI200_PSI200'
    input_dir = Path(settings.DIR_INPUT)
    dados_dir = Path(settings.DIR_DADOS)

    cache_params = {
        'DATA_INICIAL': settings.DATA_INICIAL,
        'DATA_FINAL': settings.DATA_FINAL,
        'areas': lst_areas,
        'script_version': '1.3',
        'chi_file': CHI_FILE_NAME,
        'psi_file': PSI_FILE_NAME,
        'psi_contour_levels': PSI_CONTOUR_LEVELS.tolist(),
    }
    output_files = [str(output_dir / f'chi200_psi200_{area}.png') for area in lst_areas]

    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('CACHE VALIDO! Execucao ja foi realizada com os mesmos parametros.')
        logger.info(f'   Periodo: {settings.DATA_INICIAL} a {settings.DATA_FINAL}')
        logger.info(f'   {len(output_files)} mapas ja existem')
        logger.info(f'   Diretorio: {output_dir}')
        logger.info('   Pulando execucao')
        return

    start_time = time.time()
    logger.info(f'Periodo de analise: {settings.DATA_INICIAL} a {settings.DATA_FINAL}')
    logger.info(f'Gerando {len(lst_areas)} mapas de CHI200 + PSI200')
    logger.info('=' * 80)

    # ---- Etapa 1: gerar chi200.nc ----
    logger.info('Etapa 1: Calculando CHI200 (ERA5/GDAS + PSL)...')
    plot_chi200()

    # Libera memória dos arrays do chi200 antes de criar threads no psi200
    import gc
    gc.collect()

    # ---- Etapa 2: gerar psi200.nc ----
    logger.info('Etapa 2: Calculando PSI200 (ERA5/GDAS + PSL)...')
    plot_psi200()

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Carregar chi200.nc ----
    chi_file = dados_dir / CHI_FILE_NAME
    if not chi_file.exists():
        raise FileNotFoundError(f'Arquivo nao encontrado: {chi_file}')
    ds_chi = load_dataset(str(chi_file))

    da_chi, chi_varname = _pick_first_var(ds_chi, CHI_CANDIDATES)
    da_chi = _standardize_coords(da_chi)

    # Escala CHI para 10^5 m² s^-1 se necessário
    if chi_varname != 'chi_anom_mean_scaled':
        da_chi = da_chi / CHI_SCALE

    lat_chi = da_chi['lat'].values
    chi_cyc, lon_cyc = _add_cyclic_2d(da_chi)

    da_uchi = _standardize_coords(_pick_first_var(ds_chi, UCHI_CANDIDATES)[0])
    da_vchi = _standardize_coords(_pick_first_var(ds_chi, VCHI_CANDIDATES)[0])
    uchi_cyc, vchi_cyc, _ = _add_cyclic_uv(da_uchi, da_vchi)

    # ---- Carregar psi200.nc ----
    psi_file = dados_dir / PSI_FILE_NAME
    if not psi_file.exists():
        raise FileNotFoundError(f'Arquivo nao encontrado: {psi_file}')
    ds_psi = load_dataset(str(psi_file))

    da_psi_raw, psi_varname = _pick_first_var(ds_psi, PSI_CANDIDATES)
    da_psi = _standardize_coords(da_psi_raw)
    if psi_varname != 'psi_anom_mean_scaled':
        da_psi = da_psi / PSI_SCALE

    psi_cyc, lon_psi_cyc = _add_cyclic_2d(da_psi)
    lat_psi = da_psi['lat'].values

    levels, ticks, cmap, norm = _build_chi_levels_norm()
    info_plot = settings['areas_plotagem']

    dt_ini = datetime.strptime(settings.DATA_INICIAL, '%Y-%m-%d').strftime('%d-%m-%y')
    dt_fim = datetime.strptime(settings.DATA_FINAL, '%Y-%m-%d').strftime('%d-%m-%y')

    for area in lst_areas:
        logger.info(f'Gerando mapa CHI200+PSI200 para area: {area_display_name(area)}')

        is_polar = info_plot[area].get('projection', '') == 'orthographic_south'
        if is_polar:
            proj = ccrs.Orthographic(
                central_longitude=settings.get('ORTHO_CENTRAL_LONGITUDE', info_plot[area].get('ortho_central_longitude', -71)),
                central_latitude=settings.get('ORTHO_CENTRAL_LATITUDE', info_plot[area].get('ortho_central_latitude', -84)),
            )
        else:
            proj = ccrs.PlateCarree(central_longitude=info_plot[area]['central_longitude_mapa'])

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
                    box['x_larg'], box['y_larg'],
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
            ax.add_patch(patches.Rectangle(
                (10, -20), -40, 20, linewidth=3, edgecolor='black', facecolor='none', zorder=100,
            ))
            ax.add_patch(patches.Rectangle(
                (-15, 5), -40, 20, linewidth=3, edgecolor='blue', facecolor='none', zorder=100,
            ))

        if area == 'iod':
            ax.add_patch(patches.Rectangle(
                (50, -10), 20, 20, linewidth=3, edgecolor='black', facecolor='none', zorder=100,
            ))
            ax.add_patch(patches.Rectangle(
                (90, -10), 20, 10, linewidth=3, edgecolor='black', facecolor='none', zorder=100,
            ))

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
            for x, txt in [(-135.25, '1'), (-115.25, '2'), (-95.25, '3'), (-75.25, '4'),
                            (-55.25, '5'), (-35.25, '6'), (-15.25, '7'), (4.75, '8')]:
                t = plt.text(x, -4.64, txt, fontsize=50, color='white', weight='bold', zorder=400)
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

        # Features cartográficas
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, edgecolor='black')
        ax.add_feature(cfeature.LAND.with_scale('50m'), linewidth=0.5, facecolor='whitesmoke')
        if area != 'china':
            ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1.2, edgecolor='black', zorder=100)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, edgecolor='black', zorder=100)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, edgecolor='black', zorder=100)
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), linewidth=0.5, facecolor='white')

        data_transform = ccrs.PlateCarree(
            central_longitude=info_plot[area]['central_longitude_plot']
        )

        # CHI200 shaded
        cf = ax.contourf(
            lon_cyc, lat_chi, chi_cyc,
            levels=levels, cmap=cmap, norm=norm, extend='both',
            transform=data_transform, zorder=2,
        )

        # PSI200 isolinhas: negativas em vermelho, positivas em azul escuro
        neg_levels = PSI_CONTOUR_LEVELS[PSI_CONTOUR_LEVELS < 0]
        pos_levels = PSI_CONTOUR_LEVELS[PSI_CONTOUR_LEVELS > 0]
        if len(neg_levels):
            ax.contour(
                lon_psi_cyc, lat_psi, psi_cyc,
                levels=neg_levels,
                colors='red',
                linewidths=PSI_CONTOUR_LINEWIDTH,
                transform=data_transform,
                zorder=4,
            )
        if len(pos_levels):
            ax.contour(
                lon_psi_cyc, lat_psi, psi_cyc,
                levels=pos_levels,
                colors=PSI_CONTOUR_COLOR,
                linewidths=PSI_CONTOUR_LINEWIDTH,
                transform=data_transform,
                zorder=4,
            )

        # Vento divergente — chi200 < 0, faixa -20° a 20°
        lon_q = lon_cyc[::QUIVER_STEP]
        lat_q = lat_chi[::QUIVER_STEP]
        u_q = uchi_cyc[::QUIVER_STEP, ::QUIVER_STEP]
        v_q = vchi_cyc[::QUIVER_STEP, ::QUIVER_STEP]
        chi_q = chi_cyc[::QUIVER_STEP, ::QUIVER_STEP]
        lat_outside = (lat_q < LAT_DIV_MIN) | (lat_q > LAT_DIV_MAX)
        combined_mask = (
            np.broadcast_to(lat_outside[:, None], u_q.shape)
            | (chi_q >= 0)
            | (np.sqrt(u_q**2 + v_q**2) < QUIVER_MIN_MAG)
        )
        q = ax.quiver(
            lon_q, lat_q,
            np.ma.masked_where(combined_mask, u_q),
            np.ma.masked_where(combined_mask, v_q),
            transform=ccrs.PlateCarree(),
            color='white', pivot='mid',
            scale=QUIVER_SCALE, width=QUIVER_WIDTH,
            headwidth=3.2, headlength=4.2, headaxislength=3.8,
            zorder=200,
        )
        q.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='black')])

        # Colorbar CHI200
        if is_polar and area != 'globo_3d':
            cbar = plt.colorbar(cf, ax=ax, pad=0.05, fraction=0.04, ticks=ticks)
            cbar.set_label(label=r'10$^5$ m$^2$ s$^{-1}$', size=10)
            cbar.ax.tick_params(labelsize=10)
        elif area in {'enso', 'tropico', 'MDR', 'hemisferio_sul', 'psa'}:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)
            cbar = plt.colorbar(
                cf, cax=cax, pad=0.02, fraction=0.02375,
                location='bottom', extend='both', orientation='horizontal',
                ticks=ticks, boundaries=levels, spacing='proportional',
            )
            cbar.set_label(label=r'10$^5$ m$^2$ s$^{-1}$', size=18)
            cbar.ax.tick_params(labelsize=20)
        else:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
            cbar = plt.colorbar(
                cf, cax=cax, pad=0.02, fraction=0.02375,
                extend='both', ticks=ticks, boundaries=levels, spacing='proportional',
            )
            cbar.set_label(label=r'10$^5$ m$^2$ s$^{-1}$', size=18)
            cbar.ax.tick_params(labelsize=20)

        # Título
        titulo = f'CHI200 | PSI200 anomalia (De {dt_ini} a {dt_fim})'
        ax.set_title(titulo, fontsize=14 if is_polar else 18, loc='left')

        # Logo
        logo_path = resolve_logo_path(input_dir)
        if logo_path is not None and logo_path.exists():
            _add_logo_to_map(ax=ax, logo_path=logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=500)

        if area != 'globo_3d':
            ax.add_patch(patches.Rectangle(
                (0, 0), 1, 1,
                linewidth=0.5, edgecolor='black', facecolor='none',
                transform=ax.transAxes, zorder=1000, clip_on=False,
            ))

        filename_fig = output_dir / f'chi200_psi200_{area}.png'
        logger.info(f'Salvando a figura {filename_fig}')
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
