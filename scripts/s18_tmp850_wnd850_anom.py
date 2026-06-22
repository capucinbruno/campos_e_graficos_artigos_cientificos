"""s18 - Anomalia de Temperatura 850 hPa + Vento Anomalo 850 hPa.

Pipeline de dados:
  - Temperatura 850 hPa: ERA5/GDAS + climatologia PSL/NOAA
  - Vento 850 hPa: ERA5/GDAS + climatologia PSL/NOAA
  - Shaded: anomalia de temperatura (paleta de anomalia padrao do projeto)
  - Quiver: vetores de vento anomalo 850 hPa

Saida:
    - Mapas PNG em Saida/s18_TMP850_WND850/ (um por regiao configurada em LST_AREAS_S18)

Criado em: 2026-06-06
"""

# Bibliotecas padrão
import time
from datetime import datetime
from pathlib import Path

# Bibliotecas de terceiros
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import xarray as xr
from cartopy.util import add_cyclic_point
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import FixedLocator, MultipleLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from PIL import Image

# Módulos locais
from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.common.dataset_utils import area_display_name
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.common.logo_helper import resolve_logo_path
from app.common.logo_helper import proportional_logo_zoom
from app.src.uteis.plot_olr_wind850_anom import main as plot_wind850_anom
from app.src.uteis.plot_tmp850_anom import main as plot_tmp850_anom

# ---------------------------------------------------------------------------
# Workaround: cartopy 0.25 + matplotlib 3.10 — GeometryCollection não subscritável.
# ---------------------------------------------------------------------------
import matplotlib.path as _mpath
import numpy as _np
from cartopy.mpl.geoaxes import InterProjectionTransform as _IPT

_orig_transform_path = _IPT.transform_path_non_affine


def _safe_transform_path(self, path):
    try:
        return _orig_transform_path(self, path)
    except TypeError:
        return _mpath.Path(_np.empty((0, 2)))


_IPT.transform_path_non_affine = _safe_transform_path

# ---------------------------------------------------------------------------
# Identidade do script
# ---------------------------------------------------------------------------
SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's18'
SCRIPT_NAME = Path(__file__).stem
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
TMP_FILE_NAME = 'tmp850.nc'
WIND850_FILE_NAME = 'wind850_anom.nc'

LEVELS = np.arange(-5, 5.5, 0.5)
TICKS = np.arange(-5, 6, 1)

# Parâmetros de quiver por área (copiado do s12 — mesma fonte de dados)
_QUIVER_PARAMS: dict[str, dict] = {
    'africa_monsoon':             {'headwidth': 3, 'scale': 8,  'headlength': 5, 'width': 0.0022},
    'zona_zcit_atlantico':        {'headwidth': 3, 'scale': 8,  'headlength': 5, 'width': 0.0022},
    'MDR':                        {'headwidth': 3, 'scale': 10, 'headlength': 5, 'width': 0.002},
    'tropico':                    {'headwidth': 3, 'scale': 26, 'headlength': 5, 'width': 0.0008},
    'brasil':                     {'headwidth': 3, 'scale': 8,  'headlength': 5, 'width': 0.0032},
    'america_sul':                {'headwidth': 3, 'scale': 20, 'headlength': 5, 'width': 0.0034},
    'africa':                     {'headwidth': 3, 'scale': 15, 'headlength': 5, 'width': 0.002},
    'mjo':                        {'headwidth': 5, 'scale': 15, 'headlength': 5, 'width': 0.0009},
    'amo':                        {'headwidth': 5, 'scale': 10, 'headlength': 5, 'width': 0.0013},
    'pacifico_leste_america_sul': {'headwidth': 5, 'scale': 15, 'headlength': 5, 'width': 0.0014},
    'china':                      {'headwidth': 5, 'scale': 24, 'headlength': 5, 'width': 0.0006},
    'sad':                        {'headwidth': 5, 'scale': 12, 'headlength': 5, 'width': 0.0014},
    'iod':                        {'headwidth': 5, 'scale': 10, 'headlength': 5, 'width': 0.0013},
    'pdo':                        {'headwidth': 5, 'scale': 12, 'headlength': 5, 'width': 0.0012},
    'tna':                        {'headwidth': 5, 'scale': 10, 'headlength': 5, 'width': 0.0013},
    'tsa':                        {'headwidth': 5, 'scale': 12, 'headlength': 5, 'width': 0.0013},
    'atlantico_tropical':         {'headwidth': 5, 'scale': 12, 'headlength': 5, 'width': 0.0013},
    'enso':                       {'headwidth': 4, 'scale': 15, 'headlength': 5, 'width': 0.001},
    'america_sul_zom_out':        {'headwidth': 4, 'scale': 15, 'headlength': 5, 'width': 0.001},
    'globo':                      {'headwidth': 4, 'scale': 26, 'headlength': 5, 'width': 0.0006},
    'costa_brasil':               {'headwidth': 3, 'scale': 6,  'headlength': 5, 'width': 0.0038},
    'psa':                        {'headwidth': 5, 'scale': 26, 'headlength': 5, 'width': 0.0006},
    'hemisferio_sul':             {'headwidth': 5, 'scale': 26, 'headlength': 5, 'width': 0.0006},
    'argentina':                  {'headwidth': 3, 'scale': 5,  'headlength': 5, 'width': 0.005},
    'estados_unidos_zoom':        {'headwidth': 5, 'scale': 10, 'headlength': 5, 'width': 0.0017},
    'estados_unidos':             {'headwidth': 5, 'scale': 12, 'headlength': 5, 'width': 0.0015},
    'globo_3d':                   {'headwidth': 4, 'scale': 26, 'headlength': 5, 'width': 0.0006},
    'pacific_chile':              {'headwidth': 5, 'scale': 12, 'headlength': 5, 'width': 0.0014},
}
_QUIVER_DEFAULT = {'headwidth': 4, 'scale': 12, 'headlength': 5, 'width': 0.0012}

DEFAULT_AREAS = [
    'america_sul',
    'globo_3d',
    'pacific_chile',
    'china',
    'pacifico_leste_america_sul',
    'america_sul_zom_out',
    'MDR',
    'tropico',
    'zona_zcit_atlantico',
    'brasil',    
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
]


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------
def _get_area_list() -> list[str]:
    if hasattr(settings, 'LST_AREAS_S18'):
        return list(settings.LST_AREAS_S18)
    return list(DEFAULT_AREAS)


def _to_str_date(val) -> str:
    return val.strftime('%Y-%m-%d') if hasattr(val, 'strftime') else str(val)


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


def _configure_gridlines(gl, area):
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger = get_logger(SCRIPT_ID)

    logger.info('=' * 80)
    logger.info('📊 SCRIPT {}: {}', SCRIPT_ID.upper(), SCRIPT_DESC)
    logger.info('=' * 80)

    lst_areas = _get_area_list()
    output_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_TMP850_WND850'
    input_dir = Path(settings.DIR_INPUT)
    dados_dir = Path(settings.DIR_DADOS)

    cache_params = {
        'DATA_INICIAL': settings.DATA_INICIAL,
        'DATA_FINAL': settings.DATA_FINAL,
        'areas': lst_areas,
        'script_version': '1.0',
        'tmp_file': TMP_FILE_NAME,
        'wind_file': WIND850_FILE_NAME,
    }
    output_files = [str(output_dir / f'tmp850_wnd850_{area}.png') for area in lst_areas]

    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('🎯 CACHE VÁLIDO! Execução já foi realizada com os mesmos parâmetros.')
        logger.info('   📅 Período: {} a {}', settings.DATA_INICIAL, settings.DATA_FINAL)
        logger.info('   📊 {} mapas já existem', len(output_files))
        logger.info('   📁 Diretório: {}', output_dir)
        logger.info('   ⏭️  Pulando execução')
        return

    start_time = time.time()
    logger.info('📅 Período de análise: {} a {}', settings.DATA_INICIAL, settings.DATA_FINAL)
    logger.info('📊 Gerando {} mapas — temperatura + vento anomalo 850 hPa', len(lst_areas))
    logger.info('=' * 80)

    # Etapa 1: pipeline temperatura 850 hPa → tmp850.nc
    logger.info('Etapa 1: Preparando anomalia de temperatura 850 hPa')
    plot_tmp850_anom()

    # Etapa 2: pipeline vento 850 hPa → wind850_anom.nc
    logger.info('Etapa 2: Preparando anomalia de vento 850 hPa')
    plot_wind850_anom()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Carregar temperatura 850 hPa
    tmp_file = dados_dir / TMP_FILE_NAME
    if not tmp_file.exists():
        raise FileNotFoundError(f'Arquivo esperado não encontrado: {tmp_file}')
    ds_tmp = xr.open_dataset(tmp_file)
    da_tmp = ds_tmp['tmp']
    if 'time' in da_tmp.dims:
        da_tmp = da_tmp.isel(time=0, drop=True)
    # Normaliza lat/lon
    ren = {}
    if 'latitude' in da_tmp.coords:
        ren['latitude'] = 'lat'
    if 'longitude' in da_tmp.coords:
        ren['longitude'] = 'lon'
    if ren:
        da_tmp = da_tmp.rename(ren)
    da_tmp = da_tmp.assign_coords(lon=((da_tmp['lon'] + 360) % 360))
    da_tmp = da_tmp.sortby('lon')
    if da_tmp['lat'][0] < da_tmp['lat'][-1]:
        da_tmp = da_tmp.sortby('lat', ascending=False)
    lat_tmp = da_tmp['lat'].values
    tmp_cyc, lon_tmp_cyc = add_cyclic_point(da_tmp.values, coord=da_tmp['lon'].values)
    ds_tmp.close()

    # Carregar vento 850 hPa
    wind_file = dados_dir / WIND850_FILE_NAME
    if not wind_file.exists():
        raise FileNotFoundError(f'Arquivo esperado não encontrado: {wind_file}')
    ds_wind = xr.open_dataset(wind_file)
    u_raw = ds_wind['u_anom_mean'].values
    v_raw = ds_wind['v_anom_mean'].values
    lon_wind = ds_wind['lon'].values
    lat_wind = ds_wind['lat'].values
    ds_wind.close()

    # Subsample para quiver: ERA5 1°/1° → a cada 3 pontos (~3°)
    lon_u = lon_wind[::3]
    lat_u = lat_wind[::3]
    zonal = u_raw[::3, ::3]
    meridional = v_raw[::3, ::3]

    # Mascara vetores fracos (mesmo critério do s12)
    ws = (zonal**2 + meridional**2) ** 1.2
    zonal = ma.masked_where(ws < 1, zonal)
    meridional = ma.masked_where(ws < 1, meridional)

    cmap = LinearSegmentedColormap.from_list('anom', settings.LST_ANOM_CORRETA)
    info_plot = settings['areas_plotagem']
    dt_ini_str = _to_str_date(settings.DATA_INICIAL)
    dt_fim_str = _to_str_date(settings.DATA_FINAL)

    logo_path = resolve_logo_path(input_dir)

    for area in lst_areas:
        logger.info('Gerando mapa para area: {}', area_display_name(area))

        is_polar = info_plot[area].get('projection', '') == 'orthographic_south'
        central_lon_plot = info_plot[area]['central_longitude_plot']

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

        if info_plot[area].get('plot_box', False):
            for box in info_plot[area]['lst_boxes']:
                ax.add_patch(patches.Rectangle(
                    (box['x_anc'], box['y_anc']),
                    box['x_larg'], box['y_larg'],
                    linewidth=box['linewidth'],
                    edgecolor=box['edgecolor'],
                    facecolor='none', zorder=100,
                ))

        if area == 'MDR':
            ax.plot(
                [-86, -20, -20, -86, -86], [10, 10, 20, 20, 10],
                color='black', linewidth=3, linestyle='-',
                zorder=500, transform=ccrs.PlateCarree(),
            )

        if area == 'atlantico_tropical':
            legenda_atl = input_dir / 'legenda_atlantic.png'
            if legenda_atl.exists():
                fig.figimage(plt.imread(str(legenda_atl)), 125, 614, zorder=3, alpha=1)
            ax.add_patch(patches.Rectangle((10, -20), -40, 20, linewidth=3, edgecolor='black', facecolor='none', zorder=100))
            ax.add_patch(patches.Rectangle((-15, 5), -40, 20, linewidth=3, edgecolor='blue', facecolor='none', zorder=100))

        if area == 'iod':
            ax.add_patch(patches.Rectangle((50, -10), 20, 20, linewidth=3, edgecolor='black', facecolor='none', zorder=100))
            ax.add_patch(patches.Rectangle((90, -10), 20, 10, linewidth=3, edgecolor='black', facecolor='none', zorder=100))

        if area == 'enso':
            for txt, x, y, cor in [
                ('Niño 1+2', 66.25, -13.64, 'red'),
                ('Niño 3', 34.1, 8.45, 'blue'),
                ('Niño 3.4', 8.6, -9.45, 'black'),
                ('Niño 4', -22.5, 8.45, 'm'),
            ]:
                t = plt.text(x, y, txt, fontsize=14, color=cor, weight='bold', zorder=500)
                fg = 'black' if cor in {'red', 'm'} else 'white'
                t.set_path_effects([path_effects.Stroke(linewidth=3, foreground=fg), path_effects.Normal()])

        if area == 'mjo':
            for x, y, num in [(-135.25, -4.64, '1'), (-115.25, -4.64, '2'), (-95.25, -4.64, '3'),
                               (-75.25, -4.64, '4'), (-55.25, -4.64, '5'), (-35.25, -4.64, '6'),
                               (-15.25, -4.64, '7'), (4.75, -4.64, '8')]:
                t = plt.text(x, y, num, fontsize=50, color='white', weight='bold', zorder=400)
                t.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

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

        data_transform = ccrs.PlateCarree(central_longitude=central_lon_plot)

        # Shaded: anomalia de temperatura 850 hPa
        im = ax.contourf(
            lon_tmp_cyc, lat_tmp, tmp_cyc,
            levels=LEVELS, cmap=cmap, extend='both',
            transform=data_transform, zorder=2,
        )
        # Quiver: vetores de vento anomalo 850 hPa
        qp = _QUIVER_PARAMS.get(area, _QUIVER_DEFAULT)
        ax.quiver(
            lon_u, lat_u, zonal, meridional,
            scale_units='inches',
            color='k',
            headwidth=qp['headwidth'],
            scale=qp['scale'],
            headlength=qp['headlength'],
            width=qp['width'],
            transform=data_transform,
            zorder=50,
        )

        # Colorbar
        if is_polar and area != 'globo_3d':
            cbar = plt.colorbar(im, ax=ax, pad=0.05, fraction=0.04, ticks=TICKS)
            cbar.set_label(label='°C', size=10)
            cbar.ax.tick_params(labelsize=10)
        elif area in {'enso', 'tropico', 'MDR', 'hemisferio_sul', 'psa'}:
            cax = make_axes_locatable(ax).append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)
            cbar = plt.colorbar(
                im, cax=cax, pad=0.02, fraction=0.02375,
                location='bottom', extend='both', orientation='horizontal', ticks=TICKS,
            )
            cbar.set_label(label='°C', size=18)
            cbar.ax.tick_params(labelsize=20)
        else:
            cax = make_axes_locatable(ax).append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
            cbar = plt.colorbar(im, cax=cax, pad=0.02, fraction=0.02375, extend='both', ticks=TICKS)
            cbar.set_label(label='°C', size=18)
            cbar.ax.tick_params(labelsize=20)

        # Título
        titulo = f'Anomalia Temperatura + Vento 850 hPa\nDe {dt_ini_str} a {dt_fim_str}'
        ax.set_title(titulo, fontsize=14 if is_polar else 18, loc='left')

        # Borda do frame
        if area != 'globo_3d':
            ax.add_patch(patches.Rectangle(
                (0, 0), 1, 1,
                linewidth=0.5, edgecolor='black', facecolor='none',
                transform=ax.transAxes, zorder=1000, clip_on=False,
            ))

        if logo_path is not None and logo_path.exists():
            _add_logo_to_map(ax=ax, logo_path=logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=500)

        filename_fig = output_dir / f'tmp850_wnd850_{area}.png'
        logger.info('Salvando figura {}', filename_fig)
        plt.savefig(str(filename_fig), dpi=fig.dpi, bbox_inches='tight')
        plt.close('all')

    execution_time = time.time() - start_time
    save_cache_metadata(SCRIPT_ID, cache_params, output_files, execution_time)
    logger.info('=' * 80)
    logger.info('✅ Script {} concluído com sucesso!', SCRIPT_ID.upper())
    logger.info('⏱️  Tempo de execução: {:.1f}s ({:.1f} min)', execution_time, execution_time / 60)
    logger.info('📊 {} mapas gerados em: {}', len(output_files), output_dir)
    logger.info('=' * 80)


if __name__ == '__main__':
    main()
