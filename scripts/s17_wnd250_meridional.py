"""s17 - Anomalia de Vento Meridional 250 hPa (ERA5/GDAS + PSL).

Pipeline de dados:
  - Período recente (últimos 7 dias): GDAS via NOMADS Grib Filter
  - Período mais antigo: ERA5 via Copernicus CDS
  - Climatologia: PSL/NOAA via Playwright (cache local por período MM-DD)

Saida:
    - Mapas PNG em Saida/s17_WND250_MERIDIONAL/ (um por regiao configurada em LST_AREAS_S17)

Criado em: 2026-06-06
"""

# Bibliotecas padrão
import time
from pathlib import Path

# Bibliotecas de terceiros
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from cartopy.util import add_cyclic_point
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import FixedLocator, MultipleLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from PIL import Image

# Módulos locais
from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.common.dataset_utils import area_display_name, load_dataset
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.common.logo_helper import resolve_logo_path
from app.common.logo_helper import proportional_logo_zoom
from app.src.uteis.plot_wnd_meridional_250_anom import main as plot_wnd_meridional_250_anom

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
SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's17'
SCRIPT_NAME = Path(__file__).stem
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME

# ---------------------------------------------------------------------------
# Constantes de plotagem
# ---------------------------------------------------------------------------
LEVELS = np.arange(-20, 22, 2)
TICKS = np.arange(-20, 22, 4)

WND_MERIDIONAL_FILE_NAME = 'wnd_meridional_250_anom.nc'


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------
def _add_logo_to_map(ax, logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=500):
    logo = Image.open(logo_path).convert('RGBA')
    bbox = logo.getbbox()
    if bbox is not None:
        logo = logo.crop(bbox)
    imagebox = OffsetImage(np.array(logo), zoom=proportional_logo_zoom(ax, np.array(logo).shape[1]))
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


def _standardize_coords(da):
    ren = {}
    if 'latitude' in da.coords:
        ren['latitude'] = 'lat'
    if 'longitude' in da.coords:
        ren['longitude'] = 'lon'
    if ren:
        da = da.rename(ren)
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


def _get_area_list() -> list:
    for attr in ('LST_AREAS_S17', 'LST_AREAS_S01'):
        if hasattr(settings, attr):
            return list(getattr(settings, attr))
    raise AttributeError('Nenhuma lista de áreas encontrada (LST_AREAS_S17 ou LST_AREAS_S01).')


def _to_str_date(val) -> str:
    return val.strftime('%Y-%m-%d') if hasattr(val, 'strftime') else str(val)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger = get_logger(SCRIPT_ID)

    logger.info('=' * 80)
    logger.info('📊 SCRIPT {}: {}', SCRIPT_ID.upper(), SCRIPT_DESC)
    logger.info('=' * 80)

    lst_areas = _get_area_list()
    output_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_WND250_MERIDIONAL'
    input_dir = Path(settings.DIR_INPUT)
    dados_dir = Path(settings.DIR_DADOS)

    cache_params = {
        'DATA_INICIAL': settings.DATA_INICIAL,
        'DATA_FINAL': settings.DATA_FINAL,
        'areas': lst_areas,
        'script_version': '1.0',
        'wnd_file': WND_MERIDIONAL_FILE_NAME,
    }
    output_files = [str(output_dir / f'wnd250_meridional_anom_{area}.png') for area in lst_areas]

    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('🎯 CACHE VÁLIDO! Execução já foi realizada com os mesmos parâmetros.')
        logger.info('   📅 Período: {} a {}', settings.DATA_INICIAL, settings.DATA_FINAL)
        logger.info('   📊 {} mapas já existem', len(output_files))
        logger.info('   📁 Diretório: {}', output_dir)
        logger.info('   ⏭️  Pulando execução')
        return

    start_time = time.time()
    logger.info('📅 Período de análise: {} a {}', settings.DATA_INICIAL, settings.DATA_FINAL)
    logger.info('📊 Gerando {} mapas — vento meridional 250 hPa', len(lst_areas))
    logger.info('=' * 80)

    # Etapa 1: pipeline v-meridional 250 hPa → wnd_meridional_250_anom.nc
    logger.info('Etapa 1: Preparando anomalia de vento meridional 250 hPa')
    plot_wnd_meridional_250_anom()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Carregar wnd_meridional_250_anom.nc
    wnd_file = dados_dir / WND_MERIDIONAL_FILE_NAME
    if not wnd_file.exists():
        raise FileNotFoundError(f'Arquivo esperado não encontrado: {wnd_file}')
    ds_wnd = load_dataset(str(wnd_file))
    da_wnd = _standardize_coords(ds_wnd['v_anom_mean'])
    wnd_cyc, lon_wnd_cyc = add_cyclic_point(da_wnd.values, coord=da_wnd['lon'].values)
    lat_wnd = da_wnd['lat'].values

    cmap = LinearSegmentedColormap.from_list('anom', settings.LST_ANOM_CORRETA)
    info_plot = settings['areas_plotagem']
    dt_ini_str = _to_str_date(settings.DATA_INICIAL)
    dt_fim_str = _to_str_date(settings.DATA_FINAL)

    logo_path = resolve_logo_path(input_dir)

    for area in lst_areas:
        logger.info('Gerando mapa para area: {}', area_display_name(area))

        is_polar = info_plot[area].get('projection', '') == 'orthographic_south'
        central_lon = 0 if is_polar else info_plot[area]['central_longitude_mapa']
        data_transform = ccrs.PlateCarree(
            central_longitude=info_plot[area]['central_longitude_plot']
        )

        if is_polar:
            proj = ccrs.Orthographic(
                central_longitude=settings.get('ORTHO_CENTRAL_LONGITUDE', info_plot[area].get('ortho_central_longitude', -71)),
                central_latitude=settings.get('ORTHO_CENTRAL_LATITUDE', info_plot[area].get('ortho_central_latitude', -84)),
            )
        else:
            proj = ccrs.PlateCarree(central_longitude=central_lon)

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
            img_legenda_atlantic = plt.imread(str(input_dir / 'legenda_atlantic.png'))
            fig.figimage(img_legenda_atlantic, 125, 614, zorder=3, alpha=1)
            ax.add_patch(patches.Rectangle((10, -20), -40, 20, linewidth=3, edgecolor='black', facecolor='none', zorder=100))
            ax.add_patch(patches.Rectangle((-15, 5), -40, 20, linewidth=3, edgecolor='blue', facecolor='none', zorder=100))

        if area == 'iod':
            ax.add_patch(patches.Rectangle((50, -10), 20, 20, linewidth=3, edgecolor='black', facecolor='none', zorder=100))
            ax.add_patch(patches.Rectangle((90, -10), 20, 10, linewidth=3, edgecolor='black', facecolor='none', zorder=100))

        if area == 'enso':
            for x, y, label, color, stroke_color in [
                (66.25, -13.64, 'Niño 1+2', 'red', 'black'),
                (34.1, 8.45, 'Niño 3', 'blue', 'white'),
                (8.6, -9.45, 'Niño 3.4', 'black', 'white'),
                (-22.5, 8.45, 'Niño 4', 'm', 'black'),
            ]:
                t = plt.text(x, y, label, fontsize=14, color=color, weight='bold', zorder=500)
                t.set_path_effects([path_effects.Stroke(linewidth=3, foreground=stroke_color), path_effects.Normal()])

        if area == 'mjo':
            for x, num in zip([-135.25, -115.25, -95.25, -75.25, -55.25, -35.25, -15.25, 4.75], range(1, 9)):
                t = plt.text(x, -4.64, str(num), fontsize=50, color='white', weight='bold', zorder=400)
                t.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

        # Grade
        if is_polar:
            gl = ax.gridlines(draw_labels=False, linestyle='--', alpha=0.5)
            gl.xlocator = MultipleLocator(30)
            gl.ylocator = MultipleLocator(20)
        else:
            gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.0)
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 20, 'color': 'black'}
            gl.ylabel_style = {'size': 20, 'color': 'black'}

            _grid_config = {
                'hemisferio_sul': (40, 20), 'psa': (40, 20), 'estados_unidos': (10, 10),
                'estados_unidos_zoom': (10, 5), 'argentina': (5, 5), 'brasil': (10, 5),
                'america_sul': (20, 20), 'zona_zcit_atlantico': (10, 5), 'costa_brasil': (5, 5),
                'atlantico_tropical': (20, 10), 'tsa': (10, 5), 'tna': (10, 5),
                'pdo': (20, 10), 'iod': (20, 10), 'sad': (20, 10), 'amo': (20, 10),
                'mjo': (40, 20), 'africa': (20, 20), 'africa_monsoon': (20, 10),
            }
            if area in _grid_config:
                gl.xlocator = MultipleLocator(_grid_config[area][0])
                gl.ylocator = MultipleLocator(_grid_config[area][1])
            elif area == 'globo':
                gl.xlocator = MultipleLocator(40)
                gl.ylocator = MultipleLocator(20)
                gl.xlabel_style = {'size': 15, 'color': 'black'}
                gl.ylabel_style = {'size': 15, 'color': 'black'}
            elif area == 'enso':
                gl.xlocator = FixedLocator([-160, -140, -120, -100, -80, -60, 0, 20, 40, 60, 80, 100, 120, 140, 150, 160, 170, 180])
                gl.ylocator = MultipleLocator(10)
                gl.xlabel_style = {'size': 15, 'color': 'black'}
                gl.ylabel_style = {'size': 15, 'color': 'black'}
            elif area == 'tropico':
                gl.xlocator = FixedLocator([-160, -120, -80, -40, 0, 40, 80, 120, 160])
                gl.ylocator = MultipleLocator(20)
                gl.xlabel_style = {'size': 15, 'color': 'black'}
                gl.ylabel_style = {'size': 15, 'color': 'black'}

        if not is_polar:
            ax.set_xlim([info_plot[area]['lon_esq'], info_plot[area]['lon_dir']])
            ax.set_ylim([info_plot[area]['lat_inf'], info_plot[area]['lat_sup']])

        ax.set_facecolor('white')
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, zorder=100)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, zorder=100)
        if area != 'china':
            ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1.2, zorder=100)

        # Shaded: anomalia de vento meridional 250 hPa
        im = ax.contourf(
            lon_wnd_cyc, lat_wnd, wnd_cyc,
            levels=LEVELS, cmap=cmap, extend='both',
            transform=data_transform, zorder=2,
        )
        ax.contour(
            lon_wnd_cyc, lat_wnd, wnd_cyc,
            levels=LEVELS, colors='white', linewidths=0.6,
            transform=data_transform, zorder=3,
        )

        # Colorbar
        if is_polar and area != 'globo_3d':
            cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.05, ticks=TICKS)
            cbar.set_label(label='m s⁻¹', size=10)
            cbar.ax.tick_params(labelsize=10)
        elif area in {'america_sul', 'globo_3d'}:
            cax = make_axes_locatable(ax).append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
            cbar = plt.colorbar(im, cax=cax, pad=0.02, fraction=0.02375, extend='both', ticks=TICKS)
            cbar.set_label(label='m s⁻¹', size=18)
            cbar.ax.tick_params(labelsize=20)
        elif area in {'enso', 'tropico', 'MDR', 'hemisferio_sul', 'psa'}:
            cax = make_axes_locatable(ax).append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)
            cbar = plt.colorbar(
                im, cax=cax, pad=0.02, fraction=0.02375,
                location='bottom', extend='both', orientation='horizontal', ticks=TICKS,
            )
            cbar.set_label(label='m s⁻¹', size=18)
            cbar.ax.tick_params(labelsize=20)
        else:
            cax = make_axes_locatable(ax).append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
            cbar = plt.colorbar(im, cax=cax, pad=0.02, fraction=0.02375, extend='both', ticks=TICKS)
            cbar.set_label(label='m s⁻¹', size=18)
            cbar.ax.tick_params(labelsize=20)

        # Título
        titulo = f'Anomalia Vento Meridional 250hPa (De {dt_ini_str} a {dt_fim_str})'
        ax.set_title(titulo, fontsize=14 if is_polar else 18, loc='left')

        # Borda do frame (exceto globo_3d — projeção ortográfica não tem borda retangular)
        if area != 'globo_3d':
            ax.add_patch(patches.Rectangle(
                (0, 0), 1, 1,
                linewidth=0.5, edgecolor='black', facecolor='none',
                transform=ax.transAxes, zorder=1000, clip_on=False,
            ))

        if logo_path is not None and logo_path.exists():
            _add_logo_to_map(ax=ax, logo_path=logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=500)

        filename_fig = output_dir / f'wnd250_meridional_anom_{area}.png'
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
