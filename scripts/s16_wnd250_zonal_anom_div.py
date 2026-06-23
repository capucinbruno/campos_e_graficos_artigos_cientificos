"""s16 - Anomalia de Vento Zonal 250 hPa + Vento Divergente (ERA5/GDAS + PSL).

Pipeline de dados:
  - Vento zonal 250 hPa (u_anom_mean): ERA5/GDAS + climatologia PSL/NOAA
  - Vento divergente 200 hPa: reutiliza chi200.nc calculado pelo s03/plot_chi200
  - Shaded: anomalia de vento zonal (paleta de anomalia padrão do projeto)
  - Quiver: componente divergente/irrotacional do vento 200 hPa

Saida:
    - Mapas PNG em Saida/s16_WND250_ZONAL_ANOM/ (um por regiao configurada em LST_AREAS_S16)

Criado em: 2026-06-05
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
from app.common.dataset_utils import (
    area_display_name,
    arquivo_cobre_periodo,
    load_dataset,
)
from app.common.download_helper import DownloadEngine, download_with_progress
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.common.logo_helper import resolve_logo_path
from app.common.logo_helper import proportional_logo_zoom
from app.src.uteis.plot_chi200 import main as plot_chi200
from app.src.uteis.plot_wnd_speed_250 import main as plot_wnd_speed_250
from app.src.uteis.plot_wnd_zonal_250_anom import main as plot_wnd_zonal_250_anom
from app.src.uteis.plot_z250_mean import main as plot_z250_mean

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
SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's16'
SCRIPT_NAME = Path(__file__).stem
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME

# ---------------------------------------------------------------------------
# Constantes de plotagem
# ---------------------------------------------------------------------------
LEVELS = np.arange(-20, 22, 2)
TICKS = np.arange(-20, 22, 4)

LEVELS_POS = np.arange(1, 21, 1)
TICKS_POS = np.arange(5, 21, 5)

CMAP_POS_COLORS = [
    '#9cdafa', '#53bff7', '#5393f7',
    '#f2b0bf', '#f2849e', '#ee5278', '#f22457',
    '#c9c9c9', '#e7e7e7',
]

LEVELS_MAG = np.arange(25, 90, 5)   # magnitude de vento 250 hPa (m/s); abaixo de 25 fica transparente
TICKS_MAG = np.arange(30, 90, 10)
CMAP_MAG_COLORS = [
    '#9cdafa', '#53bff7', '#5393f7',
    '#f2b0bf', '#f2849e', '#ee5278', '#f22457',
    '#c9c9c9', '#e7e7e7',
]

LEVELS_OLR_NEG = [-60, -50, -40, -30, -20, -10]  # anomalia OLR negativa (W m-2)
CMAP_OLR_NEG_COLORS = ['#005a00', '#1a7a00', '#2e9e00', '#50c000', '#80d840']

OLR_URL = 'https://downloads.psl.noaa.gov/Datasets/cpc_blended_olr-2.5deg/olr.day.anom.nc'
OLR_FILE_NAME = 'olr.day.anom.nc'
WND_SPEED_FILE_NAME = 'wnd_speed_250.nc'
Z250_FILE_NAME = 'z250_mean.nc'

CHI_FILE_NAME = 'chi200.nc'
WND_ZONAL_FILE_NAME = 'wnd_zonal_250_anom.nc'

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

UCHI_CANDIDATES = ('uchi_anom_mean', 'uchi', 'uchi_anom', 'udiv', 'uchi_irrot')
VCHI_CANDIDATES = ('vchi_anom_mean', 'vchi', 'vchi_anom', 'vdiv', 'vchi_irrot')
CHI_CANDIDATES = ('chi_anom_mean_scaled', 'chi_anom_mean', 'chi', 'chi_anom', 'velocity_potential_anomaly')


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


def _pick_first_var(ds, candidates):
    for name in candidates:
        if name in ds.data_vars:
            return ds[name]
    raise KeyError(
        f'Nenhuma das variáveis {candidates} encontrada. '
        f'Disponíveis: {list(ds.data_vars)}'
    )


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


def _add_cyclic_2d(da2d):
    data_cyc, lon_cyc = add_cyclic_point(da2d.values, coord=da2d['lon'].values)
    return data_cyc, lon_cyc


def _add_cyclic_uv(u2d, v2d):
    u_cyc, lon_cyc = add_cyclic_point(u2d.values, coord=u2d['lon'].values)
    v_cyc, _ = add_cyclic_point(v2d.values, coord=v2d['lon'].values)
    return u_cyc, v_cyc, lon_cyc


def _prepare_quiver_masked(lon, lat, u, v, *, step, min_mag, lat_bounds=None, margin=3.0):
    lon_q = lon[::step]
    lat_q = lat[::step]
    u_q = u[::step, ::step]
    v_q = v[::step, ::step]
    mag_q = np.sqrt(u_q**2 + v_q**2)
    mask = mag_q < min_mag
    if lat_bounds is not None:
        lat_min, lat_max = sorted(lat_bounds)
        lat_grid = lat_q[:, None] if lat_q.ndim == 1 else lat_q
        outside_lat = (lat_grid < lat_min + margin) | (lat_grid > lat_max - margin)
        if lat_q.ndim == 1:
            outside_lat = np.broadcast_to(outside_lat, u_q.shape)
        mask = mask | outside_lat
    return lon_q, lat_q, np.ma.masked_where(mask, u_q), np.ma.masked_where(mask, v_q)


def _get_quiver_config(area: str) -> dict:
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


def _get_area_list() -> list:
    for attr in ('LST_AREAS_S16', 'LST_AREAS_S01'):
        if hasattr(settings, attr):
            return list(getattr(settings, attr))
    raise AttributeError('Nenhuma lista de áreas encontrada (LST_AREAS_S16 ou LST_AREAS_S01).')


def _to_str_date(val) -> str:
    return val.strftime('%Y-%m-%d') if hasattr(val, 'strftime') else str(val)


def _style_contour_labels(txts) -> None:
    """Aplica estilo 'placa branca' (bbox) aos rótulos de isolinhas."""
    for txt in txts:
        txt.set_bbox(dict(
            boxstyle='round,pad=0.25',
            facecolor='white',
            edgecolor='none',
            alpha=0.85,
        ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger = get_logger(SCRIPT_ID)

    logger.info('=' * 80)
    logger.info('📊 SCRIPT {}: {}', SCRIPT_ID.upper(), SCRIPT_DESC)
    logger.info('=' * 80)

    lst_areas = _get_area_list()
    output_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_WND250_ZONAL_ANOM'
    input_dir = Path(settings.DIR_INPUT)
    dados_dir = Path(settings.DIR_DADOS)

    cache_params = {
        'DATA_INICIAL': settings.DATA_INICIAL,
        'DATA_FINAL': settings.DATA_FINAL,
        'areas': lst_areas,
        'script_version': '1.9',  # OLR nao-fatal: pula a camada se periodo nao coberto
        'wnd_file': WND_ZONAL_FILE_NAME,
        'chi_file': CHI_FILE_NAME,
        'speed_file': WND_SPEED_FILE_NAME,
        'olr_file': OLR_FILE_NAME,
        'z250_file': Z250_FILE_NAME,
    }
    output_files = [
        str(output_dir / f'wnd250_zonal_anom_{area}{suffix}.png')
        for area in lst_areas
        for suffix in ('', '_pos', '_nodiv', '_mag')
    ]

    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('🎯 CACHE VÁLIDO! Execução já foi realizada com os mesmos parâmetros.')
        logger.info('   📅 Período: {} a {}', settings.DATA_INICIAL, settings.DATA_FINAL)
        logger.info('   📊 {} mapas já existem', len(output_files))
        logger.info('   📁 Diretório: {}', output_dir)
        logger.info('   ⏭️  Pulando execução')
        return

    start_time = time.time()
    logger.info('📅 Período de análise: {} a {}', settings.DATA_INICIAL, settings.DATA_FINAL)
    logger.info('📊 Gerando {} mapas — vento zonal 250 hPa + vento divergente', len(lst_areas))
    logger.info('=' * 80)

    # Etapa 1: pipeline vento zonal 250 hPa → wnd_zonal_250_anom.nc
    logger.info('Etapa 1: Preparando anomalia de vento zonal 250 hPa')
    plot_wnd_zonal_250_anom()

    # Etapa 2: pipeline CHI200 → chi200.nc (vento divergente)
    logger.info('Etapa 2: Preparando CHI200 e vento divergente 200 hPa')
    plot_chi200()

    # Etapa 3: magnitude vento 250 hPa → wnd_speed_250.nc
    logger.info('Etapa 3: Calculando magnitude vento 250 hPa (sqrt(u²+v²))')
    plot_wnd_speed_250()

    # Etapa 4: altura geopotencial ERA5 250 hPa → z250_mean.nc
    logger.info('Etapa 4: Calculando altura geopotencial média 250 hPa (ERA5)')
    plot_z250_mean()

    # Etapa 5: OLR anomalia (PSL/NOAA) → olr.day.anom.nc
    logger.info('Etapa 5: Preparando anomalia de OLR')
    olr_path = dados_dir / OLR_FILE_NAME
    start_date = np.datetime64(settings.DATA_INICIAL)
    end_date = np.datetime64(settings.DATA_FINAL)
    if not arquivo_cobre_periodo(olr_path, start_date, end_date):
        download_with_progress(
            url=OLR_URL,
            output_path=str(olr_path),
            description=OLR_FILE_NAME,
            max_retries=5,
            force=True,
            engine=DownloadEngine.ARIA2,
            timeout=300,
        )
    ds_olr_full = load_dataset(str(olr_path))
    # O OLR (CPC Blended) tem latencia (~5 dias) e nem sempre cobre ate DATA_FINAL. Em vez de
    # abortar o script inteiro, a camada de OLR (usada SO no mapa _mag) e PULADA quando o periodo
    # nao esta coberto — os demais mapas (full/pos/nodiv/mag sem OLR) sao gerados normalmente.
    olr_tmin = ds_olr_full['time'].values.min()
    olr_tmax = ds_olr_full['time'].values.max()
    if olr_tmin <= start_date and olr_tmax >= end_date:
        da_olr_mean = ds_olr_full.sel(time=slice(start_date, end_date)).mean(dim='time')['olr']
        # Lon 0-360 → -180..180 para consistência com add_cyclic_point
        if float(da_olr_mean['lon'].min()) >= 0:
            da_olr_mean = da_olr_mean.assign_coords(
                lon=((da_olr_mean['lon'].values + 180) % 360) - 180
            ).sortby('lon')
        # Lat descendente
        if da_olr_mean['lat'][0] < da_olr_mean['lat'][-1]:
            da_olr_mean = da_olr_mean.sortby('lat', ascending=False)
        olr_cyc, lon_olr_cyc = add_cyclic_point(da_olr_mean.values, coord=da_olr_mean['lon'].values)
        lat_olr = da_olr_mean['lat'].values
    else:
        logger.warning(
            '⚠️  OLR indisponivel para o periodo solicitado (arquivo vai ate {}, DATA_FINAL={}). '
            'Pulando a camada de OLR — os demais mapas sao gerados normalmente.',
            str(olr_tmax)[:10], _to_str_date(settings.DATA_FINAL))
        olr_cyc = lon_olr_cyc = lat_olr = None

    output_dir.mkdir(parents=True, exist_ok=True)

    # Carregar wnd_zonal_250_anom.nc
    wnd_file = dados_dir / WND_ZONAL_FILE_NAME
    if not wnd_file.exists():
        raise FileNotFoundError(f'Arquivo esperado não encontrado: {wnd_file}')
    ds_wnd = load_dataset(str(wnd_file))
    da_wnd = ds_wnd['u_anom_mean']
    da_wnd = _standardize_coords(da_wnd)
    wnd_cyc, lon_wnd_cyc = _add_cyclic_2d(da_wnd)
    lat_wnd = da_wnd['lat'].values

    # Carregar wnd_speed_250.nc
    spd_file = dados_dir / WND_SPEED_FILE_NAME
    if not spd_file.exists():
        raise FileNotFoundError(f'Arquivo esperado não encontrado: {spd_file}')
    ds_spd = load_dataset(str(spd_file))
    da_spd = _standardize_coords(ds_spd['speed_mean'])
    spd_cyc, lon_spd_cyc = _add_cyclic_2d(da_spd)
    lat_spd = da_spd['lat'].values

    # Carregar chi200.nc (vento divergente)
    chi_file = dados_dir / CHI_FILE_NAME
    if not chi_file.exists():
        raise FileNotFoundError(f'Arquivo esperado não encontrado: {chi_file}')
    ds_chi = load_dataset(str(chi_file))
    da_uchi = _standardize_coords(_pick_first_var(ds_chi, UCHI_CANDIDATES))
    da_vchi = _standardize_coords(_pick_first_var(ds_chi, VCHI_CANDIDATES))
    da_chi_scalar = _standardize_coords(_pick_first_var(ds_chi, CHI_CANDIDATES))
    uchi_cyc, vchi_cyc, lon_chi_cyc = _add_cyclic_uv(da_uchi, da_vchi)
    chi_scalar_cyc, _ = _add_cyclic_2d(da_chi_scalar)
    lat_chi = da_uchi['lat'].values

    # Carregar z250_mean.nc (altura geopotencial média 250 hPa)
    z250_file = dados_dir / Z250_FILE_NAME
    if not z250_file.exists():
        raise FileNotFoundError(f'Arquivo esperado não encontrado: {z250_file}')
    ds_z250 = load_dataset(str(z250_file))
    da_z250 = _standardize_coords(ds_z250['z_mean'])
    z250_cyc, lon_z250_cyc = _add_cyclic_2d(da_z250)
    lat_z250 = da_z250['lat'].values

    cmap_full = LinearSegmentedColormap.from_list('anom', settings.LST_ANOM_CORRETA)
    cmap_pos = LinearSegmentedColormap.from_list('anom_pos', CMAP_POS_COLORS)
    cmap_mag = LinearSegmentedColormap.from_list('wnd_speed', CMAP_MAG_COLORS)
    cmap_olr_neg = LinearSegmentedColormap.from_list('olr_neg', CMAP_OLR_NEG_COLORS)
    info_plot = settings['areas_plotagem']

    for area in lst_areas:
        logger.info('Gerando mapas para area: {}', area_display_name(area))

        is_polar = info_plot[area].get('projection', '') == 'orthographic_south'
        central_lon = 0 if is_polar else info_plot[area]['central_longitude_mapa']
        data_transform = ccrs.PlateCarree(
            central_longitude=info_plot[area]['central_longitude_plot']
        )
        qcfg = _get_quiver_config(area)
        _AREAS_COM_CORTE_BORDA = {'globo', 'psa', 'hemisferio_sul'}
        dt_ini_str = _to_str_date(settings.DATA_INICIAL)
        dt_fim_str = _to_str_date(settings.DATA_FINAL)

        # Quiver calculado uma vez por área (mesmo para os dois modos)
        step = int(qcfg['step'])
        lon_q, lat_q, u_q_mask, v_q_mask = _prepare_quiver_masked(
            lon=lon_chi_cyc,
            lat=lat_chi,
            u=uchi_cyc,
            v=vchi_cyc,
            step=step,
            min_mag=float(qcfg['min_mag']),
            lat_bounds=(info_plot[area]['lat_inf'], info_plot[area]['lat_sup'])
            if area in _AREAS_COM_CORTE_BORDA
            else None,
        )
        # Restringir quiver às regiões com chi200 negativo
        chi_q = chi_scalar_cyc[::step, ::step]
        u_q_mask = np.ma.masked_where(chi_q >= 0, u_q_mask)
        v_q_mask = np.ma.masked_where(chi_q >= 0, v_q_mask)

        logo_path = resolve_logo_path(input_dir)

        for mode in ('full', 'pos', 'nodiv', 'mag'):
            if mode == 'pos':
                use_levels, use_ticks, use_extend, use_cmap = LEVELS_POS, TICKS_POS, 'max', cmap_pos
                use_data, use_lon, use_lat = wnd_cyc, lon_wnd_cyc, lat_wnd
            elif mode == 'mag':
                use_levels, use_ticks, use_extend, use_cmap = LEVELS_MAG, TICKS_MAG, 'max', cmap_mag
                use_data, use_lon, use_lat = spd_cyc, lon_spd_cyc, lat_spd
            else:
                use_levels, use_ticks, use_extend, use_cmap = LEVELS, TICKS, 'both', cmap_full
                use_data, use_lon, use_lat = wnd_cyc, lon_wnd_cyc, lat_wnd
            file_suffix = {'full': '', 'pos': '_pos', 'nodiv': '_nodiv', 'mag': '_mag'}[mode]

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
            ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#d4d4d4', zorder=1)
            ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, zorder=100)
            ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, zorder=100)
            if area != 'china':
                ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1.2, zorder=100)

            # Shaded: vento zonal anom (full/pos/nodiv) ou magnitude 250 hPa (mag)
            im = ax.contourf(
                use_lon, use_lat, use_data,
                levels=use_levels, cmap=use_cmap, extend=use_extend,
                transform=data_transform, zorder=2,
            )

            # OLR: anomalia negativa em verde (apenas modo mag; pulado se OLR indisponivel)
            if mode == 'mag' and olr_cyc is not None:
                ax.contourf(
                    lon_olr_cyc, lat_olr, olr_cyc,
                    levels=LEVELS_OLR_NEG, cmap=cmap_olr_neg, extend='min',
                    transform=data_transform, zorder=4, alpha=0.5,
                )

            # Z250: isolinhas de altura geopotencial (apenas modo mag)
            if mode == 'mag':
                cs_blue = ax.contour(
                    lon_z250_cyc, lat_z250, z250_cyc,
                    levels=[9960.0, 10200.0],
                    colors=['blue'],
                    linewidths=2.0,
                    transform=data_transform,
                    zorder=900,
                )
                txts_blue = ax.clabel(
                    cs_blue,
                    fmt={9960.0: '9960', 10200.0: '10200'},
                    inline=False,
                    fontsize=12,
                    colors=['blue'],
                )
                _style_contour_labels(txts_blue)

                cs_red = ax.contour(
                    lon_z250_cyc, lat_z250, z250_cyc,
                    levels=[10440.0, 10680.0],
                    colors=['red'],
                    linewidths=2.2,
                    transform=data_transform,
                    zorder=900,
                )
                txts_red = ax.clabel(
                    cs_red,
                    fmt={10440.0: '10440', 10680.0: '10680'},
                    inline=False,
                    fontsize=12,
                    colors=['red'],
                )
                _style_contour_labels(txts_red)

            # Quiver: vento divergente 200 hPa (chi200.nc) — omitido no modo nodiv
            if mode != 'nodiv':
                ax.quiver(
                    lon_q, lat_q, u_q_mask, v_q_mask,
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
                cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.05, ticks=use_ticks)
                cbar.set_label(label='m s⁻¹', size=10)
                cbar.ax.tick_params(labelsize=10)
            elif area in {'america_sul', 'globo_3d'}:
                cax = make_axes_locatable(ax).append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
                cbar = plt.colorbar(im, cax=cax, pad=0.02, fraction=0.02375, extend=use_extend, ticks=use_ticks)
                cbar.set_label(label='m s⁻¹', size=18)
                cbar.ax.tick_params(labelsize=20)
            elif area in {'enso', 'tropico', 'MDR', 'hemisferio_sul', 'psa'}:
                cax = make_axes_locatable(ax).append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)
                cbar = plt.colorbar(
                    im, cax=cax, pad=0.02, fraction=0.02375,
                    location='bottom', extend=use_extend, orientation='horizontal', ticks=use_ticks,
                )
                cbar.set_label(label='m s⁻¹', size=18)
                cbar.ax.tick_params(labelsize=20)
            else:
                cax = make_axes_locatable(ax).append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
                cbar = plt.colorbar(im, cax=cax, pad=0.02, fraction=0.02375, extend=use_extend, ticks=use_ticks)
                cbar.set_label(label='m s⁻¹', size=18)
                cbar.ax.tick_params(labelsize=20)

            # Título
            if mode == 'mag':
                titulo = (
                    f'Magnitude Vento 250hPa + Vento Divergente 200hPa (chi<0) + Anom OLR<0\n'
                    f'De {dt_ini_str} a {dt_fim_str}'
                )
            else:
                titulo = (
                    f'Anomalia Vento Zonal 250hPa + Vento Divergente 200hPa\n'
                    f'De {dt_ini_str} a {dt_fim_str}'
                )
            ax.set_title(titulo, fontsize=14 if is_polar else 18, loc='left')

            # Borda do frame (exceto globo_3d — projeção ortográfica não tem borda retangular)
            if area != 'globo_3d':
                ax.add_patch(patches.Rectangle(
                    (0, 0), 1, 1,
                    linewidth=0.5, edgecolor='black', facecolor='none',
                    transform=ax.transAxes, zorder=1000, clip_on=False,
                ))

            if logo_path is not None and logo_path.exists():
                _add_logo_to_map(ax=ax, logo_path=logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=1100)

            filename_fig = output_dir / f'wnd250_zonal_anom_{area}{file_suffix}.png'
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
