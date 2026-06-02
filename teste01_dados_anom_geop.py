"""s01 (TESTE) - Anomalia de Geopotencial 250 hPa via API Ampere de Reanalise.

Versao de teste do s01 que substitui o download direto do CDS pela API de
reanalise da Ampere Consultoria. O template global anomalia_dados.json em
Entrada/ define o periodo, janela climatologica e filtros.

Como a anomalia de z (geopotencial, escalar) e calculada pelo proprio
endpoint /anomalia da API, basta UMA chamada. O resultado vem em m^2/s^2
e e dividido por g=9.80665 para virar metros geopotenciais (mgp).

Saida:
    - Mapas PNG em Saida/teste_dados_anom_geop_GEOP250/ (um por area)

Criado em: 2026-05-15
"""

# Bibliotecas padrão
import json
import time
from copy import deepcopy
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
from matplotlib.ticker import FixedLocator, MultipleLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from PIL import Image

# Módulos locais
from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.src.reanalise import ReanaliseAPI

# ---------------------------------------------------------------------------
# Identidade do script (nome completo para nao colidir com teste_dados_anom.py)
# ---------------------------------------------------------------------------
SCRIPT_ID = Path(__file__).stem  # 'teste_dados_anom_geop'
SCRIPT_NAME = Path(__file__).stem
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME

# ---------------------------------------------------------------------------
# Especificacao ERA5 do script
# ---------------------------------------------------------------------------
DATASET = 'reanalysis-era5-pressure-levels'
VAR_Z = 'z'
NIVEL_Z = 250  # hPa

# Geopotencial vem em m^2/s^2; dividir por g converte para metros geopotenciais.
G_GRAVIDADE = 9.80665

# Templates globais (mesmo arquivo usado pelo teste_dados_anom.py).
PATH_TEMPLATE_ANOMALIA = Path(settings.DIR_INPUT) / 'anomalia_dados.json'

# Chaves aceitas pelo endpoint /anomalia.
_ANOMALIA_KEYS = {
    'nm_dataset',
    'nm_variavel',
    'data_inicial',
    'data_final',
    'ano_referencia_inicio',
    'ano_referencia_fim',
    'tipo_agregacao',
    'pressure_level_hpa',
    'filtro_anos',
    'filtro_anos_referencia',
    'filtro_meses',
    'filtro_dias',
    'filtro_horas',
}


# =============================================================================
# Helpers de template / API
# =============================================================================
def _load_template(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f'Template JSON nao encontrado: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _render_template(template: dict, var: str, nivel: int | None, **overrides) -> dict:
    payload = deepcopy(template)
    payload['nm_dataset'] = DATASET
    payload['nm_variavel'] = var
    if nivel is None:
        payload.pop('pressure_level_hpa', None)
    else:
        payload['pressure_level_hpa'] = nivel
    payload.update(overrides)
    return payload


def _payload_for_anomalia(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k in _ANOMALIA_KEYS}


def _to_dataarray(ds: xr.Dataset, expected_var: str) -> xr.DataArray:
    """Extrai o DataArray (lat, lon) do Dataset da API; converte object->float64."""
    if expected_var in ds.data_vars:
        da = ds[expected_var]
    else:
        da = ds[list(ds.data_vars)[0]]
    if da.dtype == object:
        da = xr.DataArray(
            np.asarray(da.values, dtype='float64'),
            dims=da.dims, coords=da.coords, attrs=da.attrs, name=da.name,
        )
    return da


def _build_periodo_label(dias_obs: pd.DatetimeIndex, di: str, df: str) -> str:
    if len(dias_obs) == 1:
        return f'Data: {pd.Timestamp(dias_obs[0]):%d-%m-%Y}'
    di_fmt = datetime.strptime(di, '%Y-%m-%d').strftime('%d-%m-%y')
    df_fmt = datetime.strptime(df, '%Y-%m-%d').strftime('%d-%m-%y')
    return f'De {di_fmt} a {df_fmt}'


def _filter_dias_by_template(dias: pd.DatetimeIndex, template: dict) -> pd.DatetimeIndex:
    f_anos = template.get('filtro_anos') or []
    f_meses = template.get('filtro_meses') or []
    f_dias = template.get('filtro_dias') or []

    keep = np.ones(len(dias), dtype=bool)
    if f_anos:
        keep &= np.isin(dias.year, f_anos)
    if f_meses:
        keep &= np.isin(dias.month, f_meses)
    if f_dias:
        keep &= np.isin(dias.day, f_dias)
    return dias[keep]


# =============================================================================
# Helpers de plotagem (portados do s01)
# =============================================================================
def _add_logo_to_map(ax, logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=500):
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


def _apply_area_locators(gl, area: str, is_polar: bool):
    """Configura xlocator/ylocator/labels conforme area (portado do s01)."""
    if area == 'hemisferio_sul':
        gl.xlocator = MultipleLocator(40); gl.ylocator = MultipleLocator(20)
    elif area == 'psa':
        gl.xlocator = MultipleLocator(40); gl.ylocator = MultipleLocator(20)
    elif area == 'estados_unidos':
        gl.xlocator = MultipleLocator(10); gl.ylocator = MultipleLocator(10)
    elif area == 'estados_unidos_zoom':
        gl.xlocator = MultipleLocator(10); gl.ylocator = MultipleLocator(5)
    elif area == 'argentina':
        gl.xlocator = MultipleLocator(5); gl.ylocator = MultipleLocator(5)
    elif area == 'brasil':
        gl.xlocator = MultipleLocator(10); gl.ylocator = MultipleLocator(5)
    elif area == 'america_sul':
        gl.xlocator = MultipleLocator(20); gl.ylocator = MultipleLocator(20)
    elif area == 'zona_zcit_atlantico':
        gl.xlocator = MultipleLocator(10); gl.ylocator = MultipleLocator(5)
    elif area == 'costa_brasil':
        gl.xlocator = MultipleLocator(5); gl.ylocator = MultipleLocator(5)
    elif area == 'enso':
        gl.xlocator = FixedLocator([
            -160, -140, -120, -100, -80, -60,
            0, 20, 40, 60, 80, 100, 120, 140, 150, 160, 170, 180,
        ])
        gl.ylocator = MultipleLocator(10)
        gl.xlabel_style = {'size': 15, 'color': 'black'}
        gl.ylabel_style = {'size': 15, 'color': 'black'}
    elif area == 'atlantico_tropical':
        gl.xlocator = MultipleLocator(20); gl.ylocator = MultipleLocator(10)
    elif area in ('tsa', 'tna'):
        gl.xlocator = MultipleLocator(10); gl.ylocator = MultipleLocator(5)
    elif area == 'globo':
        gl.xlocator = MultipleLocator(40); gl.ylocator = MultipleLocator(20)
        gl.xlabel_style = {'size': 15, 'color': 'black'}
        gl.ylabel_style = {'size': 15, 'color': 'black'}
    elif area in ('pdo', 'iod', 'sad', 'amo'):
        gl.xlocator = MultipleLocator(20); gl.ylocator = MultipleLocator(10)
    elif area == 'mjo':
        gl.xlocator = MultipleLocator(40); gl.ylocator = MultipleLocator(20)
    elif area == 'africa':
        gl.xlocator = MultipleLocator(20); gl.ylocator = MultipleLocator(20)
    elif area == 'africa_monsoon':
        gl.xlocator = MultipleLocator(20); gl.ylocator = MultipleLocator(10)
    elif area == 'tropico':
        gl.xlocator = FixedLocator([-160, -120, -80, -40, 0, 40, 80, 120, 160])
        gl.ylocator = MultipleLocator(20)
        gl.xlabel_style = {'size': 15, 'color': 'black'}
        gl.ylabel_style = {'size': 15, 'color': 'black'}
    elif is_polar:
        gl.xlocator = MultipleLocator(30); gl.ylocator = MultipleLocator(20)


def _add_area_decorations(ax, fig, area: str, input_dir: Path):
    """Caixas, textos e legendas especificas por area (portado do s01)."""
    if area == 'MDR':
        lon_min, lon_max = -86, -20
        lat_min, lat_max = 10, 20
        ax.plot(
            [lon_min, lon_max, lon_max, lon_min, lon_min],
            [lat_min, lat_min, lat_max, lat_max, lat_min],
            color='black', linewidth=3, linestyle='-', zorder=500,
            transform=ccrs.PlateCarree(),
        )

    if area == 'atlantico_tropical':
        leg = input_dir / 'legenda_atlantic.png'
        if leg.exists():
            fig.figimage(plt.imread(str(leg)), 125, 614, zorder=3, alpha=1)
        ax.add_patch(patches.Rectangle(
            (10, -20), -40, 20,
            linewidth=3, edgecolor='black', facecolor='none', zorder=100,
        ))
        ax.add_patch(patches.Rectangle(
            (-15, 5), -40, 20,
            linewidth=3, edgecolor='blue', facecolor='none', zorder=100,
        ))

    if area == 'iod':
        ax.add_patch(patches.Rectangle(
            (50, -10), 20, 20,
            linewidth=3, edgecolor='black', facecolor='none', zorder=100,
        ))
        ax.add_patch(patches.Rectangle(
            (90, -10), 20, 10,
            linewidth=3, edgecolor='black', facecolor='none', zorder=100,
        ))

    if area == 'enso':
        for lon, lat, txt, cor, contorno in [
            (66.25, -13.64, 'Niño 1+2', 'red',   'black'),
            (34.10,   8.45, 'Niño 3',   'blue',  'white'),
            ( 8.60,  -9.45, 'Niño 3.4', 'black', 'white'),
            (-22.50,  8.45, 'Niño 4',   'm',     'black'),
        ]:
            t = plt.text(lon, lat, txt, fontsize=14, color=cor, weight='bold', zorder=500)
            t.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground=contorno),
                path_effects.Normal(),
            ])

    if area == 'mjo':
        rotulos = [(-135.25, '1'), (-115.25, '2'), (-95.25, '3'), (-75.25, '4'),
                   (-55.25, '5'), (-35.25, '6'), (-15.25, '7'), (4.75, '8')]
        for lon, txt in rotulos:
            t = plt.text(lon, -4.64, txt, fontsize=50, color='white',
                         weight='bold', zorder=400)
            t.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground='black'),
                path_effects.Normal(),
            ])


# =============================================================================
# Main
# =============================================================================
def main():
    logger = get_logger(SCRIPT_ID)

    logger.info('=' * 80)
    logger.info(f'SCRIPT {SCRIPT_ID.upper()}: {SCRIPT_DESC}')
    logger.info('=' * 80)
    start_time = time.time()

    # -------------------------------------------------------------------
    # Template e areas
    # -------------------------------------------------------------------
    template_anom = _load_template(PATH_TEMPLATE_ANOMALIA)
    di = template_anom['data_inicial']
    df = template_anom['data_final']

    dias_full = pd.date_range(start=di, end=df, freq='D')
    if len(dias_full) == 0:
        raise ValueError(f'Periodo invalido no template: {di} a {df}')
    dias_obs = _filter_dias_by_template(dias_full, template_anom)
    if len(dias_obs) == 0:
        raise ValueError('Nenhum dia satisfaz os filtros do template ANOMALIA.')
    periodo_label = _build_periodo_label(dias_obs, di, df)

    lst_areas = list(settings.LST_AREAS_S01)
    info_plot = settings['areas_plotagem']
    input_dir = Path(settings.DIR_INPUT)
    output_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_GEOP250'
    output_files = [str(output_dir / f'geop250_{area}.png') for area in lst_areas]

    cache_params = {
        'template_anomalia': template_anom,
        'areas': lst_areas,
        'dataset': DATASET,
        'var': VAR_Z,
        'nivel': NIVEL_Z,
        'fonte': 'API Ampere reanalise',
        'script_version': '1.0_api_ampere_geop250',
    }

    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('Cache valido. Pulando execucao.')
        logger.info('=' * 80)
        return

    logger.info(f'Periodo: {periodo_label} | Areas: {len(lst_areas)}')

    # -------------------------------------------------------------------
    # 1 chamada a API: anomalia de z em 250 hPa
    # -------------------------------------------------------------------
    logger.info(f'Chamando API Ampere: anomalia({VAR_Z}, {NIVEL_Z} hPa)...')
    payload = _render_template(template_anom, VAR_Z, NIVEL_Z, nm_dataset=DATASET)
    with ReanaliseAPI(logger=logger) as api:
        ds = api.anomalia(**_payload_for_anomalia(payload))
    z_anom = _to_dataarray(ds, VAR_Z)

    # Converte de m^2/s^2 -> metros geopotenciais
    z_anom = z_anom / G_GRAVIDADE
    z_anom.attrs['units'] = 'mgp'
    z_anom.name = 'anomalia_geop250'

    # add_cyclic_point para evitar faixa branca em mapas globais
    da_for_plot = z_anom.assign_coords(lon=((z_anom.lon + 360) % 360)).sortby('lon')
    hgt, lon_cyc = add_cyclic_point(da_for_plot.values, coord=da_for_plot['lon'].values)
    lat = da_for_plot['lat'].values

    # Niveis/ticks dinamicos
    if np.all((hgt >= -50) & (hgt <= 50)):
        levels = np.arange(-50, 53, 3)
        ticks = np.arange(-50, 55, 5)
    elif np.all((hgt >= -100) & (hgt <= 100)):
        levels = np.arange(-100, 120, 20)
        ticks = np.arange(-100, 150, 50)
    else:
        levels = np.arange(-200, 220, 20)
        ticks = np.arange(-200, 250, 50)

    cmap = LinearSegmentedColormap.from_list('anom_corretto', list(settings.LST_ANOM_CORRETA))

    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # Loop de plot
    # -------------------------------------------------------------------
    for area in lst_areas:
        if area not in info_plot:
            logger.warning(f'Area "{area}" nao encontrada em settings.areas_plotagem. Pulando.')
            continue

        cfg = info_plot[area]
        is_polar = cfg.get('projection', '') == 'orthographic_south'

        if is_polar:
            proj = ccrs.Orthographic(central_longitude=-71, central_latitude=-84)
        else:
            proj = ccrs.PlateCarree(central_longitude=cfg['central_longitude_mapa'])

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(1, 1, 1, projection=proj)

        if is_polar:
            import matplotlib.path as mpath
            theta = np.linspace(0, 2 * np.pi, 100)
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * 0.5 + [0.5, 0.5])
            ax.set_boundary(circle, transform=ax.transAxes)

        # Caixas configuraveis do settings (plot_box)
        if cfg.get('plot_box', False):
            for box_cfg in cfg.get('lst_boxes', []):
                ax.add_patch(patches.Rectangle(
                    (box_cfg['x_anc'], box_cfg['y_anc']),
                    box_cfg['x_larg'], box_cfg['y_larg'],
                    linewidth=box_cfg['linewidth'],
                    edgecolor=box_cfg['edgecolor'],
                    facecolor='none', zorder=100,
                ))

        _add_area_decorations(ax, fig, area, input_dir)

        if is_polar:
            gl = ax.gridlines(draw_labels=False, linestyle='--', alpha=0.5)
        else:
            gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.0)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 20, 'color': 'black'}
        gl.ylabel_style = {'size': 20, 'color': 'black'}
        _apply_area_locators(gl, area, is_polar)

        if not is_polar:
            ax.set_xlim([cfg['lon_esq'], cfg['lon_dir']])
            ax.set_ylim([cfg['lat_inf'], cfg['lat_sup']])

        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2)
        ax.add_feature(cfeature.LAND.with_scale('50m'), linewidth=0.5, facecolor='whitesmoke')
        if area != 'china':
            ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1.2, zorder=100)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, zorder=100)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, zorder=100)
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), linewidth=0.5, facecolor='white')

        data_transform = (
            ccrs.PlateCarree() if is_polar
            else ccrs.PlateCarree(central_longitude=cfg['central_longitude_plot'])
        )

        im = ax.contourf(
            lon_cyc, lat, hgt,
            levels=levels, cmap=cmap, extend='both',
            transform=data_transform,
        )

        if is_polar:
            cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.05, ticks=ticks)
        elif area in {'enso', 'tropico', 'MDR', 'hemisferio_sul', 'psa'}:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)
            cbar = plt.colorbar(
                im, cax=cax, pad=0.02, fraction=0.02375,
                location='bottom', extend='both',
                orientation='horizontal', ticks=ticks,
            )
        else:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
            cbar = plt.colorbar(im, cax=cax, pad=0.02, fraction=0.02375, extend='both', ticks=ticks)

        if is_polar:
            cbar.set_label(label='mgp', size=10)
            cbar.ax.tick_params(labelsize=10)
        else:
            cbar.set_label(label='mgp', size=18)
            cbar.ax.tick_params(labelsize=20)

        ax.contour(
            lon_cyc, lat, hgt,
            levels=levels, colors='white', linewidths=0.6,
            transform=data_transform,
        )

        titulo = f'Anomalia de Alt. Geopotencial 250mb ({periodo_label})'
        if is_polar:
            titulo = f'Anomalia de Alt. Geopotencial 250mb\n({periodo_label})'
            ax.set_title(titulo, fontsize=14, loc='left')
        else:
            ax.set_title(titulo, fontsize=18, loc='left')

        logo_path = input_dir / 'novo_logo.png'
        if logo_path.exists():
            _add_logo_to_map(ax=ax, logo_path=logo_path, zoom=0.65,
                             xoffset=0, yoffset=0, zorder=500)

        filename_fig = output_dir / f'geop250_{area}.png'
        logger.info(f'Salvando figura {filename_fig}')
        plt.savefig(str(filename_fig), dpi=fig.dpi, bbox_inches='tight')
        plt.close('all')

    execution_time = time.time() - start_time
    save_cache_metadata(SCRIPT_ID, cache_params, output_files, execution_time)

    logger.info('=' * 80)
    logger.info(f'Script {SCRIPT_ID.upper()} concluido!')
    logger.info(f'Tempo de execucao: {execution_time:.1f}s ({execution_time / 60:.1f} min)')
    logger.info(f'Arquivos gerados: {len(output_files)} em {output_dir}')
    logger.info('=' * 80)


if __name__ == '__main__':
    main()
