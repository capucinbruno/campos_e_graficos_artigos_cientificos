"""s02 (TESTE) - Anomalia de CHI200 (velocity potential) via API Ampere.

Versao de teste do s02 que substitui o download direto do CDS pela API de
reanalise da Ampere Consultoria. O template global anomalia_dados.json em
Entrada/ define o periodo, janela climatologica e filtros.

Como a decomposicao de Helmholtz (vento -> velocity potential + vento
divergente) e LINEAR no vento, a anomalia de chi pode ser obtida aplicando o
Helmholtz diretamente sobre as anomalias de u e v. Logo bastam DUAS chamadas
ao endpoint /anomalia (uma para u, outra para v) -- o servidor ja devolve a
anomalia escalar de cada componente. A matematica (divergencia, Poisson na
esfera e vento divergente) e reaproveitada de app.src.uteis.plot_chi200.

Atencao ao nivel: o s02 original calcula chi a partir do vento de 250 hPa
(downloader uv250 + climatologia UWND250/VWND250), apesar do nome "chi200".
Mantemos 250 hPa aqui para reproduzir fielmente a saida do s02. Ajuste
NIVEL_UV se desejar 200 hPa.

Saida:
    - Mapas PNG em Saida/teste_dados_anom_chi200_CHI200/ (um por area)

Criado em: 2026-06-02
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
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import FixedLocator, MultipleLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from PIL import Image

# Módulos locais
from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.src.reanalise import ReanaliseAPI
from app.src.uteis.plot_chi200 import _helmholtz_chi

# ---------------------------------------------------------------------------
# Identidade do script (nome completo para nao colidir com outros testes)
# ---------------------------------------------------------------------------
SCRIPT_ID = Path(__file__).stem  # 'teste_dados_anom_chi200'
SCRIPT_NAME = Path(__file__).stem
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME

# ---------------------------------------------------------------------------
# Especificacao ERA5 do script
# ---------------------------------------------------------------------------
DATASET = 'reanalysis-era5-pressure-levels'
VAR_U = 'u'
VAR_V = 'v'
# O s02 original usa vento de 250 hPa para o CHI200. Mantemos para fidelidade.
NIVEL_UV = 250  # hPa

# Escala final do velocity potential: 10^5 m^2 s^-1.
CHI_SCALE = 1e5

# Templates globais (mesmo arquivo usado pelos demais testes).
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

# ---------------------------------------------------------------------------
# Plotagem (paleta/levels/quiver portados do s02)
# ---------------------------------------------------------------------------
LEVELS = np.arange(-60, 65, 5)

# Paleta fixa verde -> claro -> bege -> marrom
CHI200_COLORS = [
    '#005a45', '#0f7a6c', '#2e9b96', '#62bdb7',
    '#9dd8d2', '#dff3f1', '#f7f4eb', '#e7d9a9',
    '#d6b566', '#bd8a35', '#9a6313', '#6f4300',
]

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

# Ajuste de densidade (step) e demais parametros de quiver por area.
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
}

# Areas onde os vetores das bordas sao mascarados (evita setas distorcidas).
_AREAS_COM_CORTE_BORDA = {'globo', 'psa', 'hemisferio_sul'}


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


def _get_area_list() -> list[str]:
    """Prioriza LST_AREAS_S02; cai para CHI200 e depois S01 (igual ao s02)."""
    for attr in ('LST_AREAS_S02', 'LST_AREAS_CHI200', 'LST_AREAS_S01'):
        if hasattr(settings, attr):
            return list(getattr(settings, attr))
    raise AttributeError(
        'Nenhuma lista de areas encontrada em settings '
        '(LST_AREAS_S02, LST_AREAS_CHI200 ou LST_AREAS_S01).'
    )


# =============================================================================
# Helpers de processamento (CHI / vento divergente)
# =============================================================================
def _to_plot_grid(da: xr.DataArray) -> xr.DataArray:
    """Longitude para 0..360 e ordenada (igual ao teste01) antes do cyclic point."""
    return da.assign_coords(lon=((da['lon'] + 360) % 360)).sortby('lon')


def _compute_chi_fields(
    u_anom: xr.DataArray,
    v_anom: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Aplica Helmholtz na anomalia de vento e devolve chi (escalado), uchi, vchi.

    O solver de Poisson na esfera espera lat decrescente (N->S) e lon ordenada
    e uniforme (periodica). Reproduz a preparacao do plot_chi200.main().
    """
    u_anom = u_anom.sortby('lon').sortby('lat', ascending=False)
    v_anom = v_anom.sortby('lon').sortby('lat', ascending=False)

    lat = u_anom['lat'].values
    lon = u_anom['lon'].values

    chi, u_div, v_div = _helmholtz_chi(u_anom.values, v_anom.values, lat, lon)
    chi_scaled = chi / CHI_SCALE

    coords = {'lat': lat, 'lon': lon}
    chi_da = xr.DataArray(
        chi_scaled, dims=['lat', 'lon'], coords=coords, name='chi_anom_mean_scaled',
        attrs={'long_name': 'velocity potential anomaly (scaled)', 'units': '1e5 m2 s-1'},
    )
    uchi_da = xr.DataArray(
        u_div, dims=['lat', 'lon'], coords=coords, name='uchi_anom_mean',
        attrs={'long_name': 'divergent u-wind anomaly', 'units': 'm s-1'},
    )
    vchi_da = xr.DataArray(
        v_div, dims=['lat', 'lon'], coords=coords, name='vchi_anom_mean',
        attrs={'long_name': 'divergent v-wind anomaly', 'units': 'm s-1'},
    )
    return chi_da, uchi_da, vchi_da


def _build_levels_ticks_norm():
    """Levels, ticks, cmap discreto e norm a partir de LEVELS/CHI200_COLORS."""
    n_bins = len(LEVELS) - 1
    cmap = LinearSegmentedColormap.from_list('chi200_green_brown', CHI200_COLORS, N=n_bins)
    norm = BoundaryNorm(LEVELS, ncolors=n_bins, clip=False)
    ticks = np.arange(-60, 65, 10)
    return LEVELS, ticks, cmap, norm


def _prepare_quiver_masked(lon, lat, u, v, *, step, min_mag, lat_bounds=None, margin=3.0):
    """Amostra, filtra magnitude fraca e mascara vetores fora dos limites da area."""
    lon_q = lon[::step]
    lat_q = lat[::step]
    u_q = u[::step, ::step]
    v_q = v[::step, ::step]

    mag_q = np.sqrt(u_q ** 2 + v_q ** 2)
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
    """QUIVER_POR_AREA (script) > settings.CHI200_QUIVER_* > QUIVER_DEFAULTS."""
    cfg = dict(QUIVER_DEFAULTS)

    global_overrides = getattr(settings, 'CHI200_QUIVER_DEFAULTS', None)
    if isinstance(global_overrides, dict):
        cfg.update(global_overrides)

    per_area_settings = getattr(settings, 'CHI200_QUIVER_POR_AREA', None)
    if isinstance(per_area_settings, dict) and area in per_area_settings:
        cfg.update(per_area_settings[area])

    if area in QUIVER_POR_AREA:
        cfg.update(QUIVER_POR_AREA[area])

    return cfg


# =============================================================================
# Helpers de plotagem (portados do s02)
# =============================================================================
def _add_logo_to_map(ax, logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=500):
    logo = Image.open(logo_path).convert('RGBA')
    bbox = logo.getbbox()
    if bbox is not None:
        logo = logo.crop(bbox)
    imagebox = OffsetImage(np.array(logo), zoom=zoom)
    ab = AnnotationBbox(
        imagebox, (0, 0),
        xycoords=ax.transAxes, xybox=(xoffset, yoffset),
        boxcoords='offset points', box_alignment=(0, 0),
        frameon=False, pad=0, zorder=zorder, clip_on=False,
    )
    ax.add_artist(ab)


def _configure_gridlines(gl, area: str):
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 20, 'color': 'black'}
    gl.ylabel_style = {'size': 20, 'color': 'black'}

    if area in {'hemisferio_sul', 'psa', 'globo', 'mjo'}:
        gl.xlocator = MultipleLocator(40); gl.ylocator = MultipleLocator(20)
    elif area == 'estados_unidos':
        gl.xlocator = MultipleLocator(10); gl.ylocator = MultipleLocator(10)
    elif area == 'estados_unidos_zoom':
        gl.xlocator = MultipleLocator(10); gl.ylocator = MultipleLocator(5)
    elif area in {'argentina', 'costa_brasil', 'tsa', 'tna'}:
        gl.xlocator = MultipleLocator(5 if area in {'argentina', 'costa_brasil'} else 10)
        gl.ylocator = MultipleLocator(5)
    elif area == 'brasil':
        gl.xlocator = MultipleLocator(10); gl.ylocator = MultipleLocator(5)
    elif area in {'america_sul', 'africa'}:
        gl.xlocator = MultipleLocator(20); gl.ylocator = MultipleLocator(20)
    elif area in {'zona_zcit_atlantico', 'atlantico_tropical', 'pdo', 'iod', 'sad', 'amo', 'africa_monsoon'}:
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


def _add_area_decorations(ax, fig, area: str, input_dir: Path):
    """Caixas, textos e legendas especificas por area (portado do s02)."""
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
            ('Niño 3', 34.10, 8.45, 'blue'),
            ('Niño 3.4', 8.60, -9.45, 'black'),
            ('Niño 4', -22.50, 8.45, 'm'),
        ]:
            fg = 'black' if cor in {'red', 'm'} else 'white'
            t = plt.text(x, y, txt, fontsize=14, color=cor, weight='bold', zorder=500)
            t.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground=fg),
                path_effects.Normal(),
            ])

    if area == 'mjo':
        rotulos = [(-135.25, '1'), (-115.25, '2'), (-95.25, '3'), (-75.25, '4'),
                   (-55.25, '5'), (-35.25, '6'), (-15.25, '7'), (4.75, '8')]
        for x, txt in rotulos:
            t = plt.text(x, -4.64, txt, fontsize=50, color='white', weight='bold', zorder=400)
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

    lst_areas = _get_area_list()
    info_plot = settings['areas_plotagem']
    input_dir = Path(settings.DIR_INPUT)
    output_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_CHI200'
    output_files = [str(output_dir / f'chi200_{area}.png') for area in lst_areas]

    cache_params = {
        'template_anomalia': template_anom,
        'areas': lst_areas,
        'dataset': DATASET,
        'vars': [VAR_U, VAR_V],
        'nivel': NIVEL_UV,
        'palette': CHI200_COLORS,
        'quiver_defaults': QUIVER_DEFAULTS,
        'quiver_por_area': getattr(settings, 'CHI200_QUIVER_POR_AREA', {}),
        'fonte': 'API Ampere reanalise',
        'script_version': '1.0_api_ampere_chi200',
    }

    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('Cache valido. Pulando execucao.')
        logger.info('=' * 80)
        return

    logger.info(f'Periodo: {periodo_label} | Areas: {len(lst_areas)} | Nivel: {NIVEL_UV} hPa')

    # -------------------------------------------------------------------
    # 2 chamadas a API: anomalia de u e de v em NIVEL_UV.
    # Helmholtz e linear, entao chi(anomalia) = Helmholtz(u_anom, v_anom).
    # -------------------------------------------------------------------
    logger.info(f'Chamando API Ampere: anomalia(u/v, {NIVEL_UV} hPa)...')
    payload_u = _render_template(template_anom, VAR_U, NIVEL_UV, nm_dataset=DATASET)
    payload_v = _render_template(template_anom, VAR_V, NIVEL_UV, nm_dataset=DATASET)
    with ReanaliseAPI(logger=logger) as api:
        ds_u = api.anomalia(**_payload_for_anomalia(payload_u))
        ds_v = api.anomalia(**_payload_for_anomalia(payload_v))
    u_anom = _to_dataarray(ds_u, VAR_U)
    v_anom = _to_dataarray(ds_v, VAR_V)

    # -------------------------------------------------------------------
    # Velocity potential (chi) e vento divergente via Helmholtz
    # -------------------------------------------------------------------
    logger.info('Calculando velocity potential (chi) e vento divergente...')
    da_chi, da_uchi, da_vchi = _compute_chi_fields(u_anom, v_anom)

    # Grade de plotagem: lon 0..360 ordenada + cyclic point
    da_chi_p = _to_plot_grid(da_chi)
    da_uchi_p = _to_plot_grid(da_uchi)
    da_vchi_p = _to_plot_grid(da_vchi)

    lat = da_chi_p['lat'].values
    chi_cyc, lon_cyc = add_cyclic_point(da_chi_p.values, coord=da_chi_p['lon'].values)
    uchi_cyc, lon_cyc_uv = add_cyclic_point(da_uchi_p.values, coord=da_uchi_p['lon'].values)
    vchi_cyc, _ = add_cyclic_point(da_vchi_p.values, coord=da_vchi_p['lon'].values)

    levels, ticks, cmap, norm = _build_levels_ticks_norm()

    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # Loop de plot
    # -------------------------------------------------------------------
    for area in lst_areas:
        if area not in info_plot:
            logger.warning(f'Area "{area}" nao encontrada em settings.areas_plotagem. Pulando.')
            continue

        logger.info(f'Gerando mapa CHI200 para area: {area}')
        cfg = info_plot[area]

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(
            1, 1, 1,
            projection=ccrs.PlateCarree(central_longitude=cfg['central_longitude_mapa']),
        )

        # Caixas configuraveis do settings (plot_box)
        if cfg.get('plot_box', False):
            for box in cfg.get('lst_boxes', []):
                ax.add_patch(patches.Rectangle(
                    (box['x_anc'], box['y_anc']), box['x_larg'], box['y_larg'],
                    linewidth=box['linewidth'], edgecolor=box['edgecolor'],
                    facecolor='none', zorder=100,
                ))

        _add_area_decorations(ax, fig, area, input_dir)

        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.0)
        _configure_gridlines(gl, area)

        ax.set_xlim([cfg['lon_esq'], cfg['lon_dir']])
        ax.set_ylim([cfg['lat_inf'], cfg['lat_sup']])

        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, edgecolor='black')
        ax.add_feature(cfeature.LAND.with_scale('50m'), linewidth=0.5, facecolor='whitesmoke')
        if area != 'china':
            ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1.2, edgecolor='black', zorder=100)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, edgecolor='black', zorder=100)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, edgecolor='black', zorder=100)
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), linewidth=0.5, facecolor='white')

        data_transform = ccrs.PlateCarree(central_longitude=cfg['central_longitude_plot'])

        cf = ax.contourf(
            lon_cyc, lat, chi_cyc,
            levels=levels, cmap=cmap, norm=norm, extend='both',
            transform=data_transform, zorder=2,
        )
        ax.contour(
            lon_cyc, lat, chi_cyc,
            levels=levels, colors='white', linewidths=0.6, alpha=0.55,
            transform=data_transform, zorder=3,
        )

        # Vetores do vento divergente/irrotacional
        qcfg = _get_quiver_config(area)
        lon_q, lat_q, u_q_mask, v_q_mask = _prepare_quiver_masked(
            lon=lon_cyc_uv, lat=lat, u=uchi_cyc, v=vchi_cyc,
            step=int(qcfg['step']), min_mag=float(qcfg['min_mag']),
            lat_bounds=(cfg['lat_inf'], cfg['lat_sup'])
            if area in _AREAS_COM_CORTE_BORDA else None,
        )
        ax.quiver(
            lon_q, lat_q, u_q_mask, v_q_mask,
            transform=ccrs.PlateCarree(), color=qcfg['color'], pivot='mid',
            scale=float(qcfg['scale']), width=float(qcfg['width']),
            headwidth=float(qcfg['headwidth']), headlength=float(qcfg['headlength']),
            headaxislength=float(qcfg['headaxislength']), zorder=5,
        )

        # Colorbar
        if area in {'enso', 'tropico', 'MDR', 'hemisferio_sul', 'psa'}:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)
            cbar = plt.colorbar(
                cf, cax=cax, pad=0.02, fraction=0.02375, location='bottom',
                extend='both', orientation='horizontal', ticks=ticks,
                boundaries=levels, spacing='proportional',
            )
        else:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
            cbar = plt.colorbar(
                cf, cax=cax, pad=0.02, fraction=0.02375, extend='both',
                ticks=ticks, boundaries=levels, spacing='proportional',
            )
        cbar.set_label(label=r'10$^5$ m$^2$ s$^{-1}$', size=18)
        cbar.ax.tick_params(labelsize=20)

        titulo = f'Anomalia de CHI200 ({periodo_label})'
        ax.set_title(titulo, fontsize=18, loc='left')

        logo_path = input_dir / 'novo_logo.png'
        if logo_path.exists():
            _add_logo_to_map(ax=ax, logo_path=logo_path, zoom=0.65,
                             xoffset=0, yoffset=0, zorder=500)

        filename_fig = output_dir / f'chi200_{area}.png'
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
