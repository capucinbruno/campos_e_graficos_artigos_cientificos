"""s00 (TESTE) - Vento 100m + MSLP via API Ampere de Reanalise.

Versao de teste do s00 que substitui o download direto do CDS pela API de
reanalise da Ampere Consultoria. Os parametros de busca vem de dois templates
JSON globais em Entrada/:

    - anomalia_dados.json: alimenta o mapa de anomalia % do vento 100m e a
      linha climatologica dos graficos diarios por empreendimento.
    - media_dados.json:    alimenta os vetores medios (quiver), a MSLP do
      mapa e as barras observadas dos graficos diarios por empreendimento.

Como a API trabalha por variavel escalar, a anomalia da velocidade
(np.hypot(u, v)) e reproduzida via "caminho B": busca u/v separados para o
periodo atual e para a janela climatologica e calcula localmente

    (hypot(mean_u_a, mean_v_a) - hypot(mean_u_c, mean_v_c)) /
     hypot(mean_u_c, mean_v_c) * 100

Custo de chamadas (paralelizadas):
    - Mapas: 5 fixas (u_atual, v_atual, u_clim, v_clim, msl)
    - Series por empreendimento: 4 chamadas por dia, COMPARTILHADAS entre
      todos os empreendimentos (extracao no ponto e local).

Saida:
    - Mapas PNG em Saida/s00_VENTO_EOLICAS_SEMOP/
    - Graficos diarios em Saida/s00_VENTO_EOLICAS_SEMOP/graficos_diarios/

Criado em: 2026-05-15
"""

# Bibliotecas padrão
import json
import re
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

# Bibliotecas de terceiros
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.ndimage
import xarray as xr
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from numpy import ma

# Módulos locais
from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.common.parallel_helper import parallel_process
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.src.reanalise import ReanaliseAPI

# ---------------------------------------------------------------------------
# Identidade do script (derivada do nome do arquivo)
# ---------------------------------------------------------------------------
SCRIPT_ID = Path(__file__).stem.split('_')[0]
SCRIPT_NAME = Path(__file__).stem
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME

# ---------------------------------------------------------------------------
# Especificacao ERA5 do script (substitui placeholders dos templates JSON)
# ---------------------------------------------------------------------------
DATASET = 'reanalysis-era5-single-levels'
VAR_U = '100u'
VAR_V = '100v'
VAR_MSL = 'msl'
NIVEL: int | None = None  # single-levels nao tem pressure_level_hpa

# Templates globais
PATH_TEMPLATE_ANOMALIA = Path(settings.DIR_INPUT) / 'anomalia_dados.json'
PATH_TEMPLATE_MEDIA = Path(settings.DIR_INPUT) / 'media_dados.json'

# Limite de workers para chamadas paralelas a API
PARALLEL_API_WORKERS = 12

# Mascara a anomalia % nos pontos onde o vento atual e fraquerrimo
# (mesmo limiar do quiver). Valores < esse limiar viram NaN no mapa.
MIN_WIND_PARA_ANOMALIA = 2.0  # m/s

# Piso visual da magnitude dos vetores do quiver: vetores cuja magnitude
# real for < este valor sao desenhados como se tivessem este tamanho.
# Aumenta a visibilidade da faixa 3-5 m/s sem mexer nos > MIN_MAG_VISUAL_QUIVER.
MIN_MAG_VISUAL_QUIVER = 5.0  # m/s

# Anomalia de velocidade vertical (w) em nivel de pressao: sobreposta como
# contornos cheios pretos. Plotamos apenas as anomalias NEGATIVAS (em
# coordenada de pressao, w<0 = ar ascendendo).
DATASET_PL = 'reanalysis-era5-pressure-levels'
VAR_W = 'w'
NIVEL_W = 500  # hPa
NIVEIS_CONTORNO_W_NEG = (-0.30, -0.20, -0.10)  # Pa/s (plota a partir de -0.10)
COR_CONTORNO_W = 'red'
LW_CONTORNO_W = 1.5
# Faixa de latitude onde os contornos de w sao plotados (foco tropical).
# Pontos fora da faixa sao mascarados antes da plotagem.
LAT_MIN_CONTORNO_W = -10.0
LAT_MAX_CONTORNO_W = 10.0
# Sobre terra, so plota a partir desta latitude (inclusive) em direcao ao
# norte. Pontos sobre terra com lat < esse valor sao mascarados; oceano
# permanece intacto em toda a faixa [LAT_MIN_CONTORNO_W..LAT_MAX_CONTORNO_W].
LAT_MIN_TERRA_PERMITIDA_W = -7.0

# Chaves aceitas pelo endpoint agregacao_temporal (filtra payloads vindos do
# template de anomalia, que tem campos extras como ano_referencia_*).
_AGREGACAO_KEYS = {
    'nm_dataset',
    'nm_variavel',
    'data_inicial',
    'data_final',
    'tipo_agregacao',
    'filtro_anos',
    'filtro_meses',
    'filtro_dias',
    'filtro_horas',
    'pressure_level_hpa',
}

# Chaves aceitas pelo endpoint anomalia (inclui ano_referencia_*).
_ANOMALIA_KEYS = _AGREGACAO_KEYS | {
    'ano_referencia_inicio',
    'ano_referencia_fim',
    'filtro_anos_referencia',
}


# =============================================================================
# Helpers de template
# =============================================================================
def _load_template(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f'Template JSON nao encontrado: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _render_template(template: dict, var: str, nivel: int | None, **overrides) -> dict:
    """
    Substitui placeholders DATASET/VAR/NIVEL no template e aplica overrides.

    Se nivel for None, remove pressure_level_hpa para nao enviar a API.
    """
    payload = deepcopy(template)
    payload['nm_dataset'] = DATASET
    payload['nm_variavel'] = var

    if nivel is None:
        payload.pop('pressure_level_hpa', None)
    else:
        payload['pressure_level_hpa'] = nivel

    payload.update(overrides)
    return payload


def _payload_for_agregacao(payload: dict) -> dict:
    """Mantem apenas as chaves aceitas por agregacao_temporal."""
    return {k: v for k, v in payload.items() if k in _AGREGACAO_KEYS}


def _payload_for_anomalia(payload: dict) -> dict:
    """Mantem apenas as chaves aceitas pelo endpoint /anomalia."""
    return {k: v for k, v in payload.items() if k in _ANOMALIA_KEYS}


def _mask_land(da: xr.DataArray, lat_terra_permitida: float | None = None) -> xr.DataArray:
    """
    Mascara pontos sobre terra (continentes/ilhas), mantendo oceano.

    Se lat_terra_permitida for fornecido, pontos sobre terra com
    lat >= lat_terra_permitida sao MANTIDOS (so mascaramos terra abaixo
    desse paralelo). Util para "abrir" parte do continente nos tropicos.
    """
    from shapely.ops import unary_union
    import shapely.vectorized as shp_vec

    land = unary_union(list(cfeature.LAND.with_scale('50m').geometries()))
    lon2d, lat2d = np.meshgrid(da['lon'].values, da['lat'].values)
    in_land = shp_vec.contains(land, lon2d, lat2d)
    if lat_terra_permitida is not None:
        in_land = in_land & (lat2d < lat_terra_permitida)
    return da.where(~in_land)


# =============================================================================
# Fetchers da API Ampere
# =============================================================================
def _to_dataarray(ds: xr.Dataset, expected_var: str) -> xr.DataArray:
    """Extrai o DataArray (lat, lon) do Dataset retornado pela API."""
    if expected_var in ds.data_vars:
        da = ds[expected_var]
    else:
        da = ds[list(ds.data_vars)[0]]
    if da.dtype == object:
        # A API devolve listas Python que podem conter None em pontos sem
        # dado, virando dtype=object. Converte para float64 (None -> NaN)
        # para que ufuncs como np.hypot funcionem.
        da = xr.DataArray(
            np.asarray(da.values, dtype='float64'),
            dims=da.dims,
            coords=da.coords,
            attrs=da.attrs,
            name=da.name,
        )
    return da


def _run_api_task(task: tuple[str, dict], api: ReanaliseAPI) -> tuple[str, xr.Dataset]:
    """Worker para parallel_process: executa uma chamada agregacao_temporal."""
    tag, payload = task
    ds = api.agregacao_temporal(**_payload_for_agregacao(payload))
    return (tag, ds)


def _build_tasks_mapas(
    template_anom: dict,
    template_media: dict,
    anos_clim: list[int],
) -> list[tuple[str, dict]]:
    """
    Tasks dos mapas:
      - 2 fixas: u/v no periodo atual (alimentam anomalia + quiver)
      - 1 fixa : msl medio (isobaras)
      - 2 por ano de clim: u/v de cada ano em [ano_ref_ini..ano_ref_fim]

    O calculo correto da anomalia de velocidade (igual ao s00) precisa do
    hypot dentro de cada ano antes da media entre anos. Por isso emitimos um
    par u/v por ano, em vez de uma unica chamada agregada.
    """
    tasks: list[tuple[str, dict]] = [
        ('mapa_u_atual', _render_template(template_anom, VAR_U, NIVEL)),
        ('mapa_v_atual', _render_template(template_anom, VAR_V, NIVEL)),
        ('mapa_msl', _render_template(template_media, VAR_MSL, NIVEL)),
    ]

    for ano in anos_clim:
        tasks.append((
            f'mapa_u_clim_{ano}',
            _render_template(
                template_anom, VAR_U, NIVEL,
                data_inicial=f'{ano}-01-01', data_final=f'{ano}-12-31',
            ),
        ))
        tasks.append((
            f'mapa_v_clim_{ano}',
            _render_template(
                template_anom, VAR_V, NIVEL,
                data_inicial=f'{ano}-01-01', data_final=f'{ano}-12-31',
            ),
        ))

    return tasks


def _anos_climatologicos(template_anom: dict) -> list[int]:
    """
    Anos da janela climatologica que entram no calculo.

    - Base: range(ano_referencia_inicio, ano_referencia_fim + 1)
    - Se filtro_anos_referencia estiver setado, restringe a esses anos.
    """
    ano_ini = int(template_anom['ano_referencia_inicio'])
    ano_fim = int(template_anom['ano_referencia_fim'])
    todos = list(range(ano_ini, ano_fim + 1))
    filtro = template_anom.get('filtro_anos_referencia') or []
    if filtro:
        return [a for a in todos if a in filtro]
    return todos


def _build_periodo_label(dias_obs: pd.DatetimeIndex, di: str, df: str) -> str:
    """
    Label de titulo derivado dos dias efetivos (apos filtros).
    - 1 dia  -> 'Data: YYYY-MM-DD' (resolve transicoes de mes/ano automaticamente)
    - >1 dia -> 'Periodo: DI a DF'
    """
    if len(dias_obs) == 1:
        return f'Data: {pd.Timestamp(dias_obs[0]):%Y-%m-%d}'
    return f'Periodo: {di} a {df}'


def _filter_dias_by_template(dias: pd.DatetimeIndex, template: dict) -> pd.DatetimeIndex:
    """
    Mantem apenas os dias que satisfazem filtro_anos/meses/dias do template.
    Filtros vazios sao tratados como "tudo passa".
    """
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


def _build_tasks_series(
    template_anom: dict,
    template_media: dict,
    dias_obs: pd.DatetimeIndex,
) -> list[tuple[str, dict]]:
    """
    4 chamadas por dia (compartilhadas entre todos os empreendimentos):
      - u/v atual no dia D
      - u/v clim no dia D (anos de referencia, filtrados por mes/dia de D)
    """
    ano_ref_ini = int(template_anom['ano_referencia_inicio'])
    ano_ref_fim = int(template_anom['ano_referencia_fim'])

    tasks: list[tuple[str, dict]] = []
    for d in dias_obs:
        date_str = f'{d:%Y-%m-%d}'

        # Observado: vento medio do proprio dia. Como o periodo de 1 dia ja
        # constrange ano/mes/dia, zeramos esses filtros para nao colidir com
        # valores do template (ex.: filtro_dias=[21] inviabiliza um periodo
        # data_inicial=data_final=20). filtro_horas do template e preservado.
        tasks.append((
            f'serie_{date_str}_u_atual',
            _render_template(
                template_media, VAR_U, NIVEL,
                data_inicial=date_str, data_final=date_str,
                filtro_anos=[], filtro_meses=[], filtro_dias=[],
            ),
        ))
        tasks.append((
            f'serie_{date_str}_v_atual',
            _render_template(
                template_media, VAR_V, NIVEL,
                data_inicial=date_str, data_final=date_str,
                filtro_anos=[], filtro_meses=[], filtro_dias=[],
            ),
        ))

        # Climatologia: media nos anos de referencia, restrita ao mes/dia de D
        tasks.append((
            f'serie_{date_str}_u_clim',
            _render_template(
                template_anom,
                VAR_U,
                NIVEL,
                data_inicial=f'{ano_ref_ini}-01-01',
                data_final=f'{ano_ref_fim}-12-31',
                filtro_meses=[int(d.month)],
                filtro_dias=[int(d.day)],
            ),
        ))
        tasks.append((
            f'serie_{date_str}_v_clim',
            _render_template(
                template_anom,
                VAR_V,
                NIVEL,
                data_inicial=f'{ano_ref_ini}-01-01',
                data_final=f'{ano_ref_fim}-12-31',
                filtro_meses=[int(d.month)],
                filtro_dias=[int(d.day)],
            ),
        ))

    return tasks


def _execute_tasks_parallel(
    tasks: list[tuple[str, dict]],
    api: ReanaliseAPI,
    logger,
    description: str,
) -> dict[str, xr.Dataset]:
    """Roda todas as tasks em paralelo e devolve dict tag -> Dataset."""
    if not tasks:
        return {}

    workers = min(PARALLEL_API_WORKERS, len(tasks))
    results = parallel_process(
        items=tasks,
        process_func=_run_api_task,
        func_args=(api,),
        batch_size=len(tasks),
        max_workers=workers,
        logger=logger,
        description=description,
        log_each_item=False,
        use_threads=True,
    )

    out: dict[str, xr.Dataset] = {}
    for ok, res in results:
        if not ok:
            err_msg = res.get('error_msg', res) if isinstance(res, dict) else res
            raise RuntimeError(f'Falha em chamada a API Ampere: {err_msg}')
        tag, ds = res
        out[tag] = ds
    return out


# =============================================================================
# Helpers de plotagem
# =============================================================================
def _set_extent_from_area(ax, area_cfg: dict):
    ax.set_extent(
        [
            area_cfg['lon_min'],
            area_cfg['lon_max'],
            area_cfg['lat_min'],
            area_cfg['lat_max'],
        ],
        crs=ccrs.PlateCarree(),
    )


def _add_auto_gridlines(ax):
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.7,
        color='gray',
        alpha=0.5,
        linestyle='--',
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 22}
    gl.ylabel_style = {'size': 22}
    gl.xlocator = mticker.AutoLocator()
    gl.ylocator = mticker.AutoLocator()
    return gl


def _sanitize_filename(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r'[^\w\-]+', '_', name)
    name = re.sub(r'_+', '_', name)
    return name.strip('_')


def _plot_daily_wind_timeseries(
    df: pd.DataFrame,
    ponto_cfg: dict,
    output_path: Path,
    logger,
    periodo_label: str,
):
    fig, ax = plt.subplots(figsize=(16, 7))

    ax.bar(
        df['data'],
        df['vento_observado'],
        width=0.85,
        alpha=1,
        color='#060054',
        label='Vento diario observado (100 m)',
    )

    ax.plot(
        df['data'],
        df['vento_climatologico'],
        linewidth=2.5,
        marker='o',
        markersize=3,
        color='#00f2ff',
        label='Climatologia diaria (anos de referencia)',
    )

    ymin, ymax = ax.get_ylim()
    ax.axhspan(3, 20, color='skyblue', alpha=0.18, zorder=0)
    ax.set_ylim(ymin, ymax)

    ax.set_ylabel('Velocidade do vento (m/s)', fontsize=18)
    ax.set_xlabel('Data', fontsize=18)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    titulo = (
        f'{ponto_cfg["nome"]} | {ponto_cfg["subsistema"]} | '
        f'Potencia: {ponto_cfg["potencia_mw"]:.3f} MW\n'
        f'{periodo_label}'
    )
    ax.set_title(titulo, fontsize=22, loc='left')

    n_dias = len(df)
    if n_dias <= 15:
        locator = mdates.DayLocator(interval=1)
        formatter = mdates.DateFormatter('%d/%m')
    elif n_dias <= 45:
        locator = mdates.DayLocator(interval=2)
        formatter = mdates.DateFormatter('%d/%m')
    elif n_dias <= 120:
        locator = mdates.DayLocator(interval=5)
        formatter = mdates.DateFormatter('%d/%m')
    else:
        locator = mdates.AutoDateLocator()
        formatter = mdates.DateFormatter('%d/%m/%Y')

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.tick_params(axis='x', labelsize=18)

    ax.legend(fontsize=12, loc='upper right', bbox_to_anchor=(0.98, 0.98), frameon=True)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f'Grafico diario salvo em: {output_path}')


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
    # Templates JSON (globais em Entrada/)
    # -------------------------------------------------------------------
    template_anom = _load_template(PATH_TEMPLATE_ANOMALIA)
    template_media = _load_template(PATH_TEMPLATE_MEDIA)

    di_anom = template_anom['data_inicial']
    df_anom = template_anom['data_final']
    di_media = template_media['data_inicial']
    df_media = template_media['data_final']

    if (di_anom, df_anom) != (di_media, df_media):
        logger.warning(
            f'Periodos divergem entre templates: anomalia={di_anom} a {df_anom} | '
            f'media={di_media} a {df_media}. Mapa de anomalia usa periodo da '
            'anomalia; quiver/MSLP usa periodo da media.'
        )

    # -------------------------------------------------------------------
    # Dias efetivos da serie (apos filtros do template MEDIA).
    # Tambem definem o label do titulo (Opcao B: filtros estreitam o periodo
    # exibido, ate degenerar para uma unica data).
    # -------------------------------------------------------------------
    dias_obs_full = pd.date_range(start=di_media, end=df_media, freq='D')
    if len(dias_obs_full) == 0:
        raise ValueError(f'Periodo do template MEDIA invalido: {di_media} a {df_media}')

    dias_obs = _filter_dias_by_template(dias_obs_full, template_media)
    if len(dias_obs) == 0:
        raise ValueError(
            f'Nenhum dia do periodo {di_media}..{df_media} satisfaz os filtros do '
            f'template MEDIA (filtro_anos={template_media.get("filtro_anos")}, '
            f'filtro_meses={template_media.get("filtro_meses")}, '
            f'filtro_dias={template_media.get("filtro_dias")}).'
        )
    if len(dias_obs) < len(dias_obs_full):
        logger.info(
            f'Filtros do template MEDIA reduziram a serie diaria de '
            f'{len(dias_obs_full)} para {len(dias_obs)} dias.'
        )

    periodo_label = _build_periodo_label(dias_obs, di_media, df_media)

    # -------------------------------------------------------------------
    # Areas de plotagem e empreendimentos
    # -------------------------------------------------------------------
    plot_areas = {
        'america_do_sul_atlantico': {
            'lon_min': -80,
            'lon_max': 15,
            'lat_min': -40,
            'lat_max': 10,
            'cbar_label': 'Wind Speed Anomaly (%)',
            'title': f'Vento em 100 metros e PNMM ({periodo_label})\nReanalise ERA5',
            'logo_zoom': 1.5,
            'logo_x': 80,
            'logo_y': 165,
        },
        'nordeste_brasil': {
            'lon_min': -50,
            'lon_max': -25,
            'lat_min': -20,
            'lat_max': 5,
            'cbar_label': 'Wind Speed Anomaly (%)',
            'title': f'Vento em 100 metros e PNMM\n{periodo_label}\nReanalise ERA5',
            'logo_zoom': 1.5,
            'logo_x': 80,
            'logo_y': 165,
        },
    }

    empreendimentos = [
        {'nome': 'Conj. Caju', 'potencia_mw': 686, 'lat': -5.747012868, 'lon': -36.29595588, 'subsistema': 'NE', 'cor': '#00f7ff'},
        {'nome': 'Conj. Lagoa dos Ventos', 'potencia_mw': 717, 'lat': -8.651041667, 'lon': -41.48511905, 'subsistema': 'NE', 'cor': '#00ff59'},
        {'nome': 'Conj. Sao Roque', 'potencia_mw': 795, 'lat': -8.901654412, 'lon': -41.61948529, 'subsistema': 'NE', 'cor': '#fffb00'},
        {'nome': 'Conj. Campo Largo', 'potencia_mw': 688, 'lat': -10.45987216, 'lon': -41.48153409, 'subsistema': 'NE', 'cor': '#ffae00'},
        {'nome': 'Conj. Serra do Assurua', 'potencia_mw': 858, 'lat': -11.50455729, 'lon': -42.57942708, 'subsistema': 'NE', 'cor': '#ff0000'},
    ]

    lst_areas = list(plot_areas.keys())
    output_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_VENTO_EOLICAS_SEMOP'
    output_dir_graphs = output_dir / 'graficos_diarios'
    map_output_files = [str(output_dir / f'vento_SEMOP_{area}.png') for area in lst_areas]
    graph_output_files = [
        str(output_dir_graphs / f'vento_diario_{_sanitize_filename(emp["nome"])}.png')
        for emp in empreendimentos
    ]
    output_files = map_output_files + graph_output_files

    cache_params = {
        'template_anomalia': template_anom,
        'template_media': template_media,
        'areas': lst_areas,
        'empreendimentos': empreendimentos,
        'dataset': DATASET,
        'vars': [VAR_U, VAR_V, VAR_MSL],
        'nivel': NIVEL,
        'fonte': 'API Ampere reanalise',
        'script_version': '3.2.1_w_tropico_oceano',
    }

    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('Cache valido. Pulando execucao.')
        logger.info('=' * 80)
        return

    logger.info('Cache invalido. Executando processamento via API Ampere...')

    # -------------------------------------------------------------------
    # Monta lista de tasks (mapas + series por dia)
    # -------------------------------------------------------------------
    anos_clim = _anos_climatologicos(template_anom)
    if not anos_clim:
        raise ValueError(
            'Nenhum ano selecionado para a climatologia. Verifique '
            'ano_referencia_inicio/fim e filtro_anos_referencia.'
        )

    tasks_mapas = _build_tasks_mapas(template_anom, template_media, anos_clim)
    tasks_series = _build_tasks_series(template_anom, template_media, dias_obs)
    all_tasks = tasks_mapas + tasks_series

    logger.info(
        f'Tasks: {len(tasks_mapas)} mapas (3 fixas + {len(anos_clim)} anos x 2) + '
        f'{len(tasks_series)} series ({len(dias_obs)} dias x 4) = '
        f'{len(all_tasks)} chamadas'
    )

    # -------------------------------------------------------------------
    # Executa todas as chamadas a API Ampere em paralelo
    # -------------------------------------------------------------------
    with ReanaliseAPI(logger=logger) as api:
        ds_by_tag = _execute_tasks_parallel(
            tasks=all_tasks,
            api=api,
            logger=logger,
            description='Chamadas API Ampere (mapas + series)',
        )

        # Anomalia escalar de w (1 chamada, endpoint /anomalia). E escalar,
        # entao nao precisa do caminho B; o servidor calcula a anomalia.
        logger.info(f'Calculando anomalia de {VAR_W} em {NIVEL_W} hPa...')
        payload_w = _render_template(
            template_anom, VAR_W, NIVEL_W, nm_dataset=DATASET_PL,
        )
        ds_w = api.anomalia(**_payload_for_anomalia(payload_w))
        w_anom = _to_dataarray(ds_w, VAR_W)

    # Filtros para o contorno de w:
    #   1) so latitudes [LAT_MIN_CONTORNO_W, LAT_MAX_CONTORNO_W] (tropical)
    #   2) mascara pontos sobre terra APENAS abaixo de LAT_MIN_TERRA_PERMITIDA_W
    #      (do paralelo pra cima, continente fica visivel)
    lat_mask = (w_anom['lat'] >= LAT_MIN_CONTORNO_W) & (w_anom['lat'] <= LAT_MAX_CONTORNO_W)
    w_anom = w_anom.where(lat_mask)
    w_anom = _mask_land(w_anom, lat_terra_permitida=LAT_MIN_TERRA_PERMITIDA_W)

    # -------------------------------------------------------------------
    # Mapas: anomalia % do vento, quiver e MSLP
    #
    # Anomalia fiel ao s00: hypot dentro de cada ano (preserva variabilidade
    # direcional inter-anual) e depois media entre anos. A versao anterior
    # subestimava a climatologia ao fazer hypot da media de u/v entre anos.
    # -------------------------------------------------------------------
    u_a = _to_dataarray(ds_by_tag['mapa_u_atual'], VAR_U)
    v_a = _to_dataarray(ds_by_tag['mapa_v_atual'], VAR_V)
    msl_da = _to_dataarray(ds_by_tag['mapa_msl'], VAR_MSL)

    msl_units = str(msl_da.attrs.get('units', '')).lower()
    if ('pa' in msl_units and 'hpa' not in msl_units) or float(msl_da.max()) > 2000:
        logger.info('Convertendo MSLP de Pa para hPa.')
        msl_da = msl_da / 100.0
        msl_da.attrs['units'] = 'hPa'

    # Magnitude do periodo atual (1 par u/v)
    mag_a = np.hypot(u_a, v_a)

    # Climatologia: hypot ano-a-ano, depois media entre anos
    mags_clim = []
    for ano in anos_clim:
        u_ano = _to_dataarray(ds_by_tag[f'mapa_u_clim_{ano}'], VAR_U)
        v_ano = _to_dataarray(ds_by_tag[f'mapa_v_clim_{ano}'], VAR_V)
        mags_clim.append(np.hypot(u_ano, v_ano))
    mag_c = xr.concat(mags_clim, dim='ano').mean(dim='ano', skipna=True)

    mag_c_safe = xr.where(mag_c <= 0.1, np.nan, mag_c)
    anomalia_pct = ((mag_a - mag_c_safe) / mag_c_safe) * 100.0
    # Omite anomalia onde o vento atual e fraquerrimo (coerente com o quiver,
    # que tambem mascara mag_a < 2 m/s).
    anomalia_pct = xr.where(mag_a < MIN_WIND_PARA_ANOMALIA, np.nan, anomalia_pct)
    anomalia_pct.name = 'anomalia_percentual_vento100m'

    # Quiver: mascara vetores fracos (< 2 m/s) e aplica piso visual em
    # magnitudes < MIN_MAG_VISUAL_QUIVER (amplifica faixa 3-5 m/s sem mexer
    # nos vetores maiores).
    mag_a_vals = mag_a.values
    safe = np.where(mag_a_vals > 0, mag_a_vals, 1.0)
    mag_plot = np.maximum(mag_a_vals, MIN_MAG_VISUAL_QUIVER)
    boost = mag_plot / safe
    zonal_vec = ma.masked_where(mag_a_vals < 2, u_a.values * boost)
    meridional_vec = ma.masked_where(mag_a_vals < 2, v_a.values * boost)
    mslp_smooth = scipy.ndimage.gaussian_filter(msl_da.values, sigma=1)

    # A API pode devolver grades diferentes por variavel (ex.: vento recortado,
    # MSLP global, w em outra grade). Um meshgrid por grade.
    lon2d_w, lat2d_w = np.meshgrid(u_a['lon'].values, u_a['lat'].values)
    lon2d_p, lat2d_p = np.meshgrid(msl_da['lon'].values, msl_da['lat'].values)
    lon2d_wa, lat2d_wa = np.meshgrid(w_anom['lon'].values, w_anom['lat'].values)

    colors = [
        '#c8334d', '#db5054', '#ea6e5d', '#f68b67',
        '#fea973', '#ffc887', '#ffe5a5', '#ffffff',
        '#ffffff', '#d2f2df', '#b7dfd9', '#a1cbd3',
        '#8db8cc', '#7ba4c5', '#6b91be', '#5b7db7',
    ]
    cmap = LinearSegmentedColormap.from_list('wind_colormap', colors)
    shaded_levels = np.arange(-100, 105, 5)

    output_dir.mkdir(exist_ok=True, parents=True)

    for area_name, area_cfg in plot_areas.items():
        logger.info(f'Gerando figura para a area: {area_name}')

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1.9)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.9)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.9)
        ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='white')
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='white')

        _set_extent_from_area(ax, area_cfg)

        im = ax.contourf(
            lon2d_w, lat2d_w, anomalia_pct.values,
            cmap=cmap, extend='both', levels=shaded_levels,
            transform=ccrs.PlateCarree(),
        )

        if area_name == 'nordeste_brasil':
            skip = slice(None, None, 10)
            quiver_scale = 25
            quiver_width = 0.0050
            quiver_headwidth = 3.2
            quiver_headlength = 5.2
        else:
            skip = slice(None, None, 9)
            quiver_scale = 40
            quiver_width = 0.002
            quiver_headwidth = 3
            quiver_headlength = 5

        ax.quiver(
            lon2d_w[skip, skip], lat2d_w[skip, skip],
            zonal_vec[skip, skip], meridional_vec[skip, skip],
            scale_units='inches', color='k',
            headwidth=quiver_headwidth, scale=quiver_scale,
            headlength=quiver_headlength, width=quiver_width,
            transform=ccrs.PlateCarree(),
        )

        if area_name == 'nordeste_brasil':
            handles = []
            for emp in empreendimentos:
                h = ax.scatter(
                    emp['lon'], emp['lat'],
                    s=160, c=emp['cor'], edgecolors='black', linewidths=2.0,
                    marker='o', transform=ccrs.PlateCarree(), zorder=100,
                    label=emp['nome'],
                )
                handles.append(h)

            legenda = ax.legend(
                handles=handles, loc='upper left', bbox_to_anchor=(0.015, 0.985),
                fontsize=12, frameon=True, fancybox=True, framealpha=0.90,
                borderpad=0.4, labelspacing=0.35, handlelength=0.8, handletextpad=0.4,
            )
            legenda.get_frame().set_facecolor('white')
            legenda.get_frame().set_edgecolor('black')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)

        cbar = plt.colorbar(
            im, cax=cax, fraction=0.02, orientation='horizontal',
            pad=0.05, extend='both', boundaries=shaded_levels,
            ticks=(-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100),
        )
        cbar.set_label(label=area_cfg['cbar_label'], labelpad=18, size=18)
        cbar.ax.tick_params(labelsize=18)

        im2 = ax.contour(
            lon2d_p, lat2d_p, mslp_smooth,
            levels=range(980, 1045, 3), colors='black', linewidths=1.0,
            transform=ccrs.PlateCarree(),
        )
        clabels = ax.clabel(im2, inline=True, fmt='%1.0f', fontsize=12, colors='black')
        for txt in clabels:
            txt.set_bbox(dict(facecolor='white', edgecolor='white', pad=0.8))

        # Contornos cheios da anomalia NEGATIVA de w (movimento vertical
        # ascendente). Sem clabel (sem rotulos/labels).
        ax.contour(
            lon2d_wa, lat2d_wa, w_anom.values,
            levels=NIVEIS_CONTORNO_W_NEG,
            colors=COR_CONTORNO_W,
            linewidths=LW_CONTORNO_W,
            linestyles='-',
            transform=ccrs.PlateCarree(),
        )

        _add_auto_gridlines(ax)
        ax.set_title(area_cfg['title'], fontsize=22, loc='left')

        filename_fig = output_dir / f'vento_SEMOP_{area_name}.png'
        logger.info(f'Salvando a figura {filename_fig}')

        logo_path = Path(settings.DIR_INPUT) / 'logo_ampere.png'
        if logo_path.exists():
            try:
                img_logotipo = plt.imread(logo_path)
                fz = area_cfg['logo_zoom']
                if img_logotipo.ndim == 3:
                    img_logotipo = scipy.ndimage.zoom(img_logotipo, (fz, fz, 1), order=1)
                else:
                    img_logotipo = scipy.ndimage.zoom(img_logotipo, (fz, fz), order=1)
                fig.figimage(img_logotipo, area_cfg['logo_x'], area_cfg['logo_y'], zorder=3, alpha=1)
            except Exception as e:
                logger.warning(f'Nao foi possivel inserir logo da Ampere: {e}')
        else:
            logger.warning(f'Logo da Ampere nao encontrada em {logo_path}')

        plt.savefig(filename_fig, dpi=fig.dpi, bbox_inches='tight')
        plt.close(fig)

    # -------------------------------------------------------------------
    # Graficos diarios por empreendimento (extracao no ponto)
    # -------------------------------------------------------------------
    logger.info('Gerando graficos diarios dos empreendimentos...')
    output_dir_graphs.mkdir(exist_ok=True, parents=True)

    for emp in empreendimentos:
        obs_vals: list[float] = []
        clim_vals: list[float] = []

        for d in dias_obs:
            ds = f'{d:%Y-%m-%d}'

            u_a_pt = float(
                _to_dataarray(ds_by_tag[f'serie_{ds}_u_atual'], VAR_U)
                .sel(lat=emp['lat'], lon=emp['lon'], method='nearest')
                .values
            )
            v_a_pt = float(
                _to_dataarray(ds_by_tag[f'serie_{ds}_v_atual'], VAR_V)
                .sel(lat=emp['lat'], lon=emp['lon'], method='nearest')
                .values
            )
            obs_vals.append(float(np.hypot(u_a_pt, v_a_pt)))

            u_c_pt = float(
                _to_dataarray(ds_by_tag[f'serie_{ds}_u_clim'], VAR_U)
                .sel(lat=emp['lat'], lon=emp['lon'], method='nearest')
                .values
            )
            v_c_pt = float(
                _to_dataarray(ds_by_tag[f'serie_{ds}_v_clim'], VAR_V)
                .sel(lat=emp['lat'], lon=emp['lon'], method='nearest')
                .values
            )
            clim_vals.append(float(np.hypot(u_c_pt, v_c_pt)))

        df_point = pd.DataFrame({
            'data': dias_obs,
            'vento_observado': obs_vals,
            'vento_climatologico': clim_vals,
        })

        filename_graph = output_dir_graphs / f'vento_diario_{_sanitize_filename(emp["nome"])}.png'
        _plot_daily_wind_timeseries(
            df=df_point,
            ponto_cfg=emp,
            output_path=filename_graph,
            logger=logger,
            periodo_label=periodo_label,
        )

    execution_time = time.time() - start_time
    save_cache_metadata(SCRIPT_ID, cache_params, output_files, execution_time)

    logger.info('=' * 80)
    logger.info(f'Script {SCRIPT_ID.upper()} concluido!')
    logger.info(f'Tempo de execucao: {execution_time:.1f}s ({execution_time / 60:.1f} min)')
    logger.info(f'Arquivos gerados: {len(output_files)}')
    logger.info('=' * 80)


if __name__ == '__main__':
    main()
