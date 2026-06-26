# -*- coding: utf-8 -*-
"""Motor generico de animacao de globo 3D (voo da camera + evolucao temporal).

Renderiza uma variavel meteorologica sobre um globo flutuante (projecao
NearsidePerspective/Orthographic), combinando DOIS movimentos no mesmo video:

  1. Voo da camera ao redor do globo -> o ponto central (lon/lat) viaja de uma
     posicao INICIAL para uma posicao FINAL ao longo do clipe (com voltas extras
     opcionais). Ex.: comecar no meio do Pacifico e terminar na costa do Brasil.
  2. Evolucao temporal da variavel    -> o campo diario avanca (com interpolacao
     entre dias para suavizar).

O motor NAO sabe nada sobre uma variavel especifica. Cada variavel e descrita
por uma "ficha" no registro ``VARIAVEIS`` (titulo, unidade, paleta e builders
que devolvem a serie diaria de anomalia como ``xr.DataArray`` (time, lat, lon)).

Cada ficha tem builders para os DOIS modos de dados:
  - 'reanalise' -> anomalia observada (ERA5/GDAS) por DATA_INICIAL/DATA_FINAL
  - 'forecast'  -> anomalia prevista por modelo (GFS/GEFS/ECMWF/AIFS), um MP4
                   por modelo habilitado (flags RUN_GFS/RUN_GEFS/...)

Para adicionar uma variavel nova: escreva os builders e registre uma ficha.
O motor (voo + tempo + encode MP4) permanece inalterado.

Settings (defaults entre parenteses):
  VARIAVEL_GLOBO_3D            chave do registro a animar ('z250_anom')
  GLOBO_3D_MODO               'reanalise' | 'forecast'  ('reanalise')
  DATA_INICIAL / DATA_FINAL    periodo (modo reanalise)
  RODADA / NUM_RODADA / FORECAST_INIT / GEFS_FORECAST_LEAD_DAYS  (modo forecast)
  --- voo da camera ---
  GLOBO_3D_LON_INICIAL         longitude central do 1o frame (-150 = meio Pacifico)
  GLOBO_3D_LAT_INICIAL         latitude central do 1o frame  (0)
  GLOBO_3D_LON_FINAL           longitude central do ultimo frame (-45 = costa BR)
  GLOBO_3D_LAT_FINAL           latitude central do ultimo frame  (-15)
  GLOBO_3D_VOLTAS_EXTRA        voltas completas extras antes de assentar (0)
  --- render ---
  GLOBO_3D_FRAMES_POR_DIA      frames interpolados por dia (4)
  GLOBO_3D_FPS                 quadros por segundo do MP4 (20)
  GLOBO_3D_COARSEN             subamostragem da grade (4 = 0.25°->1°)
  GLOBO_3D_PROJECTION          'nearside' (globo flutuante) ou 'orthographic'

Criado em: 2026-06-25
"""

from __future__ import annotations

# Bibliotecas padrao
import os
import time as _time
from datetime import datetime, timedelta
from multiprocessing import get_context
from pathlib import Path

# Bibliotecas de terceiros
import matplotlib

matplotlib.use('Agg')  # backend sem display: necessario para grab de buffer

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.util import add_cyclic_point
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# Modulos locais
from app.shared.logger import get_logger
from app.src.uteis.forecast_daily import (
    daily_scalar_on_grid as _daily_scalar_on_grid,
    lagged_ensemble_mean as _lagged_ensemble_mean,
    resolve_forecast_lead_init as _resolve_forecast_lead_init,
)
from app.shared.settings_factory import settings
from app.src.uteis.clim_diaria_uv200_ltm import clim_hgt250_daily
from app.src.uteis.plot_geop250 import DEFAULT_SYNOPTIC_HOURS, _get_data_sources

logger = get_logger('s38')

# ---------------------------------------------------------------------------
# Fontes estilo WaPo (clones OFL embutidos em Entrada/fonts):
#   titulo serifado Didone (~Postoni)  -> Playfair Display
#   rotulos/sans (~Franklin Gothic)    -> Libre Franklin
# Fallback para serif/sans-serif genericas se os arquivos nao existirem.
# ---------------------------------------------------------------------------
from matplotlib import font_manager as _fm  # noqa: E402

FONT_SERIF = 'serif'
FONT_SANS = 'sans-serif'


def _register_fonts() -> None:
    global FONT_SERIF, FONT_SANS
    fdir = Path(settings.DIR_INPUT) / 'fonts'
    if not fdir.is_dir():
        return
    nomes = set()
    for ttf in fdir.glob('*.ttf'):
        try:
            _fm.fontManager.addfont(str(ttf))
            nomes.add(_fm.FontProperties(fname=str(ttf)).get_name())
        except Exception:
            pass
    if 'Playfair Display' in nomes:
        FONT_SERIF = 'Playfair Display'
    if 'Libre Franklin' in nomes:
        FONT_SANS = 'Libre Franklin'
    logger.info('Fontes s38: titulo={} | rotulos={}', FONT_SERIF, FONT_SANS)


_register_fonts()


def _resolve_family(nome: str, fallback: str) -> str:
    """Retorna `nome` se a familia de fonte existir (sistema ou Entrada/fonts);
    senao retorna `fallback` (com aviso). Usado p/ a fonte da legenda (ex.: Aptos)."""
    nome = str(nome or '').strip()
    if not nome:
        return fallback
    try:
        _fm.findfont(_fm.FontProperties(family=nome), fallback_to_default=False)
        return nome
    except Exception:
        logger.warning(
            "Fonte '{}' nao encontrada — usando '{}'. Coloque o .ttf em Entrada/fonts/ p/ usa-la.",
            nome, fallback,
        )
        return fallback

# Nomes possiveis da variavel de altura geopotencial nos arquivos (ERA5/GDAS/modelos)
HGT_VARS = ('hgt', 'z', 'gh', 'geopotential')

def _target_grid() -> tuple[np.ndarray, np.ndarray]:
    """Grade-alvo (fina) das series. A climatologia NCEP 2.5° e interpolada
    PARA CIMA nesta grade, preservando a textura fina do dado (ERA5/modelos).
    Resolucao via GLOBO_3D_GRID_DEG (graus); default 0.5° (~720x360)."""
    d = float(getattr(settings, 'GLOBO_3D_GRID_DEG', 0.5))
    lat = np.arange(-90.0, 90.0 + d / 2.0, d)
    lon = np.arange(0.0, 360.0, d)
    return lat, lon

# ---------------------------------------------------------------------------
# Workaround cartopy 0.25 + matplotlib 3.10 (mesmo patch do s02): a reprojecao
# em globo gera GeometryCollection nao-subscritavel em geometrias fragmentadas.
# ---------------------------------------------------------------------------
import matplotlib.path as _mpath  # noqa: E402
from cartopy.mpl.geoaxes import InterProjectionTransform as _IPT  # noqa: E402

_orig_transform_path = _IPT.transform_path_non_affine


def _safe_transform_path(self, path):
    try:
        return _orig_transform_path(self, path)
    except TypeError:
        return _mpath.Path(np.empty((0, 2)))


_IPT.transform_path_non_affine = _safe_transform_path


# ---------------------------------------------------------------------------
# Utilitarios
# ---------------------------------------------------------------------------
def _to_datetime(val) -> datetime:
    if isinstance(val, datetime):
        return val
    if hasattr(val, 'year') and not isinstance(val, str):
        return datetime(val.year, val.month, val.day)
    return datetime.strptime(str(val), '%Y-%m-%d')


# Modelos de forecast suportados para z250 (tem downloader de hgt250 dedicado).
_FCST_FLAGS = {'gfs': 'RUN_GFS', 'gefs': 'RUN_GEFS', 'ecmwf': 'RUN_ECMWF', 'aifs': 'RUN_AIFS'}


def _enabled_forecast_models() -> list[str]:
    """Modelos habilitados via flags (default: so GFS, como no s34)."""
    return [m for m, flag in _FCST_FLAGS.items() if bool(settings.get(flag, m == 'gfs'))]


def _hgt250_fcst_downloader(model: str):
    """Resolve o downloader de Z250 (forecast) do modelo (import tardio)."""
    if model == 'gfs':
        from app.src.uteis.downloaders_gfs_hgt250 import ensure_gfs_hgt250_fcst_for_period as fn
    elif model == 'gefs':
        from app.src.uteis.downloaders_gefs_hgt250 import ensure_gefs_hgt250_fcst_for_period as fn
    elif model == 'ecmwf':
        from app.src.uteis.downloaders_ecmwf_hgt250 import ensure_ecmwf_hgt250_fcst_for_period as fn
    elif model == 'aifs':
        from app.src.uteis.downloaders_aifs_hgt250 import ensure_aifs_hgt250_fcst_for_period as fn
    else:
        raise ValueError(f'Modelo de forecast sem suporte a Z250: {model}')
    return fn


# ===========================================================================
# BUILDERS Z250 — devolvem xr.DataArray(time, lat, lon) de ANOMALIA diaria
# ===========================================================================
def _z250_anom_from_daily(daily: xr.DataArray) -> xr.DataArray:
    """anomalia = campo diario - climatologia diaria LTM 1991-2020, alinhada dia-a-dia.

    `daily` ja vem na grade NCEP 2.5° (_NCEP_LAT/_NCEP_LON). A climatologia
    (clim_hgt250_daily) decodifica o eixo cftime e devolve o (mes, dia) de cada data.
    """
    clim_arr, clat, clon = clim_hgt250_daily(daily['time'].values)
    clim = xr.DataArray(
        clim_arr, dims=['time', 'lat', 'lon'],
        coords={'time': daily['time'].values, 'lat': clat, 'lon': clon},
    ).sortby('lat')
    # Ponto ciclico em longitude: a climatologia NCEP vai so ate 357.5°; sem o
    # ponto em 360° a interpolacao p/ a grade fina gera NaN em 357.5..360°, criando
    # uma faixa preta no meridiano de Greenwich (0°).
    cyc_vals, cyc_lon = add_cyclic_point(clim.values, coord=clim['lon'].values)
    clim = xr.DataArray(
        cyc_vals, dims=['time', 'lat', 'lon'],
        coords={'time': clim['time'].values, 'lat': clim['lat'].values, 'lon': cyc_lon},
    )
    clim = clim.interp(lat=daily['lat'], lon=daily['lon'], method='linear')

    anom = (daily - clim).transpose('time', 'lat', 'lon')
    anom.name = 'z250_anom'
    anom.attrs['units'] = 'm'
    logger.info(
        'Anomalia Z250: {} dias | min={:.0f} max={:.0f} mgp',
        anom.sizes['time'], float(anom.min()), float(anom.max()),
    )
    return anom


def _build_z250_reanalise() -> xr.DataArray:
    """Serie diaria de anomalia de Z250 observada (ERA5/CDS + GDAS/NOMADS)."""
    from app.src.uteis.downloaders_gdas_hgt250 import ensure_gdas_hgt250_for_period
    from app.src.uteis.downloaders_hgt250_ERA5 import (
        ensure_era5_altura_geopotencial_250_global_for_period_grib,
    )

    dt_ini = _to_datetime(settings.DATA_INICIAL)
    dt_fim = _to_datetime(settings.DATA_FINAL)
    force = bool(getattr(settings, 'FORCE_DOWNLOAD', False))

    hoje = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if dt_fim >= hoje:
        dt_fim = hoje - timedelta(days=1)
        logger.warning('DATA_FINAL ajustada para ontem ({}) — limite do GDAS', dt_fim.date())

    era5_period, gdas_period = _get_data_sources(dt_ini, dt_fim)
    files: list[Path] = []
    if era5_period:
        logger.info('Download ERA5 Z250: {} -> {}', era5_period[0].date(), era5_period[1].date())
        files += ensure_era5_altura_geopotencial_250_global_for_period_grib(
            start=era5_period[0], end=era5_period[1], hours_utc=list(DEFAULT_SYNOPTIC_HOURS),
            force_redownload=force, convert_to_height_netcdf=True,
        )
    if gdas_period:
        logger.info('Download GDAS Z250: {} -> {}', gdas_period[0].date(), gdas_period[1].date())
        files += ensure_gdas_hgt250_for_period(
            start=gdas_period[0], end=gdas_period[1], force_redownload=force,
        )

    tgt_lat, tgt_lon = _target_grid()
    daily = _daily_scalar_on_grid(files, HGT_VARS, dt_ini, dt_fim, tgt_lat, tgt_lon, logger)
    return _z250_anom_from_daily(daily)


def _build_z250_forecast(model: str) -> xr.DataArray:
    """Serie diaria de anomalia de Z250 prevista (lagged ensemble, padrao s34)."""
    rodada = int(settings.get('RODADA', 0))
    if rodada not in (0, 6, 12, 18):
        raise ValueError(f'RODADA deve ser 00/06/12/18 (UTC). Recebido: {rodada:02d}')

    run_inits, lead_hours = _resolve_forecast_lead_init(
        model,
        rodada=rodada,
        num_rodada=int(settings.get('NUM_RODADA', 1)),
        forecast_init=settings.get('FORECAST_INIT', 'latest'),
        gefs_lead_days=int(settings.get('GEFS_FORECAST_LEAD_DAYS', settings.get('FORECAST_LEAD_DAYS', 35))),
        cfs_lead_days=45,
    )
    init0 = run_inits[0]
    # Janela de previsao a plotar = DATA_INICIAL/DATA_FINAL do settings (mesmas do
    # modo reanalise), recortada ao horizonte disponivel do modelo.
    dt_ini = _to_datetime(settings.DATA_INICIAL)
    dt_fim = _to_datetime(settings.DATA_FINAL)
    avail_ini = datetime(init0.year, init0.month, init0.day) + timedelta(days=1)  # descarta o init
    avail_fim = init0 + timedelta(hours=lead_hours)
    win_ini = max(dt_ini, avail_ini)
    win_fim = min(dt_fim, avail_fim)
    if win_ini.date() > win_fim.date():
        raise RuntimeError(
            f'Janela [{dt_ini.date()} a {dt_fim.date()}] fora do horizonte do {model.upper()}. '
            f'Previsao disponivel: {avail_ini.date()} a {avail_fim.date()} '
            f'(init {init0:%Y-%m-%d %H}Z). Ajuste DATA_INICIAL/DATA_FINAL para datas FUTURAS '
            f'dentro desse intervalo.'
        )
    logger.info(
        'FORECAST {}: init {:%Y-%m-%d %H}Z, lead {}h | janela plotada {} a {}',
        model.upper(), init0, lead_hours, win_ini.date(), win_fim.date(),
    )

    hgt250_fn = _hgt250_fcst_downloader(model)
    tgt_lat, tgt_lon = _target_grid()
    per_run: list[xr.DataArray] = []
    for init_k in run_inits:
        files_k = list(hgt250_fn(
            init=init_k, lead_hours=lead_hours, hours=list(DEFAULT_SYNOPTIC_HOURS),
        ))
        if files_k:
            per_run.append(_daily_scalar_on_grid(
                files_k, HGT_VARS, win_ini, win_fim, tgt_lat, tgt_lon, logger,
            ))
    if not per_run:
        raise RuntimeError(f'Sem dados de Z250 do modelo {model.upper()} no horizonte previsto.')

    daily = _lagged_ensemble_mean(per_run)
    return _z250_anom_from_daily(daily)


# ===========================================================================
# REGISTRO DE VARIAVEIS — adicione uma ficha por variavel plotavel
# ===========================================================================
VARIAVEIS: dict[str, dict] = {
    'z250_anom': {
        'titulo': 'Anomalia de Altura Geopotencial em 250 hPa',
        'titulo_en': '250-hPa geopotential height',  # usado no titulo da tarja (ingles)
        'unidade': 'mgp',
        'cmap_colors': list(settings.LST_ANOM_CORRETA),
        'simetrico': True,   # paleta centrada em zero
        'vmax': None,        # None = automatico (percentil robusto)
        'builders': {
            'reanalise': _build_z250_reanalise,
            'forecast': _build_z250_forecast,
        },
    },
}


# ---------------------------------------------------------------------------
# Projecao do globo
# ---------------------------------------------------------------------------
# Divisas estaduais APENAS destes paises (codigo adm0_a3 do Natural Earth).
_ESTADOS_PAISES = {'USA', 'BRA', 'AUS'}
_STATE_GEOMS_CACHE: list | None = None


def _state_line_geoms() -> list:
    """Geometrias das divisas estaduais de EUA/Brasil/Australia (carregadas 1x).

    Filtra o Natural Earth admin_1 por pais — evita plotar os estados/provincias
    de todos os paises do mundo (so divisas nacionais nos demais).
    """
    global _STATE_GEOMS_CACHE
    if _STATE_GEOMS_CACHE is None:
        import cartopy.io.shapereader as shpreader

        path = shpreader.natural_earth(
            resolution='50m', category='cultural', name='admin_1_states_provinces_lines',
        )
        geoms = []
        for rec in shpreader.Reader(path).records():
            a3 = rec.attributes.get('adm0_a3') or rec.attributes.get('ADM0_A3')
            if a3 in _ESTADOS_PAISES:
                geoms.append(rec.geometry)
        _STATE_GEOMS_CACHE = geoms
        logger.info('Divisas estaduais carregadas (EUA/Brasil/Australia): {} feicoes', len(geoms))
    return _STATE_GEOMS_CACHE


_VIGNETTE_CACHE: np.ndarray | None = None


def _vignette_rgba(n: int = 512) -> np.ndarray:
    """Mascara RGBA de vinheta (preto transparente no centro, escuro nos cantos).

    Reproduz o escurecimento radial do mapa do WaPo: o centro do globo fica claro
    e as bordas/cantos escurecem progressivamente. Calculada uma unica vez.
    """
    global _VIGNETTE_CACHE
    if _VIGNETTE_CACHE is None:
        yy, xx = np.mgrid[-1:1:complex(0, n), -1:1:complex(0, n)]
        r = np.sqrt(xx ** 2 + yy ** 2)  # 0 no centro, ~1.41 nos cantos
        alpha = np.clip((r - 0.30) / (1.10 - 0.30), 0.0, 1.0) ** 1.5
        rgba = np.zeros((n, n, 4), dtype=float)
        rgba[..., 3] = alpha * 0.85  # ate 85% de preto nos cantos
        _VIGNETTE_CACHE = rgba
    return _VIGNETTE_CACHE


def _make_projection(central_lon: float, central_lat: float):
    nome = str(getattr(settings, 'GLOBO_3D_PROJECTION', 'nearside')).lower()
    if nome.startswith('ortho'):
        return ccrs.Orthographic(central_longitude=central_lon, central_latitude=central_lat)
    return ccrs.NearsidePerspective(
        central_longitude=central_lon, central_latitude=central_lat, satellite_height=35785831,
    )


def _ease(prog: np.ndarray, mode: str) -> np.ndarray:
    """Aplica o perfil de velocidade (easing) ao progresso 0..1 do voo.

    - 'linear'      velocidade constante do inicio ao fim.
    - 'ease_out'    rapido no inicio e DESACELERA, assentando suave no destino.
    - 'ease_in'     lento no inicio e acelera ao chegar.
    - 'ease_in_out' lento nas duas pontas, mais veloz no meio (smoothstep).
    """
    m = str(mode).lower()
    if m == 'ease_out':
        return 1.0 - (1.0 - prog) ** 2
    if m == 'ease_in':
        return prog ** 2
    if m in ('ease_in_out', 'smooth', 'smoothstep'):
        return prog ** 2 * (3.0 - 2.0 * prog)
    return prog  # linear


def _camera_path(total_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Trajetoria (lon, lat) da camera do 1o ao ultimo frame.

    Viaja de (LON/LAT_INICIAL) a (LON/LAT_FINAL) pelo menor arco em longitude,
    somando VOLTAS_EXTRA giros completos (para o efeito de rotacao). O perfil de
    velocidade vem de GLOBO_3D_EASING (constante ou com desaceleracao no fim).
    """
    lon_i = float(getattr(settings, 'GLOBO_3D_LON_INICIAL', -150.0))
    lat_i = float(getattr(settings, 'GLOBO_3D_LAT_INICIAL', 0.0))
    lon_f = float(getattr(settings, 'GLOBO_3D_LON_FINAL', -45.0))
    lat_f = float(getattr(settings, 'GLOBO_3D_LAT_FINAL', -15.0))
    voltas_extra = float(getattr(settings, 'GLOBO_3D_VOLTAS_EXTRA', 0.0))
    easing = str(getattr(settings, 'GLOBO_3D_EASING', 'linear'))

    # Inclinacao FIXA do globo (opcional): sobrescreve a latitude da camera para
    # mostrar mais um hemisferio. >0 = mais Hemisferio Norte; <0 = mais Sul; 0 = equador.
    # Vazio/""/None = usa LAT_INICIAL/LAT_FINAL (inclinacao varia durante o voo).
    inclin = getattr(settings, 'GLOBO_3D_INCLINACAO', '')
    if inclin not in ('', None):
        lat_i = lat_f = float(inclin)

    base_delta = ((lon_f - lon_i + 180.0) % 360.0) - 180.0  # menor arco em (-180, 180]
    total_lon_travel = base_delta + 360.0 * voltas_extra
    prog = np.linspace(0.0, 1.0, total_frames) if total_frames > 1 else np.zeros(1)
    prog = _ease(prog, easing)
    lons = lon_i + prog * total_lon_travel
    lats = lat_i + prog * (lat_f - lat_i)
    return lons, lats


# ---------------------------------------------------------------------------
# Renderizacao de UM frame (usada em serie ou em paralelo via process pool).
# ---------------------------------------------------------------------------
_FRAME_CTX: dict | None = None


def _init_frame_worker(ctx: dict) -> None:
    """Inicializa cada processo worker com o contexto do clipe (1x por worker)."""
    global _FRAME_CTX
    _FRAME_CTX = ctx


def _build_frame(f: int, ctx: dict) -> np.ndarray:
    """Renderiza o frame `f` e devolve um array RGB uint8 (HxWx3)."""
    vals_cyc = ctx['vals_cyc']
    lon_cyc, lat, levels = ctx['lon_cyc'], ctx['lat'], ctx['levels']
    n_dias, fpd, vel = ctx['n_dias'], ctx['frames_por_dia'], ctx['vel_var']
    cmap = LinearSegmentedColormap.from_list('globo3d', list(ctx['paleta']))
    data_transform = ccrs.PlateCarree()

    # Tempo (variavel) avanca a `vel` dias-de-frame por frame, desacoplado do voo;
    # clampa no ultimo dia se terminar antes do fim do voo.
    pos = min(f * vel / fpd, n_dias - 1) if n_dias > 1 else 0.0
    i0 = min(int(np.floor(pos)), n_dias - 1)
    i1 = min(i0 + 1, n_dias - 1)
    w = pos - i0
    campo = (1.0 - w) * vals_cyc[i0] + w * vals_cyc[i1]
    data_en = ctx['dates_en'][min(int(round(pos)), n_dias - 1)]

    proj = _make_projection(float(ctx['lons'][f]), float(ctx['lats'][f]))
    fig = plt.figure(figsize=(8, 8), dpi=ctx['dpi'])
    fig.patch.set_facecolor('black')
    ax = fig.add_axes([0.03, 0.22, 0.94, 0.76], projection=proj)
    ax.patch.set_facecolor('black')
    ax.set_global()

    ax.contourf(lon_cyc, lat, campo, levels=levels, cmap=cmap, extend='both',
                transform=data_transform, zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.7, edgecolor='black', zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5, edgecolor='black', zorder=5)
    estados = _state_line_geoms()
    if estados:
        ax.add_geometries(estados, data_transform, edgecolor='black',
                          facecolor='none', linewidth=0.35, zorder=5)
    ax.gridlines(linewidth=0.3, color='white', alpha=0.2, zorder=4)
    ax.set_zorder(1)

    if ctx['usar_vinheta']:
        vax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        vax.set_axis_off()
        vax.set_zorder(10)
        vax.imshow(_vignette_rgba(), extent=[0, 1, 0, 1], origin='lower',
                   aspect='auto', interpolation='bilinear', zorder=10)
        vax.set_xlim(0, 1)
        vax.set_ylim(0, 1)

    # ── Tarja inferior (estilo WaPo) ──
    fig.text(0.5, 0.150, f"{ctx['titulo_en']} anomalies on {data_en}", color='white',
             fontsize=18, ha='center', va='center', weight='bold',
             family=ctx['font_titulo'], zorder=20)

    sw_w, sw_h, gap = 0.085, 0.024, 0.050
    x0 = 0.5 - (4 * sw_w + 3 * gap) / 2.0
    y_sw = 0.080
    legenda_cores = [cmap(x) for x in (0.12, 0.34, 0.66, 0.88)]
    for i, (cor, lab) in enumerate(zip(legenda_cores, ctx['legenda_labels'])):
        x = x0 + i * (sw_w + gap)
        fig.add_artist(Rectangle((x, y_sw), sw_w, sw_h, transform=fig.transFigure,
                                 facecolor=cor, edgecolor='none', zorder=20))
        fig.text(x + sw_w / 2.0, y_sw - 0.028, lab, color='white', fontsize=11,
                 ha='center', va='center', weight='bold', family=ctx['font_legenda'], zorder=20)

    fig.text(0.020, 0.022, ctx['fonte_label'].upper(), color='#bdbdbd', fontsize=9,
             ha='left', va='center', family=FONT_SANS, zorder=20)
    fig.text(0.980, 0.022, ctx['credito'], color='#bdbdbd', fontsize=9,
             ha='right', va='center', family=FONT_SANS, zorder=20)

    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return arr


def _render_one_frame(f: int) -> np.ndarray:
    """Wrapper picklavel p/ o process pool (usa o contexto global do worker)."""
    return _build_frame(f, _FRAME_CTX)


# ---------------------------------------------------------------------------
# Renderizacao de um clipe MP4 a partir de uma serie diaria de anomalia
# ---------------------------------------------------------------------------
def _render_clip(anom: xr.DataArray, ficha: dict, variavel_key: str,
                 output_dir: Path, fonte_label: str) -> Path:
    frames_por_dia = int(getattr(settings, 'GLOBO_3D_FRAMES_POR_DIA', 4))
    fps = int(getattr(settings, 'GLOBO_3D_FPS', 20))
    coarsen = int(getattr(settings, 'GLOBO_3D_COARSEN', 1))

    if coarsen and coarsen > 1:
        anom = anom.coarsen(lat=coarsen, lon=coarsen, boundary='trim').mean()

    lat = anom['lat'].values
    vals_cyc, lon_cyc = add_cyclic_point(anom.values, coord=anom['lon'].values)
    dates = pd.DatetimeIndex(pd.to_datetime(anom['time'].values))
    n_dias = vals_cyc.shape[0]

    # Escala de cores fixa (estavel durante todo o clipe)
    vmax = ficha.get('vmax')
    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(vals_cyc), 98))
        vmax = max(10.0, round(vmax / 10.0) * 10.0)
    vmin = -vmax if ficha.get('simetrico', True) else float(np.nanmin(vals_cyc))
    paleta = settings.get('GLOBO_3D_PALETA', None) or ficha['cmap_colors']
    niveis = int(getattr(settings, 'GLOBO_3D_NIVEIS', 16))
    levels = np.linspace(vmin, vmax, niveis + 1)

    total_frames = (n_dias - 1) * frames_por_dia + 1 if n_dias > 1 else max(frames_por_dia, 1)
    lons, lats = _camera_path(total_frames)
    vel_var = max(0.05, float(getattr(settings, 'GLOBO_3D_VELOCIDADE_VAR', 1.0)))

    # Resolucao de saida (px). dpi tal que 8in * dpi = px (figsize fixo 8).
    px = int(getattr(settings, 'GLOBO_3D_TAMANHO_PX', 1080))
    dpi = px / 8.0

    # Fontes: titulo e legenda (ex.: "Aptos Display" se o .ttf existir em Entrada/fonts).
    font_titulo = _resolve_family(getattr(settings, 'GLOBO_3D_FONTE_TITULO', ''), FONT_SERIF)
    font_legenda = _resolve_family(getattr(settings, 'GLOBO_3D_FONTE_LEGENDA', ''), FONT_SANS)

    # Workers: GLOBO_3D_WORKERS (0 = auto = todos os nucleos).
    workers = int(getattr(settings, 'GLOBO_3D_WORKERS', 0)) or (os.cpu_count() or 1)
    workers = max(1, min(workers, total_frames))

    ctx = {
        'vals_cyc': vals_cyc.astype(np.float32),
        'lon_cyc': lon_cyc, 'lat': lat, 'levels': levels,
        'paleta': list(paleta),
        'lons': lons, 'lats': lats,
        'frames_por_dia': frames_por_dia, 'vel_var': vel_var, 'n_dias': n_dias,
        'dates_en': [d.strftime('%B %-d') for d in dates],
        'titulo_en': ficha.get('titulo_en', ficha['titulo']),
        'fonte_label': fonte_label,
        'credito': str(getattr(settings, 'GLOBO_3D_CREDITO', 'Bruno Capucin')).upper(),
        'font_titulo': font_titulo, 'font_legenda': font_legenda,
        'usar_vinheta': bool(getattr(settings, 'GLOBO_3D_VINHETA', True)),
        'legenda_labels': ['Well below', 'Below', 'Above', 'Well above'],
        'dpi': dpi,
    }

    logger.info('{} dias -> {} frames | {} fps | {}px | {} workers | {}',
                n_dias, total_frames, fps, px, workers, fonte_label)

    output_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = output_dir / f's38_{variavel_key}.mp4'
    writer = imageio.get_writer(
        str(mp4_path), fps=fps, codec='libx264', quality=8, macro_block_size=8,
    )
    t0 = _time.time()

    try:
        if workers > 1:
            # Render paralelo: frames independentes em process pool (fork), em ordem.
            pool = get_context('fork').Pool(
                workers, initializer=_init_frame_worker, initargs=(ctx,))
            try:
                for i, frame in enumerate(
                        pool.imap(_render_one_frame, range(total_frames), chunksize=1)):
                    writer.append_data(frame)
                    if (i + 1) % 20 == 0 or i == total_frames - 1:
                        logger.info('  frame {}/{}', i + 1, total_frames)
            finally:
                pool.close()
                pool.join()
        else:
            _init_frame_worker(ctx)
            for f in range(total_frames):
                writer.append_data(_build_frame(f, ctx))
                if (f + 1) % 20 == 0 or f == total_frames - 1:
                    logger.info('  frame {}/{}', f + 1, total_frames)
    finally:
        writer.close()

    logger.info('MP4 salvo: {} ({:.1f}s)', mp4_path, _time.time() - t0)
    return mp4_path


# ---------------------------------------------------------------------------
# Engine principal
# ---------------------------------------------------------------------------
def gerar_animacao(variavel_key: str, output_base: Path) -> list[Path]:
    """Gera o(s) MP4(s) do globo animado para a variavel do registro.

    Reanalise -> 1 MP4. Forecast -> 1 MP4 por modelo habilitado.
    Retorna a lista de caminhos gerados.
    """
    if variavel_key not in VARIAVEIS:
        raise ValueError(
            f"Variavel '{variavel_key}' nao registrada. Disponiveis: {list(VARIAVEIS.keys())}"
        )

    ficha = VARIAVEIS[variavel_key]
    modo = str(getattr(settings, 'GLOBO_3D_MODO', 'reanalise')).lower()
    logger.info('=' * 70)
    logger.info('GLOBO 3D MIDIA [{}]: {}', modo.upper(), ficha['titulo'])
    logger.info('=' * 70)

    outputs: list[Path] = []
    if modo.startswith('rean'):
        anom = ficha['builders']['reanalise']()
        outputs.append(_render_clip(
            anom, ficha, variavel_key, output_base / 'REANALISE', 'Reanalysis',
        ))
    elif modo.startswith('fore') or modo.startswith('prev'):
        models = _enabled_forecast_models()
        if not models:
            raise RuntimeError(
                'Nenhum modelo de forecast habilitado. Ative RUN_GFS/RUN_GEFS/RUN_ECMWF/RUN_AIFS.'
            )
        logger.info('Modelos habilitados: {}', ', '.join(m.upper() for m in models))
        for model in models:
            anom = ficha['builders']['forecast'](model)
            outputs.append(_render_clip(
                anom, ficha, variavel_key,
                output_base / 'FORECAST' / model.upper(), model.upper(),
            ))
    else:
        raise ValueError(f"GLOBO_3D_MODO invalido: '{modo}'. Use 'reanalise' ou 'forecast'.")

    return outputs
