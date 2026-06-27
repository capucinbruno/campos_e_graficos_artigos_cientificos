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
  (modo automatico: datas passadas->reanalise, futuras->previsao, cruza hoje->emenda)
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
import textwrap
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
from matplotlib.patches import FancyBboxPatch, Rectangle

# Modulos locais
from app.shared.logger import get_logger
from app.src.uteis.forecast_daily import (
    daily_scalar_on_grid as _daily_scalar_on_grid,
    lagged_ensemble_mean as _lagged_ensemble_mean,
    resolve_forecast_lead_init as _resolve_forecast_lead_init,
)
from app.shared.settings_factory import settings
from app.src.uteis.clim_diaria_uv200_ltm import clim_hgt250_daily, clim_t850_daily
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


# Candidatos de nome da variavel de temperatura nos arquivos (ERA5/GDAS/modelos).
TMP_VARS = ('t', 'tmp', 'air', 'temperature')


def _fcst_downloader(model: str, kind: str):
    """Resolve o downloader de forecast (import tardio) por modelo e variavel.

    kind: 'hgt250' (Z250) ou 'tmp850' (temperatura 850 hPa).
    """
    table = {
        ('gfs', 'hgt250'): ('downloaders_gfs_hgt250', 'ensure_gfs_hgt250_fcst_for_period'),
        ('gefs', 'hgt250'): ('downloaders_gefs_hgt250', 'ensure_gefs_hgt250_fcst_for_period'),
        ('ecmwf', 'hgt250'): ('downloaders_ecmwf_hgt250', 'ensure_ecmwf_hgt250_fcst_for_period'),
        ('aifs', 'hgt250'): ('downloaders_aifs_hgt250', 'ensure_aifs_hgt250_fcst_for_period'),
        ('gfs', 'tmp850'): ('downloaders_gfs_tmp850', 'ensure_gfs_tmp850_fcst_for_period'),
        ('gefs', 'tmp850'): ('downloaders_gefs_tmp850', 'ensure_gefs_tmp850_fcst_for_period'),
        ('ecmwf', 'tmp850'): ('downloaders_ecmwf_tmp850', 'ensure_ecmwf_tmp850_fcst_for_period'),
        ('aifs', 'tmp850'): ('downloaders_aifs_tmp850', 'ensure_aifs_tmp850_fcst_for_period'),
    }
    key = (model, kind)
    if key not in table:
        raise ValueError(f'Modelo {model!r} sem suporte a {kind} no s38.')
    mod_name, fn_name = table[key]
    import importlib
    return getattr(importlib.import_module(f'app.src.uteis.{mod_name}'), fn_name)


# ERA5/GDAS (reanalise) — wrappers com assinatura uniforme.
def _era5_z250(start, end, force):
    from app.src.uteis.downloaders_hgt250_ERA5 import (
        ensure_era5_altura_geopotencial_250_global_for_period_grib as fn)
    return fn(start=start, end=end, hours_utc=list(DEFAULT_SYNOPTIC_HOURS),
              force_redownload=force, convert_to_height_netcdf=True)


def _gdas_z250(start, end, force):
    from app.src.uteis.downloaders_gdas_hgt250 import ensure_gdas_hgt250_for_period as fn
    return fn(start=start, end=end, force_redownload=force)


def _era5_t850(start, end, force):
    from app.src.uteis.downloaders_tmp850_ERA5 import ensure_era5_t850_for_period as fn
    return fn(start=start, end=end, hours_utc=list(DEFAULT_SYNOPTIC_HOURS), force_redownload=force)


def _gdas_t850(start, end, force):
    from app.src.uteis.downloaders_gdas_tmp850 import ensure_gdas_tmp850_for_period as fn
    return fn(start=start, end=end, force_redownload=force)


# ===========================================================================
# BUILDERS GENERICOS — devolvem xr.DataArray(time, lat, lon) de ANOMALIA diaria
# ===========================================================================
def _to_celsius_da(da: xr.DataArray) -> xr.DataArray:
    """K -> °C se o campo estiver em escala Kelvin (media > 100)."""
    return da - 273.15 if float(da.mean()) > 100 else da


def _anom_from_clim(daily: xr.DataArray, clim_fn, *, celsius: bool,
                    nome: str, unidade: str) -> xr.DataArray:
    """anomalia = campo diario - climatologia diaria LTM 1991-2020, alinhada (mes, dia).

    `daily` na grade fina; `clim_fn(dates)` devolve (arr, lat, lon) na grade NCEP 2.5°,
    interpolada p/ a grade do `daily`. Ponto ciclico em lon evita a faixa NaN em
    Greenwich (a clim NCEP vai so ate 357.5°). Se `celsius`, normaliza obs e clim p/ °C.
    """
    arr, clat, clon = clim_fn(daily['time'].values)
    clim = xr.DataArray(
        arr, dims=['time', 'lat', 'lon'],
        coords={'time': daily['time'].values, 'lat': clat, 'lon': clon},
    ).sortby('lat')
    cyc_vals, cyc_lon = add_cyclic_point(clim.values, coord=clim['lon'].values)
    clim = xr.DataArray(
        cyc_vals, dims=['time', 'lat', 'lon'],
        coords={'time': clim['time'].values, 'lat': clim['lat'].values, 'lon': cyc_lon},
    ).interp(lat=daily['lat'], lon=daily['lon'], method='linear')

    if celsius:
        daily, clim = _to_celsius_da(daily), _to_celsius_da(clim)
    anom = (daily - clim).transpose('time', 'lat', 'lon')
    anom.name = nome
    anom.attrs['units'] = unidade
    logger.info('Anomalia {}: {} dias | min={:.1f} max={:.1f} {}',
                nome, anom.sizes['time'], float(anom.min()), float(anom.max()), unidade)
    return anom


def _reanalise_series(spec: dict, dt_ini: datetime, dt_fim: datetime) -> xr.DataArray | None:
    """Anomalia OBSERVADA (ERA5/CDS + GDAS/NOMADS) na janela [dt_ini, dt_fim]."""
    if dt_ini.date() > dt_fim.date():
        return None
    force = bool(getattr(settings, 'FORCE_DOWNLOAD', False))
    era5_period, gdas_period = _get_data_sources(dt_ini, dt_fim)
    files: list[Path] = []
    if era5_period:
        logger.info('Download ERA5 {}: {} -> {}', spec['nome'], era5_period[0].date(), era5_period[1].date())
        files += spec['era5_fn'](era5_period[0], era5_period[1], force)
    if gdas_period:
        logger.info('Download GDAS {}: {} -> {}', spec['nome'], gdas_period[0].date(), gdas_period[1].date())
        files += spec['gdas_fn'](gdas_period[0], gdas_period[1], force)
    tgt_lat, tgt_lon = _target_grid()
    daily = _daily_scalar_on_grid(files, spec['var_candidates'], dt_ini, dt_fim, tgt_lat, tgt_lon, logger)
    return _anom_from_clim(daily, spec['clim_fn'], celsius=spec['celsius'],
                           nome=spec['nome'], unidade=spec['unidade'])


def _forecast_series(spec: dict, model: str, dt_ini: datetime, dt_fim: datetime) -> xr.DataArray | None:
    """Anomalia PREVISTA (lagged ensemble) na janela [dt_ini, dt_fim], recortada ao
    horizonte do modelo. Inclui o dia do init (campos instantaneos)."""
    rodada = int(settings.get('RODADA', 0))
    if rodada not in (0, 6, 12, 18):
        raise ValueError(f'RODADA deve ser 00/06/12/18 (UTC). Recebido: {rodada:02d}')
    run_inits, lead_hours = _resolve_forecast_lead_init(
        model, rodada=rodada, num_rodada=int(settings.get('NUM_RODADA', 1)),
        forecast_init=settings.get('FORECAST_INIT', 'latest'),
        gefs_lead_days=int(settings.get('GEFS_FORECAST_LEAD_DAYS', settings.get('FORECAST_LEAD_DAYS', 35))),
        cfs_lead_days=45,
    )
    init0 = run_inits[0]
    avail_ini = datetime(init0.year, init0.month, init0.day)        # inclui o dia do init
    avail_fim = init0 + timedelta(hours=lead_hours)
    win_ini = max(dt_ini, avail_ini)
    win_fim = min(dt_fim, avail_fim)
    if win_ini.date() > win_fim.date():
        raise RuntimeError(
            f'Parte de PREVISAO [{dt_ini.date()} a {dt_fim.date()}] fora do horizonte do '
            f'{model.upper()} (disponivel {avail_ini.date()} a {avail_fim.date()}, '
            f'init {init0:%Y-%m-%d %H}Z). Reduza DATA_FINAL ou aumente o lead.'
        )
    logger.info('FORECAST {} [{}]: init {:%Y-%m-%d %H}Z, lead {}h | janela {} a {}',
                model.upper(), spec['nome'], init0, lead_hours, win_ini.date(), win_fim.date())
    fn = _fcst_downloader(model, spec['kind'])
    tgt_lat, tgt_lon = _target_grid()
    per_run: list[xr.DataArray] = []
    for init_k in run_inits:
        files_k = list(fn(init=init_k, lead_hours=lead_hours, hours=list(DEFAULT_SYNOPTIC_HOURS)))
        if files_k:
            per_run.append(_daily_scalar_on_grid(
                files_k, spec['var_candidates'], win_ini, win_fim, tgt_lat, tgt_lon, logger))
    if not per_run:
        raise RuntimeError(f'Sem dados de {spec["nome"]} do modelo {model.upper()} no horizonte.')
    daily = _lagged_ensemble_mean(per_run)
    out = _anom_from_clim(daily, spec['clim_fn'], celsius=spec['celsius'],
                          nome=spec['nome'], unidade=spec['unidade'])
    out.attrs['run_init'] = init0.strftime('%Y-%m-%d %H')  # rodada (rotulo do rodape no s39)
    return out


def _build_var_series(ficha: dict, model: str | None,
                      dt_ini: datetime, dt_fim: datetime) -> xr.DataArray:
    """Serie diaria de anomalia na janela [dt_ini, dt_fim], decidida PELAS DATAS:
    passado -> reanalise; futuro -> previsao (modelo); janela que cruza hoje ->
    EMENDA observado + previsao num unico vetor temporal continuo.
    """
    spec = ficha['spec']
    hoje = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    ontem = hoje - timedelta(days=1)

    partes: list[xr.DataArray] = []
    if dt_ini <= ontem:  # ha parte OBSERVADA (ate ontem)
        partes.append(_reanalise_series(spec, dt_ini, min(dt_fim, ontem)))
    if dt_fim >= hoje:   # ha parte PREVISTA (de hoje em diante)
        if model is None:
            raise RuntimeError('Janela tem datas futuras mas nenhum modelo foi habilitado.')
        partes.append(_forecast_series(spec, model, max(dt_ini, hoje), dt_fim))

    partes = [p for p in partes if p is not None and p.sizes.get('time', 0) > 0]
    if not partes:
        raise RuntimeError(f'Sem dados de {spec["nome"]} na janela {dt_ini.date()} a {dt_fim.date()}.')
    if len(partes) == 1:
        return partes[0]

    serie = xr.concat(partes, dim='time').sortby('time')
    _, idx = np.unique(serie['time'].values, return_index=True)
    serie = serie.isel(time=np.sort(idx))
    for p in partes:  # preserva a rodada da parte de previsao (concat mantem attrs da 1a)
        if 'run_init' in p.attrs:
            serie.attrs['run_init'] = p.attrs['run_init']
    logger.info('EMENDA observado+previsao: {} dias ({} a {})',
                serie.sizes['time'],
                pd.Timestamp(serie['time'].values[0]).date(),
                pd.Timestamp(serie['time'].values[-1]).date())
    return serie


def _output_plan(variaveis: list[str], output_base: Path):
    """Plano de saida decidido pelas datas. Retorna (lista de itens, dt_ini, dt_fim).

    Cada item: {var, model, dir, label}. Sem datas futuras -> 1 item por variavel
    (REANALISE/, model=None). Com datas futuras -> 1 item por variavel POR modelo
    habilitado (FORECAST/<MODELO>/); label indica 'ERA5 + <MODELO>' se houver emenda.
    """
    invalidas = [v for v in variaveis if v not in VARIAVEIS]
    if invalidas:
        raise ValueError(f'Variaveis nao registradas: {invalidas}. Disponiveis: {list(VARIAVEIS.keys())}')

    dt_ini = _to_datetime(settings.DATA_INICIAL)
    dt_fim = _to_datetime(settings.DATA_FINAL)
    hoje = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    ontem = hoje - timedelta(days=1)
    has_past = dt_ini <= ontem
    has_future = dt_fim >= hoje

    plano = []
    if has_future:
        models = _enabled_forecast_models()
        if not models:
            raise RuntimeError(
                'A janela tem datas FUTURAS (previsao), mas nenhum modelo esta habilitado. '
                'Ative RUN_GFS/RUN_GEFS/RUN_ECMWF/RUN_AIFS.'
            )
        for model in models:
            label = f'ERA5 + {model.upper()}' if has_past else model.upper()
            for var in variaveis:
                plano.append({'var': var, 'model': model,
                              'dir': output_base / 'FORECAST' / model.upper(), 'label': label})
    else:
        for var in variaveis:
            plano.append({'var': var, 'model': None,
                          'dir': output_base / 'REANALISE', 'label': 'Reanalysis'})
    return plano, dt_ini, dt_fim


# ===========================================================================
# REGISTRO DE VARIAVEIS — adicione uma ficha por variavel plotavel.
# A ficha so aponta downloaders/climatologia (campo 'spec'); o motor (voo, tempo,
# render, legenda) e as funcoes genericas de serie cuidam do resto.
# ===========================================================================
VARIAVEIS: dict[str, dict] = {
    'z250_anom': {
        'titulo': 'Anomalia de Altura Geopotencial em 250 hPa',
        'titulo_en': '250-hPa geopotential height',  # titulo da tarja (ingles)
        'rotulo_box': 'Geopotential Height Anomaly',  # caixa do s39 (curto, ingles)
        'subtitulo_dir': 'Geopotential height at 250 hPa',  # canto sup-dir do s39 (variavel + nivel)
        'unidade': 'mgp',
        'cmap_colors': list(settings.LST_ANOM_CORRETA),
        'simetrico': True,
        'vmax': None,
        'spec': {
            'nome': 'z250_anom', 'unidade': 'mgp', 'celsius': False,
            'var_candidates': HGT_VARS, 'clim_fn': clim_hgt250_daily, 'kind': 'hgt250',
            'era5_fn': _era5_z250, 'gdas_fn': _gdas_z250,
        },
    },
    'tmp850_anom': {
        'titulo': 'Anomalia de Temperatura do Ar em 850 hPa',
        'titulo_en': '850-hPa air temperature',
        'rotulo_box': 'Air Temperature Anomaly',  # caixa do s39 (curto, ingles)
        'subtitulo_dir': 'Air temperature at 850 hPa',  # canto sup-dir do s39 (variavel + nivel)
        'unidade': '°C',
        # Paleta amostrada PIXEL A PIXEL da barra de referencia (Entrada/paleta.jpg),
        # esquerda->direita: magenta (frio extremo) -> roxo -> azul-escuro -> azul ->
        # branco (conforme a la moyenne) -> salmao -> vermelho -> vinho -> quase-preto
        # -> cinza-carvao (quente extremo; confere com os blobs ~(28,28,28) do mapa).
        'cmap_colors': [
            '#a30d92', '#720669', '#3c0654', '#17022b', '#021323',
            '#043462', '#104b87', '#256aab', '#3d89bd', '#5ea3cc',
            '#bad9eb', '#e1ebf4', '#f7f7f7', '#f8ede7', '#f9dac8',
            '#e58366', '#cd5147', '#b72532', '#9c1526', '#821220',
            '#5f0d19', '#3c0711', '#25060b', '#0d0707', '#3d3d3d',
        ],
        'niveis': 128,       # bandas do shaded (suave, ~2x mais rapido que 256) (override: GLOBO_3D_NIVEIS_TMP850_ANOM)
        'simetrico': True,
        'vmax': 15.0,        # escala FIXA: shaded de -15 a +15 °C
        'spec': {
            'nome': 'tmp850_anom', 'unidade': '°C', 'celsius': True,
            'var_candidates': TMP_VARS, 'clim_fn': clim_t850_daily, 'kind': 'tmp850',
            'era5_fn': _era5_t850, 'gdas_fn': _gdas_t850,
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


def _overlay_guillaume(fig, ctx: dict, cmap, data_full: str) -> None:
    """Overlay estilo Guillaume Jauseau (s39): caixa do nome no topo-esquerdo + data,
    barra de gradiente continua numa caixa translucida no centro-inferior, e rodape
    com modelo/rodada (esq.) e credito (dir.). Sem vinheta."""
    # ── Caixa cinza (topo-esquerdo), enquadrada no canto e justa ao texto ──
    # Ancora o titulo no canto sup-esq e dimensiona a caixa pelo EXTENT real do texto
    # (margem minima), em vez de um tamanho fixo com sobra.
    titulo = textwrap.fill(str(ctx['titulo_box']).upper(), width=15)
    x_anchor, y_anchor = 0.016, 0.978
    t = fig.text(x_anchor, y_anchor, titulo, color='white', fontsize=11, ha='left', va='top',
                 weight='bold', family=ctx['font_legenda'], zorder=21)
    bb = t.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    pad_x, pad_y = 0.009, 0.008
    fig.add_artist(FancyBboxPatch(
        (bb.x0 - pad_x, bb.y0 - pad_y), bb.width + 2 * pad_x, bb.height + 2 * pad_y,
        boxstyle='round,pad=0,rounding_size=0.006', transform=fig.transFigure,
        facecolor='#9a9a9a', alpha=0.90, edgecolor='none', zorder=20))
    # ── Data (dia, mes, ano em ingles) ABAIXO da caixa: branco, maiusculo, sem negrito ──
    info_x = bb.x0 - pad_x + 0.002
    date_y = bb.y0 - pad_y - 0.012
    fig.text(info_x, date_y, data_full.upper(), color='white',
             fontsize=10, ha='left', va='top', weight='normal', family=ctx['font_legenda'], zorder=21)
    # ── Modelo + rodada (forecast) ou "REANALYSIS" (passado): ABAIXO da data ──
    fig.text(info_x, date_y - 0.026, ctx['rodada_label'], color='#dcdcdc',
             fontsize=9, ha='left', va='top', family=ctx['font_legenda'], zorder=21)

    # ── Subtitulo (canto superior DIREITO): subido p/ alinhar a quina, como o topo-esq ──
    fig.text(0.985, 0.978, ctx['subtitulo_dir'], color='#e6e6e6', fontsize=9.5,
             ha='right', va='top', family=ctx['font_legenda'], zorder=21)
    fig.text(0.985, 0.954, ctx['clim_ref'], color='#b8b8b8', fontsize=8.5,
             ha='right', va='top', family=ctx['font_legenda'], zorder=21)

    # ── Legenda: barra de gradiente FINA em caixa translucida arredondada (mais escura) ──
    lx0, ly0, lw, lh = 0.22, 0.072, 0.56, 0.074
    fig.add_artist(FancyBboxPatch(
        (lx0, ly0), lw, lh, boxstyle='round,pad=0,rounding_size=0.018',
        transform=fig.transFigure, facecolor='black', alpha=0.62,
        edgecolor='none', zorder=20))
    gl, gb, gw, gh = 0.245, 0.122, 0.51, 0.014
    gax = fig.add_axes([gl, gb, gw, gh])
    gax.set_zorder(21)
    gax.imshow(np.linspace(0.0, 1.0, 256).reshape(1, -1), aspect='auto', cmap=cmap,
               extent=[0, 1, 0, 1], origin='lower')
    gax.set_axis_off()
    labels = ctx['legenda5_labels']
    seg = gw / max(len(labels), 1)
    for i, lab in enumerate(labels):
        fig.text(gl + seg * (i + 0.5), gb - 0.012, str(lab).upper(), color='white',
                 fontsize=7.5, ha='center', va='top', family=ctx['font_legenda'], zorder=22)

    # ── Rodape: apenas o credito (dir.). O modelo/rodada subiu p/ baixo da data. ──
    fig.text(0.98, 0.028, ctx['credito'], color='#cfcfcf', fontsize=8.5,
             ha='right', va='center', family=FONT_SANS, zorder=21)


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
    idx_dia = min(int(round(pos)), n_dias - 1)
    data_en = ctx['dates_en'][idx_dia]
    data_full = ctx['dates_full'][idx_dia] if ctx.get('dates_full') else data_en
    guillaume = ctx.get('estilo') == 'guillaume'

    proj = _make_projection(float(ctx['lons'][f]), float(ctx['lats'][f]))
    fig = plt.figure(figsize=(8, 8), dpi=ctx['dpi'])
    fig.patch.set_facecolor('black')
    # Guillaume: globo grande (disco raio ~0.43, topo y~0.92), preenchendo o quadro
    # como na referencia; o canto sup-esq fica livre p/ a caixa compacta do nome/data
    # (no x=0.235 da caixa, o topo do globo cai p/ ~0.83, abaixo da data).
    rect = [0.07, 0.06, 0.86, 0.86] if guillaume else [0.03, 0.22, 0.94, 0.76]
    ax = fig.add_axes(rect, projection=proj)
    ax.patch.set_facecolor('black')
    ax.set_global()

    ax.contourf(lon_cyc, lat, campo, levels=levels, cmap=cmap, extend='both',
                transform=data_transform, zorder=2)
    # s39 (guillaume): contorno cinza levemente escuro; demais: preto.
    edge_color = '#444444' if guillaume else 'black'
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.7, edgecolor=edge_color, zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5, edgecolor=edge_color, zorder=5)
    estados = _state_line_geoms()
    if estados:
        ax.add_geometries(estados, data_transform, edgecolor=edge_color,
                          facecolor='none', linewidth=0.35, zorder=5)
    if not guillaume:  # s39 (guillaume): sem grade de lat/lon
        ax.gridlines(linewidth=0.3, color='white', alpha=0.2, zorder=4)
    ax.set_zorder(1)

    if guillaume:
        _overlay_guillaume(fig, ctx, cmap, data_full)
        fig.canvas.draw()
        arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return arr

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
                 output_dir: Path, fonte_label: str, script_id: str = 's38') -> Path:
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
    # Paleta POR VARIAVEL: settings GLOBO_3D_PALETA_<VAR> sobrescreve a da ficha
    # (ex.: GLOBO_3D_PALETA_TMP850_ANOM). Sem override, usa a paleta da ficha.
    override = settings.get(f'GLOBO_3D_PALETA_{variavel_key.upper()}', None)
    paleta = override or ficha['cmap_colors']
    # Nº de bandas POR VARIAVEL: settings GLOBO_3D_NIVEIS_<VAR> > ficha['niveis'] >
    # GLOBO_3D_NIVEIS global. Mais niveis = shaded mais suave (degrade mais fino).
    niveis = (settings.get(f'GLOBO_3D_NIVEIS_{variavel_key.upper()}', None)
              or ficha.get('niveis')
              or getattr(settings, 'GLOBO_3D_NIVEIS', 16))
    niveis = int(niveis)
    levels = np.linspace(vmin, vmax, niveis + 1)
    logger.info('Escala {}: shaded de {:+.1f} a {:+.1f} {} | {} bandas (passo {:.2f})',
                variavel_key, vmin, vmax, ficha.get('unidade', ''), niveis,
                (vmax - vmin) / niveis)

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

    # Estilo de layout: s39 -> 'guillaume' (caixa do nome + barra de gradiente); demais -> WaPo.
    estilo = 'guillaume' if script_id == 's39' else 'wapo'
    # Rotulo de rodada p/ o rodape (so no forecast; reanalise nao tem rodada).
    run_init = anom.attrs.get('run_init')
    if run_init:
        _ri = datetime.strptime(run_init, '%Y-%m-%d %H')
        rodada_label = f'{fonte_label.upper()}  ·  {_ri:%d/%m/%Y} run {_ri:%H}Z'
    else:
        rodada_label = fonte_label.upper()
    # Nome da variavel na caixa (override: GLOBO_3D_ROTULO_<VAR>) e 5 rotulos da legenda (ingles).
    titulo_box = str(settings.get(f'GLOBO_3D_ROTULO_{variavel_key.upper()}',
                                  ficha.get('rotulo_box', ficha['titulo_en'])))
    legenda5 = list(settings.get(
        'GLOBO_3D_LEGENDA5',
        ['Well below', 'Below', 'Average', 'Above', 'Well above'],
    ))
    # Canto sup-direito: variavel + nivel e periodo de climatologia (default 1991-2020).
    subtitulo_dir = str(settings.get(f'GLOBO_3D_SUBTITULO_{variavel_key.upper()}',
                                     ficha.get('subtitulo_dir', ficha['titulo_en'])))
    clim_ref = f"Relative to the {settings.get('GLOBO_3D_CLIM_REF', '1991-2020')} normal"

    ctx = {
        'vals_cyc': vals_cyc.astype(np.float32),
        'lon_cyc': lon_cyc, 'lat': lat, 'levels': levels,
        'paleta': list(paleta),
        'lons': lons, 'lats': lats,
        'frames_por_dia': frames_por_dia, 'vel_var': vel_var, 'n_dias': n_dias,
        'dates_en': [d.strftime('%B %-d') for d in dates],
        'dates_full': [d.strftime('%A %-d %B %Y') for d in dates],  # dia/mes/ano (ingles, s39)
        'titulo_en': ficha.get('titulo_en', ficha['titulo']),
        'fonte_label': fonte_label,
        'credito': str(getattr(settings, 'GLOBO_3D_CREDITO', 'Bruno Capucin')).upper(),
        'font_titulo': font_titulo, 'font_legenda': font_legenda,
        'usar_vinheta': bool(getattr(settings, 'GLOBO_3D_VINHETA', True)),
        'legenda_labels': ['Well below', 'Below', 'Above', 'Well above'],
        'estilo': estilo,
        'titulo_box': titulo_box,
        'subtitulo_dir': subtitulo_dir,
        'clim_ref': clim_ref,
        'legenda5_labels': legenda5,
        'rodada_label': rodada_label,
        'dpi': dpi,
    }

    logger.info('{} dias -> {} frames | {} fps | {}px | {} workers | {}',
                n_dias, total_frames, fps, px, workers, fonte_label)

    output_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = output_dir / f'{script_id}_{variavel_key}.mp4'
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
def gerar_animacao(variaveis: list[str], output_base: Path, script_id: str = 's38') -> list[Path]:
    """Gera os MP4 do globo animado para uma LISTA de variaveis.

    O MODO e decidido PELAS DATAS (DATA_INICIAL/DATA_FINAL): passado=reanalise,
    futuro=previsao, janela que cruza hoje=observado+previsao emendados. Retorna os
    caminhos gerados (1 MP4 por variavel; quando ha previsao, x cada modelo habilitado).
    O `script_id` (ex.: 's38', 's39') prefixa o nome do MP4 gerado.
    """
    plano, dt_ini, dt_fim = _output_plan(variaveis, output_base)
    logger.info('=' * 70)
    logger.info('GLOBO 3D MIDIA: {} a {} | {} clipe(s)', dt_ini.date(), dt_fim.date(), len(plano))
    logger.info('=' * 70)

    outputs: list[Path] = []
    for item in plano:
        ficha = VARIAVEIS[item['var']]
        logger.info('--- {} | {} ({}) ---',
                    item['label'], item['var'], ficha['titulo'])
        serie = _build_var_series(ficha, item['model'], dt_ini, dt_fim)
        outputs.append(_render_clip(serie, ficha, item['var'], item['dir'], item['label'], script_id))
    return outputs
