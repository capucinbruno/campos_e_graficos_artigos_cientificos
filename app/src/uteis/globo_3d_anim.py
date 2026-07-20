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
import copy
import itertools
import os
import re
import textwrap
import time as _time
from collections import deque
from datetime import datetime, timedelta
from multiprocessing import get_context
from pathlib import Path

# Bibliotecas de terceiros
import matplotlib

matplotlib.use('Agg')  # backend sem display: necessario para grab de buffer

from scipy.interpolate import CubicSpline as _CubicSpline
from scipy.ndimage import distance_transform_edt, gaussian_filter as _gaussian_filter, zoom as _zoom

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio.v2 as imageio
import numpy as np
from PIL import Image as _PIL_Image
import pandas as pd
import xarray as xr
from cartopy.util import add_cyclic_point
from matplotlib import patheffects as path_effects
from matplotlib import pyplot as plt
from matplotlib.colors import (
    BoundaryNorm, LinearSegmentedColormap, ListedColormap, hsv_to_rgb, rgb_to_hsv, to_rgba,
)
from matplotlib.patches import FancyBboxPatch, Rectangle

# Modulos locais
from app.shared.logger import get_logger
from app.src.uteis.forecast_daily import (
    daily_mslp_on_grid as _daily_mslp_on_grid,
    daily_scalar_on_grid as _daily_scalar_on_grid,
    daily_uv200_on_grid as _daily_uv200_on_grid,
    daily_wind_speed_on_grid as _daily_wind_speed_on_grid,
    lagged_ensemble_mean as _lagged_ensemble_mean,
    native_scalar_on_grid as _native_scalar_on_grid,
    resolve_forecast_lead_init as _resolve_forecast_lead_init,
    synoptic_scalar_on_grid as _synoptic_scalar_on_grid,
)
from app.shared.settings_factory import settings
from app.src.uteis.clim_diaria_uv200_ltm import (
    clim_hgt250_daily,
    clim_hgt500_daily,
    clim_t850_daily,
    clim_u250_daily,
    clim_u850_daily,
    clim_uv200_daily,
    clim_v850_daily,
)
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
OLR_VARS = ('olr', 'ulwrf', 'ttr', 'ULWRF', 'TTR')
# Componentes do vento (arquivos uv250/uv850 trazem u E v; o scalar reader pega a 1a candidata).
U_VARS = ('u', 'u_component_of_wind', 'U_GRD_L100', 'uwnd', 'ugrd')
V_VARS = ('v', 'v_component_of_wind', 'V_GRD_L100', 'vwnd', 'vgrd')

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


def _dias_janela(d0, d) -> int:
    """Dias de CALENDARIO decorridos entre `d0` (1o passo da serie) e `d` (passo atual), contando o
    1o dia (`d0`=`d` -> 1). Usado pela JANELA ACUMULADA (`_acum_rotulo`) em vez do indice `i` do
    passo -- com `ACUM_HORARIO` a serie tem varios passos NATIVOS (3h/6h) por dia de calendario, e
    contar `i+1` como "dias" inflava a janela (33 passos virava "33 dias" no rotulo/voo de camera
    quando a janela real era de so 5 dias). `.date()` descarta a hora antes de subtrair -- senao um
    passo no MESMO dia calendario que `d0` mas com hora menor arredondaria a janela pra baixo."""
    return (pd.Timestamp(d).date() - pd.Timestamp(d0).date()).days + 1


def _fmt_data_pentada(d0, dias: int, *, com_ano: bool, mostrar_hora: bool = False) -> str:
    """Rotulo de DATA da pentada movel: intervalo [d0, d0+dias-1] em ingles US.

    Ex.: 'July 20–24, 2026' (mesmo mes) ou 'July 29 – August 2, 2026' (cruza o mes).
    `com_ano=False` omite o ano (rotulo curto). dias<=1 cai p/ data unica.
    `mostrar_hora=True` (so em dias<=1, usado pelo MP4 sinotico) acrescenta a hora UTC
    (ex.: 'June 5, 2026 12Z') -- sem isso, os 4 passos sinoticos do mesmo dia teriam o
    MESMO rotulo, sem indicar qual horario esta sendo mostrado.
    """
    d0 = pd.Timestamp(d0)
    if dias <= 1:
        base = d0.strftime('%B %-d, %Y') if com_ano else d0.strftime('%B %-d')
        return f'{base} {d0.hour:02d}Z' if mostrar_hora else base
    d1 = d0 + pd.Timedelta(days=dias - 1)
    ano = f', {d1.year}' if com_ano else ''
    if d0.month == d1.month:
        return f"{d0.strftime('%B %-d')}–{d1.strftime('%-d')}{ano}"  # July 20–24, 2026
    return f"{d0.strftime('%B %-d')} – {d1.strftime('%B %-d')}{ano}"  # July 29 – August 2, 2026


def _fmt_data_br(d0, dias: int, *, mostrar_hora: bool = False, d1: 'pd.Timestamp | datetime | None' = None) -> str:
    """Rotulo de DATA no formato INGLES ABREVIADO, usado na caixa "The Weather Channel" do s42/s46.
    Espelha `_fmt_data_pentada`: dias<=1 -> data/hora sinotica unica; dias>1 -> intervalo da
    pentada/media do periodo (ou da JANELA ACUMULADA horaria, ver `d1`).

    `d1` (opcional): timestamp REAL do fim da janela, com a HORA verdadeira do passo -- usado pelo
    `ACUM_HORARIO` (precip_abs) para o rotulo virar 'Jul 18–19 03Z' (1a data trava, 2a data + hora
    andam a cada passo nativo) em vez de 'Jul 18–19' sem hora. Sem `d1`, mantem o calculo antigo
    (d0 + dias-1, sem hora no lado direito) -- usado pela pentada movel (dias = janela fixa).

    Ex.: 'Jun 25 00Z' (sinotico) ou 'Jul 5–10' (pentada, mesmo mes) ou 'Jul 18–19 03Z' (acum horario).
    """
    d0 = pd.Timestamp(d0)
    if dias <= 1:
        # `d1` (quando vem do ACUM_HORARIO) carrega a hora REAL do passo atual -- sem isso, os
        # varios passos nativos do 1o dia (ex.: 00Z, 03Z, 06Z...) cairiam todos aqui com a MESMA
        # hora congelada em `d0` (o 1o passo da serie), em vez de andar passo a passo.
        _hora_ref = pd.Timestamp(d1) if d1 is not None else d0
        base = d0.strftime('%b %-d')
        return f'{base} {_hora_ref.hour:02d}Z' if mostrar_hora else base
    d1 = pd.Timestamp(d1) if d1 is not None else d0 + pd.Timedelta(days=dias - 1)
    _hz = f' {d1.hour:02d}Z' if mostrar_hora else ''
    if d0.month == d1.month and d0.year == d1.year:
        return f"{d0.strftime('%b')} {d0.day}–{d1.day}{_hz}"
    if d0.year == d1.year:
        return f"{d0.strftime('%b %-d')} – {d1.strftime('%b %-d')}{_hz}"
    return f"{d0.strftime('%b %-d, %Y')} – {d1.strftime('%b %-d, %Y')}{_hz}"


# Modelos de forecast suportados (cada um com pelo menos 1 downloader registrado abaixo em
# _fcst_downloader). ecmwf_ens/aifs_ens/aigfs/aigefs/cfs: mesma assinatura generica (init,
# lead_hours, hours) dos demais -- ja usada por outros scripts do projeto, so faltava o registro
# aqui para o motor do globo (s38-s43) tambem os aceitar.
_FCST_FLAGS = {
    'gfs': 'RUN_GFS', 'gefs': 'RUN_GEFS', 'ecmwf': 'RUN_ECMWF', 'aifs': 'RUN_AIFS',
    'ecmwf_ens': 'RUN_ECMWF_ENS', 'aifs_ens': 'RUN_AIFS_ENS',
    'aigfs': 'RUN_AIGFS', 'aigefs': 'RUN_AIGEFS', 'cfs': 'RUN_CFS',
}


def _enabled_forecast_models() -> list[str]:
    """Modelos habilitados via flags (default: so GFS, como no s34)."""
    return [m for m, flag in _FCST_FLAGS.items() if bool(settings.get(flag, m == 'gfs'))]


# Candidatos de nome da variavel de temperatura nos arquivos (ERA5/GDAS/modelos).
TMP_VARS = ('t', 'tmp', 'air', 'temperature')


def _fcst_downloader(model: str, kind: str):
    """Resolve o downloader de forecast (import tardio) por modelo e variavel.

    kind: 'hgt250' (Z250), 'tmp850' (T 850 hPa), 'uv250' (u/v 250 hPa), 'uv850' (u/v 850 hPa)
    ou 'olr'.
    """
    table = {
        ('gfs', 'hgt250'): ('downloaders_gfs_hgt250', 'ensure_gfs_hgt250_fcst_for_period'),
        ('gefs', 'hgt250'): ('downloaders_gefs_hgt250', 'ensure_gefs_hgt250_fcst_for_period'),
        ('ecmwf', 'hgt250'): ('downloaders_ecmwf_hgt250', 'ensure_ecmwf_hgt250_fcst_for_period'),
        ('aifs', 'hgt250'): ('downloaders_aifs_hgt250', 'ensure_aifs_hgt250_fcst_for_period'),
        ('gfs', 'hgt500'): ('downloaders_gfs_hgt500', 'ensure_gfs_hgt500_fcst_for_period'),
        ('gefs', 'hgt500'): ('downloaders_gefs_hgt500', 'ensure_gefs_hgt500_fcst_for_period'),
        ('ecmwf', 'hgt500'): ('downloaders_ecmwf_hgt500', 'ensure_ecmwf_hgt500_fcst_for_period'),
        ('aifs', 'hgt500'): ('downloaders_aifs_hgt500', 'ensure_aifs_hgt500_fcst_for_period'),
        ('gfs', 'tmp850'): ('downloaders_gfs_tmp850', 'ensure_gfs_tmp850_fcst_for_period'),
        ('gefs', 'tmp850'): ('downloaders_gefs_tmp850', 'ensure_gefs_tmp850_fcst_for_period'),
        ('ecmwf', 'tmp850'): ('downloaders_ecmwf_tmp850', 'ensure_ecmwf_tmp850_fcst_for_period'),
        ('aifs', 'tmp850'): ('downloaders_aifs_tmp850', 'ensure_aifs_tmp850_fcst_for_period'),
        ('gfs', 'uv250'): ('downloaders_gfs_uv250', 'ensure_gfs_uv250_fcst_for_period'),
        ('gefs', 'uv250'): ('downloaders_gefs_uv250', 'ensure_gefs_uv250_fcst_for_period'),
        ('ecmwf', 'uv250'): ('downloaders_ecmwf_uv250', 'ensure_ecmwf_uv250_fcst_for_period'),
        ('aifs', 'uv250'): ('downloaders_aifs_uv250', 'ensure_aifs_uv250_fcst_for_period'),
        ('gfs', 'uv850'): ('downloaders_gfs_uv850', 'ensure_gfs_uv850_fcst_for_period'),
        ('gefs', 'uv850'): ('downloaders_gefs_uv850', 'ensure_gefs_uv850_fcst_for_period'),
        ('ecmwf', 'uv850'): ('downloaders_ecmwf_uv850', 'ensure_ecmwf_uv850_fcst_for_period'),
        ('aifs', 'uv850'): ('downloaders_aifs_uv850', 'ensure_aifs_uv850_fcst_for_period'),
        ('gfs', 'olr'): ('downloaders_gfs_olr', 'ensure_gfs_olr_fcst_for_period'),
        ('gefs', 'olr'): ('downloaders_gefs_olr', 'ensure_gefs_olr_fcst_for_period'),
        ('ecmwf', 'olr'): ('downloaders_ecmwf_olr', 'ensure_ecmwf_olr_fcst_for_period'),
        # Agua precipitavel (PWAT) -- so nos modelos FISICOS (GFS/GEFS/ECMWF/ECMWF_ENS/CFS);
        # os modelos de IA (AIFS/AIFS_ENS/AIGFS/AIGEFS) nao publicam esse campo.
        ('gfs', 'pwat'): ('downloaders_gfs_pwat', 'ensure_gfs_pwat_fcst_for_period'),
        ('gefs', 'pwat'): ('downloaders_gefs_pwat', 'ensure_gefs_pwat_fcst_for_period'),
        ('ecmwf', 'pwat'): ('downloaders_ecmwf_pwat', 'ensure_ecmwf_pwat_fcst_for_period'),
        ('ecmwf_ens', 'pwat'): ('downloaders_ecmwf_ens', 'ensure_ecmwf_ens_pwat_fcst_for_period'),
        ('cfs', 'pwat'): ('downloaders_cfs_ensemble', 'ensure_cfs_pwat_for_period'),
        # CHUVA (precipitacao ACUMULADA DIARIA, mm). NAO e PWAT: e a chuva que caiu (APCP/tp). Saida ja
        # e' diaria (1 valor/dia) -> daily_scalar_on_grid faz media diaria = o proprio total (preserva o
        # acumulado). Forecast-only (sem reanalise). Duas convencoes de fonte, ambas normalizadas p/ mm:
        #   tp acumulado desde o init, em metros (ECMWF Open Data: HRES e AIFS) -> diferenca entre 00Z
        #   APCP em baldes de 6 h que resetam (NOMADS: GFS, GEFS, AIGFS, AIGEFS) -> soma dos 4 do dia
        # FALTAM (sem downloader): ecmwf_ens/aifs_ens (exigem baixar e mediar 50 membros) e cfs (PRATE,
        # taxa em vez de acumulado).
        ('gfs', 'precip'): ('downloaders_gfs_precip', 'ensure_gfs_precip_fcst_for_period'),
        ('ecmwf', 'precip'): ('downloaders_ecmwf_precip', 'ensure_ecmwf_precip_fcst_for_period'),
        ('gefs', 'precip'): ('downloaders_gefs_precip', 'ensure_gefs_precip_fcst_for_period'),
        ('aifs', 'precip'): ('downloaders_aifs_precip', 'ensure_aifs_precip_fcst_for_period'),
        ('aigfs', 'precip'): ('downloaders_ai_nomads_precip', 'ensure_aigfs_precip_fcst_for_period'),
        ('aigefs', 'precip'): ('downloaders_ai_nomads_precip', 'ensure_aigefs_precip_fcst_for_period'),
        # RAJADA de vento (10fg, max diario, km/h) -- so ECMWF-HRES (unica fonte com 10fg no s47).
        ('ecmwf', 'gust'): ('downloaders_ecmwf_gust', 'ensure_ecmwf_gust_fcst_for_period'),
        # u/v 200 hPa (PSI200) — um arquivo diario com u, v e hgt em 200 hPa por modelo.
        ('gfs', 'fcst200'): ('downloaders_gfs_fcst200', 'ensure_gfs_fcst200_for_period'),
        ('gefs', 'fcst200'): ('downloaders_gefs_fcst200', 'ensure_gefs_fcst200_for_period'),
        ('ecmwf', 'fcst200'): ('downloaders_ecmwf_fcst200', 'ensure_ecmwf_fcst200_for_period'),
        ('aifs', 'fcst200'): ('downloaders_aifs_fcst200', 'ensure_aifs_fcst200_for_period'),

        # ECMWF ENS (50 membros, media do ensemble) — todas as variaveis do s38-s43.
        ('ecmwf_ens', 'hgt250'): ('downloaders_ecmwf_ens', 'ensure_ecmwf_ens_hgt250_fcst_for_period'),
        ('ecmwf_ens', 'hgt500'): ('downloaders_ecmwf_ens', 'ensure_ecmwf_ens_hgt500_fcst_for_period'),
        ('ecmwf_ens', 'tmp850'): ('downloaders_ecmwf_ens', 'ensure_ecmwf_ens_tmp850_fcst_for_period'),
        ('ecmwf_ens', 'uv250'): ('downloaders_ecmwf_ens', 'ensure_ecmwf_ens_uv250_fcst_for_period'),
        ('ecmwf_ens', 'uv850'): ('downloaders_ecmwf_ens', 'ensure_ecmwf_ens_uv850_fcst_for_period'),
        ('ecmwf_ens', 'olr'): ('downloaders_ecmwf_ens', 'ensure_ecmwf_ens_olr_fcst_for_period'),
        ('ecmwf_ens', 'fcst200'): ('downloaders_ecmwf_ens', 'ensure_ecmwf_ens_fcst200_for_period'),

        # AIFS ENS (50 membros; SEM OLR).
        ('aifs_ens', 'hgt250'): ('downloaders_aifs_ens', 'ensure_aifs_ens_hgt250_fcst_for_period'),
        ('aifs_ens', 'hgt500'): ('downloaders_aifs_ens', 'ensure_aifs_ens_hgt500_fcst_for_period'),
        ('aifs_ens', 'tmp850'): ('downloaders_aifs_ens', 'ensure_aifs_ens_tmp850_fcst_for_period'),
        ('aifs_ens', 'uv250'): ('downloaders_aifs_ens', 'ensure_aifs_ens_uv250_fcst_for_period'),
        ('aifs_ens', 'uv850'): ('downloaders_aifs_ens', 'ensure_aifs_ens_uv850_fcst_for_period'),
        ('aifs_ens', 'fcst200'): ('downloaders_aifs_ens', 'ensure_aifs_ens_fcst200_for_period'),

        # AIGFS (IA deterministica NCEP, 16d; SEM OLR).
        ('aigfs', 'hgt250'): ('downloaders_ai_nomads', 'ensure_aigfs_hgt250_fcst_for_period'),
        ('aigfs', 'hgt500'): ('downloaders_ai_nomads', 'ensure_aigfs_hgt500_fcst_for_period'),
        ('aigfs', 'tmp850'): ('downloaders_ai_nomads', 'ensure_aigfs_tmp850_fcst_for_period'),
        ('aigfs', 'uv250'): ('downloaders_ai_nomads', 'ensure_aigfs_uv250_fcst_for_period'),
        ('aigfs', 'uv850'): ('downloaders_ai_nomads', 'ensure_aigfs_uv850_fcst_for_period'),
        ('aigfs', 'fcst200'): ('downloaders_ai_nomads', 'ensure_aigfs_fcst200_for_period'),

        # AIGEFS (IA ensemble NCEP, media pronta, 16d; SEM OLR).
        ('aigefs', 'hgt250'): ('downloaders_ai_nomads', 'ensure_aigefs_hgt250_fcst_for_period'),
        ('aigefs', 'hgt500'): ('downloaders_ai_nomads', 'ensure_aigefs_hgt500_fcst_for_period'),
        ('aigefs', 'tmp850'): ('downloaders_ai_nomads', 'ensure_aigefs_tmp850_fcst_for_period'),
        ('aigefs', 'uv250'): ('downloaders_ai_nomads', 'ensure_aigefs_uv250_fcst_for_period'),
        ('aigefs', 'uv850'): ('downloaders_ai_nomads', 'ensure_aigefs_uv850_fcst_for_period'),
        ('aigefs', 'fcst200'): ('downloaders_ai_nomads', 'ensure_aigefs_fcst200_for_period'),

        # CFS (pseudo-ensemble subsazonal ~45d; SEM hgt250 nem tmp850 -- so z500/uv/olr/fcst200).
        ('cfs', 'hgt500'): ('downloaders_cfs_ensemble', 'ensure_cfs_hgt500_for_period'),
        ('cfs', 'uv250'): ('downloaders_cfs_ensemble', 'ensure_cfs_uv250_for_period'),
        ('cfs', 'uv850'): ('downloaders_cfs_ensemble', 'ensure_cfs_uv850_for_period'),
        ('cfs', 'olr'): ('downloaders_cfs_ensemble', 'ensure_cfs_olr_for_period'),
        ('cfs', 'fcst200'): ('downloaders_cfs_ensemble', 'ensure_cfs_fcst200_for_period'),
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


def _era5_z500(start, end, force):
    from app.src.uteis.downloaders_hgt500_ERA5 import (
        ensure_era5_altura_geopotencial_500_global_for_period_grib as fn)
    return fn(start=start, end=end, hours_utc=list(DEFAULT_SYNOPTIC_HOURS),
              force_redownload=force, convert_to_height_netcdf=True)


def _gdas_z500(start, end, force):
    from app.src.uteis.downloaders_gdas_hgt500 import ensure_gdas_hgt500_for_period as fn
    return fn(start=start, end=end, force_redownload=force)


def _era5_t850(start, end, force):
    from app.src.uteis.downloaders_tmp850_ERA5 import ensure_era5_t850_for_period as fn
    return fn(start=start, end=end, hours_utc=list(DEFAULT_SYNOPTIC_HOURS), force_redownload=force)


def _gdas_t850(start, end, force):
    from app.src.uteis.downloaders_gdas_tmp850 import ensure_gdas_tmp850_for_period as fn
    return fn(start=start, end=end, force_redownload=force)


def _era5_pwat(start, end, force):
    from app.src.uteis.downloaders_era5_pwat import ensure_era5_pwat_for_period as fn
    return fn(start=start, end=end, hours_utc=list(DEFAULT_SYNOPTIC_HOURS), force_redownload=force)


def _gdas_pwat(start, end, force):
    from app.src.uteis.downloaders_gdas_pwat import ensure_gdas_pwat_for_period as fn
    return fn(start=start, end=end, force_redownload=force)


def _precip_reanalise_unsupported(start, end, force):
    """Chuva (precip_abs) e' FORECAST-ONLY -- nao ha downloader de reanalise. Levanta erro claro se
    o periodo cair no passado (reanalise/emenda)."""
    raise RuntimeError(
        'Chuva (precip_abs) e FORECAST-ONLY: nao existe downloader de reanalise para precipitacao. '
        'Use DATA_INICIAL/DATA_FINAL no FUTURO (previsao GFS/ECMWF-HRES) e habilite RUN_GFS/RUN_ECMWF.')


def _gust_reanalise_unsupported(start, end, force):
    """Rajada (rajada_abs) e' FORECAST-ONLY -- so ECMWF-HRES tem downloader. Erro claro se o periodo
    cair no passado (reanalise/emenda)."""
    raise RuntimeError(
        'Rajada (rajada_abs) e FORECAST-ONLY: nao existe downloader de reanalise para rajada de vento. '
        'Use DATA_INICIAL/DATA_FINAL no FUTURO (previsao ECMWF-HRES) e habilite RUN_ECMWF.')


def _era5_uv250(start, end, force):
    from app.src.uteis.downloaders_wind250 import ensure_era5_uv250_for_period as fn
    return fn(start=start, end=end, hours_utc=list(DEFAULT_SYNOPTIC_HOURS), force_redownload=force)


def _gdas_uv250(start, end, force):
    from app.src.uteis.downloaders_gdas_uv250 import ensure_gdas_uv250_for_period as fn
    return fn(start=start, end=end, hours=list(DEFAULT_SYNOPTIC_HOURS), force_redownload=force)


def _era5_uv850(start, end, force):
    from app.src.uteis.downloaders_wind850 import ensure_era5_uv850_for_period as fn
    return fn(start=start, end=end, hours_utc=list(DEFAULT_SYNOPTIC_HOURS), force_redownload=force)


def _gdas_uv850(start, end, force):
    from app.src.uteis.downloaders_gdas_uv850 import ensure_gdas_uv850_for_period as fn
    return fn(start=start, end=end, hours=list(DEFAULT_SYNOPTIC_HOURS), force_redownload=force)


def _era5_uv200(start, end, force):
    from app.src.uteis.downloaders_wind200 import ensure_era5_uv200_for_period as fn
    return fn(start=start, end=end, hours_utc=list(DEFAULT_SYNOPTIC_HOURS), force_redownload=force)


def _gdas_uv200(start, end, force):
    from app.src.uteis.downloaders_gdas_uv200 import ensure_gdas_uv200_for_period as fn
    return fn(start=start, end=end, hours=list(DEFAULT_SYNOPTIC_HOURS), force_redownload=force)


def _olr_reanalise_series(dt_ini: datetime, dt_fim: datetime) -> xr.DataArray | None:
    """Anomalia OLR observada (CPC Blended PSL) na grade nativa 2.5° — sem interpolacao.

    Manter a grade nativa evita suavizacao excessiva (bilinear 2.5->0.5) e elimina
    a faixa NaN em 358-360° que causava o seam em Greenwich (o add_cyclic_point do
    _render_clip adiciona o ponto 360° e fecha o globo corretamente).
    """
    from app.src.uteis.clim_diaria_olr import clim_olr_daily_for_anim, _open_olr
    if dt_ini.date() > dt_fim.date():
        return None
    dates = pd.date_range(dt_ini.date(), dt_fim.date(), freq='D').values
    da_olr = _open_olr()
    slices = [
        da_olr.sel(time=np.datetime64(pd.Timestamp(d).date()), method='nearest').values
        for d in dates
    ]
    obs_arr = np.stack(slices, axis=0)
    daily = xr.DataArray(
        obs_arr, dims=['time', 'lat', 'lon'],
        coords={'time': dates, 'lat': da_olr['lat'].values, 'lon': da_olr['lon'].values},
    ).sortby('lat')
    return _anom_from_clim(daily, clim_olr_daily_for_anim, celsius=False,
                           nome='olr_anom', unidade='W/m²')


def _tsm_reanalise_series(dt_ini: datetime, dt_fim: datetime) -> xr.DataArray | None:
    """Anomalia de TSM observada (OISSTv2) — SST diaria menos a climatologia diaria OISST
    (LTM 1991-2020). SO reanalise (sem previsao). Grade ~0.5° (OISST 0.25° subamostrada).

    Mantém a grade do OISST (sem regrid p/ a grade-alvo do globo): o add_cyclic_point do
    _render_clip fecha o globo em Greenwich. Continentes ficam NaN (mascarados no globo).
    """
    from app.src.uteis.clim_diaria_sst import clim_sst_daily_for_anim, sst_obs_daily
    if dt_ini.date() > dt_fim.date():
        return None
    daily = sst_obs_daily(dt_ini, dt_fim)
    if daily.sizes.get('time', 0) == 0:
        return None
    return _anom_from_clim(daily, clim_sst_daily_for_anim, celsius=False,
                           nome='tsm_anom', unidade='°C')


def _tsm_abs_reanalise_series(dt_ini: datetime, dt_fim: datetime) -> xr.DataArray | None:
    """TSM ABSOLUTA observada (OISSTv2) em °C — SST diaria SEM subtrair climatologia.
    SO reanalise (sem previsao). Grade ~0.5° (OISST 0.25° subamostrada), land=NaN."""
    from app.src.uteis.clim_diaria_sst import sst_obs_daily
    if dt_ini.date() > dt_fim.date():
        return None
    daily = sst_obs_daily(dt_ini, dt_fim)
    if daily.sizes.get('time', 0) == 0:
        return None
    daily.name = 'tsm_abs'
    daily.attrs['unidade'] = '°C'
    return daily


def _tsm_forecast_series(model: str, dt_ini: datetime, dt_fim: datetime) -> xr.DataArray | None:
    """tsm_anom e OBSERVACIONAL (OISST) — nao ha produto de previsao de SST wired no projeto.

    Falha com mensagem clara em janelas futuras (em vez do erro cru de downloader ausente).
    """
    raise RuntimeError(
        'tsm_anom (TSM/OISST) e observacional: o projeto nao tem downloader de SST prevista, '
        f'entao {model.upper()} nao pode gerar a parte de previsao [{dt_ini.date()} a {dt_fim.date()}]. '
        'Use tsm_anom so com DATA_FINAL passada (reanalise) ou remova-o de VARIAVEIS_GLOBO_3D no forecast.'
    )


# ---------------------------------------------------------------------------
# PSI200 / CHI200 — campos DERIVADOS do vento u/v 200 (inversao de Poisson).
# Ambos sao LINEARES no vento, entao a anomalia e o campo do vento de ANOMALIA
# (u-clim, v-clim) — mesma metodologia do s04 (psi) / s03 (chi). Calculado na grade
# 2.5° da LTM (campos suaves/grande-escala; a inversao por dia fica barata):
#   campo='psi': Poisson da VORTICIDADE  -> funcao de corrente (10^6 m²/s)
#   campo='chi': Poisson da DIVERGENCIA  -> potencial de velocidade (10^5 m²/s)
# ---------------------------------------------------------------------------
# Forcante (u,v -> escalar) e escala de saida por campo. Imports tardios p/ nao
# arrastar os downloaders de plot_psi200/plot_chi200 no import do motor.
_UV200_DERIVADO = {
    'psi': {'nome': 'psi200_anom', 'escala': 1e6, 'unidade': '10⁶ m² s⁻¹',
            'mod': 'plot_psi200', 'forcante': '_compute_vorticity'},
    'chi': {'nome': 'chi200_anom', 'escala': 1e5, 'unidade': '10⁵ m² s⁻¹',
            'mod': 'plot_chi200', 'forcante': '_compute_divergence'},
}


def _uv200_derived_anom_series(u_da: xr.DataArray, v_da: xr.DataArray, *, campo: str) -> xr.DataArray:
    """Anomalia diaria em 200 hPa das series u/v 200:
      - campo='psi'          -> PSI200 (funcao de corrente, inverte Poisson da vorticidade)
      - campo='chi'          -> CHI200 (potencial de velocidade, inverte Poisson da divergencia)
      - campo='u_zonal'      -> VENTO ZONAL (u - climatologia de u, SEM Poisson)
      - campo='v_meridional' -> VENTO MERIDIONAL (v - climatologia de v, SEM Poisson)

    Subtrai a climatologia diaria de u/v (LTM NCEP 1991-2020, mesma grade 2.5°).
    """
    if campo in ('u_zonal', 'v_meridional'):
        # Anomalia de VENTO ZONAL/MERIDIONAL em 200 hPa: componente - climatologia (direto, sem Poisson).
        u_clim, v_clim, _clat, _clon = clim_uv200_daily(u_da['time'].values)
        if campo == 'u_zonal':
            vals, nome, comp = (u_da.values - u_clim), 'wnd200_zonal_anom', u_da
        else:
            vals, nome, comp = (v_da.values - v_clim), 'wnd200_meridional_anom', v_da
        anom = xr.DataArray(
            vals.astype(np.float32), dims=['time', 'lat', 'lon'],
            coords={'time': comp['time'].values, 'lat': comp['lat'].values, 'lon': comp['lon'].values},
        ).sortby('lat')
        anom.name = nome
        anom.attrs['units'] = 'ms⁻¹'
        logger.info('Anomalia {}: {} dias | min={:.1f} max={:.1f} ms⁻¹',
                    nome, anom.sizes['time'], float(anom.min()), float(anom.max()))
        return anom
    import importlib
    cfg = _UV200_DERIVADO[campo]
    mod = importlib.import_module(f'app.src.uteis.{cfg["mod"]}')
    _forcante = getattr(mod, cfg['forcante'])          # _compute_vorticity | _compute_divergence
    _solve_poisson_sphere = getattr(mod, '_solve_poisson_sphere')
    escala, nome, unidade = cfg['escala'], cfg['nome'], cfg['unidade']

    u_clim, v_clim, _clat, _clon = clim_uv200_daily(u_da['time'].values)
    u_anom = (u_da.values - u_clim).astype(np.float64)
    v_anom = (v_da.values - v_clim).astype(np.float64)
    lat = u_da['lat'].values
    lon = u_da['lon'].values
    # O solver de Poisson (s03/s04) espera lat N->S; inverte se a grade da LTM e ascendente.
    if lat[0] < lat[-1]:
        lat_s = lat[::-1]
        u_anom = u_anom[:, ::-1, :]
        v_anom = v_anom[:, ::-1, :]
    else:
        lat_s = lat
    n = u_anom.shape[0]
    out = np.empty((n, lat_s.size, lon.size), dtype=np.float32)
    # Pesos por area (cos da latitude) p/ remover a media de cada dia: psi/chi tem liberdade
    # de CALIBRE (constante aditiva fixada pela BC=0 nos polos), so os gradientes (vento) sao
    # fisicos. Centrar em 0 deixa o colormap simetrico coerente e evita falso "dia degenerado".
    w2d = np.broadcast_to(np.cos(np.deg2rad(lat_s))[:, None], (lat_s.size, lon.size))
    for k in range(n):
        forc = _forcante(u_anom[k], v_anom[k], lat_s, lon)
        pk = _solve_poisson_sphere(forc, lat_s, lon) / escala
        pk = pk - np.average(pk, weights=w2d)
        out[k] = pk.astype(np.float32)
    anom = xr.DataArray(
        out, dims=['time', 'lat', 'lon'],
        coords={'time': u_da['time'].values, 'lat': lat_s, 'lon': lon},
    ).sortby('lat')
    anom.name = nome
    anom.attrs['units'] = unidade
    logger.info('Anomalia {}: {} dias | min={:.1f} max={:.1f} {}',
                nome, anom.sizes['time'], float(anom.min()), float(anom.max()), unidade)
    return anom


def _uv200_reanalise_series(dt_ini: datetime, dt_fim: datetime, *, campo: str) -> xr.DataArray | None:
    """Anomalia PSI/CHI200 observada (ERA5/CDS + GDAS/NOMADS) na janela [dt_ini, dt_fim]."""
    if dt_ini.date() > dt_fim.date():
        return None
    force = bool(getattr(settings, 'FORCE_DOWNLOAD', False))
    era5_period, gdas_period = _get_data_sources(dt_ini, dt_fim)
    files: list[Path] = []
    if era5_period:
        logger.info('Download ERA5 uv200: {} -> {}', era5_period[0].date(), era5_period[1].date())
        files += _era5_uv200(era5_period[0], era5_period[1], force)
    if gdas_period:
        logger.info('Download GDAS uv200: {} -> {}', gdas_period[0].date(), gdas_period[1].date())
        files += _gdas_uv200(gdas_period[0], gdas_period[1], force)
    # Grade da LTM (2.5°): traz o vento p/ a mesma grade da climatologia (subtracao direta).
    _, _, clat, clon = clim_uv200_daily(np.array([np.datetime64(dt_ini.date())]))
    u_da, v_da = _daily_uv200_on_grid(files, dt_ini, dt_fim, clat, clon, logger)
    return _uv200_derived_anom_series(u_da, v_da, campo=campo)


def _uv200_forecast_series(model: str, dt_ini: datetime, dt_fim: datetime, *, campo: str) -> xr.DataArray | None:
    """Anomalia PSI/CHI200 prevista (lagged ensemble u/v 200) recortada ao horizonte do modelo."""
    # 'u_zonal'/'v_meridional' nao estao em _UV200_DERIVADO (tratados no ramo especial de
    # _uv200_derived_anom_series, sem Poisson); usam nome fixo. So p/ log — o campo real e resolvido la.
    _NOMES_VENTO = {'u_zonal': 'wnd200_zonal_anom', 'v_meridional': 'wnd200_meridional_anom'}
    nome = _UV200_DERIVADO[campo]['nome'] if campo in _UV200_DERIVADO else _NOMES_VENTO.get(campo, campo)
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
    avail_ini = datetime(init0.year, init0.month, init0.day)
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
                model.upper(), nome, init0, lead_hours, win_ini.date(), win_fim.date())
    fn = _fcst_downloader(model, 'fcst200')
    _, _, clat, clon = clim_uv200_daily(np.array([np.datetime64(win_ini.date())]))
    per_u: list[xr.DataArray] = []
    per_v: list[xr.DataArray] = []
    for init_k in run_inits:
        files_k = list(fn(init=init_k, lead_hours=lead_hours, hours=list(DEFAULT_SYNOPTIC_HOURS)))
        if files_k:
            u_k, v_k = _daily_uv200_on_grid(files_k, win_ini, win_fim, clat, clon, logger)
            per_u.append(u_k)
            per_v.append(v_k)
    if not per_u:
        raise RuntimeError(f'Sem dados de u/v 200 do modelo {model.upper()} no horizonte.')
    u_da = _lagged_ensemble_mean(per_u)
    v_da = _lagged_ensemble_mean(per_v)
    out = _uv200_derived_anom_series(u_da, v_da, campo=campo)
    out.attrs['run_init'] = init0.strftime('%Y-%m-%d %H')
    return out


# Wrappers nomeados (referenciados nas fichas): fixam o `campo` (psi/chi).
def _psi200_reanalise_series(dt_ini, dt_fim):
    return _uv200_reanalise_series(dt_ini, dt_fim, campo='psi')


def _psi200_forecast_series(model, dt_ini, dt_fim):
    return _uv200_forecast_series(model, dt_ini, dt_fim, campo='psi')


def _chi200_reanalise_series(dt_ini, dt_fim):
    return _uv200_reanalise_series(dt_ini, dt_fim, campo='chi')


def _chi200_forecast_series(model, dt_ini, dt_fim):
    return _uv200_forecast_series(model, dt_ini, dt_fim, campo='chi')


def _wnd200_zonal_reanalise_series(dt_ini, dt_fim):
    return _uv200_reanalise_series(dt_ini, dt_fim, campo='u_zonal')


def _wnd200_zonal_forecast_series(model, dt_ini, dt_fim):
    return _uv200_forecast_series(model, dt_ini, dt_fim, campo='u_zonal')


def _wnd200_meridional_reanalise_series(dt_ini, dt_fim):
    return _uv200_reanalise_series(dt_ini, dt_fim, campo='v_meridional')


def _wnd200_meridional_forecast_series(model, dt_ini, dt_fim):
    return _uv200_forecast_series(model, dt_ini, dt_fim, campo='v_meridional')


def _era5_mslp(start, end, force):
    from app.src.uteis.downloaders_wind100m_ERA5 import ensure_era5_mslp_global_for_period as fn
    return fn(start=start, end=end, hours_utc=list(DEFAULT_SYNOPTIC_HOURS), force_redownload=force)


def _mslp_reanalise_series(dt_ini: datetime, dt_fim: datetime) -> xr.DataArray | None:
    """Serie diaria de MSLP (hPa) OBSERVADA — ERA5 apenas (GDAS nao tem downloader de PNMM)."""
    if dt_ini.date() > dt_fim.date():
        return None
    force = bool(getattr(settings, 'FORCE_DOWNLOAD', False))
    era5_period, _ = _get_data_sources(dt_ini, dt_fim)
    if not era5_period:
        return None
    era5_fim = min(dt_fim, era5_period[1])
    logger.info('Download ERA5 MSLP: {} -> {}', era5_period[0].date(), era5_fim.date())
    files = _era5_mslp(era5_period[0], era5_fim, force)
    tgt_lat, tgt_lon = _target_grid()
    return _daily_mslp_on_grid(files, era5_period[0], era5_fim, tgt_lat, tgt_lon, logger)


_MSLP_FCST_DOWNLOADERS = {
    'gefs': ('downloaders_gefs_mslp', 'ensure_gefs_mslp_fcst_for_period'),   # PRMSL do pgrb2a
    'ecmwf': ('downloaders_ecmwf_mslp', 'ensure_ecmwf_mslp_fcst_for_period'),  # msl do open data
    'gfs': ('downloaders_gfs_mslp', 'ensure_gfs_mslp_fcst_for_period'),  # PRMSL do NOMADS
}

# Downloader NATIVO (passo a passo, sem resample diario) de MSLP -- so modelos com esse 2o
# downloader (ver `_mslp_forecast_series_native`). ECMWF: 3/6h (`ecmwf_native_steps`). GFS: 6h fixo
# (mesma cadencia dos buckets de APCP, pra chuva e PNMM animarem no mesmo eixo `time`).
_MSLP_FCST_DOWNLOADERS_NATIVO = {
    'ecmwf': ('downloaders_ecmwf_mslp', 'ensure_ecmwf_mslp_native_fcst_for_period'),
    'gfs': ('downloaders_gfs_mslp', 'ensure_gfs_mslp_native_fcst_for_period'),
}


def _isolinha_mslp_ativa(ficha: dict, variavel_key: str) -> bool:
    """`isolinha_mslp` RE-LIDA em runtime. O dict VARIAVEIS congela essa flag no IMPORT do modulo,
    ANTES de o header do script (aplicar_config_header) rodar no main() -- entao uma ficha que le a
    flag de um setting (ex.: rajada_abs -> GLOBO_3D_ISOLINHA_MSLP_RAJADA_ABS) pegava sempre o default.
    Mesmo padrao ja aplicado a LW/LEVELS. Override por variavel; fallback = valor da ficha (tmp850_mslp
    fixa True)."""
    return bool(settings.get(f'GLOBO_3D_ISOLINHA_MSLP_{variavel_key.upper()}',
                             ficha.get('isolinha_mslp', False)))


def _mslp_forecast_series(model: str, dt_ini: datetime, dt_fim: datetime) -> xr.DataArray | None:
    """Serie diaria de MSLP (hPa) PREVISTA (lagged ensemble). Modelos com downloader de PNMM:
    GEFS (PRMSL do pgrb2a) e ECMWF (msl do open data)."""
    if model not in _MSLP_FCST_DOWNLOADERS:
        raise RuntimeError(
            f'PNMM prevista nao esta wired p/ o modelo {model.upper()} (so GEFS e ECMWF tem downloader '
            'de MSLP). Habilite RUN_GEFS ou RUN_ECMWF para as isolinhas de PNMM no forecast.')
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
    avail_ini = datetime(init0.year, init0.month, init0.day)
    avail_fim = init0 + timedelta(hours=lead_hours)
    win_ini = max(dt_ini, avail_ini)
    win_fim = min(dt_fim, avail_fim)
    if win_ini.date() > win_fim.date():
        raise RuntimeError(
            f'Parte de PREVISAO de MSLP [{dt_ini.date()} a {dt_fim.date()}] fora do horizonte do '
            f'{model.upper()} (disponivel {avail_ini.date()} a {avail_fim.date()}, '
            f'init {init0:%Y-%m-%d %H}Z).')
    logger.info('FORECAST {} [mslp]: init {:%Y-%m-%d %H}Z, lead {}h | janela {} a {}',
                model.upper(), init0, lead_hours, win_ini.date(), win_fim.date())
    import importlib
    _mod, _fn = _MSLP_FCST_DOWNLOADERS[model]
    ensure_mslp = getattr(importlib.import_module(f'app.src.uteis.{_mod}'), _fn)
    tgt_lat, tgt_lon = _target_grid()
    per_run: list[xr.DataArray] = []
    for init_k in run_inits:
        files_k = list(ensure_mslp(
            init=init_k, lead_hours=lead_hours, hours=list(DEFAULT_SYNOPTIC_HOURS)))
        if files_k:
            per_run.append(_daily_mslp_on_grid(files_k, win_ini, win_fim, tgt_lat, tgt_lon, logger))
    if not per_run:
        raise RuntimeError(f'Sem dados de MSLP do modelo {model.upper()} no horizonte.')
    return _lagged_ensemble_mean(per_run)


def _build_mslp_series(model: str | None, dt_ini: datetime, dt_fim: datetime) -> xr.DataArray | None:
    """Serie diaria de MSLP (hPa) na janela [dt_ini, dt_fim], decidida PELAS DATAS: passado ->
    ERA5 (reanalise); futuro -> GEFS (previsao); janela que cruza hoje -> EMENDA das duas partes.

    Cada parte e resiliente: se uma falha (ex.: ERA5 fora do periodo, GEFS sem downloader p/ o
    modelo), avisa e segue com a(s) outra(s). Retorna None se nenhuma parte estiver disponivel."""
    hoje = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    ontem = hoje - timedelta(days=1)
    partes: list[xr.DataArray] = []
    if dt_ini <= ontem:   # parte OBSERVADA (ate ontem) — ERA5
        try:
            past = _mslp_reanalise_series(dt_ini, min(dt_fim, ontem))
            if past is not None and past.sizes.get('time', 0) > 0:
                partes.append(past)
        except Exception as _e:
            logger.warning('MSLP ERA5 (passado) indisponivel: {}', _e)
    if dt_fim >= hoje and model is not None:   # parte PREVISTA (de hoje em diante) — GEFS
        try:
            fut = _mslp_forecast_series(model, max(dt_ini, hoje), dt_fim)
            if fut is not None and fut.sizes.get('time', 0) > 0:
                partes.append(fut)
        except Exception as _e:
            logger.warning('MSLP {} (previsao) indisponivel: {}',
                           model.upper() if model else '-', _e)
    if not partes:
        return None
    if len(partes) == 1:
        return partes[0]
    serie = xr.concat(partes, dim='time').sortby('time')
    _, idx = np.unique(serie['time'].values, return_index=True)
    serie = serie.isel(time=np.sort(idx))
    logger.info('MSLP EMENDA observado+previsao: {} dias ({} a {})', serie.sizes['time'],
                pd.Timestamp(serie['time'].values[0]).date(),
                pd.Timestamp(serie['time'].values[-1]).date())
    return serie


def _mslp_forecast_series_native(model: str, dt_ini: datetime, dt_fim: datetime) -> xr.DataArray:
    """Serie de MSLP (hPa) PREVISTA no PASSO NATIVO do modelo (ECMWF: 3/6h; GFS: 6h fixo, mesma
    cadencia dos buckets de APCP) -- irma de `_mslp_forecast_series`, mas sem media diaria. Usada
    quando `ACUM_HORARIO=true` (ver `_precip_native_forecast_series`) pra isolinha de PNMM animar
    junto da chuva no mesmo eixo `time`."""
    if model not in _MSLP_FCST_DOWNLOADERS_NATIVO:
        raise RuntimeError(
            f"PNMM no passo nativo so tem downloader {'/'.join(_MSLP_FCST_DOWNLOADERS_NATIVO)} "
            f"-- modelo '{model}' nao suportado.")
    rodada = int(settings.get('RODADA', 0))
    if rodada not in (0, 6, 12, 18):
        raise ValueError(f'RODADA deve ser 00/06/12/18 (UTC). Recebido: {rodada:02d}')
    run_inits, lead_hours = _resolve_forecast_lead_init(
        model, rodada=rodada, num_rodada=1,   # passo nativo: sem lagged ensemble (1 rodada)
        forecast_init=settings.get('FORECAST_INIT', 'latest'),
        gefs_lead_days=int(settings.get('GEFS_FORECAST_LEAD_DAYS', settings.get('FORECAST_LEAD_DAYS', 35))),
        cfs_lead_days=45,
    )
    init0 = run_inits[0]
    avail_ini = datetime(init0.year, init0.month, init0.day)
    avail_fim = init0 + timedelta(hours=lead_hours)
    win_ini = max(dt_ini, avail_ini)
    win_fim = min(dt_fim, avail_fim)
    if win_ini.date() > win_fim.date():
        raise RuntimeError(
            f'Parte de PREVISAO de MSLP [{dt_ini.date()} a {dt_fim.date()}] fora do horizonte do '
            f'{model.upper()} (disponivel {avail_ini.date()} a {avail_fim.date()}, '
            f'init {init0:%Y-%m-%d %H}Z).')
    import importlib
    _mod, _fn = _MSLP_FCST_DOWNLOADERS_NATIVO[model]
    ensure_mslp_native = getattr(importlib.import_module(f'app.src.uteis.{_mod}'), _fn)
    files = ensure_mslp_native(init0, lead_hours)
    if not files:
        raise RuntimeError(f'Sem dados de MSLP nativo do {model.upper()} no horizonte.')
    tgt_lat, tgt_lon = _target_grid()
    da = _native_scalar_on_grid(files, ('msl',), win_ini, win_fim, tgt_lat, tgt_lon, logger)
    return (da / 100.0).rename('msl')  # Pa -> hPa


def _precip_native_forecast_series(model: str, dt_ini: datetime, dt_fim: datetime) -> xr.DataArray:
    """Serie de CHUVA (mm) ACUMULADA no PASSO NATIVO do modelo (ECMWF: 3h ate 144h, 6h dai em
    diante; GFS: 6h fixo, os buckets do APCP) -- usada pelo `precip_abs` quando `ACUM_HORARIO=true`
    (settings). Cada frame e' o ACUMULADO (running total) desde o inicio da janela ate aquele
    passo, mesmo principio do `_precip_acumular_no_tempo`, so que por passo em vez de por dia.

    ECMWF: `tp` vem CUMULATIVO desde o init (sem reset) -> incrementa por `diff()` entre passos
    consecutivos, clipado em 0 (ruido numerico). GFS: `APCP` ja vem em BUCKETS de 6h DISCRETOS
    (reset a cada bucket, ver `downloaders_gfs_precip.py`) -> soma direto (`cumsum`), sem diff."""
    if model not in ('ecmwf', 'gfs'):
        raise RuntimeError(
            f"Chuva no passo nativo (ACUM_HORARIO) so tem downloader ECMWF e GFS -- modelo "
            f"'{model}' nao suportado. Habilite RUN_ECMWF ou RUN_GFS (desligue os demais RUN_*).")
    rodada = int(settings.get('RODADA', 0))
    if rodada not in (0, 6, 12, 18):
        raise ValueError(f'RODADA deve ser 00/06/12/18 (UTC). Recebido: {rodada:02d}')
    run_inits, lead_hours = _resolve_forecast_lead_init(
        model, rodada=rodada, num_rodada=1,   # passo nativo: sem lagged ensemble (1 rodada)
        forecast_init=settings.get('FORECAST_INIT', 'latest'),
        gefs_lead_days=int(settings.get('GEFS_FORECAST_LEAD_DAYS', settings.get('FORECAST_LEAD_DAYS', 35))),
        cfs_lead_days=45,
    )
    init0 = run_inits[0]
    avail_ini = datetime(init0.year, init0.month, init0.day)
    avail_fim = init0 + timedelta(hours=lead_hours)
    win_ini = max(dt_ini, avail_ini)
    win_fim = min(dt_fim, avail_fim)
    if win_ini.date() > win_fim.date():
        raise RuntimeError(
            f'Parte de PREVISAO de chuva [{dt_ini.date()} a {dt_fim.date()}] fora do horizonte do '
            f'{model.upper()} (disponivel {avail_ini.date()} a {avail_fim.date()}, '
            f'init {init0:%Y-%m-%d %H}Z).')
    logger.info('FORECAST {} [chuva passo nativo]: init {:%Y-%m-%d %H}Z, lead {}h | janela {} a {}',
                model.upper(), init0, lead_hours, win_ini.date(), win_fim.date())

    # win_fim ate 23h do dia (nao so a meia-noite): mesma convencao do `native_scalar_on_grid`. Corta
    # os passos ALEM da janela pedida ANTES do diff/cumsum/interp -- o modelo baixa o horizonte
    # INTEIRO (ex.: 15 dias do ECMWF) mesmo com DATA_FINAL bem mais curto, e cada passo descartado a
    # mais custa 1 interp() global (721x1440) que so seria jogado fora no `.sel()` do fim da funcao.
    # NAO corta o INICIO (`win_ini`): o cumsum precisa da serie continua desde o `init0` p/ o running
    # total ficar certo -- o corte de tras pra frente e sempre seguro (nada depois de win_fim entra
    # na conta de nenhum passo mantido).
    _win_fim_dia = np.datetime64(win_fim.date()) + np.timedelta64(23, 'h')

    if model == 'ecmwf':
        from app.src.uteis.downloaders_ecmwf_precip import ensure_ecmwf_tp_native_fcst_for_period
        files = ensure_ecmwf_tp_native_fcst_for_period(init0, lead_hours)
        if not files:
            raise RuntimeError('Sem dados de tp nativo do ECMWF no horizonte.')
        tp_da = (xr.open_mfdataset(files, combine='by_coords').sortby('time')['tp']
                 .sel(time=slice(None, _win_fim_dia)).load())
        incremento = tp_da.diff(dim='time').clip(min=0)
        zeros = xr.zeros_like(tp_da.isel(time=[0]))
        chuva_nativa = xr.concat([zeros, incremento], dim='time').cumsum(dim='time', keep_attrs=True)
        chuva_nativa['time'] = tp_da['time']
    else:
        from app.src.uteis.downloaders_gfs_precip import ensure_gfs_precip_native_fcst_for_period
        files = ensure_gfs_precip_native_fcst_for_period(init0, lead_hours)
        if not files:
            raise RuntimeError('Sem dados de APCP nativo do GFS no horizonte.')
        bucket_da = (xr.open_mfdataset(files, combine='by_coords').sortby('time')['precip']
                     .sel(time=slice(None, _win_fim_dia)).load())
        # Buckets ja sao incrementos discretos (reset a cada 6h) -- cumsum direto, sem diff. Zero
        # inicial no `init` p/ o 1o frame do video nao nascer com o bucket inteiro ja acumulado.
        zeros = xr.zeros_like(bucket_da.isel(time=[0])).assign_coords(time=[np.datetime64(init0)])
        chuva_nativa = xr.concat([zeros, bucket_da], dim='time').cumsum(dim='time', keep_attrs=True)

    tgt_lat, tgt_lon = _target_grid()
    _tgt_lat_da = xr.DataArray(tgt_lat, dims=['lat'])
    _tgt_lon_da = xr.DataArray(tgt_lon, dims=['lon'])
    chuva_nativa = chuva_nativa.assign_coords(lon=(chuva_nativa['lon'] % 360)).sortby('lon').sortby('lat')
    # .astype(float32): o xarray promove p/ float64 no interp() por padrao -- a fonte (ECMWF/GFS) ja
    # e float32, entao isso so dobrava a RAM da serie sem ganho de precisao (mm de chuva).
    chuva_acum = (chuva_nativa.interp(lat=_tgt_lat_da, lon=_tgt_lon_da, method='linear')
                  .astype('float32').reset_coords(drop=True))

    chuva_acum = chuva_acum.sel(time=slice(np.datetime64(win_ini), _win_fim_dia))
    chuva_acum = chuva_acum.rename('chuva_acum')
    chuva_acum.attrs['acumulado_no_tempo'] = 1
    chuva_acum.attrs['run_init'] = init0.strftime('%Y-%m-%d %H')
    logger.info('Chuva passo nativo ({}): {} passo(s) | max {:.1f} mm',
               model.upper(), chuva_acum.sizes['time'], float(chuva_acum.max()))
    return chuva_acum


def _ptype_neve_forecast_series(
    model: str, dt_ini: datetime, dt_fim: datetime,
) -> xr.DataArray:
    """Serie NATIVA (ECMWF: 3h ate 144h, 6h dai em diante; GFS: 6h fixo) de NEVE ACUMULADA (mm),
    classificada por tipo de precipitacao. Forecast-only. Usada como CAMADA OPCIONAL sobre um campo
    de chuva ja existente (ex.: `precip_abs`) -- ver `NEVE` (settings) e o bloco de overlay em
    `gerar_animacao`/`_render_clip`.

    NAO recalcula a chuva: a chuva primaria continua vindo do pipeline do `precip_abs` (ja
    calibrado -- paleta, upsample, blue marble). Recalcular a chuva aqui a partir do passo NATIVO
    seria DUPLICAR esse pipeline com um caminho de regrid/render proprio, que tende a descolar do
    `precip_abs` (aconteceu: sumiu o blue marble, isolinha de PNMM indevida) -- portanto so a NEVE
    (que precisa mesmo da classificacao por passo) e' calculada aqui.

    ECMWF: `ptype` e' um CODIGO categorico unico (tabela OMM 4.201) -- neve = incremento de `tp` SO
    nos passos/pixels onde o codigo esta em GLOBO_3D_PTYPE_CODIGOS_NEVE (default [5, 6]).
    GFS: nao tem codigo unico -- publica `CSNOW`, uma flag BINARIA (0/1) independente das demais
    (CRAIN/CFRZR/CICEP, nao baixadas) -- neve = bucket do APCP SO onde CSNOW=1 (ver
    `downloaders_gfs_ptype.py`). Em ambos os casos, cumsum proprio ao longo do `time` nativo (running
    total, mesmo principio do `_precip_acumular_no_tempo`, so que por passo). A CLASSIFICACAO
    acontece na grade NATIVA do modelo (chuva e tipo vem do MESMO fetch, pixel a pixel identicos) --
    so o RESULTADO (continuo, em mm) e interpolado pra grade-alvo; interpolar os CODIGOS/flags direto
    seria mistura sem sentido (ex.: rain=1 + snow=5 -> 3 pareceria "chuva congelante" por
    coincidencia numerica)."""
    if model not in ('ecmwf', 'gfs'):
        raise RuntimeError(
            f"Neve por tipo so tem downloader ECMWF (ptype) e GFS (CSNOW) -- modelo '{model}' nao "
            'suportado. Habilite RUN_ECMWF ou RUN_GFS (desligue os demais RUN_*).')
    rodada = int(settings.get('RODADA', 0))
    if rodada not in (0, 6, 12, 18):
        raise ValueError(f'RODADA deve ser 00/06/12/18 (UTC). Recebido: {rodada:02d}')
    run_inits, lead_hours = _resolve_forecast_lead_init(
        model, rodada=rodada, num_rodada=1,   # ptype/tp nativo: sem lagged ensemble (1 rodada)
        forecast_init=settings.get('FORECAST_INIT', 'latest'),
        gefs_lead_days=int(settings.get('GEFS_FORECAST_LEAD_DAYS', settings.get('FORECAST_LEAD_DAYS', 35))),
        cfs_lead_days=45,
    )
    init0 = run_inits[0]
    avail_ini = datetime(init0.year, init0.month, init0.day)
    avail_fim = init0 + timedelta(hours=lead_hours)
    win_ini = max(dt_ini, avail_ini)
    win_fim = min(dt_fim, avail_fim)
    if win_ini.date() > win_fim.date():
        raise RuntimeError(
            f'Janela [{dt_ini.date()} a {dt_fim.date()}] fora do horizonte do {model.upper()} '
            f'(disponivel {avail_ini.date()} a {avail_fim.date()}, init {init0:%Y-%m-%d %H}Z).')
    logger.info('FORECAST {} [neve por tipo]: init {:%Y-%m-%d %H}Z, lead {}h | janela {} a {}',
                model.upper(), init0, lead_hours, win_ini.date(), win_fim.date())

    # win_fim ate 23h do dia: mesma convencao do `.sel()` no fim da funcao. Corta os passos ALEM da
    # janela pedida ANTES do align/diff/cumsum/interp (ver `_precip_native_forecast_series` -- mesmo
    # raciocinio: o modelo baixa o horizonte inteiro, e aqui e' o DOBRO de trabalho por passo
    # excedente, chuva E tipo). NAO corta o inicio -- o cumsum precisa da serie continua desde `init0`.
    _win_fim_dia = np.datetime64(win_fim.date()) + np.timedelta64(23, 'h')

    # Razao neve:liquido (SLR, "snow-to-liquid ratio"): 10:1 e a regra classica (10mm de neve fofa
    # por 1mm de agua equivalente); neve mais densa/molhada tem razao menor. `tp`/`APCP` vem em mm
    # de LIQUIDO -- aqui converte pra CM de neve: cm_neve = mm_liquido * razao / 10.
    razao_neve = float(settings.get('GLOBO_3D_NEVE_RAZAO_LIQUIDO', 10.0))

    if model == 'ecmwf':
        from app.src.uteis.downloaders_ecmwf_precip import ensure_ecmwf_tp_native_fcst_for_period
        from app.src.uteis.downloaders_ecmwf_ptype import ensure_ecmwf_ptype_fcst_for_period

        tp_files = ensure_ecmwf_tp_native_fcst_for_period(init0, lead_hours)
        pt_files = ensure_ecmwf_ptype_fcst_for_period(init0, lead_hours)
        if not tp_files or not pt_files:
            raise RuntimeError('Sem dados de tp/ptype nativos do ECMWF no horizonte.')

        # Grade NATIVA (sem interp) -- tp e ptype vem do MESMO downloader/cadencia, mesmo eixo `time`
        # por construcao; `join='inner'` protege se um passo faltar (404) so num dos dois.
        tp_ds = xr.open_mfdataset(tp_files, combine='by_coords').sortby('time').sel(time=slice(None, _win_fim_dia))
        pt_ds = xr.open_mfdataset(pt_files, combine='by_coords').sortby('time').sel(time=slice(None, _win_fim_dia))
        tp_da, pt_da = xr.align(tp_ds['tp'], pt_ds['ptype'], join='inner')
        tp_da, pt_da = tp_da.load(), pt_da.load()
        tp_ds.close()
        pt_ds.close()

        codigos_neve = [int(c) for c in settings.get('GLOBO_3D_PTYPE_CODIGOS_NEVE', [5, 6])]
        incremento = tp_da.diff(dim='time').clip(min=0)
        pt_no_passo = pt_da.isel(time=slice(1, None))   # ptype do FIM do intervalo classifica a neve
        inc_neve = incremento.where(pt_no_passo.isin(codigos_neve), 0.0) * (razao_neve / 10.0)

        # cumsum reintroduz o 1o passo como zero (diff() perde o primeiro) -- eixo `time` volta a
        # bater com o de `tp_da`/`pt_da` completo, mesma convencao do `_precip_acumular_no_tempo`.
        zeros = xr.zeros_like(tp_da.isel(time=[0]))
        neve_nativa = xr.concat([zeros, inc_neve], dim='time').cumsum(dim='time', keep_attrs=True)
        neve_nativa['time'] = tp_da['time']
    else:
        from app.src.uteis.downloaders_gfs_precip import ensure_gfs_precip_native_fcst_for_period
        from app.src.uteis.downloaders_gfs_ptype import ensure_gfs_csnow_native_fcst_for_period

        tp_files = ensure_gfs_precip_native_fcst_for_period(init0, lead_hours)
        pt_files = ensure_gfs_csnow_native_fcst_for_period(init0, lead_hours)
        if not tp_files or not pt_files:
            raise RuntimeError('Sem dados de APCP/CSNOW nativos do GFS no horizonte.')

        # Grade NATIVA (sem interp) -- APCP e CSNOW vem da MESMA cadencia (6h, fim de bucket), mesmo
        # eixo `time` por construcao; `join='inner'` protege se um passo faltar (404) so num dos dois.
        tp_ds = xr.open_mfdataset(tp_files, combine='by_coords').sortby('time').sel(time=slice(None, _win_fim_dia))
        pt_ds = xr.open_mfdataset(pt_files, combine='by_coords').sortby('time').sel(time=slice(None, _win_fim_dia))
        bucket_da, csnow_da = xr.align(tp_ds['precip'], pt_ds['csnow'], join='inner')
        bucket_da, csnow_da = bucket_da.load(), csnow_da.load()
        tp_ds.close()
        pt_ds.close()

        # Buckets ja sao incrementos discretos (reset a cada 6h, ver `_precip_native_forecast_series`)
        # -- sem diff, CSNOW=1 no FIM do bucket classifica o bucket inteiro como neve.
        inc_neve = bucket_da.clip(min=0).where(csnow_da == 1, 0.0) * (razao_neve / 10.0)

        # Zero inicial no `init0` p/ o 1o frame nao nascer com o bucket inteiro ja acumulado (mesma
        # convencao do `_precip_native_forecast_series` pro GFS).
        zeros = xr.zeros_like(inc_neve.isel(time=[0])).assign_coords(time=[np.datetime64(init0)])
        neve_nativa = xr.concat([zeros, inc_neve], dim='time').cumsum(dim='time', keep_attrs=True)

    # SO agora, com o resultado ja CONTINUO (mm acumulado), regride pra grade-alvo -- classificar
    # em cima do ptype cru evita misturar codigos categoricos na interpolacao (ver docstring).
    tgt_lat, tgt_lon = _target_grid()
    _tgt_lat_da = xr.DataArray(tgt_lat, dims=['lat'])
    _tgt_lon_da = xr.DataArray(tgt_lon, dims=['lon'])
    neve_nativa = neve_nativa.assign_coords(lon=(neve_nativa['lon'] % 360)).sortby('lon').sortby('lat')
    # .astype(float32): o interp() promove p/ float64 por padrao -- fonte ja e float32 (ver
    # `_precip_native_forecast_series`), sem ganho de precisao pra cm de neve.
    neve_acum = (neve_nativa.interp(lat=_tgt_lat_da, lon=_tgt_lon_da, method='linear')
                 .astype('float32').reset_coords(drop=True))

    # win_fim ate 23h do dia (nao so a meia-noite): mesma convencao do `native_scalar_on_grid`
    # (usado pela MSLP nativa) -- senao a janela cortaria o ULTIMO dia no seu 1o passo (00Z) so.
    neve_acum = neve_acum.sel(time=slice(np.datetime64(win_ini), _win_fim_dia))
    neve_acum = neve_acum.rename('neve_acum')
    neve_acum.attrs['acumulado_no_tempo'] = 1
    neve_acum.attrs['run_init'] = init0.strftime('%Y-%m-%d %H')
    logger.info('Neve por tipo: {} passo(s) nativo(s) | neve max {:.1f} cm (razao {:.0f}:1)',
               neve_acum.sizes['time'], float(neve_acum.max()), razao_neve)
    return neve_acum


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


def _absolute_reanalise_series(ficha: dict, dt_ini: datetime, dt_fim: datetime) -> xr.DataArray | None:
    """Campo ABSOLUTO observado (ERA5/GDAS) — sem subtracao de climatologia. Generico: usa
    `_daily_scalar_on_grid` com `spec['var_candidates']` quando presente (ex.: z250_abs); cai
    p/ `_daily_wind_speed_on_grid` (magnitude de vento) quando a ficha nao tem `var_candidates`
    (ex.: jet_stream, onde o 'campo' e derivado de u/v, nao uma variavel escalar direta)."""
    if dt_ini.date() > dt_fim.date():
        return None
    spec = ficha['spec']
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
    if spec.get('var_candidates'):
        da = _daily_scalar_on_grid(files, spec['var_candidates'], dt_ini, dt_fim, tgt_lat, tgt_lon, logger)
    else:
        da = _daily_wind_speed_on_grid(files, dt_ini, dt_fim, tgt_lat, tgt_lon, logger)
    logger.info('{}: {} dias | min={:.1f} max={:.1f} {}',
                spec['nome'], da.sizes['time'], float(da.min()), float(da.max()), spec['unidade'])
    return da


def _absolute_forecast_series(ficha: dict, model: str, dt_ini: datetime, dt_fim: datetime) -> xr.DataArray | None:
    """Campo ABSOLUTO previsto (lagged ensemble) — sem subtracao de climatologia. Generico via
    `spec['var_candidates']` (ex.: z250_abs); cai p/ magnitude de vento sem eles (jet_stream)."""
    spec = ficha['spec']
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
    avail_ini = datetime(init0.year, init0.month, init0.day)
    avail_fim = init0 + timedelta(hours=lead_hours)
    win_ini = max(dt_ini, avail_ini)
    win_fim = min(dt_fim, avail_fim)
    if win_ini.date() > win_fim.date():
        raise RuntimeError(
            f'Parte de PREVISAO [{dt_ini.date()} a {dt_fim.date()}] fora do horizonte do '
            f'{model.upper()} (disponivel {avail_ini.date()} a {avail_fim.date()}, '
            f'init {init0:%Y-%m-%d %H}Z). Reduza DATA_FINAL ou aumente o lead.'
        )
    # DATA_FINAL alem do horizonte do modelo: o clamp (win_fim = avail_fim) ja limita a janela;
    # AVISA no terminal que o video sai so ate a ultima data prevista, nao ate o DATA_FINAL pedido
    # (pedido do usuario p/ nao surpreender). So dispara quando ha truncamento de fato.
    if dt_fim.date() > avail_fim.date():
        logger.warning(
            'DATA_FINAL ({}) ultrapassa o horizonte do {} (init {:%Y-%m-%d %H}Z + {}d = {}); '
            'animando so ate a ultima data prevista disponivel ({}).',
            dt_fim.date(), model.upper(), init0, lead_hours // 24, avail_fim.date(), win_fim.date())
    logger.info('FORECAST {} [{}]: init {:%Y-%m-%d %H}Z, lead {}h | janela {} a {}',
                model.upper(), spec['nome'], init0, lead_hours, win_ini.date(), win_fim.date())
    fn = _fcst_downloader(model, spec['kind'])
    tgt_lat, tgt_lon = _target_grid()
    per_run: list[xr.DataArray] = []
    for init_k in run_inits:
        files_k = list(fn(init=init_k, lead_hours=lead_hours, hours=list(DEFAULT_SYNOPTIC_HOURS)))
        if files_k:
            if spec.get('var_candidates'):
                per_run.append(_daily_scalar_on_grid(
                    files_k, spec['var_candidates'], win_ini, win_fim, tgt_lat, tgt_lon, logger))
            else:
                per_run.append(_daily_wind_speed_on_grid(files_k, win_ini, win_fim, tgt_lat, tgt_lon, logger))
    if not per_run:
        raise RuntimeError(f'Sem dados de {spec["nome"]} do modelo {model.upper()} no horizonte.')
    da = _lagged_ensemble_mean(per_run)
    da.attrs['run_init'] = init0.strftime('%Y-%m-%d %H')
    return da


def _absolute_reanalise_series_synoptic(ficha: dict, dt_ini: datetime, dt_fim: datetime) -> xr.DataArray | None:
    """Campo ABSOLUTO observado (ERA5/GDAS) NA HORA SINOTICA (00/06/12/18Z, sem media diaria) —
    sem subtracao de climatologia. Irma de `_absolute_reanalise_series`, generica (qualquer
    escalar via `spec['var_candidates']`, nao so vento) — usada pelo MP4 de fichas com
    `sinotico_mp4=True` (ex.: z250_abs)."""
    if dt_ini.date() > dt_fim.date():
        return None
    spec = ficha['spec']
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
    da = _synoptic_scalar_on_grid(files, spec['var_candidates'], dt_ini, dt_fim, tgt_lat, tgt_lon, logger)
    logger.info('{}: {} passos sinoticos | min={:.1f} max={:.1f} {}',
                spec['nome'], da.sizes['time'], float(da.min()), float(da.max()), spec['unidade'])
    return da


def _absolute_forecast_series_synoptic(ficha: dict, model: str, dt_ini: datetime, dt_fim: datetime) -> xr.DataArray | None:
    """Campo ABSOLUTO previsto (lagged ensemble) NA HORA SINOTICA (sem media diaria) — sem
    subtracao de climatologia. Irma de `_absolute_forecast_series`, generica via
    `spec['var_candidates']`."""
    spec = ficha['spec']
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
    avail_ini = datetime(init0.year, init0.month, init0.day)
    avail_fim = init0 + timedelta(hours=lead_hours)
    win_ini = max(dt_ini, avail_ini)
    win_fim = min(dt_fim, avail_fim)
    if win_ini.date() > win_fim.date():
        raise RuntimeError(
            f'Parte de PREVISAO [{dt_ini.date()} a {dt_fim.date()}] fora do horizonte do '
            f'{model.upper()} (disponivel {avail_ini.date()} a {avail_fim.date()}, '
            f'init {init0:%Y-%m-%d %H}Z). Reduza DATA_FINAL ou aumente o lead.'
        )
    logger.info('FORECAST {} [{}]: init {:%Y-%m-%d %H}Z, lead {}h | janela {} a {} (sinotico)',
                model.upper(), spec['nome'], init0, lead_hours, win_ini.date(), win_fim.date())
    fn = _fcst_downloader(model, spec['kind'])
    tgt_lat, tgt_lon = _target_grid()
    per_run: list[xr.DataArray] = []
    for init_k in run_inits:
        files_k = list(fn(init=init_k, lead_hours=lead_hours, hours=list(DEFAULT_SYNOPTIC_HOURS)))
        if files_k:
            per_run.append(_synoptic_scalar_on_grid(
                files_k, spec['var_candidates'], win_ini, win_fim, tgt_lat, tgt_lon, logger))
    if not per_run:
        raise RuntimeError(f'Sem dados de {spec["nome"]} do modelo {model.upper()} no horizonte.')
    da = _lagged_ensemble_mean(per_run)
    da.attrs['run_init'] = init0.strftime('%Y-%m-%d %H')
    return da


def _build_var_series_synoptic(ficha: dict, model: str | None,
                               dt_ini: datetime, dt_fim: datetime) -> xr.DataArray:
    """Serie ABSOLUTA na HORA SINOTICA (00/06/12/18Z) na janela [dt_ini, dt_fim], com emenda
    observado+previsao decidida PELAS DATAS (espelha o ramo `absoluto` de `_build_var_series`,
    mas usando os loaders sinoticos). Usada pelo MP4 de fichas com `sinotico_mp4=True`.

    NAO aplica pentada movel (`_aplicar_pentada_movel` nao e chamado aqui de proposito):
    `rolling(time=dias)` assume "1 passo = 1 dia", que deixa de ser verdade num eixo sinotico
    (1 passo = 6h). Se a ficha tiver uma pentada configurada, avisa que ela sera IGNORADA
    neste caminho (a serie diaria do GIF/PNG, essa sim, continua aplicando a pentada normal)."""
    spec = ficha['spec']
    _nome = spec['nome']
    _dias_pentada = settings.get(f'GLOBO_3D_PENTADA_{_nome.upper()}', None)
    if _dias_pentada is None:
        _dias_pentada = ficha.get('pentada_movel', settings.get('GLOBO_3D_PENTADA_DIAS', 0))
    if int(_dias_pentada or 0) > 1:
        logger.warning('{}: pentada movel de {} dias configurada, mas IGNORADA no MP4 sinotico '
                       '(sinotico_mp4=True) — rolling de "dias" nao se aplica a passos de 6h. '
                       'O GIF/PNG (media diaria) segue aplicando a pentada normalmente.',
                       _nome, _dias_pentada)
    hoje = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    ontem = hoje - timedelta(days=1)
    partes: list[xr.DataArray] = []
    if dt_ini <= ontem:
        partes.append(_absolute_reanalise_series_synoptic(ficha, dt_ini, min(dt_fim, ontem)))
    if dt_fim >= hoje:
        if model is None:
            raise RuntimeError('Janela tem datas futuras mas nenhum modelo foi habilitado.')
        partes.append(_absolute_forecast_series_synoptic(ficha, model, max(dt_ini, hoje), dt_fim))
    partes = [p for p in partes if p is not None and p.sizes.get('time', 0) > 0]
    if not partes:
        raise RuntimeError(
            f'Sem dados sinoticos de {spec["nome"]} na janela {dt_ini.date()} a {dt_fim.date()}.')
    if len(partes) == 1:
        return partes[0]
    serie = xr.concat(partes, dim='time').sortby('time')
    _, idx = np.unique(serie['time'].values, return_index=True)
    serie = serie.isel(time=np.sort(idx))
    for p in partes:
        if 'run_init' in p.attrs:
            serie.attrs['run_init'] = p.attrs['run_init']
    logger.info('EMENDA sinotica observado+previsao: {} passos ({} a {})',
                serie.sizes['time'],
                pd.Timestamp(serie['time'].values[0]),
                pd.Timestamp(serie['time'].values[-1]))
    return serie


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
    clim_fn = spec['clim_fn']
    if clim_fn is None and spec['kind'] == 'olr':
        from app.src.uteis.clim_diaria_olr import clim_olr_daily_for_anim
        clim_fn = clim_olr_daily_for_anim
    out = _anom_from_clim(daily, clim_fn, celsius=spec['celsius'],
                          nome=spec['nome'], unidade=spec['unidade'])
    out.attrs['run_init'] = init0.strftime('%Y-%m-%d %H')  # rodada (rotulo do rodape no s39)
    return out


def _contourf_raster(lon: np.ndarray, lat: np.ndarray, campo: np.ndarray, levels, cmap,
                     extend: str, px: int = 2048) -> np.ndarray:
    """Renderiza um `contourf` PLANO (sem projecao) num buffer RGBA e o devolve.

    Usado p/ compor o sombreado no globo via `ax.imshow` (reprojecao de RASTER), preservando as
    bandas suaves do contourf SEM o bug do cartopy 0.25+mpl 3.10 (que descarta poligonos quando o
    contourf e desenhado direto no eixo do globo -> 'washout'). Eixo cobre exatamente [lon,lat].

    `px` = LARGURA do raster; a altura sai da PROPORCAO real de [lon,lat] (no dominio global 360x180
    da exatamente px//2, como antes). Isso e o que permite passar um campo ja RECORTADO na area
    visivel (ver `_subset_campo_bbox`): o mesmo px cobre menos graus -> muitos mais px/grau.
    """
    _lon_span = float(lon.max() - lon.min())
    _lat_span = float(lat.max() - lat.min())
    h = max(2, min(px * 2, int(round(px * _lat_span / _lon_span)))) if _lon_span > 0 else max(2, px // 2)
    fig = plt.figure(figsize=(px / 100.0, h / 100.0), dpi=100)
    # Fundo TRANSPARENTE: campos com extend unilateral ('max'/'min', ex.: jet_stream) deixam
    # areas SEM fill (abaixo do vmin). Sem isto, o buffer capturaria o fundo BRANCO opaco da
    # figura auxiliar, que seria composto por cima do globo (globo todo branco). Com alpha=0
    # essas areas saem transparentes e revelam o fundo do globo. Campos 'both' preenchem o
    # dominio inteiro -> nao ha area vazia -> saida identica (fix nao os afeta).
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.patch.set_alpha(0.0)
    ax.set_xlim(float(lon.min()), float(lon.max()))
    ax.set_ylim(float(lat.min()), float(lat.max()))
    # antialiased=False: com fundo transparente, as bordas ANTI-ALIASED entre bandas ficam
    # semi-transparentes (costuras) e vazam o preto do globo -> "linhas de divisa" visiveis entre
    # as bandas (bem aparente em paletas de alto contraste, ex.: jet_stream: ~85% dos px pintados
    # viravam costura). Sem AA as bandas encostam solidas (0 costura); a suavizacao das bordas vem
    # do imshow bilinear (regrid) na reprojecao ao globo, entao o resultado final segue liso.
    # ListedColormap (paleta DISCRETA, 1 cor/banda) + extend UNILATERAL ('max'/'min', ex.: precip_abs):
    # o contourf reamostra a cmap p/ N+1 regioes e EMBARALHA as cores (banda pega a cor errada). Passar
    # BoundaryNorm(levels) mapeia EXATO por banda (extend usa set_over/set_under). Escopo em 'max'/'min'
    # p/ NAO mexer nas discretas com extend='both' de antes (ex.: tsm_anom). LinearSegmented: sem norm.
    _cf_kw = ({'norm': BoundaryNorm(levels, ncolors=cmap.N)}
              if isinstance(cmap, ListedColormap) and extend in ('max', 'min') else {})
    ax.contourf(lon, lat, campo, levels=levels, cmap=cmap, extend=extend, antialiased=False, **_cf_kw)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()  # (h, px, 4), origin='upper'
    plt.close(fig)
    return rgba


def _taper_lat(lat: np.ndarray, core: float, edge: float) -> np.ndarray:
    """Janela latitudinal SUAVE (taper cosseno) p/ a camada de OLR equatorial: 1 em |lat|<=core,
    decai como meio-cosseno (Hann) ate 0 em |lat|>=edge. Evita a borda reta do corte booleano."""
    al = np.abs(np.asarray(lat, dtype='float64'))
    w = np.ones_like(al)
    width = max(edge - core, 1e-6)
    trans = (al > core) & (al < edge)
    w[trans] = 0.5 * (1.0 + np.cos(np.pi * (al[trans] - core) / width))
    w[al >= edge] = 0.0
    return w


def _validar_dias_degenerados(serie: xr.DataArray, ficha: dict, logger) -> xr.DataArray:
    """Guarda de robustez: detecta dias com anomalia DEGENERADA (campo quase todo de um sinal).

    Um campo de anomalia global e fisicamente BALANCEADO (anomalia vs climatologia integra ~0):
    a fracao de valores negativos e positivos fica ~0,3-0,6. Um dia "todo amarelo/azul" (dado
    parcial/incompleto no download, ex.: rodada ainda publicando) e flagado por DOIS criterios:
      - ABSOLUTO: min(frac_neg, frac_pos) < GLOBO_3D_DIA_DEGEN_FRAC (default 0.02) — ~todo de um sinal;
      - RELATIVO: frac_neg (ou frac_pos) < GLOBO_3D_DIA_DEGEN_REL * mediana_da_serie (default 0.5) —
        dia com MUITO menos de um sinal que o tipico (pega o caso PARCIAL, nao so o ~0%).
    So vale para anomalia simetrica (pula `absoluto`/`simetrico=False`).

    Comportamento: avisa (logger.warning) listando as datas. Se GLOBO_3D_SKIP_DIAS_DEGENERADOS
    (default true), remove os dias ruins da serie (a animacao pula o dia). Se TODOS degenerarem,
    levanta erro (download incompleto).
    """
    if ficha.get('absoluto') or not ficha.get('simetrico', True):
        return serie
    frac_min = float(settings.get('GLOBO_3D_DIA_DEGEN_FRAC', 0.02))   # limiar ABSOLUTO
    rel = float(settings.get('GLOBO_3D_DIA_DEGEN_REL', 0.5))          # fracao da MEDIANA da serie
    skip = bool(settings.get('GLOBO_3D_SKIP_DIAS_DEGENERADOS', True))
    nome = ficha['spec']['nome']
    vals = serie.values
    n = vals.shape[0]
    fneg = np.full(n, np.nan)
    fpos = np.full(n, np.nan)
    for i in range(n):
        fin = np.isfinite(vals[i])
        if int(fin.sum()) < 100:  # poucos pontos finitos -> nao avalia (evita falso positivo)
            continue
        af = vals[i][fin]
        fneg[i] = float(np.mean(af < 0))
        fpos[i] = float(np.mean(af > 0))
    # Mediana da serie (referencia robusta): um dia parcial tem MUITO menos de um sinal que o tipico.
    med_neg = float(np.nanmedian(fneg)) if np.isfinite(fneg).any() else 0.0
    med_pos = float(np.nanmedian(fpos)) if np.isfinite(fpos).any() else 0.0
    bad = []
    for i in range(n):
        if not (np.isfinite(fneg[i]) and np.isfinite(fpos[i])):
            continue
        degen_abs = min(fneg[i], fpos[i]) < frac_min               # ~todo de um sinal
        degen_rel = (fneg[i] < rel * med_neg) or (fpos[i] < rel * med_pos)  # outlier vs mediana
        if degen_abs or degen_rel:
            bad.append(i)
    if not bad:
        return serie
    datas = ', '.join(str(pd.Timestamp(serie['time'].values[i]).date()) for i in bad)
    logger.warning(
        '⚠️  {}: {} dia(s) com anomalia DEGENERADA (campo quase todo de um sinal — provavel dado '
        'parcial/incompleto no download): {}', nome, len(bad), datas)
    if len(bad) == vals.shape[0]:
        raise RuntimeError(
            f'{nome}: TODOS os {vals.shape[0]} dias estao degenerados (anomalia quase toda de um '
            f'sinal). Provavel download incompleto da rodada — reexecute quando publicada ou use '
            f'FORECAST_INIT de uma rodada completa.')
    if skip:
        keep = [i for i in range(vals.shape[0]) if i not in set(bad)]
        logger.warning('   -> pulando {} dia(s) degenerado(s) na animacao '
                       '(GLOBO_3D_SKIP_DIAS_DEGENERADOS=true).', len(bad))
        return serie.isel(time=keep)
    return serie


def _pentada_movel_serie(serie: xr.DataArray, dias, nome: str) -> xr.DataArray:
    """Converte a serie DIARIA em PENTADAS MOVEIS de `dias` dias (espelha o s34): cada frame passa
    a ser a media de uma janela que desliza 1 dia por vez, ROTULADA pelo dia INICIAL (frame do dia
    d = media de [d, d+dias-1]). dias<=1 ou serie mais curta que a janela -> retorna intacta.
    """
    dias = int(dias or 0)
    if dias <= 1:
        return serie
    n = serie.sizes.get('time', 0)
    if n < dias:
        logger.warning('{}: pentada movel de {} dias pulada — serie tem so {} dia(s).',
                       nome, dias, n)
        return serie
    # rolling trailing rotula no ULTIMO dia da janela -> media em j = [j-dias+1, j].
    # A janela FORWARD que comeca no dia i e o rolling no indice i+dias-1; recorto e re-rotulo
    # pelos primeiros (n-dias+1) dias iniciais.
    roll = serie.rolling(time=dias, center=False).mean()
    fwd = roll.isel(time=slice(dias - 1, None))
    fwd = fwd.assign_coords(time=serie['time'].values[:fwd.sizes['time']])
    fwd.name = serie.name
    fwd.attrs.update(serie.attrs)
    fwd.attrs['pentada_dias'] = dias  # sinaliza ao render p/ rotular a DATA como intervalo
    logger.info('{}: pentadas moveis de {} dias -> {} frames (de {} a {}, passo 1 dia)',
                nome, dias, fwd.sizes['time'],
                pd.Timestamp(fwd['time'].values[0]).date(),
                pd.Timestamp(fwd['time'].values[-1]).date())
    return fwd


def _aplicar_pentada_movel(serie: xr.DataArray, ficha: dict) -> xr.DataArray:
    """Pentada movel decidida pela FICHA da propria variavel (espelha o s34).

    Janela (em dias) por precedencia: GLOBO_3D_PENTADA_<VAR> > ficha['pentada_movel'] >
    GLOBO_3D_PENTADA_DIAS (global). 0/1/ausente = sem pentada (mantem diario).
    """
    nome = ficha['spec']['nome']
    dias = settings.get(f'GLOBO_3D_PENTADA_{nome.upper()}', None)
    if dias is None:
        dias = ficha.get('pentada_movel', settings.get('GLOBO_3D_PENTADA_DIAS', 0))
    return _pentada_movel_serie(serie, dias, nome)


def _alinhar_grade_partes(partes: list[xr.DataArray]) -> list[xr.DataArray]:
    """Realinha as partes da EMENDA (observado + previsao) numa grade lat/lon comum.

    Quando o trecho observado e o previsto vem em grades diferentes — ex.: OLR observado (CPC
    Blended 2.5°) vs OLR previsto (GEFS 0.5°) — o xr.concat(join='outer') faz a UNIAO das grades e
    enche de NaN os pontos que so existem numa delas. O dia observado, esparso na grade fina, fica
    ~96% NaN e o shaded sai em BRANCO (o seam observado↔previsao). Fields cujo observado e previsto
    ja compartilham grade (ex.: psi200 na LTM 2.5°) nao passam por nada aqui.

    Reamostra (interp linear) todas as partes para a grade de MAIOR resolucao (mais pontos). Fecha
    a longitude com um ponto ciclico (lon0+360) antes do interp p/ NAO deixar faixa NaN no limbo
    leste (358-360°) — o mesmo cuidado que _olr_reanalise_series tinha ao manter a grade nativa.
    """
    if len(partes) < 2:
        return partes
    alvo = max(partes, key=lambda p: p.sizes.get('lat', 0) * p.sizes.get('lon', 0))
    lat_alvo, lon_alvo = alvo['lat'].values, alvo['lon'].values
    out, mudou = [], False
    for p in partes:
        if (p.sizes.get('lat') == alvo.sizes.get('lat') and p.sizes.get('lon') == alvo.sizes.get('lon')
                and np.array_equal(p['lat'].values, lat_alvo)
                and np.array_equal(p['lon'].values, lon_alvo)):
            out.append(p)
            continue
        q = p
        if float(q['lon'].max()) < float(lon_alvo.max()):
            wrap = q.isel(lon=0).assign_coords(lon=float(q['lon'].values[0]) + 360.0)
            q = xr.concat([q, wrap], dim='lon')
        qi = q.interp(lat=lat_alvo, lon=lon_alvo)
        qi.attrs.update(p.attrs)
        out.append(qi)
        mudou = True
    if mudou:
        logger.info('Emenda: grades diferentes reamostradas p/ {} lat × {} lon (evita branco no dia observado)',
                    lat_alvo.size, lon_alvo.size)
    return out


def _precip_acumular_no_tempo(serie: xr.DataArray, spec: dict) -> xr.DataArray:
    """Modo de acumulacao da CHUVA (so `kind=='precip'`):
      - ACUMULAR_NO_TEMPO=true  -> trava a DATA_INICIAL e SOMA a chuva ao longo do tempo (running
        total = cumsum no eixo `time`). Cada frame vira a soma do inicio ate aquele dia (a animacao
        mostra o acumulado CRESCENDO ate a DATA_FINAL).
      - senao (ACUMULADO_DIARIO, default) -> retorna a serie DIARIA como esta (cada frame = 1 dia).
    So afeta precip; qualquer outra variavel passa direto."""
    if spec.get('kind') != 'precip' or not bool(settings.get('ACUMULAR_NO_TEMPO', False)):
        return serie
    out = serie.cumsum(dim='time', keep_attrs=True)
    out.attrs.update(serie.attrs)
    # Marca a serie como running total -> `_agg_estatico` sabe que a figura estatica desta serie e o
    # TOTAL (ultimo passo), nao a media (media de acumulado crescente = ~55% do total = numero que
    # nao existe). Qualquer outra serie (hgt/olr/mslp) segue na media normal.
    out.attrs['acumulado_no_tempo'] = 1
    logger.info('Chuva ACUMULAR_NO_TEMPO: running total (cumsum) sobre {} dia(s) — trava a data inicial',
                serie.sizes.get('time', 0))
    return out


def _build_var_series(ficha: dict, model: str | None,
                      dt_ini: datetime, dt_fim: datetime,
                      aplicar_pentada: bool = True) -> xr.DataArray:
    """Serie diaria na janela [dt_ini, dt_fim], decidida PELAS DATAS:
    passado -> reanalise; futuro -> previsao (modelo); janela que cruza hoje ->
    EMENDA observado + previsao num unico vetor temporal continuo.
    Variaveis com 'absoluto': True retornam o campo bruto sem subtracao de climatologia.

    `aplicar_pentada` (default True) aplica a pentada movel da propria ficha (s38/s39). O s40
    (figuras estaticas) passa False p/ obter a serie DIARIA CRUA e fazer suas proprias agregacoes
    (diario / media movel / pentadas fixas / media total).
    """
    spec = ficha['spec']
    hoje = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    ontem = hoje - timedelta(days=1)

    if ficha.get('absoluto'):
        abs_partes: list[xr.DataArray] = []
        if dt_ini <= ontem:
            abs_partes.append(_absolute_reanalise_series(ficha, dt_ini, min(dt_fim, ontem)))
        if dt_fim >= hoje:
            if model is None:
                raise RuntimeError('Janela tem datas futuras mas nenhum modelo foi habilitado.')
            abs_partes.append(_absolute_forecast_series(ficha, model, max(dt_ini, hoje), dt_fim))
        abs_partes = [p for p in abs_partes if p is not None and p.sizes.get('time', 0) > 0]
        if not abs_partes:
            raise RuntimeError(
                f'Sem dados de {spec["nome"]} na janela {dt_ini.date()} a {dt_fim.date()}.')
        if len(abs_partes) == 1:
            return _precip_acumular_no_tempo(abs_partes[0], spec)
        abs_partes = _alinhar_grade_partes(abs_partes)
        abs_serie = xr.concat(abs_partes, dim='time').sortby('time')
        _, abs_idx = np.unique(abs_serie['time'].values, return_index=True)
        abs_serie = abs_serie.isel(time=np.sort(abs_idx))
        for p in abs_partes:
            if 'run_init' in p.attrs:
                abs_serie.attrs['run_init'] = p.attrs['run_init']
        return _precip_acumular_no_tempo(abs_serie, spec)

    partes: list[xr.DataArray] = []
    if dt_ini <= ontem:  # ha parte OBSERVADA (ate ontem)
        _rfn = spec.get('reanalise_fn')
        if _rfn is not None:
            partes.append(_rfn(dt_ini, min(dt_fim, ontem)))
        else:
            partes.append(_reanalise_series(spec, dt_ini, min(dt_fim, ontem)))
    if dt_fim >= hoje:   # ha parte PREVISTA (de hoje em diante)
        if model is None:
            raise RuntimeError('Janela tem datas futuras mas nenhum modelo foi habilitado.')
        _ffn = spec.get('forecast_fn')  # hook p/ campos derivados (ex.: psi200_anom via u/v200)
        if _ffn is not None:
            partes.append(_ffn(model, max(dt_ini, hoje), dt_fim))
        else:
            partes.append(_forecast_series(spec, model, max(dt_ini, hoje), dt_fim))

    partes = [p for p in partes if p is not None and p.sizes.get('time', 0) > 0]
    if not partes:
        raise RuntimeError(f'Sem dados de {spec["nome"]} na janela {dt_ini.date()} a {dt_fim.date()}.')
    if len(partes) == 1:
        out = _validar_dias_degenerados(partes[0], ficha, logger)
        return _aplicar_pentada_movel(out, ficha) if aplicar_pentada else out

    partes = _alinhar_grade_partes(partes)
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
    serie = _validar_dias_degenerados(serie, ficha, logger)
    return _aplicar_pentada_movel(serie, ficha) if aplicar_pentada else serie


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


# Paleta da anomalia de hgt250 (Guillaume) — reutilizada por z250_anom e pelos campos de
# anomalia de vento (wnd250_zonal_anom, wnd850_meridional_anom), conforme pedido.
_PALETA_ANOM_HGT250 = [
    '#8A83A2',  # -200  roxo acinzentado (extremo neg)
    '#6656AC',  # -175  roxo-azul
    '#393FAE',  # -150  azul-índigo
    '#1A56AD',  # -125  azul médio-escuro
    '#2579D9',  # -100  azul
    '#62A6E4',  #  -75  azul claro
    '#91C2EF',  #  -50  azul pálido
    '#F1F1F9',  #  -25  quase neutro frio
    '#EDEDEC',  #    0  CINZA NEUTRO
    '#FDFED4',  #  +25  quase neutro quente
    '#FEEA8C',  #  +50  amarelo pálido
    '#FEBE56',  #  +75  âmbar
    '#FB6410',  # +100  laranja
    '#E01801',  # +125  vermelho
    '#C70B13',  # +150  vermelho escuro
    '#D91B57',  # +175  rosa-vermelho
    '#CF5A96',  # +200  rosa (extremo pos)
]


# Paleta verde->bege->marrom da anomalia de CHI200 (identica ao s03: CHI200_COLORS).
# Verde = chi NEGATIVO (divergencia em altos niveis/conveccao); marrom = chi POSITIVO (subsidencia).
# Reutilizada por chi200_anom e por chi200_cores_psi200_contornos (mesmo shaded).
_PALETA_CHI200 = [
    '#005a45', '#0f7a6c', '#2e9b96', '#62bdb7', '#9dd8d2', '#dff3f1',
    '#f7f4eb', '#e7d9a9', '#d6b566', '#bd8a35', '#9a6313', '#6f4300',
]


# Paleta BrBG_r da anomalia de OLR (igual ao s05): verde-azulado (OLR- = mais convecção/úmido)
# -> branco (neutro) -> marrom (OLR+ = convecção suprimida/seco). Reutilizada por olr_anom e por
# olr_cores_psi200_contornos (mesmo shaded).
_PALETA_OLR_ANOM = [
    '#003c30',  # verde-azulado escuro (extremo neg, muito úmido)
    '#005046', '#01655d', '#1a7e76', '#35978f', '#5bb3a8', '#7fccc0',
    '#a3dbd3', '#c7eae5', '#def0ed',
    '#f5f5f4',  # neutro
    '#f5efdc', '#f6e8c3', '#ead59f', '#dec17b', '#cea053', '#bf812d',
    '#a5691b', '#8b500a', '#6e4007',
    '#543005',  # marrom escuro (extremo pos, muito seco)
]

# TSM ABSOLUTA — levels 0..32°C (passo 0.5) e 16 cores, IDENTICOS ao s30 (SST_MEAN_LEVELS/COLORS).
# 16 cores interpoladas sobre as 64 bandas (motor usa LinearSegmentedColormap quando len!=niveis).
_SST_ABS_LEVELS = [round(i * 0.5, 1) for i in range(65)]
_SST_ABS_COLORS = [
    'white', 'blueviolet', 'blue', 'cyan', 'limegreen', 'greenyellow',
    'yellow', 'gold', 'orange', 'darkorange', 'orangered', 'red',
    'darkred', 'crimson', 'magenta', 'white',
]


# Paleta da anomalia de T850 amostrada PIXEL A PIXEL da barra de referencia
# (Entrada/paleta.jpg), esquerda->direita: magenta (frio extremo) -> roxo ->
# azul-escuro -> azul -> branco (conforme a la moyenne) -> salmao -> vermelho ->
# vinho -> quase-preto -> cinza-carvao (quente extremo). Reutilizada por
# tmp850_anom, tmp850_mslp e tmp850_cores_psi200_contornos (mesmo shaded).
_PALETA_T850_ANOM = [
    '#a30d92', '#720669', '#3c0654', '#17022b', '#021323',
    '#043462', '#104b87', '#256aab', '#3d89bd', '#5ea3cc',
    '#bad9eb', '#e1ebf4', '#f7f7f7', '#f8ede7', '#f9dac8',
    '#e58366', '#cd5147', '#b72532', '#9c1526', '#821220',
    '#5f0d19', '#3c0711', '#25060b', '#0d0707', '#3d3d3d',
]


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
        # Paleta idêntica ao s39 (Guillaume): s38 e s39 compartilham shaded e contorno.
        'cmap_colors': [
            '#8A83A2',  # [0]  -200  roxo acinzentado (extremo neg)
            '#6656AC',  # [1]  -175  roxo-azul
            '#393FAE',  # [2]  -150  azul-índigo
            '#1A56AD',  # [3]  -125  azul médio-escuro
            '#2579D9',  # [4]  -100  azul
            '#62A6E4',  # [5]   -75  azul claro
            '#91C2EF',  # [6]   -50  azul pálido
            '#F1F1F9',  # [7]   -25  quase neutro frio
            '#EDEDEC',  # [8]     0  CINZA NEUTRO
            '#FDFED4',  # [9]   +25  quase neutro quente
            '#FEEA8C',  # [10]  +50  amarelo pálido
            '#FEBE56',  # [11]  +75  âmbar
            '#FB6410',  # [12] +100  laranja
            '#E01801',  # [13] +125  vermelho
            '#C70B13',  # [14] +150  vermelho escuro
            '#D91B57',  # [15] +175  rosa-vermelho
            '#CF5A96',  # [16] +200  rosa (extremo pos)
        ],
        'niveis': 50,
        'simetrico': True,
        'vmax': float(settings.get('GLOBO_3D_VMAX_Z250_ANOM', 200)),
        'isolinha_hgt_abs': True,  # isolinhas de geopotencial absoluto em cinza escuro
        'spec': {
            'nome': 'z250_anom', 'unidade': 'mgp', 'celsius': False,
            'var_candidates': HGT_VARS, 'clim_fn': clim_hgt250_daily, 'kind': 'hgt250',
            'era5_fn': _era5_z250, 'gdas_fn': _gdas_z250,
        },
    },
    # Irma do z250_anom, mesma paleta -- serve so como CAMPO-GUIA dedicado do jato quando
    # GLOBO_3D_SUBTROPICAL_JET_NIVEL (ou JET_STREAM_NIVEL) cai na escala de Z500 (~4700-6200 mgp)
    # e a variavel shaded ativa NAO e z500_abs (kind='hgt500', que ja se auto-reusa). Nao listada
    # nas fichas usuais do s38-s41 (nada a ver com o "z250_anom gera par 5d" -- essa nao gera).
    'z500_anom': {
        'titulo': 'Anomalia de Altura Geopotencial em 500 hPa',
        'titulo_en': '500-hPa geopotential height',
        'rotulo_box': 'Geopotential Height Anomaly',
        'subtitulo_dir': 'Geopotential height at 500 hPa',
        'unidade': 'mgp',
        'cmap_colors': list(_PALETA_ANOM_HGT250),
        'niveis': 50,
        'simetrico': True,
        'vmax': float(settings.get('GLOBO_3D_VMAX_Z500_ANOM', 200)),
        'spec': {
            'nome': 'z500_anom', 'unidade': 'mgp', 'celsius': False,
            'var_candidates': HGT_VARS, 'clim_fn': clim_hgt500_daily, 'kind': 'hgt500',
            'era5_fn': _era5_z500, 'gdas_fn': _gdas_z500,
        },
    },
    'z250_abs': {
        'titulo': 'Altura Geopotencial Absoluta em 250 hPa',
        'titulo_en': '250-hPa geopotential height',
        'rotulo_box': 'Geopotential Height',
        'subtitulo_dir': 'Geopotential height at 250 hPa',
        'unidade': 'mgp',
        'absoluto': True,          # sem subtracao de climatologia
        'simetrico': False,        # nao centrado em 0 (e altura absoluta, nao anomalia)
        'sinotico_mp4': True,      # MP4 usa passos de HORA SINOTICA (00/06/12/18Z), nao media diaria
        # vmin/vmax/niveis/paleta/opacidade -- tudo overridavel em settings.local.toml sem tocar
        # em codigo (mesmo mecanismo generico ja usado por outras variaveis, ex. jet_stream).
        'vmin': float(settings.get('GLOBO_3D_VMIN_Z250_ABS', 4800.0)),
        'vmax': float(settings.get('GLOBO_3D_VMAX_Z250_ABS', 11200.0)),
        'niveis': int(settings.get('GLOBO_3D_NIVEIS_Z250_ABS', 50)),
        'cmap_colors': list(settings.get('GLOBO_3D_PALETA_Z250_ABS', [
            'purple', 'blueviolet', 'mediumpurple', 'mediumblue', 'aqua',
            'limegreen', 'yellow', 'orange', 'firebrick', 'darkred',
        ])),
        'shaded_alpha': float(settings.get('GLOBO_3D_ALPHA_Z250_ABS', 1.0)),
        'cor_oceano': str(settings.get('GLOBO_3D_COR_OCEANO_Z250_ABS', '#0e426d')),
        # Costa/fronteiras/estados PRETOS e mais grossos (contraste com o blue marble por baixo).
        # Default do estilo guillaume (s39/s41/s42) seria cinza #444444 -- sobrescrito aqui.
        'cor_fronteiras': str(settings.get('GLOBO_3D_COR_FRONTEIRAS_Z250_ABS', 'black')),
        'lw_coast':  float(settings.get('GLOBO_3D_LW_COAST_Z250_ABS', 1.0)),
        'lw_border': float(settings.get('GLOBO_3D_LW_BORDER_Z250_ABS', 0.8)),
        'lw_states': float(settings.get('GLOBO_3D_LW_STATES_Z250_ABS', 0.6)),
        'spec': {
            'nome': 'z250_abs', 'unidade': 'mgp', 'celsius': False,
            'var_candidates': HGT_VARS, 'kind': 'hgt250',
            'era5_fn': _era5_z250, 'gdas_fn': _gdas_z250,
        },
    },
    'z500_abs': {
        'titulo': 'Altura Geopotencial Absoluta em 500 hPa',
        'titulo_en': '500-hPa geopotential height',
        'rotulo_box': 'Geopotential Height',
        'subtitulo_dir': 'Geopotential height at 500 hPa',
        'unidade': 'mgp',
        'absoluto': True,          # sem subtracao de climatologia
        'simetrico': False,        # nao centrado em 0 (e altura absoluta, nao anomalia)
        'sinotico_mp4': True,      # MP4 usa passos de HORA SINOTICA (00/06/12/18Z), nao media diaria
        # Faixa de valores calibrada pela carta de referencia (GFS 500 hPa, 468-600 dam = 4680-6000
        # mgp). Paleta igual a do z250_abs (mesmas cores, override proprio p/ nao acoplar as duas).
        'vmin': float(settings.get('GLOBO_3D_VMIN_Z500_ABS', 4680.0)),
        'vmax': float(settings.get('GLOBO_3D_VMAX_Z500_ABS', 6000.0)),
        'niveis': int(settings.get('GLOBO_3D_NIVEIS_Z500_ABS', 50)),
        'cmap_colors': list(settings.get('GLOBO_3D_PALETA_Z500_ABS', [
            'purple', 'blueviolet', 'mediumpurple', 'mediumblue', 'aqua',
            'limegreen', 'yellow', 'orange', 'firebrick', 'darkred',
        ])),
        'shaded_alpha': float(settings.get('GLOBO_3D_ALPHA_Z500_ABS', 1.0)),
        'cor_oceano': str(settings.get('GLOBO_3D_COR_OCEANO_Z500_ABS', '#0e426d')),
        'cor_fronteiras': str(settings.get('GLOBO_3D_COR_FRONTEIRAS_Z500_ABS', 'black')),
        'lw_coast':  float(settings.get('GLOBO_3D_LW_COAST_Z500_ABS', 1.0)),
        'lw_border': float(settings.get('GLOBO_3D_LW_BORDER_Z500_ABS', 0.8)),
        'lw_states': float(settings.get('GLOBO_3D_LW_STATES_Z500_ABS', 0.6)),
        'spec': {
            'nome': 'z500_abs', 'unidade': 'mgp', 'celsius': False,
            # kind='hgt500' (nao 'hgt250'): o jato sempre usa Z250 dedicado p/ a isolinha-guia,
            # baixado a parte independente da variavel shaded (ver _render_clip).
            'var_candidates': HGT_VARS, 'kind': 'hgt500',
            'era5_fn': _era5_z500, 'gdas_fn': _gdas_z500,
        },
    },
    # Ficha "vazia" -- SO s42/s43 (ver _get_variaveis dos dois scripts): quando
    # GLOBO_3D_VARIAVEIS_S42/S43 vem [] (lista vazia), plota o globo/mapa 2D SO com as features
    # habilitadas (jato, icones de pressao, caixas de texto, credito), SEM nenhum campo shaded.
    # Reusa o download de Z500 (mesmo spec do z500_abs) so pra ter grade/eixo de tempo validos e
    # servir de campo-guia do jato de graca (kind='hgt500' cai no atalho "_e_z250_absoluto" que
    # reaproveita o proprio campo, sem baixar Z250 a parte) -- 'sem_variavel': True faz o render
    # pular o desenho do shaded e a caixa de titulo/legenda (ver _build_frame/_overlay_guillaume).
    'sem_variavel': {
        'titulo': '', 'titulo_en': '', 'rotulo_box': '', 'subtitulo_dir': '',
        'unidade': '', 'absoluto': True, 'simetrico': False,
        'sem_variavel': True,
        'vmin': 0.0, 'vmax': 1.0, 'niveis': 2,
        'cmap_colors': ['#000000', '#000000'],
        'shaded_alpha': 0.0,
        # Sem isso, o motor cai no default generico (#444444 fino) -- quase invisivel sobre o
        # blue marble. Mesmos valores do z250_abs/z500_abs (contorno preto, bem visivel).
        'cor_fronteiras': str(settings.get('GLOBO_3D_COR_FRONTEIRAS_SEM_VARIAVEL', 'black')),
        'lw_coast':  float(settings.get('GLOBO_3D_LW_COAST_SEM_VARIAVEL', 1.0)),
        'lw_border': float(settings.get('GLOBO_3D_LW_BORDER_SEM_VARIAVEL', 0.8)),
        'lw_states': float(settings.get('GLOBO_3D_LW_STATES_SEM_VARIAVEL', 0.6)),
        # Cor SOLIDA do oceano (sem shaded nenhum sobre a agua, ja que 'sem_variavel' nao
        # desenha nenhum campo) -- ajustavel livremente, independente da variavel do s38-s43.
        'cor_oceano': str(settings.get('GLOBO_3D_COR_OCEANO_SEM_VARIAVEL', '#0e426d')),
        'spec': {
            'nome': 'sem_variavel', 'unidade': '', 'celsius': False,
            'var_candidates': HGT_VARS, 'kind': 'hgt500',
            'era5_fn': _era5_z500, 'gdas_fn': _gdas_z500,
        },
    },
    # Agua precipitavel (PWAT) -- campo ABSOLUTO (sem climatologia), so nos modelos FISICOS
    # (GFS/GEFS/ECMWF/ECMWF_ENS/CFS); AIFS/AIFS_ENS/AIGFS/AIGEFS nao publicam esse campo.
    # Faixa 0-70mm, passo de 0.5mm (140 bandas) -> len(paleta)=14 != niveis=140 cai no caminho
    # LinearSegmentedColormap (gradiente CONTINUO, nao blocos discretos). Paleta com posicao
    # EXPLICITA (fracao 0..1, nao uniforme): 7 tons terrosos de 0 a 20mm (fracao 0..20/70) e 7
    # tons de verde de 20 a 70mm (fracao 20/70..1) -- a transicao terra->verde cai EXATAMENTE
    # nos 20mm (dois stops na mesma fracao 0.2857 = salto limpo, sem interpolar terra<->verde).
    # Colormap.__call__ ja clampa fora de [0,1] na 1a/ultima cor por default (sem estouro).
    'pwat_abs': {
        'titulo': 'Água Precipitável', 'titulo_en': 'Precipitable Water',
        'rotulo_box': 'Precipitable Water',
        'subtitulo_dir': 'Precipitable water (entire atmosphere)',
        'unidade': 'mm',
        'absoluto': True, 'simetrico': False,
        'legenda_numerica': True, 'legenda_num_step': 5.0,
        'vmin': float(settings.get('GLOBO_3D_VMIN_PWAT_ABS', 0.0)),
        'vmax': float(settings.get('GLOBO_3D_VMAX_PWAT_ABS', 70.0)),
        'niveis': int(settings.get('GLOBO_3D_NIVEIS_PWAT_ABS', 140)),
        # Verdes com posicao NAO-uniforme (20,23,26,29,32,35,70mm): o verde escuro (169f15/077205)
        # fica concentrado a partir de 35mm (169f15 comeca exatamente em 35, vai ate 077205 em 70);
        # os verdes mais claros ficam comprimidos entre 20-35mm.
        'cmap_colors': list(settings.get('GLOBO_3D_PALETA_PWAT_ABS', [
            (0.0,    '#debd70'), (0.0476, '#a78a4f'), (0.0952, '#745c37'), (0.1429, '#4c3521'),
            (0.1905, '#3b2919'), (0.2381, '#7b7666'), (0.2857, '#bfc8ba'),
            (0.2857, '#d5ebd5'), (0.3286, '#aee8ae'), (0.3714, '#6feb75'), (0.4143, '#35e935'),
            (0.4571, '#2ad02b'), (0.5,    '#169f15'), (1.0,    '#077205'),
        ])),
        'shaded_alpha': float(settings.get('GLOBO_3D_ALPHA_PWAT_ABS', 1.0)),
        'cor_oceano': str(settings.get('GLOBO_3D_COR_OCEANO_PWAT_ABS', '#0e426d')),
        'cor_fronteiras': str(settings.get('GLOBO_3D_COR_FRONTEIRAS_PWAT_ABS', 'black')),
        'lw_coast':  float(settings.get('GLOBO_3D_LW_COAST_PWAT_ABS', 1.0)),
        'lw_border': float(settings.get('GLOBO_3D_LW_BORDER_PWAT_ABS', 0.8)),
        'lw_states': float(settings.get('GLOBO_3D_LW_STATES_PWAT_ABS', 0.6)),
        'spec': {
            'nome': 'pwat_abs', 'unidade': 'mm', 'celsius': False,
            'var_candidates': ('pwat', 'tcwv'), 'kind': 'pwat',
            'era5_fn': _era5_pwat, 'gdas_fn': _gdas_pwat,
        },
    },
    # CHUVA (precipitacao ACUMULADA DIARIA, mm) -- campo ABSOLUTO, FORECAST-ONLY (GFS + ECMWF-HRES).
    # NAO e PWAT. Levels NAO-uniformes (paleta do usuario): abaixo de 1 mm = TRANSPARENTE (extend='max'
    # nao preenche abaixo do 1o nivel -> area seca revela o fundo). 6 bandas discretas (1 cor por banda
    # -> ListedColormap exato). Usa CONTOURF (bandas suaves) mesmo no s46 (GLOBO_3D_PCOLORMESH_PRECIP_ABS
    # =false), pois o s46 e' gouraud por padrao. A serie ja e' diaria (1 valor/dia = o total).
    'precip_abs': {
        'titulo': 'Chuva', 'titulo_en': 'Precipitation',
        'rotulo_box': 'Precipitation',
        'subtitulo_dir': 'Daily accumulated precipitation',
        'unidade': 'mm',
        'absoluto': True, 'simetrico': False,
        'levels': list(settings.get('GLOBO_3D_LEVELS_PRECIP_ABS',
                                    [1, 25, 50, 75, 125, 200, 250, 300, 350, 400, 450, 500])),
        'cmap_colors': list(settings.get('GLOBO_3D_PALETA_PRECIP_ABS', [
            '#23c412', '#209511', '#1f7311', '#fec60f', '#ff6a1a', '#ff0d04',
            '#fd41d3', '#fdb3fe', '#ebebeb', '#d3d3d3', '#bbbbbb',
        ])),
        'extend_contourf': 'max',  # abaixo do 1o nivel (1 mm) sem preenchimento = transparente (seco)
        'shaded_alpha': float(settings.get('GLOBO_3D_ALPHA_PRECIP_ABS', 1.0)),
        'cor_oceano': (None if str(settings.get('GLOBO_3D_COR_OCEANO_PRECIP_ABS', 'none')).strip().lower()
                       in ('none', '') else str(settings.get('GLOBO_3D_COR_OCEANO_PRECIP_ABS'))),
        'cor_fronteiras': str(settings.get('GLOBO_3D_COR_FRONTEIRAS_PRECIP_ABS', 'black')),
        'lw_coast':  float(settings.get('GLOBO_3D_LW_COAST_PRECIP_ABS', 1.0)),
        'lw_border': float(settings.get('GLOBO_3D_LW_BORDER_PRECIP_ABS', 0.8)),
        'lw_states': float(settings.get('GLOBO_3D_LW_STATES_PRECIP_ABS', 0.6)),
        # Camada OPCIONAL de neve por cima (ver `NEVE` no settings + bloco de overlay em
        # `gerar_animacao`/`_render_clip`): a chuva em si nao muda, so ganha uma 2a serie sombreada
        # (`neve_levels`/`neve_cmap_colors`) classificada por tipo de precipitacao (ECMWF `ptype` ou
        # GFS `CSNOW`, ver `_ptype_neve_forecast_series`). Propositalmente NAO um campo/ficha a parte
        # (duplicar o pipeline de chuva descolava do precip_abs: sumia o blue marble, isolinha extra).
        'neve_levels': list(settings.get('GLOBO_3D_NEVE_LEVELS',
                                         [1, 3, 5, 10, 20, 30, 50, 75, 100])),
        'neve_cmap_colors': list(settings.get('GLOBO_3D_NEVE_PALETA', [
            '#07b5d7', '#058fb2', '#097591', '#9187ef',
            '#8c59e9', '#5f2ab0', '#ea478b', '#eca6e5',
        ])),
        'spec': {
            'nome': 'precip_abs', 'unidade': 'mm', 'celsius': False,
            'var_candidates': ('precip', 'tp', 'apcp'), 'kind': 'precip',
            'era5_fn': _precip_reanalise_unsupported, 'gdas_fn': _precip_reanalise_unsupported,
        },
    },
    # RAJADA DE VENTO a 10 m (MAXIMA DIARIA, km/h) -- campo ABSOLUTO, FORECAST-ONLY (so ECMWF-HRES:
    # unica fonte com 10fg no s47). A serie ja e' diaria (1 valor/dia = o max do dia); o downloader
    # converte m/s->km/h. abaixo do 1o nivel = TRANSPARENTE (extend='max'). Isolinhas de PNMM OPCIONAIS
    # por cima (GLOBO_3D_ISOLINHA_MSLP_RAJADA_ABS, default false) -- media diaria de `msl` do ECMWF.
    'rajada_abs': {
        'titulo': 'Rajada de Vento', 'titulo_en': 'Wind Gust',
        'rotulo_box': 'Wind Gust',
        'subtitulo_dir': 'Daily maximum 10 m wind gust',
        'unidade': 'km/h',
        'absoluto': True, 'simetrico': False,
        'levels': list(settings.get('GLOBO_3D_LEVELS_RAJADA_ABS',
                                    [40, 50, 60, 70, 80, 90, 100, 120, 140, 160])),
        'cmap_colors': list(settings.get('GLOBO_3D_PALETA_RAJADA_ABS', [
            '#3ac5ff', '#0037fe', '#fd38e8', '#8b0100', '#b81310',
            '#e31302', '#eea5bd', '#ffe6ff', '#ceffff',
        ])),
        'extend_contourf': 'max',   # abaixo do 1o nivel (40 km/h) sem preenchimento = transparente
        'shaded_alpha': float(settings.get('GLOBO_3D_ALPHA_RAJADA_ABS', 1.0)),
        'cor_oceano': (None if str(settings.get('GLOBO_3D_COR_OCEANO_RAJADA_ABS', 'none')).strip().lower()
                       in ('none', '') else str(settings.get('GLOBO_3D_COR_OCEANO_RAJADA_ABS'))),
        'cor_fronteiras': str(settings.get('GLOBO_3D_COR_FRONTEIRAS_RAJADA_ABS', 'black')),
        'lw_coast':  float(settings.get('GLOBO_3D_LW_COAST_RAJADA_ABS', 1.0)),
        'lw_border': float(settings.get('GLOBO_3D_LW_BORDER_RAJADA_ABS', 0.8)),
        'lw_states': float(settings.get('GLOBO_3D_LW_STATES_RAJADA_ABS', 0.6)),
        # PNMM em isolinhas por cima: LIGADA/DESLIGADA pelo settings (o usuario decide se plota).
        'isolinha_mslp': bool(settings.get('GLOBO_3D_ISOLINHA_MSLP_RAJADA_ABS', False)),
        'spec': {
            'nome': 'rajada_abs', 'unidade': 'km/h', 'celsius': False,
            'var_candidates': ('gust', '10fg', 'i10fg', 'fg10'), 'kind': 'gust',
            'era5_fn': _gust_reanalise_unsupported, 'gdas_fn': _gust_reanalise_unsupported,
        },
    },
    'psi200_anom': {
        'titulo': 'Anomalia da Função de Corrente em 200 hPa',
        'titulo_en': '200-hPa streamfunction',
        'rotulo_box': 'Streamfunction Anomaly',
        'subtitulo_dir': 'Streamfunction at 200 hPa',
        'unidade': '10⁶ m² s⁻¹',
        # Paleta da anomalia de Z250 (s38/s39) INVERTIDA: tons FRIOS (azul/roxo) -> psi POSITIVO
        # e tons QUENTES (vermelho/laranja) -> psi NEGATIVO, conforme a convencao do HS pedida.
        'cmap_colors': list(reversed(_PALETA_ANOM_HGT250)),
        'niveis': 40,           # 40 bandas em ±50 -> passo de 2,5 ×10⁶ m²/s (contorno branco idem)
        'simetrico': True,
        'vmax': float(settings.get('GLOBO_3D_VMAX_PSI200_ANOM', 50.0)),
        'pentada_movel': 5,     # cada frame = media movel de 5 dias [d, d+4] (espelha o s34)
        'spec': {
            'nome': 'psi200_anom', 'unidade': '10⁶ m² s⁻¹', 'celsius': False,
            'var_candidates': U_VARS, 'clim_fn': None, 'kind': 'fcst200',
            'reanalise_fn': _psi200_reanalise_series,
            'forecast_fn': _psi200_forecast_series,
            'era5_fn': None, 'gdas_fn': None,
        },
    },
    'chi200_anom': {
        'titulo': 'Anomalia do Potencial de Velocidade em 200 hPa',
        'titulo_en': '200-hPa velocity potential',
        'rotulo_box': 'Velocity Potential Anomaly',
        'subtitulo_dir': 'Velocity potential at 200 hPa',
        'unidade': '10⁵ m² s⁻¹',
        # Paleta verde->bege->marrom (s03): VERDE = chi NEGATIVO (divergencia/conveccao),
        # MARROM = chi POSITIVO (subsidencia).
        'cmap_colors': _PALETA_CHI200,
        'niveis': 20,           # ±100 em 20 bandas -> passo de 10 ×10⁵ m²/s (contorno branco idem)
        'simetrico': True,
        'vmax': float(settings.get('GLOBO_3D_VMAX_CHI200_ANOM', 100.0)),
        'pentada_movel': 5,     # cada frame = media movel de 5 dias [d, d+4] (espelha o s34)
        'spec': {
            'nome': 'chi200_anom', 'unidade': '10⁵ m² s⁻¹', 'celsius': False,
            'var_candidates': U_VARS, 'clim_fn': None, 'kind': 'fcst200',
            'reanalise_fn': _chi200_reanalise_series,
            'forecast_fn': _chi200_forecast_series,
            'era5_fn': None, 'gdas_fn': None,
        },
    },
    'chi200_cores_psi200_contornos': {
        'titulo': 'Anomalia de Potencial de Velocidade (cores) e Função de Corrente (linhas) em 200 hPa',
        'titulo_en': '200-hPa velocity potential (shaded) with streamfunction contours',
        'rotulo_box': 'Velocity Potential & Streamfunction',
        'subtitulo_dir': 'Velocity potential (shaded) + streamfunction (lines) at 200 hPa',
        'unidade': '10⁵ m² s⁻¹',
        # SHADED = chi200 (mesma paleta/escala do chi200_anom: verde->marrom, ±100, passo 10).
        'cmap_colors': _PALETA_CHI200,
        'niveis': 20,
        'simetrico': True,
        'vmax': float(settings.get('GLOBO_3D_VMAX_CHI200_ANOM', 100.0)),
        'pentada_movel': 5,     # chi200 shaded em media movel de 5 dias
        # ISOLINHAS PRETAS = psi200 (funcao de corrente), tambem em pentada movel (a ficha do
        # psi200_anom ja aplica os 5 dias). SEM contorno branco no shaded (GLOBO_3D_CONTORNO_<VAR>
        # fica desligado por default p/ esta variavel).
        'contorno_serie_var': 'psi200_anom',
        'contorno_serie_cor': 'black',
        'contorno_serie_intervalo': 5.0,   # 10⁶ m²/s entre isolinhas de psi200
        'contorno_serie_lw': 0.5,
        'spec': {
            'nome': 'chi200_cores_psi200_contornos', 'unidade': '10⁵ m² s⁻¹', 'celsius': False,
            'var_candidates': U_VARS, 'clim_fn': None, 'kind': 'fcst200',
            'reanalise_fn': _chi200_reanalise_series,
            'forecast_fn': _chi200_forecast_series,
            'era5_fn': None, 'gdas_fn': None,
        },
    },
    'wnd200_zonal_cores_psi200_contornos': {
        'titulo': 'Anomalia de Vento Zonal (cores) e Função de Corrente (linhas) em 200 hPa',
        'titulo_en': '200-hPa zonal wind (shaded) with streamfunction contours',
        'rotulo_box': 'Zonal Wind & Streamfunction',
        'subtitulo_dir': 'Zonal wind (shaded) + streamfunction (lines) at 200 hPa',
        'unidade': 'ms⁻¹',
        # SHADED = anomalia de vento zonal 200 hPa, MESMA paleta/levels do wnd250_zonal_anom
        # (_PALETA_ANOM_HGT250, 30 niveis, ±20 m/s). SÓ media movel de 5 dias (como olr_cores_*).
        'cmap_colors': _PALETA_ANOM_HGT250,
        'niveis': 30,
        'simetrico': True,
        'vmax': float(settings.get('GLOBO_3D_VMAX_WND200_ZONAL_ANOM', 20.0)),
        'pentada_movel': 5,     # vento zonal 200 shaded em media movel de 5 dias
        # ISOLINHAS PRETAS = psi200 (a ficha do psi200_anom ja aplica os 5 dias).
        'contorno_serie_var': 'psi200_anom',
        'contorno_serie_cor': 'black',
        'contorno_serie_intervalo': 5.0,   # 10⁶ m²/s entre isolinhas de psi200
        'contorno_serie_lw': 0.5,
        'spec': {
            'nome': 'wnd200_zonal_cores_psi200_contornos', 'unidade': 'ms⁻¹', 'celsius': False,
            'var_candidates': U_VARS, 'clim_fn': None, 'kind': 'fcst200',
            'reanalise_fn': _wnd200_zonal_reanalise_series,
            'forecast_fn': _wnd200_zonal_forecast_series,
            'era5_fn': None, 'gdas_fn': None,
        },
    },
    'wnd200_meridional_cores_psi200_contornos': {
        'titulo': 'Anomalia de Vento Meridional (cores) e Função de Corrente (linhas) em 200 hPa',
        'titulo_en': '200-hPa meridional wind (shaded) with streamfunction contours',
        'rotulo_box': 'Meridional Wind &\nStreamfunction',   # '\n' força "MERIDIONAL WIND &" / "STREAMFUNCTION"
        'subtitulo_dir': 'Meridional wind (shaded) + streamfunction (lines) at 200 hPa',
        'unidade': 'ms⁻¹',
        # SHADED = anomalia de vento MERIDIONAL 200 hPa, MESMA paleta/levels do vento zonal
        # (_PALETA_ANOM_HGT250, 30 niveis, ±20 m/s). SÓ media movel de 5 dias (como olr_cores_*).
        'cmap_colors': _PALETA_ANOM_HGT250,
        'niveis': 30,
        'simetrico': True,
        'vmax': float(settings.get('GLOBO_3D_VMAX_WND200_MERIDIONAL_ANOM', 20.0)),
        'pentada_movel': 5,     # vento meridional 200 shaded em media movel de 5 dias
        # ISOLINHAS PRETAS = psi200 (a ficha do psi200_anom ja aplica os 5 dias).
        'contorno_serie_var': 'psi200_anom',
        'contorno_serie_cor': 'black',
        'contorno_serie_intervalo': 5.0,   # 10⁶ m²/s entre isolinhas de psi200
        'contorno_serie_lw': 0.5,
        'spec': {
            'nome': 'wnd200_meridional_cores_psi200_contornos', 'unidade': 'ms⁻¹', 'celsius': False,
            'var_candidates': V_VARS, 'clim_fn': None, 'kind': 'fcst200',
            'reanalise_fn': _wnd200_meridional_reanalise_series,
            'forecast_fn': _wnd200_meridional_forecast_series,
            'era5_fn': None, 'gdas_fn': None,
        },
    },
    'chi200_cores_z250_contornos': {
        'titulo': 'Anomalia de Potencial de Velocidade em 200 hPa (cores) e Altura Geopotencial em 250 hPa (linhas)',
        'titulo_en': '200-hPa velocity potential (shaded) with 250-hPa geopotential height contours',
        'rotulo_box': 'Velocity Potential & 250-hPa Height',
        'subtitulo_dir': 'Velocity potential at 200 hPa (shaded) + 250-hPa height (lines)',
        'unidade': '10⁵ m² s⁻¹',
        # SHADED = chi200 (mesma paleta/escala do chi200_anom: verde->marrom, ±100, passo 10). SEM
        # contorno branco no shaded (GLOBO_3D_CONTORNO_<VAR> fica desligado por default aqui).
        'cmap_colors': _PALETA_CHI200,
        'niveis': 20,
        'simetrico': True,
        'vmax': float(settings.get('GLOBO_3D_VMAX_CHI200_ANOM', 100.0)),
        'pentada_movel': 5,     # chi200 shaded em media movel de 5 dias
        # ISOLINHAS PRETAS = anomalia de Z250 (altura geopotencial 250 hPa), TAMBEM em pentada movel
        # de 5 dias — via contorno_serie_pentada, pois a ficha z250_anom nao tem pentada propria.
        'contorno_serie_var': 'z250_anom',
        'contorno_serie_pentada': 5,
        'contorno_serie_cor': 'black',
        'contorno_serie_intervalo': 40.0,   # mgp entre isolinhas de Z250 anomalia
        'contorno_serie_lw': 0.5,
        'spec': {
            'nome': 'chi200_cores_z250_contornos', 'unidade': '10⁵ m² s⁻¹', 'celsius': False,
            'var_candidates': U_VARS, 'clim_fn': None, 'kind': 'fcst200',
            'reanalise_fn': _chi200_reanalise_series,
            'forecast_fn': _chi200_forecast_series,
            'era5_fn': None, 'gdas_fn': None,
        },
    },
    'wnd250_zonal_anom': {
        'titulo': 'Anomalia de Vento Zonal em 250 hPa',
        'titulo_en': '250-hPa zonal wind',
        'rotulo_box': 'Zonal Wind Anomaly',
        'subtitulo_dir': 'Zonal wind at 250 hPa',
        'unidade': 'ms⁻¹',
        'cmap_colors': _PALETA_ANOM_HGT250,  # mesma paleta da anomalia de hgt250
        'niveis': 30,
        'simetrico': True,
        'vmax': float(settings.get('GLOBO_3D_VMAX_WND250_ZONAL_ANOM', 20.0)),
        'spec': {
            'nome': 'wnd250_zonal_anom', 'unidade': 'ms⁻¹', 'celsius': False,
            'var_candidates': U_VARS, 'clim_fn': clim_u250_daily, 'kind': 'uv250',
            'era5_fn': _era5_uv250, 'gdas_fn': _gdas_uv250,
        },
    },
    'wnd850_meridional_anom': {
        'titulo': 'Anomalia de Vento Meridional em 850 hPa',
        'titulo_en': '850-hPa meridional wind',
        'rotulo_box': 'Meridional Wind Anomaly',
        'subtitulo_dir': 'Meridional wind at 850 hPa',
        'unidade': 'ms⁻¹',
        'cmap_colors': _PALETA_ANOM_HGT250,  # mesma paleta da anomalia de hgt250
        'niveis': 30,
        'simetrico': True,
        'vmax': float(settings.get('GLOBO_3D_VMAX_WND850_MERIDIONAL_ANOM', 8.0)),
        'spec': {
            'nome': 'wnd850_meridional_anom', 'unidade': 'ms⁻¹', 'celsius': False,
            'var_candidates': V_VARS, 'clim_fn': clim_v850_daily, 'kind': 'uv850',
            'era5_fn': _era5_uv850, 'gdas_fn': _gdas_uv850,
        },
    },
    'wnd850_zonal_anom': {
        'titulo': 'Anomalia de Vento Zonal em 850 hPa',
        'titulo_en': '850-hPa zonal wind',
        'rotulo_box': 'Zonal Wind Anomaly',
        'subtitulo_dir': 'Zonal wind at 850 hPa',
        'unidade': 'ms⁻¹',
        'cmap_colors': _PALETA_ANOM_HGT250,  # mesma paleta da anomalia de hgt250
        'niveis': 30,
        'simetrico': True,
        'vmax': float(settings.get('GLOBO_3D_VMAX_WND850_ZONAL_ANOM', 10.0)),
        'spec': {
            'nome': 'wnd850_zonal_anom', 'unidade': 'ms⁻¹', 'celsius': False,
            'var_candidates': U_VARS, 'clim_fn': clim_u850_daily, 'kind': 'uv850',
            'era5_fn': _era5_uv850, 'gdas_fn': _gdas_uv850,
        },
    },
    'tmp850_anom': {
        'titulo': 'Anomalia de Temperatura do Ar em 850 hPa',
        'titulo_en': '850-hPa air temperature',
        'rotulo_box': 'Air Temperature Anomaly',  # caixa do s39 (curto, ingles)
        'subtitulo_dir': 'Air temperature at 850 hPa',  # canto sup-dir do s39 (variavel + nivel)
        'unidade': '°C',
        'isolinha_abs_0': True,   # desenha isolinha branca onde T850 absoluta = 0°C
        # Paleta amostrada PIXEL A PIXEL da barra de referencia (Entrada/paleta.jpg),
        # magenta (frio extremo) -> ... -> cinza-carvao (quente extremo).
        'cmap_colors': _PALETA_T850_ANOM,
        'niveis': 128,       # bandas do shaded (suave, ~2x mais rapido que 256) (override: GLOBO_3D_NIVEIS_TMP850_ANOM)
        'simetrico': True,
        'vmax': float(settings.get('GLOBO_3D_VMAX_TMP850_ANOM', 15.0)),  # escala FIXA ±15 °C (s38/s39); s42 usa ±10 via GLOBO_3D_VMAX_TMP850_ANOM_S42
        'spec': {
            'nome': 'tmp850_anom', 'unidade': '°C', 'celsius': True,
            'var_candidates': TMP_VARS, 'clim_fn': clim_t850_daily, 'kind': 'tmp850',
            'era5_fn': _era5_t850, 'gdas_fn': _gdas_t850,
        },
    },
    'tmp850_mslp': {
        'titulo': 'Anomalia de Temperatura do Ar em 850 hPa + PNMM',
        'titulo_en': '850-hPa air temperature',
        'rotulo_box': 'Air Temperature Anomaly',
        'subtitulo_dir': 'Air temperature at 850 hPa + MSLP',
        'unidade': '°C',
        'isolinha_abs_0': True,
        'isolinha_mslp': True,   # isolinhas de PNMM em hPa (ERA5 apenas)
        'cmap_colors': _PALETA_T850_ANOM,
        'niveis': 128,
        'simetrico': True,
        'vmax': 15.0,
        'spec': {
            'nome': 'tmp850_mslp', 'unidade': '°C', 'celsius': True,
            'var_candidates': TMP_VARS, 'clim_fn': clim_t850_daily, 'kind': 'tmp850',
            'era5_fn': _era5_t850, 'gdas_fn': _gdas_t850,
        },
    },
    'tmp850_cores_psi200_contornos': {
        'titulo': 'Anomalia de Temperatura do Ar em 850 hPa (cores) e Função de Corrente em 200 hPa (linhas)',
        'titulo_en': '850-hPa air temperature (shaded) with 200-hPa streamfunction contours',
        'rotulo_box': 'Air Temperature & Streamfunction',
        'subtitulo_dir': 'Air temperature at 850 hPa (shaded) + 200-hPa streamfunction (lines)',
        'unidade': '°C',
        # SHADED = T850 (mesma paleta/escala do tmp850_anom: -15..+15 °C, 128 bandas).
        'cmap_colors': _PALETA_T850_ANOM,
        'niveis': 128,
        'simetrico': True,
        'vmax': 15.0,
        'pentada_movel': 5,     # T850 shaded em media movel de 5 dias
        # ISOLINHAS PRETAS = psi200 (funcao de corrente), tambem em media movel de 5 dias (a
        # ficha do psi200_anom ja aplica os 5 dias). SEM contorno branco no shaded.
        'contorno_serie_var': 'psi200_anom',
        'contorno_serie_cor': 'black',
        'contorno_serie_intervalo': 5.0,   # 10⁶ m²/s entre isolinhas de psi200
        'contorno_serie_lw': 0.5,
        'spec': {
            'nome': 'tmp850_cores_psi200_contornos', 'unidade': '°C', 'celsius': True,
            'var_candidates': TMP_VARS, 'clim_fn': clim_t850_daily, 'kind': 'tmp850',
            'era5_fn': _era5_t850, 'gdas_fn': _gdas_t850,
        },
    },
    'olr_anom': {
        'titulo': 'Anomalia de Radiação de Onda Longa Emergente (OLR)',
        'titulo_en': 'Outgoing longwave radiation',
        'rotulo_box': 'OLR Anomaly',
        'subtitulo_dir': 'Outgoing longwave radiation (OLR)',
        'unidade': 'W/m²',
        # Paleta BrBG_r (igual ao s05): azul-esverdeado (OLR- = mais convecção = úmido) →
        # branco (neutro) → marrom (OLR+ = convecção suprimida = seco)
        'cmap_colors': _PALETA_OLR_ANOM,
        'niveis': 20,
        'simetrico': True,
        'vmax': float(settings.get('GLOBO_3D_VMAX_OLR_ANOM', 40.0)),
        'sem_clim_ref': True,  # OLR usa fonte CPC/PSL, não ERA5 — omite "Relative to 1991-2020"
        'spec': {
            'nome': 'olr_anom', 'unidade': 'W/m²', 'celsius': False,
            'var_candidates': OLR_VARS, 'clim_fn': None, 'kind': 'olr',
            'reanalise_fn': _olr_reanalise_series,
            'era5_fn': None, 'gdas_fn': None,
        },
    },
    'olr_cores_psi200_contornos': {
        'titulo': 'Anomalia de OLR (cores) e Função de Corrente em 200 hPa (linhas)',
        'titulo_en': 'Outgoing longwave radiation (shaded) with 200-hPa streamfunction contours',
        'rotulo_box': 'OLR & Streamfunction',
        'subtitulo_dir': 'OLR (shaded) + 200-hPa streamfunction (lines)',
        'unidade': 'W/m²',
        # SHADED = OLR (mesma paleta/escala do olr_anom: BrBG_r, ±GLOBO_3D_VMAX_OLR_ANOM).
        'cmap_colors': _PALETA_OLR_ANOM,
        'niveis': 20,
        'simetrico': True,
        'vmax': float(settings.get('GLOBO_3D_VMAX_OLR_ANOM', 40.0)),
        'sem_clim_ref': True,   # OLR usa CPC/PSL, nao ERA5
        'pentada_movel': 5,     # OLR shaded em media movel de 5 dias
        # ISOLINHAS PRETAS = psi200 (funcao de corrente), tambem em pentada movel. SEM contorno
        # branco no shaded (GLOBO_3D_CONTORNO_<VAR> desligado por default p/ esta variavel).
        'contorno_serie_var': 'psi200_anom',
        'contorno_serie_cor': 'black',
        'contorno_serie_intervalo': 5.0,   # 10⁶ m²/s entre isolinhas de psi200
        'contorno_serie_lw': 0.5,
        'spec': {
            'nome': 'olr_cores_psi200_contornos', 'unidade': 'W/m²', 'celsius': False,
            'var_candidates': OLR_VARS, 'clim_fn': None, 'kind': 'olr',
            'reanalise_fn': _olr_reanalise_series,
            'era5_fn': None, 'gdas_fn': None,
        },
    },
    'olr_cores_z250_contornos': {
        'titulo': 'Anomalia de OLR (cores) e Altura Geopotencial em 250 hPa (linhas)',
        'titulo_en': 'Outgoing longwave radiation (shaded) with 250-hPa geopotential height contours',
        'rotulo_box': 'OLR & 250-hPa Height',
        'subtitulo_dir': 'OLR (shaded) + 250-hPa geopotential height (lines)',
        'unidade': 'W/m²',
        # SHADED = OLR (mesma paleta/escala do olr_anom: BrBG_r, ±GLOBO_3D_VMAX_OLR_ANOM). SEM
        # contorno branco no shaded.
        'cmap_colors': _PALETA_OLR_ANOM,
        'niveis': 20,
        'simetrico': True,
        'vmax': float(settings.get('GLOBO_3D_VMAX_OLR_ANOM', 40.0)),
        'sem_clim_ref': True,   # OLR usa CPC/PSL, nao ERA5
        'pentada_movel': 5,     # OLR shaded em media movel de 5 dias
        # ISOLINHAS PRETAS = anomalia de Z250, TAMBEM em pentada movel de 5 dias (contorno_serie_pentada).
        'contorno_serie_var': 'z250_anom',
        'contorno_serie_pentada': 5,
        'contorno_serie_cor': 'black',
        'contorno_serie_intervalo': 40.0,   # mgp entre isolinhas de Z250 anomalia
        'contorno_serie_lw': 0.5,
        'spec': {
            'nome': 'olr_cores_z250_contornos', 'unidade': 'W/m²', 'celsius': False,
            'var_candidates': OLR_VARS, 'clim_fn': None, 'kind': 'olr',
            'reanalise_fn': _olr_reanalise_series,
            'era5_fn': None, 'gdas_fn': None,
        },
    },
    'tsm_anom': {
        'titulo': 'Anomalia de Temperatura da Superfície do Mar (TSM)',
        'titulo_en': 'Sea surface temperature',
        'rotulo_box': 'Sea Surface Temperature Anomaly',
        'subtitulo_dir': 'Sea surface temperature (SST)',
        'unidade': '°C',
        # ── Paleta ATIVA: padrao do projeto LST_ANOM_CORRETA + niveis LST_SSTA_NEW_GREC do s11
        #    (escala -5..+5 °C; BRANCO no centro; refino de ±0.2 no zero). 'levels' EXPLICITO
        #    (NAO-UNIFORME) -> o motor monta BoundaryNorm. ──
        'cmap_colors': [str(c) for c in settings.LST_ANOM_CORRETA],
        'levels': [float(x) for x in settings.LST_SSTA_NEW_GREC],
        # Continentes WHITESMOKE: SSTA e NaN sobre a TERRA; com o raster transparente, essa area
        # revela o fundo do disco. O centro da paleta e branco puro (#ffffff), o que deixaria a terra
        # branca; whitesmoke da o mesmo tom claro do tsm_abs (pedido do usuario p/ s38/s39). Oceano
        # 100% preenchido (extend='both'), entao o fundo so aparece na terra.
        'cor_fundo_globo_default': 'whitesmoke',
        # ── STANDBY: paleta amostrada da colorbar da sigma (31 bandas, -6.2..+6.2, branco central
        #    [-0.2,+0.2]). Para REATIVAR: troque 'cmap_colors' acima por 'cmap_colors_sigma',
        #    remova 'levels' e adicione 'niveis': 31. ──
        'cmap_colors_sigma': [
            '#fe93f3',  # [-6.2,-5.8] magenta (extremo frio)
            '#fe4eef',  # [-5.8,-5.4]
            '#bf12ac',  # [-5.4,-5.0] magenta-roxo
            '#820b73',  # [-5.0,-4.6] roxo
            '#3d1381',  # [-4.6,-4.2] violeta escuro
            '#3c2ab4',  # [-4.2,-3.8] indigo
            '#7260da',  # [-3.8,-3.4] azul-violeta
            '#a594fe',  # [-3.4,-3.0] lavanda
            '#d7dff4',  # [-3.0,-2.6] lavanda palida
            '#b2fba9',  # [-2.6,-2.2] verde claro
            '#59ef5c',  # [-2.2,-1.8] verde
            '#3bd13a',  # [-1.8,-1.4] verde
            '#108c4d',  # [-1.4,-1.0] verde escuro
            '#1b71e0',  # [-1.0,-0.6] azul
            '#55a4ef',  # [-0.6,-0.2] azul medio
            '#ffffff',  # [-0.2,+0.2] BRANCO (neutro — divide frio/quente)
            '#ffc176',  # [+0.2,+0.6] amarelo-laranja
            '#ff8744',  # [+0.6,+1.0] laranja
            '#ff4d18',  # [+1.0,+1.4] laranja-vermelho
            '#d02602',  # [+1.4,+1.8] vermelho
            '#a21704',  # [+1.8,+2.2] vermelho escuro
            '#770a07',  # [+2.2,+2.6] vinho
            '#430800',  # [+2.6,+3.0] vinho escuro
            '#775144',  # [+3.0,+3.4] marrom
            '#a27c6f',  # [+3.4,+3.8] marrom claro
            '#d2ac9f',  # [+3.8,+4.2] bege
            '#f0dcd5',  # [+4.2,+4.6] bege palido
            '#ffe8e3',  # [+4.6,+5.0] rosa palido
            '#f6a09f',  # [+5.0,+5.4] rosa
            '#e26062',  # [+5.4,+5.8] rosa-vermelho
            '#bd3134',  # [+5.8,+6.2] vermelho (extremo quente)
        ],
        'simetrico': True,       # niveis/escala vêm de 'levels' (LST_SSTA_NEW_GREC, -5..+5)
        'sem_clim_ref': True,    # TSM usa OISST (1991-2020), nao ERA5 — rotulo proprio se quiser
        'legenda_numerica': True,  # cbar com VALORES numericos (nao "above/below"), passo 0.5 °C
        'legenda_unidade': '°C',
        'legenda_num_step': 0.5,
        'box_nino34': True,        # caixa do Niño 3.4 (170°W–120°W, 5°S–5°N) + rotulo no globo
        'nino34_valor': True,      # rotulo DINAMICO "Niño 3.4 = +x.x°C" = media da ANOMALIA na area do box
        'nino34_anom': True,       # rotulo com SINAL (+/-), pois e' anomalia (nao TSM absoluta)
        'spec': {
            'nome': 'tsm_anom', 'unidade': '°C', 'celsius': False,
            'var_candidates': ('sst',), 'clim_fn': None, 'kind': 'sst',
            'reanalise_fn': _tsm_reanalise_series,
            'forecast_fn': _tsm_forecast_series,  # SST nao tem previsao -> erro claro no forecast
            'era5_fn': None, 'gdas_fn': None,
        },
    },
    'tsm_abs': {
        'titulo': 'Temperatura da Superfície do Mar (TSM) — absoluta',
        'titulo_en': 'Sea surface temperature',
        'rotulo_box': 'Sea Surface Temperature',
        'subtitulo_dir': 'Sea surface temperature (SST)',
        'unidade': '°C',
        # SHADED ABSOLUTO 0..32°C — MESMOS levels/cores do s30 (SST_MEAN). Escala NAO-simetrica;
        # 'levels' explicito -> motor monta BoundaryNorm; 16 cores interpoladas nas 64 bandas.
        'cmap_colors': _SST_ABS_COLORS,
        'levels': _SST_ABS_LEVELS,
        'simetrico': False,
        # Fundo do disco WHITESMOKE: a SST e NaN sobre a TERRA (contourf nao preenche) e, com o raster
        # transparente, essa area revela o fundo do globo -> vira a "cor do continente". Sem este
        # default, o fundo cairia no CENTRO da paleta SST (= 'orange') e a terra sairia LARANJA.
        # whitesmoke = continentes claros (pedido do usuario p/ tsm_abs no s38/s39); o oceano e 100%
        # preenchido via extend='both', entao o fundo so aparece na terra.
        'cor_fundo_globo_default': 'whitesmoke',
        'sem_clim_ref': True,      # OISST, nao ERA5
        'legenda_numerica': True,  # cbar com VALORES numericos absolutos (nao "above/below")
        'legenda_unidade': '°C',
        'legenda_num_step': 4.0,   # rotulos de 4 em 4 °C (0..32)
        'box_nino34': True,        # caixa do Niño 3.4 (default; controlavel por GLOBO_3D_BOX_NINO34)
        'nino34_valor': True,      # rotulo DINAMICO "Niño 3.4 = xx.x°C" (media diaria da area do box)
        'spec': {
            'nome': 'tsm_abs', 'unidade': '°C', 'celsius': False,
            'var_candidates': ('sst',), 'clim_fn': None, 'kind': 'sst',
            'reanalise_fn': _tsm_abs_reanalise_series,
            'forecast_fn': _tsm_forecast_series,  # SST nao tem previsao -> erro claro no forecast
            'era5_fn': None, 'gdas_fn': None,
        },
    },
    'jet_stream': {
        'titulo': 'Magnitude do Vento em 250 hPa (Jet Stream)',
        'titulo_en': '250-hPa wind speed (m/s)',  # unidade no titulo (s38/s39)
        'rotulo_box': 'Jet Stream',               # caixa do s39
        'subtitulo_dir': 'Wind speed at 250 hPa',
        'unidade': 'ms⁻¹',
        'absoluto': True,                     # campo absoluto — sem subtracao de climatologia
        'simetrico': False,
        'vmin': float(settings.get('GLOBO_3D_VMIN_JET_STREAM', 30.0)),  # abaixo = transparente
        'vmax': float(settings.get('GLOBO_3D_VMAX_JET_STREAM', 90.0)),
        'niveis': 64,
        # ── Paleta v1 (ciano→azul→indigo→magenta) ────────────────────────────
        # 'cmap_colors': [
        #     '#00FFEE',  # ciano-aqua  (vmin ~30 ms⁻¹)
        #     '#00E8FF',
        #     '#00C8FF',
        #     '#00A0FF',
        #     '#0070FF',
        #     '#0040EE',
        #     '#2200DD',
        #     '#5500CC',
        #     '#8800BB',
        #     '#BB00AA',
        #     '#EE0099',
        #     '#FF0077',  # magenta/rosa  (vmax ~90 ms⁻¹)
        # ],
        # ── Paleta v2 ────────────────────────────────────────────────────────────
        'cmap_colors': [
            '#2b3494',  # índigo escuro       (vmin ~30 ms⁻¹)
            '#2849a4',  # índigo
            '#1665b8',  # azul-índigo
            '#00a2e6',  # azul cyan
            '#4fa2e6',  # azul claro
            '#8581da',  # roxo-azulado
            '#9867ca',  # roxo médio
            '#ab3db2',  # roxo/violeta
            '#b531b6',  # magenta escuro (inserido entre 65 e 70 ms⁻¹)
            '#cf57c0',  # magenta
            '#e692d8',  # rosa médio
            '#f7ceef',  # rosa claro
            '#f9e6f7',  # rosa pastel
            '#fcf7fb',  # quase branco        (vmax ~90 ms⁻¹)
        ],
        # Fundo do globo preto; continentes em cinza claro (revelados abaixo de vmin)
        'cor_fundo_globo_default': 'black',
        'cor_continente': 'silver',            # continentes cinza médio
        'cor_fronteiras': 'white',            # divisas de continentes/estados/países brancas
        # contourf com extend='max': abaixo de vmin nao ha fill (transparente)
        'extend_contourf': 'max',
        # Labels da legenda (4 swatches WaPo sem unidade — unidade vai no titulo)
        'legenda_labels': ['35', '50', '65', '80+'],
        # 5 labels do gradiente s39 e unidade abaixo da barra (via legenda_unidade)
        'legenda5_labels': ['30', '45', '60', '75', '90+'],
        'legenda_unidade': 'm/s',
        # Isolinhas de Z250 climatologico whitesmoke (controladas por flag separado)
        'isolinha_hgt_abs': True,
        # Isolinhas fixas de Z250 absoluto real (sempre plotadas)
        'isolinhas_fixas_hgt': [
            (10680, 'red',    2.0),   # 10.680 mgp — vermelho
            (10200, 'orange', 2.0),   # 10.200 mgp — laranja
            (10080, 'white',  2.0),   # 10.080 mgp — branco
        ],
        'spec': {
            'nome': 'jet_stream', 'unidade': 'ms⁻¹', 'kind': 'uv250',
            'era5_fn': _era5_uv250, 'gdas_fn': _gdas_uv250,
            'hgt_clim_fn': clim_hgt250_daily,  # Z250 clim p/ isolinhas no jet stream
        },
    },
}


# ---------------------------------------------------------------------------
# Variante MEDIA MOVEL de 5 dias de z250_anom — mesma ficha, cada frame vira a
# media de uma janela deslizante de 5 dias (espelha psi/chi200). Construida como
# copia profunda da ficha diaria p/ nao duplicar paleta/spec e ficar em sincronia.
# ---------------------------------------------------------------------------
_z250_5d = copy.deepcopy(VARIAVEIS['z250_anom'])
_z250_5d.update({
    'titulo': 'Anomalia de Altura Geopotencial em 250 hPa (média móvel de 5 dias)',
    'titulo_en': '250-hPa geopotential height (5-day mean)',
    # Caixa cinza do s39 IDENTICA a da versao diaria: a media movel ja fica clara pelo
    # intervalo de datas mostrado abaixo da caixa. So os titulos internos/tarja/canto marcam "5-day mean".
    'rotulo_box': VARIAVEIS['z250_anom']['rotulo_box'],
    'subtitulo_dir': 'Geopotential height at 250 hPa (5-day mean)',
    'pentada_movel': 5,  # cada frame = media movel de 5 dias [d, d+4]
})
_z250_5d['spec']['nome'] = 'z250_anom_5d'
VARIAVEIS['z250_anom_5d'] = _z250_5d


# Variante do jet_stream com isolinhas BRANCAS de psi200 sobrepostas: MESMISSIMO shaded/paleta/
# continente/oceano/jato/isolinhas de Z250 do jet_stream (copia profunda) — so ACRESCENTA o contorno
# de psi200_anom em branco (mecanismo generico `contorno_serie_var`, igual chi200_cores_psi200_contornos).
# Aqui o psi200 vai em anomalia DIARIA (ver `contorno_serie_sem_pentada` abaixo), nao na pentada movel
# propria da ficha psi200_anom — p/ casar com o shaded diario do jet_stream.
_jet_psi = copy.deepcopy(VARIAVEIS['jet_stream'])
_jet_psi.update({
    'titulo': 'Magnitude do Vento em 250 hPa (Jet Stream) e Função de Corrente (linhas) em 200 hPa',
    'titulo_en': '250-hPa wind speed (m/s) with 200-hPa streamfunction contours',
    'rotulo_box': 'Jet Stream & Streamfunction',
    'subtitulo_dir': 'Wind speed at 250 hPa + streamfunction (lines) at 200 hPa',
    'contorno_serie_var': 'psi200_anom',
    'contorno_serie_cor': 'white',
    'contorno_serie_intervalo': 5.0,   # 10⁶ m²/s entre isolinhas de psi200
    'contorno_serie_lw': 0.5,
    # PSI200 em anomalia DIARIA (nao pentada movel): o shaded do jet_stream e diario, entao a media
    # movel de 5 dias propria do psi200_anom encurtaria a serie em 4 dias e a isolinha CONGELARIA no
    # fim (enquanto o jato continua). Diaria -> as duas terminam/animam juntas.
    'contorno_serie_sem_pentada': True,
})
_jet_psi['spec']['nome'] = 'jet_stream_psi200_contour'
VARIAVEIS['jet_stream_psi200_contour'] = _jet_psi


# Z500 SO em isolinhas BRANCAS (sem nenhum campo sombreado): copia profunda da ficha 'sem_variavel'
# (globo sem shaded, oceano solido, sem caixa de titulo/legenda) e ACRESCENTA o contorno de z500_abs
# em branco via o mecanismo generico `contorno_serie_var` (igual jet_stream_psi200_contour, mas aqui
# o unico elemento do campo e a propria isolinha). z500_abs e ABSOLUTO e nao tem pentada_movel, entao
# a serie fica sinotica/diaria e a isolinha acompanha o eixo de tempo do frame -- sem descasamento.
_z500_ct = copy.deepcopy(VARIAVEIS['sem_variavel'])
_z500_ct.update({
    'titulo': 'Altura Geopotencial em 500 hPa (isolinhas)',
    'titulo_en': '500-hPa geopotential height (contours)',
    'rotulo_box': 'Geopotential Height',
    'subtitulo_dir': 'Geopotential height at 500 hPa',
    'contorno_serie_var': 'z500_abs',
    'contorno_serie_cor': 'white',
    'contorno_serie_intervalo': 60.0,   # mgp entre isolinhas (padrao de carta de 500 hPa = 6 dam)
    'contorno_serie_lw': 0.6,           # unico elemento do campo -> um tico mais grosso que o 0.5 padrao
})
_z500_ct['spec']['nome'] = 'z500_contour'
VARIAVEIS['z500_contour'] = _z500_ct


# Combinada: PWAT em CORES (shaded) + Z500 em isolinhas BRANCAS por cima -- mesmo padrao das combos
# ja existentes (chi200_cores_psi200_contornos, olr_cores_z250_contornos etc): copia profunda da ficha
# shaded (pwat_abs, herda paleta/escala/legenda numerica/oceano) e ACRESCENTA o contorno de z500_abs
# via `contorno_serie_var`. z500_abs e absoluto e sem pentada -> a isolinha acompanha o eixo de tempo
# do frame (sem descasamento). 1 chave na lista = 1 MP4 com as DUAS variaveis.
_pwat_z500 = copy.deepcopy(VARIAVEIS['pwat_abs'])
_pwat_z500.update({
    'titulo': 'Água Precipitável (cores) e Altura Geopotencial em 500 hPa (linhas)',
    'titulo_en': 'Precipitable water (shaded) with 500-hPa geopotential height contours',
    'rotulo_box': 'Precipitable Water & Z500',
    'subtitulo_dir': 'Precipitable water (shaded) + geopotential height at 500 hPa (lines)',
    'contorno_serie_var': 'z500_abs',
    'contorno_serie_cor': 'white',
    'contorno_serie_intervalo': 60.0,   # mgp entre isolinhas (padrao de carta de 500 hPa = 6 dam)
    'contorno_serie_lw': 0.6,
})
_pwat_z500['spec']['nome'] = 'pwat_cores_z500_contornos'
VARIAVEIS['pwat_cores_z500_contornos'] = _pwat_z500


# -----------------------------------------------------------------------------------------------
# PWAT "alicia" (ficha 'pwat_abs_alicia', s38 em diante) -- paleta estilo GrADS `color.gs 20 70 1`:
# white->green->limegreen->yellow->orange->red->magenta, faixa 20-70mm. O BRANCO e '#ffffff00'
# (alpha=0): some no SHADED (revela o fundo/blue marble onde a agua precipitavel e baixa) mas
# aparece OPACO na LEGENDA -- o motor tira o alpha de '#RRGGBBAA' so na paleta da legenda
# (paleta_opaca em _run_globo). extend='max' -> abaixo de 20mm nao ha fill = transparente. Ficha
# SEPARADA: a pwat_abs original (terroso->verde, 0-70) fica intacta. Reusa os MESMOS downloaders
# de PWAT (spec identico) -> nao rebaixa dado, so muda a aparencia.
_pwat_alicia = copy.deepcopy(VARIAVEIS['pwat_abs'])
_alicia_oceano = str(settings.get('GLOBO_3D_COR_OCEANO_PWAT_ABS_ALICIA', 'none'))
_pwat_alicia.update({
    'vmin': float(settings.get('GLOBO_3D_VMIN_PWAT_ABS_ALICIA', 20.0)),
    'vmax': float(settings.get('GLOBO_3D_VMAX_PWAT_ABS_ALICIA', 70.0)),
    'niveis': int(settings.get('GLOBO_3D_NIVEIS_PWAT_ABS_ALICIA', 50)),  # 20->70 passo 1 (color.gs 20 70 1)
    'legenda_num_step': float(settings.get('GLOBO_3D_LEGENDA_STEP_PWAT_ABS_ALICIA', 5.0)),
    'cmap_colors': list(settings.get('GLOBO_3D_PALETA_PWAT_ABS_ALICIA', [
        '#ffffff00', 'green', 'limegreen', 'yellow', 'orange', 'red', 'magenta',
    ])),
    'extend_contourf': 'max',  # abaixo de vmin(20mm) sem preenchimento = transparente
    'cor_oceano': (None if _alicia_oceano.strip().lower() in ('none', '') else _alicia_oceano),
})
_pwat_alicia['spec']['nome'] = 'pwat_abs_alicia'
VARIAVEIS['pwat_abs_alicia'] = _pwat_alicia


# Combinada: OLR em CORES + Z500 em isolinhas BRANCAS (mesmo padrao da pwat_cores_z500_contornos).
# olr_anom nao tem pentada propria -> shaded e contorno seguem a mesma logica de pentada (global), logo
# ficam sempre alinhados no tempo, sem precisar de contorno_serie_pentada.
_olr_z500 = copy.deepcopy(VARIAVEIS['olr_anom'])
_olr_z500.update({
    'titulo': 'Anomalia de OLR (cores) e Altura Geopotencial em 500 hPa (linhas)',
    'titulo_en': 'Outgoing longwave radiation (shaded) with 500-hPa geopotential height contours',
    'rotulo_box': 'OLR & Z500',
    'subtitulo_dir': 'OLR (shaded) + geopotential height at 500 hPa (lines)',
    'contorno_serie_var': 'z500_abs',
    'contorno_serie_cor': 'white',
    'contorno_serie_intervalo': 60.0,   # mgp entre isolinhas (padrao de carta de 500 hPa = 6 dam)
    'contorno_serie_lw': 0.6,
})
_olr_z500['spec']['nome'] = 'olr_cores_z500_contornos'
VARIAVEIS['olr_cores_z500_contornos'] = _olr_z500


# Combinada: T850 (anomalia) em CORES + Z500 em isolinhas BRANCAS (mesmo padrao das duas acima).
_tmp850_z500 = copy.deepcopy(VARIAVEIS['tmp850_anom'])
_tmp850_z500.update({
    'titulo': 'Anomalia de Temperatura em 850 hPa (cores) e Altura Geopotencial em 500 hPa (linhas)',
    'titulo_en': '850-hPa air temperature (shaded) with 500-hPa geopotential height contours',
    'rotulo_box': 'T850 & Z500',
    'subtitulo_dir': 'Air temperature at 850 hPa (shaded) + geopotential height at 500 hPa (lines)',
    'contorno_serie_var': 'z500_abs',
    'contorno_serie_cor': 'white',
    'contorno_serie_intervalo': 60.0,   # mgp entre isolinhas (padrao de carta de 500 hPa = 6 dam)
    'contorno_serie_lw': 0.6,
})
_tmp850_z500['spec']['nome'] = 'tmp850_cores_z500_contornos'
VARIAVEIS['tmp850_cores_z500_contornos'] = _tmp850_z500


# Variantes AUTOMATICAS: pedir a chave-base gera TAMBEM as variantes listadas.
# z250_anom -> sempre acompanha a media movel de 5 dias (dois MP4s por execucao).
VARIANTES_AUTO: dict[str, list[str]] = {
    'z250_anom': ['z250_anom_5d'],
}


def expandir_variaveis(variaveis: list[str]) -> list[str]:
    """Insere as variantes automaticas (VARIANTES_AUTO) logo apos a variavel-base,
    preservando a ordem e sem duplicar as ja pedidas explicitamente.

    GLOBO_3D_VARIANTES_AUTO=false (GLOBAL, vale p/ todos os globos s38/s39/s40/s41) desliga a
    expansao: ex.: `z250_anom` gera SO a versao diaria (3 saidas), sem a media movel de 5 dias.
    Variantes pedidas EXPLICITAMENTE na lista seguem valendo."""
    if not bool(settings.get('GLOBO_3D_VARIANTES_AUTO', True)):
        return [v for i, v in enumerate(variaveis) if v not in variaveis[:i]]
    out: list[str] = []
    for v in variaveis:
        if v not in out:
            out.append(v)
        for extra in VARIANTES_AUTO.get(v, []):
            if extra not in out:
                out.append(extra)
    return out


# Threshold (mgp) entre a escala de Z500 (~4700-6200) e Z250 (~9200-11500) -- fica no meio do
# "vao" entre as duas faixas reais, entao nunca ambiguo pros niveis tipicos de jato.
_JATO_NIVEL_THRESHOLD = 7500.0


def _jato_kind_from_nivel(nivel: float) -> str:
    """Decide, PELO VALOR do nivel configurado (GLOBO_3D_JET_STREAM_NIVEL /
    GLOBO_3D_SUBTROPICAL_JET_NIVEL), se o campo-guia desse jato precisa ser Z250 ou Z500 --
    essas duas variaveis SAO a fonte da verdade: o motor baixa o HGT do nivel certo (250 ou
    500 hPa) pra cada jato, independente da variavel shaded ativa."""
    return 'hgt250' if float(nivel) >= _JATO_NIVEL_THRESHOLD else 'hgt500'


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


_BLUE_MARBLE_CACHE: np.ndarray | None = None


def _blue_marble_rgba() -> np.ndarray | None:
    """Imagem de satelite `Entrada/blue_marble.png` (mesmo arquivo usado pelo s23) como fundo do
    globo. Calculada/redimensionada uma unica vez (mesmo padrao de `_vignette_rgba`) -- o mapa nao
    muda entre frames, so a reprojecao via imshow(transform=...) e refeita a cada frame."""
    global _BLUE_MARBLE_CACHE
    if _BLUE_MARBLE_CACHE is None:
        path = Path(settings.get('DIR_INPUT', 'Entrada')) / 'blue_marble.png'
        if not path.exists():
            logger.warning('blue_marble.png nao encontrado em {} -- fundo do globo fica so a cor solida', path)
            return None
        img = _PIL_Image.open(path)
        # Teto da resolucao da FONTE (GLOBO_3D_BLUE_MARBLE_MAX_PX, default 4096 = comportamento
        # historico). O arquivo tem 8192px (4,9 km/px); em 4096 cai p/ 9,8 km/px, o que BASTA nos
        # globos flutuantes e no s42, mas BLOQUEIA o zoom do inclinado: abaixo de ~2000000 m de
        # altura o quadro pede ~2 km/px e o satelite quebra em blocos. Subir p/ 8192 dobra a
        # nitidez e QUADRUPLICA este array (25 MB -> 100 MB), alocado UMA vez e herdado pelos
        # workers via fork/CoW.
        # ATENCAO: esse 1x100 MB NAO e o custo real de subir o teto -- o `imshow` do cartopy
        # reprojeta com uma KDTree sobre TODOS os pixels da FONTE, POR FRAME E POR WORKER (ver
        # `_bbox_visivel_lonlat`). Em 8192 isso media 2,8 GB por warp (vs 0,87 GB em 4096) e
        # derrubava a maquina. Por isso o recorte no bbox visivel: ele e que torna o 8192 viavel
        # (~0,36 GB), nao o CoW. Nao confundir com GLOBO_3D_INC_BLUE_MARBLE_REGRID, que e a malha
        # de DESTINO -- subir aquele nao resolve o bloco, porque o gargalo e a fonte.
        _max_px = int(settings.get('GLOBO_3D_BLUE_MARBLE_MAX_PX', 4096))
        if img.width > _max_px:
            ratio = _max_px / img.width
            img = img.resize((_max_px, int(img.height * ratio)), _PIL_Image.LANCZOS)
        _BLUE_MARBLE_CACHE = np.array(img.convert('RGB'))
        logger.info('Blue marble: fonte {}x{} px (~{:.1f} km/px no equador) — {:.0f} MB',
                    _BLUE_MARBLE_CACHE.shape[1], _BLUE_MARBLE_CACHE.shape[0],
                    360.0 / _BLUE_MARBLE_CACHE.shape[1] * 111.0, _BLUE_MARBLE_CACHE.nbytes / 1e6)
    return _BLUE_MARBLE_CACHE


# Acima deste span (graus) a vista e larga demais p/ o recorte compensar -- globos flutuantes
# (s38-s43) veem um hemisferio inteiro (~180°). Tambem pega o caso do antimeridiano: uma faixa
# estreita cruzando ±180 da min/max ~-180/180 -> span ~360 -> sem recorte (correto, so nao otimizado).
_BM_CROP_SPAN_MAX = 120.0


def _bbox_visivel_lonlat(ax, proj, margem: float = 2.0) -> tuple[float, float, float, float] | None:
    """Bbox (lon0, lon1, lat0, lat1) em graus do pedaco de mundo que a janela do `ax` enxerga, ou
    None quando nao vale a pena recortar (ver `_BM_CROP_SPAN_MAX`).

    POR QUE EXISTE: o `imshow(transform=...)` do cartopy reprojeta via `img_transform.regrid`, que
    monta uma KDTree sobre TODOS os pixels da FONTE e busca o vizinho mais proximo de cada ponto de
    destino. Logo o custo de RAM escala com a FONTE (nao com `regrid_shape`) e e transitorio, POR
    FRAME E POR WORKER -- o array cacheado do blue marble e compartilhado via CoW, mas o warp nao.
    A janela inclinada (s44/s46) enxerga ~13-22° e mandava o planeta inteiro pra KDTree: 33,5 M de
    pontos = 2,8 GB por warp em 8192. Recortado, a mesma nitidez custa ~0,36 GB.

    COMO: amostra a propria grade de destino (limites do `ax` em coords nativas da projecao) e
    projeta de volta p/ lon/lat -- pontos de CEU (fora do globo) voltam inf/nan e sao descartados.
    Como o bbox sai da MESMA grade que o warp vai varrer, todo ponto visivel cai dentro dele: por
    isso o recorte nao borra as bordas (o nearest-neighbor so smearia se faltasse fonte)."""
    X, Y = np.meshgrid(np.linspace(*ax.get_xlim(), 129), np.linspace(*ax.get_ylim(), 129))
    pts = ccrs.PlateCarree().transform_points(proj, X.ravel(), Y.ravel())
    lon = pts[:, 0].reshape(X.shape)
    lat = pts[:, 1].reshape(X.shape)
    lon[~np.isfinite(lon)] = np.nan     # ceu (fora do globo) volta inf/nan do transform_points
    lat[~np.isfinite(lat)] = np.nan
    ok = np.isfinite(lon) & np.isfinite(lat)
    if not ok.any():
        return None                     # janela toda no ceu (nao deveria acontecer) -> sem recorte
    lon0, lon1 = float(np.nanmin(lon)), float(np.nanmax(lon))
    lat0, lat1 = float(np.nanmin(lat)), float(np.nanmax(lat))
    if (lon1 - lon0) > _BM_CROP_SPAN_MAX or (lat1 - lat0) > _BM_CROP_SPAN_MAX:
        return None
    # MARGEM ADAPTATIVA: perto do HORIZONTE (limbo) o chao fica muito escorcado -- uma celula desta
    # amostra pode valer varios graus, e a grade do warp e mais densa que ela. Sem folga, o bbox sairia
    # menor que o alcance real do warp e a faixa junto ao horizonte borraria (nearest-neighbor sem
    # fonte). Usa o maior salto entre celulas vizinhas como folga; teto p/ nao anular o recorte.
    saltos = [np.abs(np.diff(a, axis=eixo)) for a in (lon, lat) for eixo in (0, 1)]
    saltos = np.concatenate([s[np.isfinite(s)].ravel() for s in saltos]) if ok.sum() > 1 else np.array([])
    folga = max(margem, min(float(saltos.max()), 10.0)) if saltos.size else margem
    return (max(-180.0, lon0 - folga), min(180.0, lon1 + folga),
            max(-90.0, lat0 - folga), min(90.0, lat1 + folga))


def _subset_campo_bbox(lon: np.ndarray, lat: np.ndarray, campo: np.ndarray, bbox: tuple | None,
                       margem_cel: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recorta o CAMPO (lat, lon) no `bbox`, com folga de `margem_cel` celulas p/ o contourf ter
    vizinhos e fechar as bandas na borda. `bbox=None` -> devolve tudo (vista larga = sem recorte).

    POR QUE: o raster do shaded (`_contourf_raster`) cobre o dominio inteiro do campo. Global, com
    `GLOBO_3D_SHADE_PX=3600`, isso da 10 px/grau -- e a janela do inclinado (~26°) recebe so ~260 px
    de fonte, que o warp NEAREST-NEIGHBOR do cartopy estica p/ os ~1920 px do quadro = SERRILHADO em
    blocos. Subir o px global nao resolve (1 px de fonte por px de tela exigiria ~26600 px = raster de
    1,4 GB). Recortando, `GLOBO_3D_SHADE_CROP_PX=2048` cobre so os ~26° -> ~79 px/grau (~1:1 com a
    tela) com um raster MENOR que o global -> mais nitido E menos RAM no warp. Nao mexe no dado: a
    alternativa (suavizar com GLOBO_3D_SIGMA_<VAR>) achataria o pico do campo."""
    if bbox is None:
        return lon, lat, campo
    lon0, lon1, lat0, lat1 = bbox
    # CONVENCAO DE LONGITUDE: o bbox vem em -180..180 (saida do transform_points), mas a grade das
    # series e 0..360 (`_target_grid`) -- sem converter, o `where` abaixo nao casa NADA e o recorte
    # vira um no-op SILENCIOSO (foi exatamente o bug que fez o serrilhado sobreviver ao fix).
    if float(lon.max()) > 180.0:
        lon0, lon1 = lon0 % 360.0, lon1 % 360.0
        if lon0 > lon1:      # faixa cruza o meridiano 0 na convencao 0..360 -> recorte nao e contiguo
            return lon, lat, campo
    ilon = np.where((lon >= lon0) & (lon <= lon1))[0]
    ilat = np.where((lat >= lat0) & (lat <= lat1))[0]   # contiguo com lat crescente OU decrescente
    if ilon.size < 4 or ilat.size < 4:
        # NAO silenciar: se o bbox nao casa com a grade, o shaded sai serrilhado e ninguem fica sabendo.
        logger.warning('Recorte do shaded NAO aplicou (bbox lon {:.1f}..{:.1f} lat {:.1f}..{:.1f} nao casa '
                       'com a grade lon {:.1f}..{:.1f} lat {:.1f}..{:.1f}) -- campo segue GLOBAL e o '
                       'sombreado pode sair serrilhado', lon0, lon1, lat0, lat1, float(lon.min()),
                       float(lon.max()), float(lat.min()), float(lat.max()))
        return lon, lat, campo
    i0, i1 = max(0, ilon[0] - margem_cel), min(lon.size, ilon[-1] + 1 + margem_cel)
    j0, j1 = max(0, ilat[0] - margem_cel), min(lat.size, ilat[-1] + 1 + margem_cel)
    _lon, _lat, _campo = lon[i0:i1], lat[j0:j1], campo[j0:j1, i0:i1]
    # De volta p/ -180..180: o `extent` do imshow vai num transform PlateCarree, cujo dominio e
    # -180..180 -- entregar 294..317 (a fatia na convencao 0..360) faz o shaded sumir do quadro.
    # So converte se a fatia continuar MONOTONICA (nao cruza o meridiano 0); senao segue global.
    if float(lon.max()) > 180.0:
        _lon = ((_lon + 180.0) % 360.0) - 180.0
        if not np.all(np.diff(_lon) > 0):
            return lon, lat, campo
    return _lon, _lat, _campo


_GEOM_PINTADA_CACHE: dict = {}


def _geom_area_pintada(somente_brasil: bool, somente_continente: bool):
    """Geometria (lon/lat) da area onde o shaded REALMENTE aparece, ou None (aparece em tudo).

    Le a MESMA fonte dos recortes de desenho (`_brasil_clip_path`/`_land_clip_path`): se a legenda
    usasse outra geometria, ela poderia anunciar uma faixa que o mapa nao pinta. Prioridade igual a
    do render: BRASIL > CONTINENTE."""
    key = (bool(somente_brasil), bool(somente_continente))
    if key in _GEOM_PINTADA_CACHE:
        return _GEOM_PINTADA_CACHE[key]
    from shapely.ops import unary_union
    geom = None
    if somente_brasil:
        import cartopy.io.shapereader as shpreader
        _ne = shpreader.natural_earth(resolution='50m', category='cultural', name='admin_0_countries')
        geom = unary_union([r.geometry for r in shpreader.Reader(_ne).records()
                            if (r.attributes.get('ADM0_A3') or r.attributes.get('adm0_a3')) == 'BRA'])
    elif somente_continente:
        import cartopy.feature as _cf
        geom = unary_union(list(_cf.LAND.with_scale('50m').geometries()))
    _GEOM_PINTADA_CACHE[key] = geom
    return geom


def _ultima_banda_pintada(campo: np.ndarray, lon: np.ndarray, lat: np.ndarray, levels: list,
                          bbox: tuple | None, geom) -> int | None:
    """Indice (0-based) da ULTIMA banda de `levels` que aparece PINTADA no quadro, ou None se nada
    pinta (tudo abaixo do 1o nivel). A legenda desenha as bandas 0..este indice.

    'Aparece' = dentro do enquadramento (`bbox`) E dentro da area pintada (`geom`, ex.: Brasil
    quando PLOTAR_SOMENTE_BRASIL) -- senao chuva recortada fora do mapa entraria na legenda.
    `campo` e o maximo do PERIODO (legenda fixa: nao muda entre frames)."""
    _lon, _lat, _campo = _subset_campo_bbox(lon, lat, campo, bbox, margem_cel=0)
    if geom is not None:
        import shapely
        _lo, _la = np.meshgrid(_lon, _lat)
        dentro = shapely.contains_xy(geom, _lo, _la)
        if not dentro.any():
            return None
        _campo = np.where(dentro, _campo, np.nan)
    if not np.isfinite(_campo).any():
        return None
    mx = float(np.nanmax(_campo))
    if mx < float(levels[0]):
        return None                      # nada atinge o 1o nivel -> mapa sem chuva pintada
    # levels[k] <= mx < levels[k+1]. Acima do ultimo nivel, o render pinta a cor da ULTIMA banda
    # (extend='max' herda o set_over) -> a legenda tambem para nela.
    k = int(np.searchsorted(np.asarray(levels, dtype=float), mx, side='right')) - 1
    return int(min(max(k, 0), len(levels) - 2))


def _upsample_campo(lon: np.ndarray, lat: np.ndarray, campo: np.ndarray, fator: float,
                    max_celulas: int = 4_000_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolacao CUBICA do campo p/ o shaded sair com bandas LISAS, SEM alterar os valores.

    POR QUE NAO O `GLOBO_3D_SIGMA_<VAR>`: o sigma e uma media ponderada -- ele ACHATA o campo (medido
    na chuva real: sigma 1.0 = -17% no pico, 3.0 = -54%). Inaceitavel num produto de ALERTA, onde o
    numero E a informacao. A spline cubica e um INTERPOLANTE: passa EXATAMENTE pelos pontos da grade
    original, entao todo valor do modelo continua la; ela so desenha o caminho ENTRE os pontos com
    curva em vez de degrau. O overshoot tipico do cubico (ringing perto de gradiente forte) e contido
    pelo clip no [min, max] do proprio campo -> nenhum valor fora da faixa que o modelo previu.

    PARA QUE: no zoom do inclinado a celula de 0.25° (~28 km) do ECMWF vira ~21 px de tela, e as
    facetas do marching-squares do contourf viram degraus visiveis. Nao confundir com o serrilhado
    do RASTER (`_subset_campo_bbox`): sao limites diferentes -- o do raster ja e resolvido la, e este
    aqui e o limite do DADO.

    NaN e preservado (mascara reamostrada por vizinho): o cubico espalharia o NaN pela vizinhanca.
    """
    f = float(fator)
    if f <= 1.0 or campo.size == 0:
        return lon, lat, campo
    # Teto de seguranca: so faz sentido no campo JA recortado. Um campo global 6x viraria 4326x8640
    # (~150 MB e um raster gigante) -- aqui o fator cai ate caber.
    while f > 1.0 and campo.size * f * f > max_celulas:
        f -= 0.5
    if f <= 1.0:
        return lon, lat, campo
    _nan = np.isnan(campo)
    _src = np.where(_nan, float(np.nanmin(campo)) if not _nan.all() else 0.0, campo)
    c = _zoom(_src, f, order=3, mode='nearest')
    c = np.clip(c, float(np.nanmin(campo)), float(np.nanmax(campo)))
    if _nan.any():
        c[_zoom(_nan.astype(np.uint8), f, order=0, mode='nearest').astype(bool)] = np.nan
    return (np.linspace(lon[0], lon[-1], c.shape[1]),
            np.linspace(lat[0], lat[-1], c.shape[0]), c)


def _recorta_lonlat(img: np.ndarray, extent: tuple, bbox: tuple | None) -> tuple[np.ndarray, tuple]:
    """Recorta `img` (origin='upper', `extent` = (lon0, lon1, lat0, lat1)) no `bbox`, devolvendo o
    array menor + o extent correspondente. `bbox=None` -> devolve o original intacto."""
    if bbox is None:
        return img, extent
    ny, nx = img.shape[0], img.shape[1]
    e_lon0, e_lon1, e_lat0, e_lat1 = extent
    ppd_x = nx / (e_lon1 - e_lon0)
    ppd_y = ny / (e_lat1 - e_lat0)
    b_lon0, b_lon1, b_lat0, b_lat1 = bbox
    i0 = max(0, int(np.floor((b_lon0 - e_lon0) * ppd_x)))
    i1 = min(nx, int(np.ceil((b_lon1 - e_lon0) * ppd_x)))
    j0 = max(0, int(np.floor((e_lat1 - b_lat1) * ppd_y)))   # origin='upper': linha 0 = lat MAXIMA
    j1 = min(ny, int(np.ceil((e_lat1 - b_lat0) * ppd_y)))
    if i1 - i0 < 2 or j1 - j0 < 2:
        return img, extent
    # ascontiguousarray: a fatia e uma VIEW do cache de 100 MB; a copia explicita (poucos MB) evita
    # que o reshape la dentro do cartopy a materialize de forma menos previsivel.
    return np.ascontiguousarray(img[j0:j1, i0:i1]), (e_lon0 + i0 / ppd_x, e_lon0 + i1 / ppd_x,
                                                     e_lat1 - j1 / ppd_y, e_lat1 - j0 / ppd_y)


_LAND_GEOM_CACHE: list = []


def _land_geom_raw():
    """Multipolígono de TERRA 50m em lon/lat (sem projeção), cacheado -- p/ recortar a GEOMETRIA
    de um polígono (`shapely.intersection`) antes de desenhar, garantindo que ele nunca contenha
    área de oceano mesmo se `_land_clip_path` (recorte visual, dependente de reprojeção) falhar."""
    if not _LAND_GEOM_CACHE:
        import cartopy.feature as _cf
        from shapely.ops import unary_union
        _LAND_GEOM_CACHE.append(unary_union(list(_cf.LAND.with_scale('50m').geometries())))
    return _LAND_GEOM_CACHE[0]


_LAND_CLIP_CACHE: dict = {}


def _land_clip_path(proj, src):
    """Path (coords PROJETADAS do eixo) da geometria de TERRA 50m, p/ RECORTAR o shaded no litoral
    REAL (liso) via `artist.set_clip_path` -- em vez de mascarar na grade grossa do modelo (que
    serrilha a costa). O oceano fica de fora do recorte -> aparece o blue marble. Cacheado por
    projeção (a câmera do s42 é fixa, então roda 1x por render, ~1s)."""
    _p4 = getattr(proj, 'proj4_params', {}) or {}
    key = (type(proj).__name__, round(float(_p4.get('lon_0', 0.0)), 2),
           round(float(_p4.get('lat_0', 0.0)), 2), round(float(_p4.get('h', 0.0)), 0))
    cached = _LAND_CLIP_CACHE.get(key)
    if cached is not None:
        return cached
    import cartopy.feature as _cf
    from cartopy.mpl.path import shapely_to_path
    from shapely.ops import unary_union
    _land = unary_union(list(_cf.LAND.with_scale('50m').geometries()))
    try:
        path = shapely_to_path(proj.project_geometry(_land, src))
    except Exception as _e:
        logger.warning('recorte vetorial de terra falhou ({}); shaded sem recorte de costa', _e)
        path = None
    _LAND_CLIP_CACHE[key] = path
    return path


_BRASIL_CLIP_CACHE: dict = {}


def _brasil_clip_path(proj, src):
    """Path (coords PROJETADAS do eixo) da geometria do BRASIL (Natural Earth admin_0), p/
    RECORTAR o shaded via `artist.set_clip_path` -- so aparecem dados DENTRO do territorio
    brasileiro (`PLOTAR_SOMENTE_BRASIL`), independente de camera/voo (s42) ou area (s43).
    Mesmo padrao/cache de `_land_clip_path`."""
    _p4 = getattr(proj, 'proj4_params', {}) or {}
    key = (type(proj).__name__, round(float(_p4.get('lon_0', 0.0)), 2),
           round(float(_p4.get('lat_0', 0.0)), 2), round(float(_p4.get('h', 0.0)), 0))
    cached = _BRASIL_CLIP_CACHE.get(key)
    if cached is not None:
        return cached
    import cartopy.io.shapereader as shpreader
    from cartopy.mpl.path import shapely_to_path
    from shapely.ops import unary_union
    ne_path = shpreader.natural_earth(resolution='50m', category='cultural', name='admin_0_countries')
    geoms = [rec.geometry for rec in shpreader.Reader(ne_path).records()
             if (rec.attributes.get('ADM0_A3') or rec.attributes.get('adm0_a3')) == 'BRA']
    _brasil = unary_union(geoms)
    try:
        path = shapely_to_path(proj.project_geometry(_brasil, src))
    except Exception as _e:
        logger.warning('recorte vetorial do Brasil falhou ({}); shaded sem recorte de Brasil', _e)
        path = None
    _BRASIL_CLIP_CACHE[key] = path
    return path


def _chaikin_smooth_closed(pontos: list[tuple[float, float]], iteracoes: int = 4) -> list[tuple[float, float]]:
    """Arredonda as quinas de um polígono fechado (corner-cutting de Chaikin) -- os pontos digitados
    manualmente formam um contorno reto/quadrado; cada iteração substitui cada vértice por dois
    pontos a 1/4 e 3/4 da aresta, convergindo para uma curva suave."""
    pts = list(pontos)
    for _ in range(iteracoes):
        n = len(pts)
        novos = []
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            novos.append((0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1]))
            novos.append((0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1]))
        pts = novos
    return pts


def _draw_poligonos_severo(ax, ctx: dict, proj, data_transform) -> None:
    """Desenha os polígonos de tempo severo (`ctx['poligonos_severo']`, ver GLOBO_3D_POLIGONOS_
    SEVERO): preenchimento + borda por cima do shaded (zorder=3, entre o shaded em 2 e a costa/
    fronteiras/estados em 5 -- nunca cobre as linhas de divisa). A GEOMETRIA e recortada no
    CONTINENTE via `shapely.intersection` com `_land_geom_raw()` ANTES de desenhar (garante que o
    polígono nunca contem área de oceano, mesmo que os pontos informados cruzem a costa -- ao
    contrario de um recorte so visual via `set_clip_path`, que depende da reprojecao vetorial ter
    sucesso). `_land_clip_path` e aplicado por cima como reforço (defesa em profundidade), mas a
    garantia real e a intersecao. As quinas retas dos pontos informados sao arredondadas (Chaikin)
    antes de virar geometria."""
    poligonos = ctx.get('poligonos_severo')
    if not poligonos:
        return
    from shapely.geometry import Polygon as _ShpPolygon
    from shapely.geometry.base import BaseGeometry as _ShpBaseGeom
    _land_cp = _land_clip_path(proj, data_transform)
    _land_geom = _land_geom_raw()
    for pg in poligonos:
        try:
            _poly = _ShpPolygon(_chaikin_smooth_closed(pg['pontos_lonlat']))
        except Exception as _e:
            logger.warning('Polígono de tempo severo inválido ({}); pulando', _e)
            continue
        _poly_land = _poly.intersection(_land_geom)
        if _poly_land.is_empty:
            logger.warning('Polígono de tempo severo cai inteiramente no oceano; pulando')
            continue
        # intersection pode devolver Polygon/MultiPolygon/GeometryCollection (com pontos/linhas
        # residuais de toques na costa) -- add_geometries so aceita (Multi)Polygon.
        _partes = ([g for g in _poly_land.geoms if isinstance(g, _ShpBaseGeom) and g.geom_type == 'Polygon']
                   if _poly_land.geom_type == 'GeometryCollection' else [_poly_land])
        if not _partes:
            continue
        _art = ax.add_geometries(
            _partes, crs=data_transform, facecolor=pg['cor'], alpha=pg['alpha'],
            edgecolor=pg['cor_borda'], linewidth=pg['lw_borda'], zorder=3)
        if _land_cp is not None:
            _art.set_clip_path(_land_cp, ax.transData)
        if pg.get('texto'):
            _cx, _cy = _poly_land.centroid.x, _poly_land.centroid.y
            _txt = ax.text(_cx, _cy, str(pg['texto']), transform=data_transform, ha='center', va='center',
                           color=pg['cor_borda'], fontsize=9, fontweight='bold', zorder=4)
            if _land_cp is not None:
                _txt.set_clip_path(_land_cp, ax.transData)


_ATMOSPHERE_CACHE: dict[tuple, np.ndarray] = {}


def _atmosphere_glow_rgba(h: int, w: int, cx: float = 0.50, cy: float = 0.49,
                           r_frac: float = 0.433) -> np.ndarray:
    """Halo azul-ciano ao redor do disco do globo (estilo atmosfera Google Earth).

    Overlay RGBA calculado uma vez por resolucao: transparente sobre os dados,
    anel brilhante na borda do disco e halo que se esvai no espaco exterior.

    cx, cy : centro do disco em fracoes de figura (y=0 e inferior)
    r_frac : raio do disco como fracao da dimensao do frame (default ~43% da figura
             para o rect [0.07, 0.06, 0.86, 0.86] da projecao NearsidePerspective GEO)
    """
    key = (h, w, cx, cy, r_frac)
    if key in _ATMOSPHERE_CACHE:
        return _ATMOSPHERE_CACHE[key]

    rows, cols = np.mgrid[0:h, 0:w]
    cx_px = cx * w
    cy_px = (1.0 - cy) * h      # y de figura (0=inferior) -> linha de imagem (0=superior)
    r_px = np.sqrt((cols - cx_px) ** 2 + (rows - cy_px) ** 2)
    R = r_frac * min(h, w)      # raio do disco em pixels

    # Halo externo: anel que se esvai suavemente fora do disco (15% do raio de espessura)
    glow_w = 0.15 * R
    t_out = np.clip((r_px - R) / glow_w, 0.0, 1.0)
    alpha_out = (1.0 - t_out) ** 2.2 * 0.70

    # Borda interna: fade de 6px que iguala o alpha do halo externo na borda do disco.
    # Aplicado so em pixels escuros (mascara is_dark em _build_frame) para nao afetar
    # os dados coloridos — cobre o clip boundary do cartopy sem alterar o mapa.
    edge_px = 6.0
    t_in = np.clip((r_px - (R - edge_px)) / edge_px, 0.0, 1.0)
    alpha_in = t_in ** 1.5 * 0.70  # bate no alpha do halo externo na borda (continuidade)

    alpha = np.where(r_px >= R, alpha_out, alpha_in)

    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[..., 0] = 0.42     # R  ┐
    rgba[..., 1] = 0.76     # G  ├ azul-ciano claro (~#6bc2ff)
    rgba[..., 2] = 1.00     # B  ┘
    rgba[..., 3] = alpha.astype(np.float32)

    _ATMOSPHERE_CACHE[key] = rgba
    return rgba


_STARFIELD_CACHE: dict[tuple, np.ndarray] = {}


def _starfield_rgba(h: int, w: int, cx: float = 0.50, cy: float = 0.49,
                    r_disc: float = 0.433, n_stars: int = 700, seed: int = 42) -> np.ndarray:
    """Campo de estrelas aleatorias no fundo escuro, fora do disco terrestre.

    Pontos brancos com brilho variado: maioria fracas (1px), algumas mais brilhantes
    com halo de 1px ao redor. Posicoes fixas (seed deterministico) para nao piscar
    entre frames. Calculado uma vez por resolucao.
    """
    key = (h, w, cx, cy, r_disc, n_stars, seed)
    if key in _STARFIELD_CACHE:
        return _STARFIELD_CACHE[key]

    rng = np.random.default_rng(seed)
    overlay = np.zeros((h, w, 4), dtype=np.float32)

    cx_px, cy_px = cx * w, (1.0 - cy) * h
    R = r_disc * min(h, w)

    # Gera candidatos e filtra os que estao fora do disco
    n_cand = n_stars * 6
    xs = rng.integers(2, w - 2, size=n_cand)
    ys = rng.integers(2, h - 2, size=n_cand)
    r = np.sqrt((xs - cx_px) ** 2 + (ys - cy_px) ** 2)
    mask = r > R
    xs, ys = xs[mask][:n_stars], ys[mask][:n_stars]

    # Brilho: distribuicao de potencia — muitas estrelas fracas, poucas brilhantes
    brightness = rng.uniform(0.0, 1.0, size=len(xs)) ** 1.8
    brightness = np.clip(brightness * 0.85 + 0.15, 0.15, 1.0)  # [0.15 .. 1.0]

    # Estrelas brilhantes (brightness > 0.65): halo de 1px ao redor
    bright = brightness > 0.65
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny = np.clip(ys[bright] + dy, 0, h - 1)
        nx = np.clip(xs[bright] + dx, 0, w - 1)
        cur = overlay[ny, nx, 3]
        halo_b = brightness[bright] * 0.25
        overlay[ny, nx, :3] = 1.0
        overlay[ny, nx, 3] = np.maximum(cur, halo_b)

    # Ponto central de todas as estrelas (sobrepoe o halo com brilho pleno)
    overlay[ys, xs, :3] = 1.0
    overlay[ys, xs, 3] = brightness

    _STARFIELD_CACHE[key] = overlay
    return overlay


_SAT_HEIGHT_GEO = 35785831.0  # altura geoestacionaria (m) — globo "flutuante" padrao (s38/s39)


def _make_projection(central_lon: float, central_lat: float,
                     mode: str = 'nearside', sat_height: float = _SAT_HEIGHT_GEO):
    """Projecao do globo centrada em (lon, lat).

    mode:
      'orthographic'  -> Orthographic (sem perspectiva; disco cheio)
      'nearside'      -> NearsidePerspective a altura geoestacionaria (globo flutuante; s38/s39)
      'google_earth'  -> NearsidePerspective a altura MENOR (`sat_height`) -> camera mais perto,
                         regiao ampliada e com mais curvatura (estilo Google Earth; s41)
    """
    m = str(mode).lower()
    if m.startswith('ortho'):
        proj = ccrs.Orthographic(central_longitude=central_lon, central_latitude=central_lat)
    else:
        proj = ccrs.NearsidePerspective(
            central_longitude=central_lon, central_latitude=central_lat, satellite_height=sat_height,
        )
    # `threshold` (cartopy) controla a precisao da tesselacao usada p/ recortar geometrias (costa/
    # fronteiras/estados) contra o limite do disco visivel -- o padrao e' grosso demais em
    # enquadramentos bem recortados/inclinados (ex.: `chile_central`, s46), causando GAPS visiveis
    # em linhas de fronteira perto da borda do disco (bug real: divisa Bolivia-Peru-Chile quebrada,
    # sumindo um trecho inteiro). Reduzir por 50x (validado: fecha o gap sem custo perceptivel de
    # render) deixa a tesselacao fina o bastante pra nao cortar geometria por engano.
    proj.threshold /= 50.0
    return proj


def _aplicar_janela_inclinada(ax, proj, inc: dict | None) -> None:
    """Globo 'deitado'/inclinado (SO s44): troca o `set_global()` (disco centralizado, visto de cima)
    por uma JANELA DE RECORTE DESCENTRALIZADA em coords NATIVAS da projecao (metros), deslocada p/
    baixo do nadir -> o horizonte curva so no topo do quadro e o chao preenche o resto, simulando o
    pitch que a NearsidePerspective do Cartopy nao tem (ela e sempre vista de cima). Tecnica portada
    do modulo `globo_inclinado_3d` (mapas de chuva), generalizada.

    `inc=None` (TODOS os outros scripts, s38-s43) -> no-op: nao mexe no `set_global()` ja aplicado,
    entao o comportamento deles fica IDENTICO ao de antes. `inc` = dict com 'inclinacao' (fracao do
    raio p/ descer o centro da janela), 'janela_frac' (largura da janela / diametro do disco) e
    'aspecto' (largura/altura do quadro; a altura da janela = largura / aspecto)."""
    if not inc:
        return
    raio = proj.x_limits[1]                       # raio do disco em metros (NearsidePerspective simetrica)
    larg = float(inc['janela_frac']) * 2.0 * raio
    alt = larg / float(inc['aspecto'])
    cy = float(inc['deitar']) * raio              # >0 desloca a janela p/ norte (nadir sai por baixo)
    ax.set_xlim(-larg / 2.0, larg / 2.0)
    ax.set_ylim(cy - alt / 2.0, cy + alt / 2.0)


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


def _script_setting(script_id: str, suffix: str, default):
    """Setting com override POR-SCRIPT do s41/s42: GLOBO_3D_GE_<suffix> (s41/s42) tem precedencia
    sobre GLOBO_3D_<suffix> (compartilhado com s38/s39). Para os demais scripts, so o compartilhado.
    Permite regular voo/inclinacao/zoom/velocidade do s41/s42 sem afetar o s39. O s42 e copia fiel
    do s41 (mesmo namespace GE_) ate divergir com sua propria config.

    EXCECAO: as flags das CORRENTES DE JATO (JATO*, JET_STREAM*, SUBTROPICAL_JET*) sao UNIFICADAS
    entre s38/s39/s40/s41/s42 -> leem SEMPRE o GLOBO_3D_<suffix> compartilhado (s41/s42 NAO tem
    override GE para o jato). Assim um unico master GLOBO_3D_JATO liga/estiliza o jato em todos."""
    base = settings.get(f'GLOBO_3D_{suffix}', default)
    if script_id in ('s41', 's42') and not suffix.startswith(('JATO', 'JET_STREAM', 'SUBTROPICAL_JET')):
        return settings.get(f'GLOBO_3D_GE_{suffix}', base)
    # s44/s46 (globo inclinado): namespace proprio GLOBO_3D_INC_<suffix> (mesma logica do GE_ do s41/s42)
    # -> podem regular voo/altura/enquadramento SEM afetar s38-s43 (que nunca leem INC_). s46 e copia do
    # s44 (globo inclinado de ALERTA) -> compartilha o mesmo namespace INC_ e comportamento inclinado.
    if script_id in ('s44', 's46', 's47') and not suffix.startswith(('JATO', 'JET_STREAM', 'SUBTROPICAL_JET')):
        return settings.get(f'GLOBO_3D_INC_{suffix}', base)
    return base


def _camera_path(total_frames: int, script_id: str = '') -> tuple[np.ndarray, np.ndarray]:
    """Trajetoria (lon, lat) da camera do 1o ao ultimo frame.

    Viaja de (LON/LAT_INICIAL) a (LON/LAT_FINAL) pelo menor arco em longitude,
    somando VOLTAS_EXTRA giros completos (para o efeito de rotacao). O perfil de
    velocidade vem de GLOBO_3D_EASING (constante ou com desaceleracao no fim).
    O s41 (Google Earth) pode sobrescrever cada parametro via GLOBO_3D_GE_<suffix>.
    """
    lon_i = float(_script_setting(script_id, 'LON_INICIAL', -150.0))
    lat_i = float(_script_setting(script_id, 'LAT_INICIAL', 0.0))
    lon_f = float(_script_setting(script_id, 'LON_FINAL', -45.0))
    lat_f = float(_script_setting(script_id, 'LAT_FINAL', -15.0))
    voltas_extra = float(_script_setting(script_id, 'VOLTAS_EXTRA', 0.0))
    easing = str(_script_setting(script_id, 'EASING', 'linear'))

    # Inclinacao FIXA do globo (opcional): sobrescreve a latitude da camera para
    # mostrar mais um hemisferio. >0 = mais Hemisferio Norte; <0 = mais Sul; 0 = equador.
    # Vazio/""/None = usa LAT_INICIAL/LAT_FINAL (inclinacao varia durante o voo).
    # s41 tem o proprio corte HS/HN via GLOBO_3D_GE_INCLINACAO.
    inclin = _script_setting(script_id, 'INCLINACAO', '')
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
# Costura da grade ciclica LONGE do Meridiano de Greenwich (s38-s42): a corrente de jato e
# reconstruida (`_jet_segments`, isolinha circumpolar) e re-rasterizada (`_jato_raster`) tratando
# o array de longitude como um retangulo comum, SEM saber que a borda esquerda (lon.min()) e a
# direita (lon.max()) sao o MESMO lugar fisico no globo. Qualquer efeito de borda dessa
# reconstrucao/composicao acontece exatamente ONDE o array comeca/termina -- e por padrao isso e
# sempre em 0°/360° (Meridiano de Greenwich), pois `add_cyclic_point` fecha o array ali. Quando a
# camera enquadra essa regiao (Europa/Atlantico), qualquer artefato cai bem no meio da area
# visivel. A solucao: girar a ORDEM de armazenamento da grade (sem alterar nenhum ponto fisico)
# para que a borda do array fique em ~160°E (meio do Pacifico) em vez de 0° -- fora de vista em
# qualquer enquadramento tipico deste projeto (Europa, Atlantico, Americas).
# ---------------------------------------------------------------------------
def _lon_seam_alvo(lons: np.ndarray, alvo: float = 160.0, margem: float = 100.0) -> float:
    """Decide se a costura da grade ciclica deve ser deslocada para `alvo` (graus leste).

    Verifica se a camera (fixa ou em voo -- `lons` traz TODAS as posicoes do clipe) chega perto
    do Meridiano de Greenwich (lon 0°) em algum frame, dentro de `margem` graus (folga alem do
    hemisferio visivel tipico, ~90°). Retorna `alvo` se sim, ou `0.0` (sem deslocamento,
    comportamento padrao) se a camera nunca se aproxima de Greenwich durante o clipe inteiro."""
    dist = np.abs(((np.asarray(lons, dtype=float) - 0.0 + 180.0) % 360.0) - 180.0)
    return float(alvo) if bool(np.any(dist <= margem)) else 0.0


def _lon_seam_roll_index(lon_cyc: np.ndarray, seam_alvo: float) -> int:
    """Indice em `lon_cyc[:-1]` (saida de `add_cyclic_point` SEM o ponto de fechamento duplicado
    no fim) mais proximo de `seam_alvo` -- usado para saber quantas posicoes rolar a grade em
    `_lon_seam_roll`/`_lon_seam_roll_lon`, que recriam o fechamento no novo lugar."""
    lon0 = np.asarray(lon_cyc, dtype=float)[:-1]
    return int(np.argmin(np.abs(((lon0 - seam_alvo + 180.0) % 360.0) - 180.0)))


def _lon_seam_roll_lon(lon_cyc: np.ndarray, k: int) -> np.ndarray:
    """Rola o array de COORDENADAS `lon_cyc` (saida de `add_cyclic_point`) por `k` posicoes.
    Descarta o ponto de fechamento duplicado (`lon_cyc[-1] == lon_cyc[0] + 360`) antes de rolar
    e RECRIA-o no novo lugar -- sem isso o duplicado ficaria preso no MEIO do array apos a
    rolagem (dois pontos com a mesma longitude), reintroduzindo um mini-artefato de borda.
    Soma +360° a parte deslocada para manter a ordem CRESCENTE (sem isso o array teria um salto
    decrescente no meio, quebrando contourf/interp/contourpy). Mesmos pontos fisicos, so a
    ORDEM/rotulagem muda."""
    if not k:
        return lon_cyc
    lon0 = np.asarray(lon_cyc, dtype=float)[:-1]
    lon_r = np.roll(lon0, -k)
    lon_r[-k:] += 360.0
    return np.concatenate([lon_r, [lon_r[0] + 360.0]])


def _lon_seam_roll(vals: np.ndarray, k: int) -> np.ndarray:
    """Rola `vals` (ultimo eixo = longitude, saida de `add_cyclic_point`) por `k` posicoes --
    par de `_lon_seam_roll_lon` (mesma logica de descartar e recriar o ponto de fechamento) para
    mover a costura da grade ciclica sem alterar nenhum ponto fisico. contourf/pcolormesh/contour
    renderizam identico (a projecao trata longitude como periodica); so o local onde efeitos de
    BORDA do array acontecem (isolinha circumpolar do jato) muda de lugar."""
    if not k:
        return vals
    vals0 = vals[..., :-1]
    vals_r = np.roll(vals0, -k, axis=-1)
    return np.concatenate([vals_r, vals_r[..., :1]], axis=-1)


# ---------------------------------------------------------------------------
# Renderizacao de UM frame (usada em serie ou em paralelo via process pool).
# ---------------------------------------------------------------------------
_FRAME_CTX: dict | None = None
_OV_CACHE_KEY: int | None = None   # último idx_dia do overlay Guillaume (por worker)
_OV_CACHE_VAL: np.ndarray | None = None  # overlay correspondente


def _init_frame_worker(ctx: dict) -> None:
    """Inicializa cada processo worker com o contexto do clipe (1x por worker)."""
    global _FRAME_CTX
    _FRAME_CTX = ctx


def _fmt_nivel_hpa(s: str) -> str:
    """Normaliza o nivel de pressao nos titulos do globo: 'NNN-hPa' / 'NNN-HPA' / 'NNNHPA'
    (qualquer caixa, com ou sem hifen) -> 'NNN hPa'. Necessario porque a caixa de texto do s39
    e escrita toda em MAIUSCULA (viraria '250-HPA'); mantem o 'hPa' correto e com espaco."""
    return re.sub(r'(\d+)\s*-?\s*hpa', r'\1 hPa', str(s), flags=re.IGNORECASE)


def _numeric_legend_ticks(vmin: float, vmax: float, step: float) -> list[float]:
    """Ticks numericos simetricos em torno de 0, de `step` em `step`, dentro de [vmin, vmax]."""
    if step <= 0:
        step = 0.5
    n = int(np.floor(max(abs(vmin), abs(vmax)) / step + 1e-9))
    vals = [round(k * step, 6) for k in range(-n, n + 1)]
    return [v for v in vals if vmin - 1e-9 <= v <= vmax + 1e-9]


def _draw_logo_canto(fig, ctx: dict) -> None:
    """Logo no canto inferior DIREITO (so quando ctx['logo_path'] esta setado -- s44). O caminho ja
    vem resolvido por `resolve_logo_path` (prioridade CAPUCIN > GREC > AMPERE; None = nenhuma flag
    ligada -> sem logo). Desenhado via fig.figimage (pixels), acima da linha do credito."""
    _lp = ctx.get('logo_path')
    if not _lp or not Path(_lp).exists():
        return
    w_in, h_in = fig.get_size_inches()
    dpi = float(fig.dpi)
    w_px, h_px = int(round(w_in * dpi)), int(round(h_in * dpi))
    _frac = float(ctx.get('logo_largura_frac', 0.16))
    larg = max(1, round(_frac * min(w_px, h_px)))
    img = _PIL_Image.open(_lp).convert('RGBA')
    alt = max(1, round(img.height * larg / img.width))
    img = img.resize((larg, alt), _PIL_Image.LANCZOS)
    # Margens x (borda DIREITA) e y (borda INFERIOR) como fracao — permitem enquadrar o logo na
    # quina inf. direita (s44: GLOBO_3D_INC_LOGO_MARGEM_X/Y_FRAC via ctx). Antes eram fixas
    # (0.02*min p/ x e 0.02*min + 0.04*h p/ y = ~0.06*h, p/ subir acima do credito).
    _mx = float(ctx.get('logo_margem_x_frac', 0.02))
    _my = float(ctx.get('logo_margem_y_frac', 0.06))
    xo = max(0, w_px - larg - round(_mx * min(w_px, h_px)))   # canto DIREITO
    yo = max(0, round(_my * h_px))                            # a partir da borda inferior (acima do credito)
    fig.figimage(np.asarray(img), xo=xo, yo=yo, zorder=22)


def _cor_texto_contraste(cor) -> str:
    """Preto ou branco, o que ler melhor sobre `cor` (luminancia perceptual). As faixas claras da
    chuva (amarelo, rosa, cinzas) precisam de texto PRETO; as escuras, branco."""
    r, g, b = to_rgba(cor)[:3]
    return 'black' if (0.299 * r + 0.587 * g + 0.114 * b) > 0.62 else 'white'


def _legenda_faixas_twc(fig, ctx: dict, cmap, x_ini: float, x_dir: float, y_topo: float,
                        font: str) -> None:
    """Legenda de FAIXAS do estilo TWC: uma caixinha por banda de `levels`, enfileiradas, com borda
    branca e o rotulo dentro (ex.: '<25mm', '25-50mm').

    Regras (todas visiveis no layout):
      * so as bandas que APARECEM no mapa -- ate `ctx['twc_legenda_ate']` (ver `_ultima_banda_pintada`);
      * TODAS as linhas comecam em `x_ini` (coladas na caixa do modelo, alinhadas entre si);
      * `x_dir` (borda direita da caixa cinza da data) e o LIMITE: a caixinha que passaria dele desce
        p/ a linha de baixo;
      * cor do texto por contraste com a faixa (`_cor_texto_contraste`).
    Os rotulos saem dos PROPRIOS `levels`/cores do render -- nao ha texto fixo p/ desalinhar da escala.
    """
    ate = ctx.get('twc_legenda_ate')
    _lv = ctx.get('levels')                  # numpy array -> `or []` daria "truth value ambiguous"
    levels = [] if _lv is None else [float(v) for v in _lv]
    if ate is None or len(levels) < 2:
        return
    n = min(int(ate) + 1, len(levels) - 1)
    if isinstance(cmap, ListedColormap) and len(cmap.colors) >= n:
        cores = [cmap.colors[i] for i in range(n)]
    else:                                    # cmap continuo -> amostra no meio de cada banda
        cores = [cmap(i / max(len(levels) - 2, 1)) for i in range(n)]
    un = str(ctx.get('twc_legenda_unidade', ''))
    # 1a faixa como '<X': o render nao pinta abaixo de levels[0] (extend='max'), entao o piso real e
    # levels[0] -- mas o rotulo '<25mm' e o que o usuario le como "pouca chuva".
    labs = [f'<{levels[1]:g}{un}' if i == 0 else f'{levels[i]:g}-{levels[i + 1]:g}{un}'
            for i in range(n)]
    fs = float(ctx.get('twc_legenda_fontsize', 8.0))
    pad_x, pad_y = 0.005, 0.006
    # Vao ENTRE caixinhas (fracao da largura do quadro; 0.001 ~ 1 px em 1080). A faixa branca que se ve
    # entre duas caixinhas e este vao + as DUAS meias-bordas vizinhas (lw em pontos: 1.2 pt ~ 2.2 px a
    # 135 dpi) -- entao p/ afinar a faixa da p/ mexer nos dois (aqui e no _BORDA_LW).
    gap = float(ctx.get('twc_legenda_gap', 0.001))
    lw = float(ctx.get('twc_legenda_borda_lw', 1.2))
    rend = fig.canvas.get_renderer()

    # Largura de cada caixinha (medida no texto real, sem chute).
    larguras = []
    for lab in labs:
        _t = fig.text(0, 0, lab, fontsize=fs, weight='bold', family=font)
        _bb = _t.get_window_extent(rend).transformed(fig.transFigure.inverted())
        larguras.append(_bb.width + 2 * pad_x)
        _t.remove()

    # Quebra em linhas: cada caixinha tem a largura do SEU texto (justa) e as linhas comecam todas em
    # `x_ini`; a caixinha que passaria de `x_dir` (fim da caixa da data) desce p/ a linha de baixo.
    linhas, atual, acc = [], [], 0.0
    disp = x_dir - x_ini
    for i, w in enumerate(larguras):
        if atual and acc + gap + w > disp:
            linhas.append(atual)
            atual, acc = [], 0.0
        acc += (gap if atual else 0.0) + w
        atual.append(i)
    if atual:
        linhas.append(atual)

    y = y_topo
    for linha in linhas:
        x = x_ini          # toda linha comeca colada na caixa do modelo (e alinhada com as outras)
        alt = None
        for i in linha:
            t = fig.text(x + pad_x, y - pad_y, labs[i], fontsize=fs, ha='left', va='top',
                         weight='bold', color=_cor_texto_contraste(cores[i]), family=font, zorder=22)
            bb = t.get_window_extent(rend).transformed(fig.transFigure.inverted())
            alt = bb.height + 2 * pad_y
            fig.add_artist(FancyBboxPatch(
                (x, bb.y0 - pad_y), larguras[i], alt, boxstyle='square,pad=0',
                transform=fig.transFigure, facecolor=cores[i], edgecolor='white', linewidth=lw,
                zorder=21))
            x += larguras[i] + gap
        y -= (alt or 0.03) + gap


def _ticks_uniformes(vmin: float, vmax: float, alvo: int = 6) -> list[float]:
    """Ticks 'redondos' (passo 1/2/2.5/5 x 10^k) entre `vmin` e `vmax`, SEMPRE incluindo os dois
    extremos -- usado pela legenda-gradiente de chuva/neve (`_legenda_gradiente_chuva_neve`, NEVE=true)
    em vez dos `levels` brutos da ficha (nao-uniformes: 1,25,50,75,125,200,250... -- lotaria de
    numero numa barra continua pequena). `alvo` e a contagem de ticks DESEJADA (aproximada: o passo
    arredondado raramente bate exato, ver `passo_frac`)."""
    if vmax <= vmin:
        return [vmin]
    bruto = (vmax - vmin) / max(alvo - 1, 1)
    exp = np.floor(np.log10(bruto)) if bruto > 0 else 0.0
    frac = bruto / (10 ** exp)
    passo_frac = next((f for f in (1, 2, 2.5, 5, 10) if f >= frac - 1e-9), 10)
    passo = passo_frac * (10 ** exp)
    meio: list[float] = []
    v = np.ceil(vmin / passo) * passo
    while v < vmax - passo * 0.4:
        if v > vmin + passo * 0.4:
            meio.append(round(float(v), 6))
        v += passo
    return [vmin] + meio + [vmax]


def _gradient_rgba(cmap, w_px: int, h_px: int) -> np.ndarray:
    """Array RGBA (h_px, w_px, 4) uint8 com gradiente horizontal CONTINUO de `cmap` (0..1 esq->dir)."""
    grad = np.linspace(0.0, 1.0, max(int(w_px), 2))
    rgba = (np.asarray(cmap(grad)) * 255).astype(np.uint8)          # (w_px, 4)
    return np.repeat(rgba[np.newaxis, :, :], max(int(h_px), 1), axis=0)  # (h_px, w_px, 4)


def _legenda_gradiente_chuva_neve(fig, ctx: dict, x_ini: float, y_topo: float, font: str) -> None:
    """Legenda-gradiente translucida de chuva+neve (SO quando `NEVE=true`, ver `ctx['neve_overlay']`):
    substitui a `_legenda_faixas_twc` (caixinhas por banda, ao LADO da caixa do modelo) por UMA caixa
    translucida (#14223d) colada ABAIXO da caixa do modelo, com DUAS barras continuas empilhadas --
    "Acumulado de chuva (mm)" em cima, "Acumulado de neve (cm)" embaixo -- cada uma com as CORES já
    definidas em `GLOBO_3D_PALETA_PRECIP_ABS`/`GLOBO_3D_NEVE_PALETA` (via `ctx['paleta']`/
    `ctx['neve_overlay']['cmap']`), mostrando SEMPRE o minimo/maximo dos `levels` da ficha nas pontas
    e ticks intermediarios ARREDONDADOS (`_ticks_uniformes`) por legibilidade -- pedido do usuario:
    com neve ligada, as caixinhas de faixa (so da chuva) escondiam a paleta de neve por completo.
    Escala FIXA (chuva ate 500mm, neve ate 100cm, sem teto dinamico por periodo/enquadramento);
    valores acima disso NAO viram buraco/cor errada -- o `extend='max'` do contourf (ver
    `_contourf_raster`) ja garante isso sozinho, sem precisar de nenhum indicador na legenda.

    `x_ini`: borda esquerda (mesma usada pela `_legenda_faixas_twc` -- borda esquerda da caixa do
    modelo). LARGURA da caixa e' FIXA (`_LARGURA_FIXA`, ver abaixo) -- pedido do usuario: antes a
    largura era um % do vao ate a borda direita da caixa da data, entao oscilava se o titulo (azul),
    a data (cinza) ou o nome do modelo mudassem de tamanho/texto. Congelada no valor ja aprovado, faz
    a legenda ficar 100% imune a essas 3 caixas -- so `y_topo` (posicao vertical, nao tamanho) ainda
    acompanha a borda de baixo da caixa do modelo, pra continuar colada logo abaixo dela."""
    neve_ov = ctx.get('neve_overlay')
    if neve_ov is None:
        return
    _lv = ctx.get('levels')
    chuva_levels = [] if _lv is None else [float(v) for v in _lv]
    chuva_cores = list(ctx.get('paleta') or [])
    # `neve_ov['levels']` e' numpy array (np.asarray na montagem do overlay, ver `neve_overlay = {`
    # mais acima) -- `array or []` explode ("truth value... ambiguous") com mais de 1 elemento; por
    # isso o mesmo padrao `is None` do `chuva_levels` acima, em vez de `or`.
    _nlv = neve_ov.get('levels')
    neve_levels = [] if _nlv is None else [float(v) for v in _nlv]
    neve_cores = list(getattr(neve_ov.get('cmap'), 'colors', None) or [])
    if len(chuva_levels) < 2 or len(chuva_cores) < 2 or len(neve_levels) < 2 or len(neve_cores) < 2:
        return

    w_in, h_in = fig.get_size_inches()
    dpi = float(fig.dpi)
    w_px, h_px = int(round(w_in * dpi)), int(round(h_in * dpi))
    rend = fig.canvas.get_renderer()
    # `_escala` (~1/3 da area = ~0.55 linear): pedido do usuario -- 1a versao saiu grande demais
    # (fonte/barra do tamanho da caixa "Modelo X" e largura = header inteiro). So a LARGURA usa a
    # escala diretamente sobre o vao disponivel (x_ini..x_dir); fonte/barra/paddings tambem escalam
    # pra manter as PROPORCOES da caixa (nao só encolher o texto dentro de uma caixa ainda grande).
    _escala = 0.55
    # `fs_tick` NAO acompanha a `_escala` da caixa (pedido do usuario: numeros maiores que o resto,
    # mesmo que a caixa precise crescer um pouco pra caber -- por isso o `gap_barra_ticks`/altura da
    # caixa continuam medidos no texto REAL via `get_window_extent`, entao acompanham sozinhos).
    fs_titulo, fs_tick = 9.5 * _escala, 5.8
    pad_x, pad_y = 0.010 * _escala, 0.008 * _escala
    barra_h_px = max(3, round(0.019 * _escala * h_px))
    gap_titulo_barra, gap_barra_ticks, gap_secoes = (0.003 * _escala, 0.005 * _escala,
                                                      0.012 * _escala)
    # Largura FIXA (fracao da figura), congelada a partir do ajuste ja aprovado pelo usuario --
    # NAO depende mais da caixa azul (titulo), cinza (data) ou do nome do modelo: antes era um % do
    # vao ate a borda direita da caixa da data, e oscilava se qualquer uma delas mudasse de texto.
    _LARGURA_FIXA = 0.2427
    x_dir_efetivo = x_ini + _LARGURA_FIXA

    def _secao(y: float, titulo: str, cores: list, vmin: float, vmax: float) -> float:
        t = fig.text(x_ini + pad_x, y, titulo, color='white', fontsize=fs_titulo, ha='left', va='top',
                     weight='bold', family=font, zorder=22)
        bb_t = t.get_window_extent(rend).transformed(fig.transFigure.inverted())
        y_barra_topo = bb_t.y0 - gap_titulo_barra
        larg_barra = max(0.01, (x_dir_efetivo - pad_x) - (x_ini + pad_x))
        grad = _gradient_rgba(LinearSegmentedColormap.from_list('legenda_grad', cores),
                              larg_barra * w_px, barra_h_px)
        xo = round((x_ini + pad_x) * w_px)
        yo = round(y_barra_topo * h_px) - barra_h_px
        fig.figimage(grad, xo=xo, yo=yo, zorder=22)
        y_tick = (yo / h_px) - gap_barra_ticks
        for v in _ticks_uniformes(vmin, vmax):
            frac = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.0
            xt = (x_ini + pad_x) + frac * larg_barra
            ha = 'left' if frac <= 0.02 else ('right' if frac >= 0.98 else 'center')
            fig.text(xt, y_tick, f'{v:g}', color='white', fontsize=fs_tick, ha=ha, va='top',
                     weight='bold', family=font, zorder=22)
        _tm = fig.text(0, 0, '0', fontsize=fs_tick, family=font)
        alt_tick = _tm.get_window_extent(rend).transformed(fig.transFigure.inverted()).height
        _tm.remove()
        return y_tick - alt_tick

    y = y_topo - pad_y
    # Chuva: minimo mostrado = levels[1] (25 mm, a 1a banda "de verdade" -- levels[0]=1 e so o piso
    # "comecou a chover" do contourf, mesma convencao do rotulo '<25mm' da `_legenda_faixas_twc`).
    # `chuva_cores[1:]` acompanha o corte -- senao a cor da 1a banda (1-25mm) ainda tingiria a ponta
    # esquerda do gradiente mesmo com o tick comecando em 25.
    y = _secao(y, 'Acumulado de chuva (mm)', chuva_cores[1:] or chuva_cores,
              chuva_levels[1] if len(chuva_levels) > 1 else chuva_levels[0], chuva_levels[-1])
    y -= gap_secoes
    y = _secao(y, 'Acumulado de neve (cm)', neve_cores, neve_levels[0], neve_levels[-1])
    y_fim = y - pad_y

    fig.add_artist(FancyBboxPatch(
        (x_ini, y_fim), x_dir_efetivo - x_ini, y_topo - y_fim, boxstyle='square,pad=0',
        transform=fig.transFigure, facecolor='#14223d', edgecolor='none', alpha=0.72, zorder=20))


def _overlay_guillaume(fig, ctx: dict, cmap, data_full: str, data_br: str = '') -> None:
    """Overlay estilo Guillaume Jauseau (s39): caixa do nome no topo-esquerdo + data,
    barra de gradiente continua numa caixa translucida no centro-inferior, e rodape
    com modelo/rodada (esq.) e credito (dir.). Sem vinheta.

    `ctx['so_credito']` (so s42, opt-in): pula TUDO (caixa do nome, data, subtitulo,
    barra/legenda) e deixa SO o credito no rodape -- exceto a caixa "The Weather Channel"
    (`ctx['titulo_twc']`, opcional): titulo azul quadrado + caixa cinza com a data/hora
    sinotica em formato BR (`data_br`), lado a lado, se um titulo estiver configurado.
    A caixa cinza mostra `data_br` por default; `ctx['titulo_twc_data']` (se preenchido)
    sobrescreve esse texto -- a caixa continua se auto-ajustando ao tamanho da string.

    `ctx['sem_variavel']` (SO s42/s43, ficha 'sem_variavel'): sem campo shaded nao ha o que
    rotular/legendar, entao segue o MESMO caminho do `so_credito` acima (so credito + TWC
    opcional), independente do `GLOBO_3D_SO_CREDITO` estar ligado ou nao."""
    _draw_logo_canto(fig, ctx)   # logo no canto inf. direito (so s44, via ctx['logo_path'])
    if ctx.get('so_credito') or ctx.get('sem_variavel'):
        _titulo_twc = str(ctx.get('titulo_twc', '')).strip()
        if _titulo_twc:
            _font_twc = ctx.get('font_twc') or ctx.get('font_legenda', FONT_SANS)
            _texto_data = str(ctx.get('titulo_twc_data', '')).strip() or data_br
            x0, y0, pad_x, pad_y, gap = 0.045, 0.930, 0.010, 0.012, 0.0
            # ── Caixa AZUL: titulo (quinas quadradas, texto branco) ──
            t1 = fig.text(x0, y0, _titulo_twc, color='white', fontsize=19, ha='left', va='top',
                         weight='bold', family=_font_twc, zorder=21)
            bb1 = t1.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
            fig.add_artist(FancyBboxPatch(
                (bb1.x0 - pad_x, bb1.y0 - pad_y), bb1.width + 2 * pad_x, bb1.height + 2 * pad_y,
                boxstyle='square,pad=0', transform=fig.transFigure,
                facecolor=ctx.get('titulo_twc_cor', '#0077a7'), edgecolor='none', zorder=20))
            # ── Caixa CINZA: data/hora BR por default, ou `titulo_twc_data` se preenchido
            # (quinas quadradas, colada na azul) ──
            # LARGURA FIXA NO CLIPE, ditada pelo rotulo MAIS LARGO de `dates_br` (nao pelo texto do
            # frame): com ACUMULAR_NO_TEMPO o rotulo cresce ('Jul 17' -> 'Jul 17–25') e uma caixa
            # justa ao texto INCHARIA durante o video. Texto CENTRADO nela (justo a esquerda ficaria
            # torto quando o rotulo e curto). Com texto fixo em `titulo_twc_data` ha um rotulo so ->
            # a largura e a dele, como antes.
            _rend = fig.canvas.get_renderer()
            x2 = bb1.x0 - pad_x + (bb1.width + 2 * pad_x) + gap + pad_x
            _cands = ({_texto_data} if str(ctx.get('titulo_twc_data', '')).strip()
                      else set(str(d) for d in (ctx.get('dates_br') or [])) or {_texto_data})
            _w_data = 0.0
            for _lab in _cands:
                _tm = fig.text(0, 0, _lab, fontsize=19, weight='bold', family=_font_twc)
                _w_data = max(_w_data, _tm.get_window_extent(_rend)
                              .transformed(fig.transFigure.inverted()).width)
                _tm.remove()
            t2 = fig.text(x2 + _w_data / 2.0, y0, _texto_data, color='#36566a', fontsize=19,
                          ha='center', va='top', weight='bold', family=_font_twc, zorder=21)
            bb2 = t2.get_window_extent(_rend).transformed(fig.transFigure.inverted())
            fig.add_artist(FancyBboxPatch(
                (x2 - pad_x, bb2.y0 - pad_y), _w_data + 2 * pad_x, bb2.height + 2 * pad_y,
                boxstyle='square,pad=0', transform=fig.transFigure,
                facecolor='#eaebed', edgecolor='none', zorder=20))
            _x_data_dir = x2 + _w_data + pad_x   # borda direita da caixa: agora CONSTANTE no clipe
            # ── Caixa MODELO (opcional): "Modelo <MODELO>" COLADA sob a azul e alinhada a esquerda
            # com ela. Ancorada no PATCH azul, nao no texto: a borda dele e (bb1.x0-pad_x, bb1.y0-pad_y),
            # entao somar o pad PROPRIO desta caixa ao ancorar faz as duas bordas coincidirem exatamente,
            # independente do texto/fonte. O modelo vem do `fonte_label` (o mesmo que nomeia a pasta de
            # saida e o rodape) -> segue sozinho o RUN_* habilitado, e vira 'ERA5 + <MODELO>' na emenda.
            if ctx.get('twc_modelo') and str(ctx.get('fonte_label', '')).strip():
                pad_x3, pad_y3 = 0.008, 0.009
                t3 = fig.text((bb1.x0 - pad_x) + pad_x3, (bb1.y0 - pad_y) - pad_y3,
                              f"{ctx.get('twc_modelo_prefixo', 'Modelo')} "
                              f"{str(ctx['fonte_label']).upper()}".strip(),
                              color='white', fontsize=float(ctx.get('twc_modelo_fontsize', 12.0)),
                              ha='left', va='top', weight='bold', family=_font_twc, zorder=21)
                bb3 = t3.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
                fig.add_artist(FancyBboxPatch(
                    (bb3.x0 - pad_x3, bb3.y0 - pad_y3), bb3.width + 2 * pad_x3, bb3.height + 2 * pad_y3,
                    boxstyle='square,pad=0', transform=fig.transFigure,
                    facecolor=str(ctx.get('twc_modelo_cor', '#14223d')), edgecolor='none', zorder=20))
                if ctx.get('neve_overlay') is not None:
                    # NEVE=true: a legenda de FAIXAS (so chuva) escondia a paleta de neve por
                    # completo -- troca pela legenda-gradiente das DUAS, colada ABAIXO da caixa do
                    # modelo (mesma borda esquerda dela) em vez de ao lado (ver
                    # `_legenda_gradiente_chuva_neve`). LARGURA FIXA (pedido do usuario) -- nao usa
                    # mais `_x_data_dir` (borda direita da caixa da data): a legenda nao muda de
                    # tamanho mais se o titulo, a data ou o nome do modelo mudarem de texto.
                    _legenda_gradiente_chuva_neve(fig, ctx, x_ini=bb3.x0 - pad_x3,
                                                  y_topo=bb3.y0 - pad_y3, font=_font_twc)
                else:
                    # Legenda de faixas: comeca APOS a caixa do modelo (x_ini = a propria borda direita
                    # do PATCH dele, sem respiro -> encosta, como a caixa da data encosta na azul) e
                    # quebra na borda direita da caixa da data. Esse limite e CONSTANTE no clipe porque
                    # a caixa da data tem largura fixa (ditada pelo rotulo mais largo) -- sem isso, o
                    # limite cresceria junto com a data e uma caixinha da 2a linha "pularia" p/ a 1a no
                    # meio do video.
                    _legenda_faixas_twc(fig, ctx, cmap, x_ini=bb3.x1 + pad_x3,
                                        x_dir=_x_data_dir, y_topo=bb1.y0 - pad_y, font=_font_twc)
        fig.text(0.98, 0.028, ctx['credito'], color='#cfcfcf', fontsize=8.5,
                 ha='right', va='center', family=FONT_SANS, zorder=21)
        return
    # ── Caixa cinza (topo-esquerdo), enquadrada no canto e justa ao texto ──
    # Ancora o titulo no canto sup-esq e dimensiona a caixa pelo EXTENT real do texto
    # (margem minima), em vez de um tamanho fixo com sobra.
    # Quebra da caixa: se o rotulo_box ja traz '\n' explicito, RESPEITA (o autor decidiu as linhas,
    # ex.: "Meridional Wind &\nStreamfunction" p/ nao deixar o "&" sozinho); senao, quebra em width=15.
    _titulo_raw = _fmt_nivel_hpa(str(ctx['titulo_box']).upper())
    titulo = _titulo_raw if '\n' in _titulo_raw else textwrap.fill(_titulo_raw, width=15)
    x_anchor, y_anchor = 0.016, 0.978
    # Estilo do overlay PARAMETRIZAVEL via ctx (defaults = valores do s39 -> s38-s44 inalterados; o s45
    # sobrepoe): `overlay_box_cor` = cor de fundo da caixa do nome (default cinza); `overlay_info_cor` =
    # cor dos escritos livres (data, modelo, subtitulo, clim, credito) -- None mantem as cores originais
    # por elemento; `overlay_info_bold` = negrito nesses escritos.
    _ov_box_cor = ctx.get('overlay_box_cor') or '#9a9a9a'
    _ov_info_cor = ctx.get('overlay_info_cor')
    _ov_info_w = 'bold' if ctx.get('overlay_info_bold', False) else None
    t = fig.text(x_anchor, y_anchor, titulo, color='white', fontsize=11, ha='left', va='top',
                 weight='bold', family=ctx['font_legenda'], zorder=21)
    bb = t.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    pad_x, pad_y = 0.009, 0.008
    fig.add_artist(FancyBboxPatch(
        (bb.x0 - pad_x, bb.y0 - pad_y), bb.width + 2 * pad_x, bb.height + 2 * pad_y,
        boxstyle='round,pad=0,rounding_size=0.006', transform=fig.transFigure,
        facecolor=_ov_box_cor, alpha=1.0, edgecolor='none', zorder=20))
    # ── Data (dia, mes, ano em ingles) ABAIXO da caixa: branco, maiusculo, sem negrito ──
    info_x = bb.x0 - pad_x + 0.002
    date_y = bb.y0 - pad_y - 0.012
    fig.text(info_x, date_y, data_full.upper(), color=(_ov_info_cor or 'white'),
             fontsize=10, ha='left', va='top', weight=(_ov_info_w or 'normal'),
             family=ctx['font_legenda'], zorder=21)
    # ── Modelo + rodada (forecast) ou "REANALYSIS" (passado): ABAIXO da data ──
    fig.text(info_x, date_y - 0.026, ctx['rodada_label'], color=(_ov_info_cor or '#dcdcdc'),
             fontsize=9, ha='left', va='top', weight=(_ov_info_w or 'normal'),
             family=ctx['font_legenda'], zorder=21)

    # ── Subtitulo (canto superior DIREITO): subido p/ alinhar a quina, como o topo-esq ──
    fig.text(0.985, 0.978, ctx['subtitulo_dir'], color=(_ov_info_cor or '#e6e6e6'), fontsize=9.5,
             ha='right', va='top', weight=(_ov_info_w or 'normal'), family=ctx['font_legenda'], zorder=21)
    fig.text(0.985, 0.954, ctx['clim_ref'], color=(_ov_info_cor or '#b8b8b8'), fontsize=8.5,
             ha='right', va='top', weight=(_ov_info_w or 'normal'), family=ctx['font_legenda'], zorder=21)

    # ── Legenda: barra de gradiente FINA em caixa translucida arredondada (mais escura) ──
    lx0, ly0, lw, lh = 0.22, 0.072, 0.56, 0.074
    fig.add_artist(FancyBboxPatch(
        (lx0, ly0), lw, lh, boxstyle='round,pad=0,rounding_size=0.018',
        transform=fig.transFigure, facecolor='black', alpha=0.72,
        edgecolor='none', zorder=20))
    gl, gb, gw, gh = 0.245, 0.122, 0.51, 0.014
    gax = fig.add_axes([gl, gb, gw, gh])
    gax.set_zorder(21)
    gax.imshow(np.linspace(0.0, 1.0, 256).reshape(1, -1), aspect='auto', cmap=cmap,
               extent=[0, 1, 0, 1], origin='lower')
    gax.set_axis_off()
    if ctx.get('legenda_numerica'):
        # Rotulos NUMERICOS (ex.: tsm_anom) — valores de `legenda_num_step` em `step`, sem palavras.
        vmin, vmax = float(ctx['levels'][0]), float(ctx['levels'][-1])
        for t in _numeric_legend_ticks(vmin, vmax, ctx.get('legenda_num_step', 0.5)):
            xpos = gl + gw * (t - vmin) / (vmax - vmin)
            fig.text(xpos, gb - 0.010, f'{t:g}', color='white', fontsize=6,
                     ha='center', va='top', family=ctx['font_legenda'], zorder=22)
        if ctx.get('legenda_unidade'):
            fig.text(gl + gw / 2, gb - 0.030, ctx['legenda_unidade'], color='#dcdcdc',
                     fontsize=8, ha='center', va='top', family=ctx['font_legenda'], zorder=22)
    else:
        labels = ctx['legenda5_labels']
        seg = gw / max(len(labels), 1)
        for i, lab in enumerate(labels):
            fig.text(gl + seg * (i + 0.5), gb - 0.012, str(lab).upper(), color='white',
                     fontsize=7.5, ha='center', va='top', family=ctx['font_legenda'], zorder=22)
        # Unidade centralizada abaixo dos labels (ex.: 'm/s' para jet_stream)
        if ctx.get('legenda_unidade'):
            fig.text(gl + gw / 2, gb - 0.026, ctx['legenda_unidade'], color='#c0c0c0',
                     fontsize=7.5, ha='center', va='top', family=ctx['font_legenda'], zorder=22)

    # ── Rodape: apenas o credito (dir.). O modelo/rodada subiu p/ baixo da data. ──
    fig.text(0.98, 0.028, ctx['credito'], color=(_ov_info_cor or '#cfcfcf'), fontsize=8.5,
             ha='right', va='center', weight=(_ov_info_w or 'normal'), family=FONT_SANS, zorder=21)


def _jet_segments(lon: np.ndarray, lat: np.ndarray, field: np.ndarray, nivel: float,
                  min_pts: int = 12,
                  lat_band: tuple[float, float] | None = None,
                  hemisferios: tuple[str, ...] = ('N', 'S')) -> list[np.ndarray]:
    """Extrai as polilinhas (lon,lat) da isolinha ``field == nivel`` via contourpy (sem criar
    artista no eixo — so geometria). Cada segmento e orientado W->E e filtrado por numero minimo
    de pontos, por banda de |latitude| (opcional; a corrente de jato e extratropical) e por
    HEMISFERIO (`hemisferios`: 'N' e/ou 'S' — permite plotar o jato so no HN, so no HS, ou nos dois).

    Devolve lista de arrays float (N,2) em coordenadas geograficas (lon, lat). A coluna de
    longitude vem DESENROLADA (`np.unwrap`): a grade e ciclica (0..360, ponto extra em 360 do
    `add_cyclic_point`), entao um segmento que cruza o meridiano de Greenwich pode vir do
    contourpy como pontos consecutivos tipo 359.8 -> 0.3 (salto de quase 360°). Sem desenrolar
    aqui na ORIGEM, esse salto se propaga pra tudo que consome o segmento depois: o arco
    (`np.diff(lo)`/`np.cumsum`) fica errado dali em diante (desloca texto/setas), a tangente de
    `_offset_polyline` gira pro lado errado (quebra as faixas finas), e a deteccao de isolinha
    CIRCUMPOLAR (`lo.max()-lo.min() >= 350`) classifica errado um segmento comum que so cruza o
    meridiano (nao da volta no globo) como se fosse o anel inteiro. Cartopy aceita longitude fora
    de [0,360)/[-180,180) sem problema (reprojeta modularmente), entao desenrolar e seguro."""
    import contourpy

    gen = contourpy.contour_generator(x=lon, y=lat, z=np.asarray(field, dtype=float))
    segs: list[np.ndarray] = []
    for arr in gen.lines(float(nivel)):
        if arr is None or len(arr) < min_pts:
            continue
        a = np.asarray(arr, dtype=float)
        a[:, 0] = np.degrees(np.unwrap(np.deg2rad(a[:, 0])))
        _la = a[:, 1]
        dentro = np.ones(len(a), dtype=bool)
        if lat_band is not None:
            dentro &= (np.abs(_la) >= lat_band[0]) & (np.abs(_la) <= lat_band[1])
        # Hemisferio: mantem so os pontos do(s) hemisferio(s) habilitado(s) (HS: lat<0, HN: lat>0).
        if 'N' not in hemisferios:
            dentro &= (_la < 0.0)
        if 'S' not in hemisferios:
            dentro &= (_la > 0.0)
        if dentro.mean() < 0.5:  # descarta segmentos que quase nao caem na regiao permitida
            continue
        if a[-1, 0] < a[0, 0]:       # orienta W->E: se o fim esta a oeste do inicio, inverte
            a = a[::-1]
        # Descarta VORTICES fechados (baixa/alta de corte isolada, isolinha que fecha num loop
        # pequeno) -- a corrente de jato e uma ONDA LARGA, nao um vortice; diferente do anel
        # CIRCUMPOLAR legitimo (fecha dando a volta no globo INTEIRO, span de longitude ~360°,
        # tratado especialmente em `_jet_flow_sequence`/`_draw_text_on_path` via `closed=True`).
        _fechado = np.hypot(a[-1, 0] - a[0, 0], a[-1, 1] - a[0, 1]) < 1.0
        _span_lon = float(a[:, 0].max() - a[:, 0].min())
        if _fechado and _span_lon < 60.0:
            continue
        segs.append(a)
    return segs


def _offset_polyline(lo: np.ndarray, la: np.ndarray, d: float, closed: bool = False,
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Desloca a polilinha (lo, la) por ``d`` graus na direcao NORMAL (perpendicular ao fluxo),
    em espaco metrico local (lon ponderado por cos(lat), para o passo ser ~isometrico no globo).
    Serve para as faixas finas paralelas acima/abaixo da faixa central. d>0 = lado esquerdo do
    sentido W->E.

    `lo` e DESENROLADA (`np.unwrap`) antes do gradiente: sem isso, um segmento que cruza o
    meridiano de Greenwich (grade cíclica 0..360 -- pontos consecutivos tipo 359.8 -> 0.3) gera
    um salto espurio de quase 360° no `np.gradient`, girando a normal na direcao errada e
    quebrando as faixas finas paralelas exatamente ali (a faixa central nao quebra pois nao usa
    este offset). So a DIRECAO (tangente/normal) usa a versao desenrolada; a posicao de saida
    usa `lo` original, para nao alterar as coordenadas geograficas reais.

    `closed=True` (isolinha CIRCUMPOLAR, anel que fecha dando a volta no globo): `np.gradient`
    usa diferenca de UMA PONTA SO nos extremos do array (nao sabe que index 0 e index -1 sao
    VIZINHOS no anel) -> tangente errada bem onde o contourpy comecou/terminou o traco do anel
    (coincide com o meridiano de Greenwich, pois e ali que a grade ciclica tem sua borda) ->
    mesma quebra visual, so que so na ponta. Corrigido com diferenca CENTRADA PERIODICA nos
    extremos (o vizinho "antes do 1o ponto" e o ULTIMO ponto, e vice-versa)."""
    coslat = np.cos(np.deg2rad(la))
    coslat_safe = np.where(np.abs(coslat) < 1e-6, 1e-6, coslat)
    lo_unwrap = np.degrees(np.unwrap(np.deg2rad(lo)))
    if closed and len(lo) > 2:
        lo_ext = np.concatenate([[lo_unwrap[-1] - 360.0], lo_unwrap, [lo_unwrap[0] + 360.0]])
        la_ext = np.concatenate([la[-1:], la, la[:1]])
        tx = (lo_ext[2:] - lo_ext[:-2]) / 2.0 * coslat
        ty = (la_ext[2:] - la_ext[:-2]) / 2.0
    else:
        tx = np.gradient(lo_unwrap) * coslat  # tangente em espaco metrico (lon desenrolada)
        ty = np.gradient(la)
    n = np.hypot(tx, ty)
    n[n == 0] = 1.0
    tx, ty = tx / n, ty / n
    nx, ny = -ty, tx                     # normal = tangente girada +90
    return lo + nx * d / coslat_safe, la + ny * d


def _screen_tangent(ax, lo: np.ndarray, la: np.ndarray, s: np.ndarray, sm: float,
                    src, ds: float = 1.2) -> tuple[float, float, float, bool]:
    """Ponto (lon, lat) em arc-length ``sm`` e o angulo (graus, coords de TELA) da tangente ao
    fluxo ali. Diferenca CENTRADA numa janela +-``ds`` (arco) -> tangente SUAVE: setas e texto
    seguem o fluxo geral sem jitter e sem apontar 'pra tras' onde a isolinha faz microcurvas.
    Retorna ``ok=False`` se o ponto cai atras do horizonte (projecao nao-finita)."""
    L = float(s[-1])
    lon_m = float(np.interp(sm, s, lo))
    lat_m = float(np.interp(sm, s, la))
    sa, sb = max(0.0, sm - ds), min(L, sm + ds)
    proj = ax.projection
    pm = proj.transform_point(lon_m, lat_m, src)
    p1 = proj.transform_point(float(np.interp(sa, s, lo)), float(np.interp(sa, s, la)), src)
    p2 = proj.transform_point(float(np.interp(sb, s, lo)), float(np.interp(sb, s, la)), src)
    if not (np.isfinite(pm[0]) and np.isfinite(pm[1]) and np.isfinite(p1[0]) and np.isfinite(p1[1])
            and np.isfinite(p2[0]) and np.isfinite(p2[1])):
        return lon_m, lat_m, 0.0, False
    d1 = ax.transData.transform(p1)
    d2 = ax.transData.transform(p2)
    ang = float(np.degrees(np.arctan2(d2[1] - d1[1], d2[0] - d1[0])))
    return lon_m, lat_m, ang, True


_CHAR_W_CACHE: dict = {}

# Seta estilo "enviar/telegram" (ref. Entrada/seta.png): ponta a DIREITA (+x), duas asas atras e um
# ENTALHE concavo no meio da traseira (aponta p/ a ponta) — nao e um triangulo simples. Coords locais
# normalizadas ~[-0.5, 0.5]; o modo drape escala em graus, o modo tela usa como marker (pontos).
_ARROW_VERTS = np.array([
    (0.50, 0.00),    # ponta (direita)
    (-0.50, 0.45),   # asa superior (tras)
    (-0.22, 0.00),   # entalhe concavo (aponta p/ a ponta)
    (-0.50, -0.45),  # asa inferior (tras)
])


def _char_advances_pt(text: str, fontsize: float, weight: str = 'bold',
                      family: str | None = None) -> list[float]:
    """Largura de AVANCO (em pontos) de cada caractere de `text`, para dispor a palavra letra a
    letra ao longo de uma curva (ou medir largura real de texto p/ quebra de linha balanceada).
    Cacheado por (text, fontsize, weight, family)."""
    key = (text, round(fontsize, 2), weight, family)
    cached = _CHAR_W_CACHE.get(key)
    if cached is not None:
        return cached
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    fp = FontProperties(weight=weight, size=fontsize, family=family)
    advs: list[float] = []
    for ch in text:
        if ch == ' ':
            advs.append(fontsize * 0.42)
        else:
            w = float(TextPath((0, 0), ch, prop=fp).get_extents().width)
            advs.append(w + fontsize * 0.14)   # largura do glifo + tracking leve
    _CHAR_W_CACHE[key] = advs
    return advs


def _wrap_balanceado(texto: str, largura_chars: int, fontsize: float, family: str | None,
                     weight: str = 'bold') -> str:
    """Quebra `texto` em linhas BALANCEADAS: em vez de encher a 1a linha o quanto der
    (`textwrap.fill`, que deixa uma linha enorme + outra so com 1 palavra sobrando -> caixa mais
    LARGA do que precisava), distribui as palavras pra MINIMIZAR a largura da linha MAIS LARGA,
    usando a largura REAL renderizada (fonte/tamanho/peso) em vez de contagem de caracteres.

    O NUMERO de linhas alvo continua vindo do `textwrap.wrap` de sempre (mesma logica de
    `largura_chars`); so a DIVISAO das palavras dentro desse numero de linhas e otimizada (busca
    binaria na largura maxima permitida por linha — problema classico de "split array em K partes
    minimizando a maior soma")."""
    palavras = texto.split()
    if len(palavras) <= 1 or largura_chars <= 0:
        return texto
    n_linhas_alvo = len(textwrap.wrap(texto, width=largura_chars))
    if n_linhas_alvo <= 1:
        return texto
    larguras = [sum(_char_advances_pt(p, fontsize, weight, family)) for p in palavras]
    esp = sum(_char_advances_pt(' ', fontsize, weight, family))

    def _linhas_necessarias(max_w: float) -> int:
        n, atual = 1, 0.0
        for w in larguras:
            novo = w if atual == 0.0 else atual + esp + w
            if atual > 0.0 and novo > max_w:
                n += 1
                atual = w
            else:
                atual = novo
        return n

    lo, hi = max(larguras), sum(larguras) + esp * (len(palavras) - 1)
    while hi - lo > 0.05:
        mid = (lo + hi) / 2.0
        if _linhas_necessarias(mid) <= n_linhas_alvo:
            hi = mid
        else:
            lo = mid
    max_w = hi

    linhas: list[str] = []
    atual_p: list[str] = []
    atual_w = 0.0
    for p, w in zip(palavras, larguras):
        novo = w if not atual_p else atual_w + esp + w
        if atual_p and novo > max_w:
            linhas.append(' '.join(atual_p))
            atual_p, atual_w = [p], w
        else:
            atual_p.append(p)
            atual_w = novo
    if atual_p:
        linhas.append(' '.join(atual_p))
    return '\n'.join(linhas)


def _jet_flow_sequence(L: float, esp_txt: float, setas_entre: int, frac: float,
                       half: float, draw_word, draw_arrow, closed: bool = False) -> None:
    """Sequencia REGULAR tipo Entrada/JETSTREAM.png: uma palavra a cada `esp_txt` (arco) e
    `setas_entre` setas IGUALMENTE ESPACADAS no VAO entre as bordas de palavras consecutivas — as
    setas nunca invadem a palavra (`half` = meia-largura da palavra em arco, ja embute o texto real).
    `frac` in [0,1) = fase do LOOP: ao longo do GIF avanca exatamente UM espacamento de palavra ->
    emenda perfeita p/ QUALQUER texto e p/ os dois jatos juntos. Se a palavra nao couber (curva
    extrema), poe uma seta no lugar.

    `closed=True` (isolinha CIRCUMPOLAR, anel que fecha) -> a sequencia DA A VOLTA: posicoes modulo L,
    numero INTEIRO de palavras (L/step exato) -> sem 'costura' onde as pontas se encontram."""
    if esp_txt <= 0:
        return
    na = max(0, int(setas_entre))
    if closed:
        k = max(1, int(round(L / esp_txt)))            # nº INTEIRO de palavras ao redor do anel
        step = L / k
        base = (frac * step) % step                    # avanca exatamente 1 `step` no loop -> seamless
        for i in range(k):
            c = (base + i * step) % L
            if not draw_word(c):
                draw_arrow(c)
            g0, g1 = c + half, c + step - half
            if na > 0 and g1 > g0:
                for a in range(1, na + 1):
                    draw_arrow((g0 + (g1 - g0) * a / (na + 1)) % L)
        return
    # Segmento ABERTO: grade em [0, L] com corte nas pontas.
    shift = (frac * esp_txt) % esp_txt
    j0 = int(np.floor((0.0 - shift) / esp_txt)) - 1
    jmax = int(L / esp_txt) + 1
    for j in range(j0, jmax + 1):
        c = shift + j * esp_txt                        # centro da palavra
        if 0.0 <= c <= L and not draw_word(c):
            draw_arrow(c)
        g0, g1 = c + half, c + esp_txt - half          # vao entre esta palavra e a proxima
        if na > 0 and g1 > g0:
            for a in range(1, na + 1):
                pos = g0 + (g1 - g0) * a / (na + 1)
                if 0.0 <= pos <= L:
                    draw_arrow(pos)


def _seamless_loop_multiplier(vels: list[float], cap: int = 6) -> int:
    """Menor inteiro m>=1 tal que m*v seja INTEIRO para toda velocidade v>0 da lista (denominador
    limitado a `cap`). Usado no loop do GIF: com m, o loop passa a cobrir um numero INTEIRO de
    espacamentos de palavra para CADA jato (m*v por jato) -> emenda perfeita mesmo com velocidades
    FRACIONARIAS (ex.: 0.5 -> m=2, avanca 1 espacamento inteiro em 2x os frames = metade da
    velocidade, sem o 'salto' no fim do loop). Velocidades inteiras (1, 2, ...) dao m=1 (sem efeito)."""
    from fractions import Fraction
    from math import gcd
    m = 1
    for v in vels:
        if v <= 0:
            continue
        den = Fraction(v).limit_denominator(cap).denominator
        m = m * den // gcd(m, den)
    return max(1, m)


def _draw_text_on_path(ax, lo: np.ndarray, la: np.ndarray, s: np.ndarray, sm_center: float,
                       txt: str, cfg: dict, src, zorder: int, closed: bool = False) -> float | None:
    """Desenha `txt` LETRA A LETRA ao longo da curva (lo, la), centrado no arc-length `sm_center`.
    Cada glifo e posicionado e girado pela tangente local em TELA -> a palavra ACOMPANHA a curva
    (nao mais como bloco rigido). O passo entre letras (graus de arco) vem da largura do glifo (pt)
    convertida pela escala local px/grau da projecao no ponto da palavra.

    Devolve a META-EXTENSAO da palavra (graus de arco) para a supressao das setas, ou None se a
    palavra nao foi desenhada (atras do horizonte, ou numa curva fechada DEMAIS na tela — perto do
    limbo do globo — onde as letras se embaralhariam)."""
    L = float(s[-1])
    fontsize = float(cfg['texto_tam'])
    proj = ax.projection

    def _disp(sm: float):
        lon = float(np.interp(sm, s, lo))
        lat = float(np.interp(sm, s, la))
        p = proj.transform_point(lon, lat, src)
        if not (np.isfinite(p[0]) and np.isfinite(p[1])):
            return None
        return np.asarray(ax.transData.transform(p), dtype=float)

    # Escala local: px na tela por grau de arco, medida em torno do centro da palavra.
    _ds = 0.2
    pa, pb = _disp(sm_center - _ds), _disp(sm_center + _ds)
    if pa is None or pb is None:
        return None
    px_per_deg = float(np.hypot(*(pb - pa))) / (2.0 * _ds)
    if px_per_deg <= 1e-6:
        return None
    deg_per_pt = (ax.figure.dpi / 72.0) / px_per_deg    # 1 ponto de fonte -> graus de arco
    advs_deg = [a * deg_per_pt for a in _char_advances_pt(txt, fontsize, 'bold')]
    total = float(sum(advs_deg))
    half = total / 2.0

    # Direcao de leitura: se a tangente no centro aponta p/ a esquerda na tela (|ang|>90), inverte
    # a ordem das letras e gira 180 -> palavra legivel mesmo com o fluxo indo E->W na tela.
    _, _, ang_c, ok_c = _screen_tangent(ax, lo, la, s, sm_center, src)
    if not ok_c:
        return None
    # Pula a palavra se ela cai numa curva muito fechada NA TELA: se a tangente gira mais que
    # `texto_max_curva` graus entre o inicio e o fim da palavra, as letras se sobrepoem/embaralham
    # (tipico perto do limbo do globo, onde a projecao comprime o arco). Melhor um vao do que a
    # bagunca — as setas seguem preenchendo essa regiao.
    _, _, a_ini, oki = _screen_tangent(ax, lo, la, s, max(0.0, sm_center - half), src)
    _, _, a_fim, okf = _screen_tangent(ax, lo, la, s, min(L, sm_center + half), src)
    if oki and okf:
        dbend = abs((a_fim - a_ini + 180.0) % 360.0 - 180.0)
        if dbend > float(cfg.get('texto_max_curva', 110.0)):
            return None

    flip = abs(ang_c) > 90.0
    chars = list(txt)
    if flip:
        chars, advs_deg = chars[::-1], advs_deg[::-1]

    s0 = sm_center - half
    cum = 0.0
    for ch, adeg in zip(chars, advs_deg):
        s_ch = s0 + cum + adeg / 2.0
        cum += adeg
        if ch == ' ':
            continue
        if closed:
            s_ch %= L                          # da a volta na costura (anel circumpolar)
        elif s_ch < 0.0 or s_ch > L:
            continue
        lon_c, lat_c, ang, ok = _screen_tangent(ax, lo, la, s, s_ch, src)
        if not ok:
            continue
        r = ang + 180.0 if flip else ang
        ax.text(lon_c, lat_c, ch, transform=src, rotation=r, rotation_mode='anchor',
                ha='center', va='center', color=cfg['texto_cor'], zorder=zorder + 1,
                fontsize=fontsize, fontweight='bold', clip_on=True)
    return half


def _draw_jet_overlay(ax, lo: np.ndarray, la: np.ndarray, cfg: dict, frame_idx: int,
                      src, zorder: int) -> None:
    """Desenha, deslizando W->E (fase = frame_idx), a sequencia REGULAR 'JET STREAM' + setas sobre a
    faixa central (lo, la), numa UNICA grade de slots (ver `_jet_flow_sequence`) -> ordem e
    espacamento sempre iguais (padrao Entrada/JETSTREAM.png). Cada elemento e girado pela tangente
    local em TELA -> acompanha a curva do globo; pontos atras do horizonte sao pulados."""
    dlon = np.diff(lo) * np.cos(np.deg2rad(0.5 * (la[:-1] + la[1:])))
    seg_len = np.hypot(dlon, np.diff(la))
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    L = float(s[-1])
    if L <= 0:
        return
    vel = float(cfg['velocidade'])
    txt = str(cfg['texto'])
    setas_entre = int(cfg.get('setas_entre', 2))
    frac = frame_idx * vel     # [0,1) no loop do GIF (0 no PNG)
    closed = (float(lo.max()) - float(lo.min())) >= 350.0   # isolinha circumpolar (anel)

    # Meia-largura da palavra em ARCO (graus) p/ espacar as setas fora dela: largura em pontos
    # convertida pela escala px/grau medida no meio do arco (aprox., a escala varia no globo).
    adv_pt = float(sum(_char_advances_pt(txt, float(cfg['texto_tam']), 'bold')))
    proj = ax.projection

    def _px(sm):
        p = proj.transform_point(float(np.interp(sm, s, lo)), float(np.interp(sm, s, la)), src)
        return np.asarray(ax.transData.transform(p), float) if np.isfinite(p[0]) else None

    _pa, _pb = _px(L / 2 - 0.2), _px(L / 2 + 0.2)
    if _pa is not None and _pb is not None and np.hypot(*(_pb - _pa)) > 1e-6:
        half = 0.5 * adv_pt * (ax.figure.dpi / 72.0) / (np.hypot(*(_pb - _pa)) / 0.4)
    else:
        half = 2.0
    # Espacamento entre palavras AUTOMATICO (texto + vao das setas) -> nunca sobrepoe (qualquer string).
    passo = max(0.5, float(cfg.get('setas_passo', 6.0)))
    esp_txt = 2.0 * half + (setas_entre + 1) * passo

    from matplotlib.markers import MarkerStyle
    from matplotlib.path import Path as _MPath
    from matplotlib.transforms import Affine2D
    _apath = _MPath(np.vstack([_ARROW_VERTS, _ARROW_VERTS[:1]]), closed=True)

    def _arrow(pos: float) -> None:
        lon_m, lat_m, ang, ok = _screen_tangent(ax, lo, la, s, pos % L if closed else pos, src)
        if not ok:
            return
        ms = MarkerStyle(_apath, transform=Affine2D().rotate_deg(ang))   # seta "enviar" girada
        ax.plot(lon_m, lat_m, transform=src, marker=ms,
                markersize=float(cfg['arrow_tam']) * 1.6, color=cfg['arrow_cor'],
                markeredgecolor='none', linestyle='none', zorder=zorder, clip_on=True)

    def _word(pos: float) -> bool:
        _, _, _, ok = _screen_tangent(ax, lo, la, s, pos % L if closed else pos, src)
        if not ok:
            return True   # atras do horizonte: nao desenha nada (nem seta de fallback)
        return _draw_text_on_path(ax, lo, la, s, pos % L if closed else pos, txt, cfg, src, zorder,
                                  closed=closed) is not None

    _jet_flow_sequence(L, esp_txt, setas_entre, frac, half, _word, _arrow, closed=closed)


def _draw_jet_stream(ax, lon: np.ndarray, lat: np.ndarray, field: np.ndarray,
                     cfg: dict, frame_idx: int, src) -> None:
    """Desenha a corrente de jato (ref. Entrada/JETSTREAM.png): faixa central OPACA sobre a
    isolinha ``field == cfg['nivel']``, com faixas finas TRANSLUCIDAS paralelas acima/abaixo
    (estaticas) e, por cima, a palavra 'JET STREAM' + setas deslizando W->E. A POSICAO vem do dado
    (isolinha muda dia a dia); so o overlay (texto/setas) e animado. Curvatura do globo via
    ``transform=src`` (PlateCarree)."""
    segs = _jet_segments(lon, lat, field, cfg['nivel'], min_pts=int(cfg['min_pts']),
                         lat_band=cfg.get('lat_band'),
                         hemisferios=cfg.get('hemisferios', ('N', 'S')))
    if not segs:
        return
    lw = float(cfg['largura'])
    stripe_lw = float(cfg['stripe_largura'])
    n_stripe = int(cfg['stripe_n'])
    gap0 = float(cfg['stripe_gap0'])
    gap = float(cfg['stripe_gap'])
    z = 8
    for a in segs:
        lo, la = a[:, 0], a[:, 1]
        # Faixas finas translucidas, paralelas, acima e abaixo (estaticas).
        for k in range(1, n_stripe + 1):
            d = gap0 + (k - 1) * gap
            for sgn in (+1.0, -1.0):
                lo_s, la_s = _offset_polyline(lo, la, sgn * d)
                ax.plot(lo_s, la_s, transform=src, color=cfg['cor'], lw=stripe_lw,
                        alpha=float(cfg['stripe_alpha']), solid_capstyle='round', zorder=z)
        # Faixa central OPACA.
        ax.plot(lo, la, transform=src, color=cfg['cor'], lw=lw, alpha=float(cfg['alpha']),
                solid_capstyle='round', solid_joinstyle='round', zorder=z + 1)
        # Overlay animado (texto + setas) por cima.
        _draw_jet_overlay(ax, lo, la, cfg, frame_idx, src, zorder=z + 2)


# ---------------------------------------------------------------------------
# Corrente de jato DRAPEJADA na superficie (GLOBO_3D_JATO_DRAPE): em vez de desenhar o jato
# como adesivos no plano da TELA (tamanho fixo em pontos -> embaralha no limbo), o jato inteiro e
# renderizado num RASTER PLANO equirectangular (lon x lat) e "colado" na esfera via imshow — como
# o sombreado das variaveis. Assim ele fica RENTE A SUPERFICIE e ganha a perspectiva 3D correta
# (comprime graciosamente no limbo em vez de embaralhar). Sizes vem em GRAUS geograficos.
# ---------------------------------------------------------------------------
def _parse_jet_cut(raw) -> dict | None:
    """Interpreta GLOBO_3D_<JATO>_CUT = {lat, lon, fade_out} (tabela inline TOML -> dict). Faz o jato
    SUMIR esmaecendo a LESTE (jusante) do ponto {lat,lon}, ao longo de `fade_out` graus de arco.
    Retorna {'lat','lon','fade_out'} validado, ou None (ausente/invalido)."""
    if not raw or not isinstance(raw, dict):
        return None
    try:
        lat, lon = float(raw.get('lat')), float(raw.get('lon'))
    except (TypeError, ValueError):
        logger.warning('JATO CUT ignorado (lat/lon ausente ou invalido): {}', raw)
        return None
    return {'lat': lat, 'lon': lon, 'fade_out': max(0.0, float(raw.get('fade_out', 0.0) or 0.0))}


def _jet_cut_taper(lo: np.ndarray, la: np.ndarray, s: np.ndarray, cut: dict | None):
    """Devolve w(c)->[0,1] p/ o corte do jato: 1 a OESTE (montante) do ponto {lat,lon} do `cut`,
    esmaece 1->0 (cosseno) ao longo de `fade_out` graus de arco a LESTE (jusante) e 0 dali em diante.
    None se `cut` ausente. O jato vem orientado W->E (`_jet_segments`), entao arco crescente = leste."""
    if not cut:
        return None
    coslat = np.cos(np.deg2rad(la))
    dlon = (lo - cut['lon'] + 180.0) % 360.0 - 180.0   # menor dif. angular (grade ciclica/lon unwrap)
    d2 = (dlon * coslat) ** 2 + (la - cut['lat']) ** 2
    s_cut = float(s[int(np.argmin(d2))])
    fade = float(cut['fade_out'])

    def w(c: float) -> float:
        if c <= s_cut:
            return 1.0
        if fade <= 0.0 or c >= s_cut + fade:
            return 0.0
        return float(0.5 + 0.5 * np.cos(np.pi * (c - s_cut) / fade))
    return w


def _plot_line_tapered(ax, lo, la, s, color, lw, base_alpha, w_fn, zorder) -> None:
    """Desenha a polilinha (lo,la) como LineCollection com alpha POR SEGMENTO = base_alpha*w_fn(arco)
    -- o `ax.plot` so aceita alpha escalar. Usado no corte do jato (modo drape)."""
    from matplotlib.collections import LineCollection
    pts = np.column_stack([lo, la])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    smid = 0.5 * (s[:-1] + s[1:])
    cols = np.tile(np.array(to_rgba(color), dtype=float), (len(segs), 1))
    cols[:, 3] = base_alpha * np.array([w_fn(float(c)) for c in smid])
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw, zorder=zorder,
                                     capstyle='round', joinstyle='round'))


def _jato_flat_overlay(ax, lo: np.ndarray, la: np.ndarray, cfg: dict, frame_idx: int,
                       fs_pt: float, arrow_ms_pt: float, deg_per_pt: float, zorder: int) -> None:
    """Texto 'JET STREAM' (letra a letra) + setas num eixo PLANO equirectangular (data = lon, lat,
    escala uniforme). Sem projecao: a tangente e atan2(dlat, dlon) direto. O raster inteiro e
    depois drapejado na esfera (a curvatura 3D vem da reprojecao, nao daqui)."""
    dlon, dlat = np.diff(lo), np.diff(la)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(dlon, dlat))])   # arco EQUIRECTANGULAR (graus)
    L = float(s[-1])
    if L <= 0:
        return
    _cut_w = _jet_cut_taper(lo, la, s, cfg.get('cut'))   # corte {lat,lon,fade_out}: esmaece p/ leste
    vel = float(cfg['velocidade'])
    txt = str(cfg['texto'])
    setas_entre = int(cfg.get('setas_entre', 2))
    advs_deg = [wd * deg_per_pt for wd in _char_advances_pt(txt, fs_pt, 'bold')]
    half = float(sum(advs_deg)) / 2.0
    # Espacamento entre palavras AUTOMATICO: comprimento do texto (2*half) + vao p/ as setas
    # (setas_entre+1 passos). Assim, qualquer string (curta ou longa) nunca sobrepoe as setas.
    passo = max(0.5, float(cfg.get('setas_passo', 6.0)))
    esp_txt = 2.0 * half + (setas_entre + 1) * passo
    frac = frame_idx * vel     # [0,1) no loop do GIF (0 no PNG)
    # Isolinha CIRCUMPOLAR (fecha o anel): lon cobre ~360° -> arco periodico (as pontas coincidem).
    closed = (float(lo.max()) - float(lo.min())) >= 350.0

    # Tangente SUAVE: diferenca CENTRADA numa janela +-`wsm` (arco, graus). Sem isso a seta/letra
    # aponta pra baixo/pra tras nas microcurvas da isolinha. Janela ~ tamanho de uma letra.
    wsm = max(0.8, 1.4 * max(advs_deg) if advs_deg else 1.4)

    def _tan(sm: float):
        if closed:
            sm = sm % L                        # posicao da a volta na costura (lon 0 = 360)
        lon_m = float(np.interp(sm, s, lo)); lat_m = float(np.interp(sm, s, la))
        sa, sb = max(0.0, sm - wsm), min(L, sm + wsm)
        lon_a = float(np.interp(sa, s, lo)); lat_a = float(np.interp(sa, s, la))
        lon_b = float(np.interp(sb, s, lo)); lat_b = float(np.interp(sb, s, la))
        return lon_m, lat_m, float(np.degrees(np.arctan2(lat_b - lat_a, lon_b - lon_a)))

    # Sprite Entrada/seta.png (mesmo modelo do escoamento de baixos niveis) no lugar do poligono
    # _ARROW_VERTS -- o TAMANHO continua vindo de `arrow_tam_deg` (comprimento ponta-cauda), so o
    # MODELO (forma) muda; a largura acompanha a proporcao original do sprite.
    _seta_sprite = _carrega_seta_sprite(str(cfg['arrow_cor']))
    _seta_deg_por_px = float(cfg['arrow_tam_deg']) / _seta_sprite.size[0]

    def _arrow(pos: float) -> None:
        _wa = 1.0 if _cut_w is None else _cut_w(pos % L if closed else pos)
        if _wa <= 0.003:
            return   # a leste do corte: seta ja sumiu
        lon_m, lat_m, ang = _tan(pos)
        spr_rot = _seta_sprite.rotate(ang, expand=True, resample=_PIL_Image.BICUBIC)
        arr = np.asarray(spr_rot)
        if _wa < 0.999:
            arr = arr.copy()
            arr[..., 3] = (arr[..., 3].astype(np.float32) * _wa).astype(arr.dtype)
        w_deg = spr_rot.width * _seta_deg_por_px
        h_deg = spr_rot.height * _seta_deg_por_px
        ax.imshow(arr, origin='upper', zorder=zorder, interpolation='bilinear',
                  extent=(lon_m - w_deg / 2, lon_m + w_deg / 2, lat_m - h_deg / 2, lat_m + h_deg / 2))

    def _word(pos: float) -> bool:
        # 'JET STREAM' em tamanho CHEIO, NUNCA some (nem nas curvas fechadas) e SEMPRE SEGUE A DIRECAO
        # DO FLUXO -- igual as setas (tangente +s, SEM flip de tela). Desenha letra a letra, cada glifo
        # ancorado no EIXO do jato com a tangente LOCAL. Espacamento por CORDA (distancia RETA entre
        # letras), nao por arco: numa curva o arco entre letras > corda, entao espacar por arco=largura
        # as deixa mais proximas que a largura em linha reta e elas se empilham no lado concavo; por
        # corda avanca mais no arco e os glifos se espalham. Em curva MUITO fechada ainda pode restar um
        # leve embaralho (a palavra e longa) -- aceito a pedido do usuario, que prefere isso a
        # encolher/pular/inverter a palavra.
        chars = list(txt)
        n = len(chars)
        adv = list(advs_deg)
        fs = fs_pt

        def _pt(sm: float):
            sm = sm % L if closed else sm
            return float(np.interp(sm, s, lo)), float(np.interp(sm, s, la))

        def _adv_chord(s_from: float, chord: float, direction: float):
            """Arc-pos a uma distancia RETA (corda) `chord` de `s_from`, andando no sentido +1/-1."""
            x0, y0 = _pt(s_from)
            step = 0.25
            sm = s_from
            for _ in range(4000):
                sm2 = sm + direction * step
                if not closed and (sm2 < 0.0 or sm2 > L):
                    return None
                x, y = _pt(sm2)
                if np.hypot(x - x0, y - y0) >= chord:
                    return sm2
                sm = sm2
            return sm

        # Arc-pos de cada letra: a CENTRAL ancora em `pos`; as vizinhas espacadas por corda = soma das
        # meias-larguras dos glifos adjacentes (em reta ~= largura da letra). Segue o fluxo (tangente +s).
        ic = n // 2
        arc_pos: list[float | None] = [None] * n
        arc_pos[ic] = pos
        sm = pos
        for i in range(ic + 1, n):
            nxt = _adv_chord(sm, (adv[i - 1] + adv[i]) / 2.0, +1.0)
            if nxt is None:
                break
            arc_pos[i] = nxt
            sm = nxt
        sm = pos
        for i in range(ic - 1, -1, -1):
            prv = _adv_chord(sm, (adv[i] + adv[i + 1]) / 2.0, -1.0)
            if prv is None:
                break
            arc_pos[i] = prv
            sm = prv
        for ch, ap in zip(chars, arc_pos):
            if ch == ' ' or ap is None:
                continue
            _wt = 1.0 if _cut_w is None else _cut_w(ap % L if closed else ap)
            if _wt <= 0.003:
                continue   # letra a leste do corte: ja sumiu
            lon_c, lat_c, ang = _tan(ap)
            ax.text(lon_c, lat_c, ch, rotation=ang, rotation_mode='anchor', ha='center', va='center',
                    color=cfg['texto_cor'], zorder=zorder + 1, fontsize=fs, fontweight='bold',
                    clip_on=True, alpha=_wt)
        return True

    _jet_flow_sequence(L, esp_txt, setas_entre, frac, half, _word, _arrow, closed=closed)


def _draw_jet_flat(ax, lon: np.ndarray, lat: np.ndarray, field: np.ndarray, cfg: dict,
                   frame_idx: int, deg_per_pt: float) -> None:
    """Desenha o jato (faixa + faixas finas + texto + setas) num eixo PLANO equirectangular, com os
    tamanhos vindos de GRAUS geograficos (convertidos p/ pontos via deg_per_pt). Serve de fonte p/
    o raster que sera drapejado na esfera."""
    segs = _jet_segments(lon, lat, field, cfg['nivel'], min_pts=int(cfg['min_pts']),
                         lat_band=cfg.get('lat_band'),
                         hemisferios=cfg.get('hemisferios', ('N', 'S')))
    if not segs:
        return
    lw = float(cfg['largura_deg']) / deg_per_pt
    stripe_lw = float(cfg['stripe_largura_deg']) / deg_per_pt
    fs_pt = float(cfg['texto_tam_deg']) / deg_per_pt
    arrow_ms = float(cfg['arrow_tam_deg']) / deg_per_pt
    n_stripe = int(cfg['stripe_n'])
    gap0, gap = float(cfg['stripe_gap0']), float(cfg['stripe_gap'])
    for a in segs:
        lo, la = a[:, 0], a[:, 1]
        # Corte {lat,lon,fade_out}: esmaece faixa+faixas finas p/ leste (jusante) do ponto. Sem corte,
        # segue o caminho ax.plot original (alpha escalar, zero mudanca). `_s` = arco equirectangular.
        _s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(lo), np.diff(la)))])
        _cw = _jet_cut_taper(lo, la, _s, cfg.get('cut'))
        for k in range(1, n_stripe + 1):
            d = gap0 + (k - 1) * gap
            for sgn in (1.0, -1.0):
                los, las = _offset_polyline(lo, la, sgn * d)
                if _cw is None:
                    ax.plot(los, las, color=cfg['cor'], lw=stripe_lw, alpha=float(cfg['stripe_alpha']),
                            solid_capstyle='round', zorder=3)
                else:
                    _plot_line_tapered(ax, los, las, _s, cfg['cor'], stripe_lw,
                                       float(cfg['stripe_alpha']), _cw, zorder=3)
        if _cw is None:
            ax.plot(lo, la, color=cfg['cor'], lw=lw, alpha=float(cfg['alpha']),
                    solid_capstyle='round', solid_joinstyle='round', zorder=4)
        else:
            _plot_line_tapered(ax, lo, la, _s, cfg['cor'], lw, float(cfg['alpha']), _cw, zorder=4)
        _jato_flat_overlay(ax, lo, la, cfg, frame_idx, fs_pt, arrow_ms, deg_per_pt, zorder=5)


def _jato_raster(lon: np.ndarray, lat: np.ndarray, fields_by_kind: dict, jatos: list,
                 frame_idx: int, px: int, pad: float = 6.0) -> tuple[np.ndarray | None, list | None]:
    """Renderiza TODOS os jatos (lista de cfgs) num RASTER PLANO equirectangular (lon x lat) e devolve
    (RGBA uint8, extent [lon0,lon1,lat0,lat1]). O extent em latitude cobre a UNIAO das bandas de
    todos os jatos, p/ maximizar a resolucao. Retorna (None, None) se nenhum jato tem isolinha.
    Depois e drapejado na esfera via imshow (uma unica reprojecao para todos os jatos).

    `fields_by_kind`: {'hgt250': campo|None, 'hgt500': campo|None} -- cada `cfg` usa o campo do
    SEU proprio `cfg['kind']` (GLOBO_3D_JET_STREAM_NIVEL/SUBTROPICAL_JET_NIVEL decidem a escala,
    ver `_jato_kind_from_nivel`), pulando o jato se o campo daquele kind nao estiver disponivel."""
    lat_all = []
    for cfg in jatos:
        field = fields_by_kind.get(cfg.get('kind', 'hgt250'))
        if field is None:
            continue
        segs = _jet_segments(lon, lat, field, cfg['nivel'], min_pts=int(cfg['min_pts']),
                             lat_band=cfg.get('lat_band'),
                         hemisferios=cfg.get('hemisferios', ('N', 'S')))
        if segs:
            lat_all.append(np.concatenate([a[:, 1] for a in segs]))
    if not lat_all:
        return None, None
    lat_all = np.concatenate(lat_all)
    lat0 = max(float(lat.min()), float(lat_all.min()) - pad)
    lat1 = min(float(lat.max()), float(lat_all.max()) + pad)
    lon0, lon1 = float(lon.min()), float(lon.max())
    w_deg, h_deg = lon1 - lon0, max(1.0, lat1 - lat0)
    if w_deg <= 0:
        return None, None
    fig_w_in = 8.0
    dpi = max(50.0, px / fig_w_in)
    fig = plt.figure(figsize=(fig_w_in, fig_w_in * h_deg / w_deg), dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_facecolor('none')
    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    deg_per_pt = w_deg / 576.0     # 1 ponto de fonte -> graus (fig 8in cobre `w_deg`; independe de px)
    for cfg in jatos:
        field = fields_by_kind.get(cfg.get('kind', 'hgt250'))
        if field is None:
            continue
        _draw_jet_flat(ax, lon, lat, field, cfg, frame_idx, deg_per_pt)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba, [lon0, lon1, lat0, lat1]


def _draw_jet_layer(ax, ctx: dict, f: int, hgt_jato_by_kind: dict, lon_cyc, lat, data_transform) -> None:
    """Desenha SO a(s) corrente(s) de jato do frame `f` no eixo `ax` (drape ou adesivos de tela).
    Fatorado de `_build_frame` para ser reusado no render isolado do jato (cache de fundo). O jato
    e SEMPRE desenhado por frame (fase = `f`) -> a ANIMACAO do jato nao e cacheada nem alterada.

    `hgt_jato_by_kind`: {'hgt250': campo|None, 'hgt500': campo|None} -- ver `_jato_raster`."""
    if not (ctx.get('jatos') and hgt_jato_by_kind):
        return
    # Fade-in sincronizado com o inicio da cauda (teste s42, GLOBO_3D_FADE_CAUDA): invisivel antes
    # do campo congelar, esmaece durante GLOBO_3D_FADE_CAUDA_DUR_SEG segundos.
    _fade = 1.0
    if ctx.get('fade_cauda_on'):
        _total = max(int(ctx.get('total_frames', 1)) - 1, 1)
        _prog = f / _total
        _fi = float(ctx.get('fade_cauda_inicio', 0.0))
        _fd = max(float(ctx.get('fade_cauda_dur', 0.0)), 1e-6)
        _fade = float(np.clip((_prog - _fi) / _fd, 0.0, 1.0))
        if _fade <= 0.003:
            return   # ainda invisivel -> nem renderiza (economiza o raster/regrid do drape)
    if ctx.get('jato_drape'):
        _rgba, _ext = _jato_raster(lon_cyc, lat, hgt_jato_by_kind, ctx['jatos'], f,
                                   int(ctx.get('jato_drape_px', 5000)),
                                   float(ctx.get('jato_drape_pad', 6.0)))
        if _rgba is not None:
            if _fade < 1.0:
                _rgba = _rgba.copy()
                _rgba[..., 3] = (_rgba[..., 3].astype(np.float32) * _fade).astype(_rgba.dtype)
            ax.imshow(_rgba, origin='upper', transform=data_transform, extent=_ext, zorder=9,
                      interpolation='bilinear',
                      regrid_shape=int(ctx.get('jato_drape_regrid', 2048)))
    else:
        # Modo tela: fade so binario (invisivel/visivel) -- os desenhos de faixa/seta/texto nao
        # tem canal de alpha individual pra escalar suavemente feito o raster do drape.
        for _jc in ctx['jatos']:
            _field = hgt_jato_by_kind.get(_jc.get('kind', 'hgt250'))
            if _field is None:
                continue
            _draw_jet_stream(ax, lon_cyc, lat, _field, _jc, f, data_transform)


# ---------------------------------------------------------------------------
# Escoamento de baixos niveis: setas animadas (Entrada/seta.png) ao longo de uma
# trajetoria lat/lon desenhada a mao (origem -> pontos_meio -> destino) -- anotacao
# ilustrativa, NAO derivada de dado real de vento. Reusa o mecanismo de "drapear na
# esfera" do jato (raster plano + imshow com transform), so que aqui o sprite e uma
# imagem (nao um poligono) e a posicao vem de uma curva manual (nao de uma isolinha).
# ---------------------------------------------------------------------------
def _interp_curva_lonlat(
    origem: tuple[float, float], pontos_meio: list, destino: tuple[float, float], n: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpola uma curva suave passando por origem -> pontos_meio -> destino (cada ponto =
    (lat, lon)): spline cubica parametrizada pelo indice do ponto de controle (>=3 pontos) ou
    reta simples (so origem/destino). Retorna (lon_curva, lat_curva, s_curva) com `n` amostras
    densas; `s_curva` = comprimento de arco acumulado (graus), usado p/ animar as setas."""
    pontos = [tuple(origem)] + [tuple(p) for p in pontos_meio] + [tuple(destino)]
    lat_ctrl = np.array([p[0] for p in pontos], dtype=np.float64)
    lon_ctrl = np.array([p[1] for p in pontos], dtype=np.float64)
    t_ctrl = np.arange(len(pontos), dtype=np.float64)
    t_dense = np.linspace(0.0, float(len(pontos) - 1), n)
    if len(pontos) >= 3:
        lat_c = _CubicSpline(t_ctrl, lat_ctrl)(t_dense)
        lon_c = _CubicSpline(t_ctrl, lon_ctrl)(t_dense)
    else:
        lat_c = np.interp(t_dense, t_ctrl, lat_ctrl)
        lon_c = np.interp(t_dense, t_ctrl, lon_ctrl)
    ds = np.hypot(np.diff(lon_c), np.diff(lat_c))
    s_c = np.concatenate([[0.0], np.cumsum(ds)])
    return lon_c.astype(np.float64), lat_c.astype(np.float64), s_c.astype(np.float64)


_SETA_SPRITE_CACHE: dict[str, '_PIL_Image.Image'] = {}


def _carrega_seta_sprite(cor: str) -> '_PIL_Image.Image':
    """Carrega Entrada/seta.png (triangulo branco solido, SEM canal alpha) e recolore pra `cor`:
    usa o BRILHO (grayscale) como canal ALPHA (branco=opaco, preto=transparente -- preserva o
    anti-aliasing das bordas do sprite) e pinta o RGB inteiro com `cor`. Cacheado por cor (nao
    reprocessa a imagem a cada frame/seta)."""
    if cor in _SETA_SPRITE_CACHE:
        return _SETA_SPRITE_CACHE[cor]
    path = Path(str(settings.get('DIR_INPUT', 'Entrada'))) / 'seta.png'
    cinza = _PIL_Image.open(path).convert('L')
    bbox = cinza.getbbox()
    if bbox is not None:
        cinza = cinza.crop(bbox)
    alpha = np.array(cinza, dtype=np.uint8)
    r, g, b = (int(round(c * 255)) for c in to_rgba(cor)[:3])
    rgba = np.zeros((*alpha.shape, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3] = r, g, b, alpha
    img = _PIL_Image.fromarray(rgba, mode='RGBA')
    _SETA_SPRITE_CACHE[cor] = img
    return img


def _build_escoamentos_cfg() -> list[dict]:
    """Le GLOBO_3D_ESCOAMENTO_BAIXOS_NIVEIS (lista de trajetorias manuais lat/lon) e pre-computa,
    por escoamento, a curva densa (spline) e o sprite ja recolorido -- uma vez so, nao por frame."""
    itens = list(settings.get('GLOBO_3D_ESCOAMENTO_BAIXOS_NIVEIS', []) or [])
    pad_default = float(settings.get('GLOBO_3D_ESCOAMENTO_DRAPE_PAD', 4.0))
    out = []
    for it in itens:
        origem, destino = it.get('origem'), it.get('destino')
        if not origem or not destino:
            logger.warning('Escoamento de baixos niveis sem origem/destino — ignorado: {}', it)
            continue
        lon_c, lat_c, s_c = _interp_curva_lonlat(origem, list(it.get('pontos_meio', []) or []), destino)
        out.append({
            'lon_curva': lon_c, 'lat_curva': lat_c, 's_curva': s_c,
            'cor': str(it.get('cor', 'white')),
            'sprite': _carrega_seta_sprite(str(it.get('cor', 'white'))),
            'velocidade': float(it.get('velocidade', 1.0)),
            'espacamento_deg': float(it.get('espacamento_deg', 6.0)),
            'tamanho_deg': float(it.get('tamanho_deg', 2.5)),
            'largura_relativa': float(it.get('largura_relativa', 1.0)),
            'alpha': float(it.get('alpha', 1.0)),
            'fade_deg': float(it.get('fade_deg', 3.0)),
            'pad': float(it.get('pad', pad_default)),
        })
    if out:
        logger.info('{} escoamento(s) de baixos niveis configurado(s)', len(out))
    return out


def _fade_arco(c: float, comprimento: float, fade_deg: float) -> float:
    """Peso 0..1 (cosseno, sem corte reto) que esmaece a seta perto da ORIGEM (c perto de 0) e do
    DESTINO (c perto de `comprimento`) do escoamento -- evita a seta aparecer/sumir seca nas pontas
    do traçado. `fade_deg` = distancia em arco (graus) onde ocorre a transicao; 0 = sem esmaecer."""
    if fade_deg <= 0:
        return 1.0
    w = 1.0
    if c < fade_deg:
        w = min(w, 0.5 - 0.5 * np.cos(np.pi * c / fade_deg))
    if c > comprimento - fade_deg:
        w = min(w, 0.5 - 0.5 * np.cos(np.pi * (comprimento - c) / fade_deg))
    return float(np.clip(w, 0.0, 1.0))


def _escoamento_raster(
    escoamentos: list, frame_idx: int, px: int, total_frames: int = 1,
) -> tuple[np.ndarray | None, list | None]:
    """Renderiza as setas de TODOS os escoamentos num raster plano equirretangular RGBA e devolve
    (rgba, extent [lon0,lon1,lat0,lat1]) -- mesmo contrato de `_jato_raster`, so que aqui a
    posicao das setas vem de uma curva manual (nao de uma isolinha calculada do dado) e o sprite
    e uma imagem (`Entrada/seta.png`), nao um poligono vetorial. `total_frames` normaliza a fase
    do fluxo (`velocidade` vira fracao POR LOOP, nao por frame -- do contrario, com `frame_idx *
    velocidade` cru, o deslocamento cai sempre em multiplos EXATOS do espacamento e a animacao
    fica parada; mesma pegadinha que o jato evita multiplicando por `_phase_unit = 1/total_frames`
    antes de guardar `cfg['velocidade']`)."""
    if not escoamentos:
        return None, None
    pad = max(e['pad'] for e in escoamentos)
    lon0 = min(float(e['lon_curva'].min()) for e in escoamentos) - pad
    lon1 = max(float(e['lon_curva'].max()) for e in escoamentos) + pad
    lat0 = min(float(e['lat_curva'].min()) for e in escoamentos) - pad
    lat1 = max(float(e['lat_curva'].max()) for e in escoamentos) + pad
    w_deg, h_deg = lon1 - lon0, max(1.0, lat1 - lat0)
    if w_deg <= 0:
        return None, None
    w_px = max(64, int(px))
    h_px = max(64, int(round(w_px * h_deg / w_deg)))
    # Teto na MAIOR dimensao: uma trajetoria alta e estreita (ex.: desce ~30° de lat em ~10° de lon)
    # fazia h_px explodir (h_px = w_px * h_deg/w_deg) -> canvas de dezenas de MB RECRIADO a cada frame
    # em cada worker (transiente que ajudava a estourar a RAM). Reduz AMBAS as dimensoes na mesma
    # proporcao (pixels QUADRADOS preservados -> setas nao distorcem), so cai a resolucao do raster --
    # que e reamostrado no imshow/drape de qualquer jeito, e o sprite da seta e macio (tolera bem).
    _max_px = max(256, int(settings.get('GLOBO_3D_ESCOAMENTO_DRAPE_MAX_PX', 3000)))
    _maior = max(w_px, h_px)
    if _maior > _max_px:
        _s = _max_px / _maior
        w_px = max(64, int(round(w_px * _s)))
        h_px = max(64, int(round(h_px * _s)))
    # Fundo do canvas: NASCE na cor das setas (alpha 0), nao em preto. Motivo: o imshow reamostra
    # com bilinear no drape e mistura RGB e alpha de forma independente -> na borda das setas, um
    # pixel opaco vizinho de um transparente-PRETO virava cinza escuro ("contorno fantasma", mesmo
    # bug dos icones GIF). Ficava sub-pixel em regrid alto; ao baixar a resolucao (Parte A) apareceu.
    # Com o fundo JA na cor da seta, a interpolacao mistura cor-com-cor -> so o alpha esmaece, sem
    # halo, e SEM custo (evita o distance_transform_edt do _bleed_transparent_rgb). Cores diferentes
    # entre escoamentos: cai no bleed generico no fim (raro; canvas unico so aceita uma cor de fundo).
    _cores = {e.get('cor', 'white') for e in escoamentos}
    if len(_cores) == 1:
        _r, _g, _b = (int(round(c * 255)) for c in to_rgba(next(iter(_cores)))[:3])
        _bg = (_r, _g, _b, 0)
    else:
        _bg = (0, 0, 0, 0)
    canvas = _PIL_Image.new('RGBA', (w_px, h_px), _bg)

    def _lonlat_to_px(lo: float, la: float) -> tuple[float, float]:
        return (lo - lon0) / w_deg * w_px, (lat1 - la) / h_deg * h_px

    for esc in escoamentos:
        lon_c, lat_c, s_c = esc['lon_curva'], esc['lat_curva'], esc['s_curva']
        comprimento = float(s_c[-1])
        esp = float(esc['espacamento_deg'])
        if comprimento <= 0 or esp <= 0:
            continue
        frac = frame_idx * float(esc['velocidade']) / max(total_frames, 1)
        # Comprimento (ao longo do voo, eixo X original do sprite) escala por tamanho_deg; a
        # largura (perpendicular) preserva a proporcao original da imagem e ainda pode ser
        # afinada/engordada com `largura_relativa` (1.0 = proporcao original do seta.png).
        orig_w, orig_h = esc['sprite'].size
        comp_px = max(4, int(round(esc['tamanho_deg'] / w_deg * w_px)))
        larg_px = max(2, int(round(comp_px * (orig_h / orig_w) * esc['largura_relativa'])))
        sprite = esc['sprite'].resize((comp_px, larg_px), _PIL_Image.BICUBIC)
        if esc['alpha'] < 1.0:
            sprite = sprite.copy()
            a = np.array(sprite.getchannel('A'), dtype=np.float32) * float(esc['alpha'])
            sprite.putalpha(_PIL_Image.fromarray(a.astype(np.uint8)))

        _fade_deg = float(esc['fade_deg'])

        def _draw_arrow(c: float, _lon_c=lon_c, _lat_c=lat_c, _s_c=s_c, _L=comprimento, _spr=sprite) -> bool:
            c = min(max(c, 0.0), _L)
            i = int(np.clip(np.searchsorted(_s_c, c), 1, len(_s_c) - 1))
            t = (c - _s_c[i - 1]) / max(_s_c[i] - _s_c[i - 1], 1e-9)
            lo = _lon_c[i - 1] + t * (_lon_c[i] - _lon_c[i - 1])
            la = _lat_c[i - 1] + t * (_lat_c[i] - _lat_c[i - 1])
            ang = float(np.degrees(np.arctan2(_lat_c[i] - _lat_c[i - 1], _lon_c[i] - _lon_c[i - 1])))
            spr_rot = _spr.rotate(ang, expand=True, resample=_PIL_Image.BICUBIC)
            _w = _fade_arco(c, _L, _fade_deg)
            if _w < 0.999:
                if _w <= 0.001:
                    return True
                spr_rot = spr_rot.copy()
                _a = np.array(spr_rot.getchannel('A'), dtype=np.float32) * _w
                spr_rot.putalpha(_PIL_Image.fromarray(_a.astype(np.uint8)))
            x, y = _lonlat_to_px(lo, la)
            canvas.alpha_composite(spr_rot, dest=(int(round(x - spr_rot.width / 2)),
                                                  int(round(y - spr_rot.height / 2))))
            return True

        _jet_flow_sequence(comprimento, esp, 0, frac, 0.0, lambda _c: False, _draw_arrow, closed=False)

    _out = np.asarray(canvas)
    # Cor unica: o fundo ja nasceu na cor da seta -> sem halo, sem custo. So no caso raro de cores
    # DIFERENTES (canvas unico, um fundo so) sobra halo nas setas da cor que nao e a do fundo ->
    # ai sim aplica o bleed generico (nearest-neighbor) para acertar todas.
    if len(_cores) > 1:
        _out = _bleed_transparent_rgb(_out)
    return _out, [lon0, lon1, lat0, lat1]


def _draw_escoamento_layer(ax, ctx: dict, f: int, data_transform) -> None:
    """Desenha as setas do 'escoamento de baixos niveis' (`ctx['escoamentos']`) do frame `f`.
    Mesmo mecanismo de raster+imshow do jato (drape), sem banda/faixa -- so as setas fluindo."""
    escoamentos = ctx.get('escoamentos')
    if not escoamentos:
        return
    # zorder: o escoamento e de BAIXOS niveis -> fica ABAIXO da corrente de jato (alto nivel) e dos
    # icones de pressao. Espelha a logica de _draw_icones_pressao (jato usa 9 no drape / 7 na tela):
    # coloca o escoamento 3 niveis abaixo do jato, com piso 6 p/ nunca cair abaixo da costa/fronteiras
    # (zorder 5). Sem jato, mantem o valor alto de sempre (9, abaixo dos icones 10/11). Antes ficava
    # em 9 fixo = MESMO zorder do jato drape -> empate resolvido pela ordem de desenho (escoamento por
    # ultimo) e as setas cobriam o jato.
    if ctx.get('jatos') and (ctx.get('hgt_z250_abs_cyc') is not None
                             or ctx.get('hgt_z500_abs_cyc') is not None):
        _jato_z = 9 if ctx.get('jato_drape') else 7
        _z = max(6, _jato_z - 3)
    else:
        _z = 9
    _rgba, _ext = _escoamento_raster(escoamentos, f, int(ctx.get('escoamento_drape_px', 3000)),
                                     int(ctx.get('total_frames', 1)))
    if _rgba is not None:
        ax.imshow(_rgba, origin='upper', transform=data_transform, extent=_ext, zorder=_z,
                 interpolation='bilinear', regrid_shape=int(ctx.get('escoamento_drape_regrid', 2048)))


def _draw_caixa_livre(ax, ctx: dict, f: int, data_transform) -> None:
    """Desenha TODAS as caixas de texto livres configuradas (`ctx['caixas_livres']`, lista --
    pode ter 0, 1 ou varias). Fatorada por caixa em `_draw_uma_caixa_livre`."""
    for _cxl in ctx.get('caixas_livres') or []:
        _draw_uma_caixa_livre(ax, ctx, f, data_transform, _cxl)


def _draw_uma_caixa_livre(ax, ctx: dict, f: int, data_transform, _cxl: dict) -> None:
    """Caixa de texto LIVRE ancorada em lat/lon (segue a rotacao do globo). Cantos QUADRADOS
    (boxstyle='square'), contorno opcional. Fatorada p/ ser chamada tanto no render completo
    (`_build_frame`) quanto no overlay do fundo cacheado (`_render_overlay_rgba`) -- antes so
    existia no primeiro, entao a caixa sumia (nunca era desenhada) nos frames de fundo congelado
    (cauda do MP4 + todo o GIF) sempre que GLOBO_3D_BG_CACHE estava ligado.

    Sem `caixa_fixa`: fade-in normal um pouco antes do fim do clipe (`inicio_frac`..
    `inicio_frac+fade_frac`). Com `caixa_fixa` (s42): a caixa fica FIXA (sempre em `alpha_max`,
    sem rampa) em QUALQUER saida (MP4, GIF, PNG), INDEPENDENTE de `GLOBO_3D_FADE_CAUDA` -- ela
    nao faz parte da animacao da corrente de jato/icones de pressao (que podem ou nao esmaecer
    via FADE_CAUDA), entao nunca deve esmaecer junto com eles.

    Override POR CAIXA (`_cxl['fixa']`, de GLOBO_3D_CAIXA_LIVRE_FIXA / campo `fixa` nas extras):
    permite uma caixa especifica ESMAECER mesmo no s42 (ex.: aparecer no inicio da cauda). Como os
    icones, uma caixa que esmaece na linha do tempo do MP4 sai em alpha CHEIO no PNG/GIF
    (`saida_estatica`) -- senao o frame 0 do PNG (fracao 0) a deixaria invisivel."""
    if not _cxl.get('texto'):
        return
    _fixa = bool(_cxl.get('fixa', ctx.get('caixa_fixa')))
    if _fixa or ctx.get('saida_estatica'):
        _a = _cxl['alpha_max']
    else:
        _total = max(len(ctx['lons']) - 1, 1)
        _prog = f / _total                       # 0..1 ao longo do clipe
        _a = float(np.clip((_prog - _cxl['inicio_frac']) / max(_cxl['fade_frac'], 1e-6),
                           0.0, 1.0)) * _cxl['alpha_max']
    if _a <= 0.003:
        return
    _lw = _cxl['contorno_lw']
    _txt = _wrap_balanceado(_cxl['texto'], _cxl['largura'], _cxl['fontsize'],
                            ctx.get('font_twc') or ctx.get('font_legenda'))
    _bbox = dict(boxstyle=f"square,pad={_cxl.get('pad', 0.30)}",
                 facecolor=to_rgba(_cxl['cor_box'], _a),
                 edgecolor=(to_rgba(_cxl['contorno_cor'], _a) if _lw > 0 else 'none'),
                 linewidth=_lw)
    # ha/va='center' ancora o ponto no CENTRO; ma='center' centraliza as multiplas linhas
    # (senao ficam a esquerda, deixando "sobra" a direita). O bbox envolve o texto + pad
    # uniforme -> sem sobras para cima/baixo/lados.
    # MESMA fonte da caixa azul "The Weather Channel" (ctx['font_twc']).
    _txt_artist = ax.text(_cxl['lon'], _cxl['lat'], _txt, transform=data_transform,
                          color=to_rgba(_cxl['cor_texto'], _a), fontsize=_cxl['fontsize'],
                          fontweight='bold', ha='center', va='center', ma='center',
                          linespacing=1.2,
                          family=ctx.get('font_twc') or ctx['font_legenda'], zorder=12,
                          bbox=_bbox, clip_on=False)
    # Sombra preta LEVEMENTE esfumacada ao redor da caixa: 3 contornos pretos concentricos,
    # do mais largo/transparente ao mais estreito/opaco, desenhados ATRAS (Normal por ultimo
    # redesenha o preenchimento opaco + a borda branca por cima) -> halo suave so na borda.
    if _cxl.get('sombra', True):
        _bp = _txt_artist.get_bbox_patch()
        if _bp is not None:
            _bp.set_path_effects([
                path_effects.Stroke(linewidth=7.0, foreground='black', alpha=0.10 * _a),
                path_effects.Stroke(linewidth=5.0, foreground='black', alpha=0.14 * _a),
                path_effects.Stroke(linewidth=3.0, foreground='black', alpha=0.22 * _a),
                path_effects.Normal(),
            ])


def _render_overlay_rgba(ctx: dict, f: int) -> np.ndarray | None:
    """Renderiza jato + icones de pressao do frame `f` numa UNICA figura TRANSPARENTE, com a
    MESMA projecao/enquadramento do globo (camera fixa nos frames de fundo congelado). Devolve
    RGBA float32 (H,W,4) para compor sobre o fundo cacheado.

    Antes eram 2 figuras Agg separadas (`_render_jet_only_rgba` + `_render_icones_only_rgba`):
    cada uma aloca seu proprio canvas full-res (figsize x dpi) e faz sua propria copia float32
    do buffer. Nos frames de fundo congelado — que dominam o clipe quando ha cauda longa do MP4
    ou GIF — isso dobrava o nº de renders Agg por frame em paralelo (8 workers), quase estourando
    a RAM do WSL ao ligar os icones de pressao junto com o jato. Unificar em 1 figura corta esse
    custo pela metade sem mudar nenhum pixel (mesmo `_draw_jet_layer`/`_draw_icones_pressao`,
    mesma ordem de zorder)."""
    tem_jato = bool(ctx.get('jatos') and ctx.get('hgt_jato_frozen'))
    tem_icones = bool(ctx.get('icones_pressao'))
    tem_caixa = any(_cx.get('texto') for _cx in (ctx.get('caixas_livres') or []))
    tem_escoamento = bool(ctx.get('escoamentos'))
    tem_objeto = bool(ctx.get('objeto_texto'))
    if not tem_jato and not tem_icones and not tem_caixa and not tem_objeto:
        return None
    cam_lon, cam_lat = float(ctx['lons'][f]), float(ctx['lats'][f])
    proj = _make_projection(cam_lon, cam_lat, mode=ctx.get('proj_mode', 'nearside'),
                            sat_height=ctx.get('sat_height', _SAT_HEIGHT_GEO))
    guillaume = ctx.get('estilo') == 'guillaume'
    rect = ctx.get('globe_rect') or ([0.07, 0.06, 0.86, 0.86] if guillaume
                                      else [0.01, 0.10, 0.98, 0.83])
    data_transform = ccrs.PlateCarree()
    fig = plt.figure(figsize=ctx.get('figsize', (8, 8)), dpi=ctx['dpi'])
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes(rect, projection=proj)
    ax.patch.set_alpha(0.0)          # fundo do eixo transparente -> so jato/icones/caixa contribuem
    ax.set_global()
    _aplicar_janela_inclinada(ax, proj, ctx.get('inc'))   # so s44; no-op p/ os demais
    ax.spines['geo'].set_visible(False)  # sem anel/borda -> nada alem das camadas na composicao
    if tem_jato:
        _draw_jet_layer(ax, ctx, f, ctx['hgt_jato_frozen'], ctx['lon_cyc'], ctx['lat'],
                        data_transform)
    if tem_icones:
        _draw_icones_pressao(ax, ctx, f, data_transform, proj)
    if tem_escoamento:
        _draw_escoamento_layer(ax, ctx, f, data_transform)
    if tem_objeto:
        # No overlay (composto POR CIMA do fundo) -> permite o objeto ficar acima do jato ate nos
        # frames congelados (cauda/GIF). O zorder do dict decide a ordem RELATIVA aqui (ex.: acima
        # do jato/caixa). No fundo cacheado ele NAO e desenhado (ver _build_frame, `not skip_overlay`).
        _draw_objeto_texto(ax, ctx, data_transform, proj)
    if tem_caixa:
        _draw_caixa_livre(ax, ctx, f, data_transform)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.float32).copy()
    plt.close(fig)
    return rgba


# ── Ícones de pressão animados ────────────────────────────────────────────────

def _bleed_transparent_rgb(frame: np.ndarray) -> np.ndarray:
    """Espalha o RGB dos pixels OPACOS para dentro da area 100% TRANSPARENTE (alpha=0) antes de
    qualquer reamostragem. GIF e formato de paleta indexada: o indice 'transparente' ainda carrega
    um RGB fixo (tipicamente preto) mesmo sem nunca aparecer na tela. Quando o cartopy reprojeta o
    icone pro globo (`regrid_shape`, resample bilinear), RGB e alpha sao interpolados de forma
    INDEPENDENTE -- na borda do desenho, um pixel opaco branco vizinho de um pixel preto
    transparente vira um pixel CINZA semi-transparente, criando um contorno fantasma ao redor do
    icone. Substituindo o RGB de cada pixel transparente pelo do pixel OPACO mais proximo (nearest-
    neighbor via distance transform), a interpolacao passa a misturar branco-com-branco -- so o
    alpha esmaece, sem vazar cor errada pra borda."""
    alpha = frame[..., 3]
    transparente = alpha == 0
    if not transparente.any() or transparente.all():
        return frame
    nearest_y, nearest_x = distance_transform_edt(transparente, return_distances=False, return_indices=True)
    out = frame.copy()
    out[..., :3][transparente] = frame[..., :3][nearest_y[transparente], nearest_x[transparente]]
    return out


def _load_gif_frames(path: Path) -> list[np.ndarray]:
    """Extrai todos os frames de um GIF animado como arrays RGBA uint8 (H, W, 4).
    Usa PIL para tratar corretamente paleta, transparência e disposal de cada frame."""
    frames: list[np.ndarray] = []
    try:
        gif = _PIL_Image.open(path)
        for i in range(getattr(gif, 'n_frames', 1)):
            gif.seek(i)
            frames.append(_bleed_transparent_rgb(np.array(gif.convert('RGBA'), dtype=np.uint8)))
    except Exception as exc:
        logger.warning('Erro ao carregar GIF {}: {}', path, exc)
    return frames


def _recolor_icon_frames(frames: list[np.ndarray], cor_alvo: str) -> list[np.ndarray]:
    """Reconverte o tom de AZUL dos icones de pressao (GIFs prontos em `Entrada/icones_pressao/`,
    sem script gerador no repositorio) pra `cor_alvo` (hex/nome), preservando SATURACAO e VALOR
    (sombreado/brilho do desenho original) -- so troca o MATIZ (hue) dos pixels que ja sao
    azulados o bastante (`hue` na faixa do azul E saturacao > 0.12); branco/preto/cinza (contorno,
    fundo) ficam intocados, assim como o canal alpha (transparencia/antialiasing das bordas)."""
    h_alvo = float(rgb_to_hsv(np.asarray(to_rgba(cor_alvo)[:3], dtype=np.float32).reshape(1, 1, 3))[0, 0, 0])
    out = []
    for fr in frames:
        rgb = fr[..., :3].astype(np.float32) / 255.0
        hsv = rgb_to_hsv(rgb)
        mask = (hsv[..., 0] > 0.45) & (hsv[..., 0] < 0.72) & (hsv[..., 1] > 0.12)
        hsv[..., 0] = np.where(mask, h_alvo, hsv[..., 0])
        fr2 = fr.copy()
        fr2[..., :3] = np.clip(hsv_to_rgb(hsv) * 255.0, 0, 255).astype(np.uint8)
        out.append(fr2)
    return out


def _draw_icones_pressao(ax, ctx: dict, f: int, data_transform, proj) -> None:
    """Plota ícones de pressão animados (GIFs) ancorados em lat/lon no globo.

    Cada ícone é renderizado via ax.imshow() com transform=data_transform, o que
    permite ao cartopy reprojetar o raster na superfície esférica e ocultar
    automaticamente elementos no limbo (lado escuro do globo).

    Sombra: versão totalmente preta do frame, deslocada por (sombra_dx, sombra_dy)
    graus e com opacidade reduzida, desenhada antes do ícone.

    Zorder: SEMPRE abaixo da corrente de jato quando ela está ligada (o jato nunca fica
    escondido atrás de um ícone) — ver `z_sombra`/`z_icone` abaixo.
    """
    icones = ctx.get('icones_pressao')
    if not icones:
        return
    regrid = int(ctx.get('icone_pressao_regrid', 2048))
    total_frames = max(1, int(ctx.get('total_frames', 1)))
    # A corrente de jato tem que ficar SEMPRE acima dos icones (nunca escondida atras de um icone
    # que passe por cima da faixa). O jato usa zorder ate 7 (tela) ou 9 (drape) -- com jato ligado,
    # os icones ficam 1-2 abaixo do maior zorder do jato; sem jato, mantem o valor alto de sempre
    # (10/11), so acima do resto do mapa.
    if ctx.get('jatos') and (ctx.get('hgt_z250_abs_cyc') is not None
                             or ctx.get('hgt_z500_abs_cyc') is not None):
        _jato_z = 9 if ctx.get('jato_drape') else 7
        z_sombra, z_icone = _jato_z - 2, _jato_z - 1
    else:
        z_sombra, z_icone = 10, 11
    for ic in icones:
        lat_c = float(ic['lat'])
        lon_c = float(ic['lon'])
        frames = ic.get('_frames')
        if not frames:
            continue
        # Verifica visibilidade: centro do ícone deve estar no hemisfério visível.
        # proj.transform_point devolve nan quando o ponto está atrás do globo.
        try:
            _px, _py = proj.transform_point(lon_c, lat_c, data_transform)
            if not (np.isfinite(_px) and np.isfinite(_py)):
                continue
        except Exception:
            continue
        velocidade = float(ic.get('velocidade', 1.0))
        n_frames_gif = len(frames)
        # Numero de voltas COMPLETAS do giro ao longo de todo o clipe (arredondado pra INTEIRO):
        # garante que o ultimo frame do clipe emende sem salto com o primeiro (loop do GIF) e que
        # o giro avance em passos uniformes -- em vez de `int(f*velocidade) % n_frames_gif`, que
        # deixa sobra fracionaria no fim do loop e da a impressao de "cortar e reiniciar".
        n_voltas = max(1, round(velocidade * total_frames / n_frames_gif))
        gif_f = int((f % total_frames) / total_frames * n_voltas * n_frames_gif) % n_frames_gif
        frame_rgba = frames[gif_f]                         # uint8 (H, W, 4)
        r = float(ic.get('tamanho_deg', 8.0)) / 2.0       # raio em graus (latitude)
        # Raio em LONGITUDE compensado por cos(lat): 1° de longitude equivale a cos(lat)° de
        # latitude em distancia FISICA na esfera -- um box simetrico em graus (mesmo r nos dois
        # eixos) fica cada vez mais ELIPTICO (esticado na vertical) longe do equador. Sem isso o
        # icone ficava OVAL sobre a Europa (~50°N, cos=0.64) mas quase circular perto do equador,
        # onde cos(lat)~1 escondia o problema. Clampa perto do polo p/ o raio nao explodir.
        # `ctx['mapa_plano']` (s43): PlateCarree e uma projecao equirretangular, sem essa
        # foreshortening esferica (1° de lon vale sempre o mesmo tanto em tela, em qualquer
        # latitude) -- aplicar a MESMA compensacao aqui esticaria o icone/jato sem necessidade
        # (ex.: ~1.49x na Europa, 48°N), entao pula a divisao por cos(lat) nesse caso.
        r_lon = r if ctx.get('mapa_plano') else r / max(abs(np.cos(np.deg2rad(lat_c))), 0.05)
        alpha = float(ic.get('alpha', 1.0))
        # ── Fade-in (mesmo mecanismo da caixa de texto livre): opcional, desligado por padrao
        # (aparece direto). Com `fade_in=true`, sobe de 0 a `alpha` entre `fade_inicio` e
        # `fade_inicio+fade_duracao` (fracao 0..1 do clipe). `fade_cauda_on` (teste s42) OVERRIDE
        # esses valores pelos do inicio da cauda, independente do que o icone tem configurado.
        # O fade e um conceito da LINHA DO TEMPO do MP4: nas saidas ESTATICAS (PNG da media) e no
        # GIF (loop do campo medio fixo), o icone deve ficar em alpha CHEIO -- senao o frame 0 do
        # PNG (fracao 0) ou o loop do GIF zerariam/oscilariam o alpha. So o MP4 anima o fade.
        _fade_in_on = ((bool(ic.get('fade_in', False)) or bool(ctx.get('fade_cauda_on')))
                       and not ctx.get('saida_estatica'))
        if _fade_in_on:
            if ctx.get('fade_cauda_on'):
                _fi = float(ctx.get('fade_cauda_inicio', 0.0))
                _fd = max(float(ctx.get('fade_cauda_dur', 0.0)), 1e-6)
            else:
                _fi = float(ic.get('fade_inicio', 0.0))
                _fd = max(float(ic.get('fade_duracao', 0.15)), 1e-6)
            _prog = f / max(total_frames - 1, 1)
            alpha *= float(np.clip((_prog - _fi) / _fd, 0.0, 1.0))
            if alpha <= 0.003:   # ainda invisivel -> pula o imshow/regrid (economiza render)
                continue
        # ── Sombra ────────────────────────────────────────────────────────────
        if ic.get('sombra', False):
            s_dx    = float(ic.get('sombra_dx', 1.5))
            s_dy    = float(ic.get('sombra_dy', -1.5))
            s_alpha = float(ic.get('sombra_alpha', 0.35)) * (alpha / max(float(ic.get('alpha', 1.0)), 1e-6))
            shadow = np.zeros_like(frame_rgba)
            shadow[..., 3] = (frame_rgba[..., 3].astype(np.float32) * s_alpha).astype(np.uint8)
            ax.imshow(shadow, origin='upper', transform=data_transform, zorder=z_sombra,
                      extent=[lon_c + s_dx - r_lon, lon_c + s_dx + r_lon,
                               lat_c + s_dy - r, lat_c + s_dy + r],
                      interpolation='bilinear', regrid_shape=regrid, clip_on=True)
        # ── Ícone ─────────────────────────────────────────────────────────────
        if alpha < 1.0:
            ic_rgba = frame_rgba.copy()
            ic_rgba[..., 3] = (ic_rgba[..., 3].astype(np.float32) * alpha).astype(np.uint8)
        else:
            ic_rgba = frame_rgba
        ax.imshow(ic_rgba, origin='upper', transform=data_transform, zorder=z_icone,
                  extent=[lon_c - r_lon, lon_c + r_lon, lat_c - r, lat_c + r],
                  interpolation='bilinear', regrid_shape=regrid, clip_on=True)


_STATIC_IMG_CACHE: dict[Path, np.ndarray | None] = {}


def _load_static_image_rgba(path: Path) -> np.ndarray | None:
    """Carrega uma imagem estática (PNG) como array RGBA uint8, cacheada por caminho."""
    if path not in _STATIC_IMG_CACHE:
        try:
            _STATIC_IMG_CACHE[path] = np.array(_PIL_Image.open(path).convert('RGBA'), dtype=np.uint8)
        except Exception as exc:
            logger.warning('Erro ao carregar imagem {}: {}', path, exc)
            _STATIC_IMG_CACHE[path] = None
    return _STATIC_IMG_CACHE[path]


_EARTH_M_PER_DEG = 111_195.0   # metros por grau de arco (raio medio da Terra) -- ver _draw_objeto_texto


def _draw_objeto_texto(ax, ctx: dict, data_transform, proj) -> None:
    """Plota a imagem estática ancorada em lat/lon (`ctx['objeto_texto']`, ver settings OBJETO_
    TEXTO): a ANCORA segue lat/lon (acompanha a rotacao do globo, some no lado escuro), mas a
    IMAGEM em si e desenhada RETA em `ax.transData` (coordenadas ja projetadas), igual as caixas
    de texto (`ax.text`) -- NAO segue a curvatura/perspectiva da esfera como um dado geografico
    normal (`transform=data_transform`) seguiria. SEM animação/sombra/fade e SEM recorte de
    continente -- o ponto configurado pode ficar de propósito no litoral/oceano (mesmo uso da
    caixa de texto livre), então nunca é mascarado. `tamanho_deg` e a MAIOR dimensao (imagem pode
    ser retangular, ex. um banner -- a proporcao original e preservada, nao estica pra virar
    quadrado); convertido p/ metros (unidade nativa do globo 3D) ou usado direto em graus no mapa
    2D (`mapa_plano`, PlateCarree -- `ax.transData` ja e lon/lat)."""
    obj = ctx.get('objeto_texto')
    if not obj or obj.get('_img') is None:
        return
    lat_c, lon_c = obj['lat'], obj['lon']
    try:
        px, py = proj.transform_point(lon_c, lat_c, data_transform)
        if not (np.isfinite(px) and np.isfinite(py)):
            return
    except Exception:
        return
    img = obj['_img']
    aspect = img.shape[1] / img.shape[0]   # largura/altura em pixels
    tam = obj['tamanho_deg'] * (1.0 if ctx.get('mapa_plano') else _EARTH_M_PER_DEG)
    w, h = (tam, tam / aspect) if aspect >= 1.0 else (tam * aspect, tam)
    r_w, r_h = w / 2.0, h / 2.0
    ax.imshow(img, origin='upper', transform=ax.transData, zorder=int(obj.get('zorder', 8)),
              extent=[px - r_w, px + r_w, py - r_h, py + r_h],
              interpolation='bilinear', clip_on=True)


_CLIP_UID = itertools.count()   # id sequencial de clipe -> entra na chave do cache do overlay


def _composite_overlay_box(arr: np.ndarray, ctx: dict, idx_dia: int, data_full: str) -> np.ndarray:
    """Compoe a caixa de texto/legenda (Guillaume) por cima de `arr` (float RGB). Cache de UMA
    entrada, valida enquanto a chave nao muda — os frames de fundo congelado repetem o mesmo dia.

    A chave inclui `clip_uid` (id do clipe, unico por `_render_clip`) porque `(variavel, idx_dia)`
    NAO identifica o overlay: MP4 e PNG-resumo do MESMO campo, ou dois MODELOS seguidos, tem
    datas/rotulo de modelo diferentes no mesmo idx_dia. Sem o uid, o PNG de um modelo (idx 0) vazava
    p/ o frame 0 do MP4 do modelo seguinte -- com a data do resumo ('Jul 17–25') no 1o frame."""
    if ctx.get('estilo') != 'guillaume':
        return arr
    global _OV_CACHE_KEY, _OV_CACHE_VAL
    _ov_key = (ctx.get('clip_uid'), ctx.get('variavel_key'), idx_dia)
    if _OV_CACHE_KEY != _ov_key:
        fig_ov = plt.figure(figsize=ctx.get('figsize', (8, 8)), dpi=ctx['dpi'])
        fig_ov.patch.set_alpha(0)
        fig_ov.canvas.draw()  # inicializa renderer para calculo do extent do texto
        data_br = ctx['dates_br'][idx_dia] if ctx.get('dates_br') else data_full
        _overlay_guillaume(fig_ov, ctx, ctx['cmap_legend'], data_full, data_br)
        fig_ov.canvas.draw()
        _OV_CACHE_VAL = np.asarray(fig_ov.canvas.buffer_rgba()).copy().astype(np.float32)
        plt.close(fig_ov)
        _OV_CACHE_KEY = _ov_key
    ov = _OV_CACHE_VAL
    ov_a = ov[..., 3:4] / 255.0
    return np.clip(arr * (1.0 - ov_a) + ov[..., :3] * ov_a, 0.0, 255.0)


def _pos_variavel(f: int, ctx: dict) -> float:
    """Posicao (fracionaria) do frame `f` no eixo de PASSOS da variavel (`vals_cyc`/`hgt_*_abs_cyc`),
    0..n_dias-1. `base_frames` (o voo) e calculado a partir de `n_dias_janela` (dias de CALENDARIO —
    ver comentario em `n_dias_janela`), entao o passo-por-frame nao pode ser `vel/frames_por_dia`
    direto: isso so bate quando `n_dias` (Nº DE PASSOS) == `n_dias_janela` (serie DIARIA de sempre,
    1 passo/dia). Com ACUM_HORARIO (varios passos nativos/dia) o voo tem `(n_dias_janela-1)*fpd`
    frames mas a serie tem `n_dias-1` passos -- sem escalar, o voo inteiro so alcancava uma fracao
    pequena de `n_dias` (ficava OBSERVADO travar a chuva/PNMM por volta do dia 2 numa janela de 11
    dias, mesmo a data/jato parecendo seguir em frente: ambos usam este `pos`, so que o jato tem sua
    FASE visual [dash pattern] avancando com `f` puro por cima do campo congelado). Reduz para a
    formula antiga (`f*vel/fpd`) quando `n_dias_janela == n_dias` (1 passo/dia)."""
    n_dias, fpd, vel = ctx['n_dias'], ctx['frames_por_dia'], ctx['vel_var']
    if n_dias <= 1:
        return 0.0
    n_dias_janela = ctx.get('n_dias_janela', n_dias)
    fpd = fpd if fpd > 0 else 1
    if n_dias_janela > 1:
        passos_por_frame = (n_dias - 1) / ((n_dias_janela - 1) * fpd)
    else:
        passos_por_frame = 1.0 / fpd
    return min(f * vel * passos_por_frame, n_dias - 1)


def _build_frame(f: int, ctx: dict, skip_jet: bool = False, skip_overlay: bool = False,
                 as_float: bool = False) -> np.ndarray:
    """Renderiza o frame `f` e devolve um array RGB uint8 (HxWx3).

    `skip_jet`/`skip_overlay`: usados para montar o FUNDO cacheado (sem jato e sem a caixa de texto).
    `as_float`: devolve float32 (sem clip p/ uint8) — para o fundo servir de base na composicao.

    CACHE DE FUNDO (GLOBO_3D_BG_CACHE): nos frames de fundo CONGELADO (cauda do MP4 + todos do GIF, onde
    campo e camera nao mudam) o fundo pesado (sombreado + reprojecao + costa/fronteiras + isolinhas) e
    reaproveitado de `ctx['_bg_arr']` e SO a corrente de jato e desenhada/composta por frame -> a
    animacao/qualidade do jato ficam IDENTICAS; muda so o que fica ATRAS dele (que e constante)."""
    # ── Fast path: fundo congelado cacheado -> compoe so o jato (fase `f`) + a caixa por cima ──
    if (not skip_jet and ctx.get('_bg_arr') is not None
            and f >= int(ctx.get('_bg_from', 1 << 30))):
        arr = ctx['_bg_arr'].copy()
        overlay = _render_overlay_rgba(ctx, f)
        if overlay is not None:
            _a = overlay[..., 3:4] / 255.0
            arr = arr * (1.0 - _a) + overlay[..., :3] * _a
        _n = ctx['n_dias']
        _pos = _pos_variavel(f, ctx)
        _idx = min(int(round(_pos)), _n - 1)
        _dfull = (ctx['dates_full'][_idx] if ctx.get('dates_full')
                  else ctx['dates_en'][_idx])
        arr = _composite_overlay_box(arr, ctx, _idx, _dfull)
        return arr.astype(np.uint8)

    vals_cyc = ctx['vals_cyc']
    lon_cyc, lat, levels = ctx['lon_cyc'], ctx['lat'], ctx['levels']
    n_dias = ctx['n_dias']
    # cmap_plot/cmap_legend herdados do pai via CoW (criados 1x em _render_clip).
    cmap_plot   = ctx['cmap_plot']
    cmap_legend = ctx['cmap_legend']
    paleta      = ctx['paleta']   # necessário para detecção de transparência no contorno
    data_transform = ccrs.PlateCarree()

    # Tempo (variavel) avanca por frame, desacoplado do voo (ver `_pos_variavel` -- escala pra
    # cobrir os `n_dias` passos ao longo do voo mesmo quando este tem menos frames que passos, ex.
    # ACUM_HORARIO); clampa no ultimo passo se terminar antes do fim do voo.
    pos = _pos_variavel(f, ctx)
    i0 = min(int(np.floor(pos)), n_dias - 1)
    i1 = min(i0 + 1, n_dias - 1)
    w = pos - i0
    campo = (1.0 - w) * vals_cyc[i0] + w * vals_cyc[i1]
    # TRANSPARÊNCIA central do shaded: |anom| < GLOBO_3D_TRANSP_ATE_<VAR> vira NaN (transparente) ->
    # aparece o fundo (blue marble). A máscara de OCEANO (só continente) NÃO é feita aqui (na grade
    # grossa, que serrilha a costa) e sim por RECORTE VETORIAL do shaded no litoral 50m, abaixo.
    _transp = float(ctx.get('transp_ate', 0.0))
    if _transp > 0.0:
        campo = np.array(campo, dtype=np.float32, copy=True)
        campo[np.abs(campo) < _transp] = np.nan
    idx_dia = min(int(round(pos)), n_dias - 1)
    data_en = ctx['dates_en'][idx_dia]
    data_full = ctx['dates_full'][idx_dia] if ctx.get('dates_full') else data_en
    data_wapo = ctx['dates_wapo'][idx_dia] if ctx.get('dates_wapo') else data_en
    guillaume = ctx.get('estilo') == 'guillaume'

    proj = _make_projection(float(ctx['lons'][f]), float(ctx['lats'][f]),
                            mode=ctx.get('proj_mode', 'nearside'),
                            sat_height=ctx.get('sat_height', _SAT_HEIGHT_GEO))
    fig = plt.figure(figsize=ctx.get('figsize', (8, 8)), dpi=ctx['dpi'])
    fig.patch.set_facecolor('black')
    # Guillaume: globo grande (disco raio ~0.43, topo y~0.92), preenchendo o quadro
    # como na referencia; o canto sup-esq fica livre p/ a caixa compacta do nome/data
    # (no x=0.235 da caixa, o topo do globo cai p/ ~0.83, abaixo da data).
    # WaPo: rect [0.01, 0.10, 0.98, 0.83] -> r=0.415, cy=0.515, disc_top=0.930.
    # O disco desce até y=0.10 (dentro da tarja), mas o Rectangle preto (zorder=15)
    # mascara a parte do globo abaixo de barra_h=0.17; o bar_backup elimina halo/estrelas.
    rect = ctx.get('globe_rect') or ([0.07, 0.06, 0.86, 0.86] if guillaume
                                      else [0.01, 0.10, 0.98, 0.83])
    ax = fig.add_axes(rect, projection=proj)
    # Base do disco = cor do OCEANO quando GLOBO_3D_COR_OCEANO está setada (s42: oceano cor sólida +
    # continente com blue marble recortado na terra); senão a cor de fundo de sempre.
    ax.patch.set_facecolor(ctx.get('bg_oceano_cor') or ctx.get('cor_fundo_globo', 'black'))
    ax.set_global()
    _aplicar_janela_inclinada(ax, proj, ctx.get('inc'))   # so s44; no-op p/ os demais
    if guillaume:
        ax.spines['geo'].set_linewidth(0)  # remove o anel preto; o halo azul define a borda

    # Imagem de satelite (blue marble) como fundo do globo, ANTES de tudo (zorder=0) -- so
    # aparece de fato onde o shaded por cima for semi-transparente (ver `shaded_alpha`).
    if ctx.get('fundo_blue_marble'):
        _bm = _blue_marble_rgba()
        if _bm is not None:
            # RECORTE NO BBOX VISIVEL antes do warp: o custo de RAM da reprojecao escala com a FONTE
            # (KDTree sobre todos os pixels dela), POR FRAME E POR WORKER -- ver `_bbox_visivel_lonlat`.
            # Mandar o planeta inteiro p/ mostrar ~20° custava 2,8 GB/warp em 8192; recortado, ~0,36 GB
            # com a MESMA nitidez nativa. Vista larga (s38-s43) -> bbox None -> caminho de antes, intacto.
            _bm_ext: tuple = (-180.0, 180.0, -90.0, 90.0)
            if ctx.get('blue_marble_crop', True):
                _bm, _bm_ext = _recorta_lonlat(_bm, _bm_ext, _bbox_visivel_lonlat(ax, proj))
            _bm_art = ax.imshow(_bm, origin='upper', extent=_bm_ext, transform=data_transform,
                      interpolation='bilinear', zorder=0,
                      regrid_shape=int(ctx.get('blue_marble_regrid', 2048)))
            # CONTINENTE = satélite / OCEANO = cor sólida: recorta o blue marble na TERRA (50m vetorial)
            # -> a água mostra a base (bg_oceano_cor). So quando GLOBO_3D_COR_OCEANO está setada.
            if ctx.get('bg_oceano_cor'):
                _cp = _land_clip_path(proj, data_transform)
                if _cp is not None:
                    _bm_art.set_clip_path(_cp, ax.transData)

    # Continentes/oceanos coloridos (para variaveis absolutas onde abaixo do vmin = transparente,
    # ou pra cobrir a agua do blue marble com uma cor solida estilo TV -- terra fica com a textura
    # de satelite por baixo, oceano vira cor chapada).
    if ctx.get('cor_continente'):
        ax.add_feature(cfeature.LAND.with_scale('50m'),
                       facecolor=ctx['cor_continente'], zorder=1)
    if ctx.get('cor_oceano'):
        ax.add_feature(cfeature.OCEAN.with_scale('50m'),
                       facecolor=ctx['cor_oceano'], zorder=1)

    # Opacidade do shaded POR VARIAVEL: ficha['shaded_alpha'] (override GLOBO_3D_ALPHA_<VAR>);
    # default 1.0 (opaco, comportamento de sempre).
    _shaded_alpha = float(ctx.get('shaded_alpha', 1.0))
    _shaded_art = None
    if ctx.get('sem_variavel'):
        pass  # SO s42/s43: nenhum campo shaded desenhado -- so features (jato/icones/caixas).
    elif ctx.get('usar_pcolormesh'):
        # 'gouraud' = gradiente CONTINUO liso; 'nearest' = BANDAS DISCRETAS chapadas (BoundaryNorm).
        # alpha: passar 1.0 ESCALAR ao pcolormesh SOBRESCREVE o alpha por-pixel do colormap (o branco
        # #ffffff00 virava OPACO) -> so passa alpha quando <1.0; senao None preserva o alpha do cmap
        # (transparencia da paleta = abaixo do vmin/branco some, revela o fundo do globo).
        _shd = ctx.get('pcolormesh_shading', 'gouraud')
        _shaded_art = ax.pcolormesh(lon_cyc, lat, campo, norm=ctx['norm_fn'], cmap=cmap_plot,
                      transform=data_transform, shading=_shd, zorder=2,
                      alpha=(_shaded_alpha if _shaded_alpha < 1.0 else None))
    else:
        # BANDAS suaves do contourf (estilo WaPo) SEM o bug do cartopy: o contourf e renderizado
        # num raster PLANO (_contourf_raster) e composto no globo via imshow (reprojecao de raster).
        # Desenhar ax.contourf direto no globo aciona o bug cartopy 0.25+mpl 3.10 (geometrias ->
        # TypeError -> _safe_transform_path devolve path vazio -> poligonos descartados -> washout).
        _shade_px = int(ctx.get('shade_px', 3600))
        _regrid = int(ctx.get('shade_regrid', 2048))  # resolucao da REPROJECAO (cartopy default=750=mole)
        # RECORTE do raster na area visivel (mesma ideia do satelite): o raster global desperdica
        # px/grau fora do quadro e chega serrilhado na janela fechada do inclinado. Ver
        # `_subset_campo_bbox`. Vista larga (s38-s43) -> bbox None -> raster global, caminho de antes.
        _lon_r, _lat_r, _campo_r = lon_cyc, lat, campo
        if ctx.get('shade_crop', True):
            _lon_r, _lat_r, _campo_r = _subset_campo_bbox(lon_cyc, lat, campo,
                                                          _bbox_visivel_lonlat(ax, proj))
            if _lon_r.size < lon_cyc.size:   # recortou de fato -> px proprio (menor E mais denso)
                _shade_px = int(ctx.get('shade_crop_px', 2048))
                # Upsample CUBICO do campo recortado: tira o degrau da CELULA do modelo (limite do
                # dado, nao do raster) preservando os valores. So aqui dentro (campo pequeno) -- no
                # campo global seria uma bomba de RAM. Ver `_upsample_campo`.
                _lon_r, _lat_r, _campo_r = _upsample_campo(_lon_r, _lat_r, _campo_r,
                                                           ctx.get('shade_upsample', 1.0))
        flat_rgba = _contourf_raster(_lon_r, _lat_r, _campo_r, levels, cmap_plot,
                                     ctx.get('extend_contourf', 'both'), px=_shade_px)
        _shaded_art = ax.imshow(flat_rgba, origin='upper', transform=data_transform, zorder=2,
                  extent=[float(_lon_r.min()), float(_lon_r.max()),
                          float(_lat_r.min()), float(_lat_r.max())],
                  interpolation='bilinear', regrid_shape=_regrid, alpha=_shaded_alpha)
    # Recorte do shaded (prioridade: BRASIL > CONTINENTE > mascara_oceano por variavel).
    # PLOTAR_SOMENTE_BRASIL/CONTINENTE (globais, s42/s44) sobrepoem o `mascara_oceano` da
    # ficha -- SO CONTINENTE recorta na geometria VETORIAL da costa (50m) -> litoral liso (sem
    # serrilhado da grade) e a água fica de fora -> aparece o fundo do oceano.
    if _shaded_art is not None:
        if ctx.get('plotar_somente_brasil'):
            _cp = _brasil_clip_path(proj, data_transform)
        elif ctx.get('plotar_somente_continente') or ctx.get('mascara_oceano'):
            _cp = _land_clip_path(proj, data_transform)
        else:
            _cp = None
        if _cp is not None:
            _shaded_art.set_clip_path(_cp, ax.transData)

    # Polígonos de tempo severo: SOBRE o shaded (zorder 2), recortados no CONTINENTE (nunca aparecem
    # no oceano) e ABAIXO da costa/fronteiras/estados (zorder 5) -- por isso zorder=3 aqui, entre os
    # dois. Reusa o mesmo recorte vetorial 50m do `mascara_oceano`/PLOTAR_SOMENTE_CONTINENTE.
    _draw_poligonos_severo(ax, ctx, proj, data_transform)

    # Camada opcional de NEVE acumulada (overlay `NEVE`, ver ficha `precip_abs`): MESMA tecnica do
    # campo primario (raster de contourf via _contourf_raster + imshow, nao pcolormesh) pra banda discreta bater com a
    # chuva; abaixo do 1o nivel fica transparente por construcao do contourf (extend='max').
    _neve_ov = ctx.get('neve_overlay')
    if _neve_ov is not None:
        neve_campo = (1.0 - w) * _neve_ov['cyc'][i0] + w * _neve_ov['cyc'][i1]
        _n_lon_r, _n_lat_r, _n_campo_r = lon_cyc, lat, neve_campo
        _dbg_bbox = _bbox_visivel_lonlat(ax, proj)
        if ctx.get('shade_crop', True):
            _n_lon_r, _n_lat_r, _n_campo_r = _subset_campo_bbox(
                lon_cyc, lat, neve_campo, _dbg_bbox)
        neve_rgba = _contourf_raster(_n_lon_r, _n_lat_r, _n_campo_r, _neve_ov['levels'],
                                     _neve_ov['cmap'], 'max', px=int(ctx.get('shade_px', 3600)))
        # DEBUG TEMPORARIO (investigacao Andes x Patagonia, remover depois)
        try:
            import numpy as _np2
            _andes_mask_pre = (((lon_cyc % 360) >= 288) & ((lon_cyc % 360) <= 291))
            _andes_lat_pre = (lat >= -37) & (lat <= -30)
            _andes_slice_pre = neve_campo[_np2.ix_(_andes_lat_pre, _andes_mask_pre)]
            with open('/tmp/debug_neve2.txt', 'a') as _f:
                _f.write(f'bbox_visivel={_dbg_bbox} shade_crop={ctx.get("shade_crop", True)}\n')
                _f.write(f'neve_campo (pre-crop) Andes max={float(_np2.nanmax(_andes_slice_pre)) if _andes_slice_pre.size else "vazio"} size={_andes_slice_pre.size}\n')
                _f.write(f'pos-crop: n_lon_r range=[{float(_n_lon_r.min())},{float(_n_lon_r.max())}] '
                        f'n_lat_r range=[{float(_n_lat_r.min())},{float(_n_lat_r.max())}] '
                        f'n_campo_r shape={_n_campo_r.shape} max={float(_np2.nanmax(_n_campo_r)) if _n_campo_r.size else "vazio"} '
                        f'nan_pct={100*_np2.isnan(_n_campo_r).mean() if _n_campo_r.size else -1}\n')
                _f.write(f'neve_rgba shape={neve_rgba.shape} alpha_max={int(neve_rgba[...,3].max())} '
                        f'pct_pintado={100*(neve_rgba[...,3]>10).mean():.2f}%\n')
                _f.flush()
        except Exception as _dbg_e2:
            with open('/tmp/debug_neve2.txt', 'a') as _f:
                _f.write(f'DEBUG2 EXCEPTION: {type(_dbg_e2).__name__}: {_dbg_e2}\n')
        _neve_art = ax.imshow(neve_rgba, origin='upper', transform=data_transform, zorder=3,
                              extent=[float(_n_lon_r.min()), float(_n_lon_r.max()),
                                      float(_n_lat_r.min()), float(_n_lat_r.max())],
                              interpolation='bilinear')
        if _cp is not None:
            _neve_art.set_clip_path(_cp, ax.transData)

    # Camada de OLR equatorial: alpha = taper(lat) [suave, sem corte reto] x |OLR|/knee [some onde
    # nao ha sinal], com teto amax. Aparece so na faixa equatorial e some gradualmente nas bordas.
    _olr_ov = ctx.get('olr_overlay')
    if _olr_ov is not None:
        olr_campo = np.nan_to_num((1.0 - w) * _olr_ov['cyc'][i0] + w * _olr_ov['cyc'][i1], nan=0.0)
        mag = np.clip(np.abs(olr_campo) / max(_olr_ov['knee'], 1e-6), 0.0, 1.0)
        alpha2d = (_olr_ov['taper'][:, None] * mag * _olr_ov['amax']).astype(np.float32)
        ax.pcolormesh(_olr_ov['lon'], _olr_ov['lat'], olr_campo,
                      cmap=_olr_ov['cmap'], norm=_olr_ov['norm'], alpha=alpha2d,
                      shading='gouraud', transform=data_transform, zorder=3)

    # Isolinha de temperatura absoluta 0°C (variaveis com clim_abs_cyc no ctx)
    _iso_txts: list = []
    if ctx.get('clim_abs_cyc') is not None:
        _clim_f = (1.0 - w) * ctx['clim_abs_cyc'][i0] + w * ctx['clim_abs_cyc'][i1]
        cs0 = ax.contour(lon_cyc, lat, campo + _clim_f, levels=[0.0], colors='#444444',
                         linewidths=0.7, transform=data_transform, zorder=6)
        _iso_txts = list(ax.clabel(cs0, fmt='0°C', fontsize=8, colors='white',
                                   inline=True, inline_spacing=3))
        for _t in _iso_txts:
            _t.set_fontweight('bold')
            _t.set_zorder(50)       # acima de coastlines (5), vinheta (10), tudo
            _t.set_clip_on(False)   # nunca clipado pela projecao
            _t.set_bbox(dict(facecolor='#444444', edgecolor='none', pad=2, alpha=1.0))
    # Isolinhas de geopotencial absoluto 250 hPa (cinza escuro, sem label)
    if ctx.get('hgt_abs_cyc') is not None and ctx.get('hgt_abs_levels') is not None:
        _hgt_f = (1.0 - w) * ctx['hgt_abs_cyc'][i0] + w * ctx['hgt_abs_cyc'][i1]
        if ctx.get('campo_absoluto'):
            if ctx.get('hgt_anom_vals_cyc') is not None:
                # Z250 anomalia real + clim = Z250 absoluto do dia (identico ao z250_anom)
                _hgt_a = (1.0 - w) * ctx['hgt_anom_vals_cyc'][i0] + w * ctx['hgt_anom_vals_cyc'][i1]
                _hgt_contour = _hgt_a + _hgt_f
            else:
                _hgt_contour = _hgt_f  # fallback: apenas climatologia
        else:
            _hgt_contour = campo + _hgt_f
        _hgt_iso_color = 'whitesmoke' if ctx.get('campo_absoluto') else '#666666'
        ax.contour(lon_cyc, lat, _hgt_contour, levels=ctx['hgt_abs_levels'],
                   colors=_hgt_iso_color, linewidths=0.35, transform=data_transform, zorder=3)
    # Isolinhas de Z250 absoluto sobre um campo ESTRANHO (tmp850_anom, olr_anom, ...): tracadas direto
    # do Z250 reconstruido (anom z250 + clim), independente do campo shaded de fundo. Cor configuravel
    # (GLOBO_3D_ISOL_HGT250_COR, default cinza escuro).
    elif ctx.get('hgt_z250_iso_levels') is not None and ctx.get('hgt_z250_abs_cyc') is not None:
        _hgt_z = (1.0 - w) * ctx['hgt_z250_abs_cyc'][i0] + w * ctx['hgt_z250_abs_cyc'][i1]
        ax.contour(lon_cyc, lat, _hgt_z, levels=ctx['hgt_z250_iso_levels'],
                   colors=ctx.get('hgt_z250_iso_cor', '#666666'), linewidths=0.35,
                   transform=data_transform, zorder=3)
    # Isolinhas fixas coloridas de Z250 absoluto (ex.: 10080/10200/10680 mgp no jet stream)
    if ctx.get('isolinhas_fixas_hgt') and ctx.get('hgt_z250_abs_cyc') is not None:
        _hgt_z250 = (1.0 - w) * ctx['hgt_z250_abs_cyc'][i0] + w * ctx['hgt_z250_abs_cyc'][i1]
        for _nivel, _cor, _lw in ctx['isolinhas_fixas_hgt']:
            ax.contour(lon_cyc, lat, _hgt_z250, levels=[float(_nivel)],
                       colors=[_cor], linewidths=_lw, transform=data_transform, zorder=7)
    # Corrente de jato (s41): faixa central opaca + faixas finas translucidas ao longo da isolinha
    # de Z250 absoluto; 'JET STREAM' + setas deslizam W->E por cima (posicao do dado, overlay animado).
    # DRAPE: renderiza o jato num raster plano e o cola na esfera (perspectiva 3D correta, rente a
    # superficie); senao, desenha direto no globo como adesivos de tela (rapido, mas embaralha no limbo).
    if not skip_jet and ctx.get('jatos') and (ctx.get('hgt_z250_abs_cyc') is not None
                                              or ctx.get('hgt_z500_abs_cyc') is not None):
        _hgt_jato_by_kind: dict = {}
        if ctx.get('hgt_z250_abs_cyc') is not None:
            _hgt_jato_by_kind['hgt250'] = ((1.0 - w) * ctx['hgt_z250_abs_cyc'][i0]
                                           + w * ctx['hgt_z250_abs_cyc'][i1])
        if ctx.get('hgt_z500_abs_cyc') is not None:
            _hgt_jato_by_kind['hgt500'] = ((1.0 - w) * ctx['hgt_z500_abs_cyc'][i0]
                                           + w * ctx['hgt_z500_abs_cyc'][i1])
        _draw_jet_layer(ax, ctx, f, _hgt_jato_by_kind, lon_cyc, lat, data_transform)
    # Isolinhas de PNMM (MSLP) — para variáveis como tmp850_mslp
    if ctx.get('mslp_cyc') is not None and ctx.get('mslp_levels') is not None:
        _mslp_f = (1.0 - w) * ctx['mslp_cyc'][i0] + w * ctx['mslp_cyc'][i1]
        ax.contour(lon_cyc, lat, _mslp_f, levels=ctx['mslp_levels'],
                   colors=str(settings.get('GLOBO_3D_MSLP_COR', 'white')),
                   linewidths=float(settings.get('GLOBO_3D_MSLP_LW', 0.5)),
                   transform=data_transform, zorder=4)
    # Isolinhas AUXILIARES de uma 2a variavel (ex.: psi200 PRETO sobre o chi200 shaded).
    # Usa a grade PROPRIA do contorno (pode diferir da do shaded — ex.: psi200 2.5° sobre OLR 0.5°).
    if ctx.get('contour_cyc') is not None and ctx.get('contour_levels') is not None:
        _ct_f = (1.0 - w) * ctx['contour_cyc'][i0] + w * ctx['contour_cyc'][i1]
        _ct_lon = ctx.get('contour_lon') if ctx.get('contour_lon') is not None else lon_cyc
        _ct_lat = ctx.get('contour_lat') if ctx.get('contour_lat') is not None else lat
        ax.contour(_ct_lon, _ct_lat, _ct_f, levels=ctx['contour_levels'],
                   colors=ctx.get('contour_color', 'black'),
                   linewidths=ctx.get('contour_lw', 0.5),
                   transform=data_transform, zorder=4)
    # Isolinhas brancas — controladas por GLOBO_3D_CONTORNO ou GLOBO_3D_CONTORNO_<VAR>.
    if ctx.get('usar_contorno', False):
        _step_pal = (levels[-1] - levels[0]) / (len(paleta) - 1)
        _transp_idx = [i for i, c in enumerate(paleta)
                       if isinstance(c, str) and len(c) == 9 and c[7:9].upper() == '00']
        if _transp_idx:
            _transp_lo = levels[0] + _transp_idx[0]  * _step_pal
            _transp_hi = levels[0] + _transp_idx[-1] * _step_pal
            _campo_c = np.where((campo >= _transp_lo) & (campo <= _transp_hi), np.nan, campo)
            _clvls = [lv for lv in levels if lv < _transp_lo or lv > _transp_hi]
        else:
            _campo_c, _clvls = campo, list(levels)
        if _clvls:
            ax.contour(lon_cyc, lat, _campo_c, levels=_clvls, colors='white',
                       linewidths=0.35, transform=data_transform, zorder=3)
    edge_color = ctx.get('cor_fronteiras') or ('#444444' if guillaume else 'black')
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'),
                   linewidth=ctx['lw_coast'], edgecolor=edge_color, zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'),
                   linewidth=ctx['lw_border'], edgecolor=edge_color, zorder=5)
    _estados = _state_line_geoms()
    if _estados:
        ax.add_geometries(_estados, data_transform, edgecolor=edge_color,
                          facecolor='none', linewidth=ctx['lw_states'], zorder=5)

    # ── Ícones de pressão animados (GIFs ancorados em lat/lon) ──────────────
    if ctx.get('icones_pressao'):
        _draw_icones_pressao(ax, ctx, f, data_transform, proj)

    # ── Objeto de figura estático ancorado em lat/lon (settings OBJETO_TEXTO) ────
    # `not skip_overlay`: como a caixa livre, o objeto NAO entra no fundo cacheado (senao ficaria
    # assado ABAIXO do jato, que e composto por cima do fundo -> zorder alto nao adiantaria nos
    # frames congelados). Nos frames de fundo congelado ele e redesenhado no overlay
    # (_render_overlay_rgba), acima do jato -> o zorder do dict vale em QUALQUER frame (voo = render
    # completo aqui; cauda/GIF = overlay). No render completo, jato e objeto estao na MESMA figura,
    # entao um zorder > 9 ja poe o objeto acima do jato.
    if not skip_overlay and ctx.get('objeto_texto'):
        _draw_objeto_texto(ax, ctx, data_transform, proj)

    # ── Escoamento de baixos níveis (setas manuais animadas, Entrada/seta.png) ──
    # `not skip_jet`: mesma exclusao do jato (linha ~3772) -- o escoamento MUDA DE POSICAO por
    # frame (desliza ao longo da curva), entao nao pode ser assado no fundo cacheado (`skip_jet=
    # True` ao montar `ctx['_bg_arr']`), senao fica uma copia PARADA (no fundo) + uma ANIMADA (no
    # overlay por cima) -- exatamente o "duas series de seta" visto no teste.
    if not skip_jet and ctx.get('escoamentos'):
        _draw_escoamento_layer(ax, ctx, f, data_transform)

    # ── Caixa do Niño 3.4 (170°W–120°W, 5°S–5°N) + rotulo — flag GLOBO_3D_BOX_NINO34 (qualquer var) ──
    if ctx.get('box_nino34'):
        lon0, lon1, lat0, lat1 = -170.0, -120.0, -5.0, 5.0
        nx, ny = 80, 24
        bx = np.concatenate([np.linspace(lon0, lon1, nx), np.full(ny, lon1),
                             np.linspace(lon1, lon0, nx), np.full(ny, lon0)])
        by = np.concatenate([np.full(nx, lat0), np.linspace(lat0, lat1, ny),
                             np.full(nx, lat1), np.linspace(lat1, lat0, ny)])
        # Retangulo em coords geograficas -> ja segue a curvatura/perspectiva do globo.
        # So contorno PRETO (sem halo branco).
        ax.plot(bx, by, transform=data_transform, color='black', linewidth=1.8,
                solid_capstyle='round', zorder=8)
        # Rotulo "Niño 3.4" — texto unico (espacamento natural) ROTACIONADO pelo tangente local
        # da projecao, de modo a inclinar junto com a perspectiva do globo conforme a caixa desliza.
        _lon_c, _lat_c = 0.5 * (lon0 + lon1), lat1 + 3.0
        _p0 = proj.transform_point(_lon_c - 4.0, _lat_c, data_transform)
        _p1 = proj.transform_point(_lon_c + 4.0, _lat_c, data_transform)
        _ang = (np.degrees(np.arctan2(_p1[1] - _p0[1], _p1[0] - _p0[0]))
                if np.isfinite(_p0[0]) and np.isfinite(_p1[0]) else 0.0)
        # Rotulo DINAMICO quando a ficha traz o valor diario do box (tsm_abs): "Niño 3.4 = xx.x°C",
        # interpolado entre os dias i0/i1 como o campo. Sem serie (anomalia etc.) -> so "Niño 3.4".
        _nser = ctx.get('nino34_serie')
        _label34 = 'Niño 3.4'
        if _nser is not None and len(_nser):
            _v34 = (1.0 - w) * _nser[i0] + w * _nser[i1]
            if np.isfinite(_v34):
                # anomalia (tsm_anom) -> com sinal (+0.8/-0.5); absoluta (tsm_abs) -> sem sinal (26.5)
                _fmt = f'{_v34:+.1f}' if ctx.get('nino34_signed') else f'{_v34:.1f}'
                _label34 = f'Niño 3.4 = {_fmt}°C'
        _t34 = ax.text(_lon_c, _lat_c, _label34, transform=data_transform, rotation=_ang,
                       rotation_mode='anchor', color='white', fontsize=13, fontweight='bold',
                       ha='center', va='center', family=ctx.get('font_legenda', FONT_SANS),
                       zorder=9)
        _t34.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'),
                               path_effects.Normal()])

    # ── Caixa de texto LIVRE ancorada em lat/lon (segue a rotacao do globo) ──────────
    # `skip_overlay` (montagem do FUNDO cacheado, `_bg_arr`): pula a caixa aqui -- ela e
    # redesenhada por frame via `_render_overlay_rgba` (fast path do cache), com o alpha certo
    # p/ o frame corrente. Sem essa guarda, a caixa ficava CONGELADA no alpha do frame em que o
    # fundo foi cacheado (ou dobrada, desenhada aqui E de novo no overlay).
    if not skip_overlay:
        _draw_caixa_livre(ax, ctx, f, data_transform)
    ax.set_zorder(1)

    # ── Render do globo (sem overlay de texto/legenda) ───────────────────────
    if not guillaume:
        if ctx['usar_vinheta']:
            vax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
            vax.set_axis_off()
            vax.set_zorder(10)
            vax.imshow(_vignette_rgba(), extent=[0, 1, 0, 1], origin='lower',
                       aspect='auto', interpolation='bilinear', zorder=10)
            vax.set_xlim(0, 1)
            vax.set_ylim(0, 1)

        # ── Tarja inferior (estilo WaPo) ──
        # Rectangle sólido preto mascara a parte do disco que desce abaixo da tarja.
        _bar_h = ctx.get('barra_h', 0.17)
        fig.add_artist(Rectangle((0, 0), 1.0, _bar_h, transform=fig.transFigure,
                                 facecolor='black', edgecolor='none', zorder=15))
        _titulo_sufixo = '' if ctx.get('campo_absoluto') else ' anomalies'
        _titulo_wapo = _fmt_nivel_hpa(f"{ctx['titulo_en']}{_titulo_sufixo} on {data_wapo}")
        fig.text(0.5, 0.136, _titulo_wapo, color='white',
                 fontsize=15, ha='center', va='center', weight='bold',
                 family=ctx['font_legenda'], zorder=20)

        if ctx.get('legenda_numerica'):
            # Barra CONTINUA + ticks NUMERICOS (ex.: tsm_anom) no lugar dos 4 swatches categoricos.
            bar_w, bar_hh = 0.46, 0.022
            bx0 = 0.5 - bar_w / 2.0
            by0 = 0.082
            cbax = fig.add_axes([bx0, by0, bar_w, bar_hh])
            cbax.set_zorder(20)
            cbax.imshow(np.linspace(0.0, 1.0, 256).reshape(1, -1), aspect='auto',
                        cmap=cmap_legend, extent=[0, 1, 0, 1], origin='lower')
            cbax.set_axis_off()
            vmin, vmax = float(ctx['levels'][0]), float(ctx['levels'][-1])
            for t in _numeric_legend_ticks(vmin, vmax, ctx.get('legenda_num_step', 0.5)):
                xpos = bx0 + bar_w * (t - vmin) / (vmax - vmin)
                fig.text(xpos, by0 - 0.018, f'{t:g}', color='white', fontsize=7,
                         ha='center', va='center', weight='bold', family=ctx['font_legenda'], zorder=20)
            if ctx.get('legenda_unidade'):
                fig.text(bx0 + bar_w / 2, by0 - 0.036, ctx['legenda_unidade'], color='white',
                         fontsize=10, ha='center', va='top', weight='bold',
                         family=ctx['font_legenda'], zorder=20)
        else:
            sw_w, sw_h, gap = 0.085, 0.024, 0.050
            x0 = 0.5 - (4 * sw_w + 3 * gap) / 2.0
            y_sw = 0.080
            legenda_cores = [cmap_legend(x) for x in (0.12, 0.34, 0.66, 0.88)]
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
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy().astype(np.float32)
    # Salva bbox dos labels da isoterma 0°C enquanto o renderer ainda está disponível.
    # Necessário para restaurar as caixinhas após atmosfera/estrelas (numpy pós-render).
    _iso_label_backups: list = []
    if _iso_txts:
        _rend = fig.canvas.get_renderer()
        _ih, _iw = arr.shape[:2]
        for _txt in _iso_txts:
            try:
                _bb = _txt.get_window_extent(_rend)
                _pad = 4
                _r0 = max(0, _ih - int(_bb.y1) - _pad)
                _r1 = min(_ih, _ih - int(_bb.y0) + _pad)
                _c0 = max(0, int(_bb.x0) - _pad)
                _c1 = min(_iw, int(_bb.x1) + _pad)
                if _r0 < _r1 and _c0 < _c1:
                    _iso_label_backups.append((_r0, _r1, _c0, _c1, arr[_r0:_r1, _c0:_c1].copy()))
            except Exception:
                pass
    plt.close(fig)
    h, w = arr.shape[:2]

    # Tarja inferior (WaPo/s38): salva os pixels ANTES de atmosfera+estrelas para
    # restaurar depois — impede que estrelas e halo vazem para dentro da tarja de texto.
    bar_px = int(ctx.get('barra_h', 0.0) * h)
    bar_backup = arr[-bar_px:, :].copy() if bar_px > 0 else None

    # ── Atmosfera (halo azul) + estrelas ─────────────────────────────────────
    # Aplicados ANTES do overlay de texto/legenda para que o overlay tenha
    # prioridade máxima e nunca apareçam estrelas sobre a caixa cinza ou o cbar.
    # GLOBO_3D_SOMENTE_ESTRELAS (s38-s42): quando ATMOSFERA_ESTRELAS=false, desenha SO as
    # estrelas (pula o halo/blur do Passo 1+2) -- mesma geometria (atm_cx/cy/r) dos dois modos.
    _full_atm = ctx.get('usar_atmosfera_estrelas', False)
    _so_estrelas = ctx.get('usar_somente_estrelas', False)
    if _full_atm or _so_estrelas:
        _cx = ctx.get('atm_cx', 0.50)
        _cy = ctx.get('atm_cy', 0.49)
        _r  = ctx.get('atm_r',  0.433)

        if _full_atm:
            glow = _atmosphere_glow_rgba(h, w, cx=_cx, cy=_cy, r_frac=_r)

            # Passo 1: halo nos pixels escuros (espaço)
            is_dark = (arr.mean(axis=-1) < 30.0)[..., np.newaxis].astype(np.float32)
            a = glow[..., 3:4] * is_dark
            arr = np.clip(arr * (1.0 - a) + glow[..., :3] * 255.0 * a, 0.0, 255.0)

            # Passo 2: blur estreito (±5px) na borda do disco
            R_px = _r * min(h, w)
            _cx_px = _cx * w
            _cy_px = (1.0 - _cy) * h   # y de figura (0=inferior) -> linha de imagem (0=superior)
            rows_b, cols_b = np.mgrid[0:h, 0:w]
            r_b = np.sqrt((cols_b - _cx_px) ** 2 + (rows_b - _cy_px) ** 2)
            blend_w = np.clip(1.0 - np.abs(r_b - R_px) / 5.0, 0.0, 1.0)[..., np.newaxis]
            arr_blur = _gaussian_filter(arr, sigma=[2, 2, 0])
            arr = arr * (1.0 - blend_w) + arr_blur * blend_w

        # Passo 3: estrelas no fundo escuro (sempre que full_atm OU so_estrelas)
        stars = _starfield_rgba(h, w, cx=_cx, cy=_cy, r_disc=_r)
        s_alpha = stars[..., 3:4]
        arr = np.clip(arr + stars[..., :3] * s_alpha * 255.0, 0.0, 255.0)

    # Restaura a tarja inferior: elimina estrelas/halo que vazaram p/ a faixa de texto
    if bar_backup is not None:
        arr[-bar_px:, :] = bar_backup
    # Restaura caixinhas dos labels da isoterma: opacidade total, sem atmosfera por cima
    for (_r0, _r1, _c0, _c1, _bk) in _iso_label_backups:
        arr[_r0:_r1, _c0:_c1] = _bk

    # ── Overlay de texto/legenda (Guillaume) — sempre por cima de tudo ───────
    # Renderizado numa figura transparente separada e composto por último (via _composite_overlay_box,
    # cache por (variavel, idx_dia)), garantindo que a caixa cinza e o cbar fiquem acima do jato,
    # estrelas e halo. `skip_overlay` (montagem do FUNDO cacheado) pula esta etapa.
    if not skip_overlay:
        arr = _composite_overlay_box(arr, ctx, idx_dia, data_full)

    return arr if as_float else arr.astype(np.uint8)


def _render_one_frame(f: int) -> np.ndarray:
    """Wrapper picklavel p/ o process pool (usa o contexto global do worker)."""
    return _build_frame(f, _FRAME_CTX)


def _taper_box(x: np.ndarray, lo: float, hi: float, pad: float) -> np.ndarray:
    """Peso 1D (cosseno, sem corte reto): 1.0 dentro de [lo, hi], cai suavemente ate 0.0 em
    [lo-pad, lo] e [hi, hi+pad], 0.0 fora disso. Mesma familia do `_taper_lat` (overlay de OLR)."""
    w = np.zeros_like(x, dtype=np.float32)
    dentro = (x >= lo) & (x <= hi)
    esquerda = (x < lo) & (x >= lo - pad)
    direita = (x > hi) & (x <= hi + pad)
    w[dentro] = 1.0
    if pad > 0:
        w[esquerda] = 0.5 - 0.5 * np.cos(np.pi * (x[esquerda] - (lo - pad)) / pad)
        w[direita] = 0.5 - 0.5 * np.cos(np.pi * ((hi + pad) - x[direita]) / pad)
    return w


def _suaviza_regiao_hgt(field_cyc: np.ndarray, lon_cyc: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Suaviza o campo-guia do jato (Z250/Z500 absoluto) SO dentro de uma caixa geografica (ex.:
    Cordilheira dos Andes) — as ondas de gravidade orograficas perturbam a isolinha do jato bem
    onde o escoamento cruza a cordilheira; suavizar o campo INTEIRO (GLOBO_3D_SIGMA_HGT_CONTORNOS)
    tira nitidez do jato em regioes sem essa perturbacao. Aqui a suavizacao (sigma proprio, mais
    forte) e misturada com o campo original via mascara 2D com transicao em cosseno (sem corte reto
    na borda da caixa) — no-op se `GLOBO_3D_SUAVIZACAO_REGIONAL_HGT` estiver desligada (default)."""
    if not bool(settings.get('GLOBO_3D_SUAVIZACAO_REGIONAL_HGT', False)):
        return field_cyc
    sigma = float(settings.get('GLOBO_3D_SUAVIZACAO_REGIONAL_HGT_SIGMA', 4.0))
    if sigma <= 0:
        return field_cyc
    lon_min = float(settings.get('GLOBO_3D_SUAVIZACAO_REGIONAL_HGT_LON_MIN', -80.0))
    lon_max = float(settings.get('GLOBO_3D_SUAVIZACAO_REGIONAL_HGT_LON_MAX', -60.0))
    lat_min = float(settings.get('GLOBO_3D_SUAVIZACAO_REGIONAL_HGT_LAT_MIN', -56.0))
    lat_max = float(settings.get('GLOBO_3D_SUAVIZACAO_REGIONAL_HGT_LAT_MAX', 12.0))
    pad = float(settings.get('GLOBO_3D_SUAVIZACAO_REGIONAL_HGT_PAD', 8.0))

    lon_norm = ((lon_cyc + 180.0) % 360.0) - 180.0   # -180..180, p/ casar com lon_min/max negativos
    mask2d = (_taper_box(lon_norm, lon_min, lon_max, pad)[None, :]
              * _taper_box(lat, lat_min, lat_max, pad)[:, None]).astype(np.float32)
    if not np.any(mask2d > 0):
        return field_cyc

    out = np.empty_like(field_cyc)
    for t in range(field_cyc.shape[0]):
        suave = _gaussian_filter(field_cyc[t], sigma=sigma, mode=['reflect', 'wrap'])
        out[t] = mask2d * suave + (1.0 - mask2d) * field_cyc[t]
    return out.astype(np.float32)


def _save_gif_paleta_unica(frames: list[np.ndarray], out_path: Path, fps: int) -> None:
    """Salva um GIF com UMA UNICA paleta de 256 cores compartilhada entre TODOS os frames -- em
    vez do comportamento padrao do Pillow/imageio (usado por `imageio.mimsave(format='GIF')`), que
    quantiza CADA frame de forma independente. Isso faz o MESMO valor RGB, em frames diferentes,
    mapear pra indices de paleta ligeiramente diferentes -> "flicker" de cor visivel sobretudo onde
    ha blend complexo (ex.: a faixa translucida do jato cruzando o shaded/continente, onde a
    quantidade de cores UNICAS por frame e maior e mais instavel de um frame pro outro -- inclusive
    pode fazer trechos da faixa "sumirem"/mudarem de cor feio quando a cor exata do blend nao esta
    bem representada na paleta daquele frame especifico).

    A paleta e construida a partir de uma AMOSTRA de TODOS os frames combinados (reduzidos e
    empilhados num mosaico) -- nao so o primeiro frame -- pra cobrir bem as cores que só aparecem
    em frames do meio/fim (ex.: jato numa posicao diferente, cruzando outra parte do continente)."""
    pil_frames = [_PIL_Image.fromarray(fr) for fr in frames]
    amostras = [f.resize((max(f.width // 4, 1), max(f.height // 4, 1)), _PIL_Image.BILINEAR)
                for f in pil_frames]
    largura = amostras[0].width
    altura_total = sum(a.height for a in amostras)
    mosaico = _PIL_Image.new('RGB', (largura, altura_total))
    _y = 0
    for _a in amostras:
        mosaico.paste(_a, (0, _y))
        _y += _a.height
    paleta_img = mosaico.quantize(colors=256, method=_PIL_Image.MEDIANCUT)
    quantizados = [f.quantize(palette=paleta_img, dither=_PIL_Image.FLOYDSTEINBERG) for f in pil_frames]
    quantizados[0].save(str(out_path), format='GIF', save_all=True,
                        append_images=quantizados[1:], duration=int(round(1000 / max(fps, 1))),
                        loop=0)


# ---------------------------------------------------------------------------
# Renderizacao de um clipe MP4 a partir de uma serie diaria de anomalia
# ---------------------------------------------------------------------------
def _render_clip(anom: xr.DataArray, ficha: dict, variavel_key: str,
                 output_dir: Path, fonte_label: str, script_id: str = 's38',
                 hgt_anom_serie: xr.DataArray | None = None,
                 mslp_serie: xr.DataArray | None = None,
                 olr_serie: xr.DataArray | None = None,
                 contour_serie: xr.DataArray | None = None,
                 estatico: bool = False,
                 png_path: Path | None = None,
                 camera: tuple[float, float] | None = None,
                 gif: bool = False,
                 gif_path: Path | None = None,
                 hgt_anom_serie_500: xr.DataArray | None = None,
                 neve_serie: xr.DataArray | None = None) -> Path:
    """Renderiza uma serie diaria como MP4 (voo da camera + evolucao temporal) OU, quando
    `estatico=True`, como UMA figura PNG (`png_path`) de um unico campo, com a camera fixa em
    `camera` (lon, lat) — usada pelo s40 (figuras estaticas). No modo estatico, `anom` deve ter
    um unico passo de tempo (o campo ja agregado: dia, media movel, pentada ou media total).

    Com `gif=True` (`gif_path`): campo ESTATICO (media do periodo, 1 passo de tempo) + camera fixa,
    mas renderiza VARIOS frames onde SO o overlay da corrente de jato (setas + 'JET STREAM') desliza
    W->E -> salva um GIF em loop. Com GLOBO_3D_JATO ligado, o jato aparece nas TRES saidas: FLUINDO no
    MP4 (fase = 1/fps), PARADO no PNG (`estatico`, fase 0) e ANIMADO no GIF (fase = 1/total_frames)."""
    # frames/fps/velocidade: o s41 pode regular a animacao via GLOBO_3D_GE_* (fallback ao s39).
    frames_por_dia = int(_script_setting(script_id, 'FRAMES_POR_DIA', 4))
    fps = int(_script_setting(script_id, 'FPS', 20))
    coarsen = int(getattr(settings, 'GLOBO_3D_COARSEN', 1))
    # Janela da pentada movel (0/1 = diario) — capturada ANTES do coarsen, que descarta attrs.
    _pent_dias = int(anom.attrs.get('pentada_dias', 0))
    # ACUMULAR_NO_TEMPO: cada frame e o total DESDE A DATA INICIAL ate aquele dia (running total), entao
    # rotular com a data unica do dia MENTE -- o campo do 3o frame nao e a chuva do dia 19, e a soma
    # 17→19. Aqui a data vira uma JANELA QUE CRESCE, ancorada no 1o dia ('Jul 17' -> 'Jul 17–18' -> ...).
    # So no MP4: a figura estatica ja recebe o intervalo FECHADO do periodo via `pentada_dias`
    # (_agg_estatico), e uma pentada movel tem sua propria janela -- nos dois casos, nao mexer.
    _acum_rotulo = (bool(anom.attrs.get('acumulado_no_tempo'))
                    and _pent_dias <= 1 and anom.sizes.get('time', 0) > 1)
    # Rotulo com hora UTC (12Z etc.) so pro MP4 sinotico ou de passo nativo (dias<=1 -- pentadas/
    # medias de periodo usam intervalo de datas, a hora nao se aplica ali). `precip_abs` entra aqui
    # quando ACUM_HORARIO=true (settings) -- a serie ja chega no passo nativo nesse caso.
    _mostrar_hora = bool(ficha.get('sinotico_mp4', False) or ficha.get('passo_nativo_mp4', False)
                         or (variavel_key == 'precip_abs' and bool(settings.get('ACUM_HORARIO', False))))

    if coarsen and coarsen > 1:
        anom = anom.coarsen(lat=coarsen, lon=coarsen, boundary='trim').mean()

    lat = anom['lat'].values
    # Suavizacao gaussiana opcional por variavel (ex.: GLOBO_3D_SIGMA_OLR_ANOM = 1.5). Override POR
    # SCRIPT (GLOBO_3D_SIGMA_<VAR>_<SCRIPT>, ex.: _S42) tem precedencia sobre a global -> o s42 pode
    # suavizar sem afetar s38/s39. Aplicada antes do add_cyclic_point com mode='wrap' em lon.
    _sigma_var = float(settings.get(f'GLOBO_3D_SIGMA_{variavel_key.upper()}_{script_id.upper()}',
                                    settings.get(f'GLOBO_3D_SIGMA_{variavel_key.upper()}', 0.0)))
    _anom_vals = anom.values
    if _sigma_var > 0:
        _anom_vals = np.stack([
            _gaussian_filter(_anom_vals[t], sigma=_sigma_var, mode=['reflect', 'wrap'])
            for t in range(_anom_vals.shape[0])
        ])
    vals_cyc, lon_cyc = add_cyclic_point(_anom_vals, coord=anom['lon'].values)
    dates = pd.DatetimeIndex(pd.to_datetime(anom['time'].values))
    n_dias = vals_cyc.shape[0]
    # Dias de CALENDARIO reais cobertos por `dates` -- igual a `n_dias` na serie DIARIA de sempre (1
    # passo/dia), mas diferente com ACUM_HORARIO (varios passos NATIVOS/dia): `n_dias` ali e o Nº DE
    # PASSOS (precisa continuar assim p/ indexar `vals_cyc`/overlays, ver `pos` em `_build_frame`),
    # `n_dias_janela` e o Nº DE DIAS de verdade (usado so p/ o voo da camera nao esticar o MP4 como
    # se a janela real (ex.: 5 dias) tivesse um dia por passo nativo (ex.: 33) -- ver `_dias_janela`).
    n_dias_janela = _dias_janela(dates[0], dates[-1]) if len(dates) else 1

    # Valor DIARIO do box do Niño 3.4 (170°W–120°W = 190–240° em 0..360, 5°S–5°N), media da area
    # ponderada por cos(lat) e ignorando NaN (continente). So p/ fichas com nino34_valor (tsm_abs);
    # vira o rotulo dinamico "Niño 3.4 = xx.x°C" por frame. Calculado da propria serie plotada.
    nino34_serie: np.ndarray | None = None
    if ficha.get('nino34_valor'):
        _box = anom.sel(lat=slice(-5, 5), lon=slice(190, 240))
        if _box.sizes.get('lat', 0) and _box.sizes.get('lon', 0):
            _wlat = np.cos(np.deg2rad(_box['lat']))
            nino34_serie = _box.weighted(_wlat).mean(dim=('lat', 'lon'), skipna=True).values.astype(float)
            logger.info('Niño 3.4 (box diario) {}: {} valores | {:.1f}..{:.1f}°C',
                        variavel_key, nino34_serie.size,
                        float(np.nanmin(nino34_serie)), float(np.nanmax(nino34_serie)))

    # Série de Z250 anomalia para isolinhas no campo absoluto (jet_stream):
    # quando fornecida, permite mostrar Z250 absoluto real (anom + clim) em vez da clim sozinha.
    hgt_anom_vals_cyc: np.ndarray | None = None
    if hgt_anom_serie is not None:
        _ha = hgt_anom_serie.sel(time=anom['time'].values, method='nearest')
        if coarsen and coarsen > 1:
            _ha = _ha.coarsen(lat=coarsen, lon=coarsen, boundary='trim').mean()
        _ha_cyc, _ = add_cyclic_point(_ha.values, coord=_ha['lon'].values)
        hgt_anom_vals_cyc = _ha_cyc.astype(np.float32)

    # Climatologia absoluta para variaveis com isolinha de temperatura absoluta (ex.: 0°C)
    clim_abs_cyc = None
    if ficha.get('isolinha_abs_0') and bool(settings.get('GLOBO_3D_ISOTERMA_0C', True)):
        _spec = ficha['spec']
        _cl_raw, _cl_lat, _cl_lon = _spec['clim_fn'](anom['time'].values)
        _cl_da = xr.DataArray(
            _cl_raw, dims=['time', 'lat', 'lon'],
            coords={'time': anom['time'].values, 'lat': _cl_lat, 'lon': _cl_lon},
        ).sortby('lat')
        _cl_cyc_v, _cl_cyc_l = add_cyclic_point(_cl_da.values, coord=_cl_da['lon'].values)
        _cl_da = xr.DataArray(
            _cl_cyc_v, dims=['time', 'lat', 'lon'],
            coords={'time': _cl_da['time'].values, 'lat': _cl_da['lat'].values, 'lon': _cl_cyc_l},
        ).interp(lat=anom['lat'].values, lon=anom['lon'].values, method='linear')
        if _spec.get('celsius', False):
            _cl_da = _to_celsius_da(_cl_da)
        _cl_cyc_v2, _ = add_cyclic_point(_cl_da.values, coord=_cl_da['lon'].values)
        clim_abs_cyc = _cl_cyc_v2.astype(np.float32)
        logger.info('Climatologia absoluta p/ isolinha 0°C ({}): shape {}',
                    _spec['nome'], clim_abs_cyc.shape)

    # Climatologia Z250 para isolinhas de geopotencial.
    # Tudo (whitesmoke + fixas coloridas) controlado pela mesma flag por variavel.
    hgt_abs_cyc = None
    hgt_abs_levels: np.ndarray | None = None
    hgt_z250_abs_cyc: np.ndarray | None = None  # anom + clim = Z250 absoluto real
    # Campo-guia DEDICADO do jato quando ele precisa de Z500 (GLOBO_3D_JET_STREAM_NIVEL/
    # SUBTROPICAL_JET_NIVEL na escala de Z500, ver _jato_kind_from_nivel). Sem relacao com as
    # isolinhas Z250 acima (essas continuam Z250-only, feature separada).
    hgt_z500_abs_cyc: np.ndarray | None = None
    _isol_hgt_flag = ('GLOBO_3D_ISOL_HGT_JET_STREAM' if variavel_key == 'jet_stream'
                      else 'GLOBO_3D_ISOL_HGT250_ABS')
    _isol_flag_on = bool(settings.get(_isol_hgt_flag, False))
    # Correntes de jato (s38/s39/s40/s41 — master unico GLOBO_3D_JATO): se o master estiver ligado, o
    # jato e plotado em QUALQUER campo atmosferico e em TODAS as saidas. Precisa do Z250 absoluto (anom de z250 +
    # clim de Z250), baixado a parte para localizar a isolinha-guia — independe da variavel shaded.
    # Dois jatos independentes: JET_STREAM (azul) + SUBTROPICAL_JET (verde). _jato_on = master E >=1 jato
    # E >=1 hemisferio. HEMISFERIO_SUL/NORTE (default ambos true) escolhem onde plotar: ao olhar o HS,
    # desligue o HN p/ nao desenhar o jato do outro hemisferio (que apareceria no limbo/topo do globo).
    _jato_master = bool(_script_setting(script_id, 'JATO', False))
    _jet1_on = bool(_script_setting(script_id, 'JET_STREAM', True))
    _jet2_on = bool(_script_setting(script_id, 'SUBTROPICAL_JET', False))
    _hem_sul = bool(_script_setting(script_id, 'JATO_HEMISFERIO_SUL', True))
    _hem_norte = bool(_script_setting(script_id, 'JATO_HEMISFERIO_NORTE', True))
    _hemisferios = tuple(h for h, on in (('N', _hem_norte), ('S', _hem_sul)) if on)
    _jato_on = _jato_master and (_jet1_on or _jet2_on) and bool(_hemisferios)
    _flag_whitesmoke = ficha.get('isolinha_hgt_abs') and _isol_flag_on
    _need_hgt_clim = _isol_flag_on and (
        bool(ficha.get('isolinha_hgt_abs')) or bool(ficha.get('isolinhas_fixas_hgt'))
    )
    if _need_hgt_clim:
        _spec_h = ficha['spec']
        _hgt_clim_fn = _spec_h.get('hgt_clim_fn', _spec_h.get('clim_fn'))
        if _hgt_clim_fn is None:
            logger.warning('isolinha hgt ativa mas sem clim_fn no spec de {}', variavel_key)
        else:
            _ch_raw, _ch_lat, _ch_lon = _hgt_clim_fn(anom['time'].values)
            _ch_da = xr.DataArray(
                _ch_raw, dims=['time', 'lat', 'lon'],
                coords={'time': anom['time'].values, 'lat': _ch_lat, 'lon': _ch_lon},
            ).sortby('lat')
            _ch_cyc_v, _ch_cyc_l = add_cyclic_point(_ch_da.values, coord=_ch_da['lon'].values)
            _ch_da = xr.DataArray(
                _ch_cyc_v, dims=['time', 'lat', 'lon'],
                coords={'time': _ch_da['time'].values, 'lat': _ch_da['lat'].values, 'lon': _ch_cyc_l},
            ).interp(lat=anom['lat'].values, lon=anom['lon'].values, method='linear')
            _ch_cyc_v2, _ = add_cyclic_point(_ch_da.values, coord=_ch_da['lon'].values)
            hgt_abs_cyc = _ch_cyc_v2.astype(np.float32)
            # Z250 absoluto real = anom + clim (para isolinhas fixas coloridas)
            if hgt_anom_vals_cyc is not None:
                hgt_z250_abs_cyc = (hgt_anom_vals_cyc + hgt_abs_cyc).astype(np.float32)
            # Suavizacao gaussiana das isolinhas Z250 — mesmo padrao de GLOBO_3D_MSLP_SIGMA.
            # Sem isso, ax.contour em grade 0.5° produz degraus visiveis (serrilhado).
            _sigma_hgt = float(settings.get('GLOBO_3D_SIGMA_HGT_CONTORNOS', 1.5))
            if _sigma_hgt > 0:
                hgt_abs_cyc = np.stack([
                    _gaussian_filter(hgt_abs_cyc[t], sigma=_sigma_hgt, mode=['reflect', 'wrap'])
                    for t in range(hgt_abs_cyc.shape[0])
                ]).astype(np.float32)
                if hgt_z250_abs_cyc is not None:
                    hgt_z250_abs_cyc = np.stack([
                        _gaussian_filter(hgt_z250_abs_cyc[t], sigma=_sigma_hgt, mode=['reflect', 'wrap'])
                        for t in range(hgt_z250_abs_cyc.shape[0])
                    ]).astype(np.float32)
            if hgt_z250_abs_cyc is not None:
                hgt_z250_abs_cyc = _suaviza_regiao_hgt(hgt_z250_abs_cyc, lon_cyc, lat)
            # Niveis para isolinhas whitesmoke (somente se flag ativa)
            if _flag_whitesmoke:
                _intv = float(settings.get('GLOBO_3D_ISOL_HGT250_INTERVALO', 60))
                _hgt_full = (
                    hgt_z250_abs_cyc if (ficha.get('absoluto') and hgt_z250_abs_cyc is not None)
                    else vals_cyc + hgt_abs_cyc
                )
                _hmin = float(np.nanmin(_hgt_full))
                _hmax = float(np.nanmax(_hgt_full))
                hgt_abs_levels = np.arange(
                    np.floor(_hmin / _intv) * _intv,
                    np.ceil(_hmax / _intv) * _intv + _intv,
                    _intv,
                )
                logger.info('Z250 isolinhas {}: intervalo {} mgp | {} niveis',
                            variavel_key, _intv, len(hgt_abs_levels))

    # Campo-guia do jato = O PROPRIO campo shaded, quando a variavel JA e uma altura geopotencial
    # absoluta (kind='hgt250' ou 'hgt500' -- ex.: z250_abs, z500_abs) -- evita a reconstrucao
    # anomalia+climatologia (bloco abaixo) e o download extra (gerar_animacao ja pula esse
    # download ao detectar esta ficha). Mesma suavizacao gaussiana do caminho normal, aplicada
    # por time-step (funciona igual em daily ou sinotico). Grava no campo do PROPRIO kind da
    # ficha (hgt250 -> hgt_z250_abs_cyc; hgt500 -> hgt_z500_abs_cyc) -- os jatos que precisarem
    # do OUTRO kind ainda caem no bloco de download dedicado abaixo.
    _self_kind = ficha['spec'].get('kind') if (bool(ficha.get('absoluto'))
                                               and ficha['spec'].get('kind') in ('hgt250', 'hgt500')) else None
    if _jato_on and _self_kind is not None:
        _self_field = vals_cyc.copy()
        _sig = float(settings.get('GLOBO_3D_SIGMA_HGT_CONTORNOS', 1.5))
        if _sig > 0:
            _self_field = np.stack([
                _gaussian_filter(_self_field[t], sigma=_sig, mode=['reflect', 'wrap'])
                for t in range(_self_field.shape[0])
            ]).astype(np.float32)
        _self_field = _suaviza_regiao_hgt(_self_field, lon_cyc, lat)
        if _self_kind == 'hgt250' and hgt_z250_abs_cyc is None:
            hgt_z250_abs_cyc = _self_field
        elif _self_kind == 'hgt500' and hgt_z500_abs_cyc is None:
            hgt_z500_abs_cyc = _self_field

    # Z250/Z500 absoluto DEDICADO (independe da variavel shaded): anomalia + climatologia, AMBAS
    # reamostradas para a grade do campo shaded (`anom`) — pois o campo pode estar numa grade
    # diferente da do HGT. Usado pelos JATOS (GLOBO_3D_JATO, um campo por kind conforme o NIVEL
    # de cada jato -- ver _jato_kind_from_nivel) E pelas ISOLINHAS de Z250 absoluto (GLOBO_3D_ISOL_
    # HGT250_ABS, sempre Z250). Fichas nativas de Z250 (z250_anom/jet_stream) ja montaram
    # hgt_z250_abs_cyc/hgt_abs_cyc acima (via clim propria) e nao entram aqui. Se o modelo nao
    # tiver o nivel, `hgt_anom_serie`/`hgt_anom_serie_500` vem None (aviso em gerar_animacao).
    def _para_grade_anom(_da: xr.DataArray) -> np.ndarray:
        """(time,lat,lon) na grade propria -> grade do campo `anom` (sem cyclic)."""
        _d = _da.sortby('lat')
        _v, _l = add_cyclic_point(_d.values, coord=_d['lon'].values)
        _d = xr.DataArray(_v, dims=['time', 'lat', 'lon'],
                          coords={'time': _d['time'].values, 'lat': _d['lat'].values, 'lon': _l})
        return _d.interp(lat=anom['lat'].values, lon=anom['lon'].values, method='linear').values

    if (_jato_on or _isol_flag_on) and hgt_anom_serie is not None and hgt_z250_abs_cyc is None:
        _ha = hgt_anom_serie.sel(time=anom['time'].values, method='nearest')
        _ha_grid = _para_grade_anom(_ha)
        _zc_raw, _zc_lat, _zc_lon = clim_hgt250_daily(anom['time'].values)
        _zc_grid = _para_grade_anom(xr.DataArray(
            _zc_raw, dims=['time', 'lat', 'lon'],
            coords={'time': anom['time'].values, 'lat': _zc_lat, 'lon': _zc_lon}))
        _z250abs, _ = add_cyclic_point(_ha_grid + _zc_grid, coord=anom['lon'].values)
        hgt_z250_abs_cyc = _z250abs.astype(np.float32)   # casa com (lon_cyc, lat)
        _sig = float(settings.get('GLOBO_3D_SIGMA_HGT_CONTORNOS', 1.5))
        if _sig > 0:
            hgt_z250_abs_cyc = np.stack([
                _gaussian_filter(hgt_z250_abs_cyc[t], sigma=_sig, mode=['reflect', 'wrap'])
                for t in range(hgt_z250_abs_cyc.shape[0])
            ]).astype(np.float32)
        hgt_z250_abs_cyc = _suaviza_regiao_hgt(hgt_z250_abs_cyc, lon_cyc, lat)

    if _jato_on and hgt_anom_serie_500 is not None and hgt_z500_abs_cyc is None:
        _ha5 = hgt_anom_serie_500.sel(time=anom['time'].values, method='nearest')
        _ha5_grid = _para_grade_anom(_ha5)
        _zc5_raw, _zc5_lat, _zc5_lon = clim_hgt500_daily(anom['time'].values)
        _zc5_grid = _para_grade_anom(xr.DataArray(
            _zc5_raw, dims=['time', 'lat', 'lon'],
            coords={'time': anom['time'].values, 'lat': _zc5_lat, 'lon': _zc5_lon}))
        _z500abs, _ = add_cyclic_point(_ha5_grid + _zc5_grid, coord=anom['lon'].values)
        hgt_z500_abs_cyc = _z500abs.astype(np.float32)
        _sig5 = float(settings.get('GLOBO_3D_SIGMA_HGT_CONTORNOS', 1.5))
        if _sig5 > 0:
            hgt_z500_abs_cyc = np.stack([
                _gaussian_filter(hgt_z500_abs_cyc[t], sigma=_sig5, mode=['reflect', 'wrap'])
                for t in range(hgt_z500_abs_cyc.shape[0])
            ]).astype(np.float32)
        hgt_z500_abs_cyc = _suaviza_regiao_hgt(hgt_z500_abs_cyc, lon_cyc, lat)

    # Niveis das ISOLINHAS de Z250 absoluto sobre um campo ESTRANHO (ex.: tmp850_anom, olr_anom): so
    # quando a flag esta ligada, a ficha NAO e uma das nativas de Z250 (que ja montaram hgt_abs_levels
    # via caminho clim proprio) e ha Z250 reconstruido acima. Traca direto de hgt_z250_abs_cyc (o campo
    # shaded nao entra na conta — as isolinhas sao de geopotencial, nao da variavel de fundo).
    hgt_z250_iso_levels: np.ndarray | None = None
    if (_isol_flag_on and hgt_abs_levels is None and hgt_z250_abs_cyc is not None
            and not ficha.get('isolinha_hgt_abs')):
        _intv = float(settings.get('GLOBO_3D_ISOL_HGT250_INTERVALO', 60))
        _hmin = float(np.nanmin(hgt_z250_abs_cyc))
        _hmax = float(np.nanmax(hgt_z250_abs_cyc))
        hgt_z250_iso_levels = np.arange(
            np.floor(_hmin / _intv) * _intv,
            np.ceil(_hmax / _intv) * _intv + _intv,
            _intv,
        )
        logger.info('Z250 isolinhas absolutas sobre {}: intervalo {} mgp | {} niveis',
                    variavel_key, _intv, len(hgt_z250_iso_levels))

    # MSLP isolinhas (só para variáveis com isolinha_mslp=True, ex.: tmp850_mslp)
    mslp_cyc: np.ndarray | None = None
    mslp_levels: np.ndarray | None = None
    if mslp_serie is not None and _isolinha_mslp_ativa(ficha, variavel_key):
        _ms = mslp_serie.sel(time=anom['time'].values, method='nearest')
        if coarsen and coarsen > 1:
            _ms = _ms.coarsen(lat=coarsen, lon=coarsen, boundary='trim').mean()
        _ms_vals = _ms.values
        _sigma_mslp = float(settings.get('GLOBO_3D_MSLP_SIGMA', 2.0))
        if _sigma_mslp > 0:
            # mode=['reflect','wrap']: lat reflete nos polos, lon envolve em 0°/360° sem seam
            _ms_vals = np.stack([_gaussian_filter(_ms_vals[t], sigma=_sigma_mslp,
                                                  mode=['reflect', 'wrap'])
                                 for t in range(_ms_vals.shape[0])])
        _ms_cyc, _ = add_cyclic_point(_ms_vals, coord=_ms['lon'].values)
        mslp_cyc = _ms_cyc.astype(np.float32)
        _intv_mslp = float(settings.get('GLOBO_3D_MSLP_INTERVALO', 4.0))
        mslp_levels = np.arange(
            np.floor(np.nanmin(mslp_cyc) / _intv_mslp) * _intv_mslp,
            np.ceil(np.nanmax(mslp_cyc) / _intv_mslp) * _intv_mslp + _intv_mslp,
            _intv_mslp,
        )
        logger.info('MSLP isolinhas: intervalo {} hPa | {} niveis', _intv_mslp, len(mslp_levels))

    # Serie AUXILIAR plotada como ISOLINHAS (nao shaded) — ex.: psi200 preto sobre chi200 shaded.
    # Alinhada por tempo com o shaded (mesma pentada/dias) e desenhada num intervalo fixo.
    contour_cyc: np.ndarray | None = None
    contour_levels: np.ndarray | None = None
    # Grade PROPRIA do contorno: a serie auxiliar (ex.: psi200 em 2.5°) pode estar numa grade
    # DIFERENTE do shaded principal (ex.: OLR em 0.5°). Precisamos do lon/lat dela p/ o ax.contour
    # — usar o lon_cyc/lat do principal quebra ("Length of x must match columns in z").
    contour_lon: np.ndarray | None = None
    contour_lat: np.ndarray | None = None
    if contour_serie is not None and ficha.get('contorno_serie_var'):
        _cs = contour_serie.sel(time=anom['time'].values, method='nearest')
        if coarsen and coarsen > 1:
            _cs = _cs.coarsen(lat=coarsen, lon=coarsen, boundary='trim').mean()
        _cs_vals = _cs.values
        _sigma_cs = float(settings.get('GLOBO_3D_CONTORNO_SERIE_SIGMA', 1.0))
        if _sigma_cs > 0:
            _cs_vals = np.stack([_gaussian_filter(_cs_vals[t], sigma=_sigma_cs,
                                                  mode=['reflect', 'wrap'])
                                 for t in range(_cs_vals.shape[0])])
        _cs_cyc, _cs_loncyc = add_cyclic_point(_cs_vals, coord=_cs['lon'].values)
        contour_cyc = _cs_cyc.astype(np.float32)
        contour_lon = np.asarray(_cs_loncyc)
        contour_lat = _cs['lat'].values
        _intv_cs = float(settings.get('GLOBO_3D_CONTORNO_SERIE_INTERVALO',
                                      ficha.get('contorno_serie_intervalo', 10.0)))
        contour_levels = np.arange(
            np.floor(np.nanmin(contour_cyc) / _intv_cs) * _intv_cs,
            np.ceil(np.nanmax(contour_cyc) / _intv_cs) * _intv_cs + _intv_cs,
            _intv_cs,
        )
        logger.info('Isolinhas auxiliares ({}): intervalo {} | {} niveis',
                    ficha.get('contorno_serie_var'), _intv_cs, len(contour_levels))

    # Escala de cores fixa (estavel durante todo o clipe). Override POR SCRIPT
    # (GLOBO_3D_VMAX_<VAR>_<SCRIPT>, ex.: _S42) tem precedencia sobre a ficha/global -> o s42 usa
    # ±10 °C (TWC) sem afetar o s38/s39 (que ficam com o ±15 °C da ficha). Mesmo padrao do sigma.
    _vmax_ov = settings.get(f'GLOBO_3D_VMAX_{variavel_key.upper()}_{script_id.upper()}', None)
    vmax = float(_vmax_ov) if _vmax_ov is not None else ficha.get('vmax')
    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(vals_cyc), 98))
        vmax = max(10.0, round(vmax / 10.0) * 10.0)
    if ficha.get('vmin') is not None:
        vmin = float(ficha['vmin'])
    elif ficha.get('simetrico', True):
        vmin = -vmax
    else:
        vmin = float(np.nanmin(vals_cyc))
    # Paleta POR VARIAVEL: settings GLOBO_3D_PALETA_<VAR> sobrescreve a da ficha. Override POR SCRIPT
    # (GLOBO_3D_PALETA_<VAR>_<SCRIPT>, ex.: _S42) tem precedencia sobre a global -> a paleta TWC do
    # tmp850_anom fica SO no s42; s38/s39 mantem a paleta original da ficha. Mesmo padrao do sigma.
    override = (settings.get(f'GLOBO_3D_PALETA_{variavel_key.upper()}_{script_id.upper()}', None)
                or settings.get(f'GLOBO_3D_PALETA_{variavel_key.upper()}', None))
    paleta = override or ficha['cmap_colors']
    # Nº de bandas POR VARIAVEL: settings GLOBO_3D_NIVEIS_<VAR> > ficha['niveis'] >
    # GLOBO_3D_NIVEIS global. Mais niveis = shaded mais suave (degrade mais fino).
    # NIVEIS EXPLICITOS (ficha['levels']): grade possivelmente NAO-UNIFORME (ex.: tsm_anom usa
    # LST_SSTA_NEW_GREC do s11, com refino de ±0.2 no zero). Override de GLOBO_3D_NIVEIS_<VAR>
    # tem prioridade (volta p/ grade uniforme). Sem nenhum dos dois -> linspace uniforme.
    # RE-LIDO aqui (nao so da ficha): o `levels` da ficha e congelado no IMPORT do modulo, entao a
    # escala posta no header do script era ignorada em silencio. Precedencia igual a da paleta:
    # _<VAR>_<SCRIPT> > _<VAR> > ficha.
    explicit_levels = settings.get(f'GLOBO_3D_LEVELS_{variavel_key.upper()}_{script_id.upper()}',
                                   settings.get(f'GLOBO_3D_LEVELS_{variavel_key.upper()}',
                                                ficha.get('levels')))
    explicit_levels = list(explicit_levels) if explicit_levels is not None else None
    if explicit_levels is not None and not settings.get(f'GLOBO_3D_NIVEIS_{variavel_key.upper()}', None):
        levels = np.asarray(explicit_levels, dtype=float)
        niveis = len(levels) - 1
        vmin, vmax = float(levels[0]), float(levels[-1])
    else:
        niveis = (settings.get(f'GLOBO_3D_NIVEIS_{variavel_key.upper()}', None)
                  or ficha.get('niveis')
                  or getattr(settings, 'GLOBO_3D_NIVEIS', 16))
        niveis = int(niveis)
        levels = np.linspace(vmin, vmax, niveis + 1)
    logger.info('Escala {}: shaded de {:+.1f} a {:+.1f} {} | {} bandas (passo medio {:.2f})',
                variavel_key, vmin, vmax, ficha.get('unidade', ''), niveis,
                (vmax - vmin) / niveis)
    # Modo do pcolormesh por variavel (so vale quando usar_pcolormesh=true): 'gouraud' = gradiente
    # CONTINUO liso (default); 'nearest' = BANDAS DISCRETAS chapadas (BoundaryNorm) SEM costura de
    # contourf. Ex.: pwat_abs_alicia usa 'nearest' p/ os degraus de 5 mm. Chaveado por variavel (igual
    # GLOBO_3D_SIGMA_<VAR>), nao mexe nas demais.
    _pcm_shading = str(settings.get(f'GLOBO_3D_PCOLORMESH_SHADING_{variavel_key.upper()}', 'gouraud')).lower()

    # Colormap e norm construidos 1x aqui (no pai) e herdados por todos os workers via CoW.
    # Antes eram recriados a cada frame (N_frames × from_list() = chamadas desnecessárias).
    # Remove alpha SO de hex #RRGGBBAA -> #RRGGBB. O `startswith('#')` evita cortar cores nomeadas
    # de 9 letras (ex.: 'limegreen') que virariam 'limegre' e quebrariam o matplotlib.
    paleta_opaca = [c[:7] if isinstance(c, str) and len(c) == 9 and c.startswith('#') else c
                    for c in paleta]
    if len(paleta) == niveis:
        # 1 cor por banda -> colormap DISCRETO exato (cada banda recebe a cor correspondente,
        # sem o blend do from_list). Fiel a colorbars amostradas pixel a pixel (ex.: tsm_anom).
        cmap_plot = ListedColormap(list(paleta))
        cmap_plot.set_under(paleta[0])
        cmap_plot.set_over(paleta[-1])
        cmap_legend = ListedColormap(paleta_opaca)
        norm_fn = BoundaryNorm(levels, ncolors=niveis)
    else:
        cmap_plot   = LinearSegmentedColormap.from_list('globo3d',        list(paleta))
        cmap_legend = LinearSegmentedColormap.from_list('globo3d_legend', paleta_opaca)
        # 'nearest' (bandas discretas) ou niveis nao-uniformes (explicit_levels) -> BoundaryNorm (cor
        # chapada por banda). No 'nearest', tudo ABAIXO de vmin fica TRANSPARENTE (set_under alpha 0)
        # -> nao pinta nada, revela o fundo do globo. Caso contrario (gouraud/contourf) -> Normalize.
        if explicit_levels is not None or _pcm_shading == 'nearest':
            norm_fn = BoundaryNorm(levels, ncolors=256)
            if _pcm_shading == 'nearest':
                cmap_plot.set_under((0.0, 0.0, 0.0, 0.0))
        else:
            norm_fn = plt.Normalize(vmin=vmin, vmax=vmax)

    # Base do FLUXO do jato: nº de frames p/ o padrao avancar 1 espacamento de palavra A VELOCIDADE 1
    # (GLOBO_3D_GE_GIF_FRAMES). Vale p/ o GIF (loop = base x m) E p/ o MP4 -> a velocidade do jato tem o
    # MESMO efeito por frame nas duas saidas. Antes o MP4 usava 1/fps (escala diferente do GIF) e a
    # velocidade quase nao alterava o video; agora GIF e MP4 compartilham esta base.
    _flow_base = max(2, int(_script_setting(script_id, 'GIF_FRAMES', 48)))
    _bg_from = None  # 1o frame de FUNDO CONGELADO (cache de fundo): GIF=0; MP4=inicio da cauda; None=sem cache
    if estatico:
        # Figura estatica: 1 frame do campo (unico passo de tempo), camera FIXA.
        total_frames = 1
        cam_lon, cam_lat = camera if camera is not None else (
            float(getattr(settings, 'GLOBO_3D_LON_FINAL', -45.0)),
            float(getattr(settings, 'GLOBO_3D_LAT_FINAL', -15.0)))
        lons, lats = np.array([cam_lon], dtype=float), np.array([cam_lat], dtype=float)
    elif gif:
        # GIF: campo estatico (media) + camera FIXA. So faz sentido animar quando ha jato (setas +
        # 'JET STREAM' deslizando); sem jato, 1 frame basta (GIF estatico = PNG) e evita renderizar
        # dezenas de frames identicos.
        # EMENDA PERFEITA com velocidades FRACIONARIAS: o loop precisa cobrir um numero INTEIRO de
        # espacamentos por jato, senao o fluxo "reinicia no meio" no fim do loop. `_flow_base` e a base
        # (fase = 1/base por frame, preserva a velocidade/suavidade); o loop e multiplicado por
        # `m` = _seamless_loop_multiplier(velocidades) para que cada jato avance m*v espacamentos
        # INTEIROS (ex.: v=0.5 -> m=2 -> 1 espacamento em 2x os frames = metade da velocidade, seamless).
        if _jato_on and (hgt_z250_abs_cyc is not None or hgt_z500_abs_cyc is not None):
            _gif_vels = []
            if _jet1_on:
                _gif_vels.append(float(_script_setting(script_id, 'JET_STREAM_VELOCIDADE', 1.0)))
            if _jet2_on:
                _gif_vels.append(float(_script_setting(script_id, 'SUBTROPICAL_JET_VELOCIDADE', 1.0)))
            _gif_mult = _seamless_loop_multiplier(_gif_vels)
            total_frames = _flow_base * _gif_mult
            _bg_from = 0   # GIF: TODOS os frames tem campo/camera congelados -> cache de fundo total
            if _gif_mult > 1:
                logger.info('GIF loop seamless: base {} x m={} = {} frames (velocidades {})',
                            _flow_base, _gif_mult, total_frames, _gif_vels)
        else:
            total_frames = 1
        cam_lon, cam_lat = camera if camera is not None else (
            float(getattr(settings, 'GLOBO_3D_LON_FINAL', -45.0)),
            float(getattr(settings, 'GLOBO_3D_LAT_FINAL', -15.0)))
        lons = np.full(total_frames, cam_lon, dtype=float)
        lats = np.full(total_frames, cam_lat, dtype=float)
    else:
        # `n_dias_janela` (dias de CALENDARIO reais), NAO `n_dias` (Nº de passos -- so bate com dias
        # de calendario na serie DIARIA de sempre; com ACUM_HORARIO inflava o voo da camera).
        base_frames = ((n_dias_janela - 1) * frames_por_dia + 1
                       if n_dias_janela > 1 else max(frames_por_dia, 1))
        if camera is not None:
            # Camera FIXA explicita (ex.: GLOBO_3D_MP4_MEDIA_FIXA) -- sem voo, mesmo ponto do
            # PNG/GIF em todos os frames. Sem isso, so o voo padrao (GLOBO_3D_GE_LON/LAT_*) valia.
            lons = np.full(base_frames, float(camera[0]), dtype=float)
            lats = np.full(base_frames, float(camera[1]), dtype=float)
        else:
            lons, lats = _camera_path(base_frames, script_id)
        # CAUDA do MP4 (GLOBO_3D_JATO_MP4_CAUDA_SEG, só quando o jato sera desenhado): depois que a
        # animacao da variavel termina, acrescenta N segundos de frames onde o CAMPO e a isolinha-guia
        # ficam CONGELADOS no ultimo dia (a interpolacao temporal ja clampa `pos` em n_dias-1) e a
        # camera para na posicao final — mas a FASE do jato (funcao do indice `f`) continua avancando,
        # entao 'JET STREAM'/setas seguem FLUINDO W->E sobre o campo parado. Assume que a variavel ja
        # chegou no ultimo dia ao fim do voo (verdade p/ VELOCIDADE_VAR >= 1). 0 = sem cauda.
        _cauda_seg = float(_script_setting(script_id, 'JATO_MP4_CAUDA_SEG', 7.0))
        _tail = int(round(_cauda_seg * fps)) if (_cauda_seg > 0 and _jato_on
                                                 and (hgt_z250_abs_cyc is not None
                                                      or hgt_z500_abs_cyc is not None)) else 0
        if _tail > 0:
            lons = np.concatenate([lons, np.full(_tail, lons[-1], dtype=float)])
            lats = np.concatenate([lats, np.full(_tail, lats[-1], dtype=float)])
            _bg_from = base_frames   # MP4: a partir da cauda o campo/camera congelam -> cache de fundo
            logger.info('MP4 cauda do jato: +{} frame(s) (~{:.0f}s) com campo congelado e jato fluindo',
                        _tail, _cauda_seg)
        total_frames = base_frames + _tail

    # Corrente de jato: desloca a costura da grade ciclica p/ longe de Greenwich se a camera
    # (fixa ou em voo, `lons` ja cobre o clipe inteiro aqui) chega perto dele (ver `_lon_seam_alvo`
    # no topo do arquivo). So vale a pena quando ha jato pra desenhar (unico consumidor afetado).
    if _jato_on and (hgt_z250_abs_cyc is not None or hgt_z500_abs_cyc is not None):
        _seam_alvo = _lon_seam_alvo(lons)
        if _seam_alvo:
            _k = _lon_seam_roll_index(lon_cyc, _seam_alvo)
            if _k:
                logger.info('Corrente de jato: costura da grade deslocada p/ ~{:.0f}°E '
                            '(camera perto do Meridiano de Greenwich)', _seam_alvo)
                vals_cyc = _lon_seam_roll(vals_cyc, _k)
                if hgt_anom_vals_cyc is not None:
                    hgt_anom_vals_cyc = _lon_seam_roll(hgt_anom_vals_cyc, _k)
                if clim_abs_cyc is not None:
                    clim_abs_cyc = _lon_seam_roll(clim_abs_cyc, _k)
                if hgt_abs_cyc is not None:
                    hgt_abs_cyc = _lon_seam_roll(hgt_abs_cyc, _k)
                if hgt_z250_abs_cyc is not None:
                    hgt_z250_abs_cyc = _lon_seam_roll(hgt_z250_abs_cyc, _k)
                if hgt_z500_abs_cyc is not None:
                    hgt_z500_abs_cyc = _lon_seam_roll(hgt_z500_abs_cyc, _k)
                if mslp_cyc is not None:
                    mslp_cyc = _lon_seam_roll(mslp_cyc, _k)
                if contour_cyc is not None and contour_lon is not None:
                    _k_ct = _lon_seam_roll_index(contour_lon, _seam_alvo)
                    if _k_ct:
                        contour_lon = _lon_seam_roll_lon(contour_lon, _k_ct)
                        contour_cyc = _lon_seam_roll(contour_cyc, _k_ct)
                lon_cyc = _lon_seam_roll_lon(lon_cyc, _k)

    # Fade-in sincronizado com o INICIO DA CAUDA (teste s42): jato + icones de pressao + caixa de
    # texto livre ficam INVISIVEIS durante o voo principal (campo ainda mudando) e aparecem
    # esmaecendo assim que o campo CONGELA (inicio da cauda), ficando visiveis e em movimento
    # pelos GLOBO_3D_FADE_CAUDA_DUR_SEG segundos seguintes. So faz sentido no MP4 com cauda.
    _fade_cauda_on = bool(script_id == 's42' and settings.get('GLOBO_3D_FADE_CAUDA', False)
                         and _bg_from is not None and not estatico and not gif)
    _fade_cauda_inicio = float(_bg_from) / total_frames if _fade_cauda_on else 0.0
    _fade_cauda_dur = ((float(settings.get('GLOBO_3D_FADE_CAUDA_DUR_SEG', 1.5)) * fps) / total_frames
                       if _fade_cauda_on else 0.0)
    # Caixa de texto livre FIXA (sem fade), sempre (s42) -- INDEPENDENTE de GLOBO_3D_FADE_CAUDA:
    # a caixa e um rotulo fixo, nao faz parte da animacao do jato/icones de pressao (essa sim
    # controlada por FADE_CAUDA via `_fade_cauda_on` abaixo), entao nunca deve esmaecer com eles,
    # nem quando FADE_CAUDA=false (jato/icones aparecendo direto, sem fade nenhum).
    _caixa_fixa = bool(script_id == 's42')
    vel_var = max(0.05, float(_script_setting(script_id, 'VELOCIDADE_VAR', 1.0)))

    # Resolucao de saida (px). dpi tal que 8in * dpi = px (figsize fixo 8).
    px = int(getattr(settings, 'GLOBO_3D_TAMANHO_PX', 1080))
    dpi = px / 8.0

    # Fontes: titulo e legenda (ex.: "Aptos Display" se o .ttf existir em Entrada/fonts).
    font_titulo = _resolve_family(getattr(settings, 'GLOBO_3D_FONTE_TITULO', ''), FONT_SERIF)
    font_legenda = _resolve_family(getattr(settings, 'GLOBO_3D_FONTE_LEGENDA', ''), FONT_SANS)
    # Fonte da caixa "The Weather Channel" (titulo azul + data) -- default herda font_legenda.
    # Reusada tambem pela caixa de texto LIVRE, pra ficarem visualmente consistentes.
    font_twc = _resolve_family(str(settings.get('GLOBO_3D_FONTE_TWC', '')), font_legenda)

    # Workers: GLOBO_3D_WORKERS (0 = auto = min(nucleos, 8)).
    # Cap original de 4 foi removido: o OOM vinha do pickle de ctx via initargs (~150 MB × N),
    # resolvido com fork+CoW (_FRAME_CTX global herdado sem cópia). Agora cada worker
    # aloca apenas ~20 MB de estado local por frame; 8 workers ≈ 160 MB adicionais.
    workers = int(getattr(settings, 'GLOBO_3D_WORKERS', 0)) or min(os.cpu_count() or 1, 8)
    workers = max(1, min(workers, total_frames))

    # Estilo de layout: s39/s41/s42 -> 'guillaume' (caixa do nome + barra de gradiente); demais -> WaPo.
    # (s41 = copia fiel do s39, muda so a projecao; s42 = copia fiel do s41, ponto de partida.)
    estilo = 'guillaume' if script_id in ('s39', 's41', 's42', 's44', 's46', 's47') else 'wapo'

    # Projecao do globo. s38/s39 usam GLOBO_3D_PROJECTION (default 'nearside'); o s41/s42 usam a
    # projecao "Google Earth" (NearsidePerspective com camera mais perto = zoom/curvatura) via
    # GLOBO_3D_PROJECTION_S41 (default 'google_earth', compartilhado entre s41/s42 por enquanto).
    # No modo google_earth a camera fica a GLOBO_3D_GE_ALTURA metros (menor = mais zoom) e
    # atmosfera/estrelas/vinheta sao desligadas (assumem o disco flutuante centralizado, que nao
    # se aplica ao recorte ampliado).
    _inclinado = script_id in ('s44', 's46', 's47')   # globo "deitado" (janela de recorte descentralizada)
    # PADRAO_TWC (qualquer globo inclinado -- s44, s46 e os proximos): troca o overlay Guillaume
    # (caixa do nome, data, subtitulo, barra de legenda) pelo estilo The Weather Channel -> so
    # credito + caixa azul do titulo + caixa cinza da data (TITULO_THE_WEATHER_CHANNEL / _DATA),
    # igual o s42. false = overlay normal (estilo s39). Nome antigo: S44_PADRAO_TWC (aviso abaixo).
    if settings.get('S44_PADRAO_TWC', None) is not None:
        logger.warning('S44_PADRAO_TWC foi RENOMEADA para PADRAO_TWC (a chave vale p/ todo globo '
                       'inclinado, nao so o s44) e NAO e mais lida — renomeie no seu settings.local.toml.')
    _padrao_twc = _inclinado and bool(settings.get('PADRAO_TWC', False))
    # s44: blue marble (Entrada/blue_marble.png) no CONTINENTE + GLOBO_3D_COR_OCEANO_SEM_VARIAVEL no
    # OCEANO (o blue marble e recortado na terra pela base do disco). Override PROPRIO do s44:
    # GLOBO_3D_INC_BLUE_MARBLE tem precedencia sobre o GLOBO_3D_BLUE_MARBLE compartilhado (que liga o
    # satelite no s42) -> da pra ligar/desligar so no s44 sem tocar no s42. CUSTO no s44: a reprojecao do
    # satelite (imshow) sobre a janela inclinada 16:9 estica o warp e pesa na RAM -> mitigado pelo regrid
    # proprio GLOBO_3D_INC_BLUE_MARBLE_REGRID (default 1280, vs 2048 dos demais; ver blue_marble_regrid).
    # Default do s44 = LIGADO (satelite leve no continente); false => cai no cinza solido do continente.
    _inc_ocean = str(settings.get('GLOBO_3D_COR_OCEANO_SEM_VARIAVEL', '#0e426d')) if _inclinado else None
    _inc_blue_marble = _inclinado and bool(
        settings.get('GLOBO_3D_INC_BLUE_MARBLE', settings.get('GLOBO_3D_BLUE_MARBLE', True)))
    # Blue marble EFETIVO (satelite no continente): s42 (flag GLOBO_3D_BLUE_MARBLE) ou s44 (_inc_blue_marble).
    _fundo_blue_marble = bool((script_id == 's42' and settings.get('GLOBO_3D_BLUE_MARBLE', False))
                              or _inc_blue_marble)
    # Cor do continente. Fichas SEM shaded (sem_variavel, ex.: z500_contour) tem cor_continente=None e
    # dependiam do blue marble p/ aparecer -- com o satelite OFF o continente ficava com a MESMA cor do
    # oceano (globo todo azul). Aqui, quando NAO ha blue marble, preenche a terra com uma cor solida
    # (default CINZA CLARO) p/ separar terra/mar. So no caso sem_variavel e so sem satelite (o cinza fica
    # no zorder=1, ACIMA do blue marble no zorder=0 -- entao com satelite ligado ele o cobriria).
    _cor_continente = ficha.get('cor_continente')
    if _cor_continente is None and not _fundo_blue_marble and (ficha.get('sem_variavel') or _inclinado):
        # s44 (_inclinado): continente CINZA solido TAMBEM p/ fichas SHADED (ex.: pwat_abs_alicia) quando
        # o satelite esta OFF -- so aparece onde a variavel e transparente (terra seca); sem isto a terra
        # seca ficava com a cor do OCEANO. Fora do s44, segue so p/ sem_variavel (comportamento de antes).
        _cor_continente = str(settings.get('GLOBO_3D_COR_CONTINENTE_SEM_VARIAVEL', '#d9d9d9'))
    # Logo no canto inf. direito (so s44): prioridade CAPUCIN>GREC>AMPERE (None se nenhuma flag ligada).
    from app.common.logo_helper import resolve_logo_path
    _inc_logo = resolve_logo_path(settings.get('DIR_INPUT', 'Entrada')) if _inclinado else None
    if script_id in ('s41', 's42'):
        proj_mode = str(settings.get('GLOBO_3D_PROJECTION_S41', 'google_earth')).lower()
    elif _inclinado:
        proj_mode = 'nearside'   # NearsidePerspective; o "deitado" vem da JANELA, nao da projecao
    else:
        proj_mode = str(getattr(settings, 'GLOBO_3D_PROJECTION', 'nearside')).lower()
    if proj_mode == 'google_earth':
        sat_height = float(settings.get('GLOBO_3D_GE_ALTURA', 5_000_000.0))
    elif _inclinado:
        sat_height = float(settings.get('GLOBO_3D_INC_ALTURA', 9_000_000.0))  # camera do globo inclinado
    else:
        sat_height = _SAT_HEIGHT_GEO

    # ── Enquadramento da figura ──────────────────────────────────────────────
    # Demais scripts: quadro QUADRADO 8x8in (globo flutuante centralizado). s41 no modo
    # google_earth: quadro PAISAGEM (mais largo que alto) reproduzindo o print de referencia —
    # o globo e GRANDE (preenche a largura), com o centro do disco EMPURRADO PARA BAIXO, de modo
    # que se ve a metade de cima do disco (America do Sul ampliada + arco do limbo no topo/cantos).
    # O eixo do globo e um QUADRADO (em polegadas) maior que o quadro; a figura recorta o excesso.
    #   GLOBO_3D_GE_ASPECT      -> altura/largura do quadro (<1 = paisagem)
    #   GLOBO_3D_GE_GLOBO_FRAC  -> diametro do disco como fracao da LARGURA do quadro (>=1 preenche)
    #   GLOBO_3D_GE_GLOBO_CY    -> posicao vertical do CENTRO do disco (0=base, 1=topo; baixo => desce)
    figsize = (8.0, 8.0)
    globe_rect = None
    inc = None   # payload do globo inclinado (so s44); None => set_global() normal (todos os outros)
    if script_id in ('s41', 's42') and proj_mode == 'google_earth':
        _asp = float(settings.get('GLOBO_3D_GE_ASPECT', 0.62))
        fig_w, fig_h = 8.0, 8.0 * _asp
        _gw = float(settings.get('GLOBO_3D_GE_GLOBO_FRAC', 1.02))  # diametro / largura do quadro
        _gh = _gw * fig_w / fig_h                                  # mesma medida em polegadas => QUADRADO
        _gcy = float(settings.get('GLOBO_3D_GE_GLOBO_CY', 0.29))   # centro do disco (fracao vertical)
        figsize = (fig_w, fig_h)
        globe_rect = [(1.0 - _gw) / 2.0, _gcy - _gh / 2.0, _gw, _gh]
    elif _inclinado:
        # Quadro PAISAGEM + eixo FULL-BLEED [0,0,1,1]: o enquadramento "deitado" vem da janela de
        # recorte descentralizada (_aplicar_janela_inclinada), aplicada por frame apos o set_global().
        _asp = float(settings.get('GLOBO_3D_INC_ASPECT', 0.5625))   # altura/largura do quadro (16:9)
        figsize = (8.0, 8.0 * _asp)
        globe_rect = [0.0, 0.0, 1.0, 1.0]
        # DEITAR (nao "INCLINACAO"!): deslocamento vertical da janela = o quanto o globo "deita".
        # NAO usar o sufixo INCLINACAO aqui -- `_camera_path` le _script_setting(script_id,'INCLINACAO')
        # que, p/ s44, resolve pra GLOBO_3D_INC_INCLINACAO; esse fica reservado ao corte HS/HN (latitude
        # fixa da camera), separado do deslocamento da janela.
        inc = {
            'deitar': float(settings.get('GLOBO_3D_INC_DEITAR', 0.555)),
            'janela_frac': float(settings.get('GLOBO_3D_INC_JANELA_FRAC', 0.858)),
            'aspecto': figsize[0] / figsize[1],
        }
    # Centro e raio do disco para atmosfera/estrelas — derivado do rect de cada estilo:
    #   guillaume rect [0.07, 0.06, 0.86, 0.86] -> cy=0.49, r=0.433
    #   wapo     rect [0.01, 0.10, 0.98, 0.83] -> cy=0.515, r=0.415, disc_top=0.930
    #   google_earth: eixo do globo e um QUADRADO em POLEGADAS (ver bloco acima) -> o raio, como
    #   fracao da dimensao MENOR do quadro (altura, pois aspect<1), e 0.5*_gw/_asp (>0.5 pois o
    #   disco e maior que a moldura); cy = _gcy direto (mesma convencao fracao-da-figura). Com
    #   isso o halo/estrelas acompanham so o arco VISIVEL do limbo (topo/cantos), sem precisar de
    #   mudanca nenhuma em `_atmosphere_glow_rgba`/`_starfield_rgba` (ja trabalham com h!=w).
    if globe_rect is not None and proj_mode == 'google_earth':
        _atm_cx, _atm_cy = 0.50, _gcy
        _atm_r = 0.5 * _gw / _asp
    elif _inclinado and inc is not None:
        # Globo deitado: eixo full-bleed [0,0,1,1] + janela de recorte. O nadir e o raio mapeiam
        # LINEARMENTE p/ fracao da figura (a janela casa o aspecto do quadro). raio como fracao da
        # ALTURA (mesma convencao do google_earth) = A/(2·janela_frac); cy do nadir = 0.5 - deitar·r.
        _A = figsize[0] / figsize[1]
        _atm_r = _A / (2.0 * inc['janela_frac'])
        _atm_cx, _atm_cy = 0.50, 0.5 - inc['deitar'] * _atm_r
    elif estilo == 'guillaume':
        _atm_cx, _atm_cy, _atm_r = 0.50, 0.49, 0.433
    else:
        _atm_cx, _atm_cy, _atm_r = 0.50, 0.515, 0.415
    # Rotulo de rodada p/ o rodape (so no forecast; reanalise nao tem rodada).
    run_init = anom.attrs.get('run_init')
    if run_init:
        _ri = datetime.strptime(run_init, '%Y-%m-%d %H')
        rodada_label = f'{fonte_label.upper()}  ·  {_ri:%b %-d, %Y}  ·  run {_ri:%H}Z'
    else:
        rodada_label = fonte_label.upper()
    # Nome da variavel na caixa (override: GLOBO_3D_ROTULO_<VAR>) e 5 rotulos da legenda (ingles).
    titulo_box = str(settings.get(f'GLOBO_3D_ROTULO_{variavel_key.upper()}',
                                  ficha.get('rotulo_box', ficha['titulo_en'])))
    _legenda5_default = ficha.get('legenda5_labels', ['Well below', 'Below', 'Average', 'Above', 'Well above'])
    legenda5 = list(settings.get('GLOBO_3D_LEGENDA5', _legenda5_default))
    # Canto sup-direito: variavel + nivel e periodo de climatologia (default 1991-2020).
    subtitulo_dir = str(settings.get(f'GLOBO_3D_SUBTITULO_{variavel_key.upper()}',
                                     ficha.get('subtitulo_dir', ficha['titulo_en'])))
    _clim_ref_str = f"Relative to the {settings.get('GLOBO_3D_CLIM_REF', '1991-2020')} normal"
    clim_ref = '' if ficha.get('sem_clim_ref') else _clim_ref_str

    # Cor de fundo do globo: ficha pode definir default (ex.: 'black' p/ wind abs);
    # fallback para centro da paleta se ficha nao define.
    _centro_paleta = paleta[len(paleta) // 2]
    # A paleta pode ser lista de CORES ('#rrggbb'[aa]) ou de NOS de colormap (posicao, cor) -- ex.:
    # pwat_abs. Extrai a COR do no (elemento [1]) senao `str((0.28, '#...'))` viraria uma cor invalida
    # (so aparecia fora do s42, onde bg_oceano_cor mascarava o cor_fundo_globo).
    if isinstance(_centro_paleta, (tuple, list)) and len(_centro_paleta) == 2:
        _centro_paleta = _centro_paleta[1]
    _centro_opaco = (_centro_paleta[:7] if isinstance(_centro_paleta, str)
                     and len(_centro_paleta) > 7 and _centro_paleta.startswith('#')
                     else _centro_paleta)
    _cor_fundo_default = ficha.get('cor_fundo_globo_default', _centro_opaco)
    cor_fundo_globo = str(settings.get(f'GLOBO_3D_COR_FUNDO_{variavel_key.upper()}',
                                       settings.get('GLOBO_3D_COR_FUNDO', _cor_fundo_default)))

    # ── Camada opcional de OLR equatorial (taper latitudinal SUAVE no alpha; sem corte reto) ──
    olr_overlay = None
    _olr_suf_en = _olr_suf_box = _olr_suf_sub = ''
    if (olr_serie is not None and bool(settings.get('GLOBO_3D_OLR_OVERLAY', False))
            and variavel_key != 'olr_anom'):
        _olr = olr_serie.sel(time=anom['time'].values, method='nearest')
        if coarsen and coarsen > 1:
            _olr = _olr.coarsen(lat=coarsen, lon=coarsen, boundary='trim').mean()
        _olr_cyc, _olr_loncyc = add_cyclic_point(_olr.values, coord=_olr['lon'].values)
        _core = float(settings.get('GLOBO_3D_OLR_LAT_CORE', 5.0))   # sinal cheio em |lat|<=core
        _edge = float(settings.get('GLOBO_3D_OLR_LAT_EDGE', 12.0))  # alpha->0 em |lat|>=edge
        olr_overlay = {
            'cyc': _olr_cyc.astype(np.float32), 'lon': _olr_loncyc, 'lat': _olr['lat'].values,
            'taper': _taper_lat(_olr['lat'].values, _core, _edge).astype(np.float32),
            'cmap': LinearSegmentedColormap.from_list('olr_anom', VARIAVEIS['olr_anom']['cmap_colors']),
            'norm': plt.Normalize(vmin=-float(settings.get('GLOBO_3D_OLR_VMAX', 40.0)),
                                  vmax=float(settings.get('GLOBO_3D_OLR_VMAX', 40.0))),
            'amax': float(settings.get('GLOBO_3D_OLR_ALPHA_MAX', 0.9)),
            'knee': float(settings.get('GLOBO_3D_OLR_ALPHA_KNEE', 8.0)),  # |OLR| p/ opacidade plena
        }
        _olr_suf_en, _olr_suf_box, _olr_suf_sub = ' + equatorial OLR', ' + EQ. OLR', ' + equatorial OLR'
        logger.info('Overlay de OLR equatorial ATIVO (|lat|<={:.0f}°, taper ate {:.0f}°)', _core, _edge)

    # ── Camada opcional de NEVE acumulada (overlay `NEVE`, ver ficha `precip_abs`): 2a serie SHADED (banda discreta,
    # MESMA tecnica do campo primario via _contourf_raster -- nao pcolormesh+alpha do overlay de
    # OLR, pra ficar visualmente consistente com a chuva) desenhada por cima. Ver bloco de render
    # em _render_frame_worker (ctx['neve_overlay']).
    neve_overlay = None
    if neve_serie is not None:
        _neve = neve_serie.sel(time=anom['time'].values, method='nearest')
        if coarsen and coarsen > 1:
            _neve = _neve.coarsen(lat=coarsen, lon=coarsen, boundary='trim').mean()
        _neve_cyc, _neve_loncyc = add_cyclic_point(_neve.values, coord=_neve['lon'].values)
        _neve_lat = _neve['lat'].values
        _neve_anchor = list(ficha.get('neve_cmap_colors', [
            '#07b5d7', '#058fb2', '#097591', '#9187ef',
            '#8c59e9', '#5f2ab0', '#ea478b', '#eca6e5',
        ]))
        # Escala FIXA 1-100 cm (pedido do usuario: nao usar teto dinamico por periodo/enquadramento
        # -- so os niveis/cores da ficha, sempre). Valores acima de 100 cm nao viram buraco/cor
        # errada: o `extend='max'` do contourf (ver `_contourf_raster`) ja pinta tudo acima do ultimo
        # nivel com a ULTIMA cor (`_neve_anchor[-1]`) -- a "ponta" na legenda-gradiente
        # (`_legenda_gradiente_chuva_neve`) so torna isso visivel pro espectador.
        _neve_levels = np.asarray(ficha.get('neve_levels', [1, 3, 5, 10, 20, 30, 50, 75, 100]),
                                  dtype=float)
        _neve_colors = _neve_anchor
        neve_overlay = {
            'cyc': _neve_cyc.astype(np.float32), 'lon': _neve_loncyc, 'lat': _neve_lat,
            'levels': _neve_levels, 'cmap': ListedColormap(_neve_colors),
        }
        logger.info('Overlay de NEVE ATIVO: {} banda(s), {:.0f} a {:.0f} cm (escala fixa; acima '
                    'disso assume a ultima cor da paleta)', len(_neve_colors), _neve_levels[0],
                    _neve_levels[-1])
        # DEBUG TEMPORARIO (investigacao Andes x Patagonia, remover depois) -- file I/O direto,
        # sem depender do logger (log estava sumindo sem excecao nenhuma).
        try:
            with open('/tmp/debug_neve.txt', 'a') as _f:
                _f.write(f'--- neve_overlay ctx | cyc shape={_neve_cyc.shape} '
                        f'lon_cyc len={len(_neve_loncyc)} lat len={len(_neve["lat"].values)} ---\n')
                _dbg_lat = _neve['lat'].values
                _dbg_lon = _neve['lon'].values % 360
                _dbg_andes = (_dbg_lat[:, None] >= -37) & (_dbg_lat[:, None] <= -30) & \
                             (_dbg_lon[None, :] >= 288) & (_dbg_lon[None, :] <= 291)
                _dbg_pata = (_dbg_lat[:, None] >= -56) & (_dbg_lat[:, None] <= -50) & \
                            (_dbg_lon[None, :] >= 285) & (_dbg_lon[None, :] <= 292)
                _dbg_last = _neve.isel(time=-1).values
                _f.write(f'Andes max={np.nanmax(_dbg_last[_dbg_andes]) if _dbg_andes.any() else float("nan")} '
                        f'n_andes_px={int(_dbg_andes.sum())} '
                        f'nan_pct={100*np.isnan(_dbg_last[_dbg_andes]).mean() if _dbg_andes.any() else -1}\n')
                _f.write(f'Patagonia max={np.nanmax(_dbg_last[_dbg_pata]) if _dbg_pata.any() else float("nan")} '
                        f'n_pata_px={int(_dbg_pata.sum())} '
                        f'nan_pct={100*np.isnan(_dbg_last[_dbg_pata]).mean() if _dbg_pata.any() else -1}\n')
                _f.write(f'cyc dtype={_neve_cyc.dtype} cyc_global_max={np.nanmax(_neve_cyc)} '
                        f'cyc_nan_pct_total={100*np.isnan(_neve_cyc).mean()}\n')
                _f.flush()
        except Exception as _dbg_e:
            with open('/tmp/debug_neve.txt', 'a') as _f:
                _f.write(f'DEBUG EXCEPTION: {type(_dbg_e).__name__}: {_dbg_e}\n')

    # ── Caixa de texto LIVRE (opcional): ancorada em lat/lon, fade-in perto do fim do clipe ──
    # Caixas de texto LIVRES: a "principal" (settings singulares, GLOBO_3D_CAIXA_LIVRE_*, sempre
    # existiu) + quaisquer EXTRAS (GLOBO_3D_CAIXAS_LIVRES_EXTRA, lista de dicts -- mesmo padrao
    # de GLOBO_3D_ICONES_PRESSAO -- pra anotar mais de um lugar no mapa ao mesmo tempo).
    caixas_livres: list[dict] = []
    if bool(settings.get('GLOBO_3D_CAIXA_LIVRE', False)) and str(settings.get('GLOBO_3D_CAIXA_LIVRE_TEXTO', '')):
        caixas_livres.append({
            'lat': float(settings.get('GLOBO_3D_CAIXA_LIVRE_LAT', 0.0)),
            'lon': float(settings.get('GLOBO_3D_CAIXA_LIVRE_LON', -140.0)),
            'texto': str(settings.get('GLOBO_3D_CAIXA_LIVRE_TEXTO', '')),
            'cor_box': str(settings.get('GLOBO_3D_CAIXA_LIVRE_COR_BOX', 'black')),
            'cor_texto': str(settings.get('GLOBO_3D_CAIXA_LIVRE_COR_TEXTO', 'white')),
            'contorno_cor': str(settings.get('GLOBO_3D_CAIXA_LIVRE_CONTORNO_COR', 'white')),
            'contorno_lw': float(settings.get('GLOBO_3D_CAIXA_LIVRE_CONTORNO_LW', 0.0)),
            'fontsize': float(settings.get('GLOBO_3D_CAIXA_LIVRE_FONTSIZE', 14.0)),
            'largura': int(settings.get('GLOBO_3D_CAIXA_LIVRE_LARGURA', 22)),   # quebra de linha (0 = sem)
            'inicio_frac': float(settings.get('GLOBO_3D_CAIXA_LIVRE_INICIO', 0.80)),  # % do clipe p/ comecar o fade
            'fade_frac': float(settings.get('GLOBO_3D_CAIXA_LIVRE_FADE', 0.12)),      # duracao do fade (% do clipe)
            # fixa=true: sempre em alpha_max (default s42). false: esmaece na linha do tempo do MP4
            # (inicio_frac..+fade_frac) mesmo no s42 -- ex.: caixa que aparece no inicio da cauda.
            'fixa': bool(settings.get('GLOBO_3D_CAIXA_LIVRE_FIXA', _caixa_fixa)),
            'alpha_max': float(settings.get('GLOBO_3D_CAIXA_LIVRE_ALPHA_MAX', 1.0)),  # opacidade final (1.0 = OPACA)
            # Sombra preta esfumacada ao redor da caixa (halo suave p/ realcar a borda branca).
            'sombra': bool(settings.get('GLOBO_3D_CAIXA_LIVRE_SOMBRA', True)),
            # Espaco entre o texto e a borda da caixa (unidades de fonte; default do matplotlib
            # pro boxstyle 'square' seria 0.3 -- 0.55 ficava com sobra grande nas 4 bordas).
            'pad': float(settings.get('GLOBO_3D_CAIXA_LIVRE_PAD', 0.30)),
        })
    for _cx in (settings.get('GLOBO_3D_CAIXAS_LIVRES_EXTRA', []) or []):
        if not str(_cx.get('texto', '')):
            continue
        caixas_livres.append({
            'lat': float(_cx.get('lat', 0.0)),
            'lon': float(_cx.get('lon', 0.0)),
            'texto': str(_cx.get('texto', '')),
            'cor_box': str(_cx.get('cor_box', 'black')),
            'cor_texto': str(_cx.get('cor_texto', 'white')),
            'contorno_cor': str(_cx.get('contorno_cor', 'white')),
            'contorno_lw': float(_cx.get('contorno_lw', 0.0)),
            'fontsize': float(_cx.get('fontsize', 14.0)),
            'largura': int(_cx.get('largura', 22)),
            'inicio_frac': float(_cx.get('inicio_frac', 0.80)),
            'fade_frac': float(_cx.get('fade_frac', 0.12)),
            'fixa': bool(_cx.get('fixa', _caixa_fixa)),
            'alpha_max': float(_cx.get('alpha_max', 1.0)),
            'sombra': bool(_cx.get('sombra', True)),
            'pad': float(_cx.get('pad', 0.30)),
        })
    for _cxl_log in caixas_livres:
        logger.info('Caixa de texto livre ATIVA em (lat {:.2f}, lon {:.2f}): "{}" | fade {:.0%}..{:.0%}',
                    _cxl_log['lat'], _cxl_log['lon'], _cxl_log['texto'],
                    _cxl_log['inicio_frac'], min(1.0, _cxl_log['inicio_frac'] + _cxl_log['fade_frac']))

    # Polígonos de TEMPO SEVERO (GLOBO_3D_POLIGONOS_SEVERO): preenchimento SOBRE o shaded, recortado
    # no CONTINENTE (nunca aparece no oceano mesmo se o polígono cruzar a costa -- mesmo recorte
    # vetorial 50m do `mascara_oceano`/PLOTAR_SOMENTE_CONTINENTE) e desenhado ABAIXO das linhas de
    # costa/fronteiras/estados (zorder entre o shaded e as bordas -- ver _draw_poligonos_severo).
    poligonos_severo: list[dict] = []
    for _pg in (settings.get('GLOBO_3D_POLIGONOS_SEVERO', []) or []):
        _pontos = list(_pg.get('pontos') or [])
        if len(_pontos) < 3:
            logger.warning('Polígono de tempo severo ignorado: precisa de >=3 pontos (veio {})', len(_pontos))
            continue
        poligonos_severo.append({
            'pontos_lonlat': [(float(_lo), float(_la)) for _la, _lo in _pontos],  # shapely quer (x,y)=(lon,lat)
            'cor': str(_pg.get('cor', '#ffcc00')),
            'alpha': float(_pg.get('alpha', 0.35)),
            'cor_borda': str(_pg.get('cor_borda', _pg.get('cor', '#ffcc00'))),
            'lw_borda': float(_pg.get('lw_borda', 1.0)),
            'texto': _pg.get('texto'),
        })
        logger.info('Polígono de tempo severo ATIVO: {} pontos, cor {}', len(_pontos), poligonos_severo[-1]['cor'])

    # Ícones de pressão animados (s38–s41): GIFs de Entrada/icones_pressao/ ancorados em lat/lon.
    _icones_pressao: list[dict] = []
    _icones_cfg = list(settings.get('GLOBO_3D_ICONES_PRESSAO', []) or [])
    if _icones_cfg:
        _dir_icones = Path(settings.get('DIR_INPUT', 'Entrada')) / 'icones_pressao'
        for _ic in _icones_cfg:
            _tipo = str(_ic.get('tipo', ''))
            if not _tipo:
                continue
            _gif_path = _dir_icones / f'{_tipo}.gif'
            if not _gif_path.exists():
                logger.warning('Ícone de pressão não encontrado: {}', _gif_path)
                continue
            _frames = _load_gif_frames(_gif_path)
            if not _frames:
                continue
            # Recolore o AZUL dos icones de ALTA pressao (HIGH_HN/HS, ALTA_HN/HS) se
            # GLOBO_3D_ICONE_ALTA_COR estiver configurado -- nao mexe nos de BAIXA (vermelhos).
            _cor_alta = str(settings.get('GLOBO_3D_ICONE_ALTA_COR', '')).strip()
            if _cor_alta and _tipo.upper().startswith(('HIGH_', 'ALTA_')):
                _frames = _recolor_icon_frames(_frames, _cor_alta)
            _icones_pressao.append({**dict(_ic), '_frames': _frames})
            logger.info('Ícone de pressão: {} ({} frames, tamanho {:.1f}°)',
                        _tipo, len(_frames), float(_ic.get('tamanho_deg', 8.0)))

    # Objeto de figura estático (settings OBJETO_TEXTO): UMA imagem ancorada em lat/lon, tamanho
    # fixo em graus -- sem animação/recorte de continente (ver _draw_objeto_texto).
    objeto_texto: dict | None = None
    _obj_cfg = dict(settings.get('OBJETO_TEXTO', {}) or {})
    if _obj_cfg.get('imagem'):
        _obj_path = Path(settings.get('DIR_INPUT', 'Entrada')) / str(_obj_cfg['imagem'])
        _obj_img = _load_static_image_rgba(_obj_path)
        if _obj_img is not None:
            objeto_texto = {
                'lat': float(_obj_cfg.get('lat', 0.0)),
                'lon': float(_obj_cfg.get('lon', 0.0)),
                'tamanho_deg': float(_obj_cfg.get('tamanho_deg', 8.0)),
                'zorder': int(_obj_cfg.get('zorder', 8)),   # ordem de desenho (default 8; ver stack no _draw_objeto_texto)
                '_img': _obj_img,
            }
            logger.info('Objeto de figura ATIVO: {} em (lat {:.2f}, lon {:.2f}), tamanho {:.1f}°',
                        _obj_cfg['imagem'], objeto_texto['lat'], objeto_texto['lon'],
                        objeto_texto['tamanho_deg'])

    # Config das correntes de jato (s41): LISTA de jatos (JET_STREAM + SUBTROPICAL_JET). Cada um
    # tem nivel/cor/texto proprios (e o PROPRIO campo-guia: GLOBO_3D_JET_STREAM_NIVEL/
    # SUBTROPICAL_JET_NIVEL decidem Z250 ou Z500 via _jato_kind_from_nivel); o resto do estilo
    # (faixa, faixas finas, setas, texto, drape) e COMPARTILHADO. Cada jato so entra na lista se
    # o campo-guia do SEU kind estiver disponivel -- um jato pode ficar de fora sem derrubar o
    # outro (ex.: JET_STREAM em Z250 funciona mesmo se o Z500 do SUBTROPICAL_JET faltar).
    _jatos_cfg: list[dict] = []
    if _jato_on and (hgt_z250_abs_cyc is not None or hgt_z500_abs_cyc is not None):
        # Unidade de fase por saida (a velocidade de CADA jato multiplica isto). GIF e MP4 usam a MESMA
        # base (`_flow_base`) -> `velocidade` tem efeito IDENTICO por frame nas duas saidas:
        #   GIF -> 1/base  (avanca v/base por frame; loop = base*m cobre m*v espacamentos INTEIROS -> seamless)
        #   MP4 -> 1/base  (mesmo passo por frame; fluxo continuo, sem emenda a respeitar)
        #   PNG -> 0       (parado, fase 0)
        if estatico:
            _phase_unit = 0.0
        else:
            _phase_unit = 1.0 / float(_flow_base)
        _campo_disponivel = {'hgt250': hgt_z250_abs_cyc is not None, 'hgt500': hgt_z500_abs_cyc is not None}
        _estilo_comum = {
            'alpha': float(_script_setting(script_id, 'JATO_ALPHA', 1.0)),        # central = opaca
            'stripe_alpha': float(_script_setting(script_id, 'JATO_STRIPE_ALPHA', 0.35)),
            'stripe_n': int(_script_setting(script_id, 'JATO_STRIPE_N', 3)),      # faixas finas por lado
            'stripe_gap0': float(_script_setting(script_id, 'JATO_STRIPE_GAP0', 1.0)),  # graus: 1a faixa
            'stripe_gap': float(_script_setting(script_id, 'JATO_STRIPE_GAP', 0.7)),    # graus: entre faixas
            'setas_entre': int(_script_setting(script_id, 'JATO_SETAS_ENTRE', 2)),   # nº de setas entre palavras
            'setas_passo': float(_script_setting(script_id, 'JATO_SETAS_PASSO', 6.0)),  # graus entre setas (auto-espaco)
            'arrow_cor': str(_script_setting(script_id, 'JATO_ARROW_COR', 'white')),
            'texto_cor': str(_script_setting(script_id, 'JATO_TEXTO_COR', 'white')),
            'texto_max_curva': float(_script_setting(script_id, 'JATO_TEXTO_MAX_CURVA', 110.0)),
            'min_pts': int(_script_setting(script_id, 'JATO_MIN_PTS', 12)),
            'lat_band': (float(_script_setting(script_id, 'JATO_LAT_MIN', 15.0)),
                         float(_script_setting(script_id, 'JATO_LAT_MAX', 60.0))),
            'hemisferios': _hemisferios,   # ('N','S'), ('S',) ou ('N',) — hemisferio(s) onde plotar
            # modo TELA (nao-drape): tamanhos em pontos
            'largura': float(_script_setting(script_id, 'JATO_LARGURA', 16.0)),
            'stripe_largura': float(_script_setting(script_id, 'JATO_STRIPE_LARGURA', 3.0)),
            'arrow_tam': float(_script_setting(script_id, 'JATO_ARROW_TAM', 12.0)),
            'texto_tam': float(_script_setting(script_id, 'JATO_TEXTO_TAM', 11.0)),
            # modo DRAPE (rente a superficie): tamanhos em GRAUS geograficos
            'largura_deg': float(_script_setting(script_id, 'JATO_LARGURA_DEG', 2.2)),
            'stripe_largura_deg': float(_script_setting(script_id, 'JATO_STRIPE_LARGURA_DEG', 0.5)),
            'texto_tam_deg': float(_script_setting(script_id, 'JATO_TEXTO_TAM_DEG', 1.7)),
            'arrow_tam_deg': float(_script_setting(script_id, 'JATO_ARROW_TAM_DEG', 2.4)),
        }
        if _jet1_on:   # JET STREAM (azul)
            _nivel1 = float(_script_setting(script_id, 'JET_STREAM_NIVEL', 10200.0))
            _kind1 = _jato_kind_from_nivel(_nivel1)
            if _campo_disponivel[_kind1]:
                _jatos_cfg.append({**_estilo_comum,
                    'nome': 'JET_STREAM', 'kind': _kind1,
                    'nivel': _nivel1,
                    'cor': str(_script_setting(script_id, 'JET_STREAM_COR', '#1787ad')),
                    'texto': str(_script_setting(script_id, 'JET_STREAM_TEXTO', 'JET STREAM')),
                    'velocidade': float(_script_setting(script_id, 'JET_STREAM_VELOCIDADE', 1.0)) * _phase_unit,
                    'cut': _parse_jet_cut(_script_setting(script_id, 'JET_STREAM_CUT', None)),
                })
            else:
                logger.warning('JET_STREAM: campo-guia {} indisponivel (nivel {:.0f} mgp) — '
                               'jato NAO sera plotado.', _kind1.upper(), _nivel1)
        if _jet2_on:   # SUBTROPICAL JET (verde)
            _nivel2 = float(_script_setting(script_id, 'SUBTROPICAL_JET_NIVEL', 10600.0))
            _kind2 = _jato_kind_from_nivel(_nivel2)
            if _campo_disponivel[_kind2]:
                _jatos_cfg.append({**_estilo_comum,
                    'nome': 'SUBTROPICAL_JET', 'kind': _kind2,
                    'nivel': _nivel2,
                    'cor': str(_script_setting(script_id, 'SUBTROPICAL_JET_COR', '#2e8b57')),
                    'texto': str(_script_setting(script_id, 'SUBTROPICAL_JET_TEXTO', 'SUBTROPICAL JET')),
                    'velocidade': float(_script_setting(script_id, 'SUBTROPICAL_JET_VELOCIDADE', 1.0)) * _phase_unit,
                    'cut': _parse_jet_cut(_script_setting(script_id, 'SUBTROPICAL_JET_CUT', None)),
                })
            else:
                logger.warning('SUBTROPICAL_JET: campo-guia {} indisponivel (nivel {:.0f} mgp) — '
                               'jato NAO sera plotado.', _kind2.upper(), _nivel2)
        _saida = 'GIF' if gif else ('PNG estatico' if estatico else 'MP4')
        for _jc in _jatos_cfg:
            logger.info('Corrente de jato "{}" ({}): {}={:.0f} mgp, cor {}, texto "{}", {} frame(s)',
                        _jc['nome'], _saida, _jc['kind'].upper(), _jc['nivel'], _jc['cor'], _jc['texto'],
                        total_frames)
    _jato_drape = bool(_script_setting(script_id, 'JATO_DRAPE', False))
    _jato_drape_px = int(_script_setting(script_id, 'JATO_DRAPE_PX', 5000))
    _jato_drape_regrid = int(_script_setting(script_id, 'JATO_DRAPE_REGRID', 2048))
    _escoamentos_cfg = _build_escoamentos_cfg()
    _escoamento_drape_px = int(settings.get('GLOBO_3D_ESCOAMENTO_DRAPE_PX', 3000))
    _escoamento_drape_regrid = int(settings.get('GLOBO_3D_ESCOAMENTO_DRAPE_REGRID', 2048))
    _jato_drape_pad = float(_script_setting(script_id, 'JATO_DRAPE_PAD', 6.0))
    # regrid_shape do cartopy imshow() se aplica ao EXTENT INTEIRO do eixo (nao so ao extent do
    # icone) -- por isso precisa de um valor alto (mesma ordem do drape do jato) mesmo o icone
    # ocupando so alguns graus do globo visivel; com um valor baixo o icone fica serrilhado.
    _icone_pressao_regrid = int(_script_setting(script_id, 'ICONE_PRESSAO_REGRID', 2048))
    _shade_regrid = int(settings.get('GLOBO_3D_SHADE_REGRID', 2048))
    # s44 (globo INCLINADO): a janela e 16:9 e MUITO mais alta que larga (ver _aplicar_janela_
    # inclinada) -> o cartopy ESTICA a grade de reprojecao (regrid 2048 vira ~2048x3641) e cada warp
    # fica ~1.8x mais pesado que no s42 (disco ~quadrado). Como a SAIDA tem so ~1080px de altura
    # (GLOBO_3D_TAMANHO_PX), 2048 nas camadas que cobrem o globo INTEIRO (sombreado, jato, escoamento)
    # e oversampling INVISIVEL -> defaults mais leves SO no s44 (namespace INC_ sobrepoe; s42 e os
    # demais seguem em 2048). O regrid do ICONE fica FORA daqui (feature pequena/bitmap: baixa-lo
    # serrilha) -- ele e resolvido na Parte B (reprojecao local). `JATO*` nao passa pelo INC_ do
    # _script_setting (flags de jato sao unificadas entre os globos), por isso o override e inline.
    if script_id in ('s44', 's46', 's47'):
        _shade_regrid = int(settings.get('GLOBO_3D_INC_SHADE_REGRID', 1280))
        _jato_drape_regrid = int(settings.get('GLOBO_3D_INC_JATO_DRAPE_REGRID', 1280))
        _escoamento_drape_regrid = int(settings.get('GLOBO_3D_INC_ESCOAMENTO_DRAPE_REGRID', 1280))

    # Flags que decidem se o OCEANO fica SEM nenhum dado shaded (mascara por variavel, ficha
    # 'sem_variavel', ou os toggles globais PLOTAR_SOMENTE_BRASIL/CONTINENTE). GLOBO_3D_COR_OCEANO_
    # SEM_VARIAVEL e MANDATORIA nesses casos -- sobrepoe a cor por variavel (GLOBO_3D_COR_OCEANO_
    # <VAR>), que so vale quando o oceano PODE ter dado (shaded cobrindo tudo, sem mascara/recorte).
    _mascara_oceano_var = bool(settings.get(f'GLOBO_3D_MASCARA_OCEANO_{variavel_key.upper()}_{script_id.upper()}',
                                            settings.get(f'GLOBO_3D_MASCARA_OCEANO_{variavel_key.upper()}', False)))
    # PLOTAR_SOMENTE_BRASIL/CONTINENTE valem no s42 E no s44 (o s44 reusa a lista de variaveis do s42):
    # recortam o shaded de QUALQUER variavel na terra, sem precisar de GLOBO_3D_MASCARA_OCEANO_<VAR> por ficha.
    _plotar_so_brasil = bool(settings.get('PLOTAR_SOMENTE_BRASIL', False)) if script_id in ('s42', 's44', 's46', 's47') else False
    _plotar_so_continente = bool(settings.get('PLOTAR_SOMENTE_CONTINENTE', False)) if script_id in ('s42', 's44', 's46', 's47') else False
    _oceano_sem_dado = (bool(ficha.get('sem_variavel')) or _mascara_oceano_var
                       or _plotar_so_brasil or _plotar_so_continente)

    # ── Legenda de faixas do estilo TWC: ATE QUE BANDA desenhar ────────────────────────────────
    # Regra: so as faixas que aparecem PINTADAS no quadro. Calculada UMA vez por clipe (fixa: nao
    # muda entre frames, senao a legenda mudaria de tamanho durante o video) usando o MAXIMO DO
    # PERIODO dentro do enquadramento E da area pintada (Brasil/continente, quando recortado).
    # A camera do ultimo frame e a do assentamento (no globo inclinado ela e fixa o clipe inteiro).
    _twc_leg_on = bool(settings.get('TITULO_THE_WEATHER_CHANNEL_LEGENDA', False)
                       and (script_id == 's42' or _padrao_twc) and not ficha.get('sem_variavel'))
    _twc_leg_ate = None
    if _twc_leg_on:
        _pj = _make_projection(float(lons[-1]), float(lats[-1]), mode=proj_mode, sat_height=sat_height)
        _fg = plt.figure(figsize=figsize, dpi=40)      # descartavel, so p/ medir o enquadramento
        _ax = _fg.add_axes(globe_rect or [0.0, 0.0, 1.0, 1.0], projection=_pj)
        _ax.set_global()
        _aplicar_janela_inclinada(_ax, _pj, inc)
        _bb_leg = _bbox_visivel_lonlat(_ax, _pj)
        plt.close(_fg)
        _twc_leg_ate = _ultima_banda_pintada(
            np.nanmax(vals_cyc, axis=0), lon_cyc, lat, levels, _bb_leg,
            _geom_area_pintada(_plotar_so_brasil, _plotar_so_continente))
        logger.info('Legenda TWC: faixas ate "{}" (maximo do periodo na area pintada)',
                    'nenhuma' if _twc_leg_ate is None else
                    f'{levels[_twc_leg_ate]:g}-{levels[_twc_leg_ate + 1]:g}')

    ctx = {
        'variavel_key': variavel_key,  # chave do cache do overlay (evita reuso entre variaveis)
        'vals_cyc': vals_cyc.astype(np.float32),
        'lon_cyc': lon_cyc, 'lat': lat, 'levels': levels,
        'paleta': list(paleta),
        'cmap_plot': cmap_plot, 'cmap_legend': cmap_legend, 'norm_fn': norm_fn,
        'shade_px': int(settings.get('GLOBO_3D_SHADE_PX', 3600)),      # resolucao do raster do contourf
        'shade_regrid': _shade_regrid,  # resolucao da reprojecao imshow (s44: mais leve, ver acima)
        # Recorte do raster do shaded na area visivel + o px que ele usa quando recorta (ver
        # `_subset_campo_bbox`). So faz efeito em vista fechada (s44/s46); nos flutuantes auto-desliga.
        'shade_crop': bool(settings.get('GLOBO_3D_SHADE_CROP', True)),
        'shade_crop_px': int(settings.get('GLOBO_3D_SHADE_CROP_PX', 2048)),
        # Upsample CUBICO do campo recortado (1 = off). Override por variavel, como o SIGMA: a chuva
        # precisa (celula de 28 km vira ~21 px no zoom), um campo sinotico liso nao. NAO e suavizacao
        # -- preserva os valores da grade (ver `_upsample_campo`); e a alternativa honesta ao SIGMA.
        'shade_upsample': float(settings.get(f'GLOBO_3D_SHADE_UPSAMPLE_{variavel_key.upper()}',
                                             settings.get('GLOBO_3D_SHADE_UPSAMPLE', 1.0))),
        'cor_fundo_globo': cor_fundo_globo,
        'lons': lons, 'lats': lats,
        'frames_por_dia': frames_por_dia, 'vel_var': vel_var, 'n_dias': n_dias,
        # `n_dias_janela`: ver `_pos_variavel` -- necessario pra escalar o passo-por-frame quando a
        # serie tem mais PASSOS que dias de calendario (ACUM_HORARIO).
        'n_dias_janela': n_dias_janela,
        # Pentada movel: a data vira o INTERVALO da janela [d, d+dias-1] (ex.: 'July 20–24, 2026')
        # em vez de um dia unico — senao o rotulo engana (o campo e media de `dias` dias).
        # Com `_acum_rotulo` (ACUMULAR_NO_TEMPO no MP4) a data e a JANELA ACUMULADA ate o passo i --
        # ancorada em dates[0] e crescendo. `_dias_janela(d)` conta DIAS DE CALENDARIO reais
        # decorridos ate `d` (NAO o indice `i` do passo): com ACUM_HORARIO a serie vem no passo
        # NATIVO do modelo (3h/6h) -- varios passos por dia -- entao `i+1` inflava a janela pra
        # "33 dias" numa serie de so 5 dias reais (bug: titulo mostrava agosto com DATA_FINAL=jul22,
        # e o voo da camera esticava o MP4 pra 217 frames). `dates[0].date()` normaliza a hora do
        # 1o passo (pode ser != 00Z) antes de subtrair, senao um `d` no mesmo dia calendario mas com
        # hora menor arredondaria pra baixo.
        # id unico DESTE clipe: sem ele o cache do overlay (uma entrada, chave (var, idx_dia))
        # serve o overlay de outro clipe/modelo quando a chave coincide -- ver _composite_overlay_box.
        'clip_uid': next(_CLIP_UID),
        'dates_en': [_fmt_data_pentada(dates[0] if _acum_rotulo else d,
                                       _dias_janela(dates[0], d) if _acum_rotulo else _pent_dias,
                                       com_ano=False, mostrar_hora=_mostrar_hora)
                     for d in dates],
        'dates_full': [_fmt_data_pentada(dates[0] if _acum_rotulo else d,
                                         _dias_janela(dates[0], d) if _acum_rotulo else _pent_dias,
                                         com_ano=True, mostrar_hora=_mostrar_hora)
                       for d in dates],  # s39
        'dates_wapo': [_fmt_data_pentada(dates[0] if _acum_rotulo else d,
                                         _dias_janela(dates[0], d) if _acum_rotulo else _pent_dias,
                                         com_ano=True, mostrar_hora=_mostrar_hora)
                       for d in dates],  # s38
        # `d1=d` (so quando `_acum_rotulo`): passa a hora REAL do passo pro `_fmt_data_br` embutir na
        # 2a data ('Jul 18–19 03Z') -- 1a data trava em dates[0], 2a data + hora andam a cada passo
        # nativo (ver docstring de `_fmt_data_br`). Pedido do usuario p/ ACUM_HORARIO.
        'dates_br': [_fmt_data_br(dates[0] if _acum_rotulo else d,
                                  _dias_janela(dates[0], d) if _acum_rotulo else _pent_dias,
                                  mostrar_hora=_mostrar_hora, d1=(d if _acum_rotulo else None))
                     for d in dates],  # s42/s46 (caixa "The Weather Channel")
        'titulo_en': ficha.get('titulo_en', ficha['titulo']) + _olr_suf_en,
        'olr_overlay': olr_overlay,
        'neve_overlay': neve_overlay,
        'fonte_label': fonte_label,
        'credito': str(getattr(settings, 'GLOBO_3D_CREDITO', 'Bruno Capucin')).upper(),
        'logo_path': str(_inc_logo) if _inc_logo else None,   # logo canto inf. direito (so s44)
        # Tamanho/enquadramento do logo PROPRIOS do s44 (namespace GLOBO_3D_INC_LOGO_*), sem tocar no
        # LOGO_WIDTH_FRAC global (usado nos demais mapas). Tamanho cai no global se o INC nao existir.
        'logo_largura_frac': float(settings.get('GLOBO_3D_INC_LOGO_WIDTH_FRAC',
                                                settings.get('LOGO_WIDTH_FRAC', 0.16))),
        'logo_margem_x_frac': float(settings.get('GLOBO_3D_INC_LOGO_MARGEM_X_FRAC', 0.02)),
        'logo_margem_y_frac': float(settings.get('GLOBO_3D_INC_LOGO_MARGEM_Y_FRAC', 0.06)),
        'font_titulo': font_titulo, 'font_legenda': font_legenda, 'font_twc': font_twc,
        'proj_mode': proj_mode,
        'sat_height': sat_height,
        'figsize': figsize,      # (w, h) em polegadas — s41 google_earth = paisagem; demais = 8x8
        'globe_rect': globe_rect,  # rect [x0,y0,w,h] do globo (s41 landscape); None = usa o do estilo
        # google_earth: disco nao esta centralizado/flutuante -> vinheta off (nao teria contexto
        # visual). Atmosfera/estrelas agora usa a geometria derivada do globe_rect (ver atm_cx/
        # cy/r acima) -- funciona no arco do limbo visivel nos cantos -- mas so habilitado no s42
        # por enquanto (pedido especifico); s41 mantem o comportamento atual (desligado).
        # s44 (globo inclinado): disco NAO esta centralizado/flutuante -> vinheta/atmosfera/estrelas
        # off (mesma razao do google_earth: assumem o disco centralizado). `not _inclinado` mantem
        # s38-s43 IDENTICOS (la _inclinado=False).
        'usar_vinheta': (bool(getattr(settings, 'GLOBO_3D_VINHETA', True))
                         and proj_mode != 'google_earth' and not _inclinado),
        # s44 (inclinado): atmosfera/estrelas HABILITADAS (geometria do disco calculada p/ a janela
        # inclinada, ver _atm_cx/cy/r acima) -- so a vinheta fica off (corte de cantos assume disco
        # centralizado). Por isso `or _inclinado` aqui, mas nao no usar_vinheta.
        'usar_atmosfera_estrelas': (bool(settings.get('GLOBO_3D_ATMOSFERA_ESTRELAS', False))
                                    and (proj_mode != 'google_earth' or script_id == 's42'
                                         or _inclinado)),
        # SOMENTE_ESTRELAS (s38-s42): so vale quando ATMOSFERA_ESTRELAS=false (senao o efeito
        # completo ja inclui as estrelas); mesma restricao de geometria do google_earth.
        'usar_somente_estrelas': ((not bool(settings.get('GLOBO_3D_ATMOSFERA_ESTRELAS', False)))
                                  and bool(settings.get('GLOBO_3D_SOMENTE_ESTRELAS', False))
                                  and (proj_mode != 'google_earth' or script_id == 's42'
                                       or _inclinado)),
        'inc': inc,   # payload do globo inclinado (so s44); None => set_global() normal
        'atm_cx': _atm_cx, 'atm_cy': _atm_cy, 'atm_r': _atm_r,
        'barra_h': 0.17 if estilo == 'wapo' else 0.0,
        'clim_abs_cyc': clim_abs_cyc,
        'hgt_abs_cyc': hgt_abs_cyc,
        'hgt_abs_levels': hgt_abs_levels,
        'hgt_anom_vals_cyc': hgt_anom_vals_cyc,
        'hgt_z250_abs_cyc': hgt_z250_abs_cyc,
        'hgt_z500_abs_cyc': hgt_z500_abs_cyc,   # campo-guia dedicado do jato quando kind='hgt500'
        'hgt_z250_iso_levels': hgt_z250_iso_levels,   # isolinhas de Z250 abs sobre campo estranho
        'hgt_z250_iso_cor': str(settings.get('GLOBO_3D_ISOL_HGT250_COR', '#666666')),
        'jatos': _jatos_cfg,                 # LISTA de jatos (JET_STREAM + SUBTROPICAL_JET)
        'jato_drape': _jato_drape,
        'jato_drape_px': _jato_drape_px,
        'jato_drape_regrid': _jato_drape_regrid,
        'jato_drape_pad': _jato_drape_pad,
        'escoamentos': _escoamentos_cfg,      # LISTA de escoamentos de baixos niveis (setas manuais)
        'escoamento_drape_px': _escoamento_drape_px,
        'escoamento_drape_regrid': _escoamento_drape_regrid,
        'isolinhas_fixas_hgt': ficha.get('isolinhas_fixas_hgt', []) if _isol_flag_on else [],
        'mslp_cyc': mslp_cyc,
        'mslp_levels': mslp_levels,
        'contour_cyc': contour_cyc,
        'contour_levels': contour_levels,
        'contour_lon': contour_lon,
        'contour_lat': contour_lat,
        'contour_color': str(ficha.get('contorno_serie_cor', 'black')),
        'contour_lw': float(settings.get('GLOBO_3D_CONTORNO_SERIE_LW',
                                         ficha.get('contorno_serie_lw', 0.5))),
        'usar_contorno': bool(settings.get(f'GLOBO_3D_CONTORNO_{variavel_key.upper()}',
                                            settings.get('GLOBO_3D_CONTORNO', False))),
        'campo_absoluto': bool(ficha.get('absoluto')),
        # Caixa do Niño 3.4: flag de settings aplicavel a QUALQUER variavel. Precedencia
        # GLOBO_3D_BOX_NINO34_<VAR> > GLOBO_3D_BOX_NINO34 (global) > ficha['box_nino34']
        # (default: so tsm_anom traz True na ficha; setar a global false desliga ate nela).
        'box_nino34': bool(settings.get(f'GLOBO_3D_BOX_NINO34_{variavel_key.upper()}',
                                        settings.get('GLOBO_3D_BOX_NINO34',
                                                     ficha.get('box_nino34', False)))),
        'nino34_serie': nino34_serie,  # media diaria do box (°C) p/ rotulo dinamico; None = so "Niño 3.4"
        'nino34_signed': bool(ficha.get('nino34_anom', False)),  # true = rotulo com sinal +/- (anomalia)
        'caixas_livres': caixas_livres,    # lista de caixas de texto livres (lat/lon + fade-in); [] = nenhuma
        'poligonos_severo': poligonos_severo,   # areas de tempo severo (recorte continente); [] = nenhum
        'icones_pressao': _icones_pressao or None,   # GIFs ancorados em lat/lon; None = sem ícones
        'icone_pressao_regrid': _icone_pressao_regrid,
        'objeto_texto': objeto_texto,   # imagem estática ancorada em lat/lon (settings OBJETO_TEXTO)
        'total_frames': total_frames,   # p/ o icone de pressao fechar o giro num numero INTEIRO de voltas
        'saida_estatica': bool(estatico or gif),   # PNG/GIF: icones em alpha CHEIO (sem fade da linha do tempo do MP4)
        'cor_continente': _cor_continente,
        # MANDATORIO: sempre que o oceano ficar sem shaded (mascara/recorte/sem_variavel),
        # GLOBO_3D_COR_OCEANO_SEM_VARIAVEL manda -- sobrepoe a cor por variavel (ver _oceano_sem_dado
        # acima). Só usa a cor da ficha (GLOBO_3D_COR_OCEANO_<VAR>) quando o oceano PODE ter dado.
        'cor_oceano': (_inc_ocean if _inclinado else
                       (str(settings.get('GLOBO_3D_COR_OCEANO_SEM_VARIAVEL', ficha.get('cor_oceano')))
                        if _oceano_sem_dado else ficha.get('cor_oceano'))),
        # Mascara de OCEANO por variavel (GLOBO_3D_MASCARA_OCEANO_<VAR>): apaga o shaded sobre a agua
        # (NaN -> transparente -> aparece o blue marble). Fica so nos continentes. Override POR SCRIPT
        # (_<SCRIPT>, ex.: _S42) tem precedencia -> so o s42 recorta o T850 no continente; s38/s39 nao.
        'mascara_oceano': _mascara_oceano_var,
        # Globais (SO s42): recortam o shaded independente de camera/voo/ficha. Prioridade
        # BRASIL > CONTINENTE > mascara_oceano por variavel (ver bloco de clip em _build_frame).
        'plotar_somente_brasil': _plotar_so_brasil,
        'plotar_somente_continente': _plotar_so_continente,
        # Transparencia CENTRAL por variavel: |anom| < este valor vira transparente (aparece o fundo).
        'transp_ate': float(settings.get(f'GLOBO_3D_TRANSP_ATE_{variavel_key.upper()}_{script_id.upper()}',
                                         settings.get(f'GLOBO_3D_TRANSP_ATE_{variavel_key.upper()}', 0.0))),
        # Cor das costas/divisas por variavel, com override POR SCRIPT (_<SCRIPT>, ex.: _S42) -> as
        # costas pretas do T850 ficam so no s42; s38/s39 usam a cor da ficha (branca/default).
        'cor_fronteiras': settings.get(f'GLOBO_3D_COR_FRONTEIRAS_{variavel_key.upper()}_{script_id.upper()}',
                                       settings.get(f'GLOBO_3D_COR_FRONTEIRAS_{variavel_key.upper()}',
                                                    ficha.get('cor_fronteiras'))),
        'extend_contourf': ficha.get('extend_contourf', 'both'),
        'shaded_alpha': float(ficha.get('shaded_alpha', 1.0)),
        # SO s42/s43 (ficha 'sem_variavel'): pula o desenho do shaded + caixa titulo/legenda,
        # mantendo jato/icones/caixas/credito (ver _build_frame e _overlay_guillaume).
        'sem_variavel': bool(ficha.get('sem_variavel', False)),
        'legenda_unidade': ficha.get('legenda_unidade', ''),
        'legenda_labels': ficha.get('legenda_labels', ['Well below', 'Below', 'Above', 'Well above']),
        'estilo': estilo,
        'titulo_box': titulo_box + _olr_suf_box,
        'subtitulo_dir': subtitulo_dir + _olr_suf_sub,
        'clim_ref': clim_ref,
        'legenda5_labels': legenda5,
        'legenda_numerica': bool(ficha.get('legenda_numerica', False)),
        'legenda_num_step': float(settings.get('GLOBO_3D_LEGENDA_NUM_STEP',
                                               ficha.get('legenda_num_step', 0.5))),
        'rodada_label': rodada_label,
        'dpi': dpi,
        # Espessura das linhas POR VARIAVEL: ficha pode sobrescrever o default global (ex.:
        # z250_abs engrossa e escurece p/ contrastar com o blue marble por baixo).
        # Espessura das costas/divisas/estados. Mesma precedencia do `cor_fronteiras` acima: POR
        # VARIAVEL+SCRIPT (_<VAR>_<SCRIPT>) > POR VARIAVEL (_<VAR>) > ficha > global. Precisa ser
        # RE-LIDO aqui: o `lw_coast` da ficha e congelado no IMPORT do modulo (dict VARIAVEIS), ou
        # seja, ANTES do aplicar_config_header() -- sem esta releitura, a espessura no header do
        # script era silenciosamente ignorada (so settings.local/mae funcionavam).
        'lw_coast':  float(settings.get(f'GLOBO_3D_LW_COAST_{variavel_key.upper()}_{script_id.upper()}',
                           settings.get(f'GLOBO_3D_LW_COAST_{variavel_key.upper()}',
                                        ficha.get('lw_coast', settings.get('GLOBO_3D_COASTLINE_LW', 0.5))))),
        'lw_border': float(settings.get(f'GLOBO_3D_LW_BORDER_{variavel_key.upper()}_{script_id.upper()}',
                           settings.get(f'GLOBO_3D_LW_BORDER_{variavel_key.upper()}',
                                        ficha.get('lw_border', settings.get('GLOBO_3D_BORDERS_LW', 0.35))))),
        'lw_states': float(settings.get(f'GLOBO_3D_LW_STATES_{variavel_key.upper()}_{script_id.upper()}',
                           settings.get(f'GLOBO_3D_LW_STATES_{variavel_key.upper()}',
                                        ficha.get('lw_states', settings.get('GLOBO_3D_STATES_LW', 0.2))))),
        # Sombreado CONTÍNUO (gouraud/nearest, pcolormesh) vs bandas SUAVES do contourf. Precedência:
        # POR VARIÁVEL (GLOBO_3D_PCOLORMESH_<VAR>, p/ forçar contourf/pcolormesh numa variável -- ex.:
        # pwat_abs_alicia=false p/ bandas SUAVES do contourf em vez dos BLOCOS de célula do pcolormesh) >
        # POR SCRIPT (_<SCRIPT>, ex.: _S42) > global GLOBO_3D_PCOLORMESH.
        'usar_pcolormesh': bool(settings.get(f'GLOBO_3D_PCOLORMESH_{variavel_key.upper()}',
                                settings.get(f'GLOBO_3D_PCOLORMESH_{script_id.upper()}',
                                             settings.get('GLOBO_3D_PCOLORMESH', False)))),
        # Modo do pcolormesh POR VARIAVEL: 'gouraud' (liso, default) ou 'nearest' (bandas discretas).
        'pcolormesh_shading': _pcm_shading,
        # Fundo de satelite (blue marble) — por enquanto so no s42 (pedido especifico); gate
        # explicito em vez do namespace GE_ compartilhado com o s41, pra nao ligar no s41 tambem.
        'fundo_blue_marble': _fundo_blue_marble,
        # s44 (inclinado): regrid MAIS LEVE p/ o warp do satelite -- mesma filosofia da Parte A do INC_
        # SHADE/JATO/ESCOAMENTO (1280). A saida tem ~1080px, entao 2048 e oversampling invisivel e a
        # janela inclinada 16:9 esticaria o warp (maior custo de RAM). Outros scripts seguem em 2048.
        'blue_marble_regrid': (int(settings.get('GLOBO_3D_INC_BLUE_MARBLE_REGRID', 1280)) if _inclinado
                               else int(settings.get('GLOBO_3D_BLUE_MARBLE_REGRID', 2048))),
        # Recorte da FONTE do satelite no bbox visivel antes do warp (o que torna MAX_PX=8192 viavel
        # no inclinado). Escape hatch: false = manda o globo inteiro p/ o warp (comportamento antigo,
        # ~8x mais RAM no s44/s46). Auto-desliga sozinho em vista larga -- ver `_bbox_visivel_lonlat`.
        'blue_marble_crop': bool(settings.get('GLOBO_3D_BLUE_MARBLE_CROP', True)),
        # Cor sólida do OCEANO (s42): oceano = esta cor + continente = blue marble recortado na terra.
        'bg_oceano_cor': ((str(settings.get('GLOBO_3D_COR_OCEANO', '')) or None) if script_id == 's42'
                          else _inc_ocean),   # s44: base do disco = oceano SEM_VARIAVEL (blue marble recorta na terra)
        # Minimalista: remove caixa do nome, data, subtitulo e barra/legenda do overlay guillaume --
        # fica so o credito no rodape. s42: GLOBO_3D_SO_CREDITO. Inclinado (s44/s46): PADRAO_TWC.
        'so_credito': bool((script_id == 's42' and settings.get('GLOBO_3D_SO_CREDITO', False))
                           or _padrao_twc),
        'fade_cauda_on': _fade_cauda_on,
        'fade_cauda_inicio': _fade_cauda_inicio,
        'fade_cauda_dur': _fade_cauda_dur,
        'caixa_fixa': _caixa_fixa,
        # Caixa "The Weather Channel" (titulo azul + data/hora em formato BR) -- so aparece
        # dentro do modo 'so_credito' e so se um titulo estiver configurado (s42, ou inclinado c/ TWC).
        'titulo_twc': (str(settings.get('TITULO_THE_WEATHER_CHANNEL', ''))
                       if (script_id == 's42' or _padrao_twc) else ''),
        # Texto da caixa cinza ao lado do titulo TWC. Vazio = usa a data/hora (data_br, default).
        # Preenchido = sobrescreve a data com esse texto (caixa se auto-ajusta ao tamanho).
        'titulo_twc_data': (str(settings.get('TITULO_THE_WEATHER_CHANNEL_DATA', ''))
                            if (script_id == 's42' or _padrao_twc) else ''),
        # Cor de fundo da caixa AZUL do titulo TWC (a caixa cinza da data segue fixa).
        'titulo_twc_cor': str(settings.get('TITULO_THE_WEATHER_CHANNEL_COR', '#0077a7')),
        # Caixa "Modelo <MODELO>" colada sob a azul (default OFF -> s42 e quem nao pedir seguem iguais).
        # O nome sai do `fonte_label`, entao acompanha o RUN_* habilitado sem config duplicada.
        'twc_modelo': bool(settings.get('TITULO_THE_WEATHER_CHANNEL_MODELO', False)
                           and (script_id == 's42' or _padrao_twc)),
        'twc_modelo_cor': str(settings.get('TITULO_THE_WEATHER_CHANNEL_MODELO_COR', '#14223d')),
        'twc_modelo_prefixo': str(settings.get('TITULO_THE_WEATHER_CHANNEL_MODELO_PREFIXO', 'Modelo')),
        'twc_modelo_fontsize': float(settings.get('TITULO_THE_WEATHER_CHANNEL_MODELO_FONTSIZE', 12.0)),
        # Legenda de faixas (ao lado da caixa do modelo). `twc_legenda_ate` = ultima banda pintada no
        # quadro (None = nao desenha). Rotulos/cores saem dos proprios levels+paleta do render.
        'twc_legenda_ate': _twc_leg_ate,
        'twc_legenda_unidade': str(ficha.get('unidade', '')) if _twc_leg_on else '',
        'twc_legenda_fontsize': float(settings.get('TITULO_THE_WEATHER_CHANNEL_LEGENDA_FONTSIZE', 8.0)),
        'twc_legenda_borda_lw': float(settings.get('TITULO_THE_WEATHER_CHANNEL_LEGENDA_BORDA_LW', 1.2)),
        'twc_legenda_gap': float(settings.get('TITULO_THE_WEATHER_CHANNEL_LEGENDA_GAP', 0.001)),
    }

    _fps_log = int(_script_setting(script_id, 'GIF_FPS', 12)) if gif else fps
    logger.info('{} dias -> {} frames | {} fps{} | {}px | {} workers | {}',
                n_dias, total_frames, _fps_log, ' (GIF)' if gif else '', px, workers, fonte_label)

    # Pré-aquece caches no pai — workers herdam via CoW, sem re-computar no 1º frame.
    _state_line_geoms()
    # Shapefiles cartopy: leitura de disco ocorre 1x; filhos herdam via CoW.
    list(cfeature.COASTLINE.with_scale('50m').geometries())
    list(cfeature.BORDERS.with_scale('50m').geometries())
    # Vinheta radial (independente de tamanho da figura).
    if ctx['usar_vinheta']:
        _vignette_rgba()
    # Atmosfera e estrelas: chave inclui (h, w, cx, cy, r) — pre-aquecer com os params corretos.
    if ctx['usar_atmosfera_estrelas']:
        _atmosphere_glow_rgba(px, px, cx=ctx['atm_cx'], cy=ctx['atm_cy'], r_frac=ctx['atm_r'])
        _starfield_rgba(px, px, cx=ctx['atm_cx'], cy=ctx['atm_cy'], r_disc=ctx['atm_r'])
    elif ctx['usar_somente_estrelas']:
        _starfield_rgba(px, px, cx=ctx['atm_cx'], cy=ctx['atm_cy'], r_disc=ctx['atm_r'])

    # ── Modo ESTATICO (s40): 1 frame -> PNG. Sem writer/pool. ──
    if estatico:
        if png_path is None:
            raise ValueError('_render_clip(estatico=True) exige png_path.')
        t0 = _time.time()
        png_path.parent.mkdir(parents=True, exist_ok=True)
        # GLOBO_3D_JATO_PNG (default true = comportamento de sempre: jato PARADO tambem aparece
        # no PNG). false = PNG sai SEM jato, mesmo com GLOBO_3D_JATO ligado (GIF/MP4 nao mudam).
        _skip_jet_png = not bool(settings.get('GLOBO_3D_JATO_PNG', True))
        frame = _build_frame(0, ctx, skip_jet=_skip_jet_png)
        imageio.imwrite(str(png_path), frame)
        logger.info('PNG salvo: {} ({:.1f}s)', png_path, _time.time() - t0)
        return png_path

    # Saida: GIF (loop, campo medio + jato animado) ou MP4 (voo + evolucao). O GIF acumula os
    # frames numa lista (loop curto) e grava via mimsave; o MP4 faz streaming pelo writer.
    if gif:
        if gif_path is None:
            raise ValueError('_render_clip(gif=True) exige gif_path.')
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        out_path = gif_path
        fps_out = int(_script_setting(script_id, 'GIF_FPS', 12))
        frames_buf: list[np.ndarray] = []
        writer = None
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f'{script_id}_{variavel_key}.mp4'
        fps_out = fps
        frames_buf = None
        writer = imageio.get_writer(
            str(out_path), fps=fps_out, codec='libx264', quality=8, macro_block_size=8,
        )

    # Fade-in/out (GLOBO_3D_FADE_IN_SEG / _OUT_SEG): SO no MP4. Escurece o RGB dos primeiros/ultimos
    # N frames (curva cosseno, suave nas pontas) -- fade-in sobe do preto (0->1), fade-out cai ao
    # preto (1->0). Com HOLD, o fade-out ainda segura frames PRETOS no fim pra o video nao cortar
    # seco. Aplicado no funil unico `_emit` (indice global do frame). 0 = desligado (default). Nao
    # vale p/ GIF (loop) nem PNG estatico (frame unico) -- estes zerariam/oscilariam o brilho.
    _fade_in = (int(round(float(settings.get('GLOBO_3D_FADE_IN_SEG', 0.0)) * fps))
                if writer is not None else 0)
    _fade_out = (int(round(float(settings.get('GLOBO_3D_FADE_OUT_SEG', 0.0)) * fps))
                 if writer is not None else 0)
    _fade_hold = (int(round(float(settings.get('GLOBO_3D_FADE_OUT_HOLD_SEG', 0.4)) * fps))
                  if _fade_out > 0 else 0)
    # CONGELA o ULTIMO frame por N segundos ANTES do fade-out (GLOBO_3D_ULTIMO_FRAME_SEG). Sem isto,
    # o fade-out morde os ultimos frames RENDERIZADOS: o quadro final (no s46, o acumulado TOTAL do
    # periodo -- a informacao que importa) ja aparece escurecendo. Com o congelamento, o ultimo frame
    # e' exibido INTEIRO por N s e so entao o fade comeca, sobre ele mesmo. So MP4.
    _ultimo_seg = float(settings.get('GLOBO_3D_ULTIMO_FRAME_SEG', 0.0)) if writer is not None else 0.0
    _ultimo_hold = max(0, int(round(_ultimo_seg * fps)))
    _emit_state: dict = {}
    if _fade_in > 0 or _fade_out > 0 or _ultimo_hold > 0:
        logger.info('MP4 fim: ultimo frame congelado {} frame(s) (~{:.1f}s) | fade in={} out={} '
                    '(+{} no preto)', _ultimo_hold, _ultimo_seg, _fade_in, _fade_out, _fade_hold)

    def _fator_fade(f_idx: int, total: int) -> float:
        """Brilho (0..1) do frame `f_idx` num clipe de `total` frames -- curva cosseno nas pontas."""
        k = 1.0
        if _fade_in > 0 and f_idx < _fade_in:
            k = min(k, 0.5 - 0.5 * np.cos(np.pi * f_idx / _fade_in))                # 0.0 -> 1.0
        restam = total - 1 - f_idx                    # 0 no ultimo frame do clipe
        if _fade_out > 0 and restam < _fade_out:
            k = min(k, 0.5 - 0.5 * np.cos(np.pi * restam / _fade_out))              # 1.0 -> 0.0
        return k

    # Comprimento TOTAL do clipe = frames renderizados + congelamento do ultimo. O fade-out se mede
    # a partir daqui, entao ele cai dentro do congelamento (e nao na animacao).
    _clipe_frames = total_frames + _ultimo_hold

    def _emit(f_idx: int, fr: np.ndarray) -> None:
        _emit_state['ultimo'] = fr           # guarda SEM fade: e ele que sera congelado no fim
        _k = _fator_fade(f_idx, _clipe_frames)
        if _k < 1.0:
            fr = (fr.astype(np.float32) * _k).astype(fr.dtype)
        _emit_state['shape'], _emit_state['dtype'] = fr.shape, fr.dtype
        if writer is not None:
            writer.append_data(fr)
        else:
            frames_buf.append(fr)

    t0 = _time.time()

    # ── CACHE DE FUNDO (GLOBO_3D_BG_CACHE) ────────────────────────────────────────────────────
    # Nos frames de FUNDO CONGELADO (todos do GIF; cauda do MP4) o campo e a camera nao mudam — so o
    # jato desliza. Renderiza o fundo pesado (sombreado + reprojecao + costa/fronteiras + isolinhas)
    # UMA vez aqui e guarda em ctx['_bg_arr']; cada frame reusa o fundo e compoe SO o jato (renderizado
    # normalmente por frame -> animacao/qualidade do jato IDENTICAS). Computado antes do fork -> os
    # workers herdam o fundo via CoW. Ganho ~2x sem tocar em nada de qualidade.
    _bg_ok = (bool(settings.get('GLOBO_3D_BG_CACHE', True)) and _bg_from is not None
              and _jato_on and (hgt_z250_abs_cyc is not None or hgt_z500_abs_cyc is not None)
              and total_frames > _bg_from + 1)
    if _bg_ok:
        _frozen_by_kind: dict = {}
        if hgt_z250_abs_cyc is not None:
            _frozen_by_kind['hgt250'] = hgt_z250_abs_cyc[min(n_dias - 1, hgt_z250_abs_cyc.shape[0] - 1)]
        if hgt_z500_abs_cyc is not None:
            _frozen_by_kind['hgt500'] = hgt_z500_abs_cyc[min(n_dias - 1, hgt_z500_abs_cyc.shape[0] - 1)]
        ctx['hgt_jato_frozen'] = _frozen_by_kind
        ctx['_bg_from'] = int(_bg_from)
        # Fundo (sem jato, sem caixa) do 1o frame congelado — _bg_arr ainda None => render completo.
        ctx['_bg_arr'] = _build_frame(int(_bg_from), ctx, skip_jet=True, skip_overlay=True,
                                      as_float=True)
        logger.info('Cache de fundo ATIVO: fundo 1x + jato por frame nos frames congelados '
                    '(>= {} de {})', _bg_from, total_frames)
        # Autoteste opcional: renderiza 1 frame congelado do jeito ANTIGO (completo) e pelo cache,
        # e loga a diferenca maxima de pixel (0 = identico). Prova que nao ha impacto na imagem/jato.
        if bool(settings.get('GLOBO_3D_BG_CACHE_CHECK', False)):
            _fc = int(_bg_from) + 1
            _bg_keep = ctx['_bg_arr']
            ctx['_bg_arr'] = None                       # forca render completo
            _full = _build_frame(_fc, ctx)
            ctx['_bg_arr'] = _bg_keep                   # reativa o cache
            _fast = _build_frame(_fc, ctx)
            _d = np.abs(_full.astype(np.int16) - _fast.astype(np.int16))
            logger.info('BG cache self-check frame {}: max|diff|={} media|diff|={:.5f} '
                        '(0 = pixel-identico)', _fc, int(_d.max()), float(_d.mean()))

    # Seta o ctx como global ANTES do fork: filhos herdam via CoW sem pickle dos arrays.
    # Sem isso, cada worker receberia uma cópia completa de ctx (~150 MB) via IPC pipe.
    global _FRAME_CTX
    _FRAME_CTX = ctx

    try:
        if workers > 1:
            # Render paralelo: fork herda _FRAME_CTX via CoW — sem initargs, sem pickle.
            # maxtasksperchild recicla o worker a cada N frames: mata e recria o processo,
            # devolvendo ao SO a RAM que matplotlib/cartopy acumulam por frame (raster 3600²
            # do contourf + reprojecao do imshow + paths de contorno da 2ª variavel). Sem isso,
            # o worker vive o clipe inteiro e o RSS cresce ate estourar (OOM ~frame 80/136 no WSL).
            _maxtasks = int(settings.get('GLOBO_3D_MAXTASKS', 20)) or None
            pool = get_context('fork').Pool(workers, maxtasksperchild=_maxtasks)
            # JANELA DESLIZANTE (limita o pico de RAM): `pool.imap(range(total_frames))` despachava
            # TODOS os frames de uma vez e os workers, em paralelo, corriam na FRENTE do consumo serial
            # (`_emit` -> writer libx264, mais lento que o render). Os frames RGBA prontos (px²×4 bytes)
            # se acumulavam SEM LIMITE no buffer do imap -> quanto mais rapida a maquina, MAIOR a pilha
            # (por isso ate PC potente estourava, ainda mais sob o teto de RAM do WSL2). Aqui mantemos no
            # maximo `workers + GLOBO_3D_RENDER_BUFFER` frames em voo: submete-se um novo apply_async so
            # depois de consumir o proximo em ordem (deque FIFO = ordem de submissao = ordem do frame).
            _buffer = max(1, int(settings.get('GLOBO_3D_RENDER_BUFFER', 2)))
            _max_inflight = workers + _buffer
            try:
                inflight: deque = deque()
                next_submit = 0
                for i in range(total_frames):
                    while next_submit < total_frames and len(inflight) < _max_inflight:
                        inflight.append(pool.apply_async(_render_one_frame, (next_submit,)))
                        next_submit += 1
                    _emit(i, inflight.popleft().get())
                    if (i + 1) % 20 == 0 or i == total_frames - 1:
                        logger.info('  frame {}/{}', i + 1, total_frames)
            finally:
                pool.close()
                pool.join()
        else:
            for f in range(total_frames):
                _emit(f, _build_frame(f, ctx))
                if (f + 1) % 20 == 0 or f == total_frames - 1:
                    logger.info('  frame {}/{}', f + 1, total_frames)
        # CONGELAMENTO do ultimo frame (GLOBO_3D_ULTIMO_FRAME_SEG): repete o ultimo frame RENDERIZADO
        # (sem fade -- ver `_emit`) por N s. Como `_clipe_frames` ja conta estes frames, o `_fator_fade`
        # deixa o brilho CHEIO no comeco do congelamento e so escurece nos ultimos `_fade_out` frames
        # DELE -> o quadro final e' exibido inteiro antes de sumir.
        if _ultimo_hold > 0 and _emit_state.get('ultimo') is not None:
            _ult = _emit_state['ultimo']
            for _j in range(_ultimo_hold):
                _emit(total_frames + _j, _ult)
        # HOLD do fade-out: segura N frames PRETOS no fim (o ultimo frame renderizado ja saiu preto
        # pelo fade). So MP4 -- _fade_hold > 0 implica writer != None.
        if _fade_hold > 0 and _emit_state.get('shape') is not None:
            _black = np.zeros(_emit_state['shape'], dtype=_emit_state['dtype'])
            for _ in range(_fade_hold):
                writer.append_data(_black)
    finally:
        if writer is not None:
            writer.close()
        _FRAME_CTX = None  # libera referência após renderização

    if gif:
        # Paleta UNICA compartilhada entre os frames (ver _save_gif_paleta_unica) -- evita o
        # flicker de cor do mimsave (paleta por frame independente).
        _save_gif_paleta_unica(frames_buf, out_path, fps_out)

    logger.info('{} salvo: {} ({:.1f}s)', 'GIF' if gif else 'MP4', out_path, _time.time() - t0)
    return out_path


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
        # ACUM_HORARIO (settings, default false): troca a CHUVA (precip_abs) de acumulado DIARIO
        # p/ acumulado no PASSO NATIVO do modelo (3h/6h ECMWF, 6h GFS) -- anima hora a hora em vez
        # de dia a dia. So se aplica a `precip_abs`; as demais fichas seguem no pipeline generico
        # de sempre. Ver `_precip_native_forecast_series` (motivo de nao usar `_build_var_series`
        # aqui: essa serie precisa do passo nativo cru, nao do resample diario).
        _acum_horario = (item['var'] == 'precip_abs' and bool(settings.get('ACUM_HORARIO', False))
                         and item['model'] in ('ecmwf', 'gfs'))
        if _acum_horario:
            serie = _precip_native_forecast_series(item['model'], dt_ini, dt_fim)
            serie_mp4 = serie
        else:
            serie = _build_var_series(ficha, item['model'], dt_ini, dt_fim)
            # Serie SINOTICA (00/06/12/18Z, sem media diaria) para o MP4 de fichas com
            # `sinotico_mp4=True` (ex.: z250_abs). GIF/PNG continuam usando `serie`/`serie_m`
            # (media DIARIA de sempre) mais abaixo -- so o MP4 troca de eixo temporal.
            serie_mp4 = (_build_var_series_synoptic(ficha, item['model'], dt_ini, dt_fim)
                         if ficha.get('sinotico_mp4') else serie)
        # Camada OPCIONAL de neve por cima da chuva (`NEVE` no settings, so ficha com `neve_levels`
        # e so ECMWF/GFS -- unicos modelos com downloader de tipo de precipitacao, ver
        # `_ptype_neve_forecast_series`). NAO mexe em `serie`/`serie_mp4`: a chuva continua vindo do
        # pipeline acima (diario ou nativo, ver `_acum_horario`); a neve e' SEMPRE calculada no passo
        # nativo (precisa da classificacao por passo) -- quando `ACUM_HORARIO=false` ela so acaba
        # amostrada nos poucos frames DIARIOS do overlay (ver `_render_clip`,
        # `neve_serie.sel(..., method='nearest')`), o que já É o acumulado diário.
        neve_serie_nativa = None
        if ('neve_levels' in ficha and bool(settings.get('NEVE', False))
                and item['model'] in ('ecmwf', 'gfs')):
            neve_serie_nativa = _ptype_neve_forecast_series(item['model'], dt_ini, dt_fim)
        # Z250 e/ou Z500 (altura geopotencial), baixados conforme o NIVEL de CADA jato configurado
        # (GLOBO_3D_JET_STREAM_NIVEL/SUBTROPICAL_JET_NIVEL sao a fonte da verdade -- ver
        # _jato_kind_from_nivel): jato pode ser plotado sobre QUALQUER campo shaded, independente
        # da escala do nivel escolhido. Isolinhas (GLOBO_3D_ISOL_HGT_JET_STREAM/HGT250_ABS) sao
        # sempre Z250. Se o modelo nao tiver o nivel, avisa e essa saida fica sem aquele jato
        # (hgt_anom_serie[_500]=None). EXCECAO: quando a propria variavel JA E uma altura
        # geopotencial absoluta do MESMO kind (ficha['absoluto'] + kind in ('hgt250','hgt500')), o
        # campo-guia e o PROPRIO campo shaded -- `_render_clip` reusa `vals_cyc` direto, sem
        # download nem reconstrucao extra.
        hgt_anom_serie = None
        hgt_anom_serie_500 = None
        _e_kind_absoluto = (ficha['spec'].get('kind')
                           if (bool(ficha.get('absoluto')) and ficha['spec'].get('kind') in ('hgt250', 'hgt500'))
                           else None)
        _jato_master_g = bool(_script_setting(script_id, 'JATO', False))
        _kinds_jato_needed: set[str] = set()
        if _jato_master_g:
            if bool(_script_setting(script_id, 'JET_STREAM', True)):
                _kinds_jato_needed.add(_jato_kind_from_nivel(
                    float(_script_setting(script_id, 'JET_STREAM_NIVEL', 10200.0))))
            if bool(_script_setting(script_id, 'SUBTROPICAL_JET', False)):
                _kinds_jato_needed.add(_jato_kind_from_nivel(
                    float(_script_setting(script_id, 'SUBTROPICAL_JET_NIVEL', 10600.0))))
        _isol_precisa = (item['var'] == 'jet_stream'
                         and bool(settings.get('GLOBO_3D_ISOL_HGT_JET_STREAM', False)))
        # Isolinhas de Z250 absoluto sobre QUALQUER campo (GLOBO_3D_ISOL_HGT250_ABS): precisa do Z250
        # dedicado, exceto nas fichas nativas de Z250 (isolinha_hgt_abs) que reconstroem via clim propria.
        _isol_abs_precisa = (bool(settings.get('GLOBO_3D_ISOL_HGT250_ABS', False))
                             and not ficha.get('isolinha_hgt_abs'))
        _precisa_hgt250 = ('hgt250' in _kinds_jato_needed) or _isol_precisa or _isol_abs_precisa
        _precisa_hgt500 = 'hgt500' in _kinds_jato_needed
        if _precisa_hgt250 and _e_kind_absoluto != 'hgt250':
            try:
                logger.info('{}: carregando z250_anom para Z250 absoluto (jato/isolinhas)', item['var'])
                hgt_anom_serie = _build_var_series(VARIAVEIS['z250_anom'], item['model'], dt_ini, dt_fim)
            except Exception as _e:
                logger.warning('⚠ Modelo "{}" NÃO tem Z250 (altura geopotencial 250 hPa) — a corrente '
                               'de jato NÃO sera plotada (Z250) para {}. Detalhe: {}',
                               item['model'], item['var'], _e)
                hgt_anom_serie = None
        if _precisa_hgt500 and _e_kind_absoluto != 'hgt500':
            try:
                logger.info('{}: carregando z500_anom para Z500 absoluto (jato)', item['var'])
                hgt_anom_serie_500 = _build_var_series(VARIAVEIS['z500_anom'], item['model'], dt_ini, dt_fim)
            except Exception as _e:
                logger.warning('⚠ Modelo "{}" NÃO tem Z500 (altura geopotencial 500 hPa) — a corrente '
                               'de jato NÃO sera plotada (Z500) para {}. Detalhe: {}',
                               item['model'], item['var'], _e)
                hgt_anom_serie_500 = None
        mslp_serie = None
        if _isolinha_mslp_ativa(ficha, item['var']):
            try:
                mslp_serie = (_mslp_forecast_series_native(item['model'], dt_ini, dt_fim)
                             if (ficha.get('passo_nativo_mp4') or _acum_horario) else
                             _build_mslp_series(item['model'], dt_ini, dt_fim))
            except Exception as _e:
                logger.warning('MSLP nao disponivel para {}: {}', item['var'], _e)
        # Camada opcional de OLR equatorial sobreposta (GLOBO_3D_OLR_OVERLAY), exceto na propria OLR.
        olr_serie = None
        if bool(settings.get('GLOBO_3D_OLR_OVERLAY', False)) and item['var'] != 'olr_anom':
            try:
                logger.info('Carregando OLR para overlay equatorial sobre {}', item['var'])
                olr_serie = _build_var_series(VARIAVEIS['olr_anom'], item['model'], dt_ini, dt_fim)
            except Exception as _e:
                logger.warning('OLR overlay indisponivel para {}: {}', item['var'], _e)
        # Serie auxiliar plotada como ISOLINHAS (ex.: psi200 preto sobre chi200 shaded).
        # Construida pelo mesmo motor -> herda o mesmo modo (reanalise/forecast) e a pentada.
        contour_serie = None
        _cs_var = ficha.get('contorno_serie_var')
        if _cs_var:
            logger.info('{}: carregando {} para isolinhas auxiliares', item['var'], _cs_var)
            # `contorno_serie_sem_pentada`: forca o contorno DIARIO (ignora a pentada movel propria da
            # ficha do contorno) -- p/ quando o shaded e diario (ex.: jet_stream_psi200_contour), senao
            # a pentada de 5 dias do psi200 encurtaria a serie e a isolinha congelaria no fim do clipe.
            _cs_sem_pent = bool(ficha.get('contorno_serie_sem_pentada', False))
            contour_serie = _build_var_series(VARIAVEIS[_cs_var], item['model'], dt_ini, dt_fim,
                                              aplicar_pentada=not _cs_sem_pent)
            # Pentada movel do CONTORNO seguindo a ficha-PAI: campos como z250_anom nao tem pentada
            # propria, mas como isolinha sobre chi200/olr devem acompanhar a media movel do shaded.
            # So aplica se a serie ainda nao veio pentada'da (ex.: psi200_anom ja tem pentada propria).
            _cs_pent = int(ficha.get('contorno_serie_pentada', 0) or 0)
            if _cs_pent > 1 and 'pentada_dias' not in contour_serie.attrs:
                contour_serie = _pentada_movel_serie(contour_serie, _cs_pent, _cs_var)
        # MEDIA do periodo (usada por PNG/GIF do s41/s42, pelo PNG-resumo do s38/s39, e opcionalmente
        # pelo proprio MP4 -- ver GLOBO_3D_MP4_MEDIA_FIXA abaixo). Calculada ANTES do MP4 pois o MP4
        # pode precisar dela. s38/s39: 2a saida PNG com a media do periodo animado (GLOBO_3D_PNG_MEDIA,
        # default true) -- resume num quadro so o mesmo intervalo DATA_INICIAL..DATA_FINAL do MP4.
        _png_media_on = bool(settings.get('GLOBO_3D_PNG_MEDIA', True))
        _gif_media_on = bool(settings.get('GLOBO_3D_GIF_MEDIA', True))   # 3a saida (GIF do resumo)
        _quer_media = (script_id in ('s41', 's42', 's44', 's46', 's47')
                       or (script_id in ('s38', 's39') and _png_media_on))
        serie_m = _hgt_m = _hgt_m_500 = _mslp_m = _olr_m = _cont_m = _cam = _neve_m = None
        if _quer_media:
            _dts = pd.DatetimeIndex(pd.to_datetime(serie['time'].values))
            # rotulo_dias = DIAS cobertos pelo periodo, NAO nº de frames: a pentada movel ja colapsou a
            # serie (janela de 5 dias vira 1 frame), entao len(_dts) daria 1 e o PNG-media sairia rotulado
            # com UM dia so ("17" em vez de "17-21"). Span real = do 1o ao ultimo frame + (pentada-1) dias
            # que o ultimo frame pentado ainda cobre. Sem pentada, cai no nº de dias diarios de sempre.
            _pent = int(serie.attrs.get('pentada_dias', 0) or 0)
            _n = (_dts[-1].date() - _dts[0].date()).days + 1 + max(_pent - 1, 0)
            _mean = lambda s: _agg_estatico(s, _dts[0], _dts[-1], _n)  # noqa: E731
            serie_m = _mean(serie)
            if serie_m is not None:
                _cp = _camera_path(2, script_id)          # camera de assentamento (s41/s42 = fixa)
                _cam = (float(_cp[0][-1]), float(_cp[1][-1]))
                _hgt_m, _mslp_m = _mean(hgt_anom_serie), _mean(mslp_serie)
                _hgt_m_500 = _mean(hgt_anom_serie_500)
                _olr_m, _cont_m = _mean(olr_serie), _mean(contour_serie)
                _neve_m = _mean(neve_serie_nativa)

        # (1) MP4 do periodo: por padrao, voo da camera + evolucao dia a dia (ou sinotico p/
        # sinotico_mp4). GLOBO_3D_MP4_MEDIA_FIXA=true (s41/s42) troca isso por CAMPO MEDIO FIXO +
        # camera parada -- MP4 vira "GIF em formato de video" (mesmo campo/camera do GIF, so a
        # corrente de jato/icones/caixa livre em movimento). Com GLOBO_3D_JATO ligado, jato FLUINDO.
        _mp4_media_fixa = bool(script_id in ('s41', 's42')
                               and settings.get('GLOBO_3D_MP4_MEDIA_FIXA', False) and serie_m is not None)
        if _mp4_media_fixa:
            outputs.append(_render_clip(serie_m, ficha, item['var'], item['dir'], item['label'], script_id,
                                        _hgt_m, _mslp_m, _olr_m, _cont_m, camera=_cam,
                                        hgt_anom_serie_500=_hgt_m_500, neve_serie=_neve_m))
        else:
            outputs.append(_render_clip(serie_mp4, ficha, item['var'], item['dir'], item['label'], script_id,
                                        hgt_anom_serie, mslp_serie, olr_serie, contour_serie,
                                        hgt_anom_serie_500=hgt_anom_serie_500,
                                        neve_serie=neve_serie_nativa))

        # (2) MEDIA do periodo em PNG (estatico; jato PARADO se ligado): 2a saida que resume num quadro
        # so o mesmo intervalo DATA_INICIAL..DATA_FINAL animado no MP4. s41/s42 sempre; s38/s39 quando
        # GLOBO_3D_PNG_MEDIA (default true). Camera = ponto final do voo (vista de assentamento).
        if serie_m is not None:
            # Sufixo do arquivo = o que a figura REALMENTE mostra: '_total' quando a serie e um
            # running total de chuva (ACUMULAR_NO_TEMPO -> total do periodo), '_media' no resto.
            _sfx = 'total' if serie.attrs.get('acumulado_no_tempo') else 'media'
            _png = item['dir'] / f"{script_id}_{item['var']}_{_sfx}.png"
            outputs.append(_render_clip(
                serie_m, ficha, item['var'], item['dir'], item['label'], script_id,
                _hgt_m, _mslp_m, _olr_m, _cont_m,
                estatico=True, png_path=_png, camera=_cam, hgt_anom_serie_500=_hgt_m_500,
                neve_serie=_neve_m))
            # (3) So s41/s42/s44/s46, e so com GLOBO_3D_GIF_MEDIA (default true): alem do PNG, a
            # MEDIA/TOTAL tambem em GIF (campo fixo + 'JET STREAM'/setas deslizando W->E). No s38/s39 o
            # MP4 ja e a versao animada, entao o GIF seria redundante. O s46 desliga pelo header (o GIF
            # nao entra na entrega de alerta) -- knob em vez de tirar da tupla p/ o s44 seguir com o seu.
            if script_id in ('s41', 's42', 's44', 's46', 's47') and _gif_media_on:
                _gif = item['dir'] / f"{script_id}_{item['var']}_{_sfx}.gif"
                outputs.append(_render_clip(
                    serie_m, ficha, item['var'], item['dir'], item['label'], script_id,
                    _hgt_m, _mslp_m, _olr_m, _cont_m,
                    gif=True, gif_path=_gif, camera=_cam, hgt_anom_serie_500=_hgt_m_500,
                    neve_serie=_neve_m))
    return outputs


# ---------------------------------------------------------------------------
# Engine de FIGURAS ESTATICAS (s40) — mesma serie/motor do globo animado, mas a
# saida sao PNGs por agregacao (padrao do s34): diario, media movel, pentadas
# fixas e media do periodo todo. Cada figura e um frame estatico (camera fixa).
# ---------------------------------------------------------------------------
def _agg_estatico(serie: xr.DataArray | None, d0, d1, rotulo_dias: int) -> xr.DataArray | None:
    """Media de `serie` no intervalo de datas [d0, d1] -> DataArray de UM passo de tempo
    (coord = d0), preservando attrs. `rotulo_dias` > 1 marca a DATA como intervalo (pentada_dias),
    fazendo o render rotular 'Jul 1–5, 2026'; 1 = data unica. None/serie vazia -> None."""
    if serie is None:
        return None
    sub = serie.sel(time=slice(np.datetime64(d0), np.datetime64(d1)))
    if sub.sizes.get('time', 0) == 0:
        return None
    if serie.attrs.get('acumulado_no_tempo'):
        # Serie de CHUVA em running total (ACUMULAR_NO_TEMPO): a figura estatica do periodo e o
        # TOTAL = ultimo passo do cumsum, NAO a media (a media de um acumulado crescente daria
        # ~55% do total -- um campo que nunca existiu). O rotulo de intervalo (pentada_dias)
        # abaixo continua certo: e o total DAQUELE intervalo.
        m = sub.isel(time=[-1]).assign_coords(time=[np.datetime64(d0)])
    else:
        m = sub.mean('time', skipna=True).expand_dims(time=[np.datetime64(d0)])
    m.name = serie.name
    m.attrs.update(serie.attrs)  # preserva run_init (rotulo de rodada) e afins
    if rotulo_dias and rotulo_dias > 1:
        m.attrs['pentada_dias'] = int(rotulo_dias)
    else:
        m.attrs.pop('pentada_dias', None)
    return m


def gerar_figuras_estaticas(variaveis: list[str], output_base: Path,
                            script_id: str = 's40') -> list[Path]:
    """Gera FIGURAS ESTATICAS (PNG) do globo para uma LISTA de variaveis (s40).

    Mesmo modo automatico do globo animado (passado=reanalise, futuro=previsao, cruza hoje=emenda),
    mas em vez de 1 MP4 por variavel produz, no padrao do s34, quatro colecoes de PNGs por variavel:
      - diario/         : uma figura por dia
      - media_movel/    : media movel de MOV_AVG_DAYS dias (janelas deslizantes)
      - pentadas_fixas/ : pentadas FIXAS de 5 dias (p1, p2, ...) contiguas a partir de DATA_INICIAL
      - media_total/    : uma figura = media do periodo inteiro

    media_movel e pentadas_fixas so saem se houver >= GLOBO_3D_ESTATICO_MIN_DIAS dias (senao a
    colecao — e a pasta — simplesmente nao e criada; sem erro). diario e media_total saem sempre.
    """
    plano, dt_ini, dt_fim = _output_plan(variaveis, output_base)
    mov = int(settings.get('MOV_AVG_DAYS', 5)) or 5
    pent_dias = 5  # pentada fixa (padrao do projeto: PENTADA_DIAS)
    min_dias = int(settings.get('GLOBO_3D_ESTATICO_MIN_DIAS', 5))
    # Posicao/inclinacao do globo estatico: controlada por ORTHO_CENTRAL_LONGITUDE/LATITUDE
    # (NAO pelo voo GLOBO_3D_LON/LAT_INICIAL/FINAL, que so vale p/ os MP4 do s38/s39). Override
    # opcional por GLOBO_3D_ESTATICO_LON/LAT.
    camera = (
        float(settings.get('GLOBO_3D_ESTATICO_LON', getattr(settings, 'ORTHO_CENTRAL_LONGITUDE', -45.0))),
        float(settings.get('GLOBO_3D_ESTATICO_LAT', getattr(settings, 'ORTHO_CENTRAL_LATITUDE', -15.0))),
    )
    logger.info('=' * 70)
    logger.info('GLOBO 3D ESTATICO: {} a {} | {} colecao(oes) | camera fixa (lon {:.0f}, lat {:.0f})',
                dt_ini.date(), dt_fim.date(), len(plano), camera[0], camera[1])
    logger.info('=' * 70)

    outputs: list[Path] = []
    for item in plano:
        var, model = item['var'], item['model']
        ficha = VARIAVEIS[var]
        logger.info('--- {} | {} ({}) ---', item['label'], var, ficha['titulo'])
        # Serie DIARIA CRUA (sem a pentada da ficha) — o s40 faz suas proprias agregacoes.
        serie = _build_var_series(ficha, model, dt_ini, dt_fim, aplicar_pentada=False)
        # Series auxiliares (tambem CRUAS/diarias) — mesmos hooks do globo animado.
        hgt_anom_serie = None
        if var == 'jet_stream' and bool(settings.get('GLOBO_3D_ISOL_HGT_JET_STREAM', False)):
            hgt_anom_serie = _build_var_series(VARIAVEIS['z250_anom'], model, dt_ini, dt_fim,
                                               aplicar_pentada=False)
        mslp_serie = None
        if _isolinha_mslp_ativa(ficha, var):
            try:
                mslp_serie = _build_mslp_series(model, dt_ini, dt_fim)
            except Exception as _e:
                logger.warning('MSLP nao disponivel para {}: {}', var, _e)
        olr_serie = None
        if bool(settings.get('GLOBO_3D_OLR_OVERLAY', False)) and var != 'olr_anom':
            try:
                olr_serie = _build_var_series(VARIAVEIS['olr_anom'], model, dt_ini, dt_fim,
                                              aplicar_pentada=False)
            except Exception as _e:
                logger.warning('OLR overlay indisponivel para {}: {}', var, _e)
        contour_serie = None
        _cs_var = ficha.get('contorno_serie_var')
        if _cs_var:
            contour_serie = _build_var_series(VARIAVEIS[_cs_var], model, dt_ini, dt_fim,
                                              aplicar_pentada=False)

        dates = pd.DatetimeIndex(pd.to_datetime(serie['time'].values))
        n = len(dates)
        base = item['dir'] / var
        logger.info('{}: {} dia(s) diario(s) [{} a {}]', var, n, dates[0].date(), dates[-1].date())

        def _render(d0, d1, rotulo_dias, png_path):
            """Agrega shaded + auxiliares para [d0,d1] e renderiza 1 PNG estatico."""
            _shaded = _agg_estatico(serie, d0, d1, rotulo_dias)
            if _shaded is None:
                return
            outputs.append(_render_clip(
                _shaded, ficha, var, base, item['label'], script_id,
                _agg_estatico(hgt_anom_serie, d0, d1, rotulo_dias),
                _agg_estatico(mslp_serie, d0, d1, rotulo_dias),
                _agg_estatico(olr_serie, d0, d1, rotulo_dias),
                _agg_estatico(contour_serie, d0, d1, rotulo_dias),
                estatico=True, png_path=png_path, camera=camera))

        # ── DIARIO (sempre) ──
        for d in dates:
            _render(d, d, 1, base / 'diario' / f'{script_id}_{var}_{d:%Y%m%d}.png')

        # ── MEDIA TOTAL (sempre) ──
        _render(dates[0], dates[-1], n,
                base / 'media_total' /
                f'{script_id}_{var}_{dates[0]:%Y%m%d}_{dates[-1]:%Y%m%d}_media_total.png')

        # ── MEDIA MOVEL (so com dados suficientes) ──
        if n >= max(min_dias, mov):
            for i in range(n - mov + 1):
                d0, d1 = dates[i], dates[i + mov - 1]
                _render(d0, d1, mov,
                        base / 'media_movel' /
                        f'{script_id}_{var}_{d0:%Y%m%d}_{d1:%Y%m%d}_mm{mov}d.png')
        else:
            logger.info('{}: media movel pulada — {} dia(s) < minimo {} (sem pasta media_movel).',
                        var, n, max(min_dias, mov))

        # ── PENTADAS FIXAS de 5 dias (so com dados suficientes) ──
        if n >= max(min_dias, pent_dias):
            n_pent = n // pent_dias
            for k in range(n_pent):
                d0, d1 = dates[k * pent_dias], dates[k * pent_dias + pent_dias - 1]
                _render(d0, d1, pent_dias,
                        base / 'pentadas_fixas' /
                        f'{script_id}_{var}_{d0:%Y%m%d}_{d1:%Y%m%d}_p{k + 1}.png')
        else:
            logger.info('{}: pentadas puladas — {} dia(s) < minimo {} (sem pasta pentadas_fixas).',
                        var, n, max(min_dias, pent_dias))
    return outputs
