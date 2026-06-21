"""s34 - Guia de onda de Rossby: numero de onda estacionario (Ks) em 200 hPa.

Diagnostico de waveguide de Rossby seguindo Hoskins & Ambrizzi (1993). Calcula o
numero de onda estacionario Ks a partir do vento zonal medio basico em 200 hPa e
o sobrepoe a anomalia do vento meridional v'200 (a onda real), permitindo um
diagnostico CONCLUSIVO de propagacao/estacionariedade de ondas de Rossby.

Por que dois campos:
  - Ks (meio): diz ONDE uma onda estacionaria PODE existir (waveguide). Maximos de
    Ks = nucleos de jato (guias); regioes de leste/evanescentes = sem onda
    estacionaria (mascaradas/hachuradas).
  - v'200 (onda real): diz se a onda ESTA la. No mapa, o trem de ondas; no
    Hovmoller, bandas verticais (estacionaria) vs inclinadas (propagante).

No MAPA, a onda real e mostrada por: anomalia de altura geopotencial Z200
(isolinhas) + vetores de WAF (Takaya & Nakamura 2001, direcao da propagacao de
energia). Pareamento classico de waveguide (Hoskins & Ambrizzi; Takaya &
Nakamura). Nos HOVMOLLERS, a onda e a anomalia de vento meridional v'200.

Plotagem padronizada conforme o s07 (areas de plotagem, features, posicao das
barras de cor e tamanho dos ticks).

Modos (MODE): 'reanalysis' (ERA5/GDAS por datas) ou 'forecast'. Em forecast, os modelos
sao habilitados por flags (RUN_GFS/RUN_GEFS) e o pipeline roda para CADA modelo habilitado,
salvando em Saida/<MODELO>/. GFS = deterministico 0.25°; GEFS = media do ensemble 0.5°.

Pipeline:
  1. Download ERA5/GDAS (reanalise) ou GFS/GEFS (forecast) de u/v 200 hPa
  2. Serie diaria u/v200 regridada para a grade da LTM NCEP (2.5°)
  3. Anomalia diaria v'200 = v200 - LTM_diaria(dia-do-ano)
  4. Ks (Hoskins & Ambrizzi 1993) do vento zonal medio do periodo
  5. WAF 200 hPa: media de hgt (ERA5/GDAS) - clim PSL, anomalia Z200 e tnflux.tnf2d
  6. Mapa (por area): Ks sombreado + branco (leste) + U=0 + Z200 anom + WAF
  7. Hovmollers: v'200 medio em 2 faixas de jato (subtropical + polar)
  8. Mapas por PENTADA: 3 janelas FIXAS de 5 dias a partir do dia seguinte a data
     da rodada (sufixo pentadaX) — mesma colecao de mapas da janela movel

Saida:
    - PNG (mapas por area, janela movel) + PNG por pentada (sufixo pentadaX)
      + 2 PNG (Hovmollers) + NetCDF em Saida/s34_TELECONEXAO/<MODO>/<CONTINENTE>/<MODELO>/

Criado em: 2026-06-14
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use('Agg')  # backend headless ANTES do pyplot: sem Tk, o download em threads (ECMWF-ENS) nao crasha
import matplotlib.dates as mdates
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.util import add_cyclic_point as _acp
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from PIL import Image

import tnflux

from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.common.dataset_utils import area_display_name
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.src.uteis.clim_diaria_olr import clim_olr_daily, olr_obs_daily
from app.src.uteis.clim_diaria_uv200_ltm import (
    clim_hgt200_daily,
    clim_hgt500_daily,
    clim_t2m_daily,
    clim_t850_daily,
    clim_u850_daily,
    clim_uv200_daily,
    clim_v850_daily,
)
from app.src.uteis.chi200_intrasazonal import chi_from_wind
from app.src.uteis.downloaders_cfs_ensemble import (
    CFS_LEAD_DAYS,
    ensure_cfs_fcst200_uvz_for_period,
    ensure_cfs_hgt500_for_period,
    ensure_cfs_olr_for_period,
    ensure_cfs_t2m_for_period,
    ensure_cfs_uv250_for_period,
    ensure_cfs_uv850_for_period,
)
from app.src.uteis.downloaders_ai_nomads import (
    ensure_aigefs_fcst200_for_period,
    ensure_aigefs_hgt250_fcst_for_period,
    ensure_aigefs_hgt500_fcst_for_period,
    ensure_aigefs_t2m_fcst_for_period,
    ensure_aigefs_tmp850_fcst_for_period,
    ensure_aigefs_uv250_fcst_for_period,
    ensure_aigefs_uv850_fcst_for_period,
    ensure_aigfs_fcst200_for_period,
    ensure_aigfs_hgt250_fcst_for_period,
    ensure_aigfs_hgt500_fcst_for_period,
    ensure_aigfs_t2m_fcst_for_period,
    ensure_aigfs_tmp850_fcst_for_period,
    ensure_aigfs_uv250_fcst_for_period,
    ensure_aigfs_uv850_fcst_for_period,
)
from app.src.uteis.downloaders_aifs_ens import (
    ensure_aifs_ens_fcst200_for_period,
    ensure_aifs_ens_hgt250_fcst_for_period,
    ensure_aifs_ens_hgt500_fcst_for_period,
    ensure_aifs_ens_t2m_fcst_for_period,
    ensure_aifs_ens_tmp850_fcst_for_period,
    ensure_aifs_ens_uv250_fcst_for_period,
    ensure_aifs_ens_uv850_fcst_for_period,
)
from app.src.uteis.downloaders_aifs_fcst200 import ensure_aifs_fcst200_for_period
from app.src.uteis.downloaders_aifs_hgt250 import ensure_aifs_hgt250_fcst_for_period
from app.src.uteis.downloaders_aifs_hgt500 import ensure_aifs_hgt500_fcst_for_period
from app.src.uteis.downloaders_aifs_t2m import ensure_aifs_t2m_fcst_for_period
from app.src.uteis.downloaders_aifs_tmp850 import ensure_aifs_tmp850_fcst_for_period
from app.src.uteis.downloaders_aifs_uv250 import ensure_aifs_uv250_fcst_for_period
from app.src.uteis.downloaders_aifs_uv850 import ensure_aifs_uv850_fcst_for_period
from app.src.uteis.downloaders_ecmwf_ens import (
    ensure_ecmwf_ens_fcst200_for_period,
    ensure_ecmwf_ens_hgt250_fcst_for_period,
    ensure_ecmwf_ens_hgt500_fcst_for_period,
    ensure_ecmwf_ens_olr_fcst_for_period,
    ensure_ecmwf_ens_t2m_fcst_for_period,
    ensure_ecmwf_ens_tmp850_fcst_for_period,
    ensure_ecmwf_ens_uv250_fcst_for_period,
    ensure_ecmwf_ens_uv850_fcst_for_period,
)
from app.src.uteis.downloaders_ecmwf_fcst200 import ensure_ecmwf_fcst200_for_period
from app.src.uteis.downloaders_ecmwf_hgt250 import ensure_ecmwf_hgt250_fcst_for_period
from app.src.uteis.downloaders_ecmwf_hgt500 import ensure_ecmwf_hgt500_fcst_for_period
from app.src.uteis.downloaders_ecmwf_olr import ensure_ecmwf_olr_fcst_for_period
from app.src.uteis.downloaders_ecmwf_t2m import ensure_ecmwf_t2m_fcst_for_period
from app.src.uteis.downloaders_ecmwf_tmp850 import ensure_ecmwf_tmp850_fcst_for_period
from app.src.uteis.downloaders_ecmwf_uv250 import ensure_ecmwf_uv250_fcst_for_period
from app.src.uteis.downloaders_ecmwf_uv850 import ensure_ecmwf_uv850_fcst_for_period
from app.src.uteis.downloaders_era5_superficie import ensure_era5_superficie_for_period
from app.src.uteis.downloaders_gdas_hgt200 import ensure_gdas_hgt200_for_period
from app.src.uteis.downloaders_gdas_hgt250 import ensure_gdas_hgt250_for_period
from app.src.uteis.downloaders_gdas_hgt500 import ensure_gdas_hgt500_for_period
from app.src.uteis.downloaders_gdas_t2m import ensure_gdas_t2m_for_period
from app.src.uteis.downloaders_gdas_tmp850 import ensure_gdas_tmp850_for_period
from app.src.uteis.downloaders_gdas_uv250 import ensure_gdas_uv250_for_period
from app.src.uteis.downloaders_gdas_uv850 import ensure_gdas_uv850_for_period
from app.src.uteis.downloaders_gefs_fcst200 import ensure_gefs_fcst200_for_period
from app.src.uteis.downloaders_gefs_hgt250 import ensure_gefs_hgt250_fcst_for_period
from app.src.uteis.downloaders_gefs_hgt500 import ensure_gefs_hgt500_fcst_for_period
from app.src.uteis.downloaders_gefs_olr import ensure_gefs_olr_fcst_for_period
from app.src.uteis.downloaders_gefs_t2m import ensure_gefs_t2m_fcst_for_period
from app.src.uteis.downloaders_gefs_tmp850 import ensure_gefs_tmp850_fcst_for_period
from app.src.uteis.downloaders_gefs_uv250 import ensure_gefs_uv250_fcst_for_period
from app.src.uteis.downloaders_gefs_uv850 import ensure_gefs_uv850_fcst_for_period
from app.src.uteis.downloaders_gfs_hgt250 import ensure_gfs_hgt250_fcst_for_period
from app.src.uteis.downloaders_gfs_hgt500 import ensure_gfs_hgt500_fcst_for_period
from app.src.uteis.downloaders_gfs_olr import ensure_gfs_olr_fcst_for_period
from app.src.uteis.downloaders_gfs_t2m import ensure_gfs_t2m_fcst_for_period
from app.src.uteis.downloaders_gfs_tmp850 import ensure_gfs_tmp850_fcst_for_period
from app.src.uteis.downloaders_gfs_uv250 import ensure_gfs_uv250_fcst_for_period
from app.src.uteis.downloaders_gfs_uv850 import ensure_gfs_uv850_fcst_for_period
from app.src.uteis.downloaders_nomads_combo import (
    ensure_gefs_combo_for_period,
    ensure_gfs_combo_for_period,
)
from app.src.uteis.downloaders_tmp850_ERA5 import ensure_era5_t850_for_period
from app.src.uteis.downloaders_gdas_uv200 import ensure_gdas_uv200_for_period
from app.src.uteis.downloaders_gfs_fcst200 import ensure_gfs_fcst200_for_period
from app.src.uteis.downloaders_hgt200_ERA5 import ensure_era5_hgt200_for_period
from app.src.uteis.downloaders_hgt500_ERA5 import (
    ensure_era5_altura_geopotencial_500_global_for_period_grib,
)
from app.src.uteis.downloaders_wind250 import ensure_era5_uv250_for_period
from app.src.uteis.downloaders_wind850 import ensure_era5_uv850_for_period
from app.src.uteis.downloaders_z250_era5 import ensure_era5_z250_for_period
from app.src.uteis.downloaders_wind200 import ensure_era5_uv200_for_period
from app.src.uteis.plot_psi200 import (
    _compute_rotational_wind,
    _compute_vorticity,
    _solve_poisson_sphere,
)
from app.src.uteis.plot_rossby_waf import G, POLAR_MASK_LAT, TROPICAL_MASK_LAT, WAF_GRID_SPACING
from app.src.uteis.rossby_wave_source import rossby_wave_source
from app.src.uteis.stationary_wavenumber import stationary_wavenumber

# ---------------------------------------------------------------------------
# Identidade do script
# ---------------------------------------------------------------------------
SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's34'
SCRIPT_NAME = Path(__file__).stem
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
ERA5_LATENCY_DAYS = 7
DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)

# Modelos de previsao suportados (MODE='forecast'): nome -> (200hPa, OLR, T850, HGT500, HGT250,
# UV250, UV850, T2M). GFS=det 0.25° (NOMADS); GEFS=media ensemble (geavg) 0.5°; ECMWF=HRES (open
# data). Interfaces identicas (init, lead_hours, hours). Flags _MODEL_FLAGS (RUN_GFS/RUN_GEFS/...).
# HGT250=isolinhas jato no psi_jato; UV250=jato no chi_jato; UV850+T2M=campo t2m_wnd850.
_FCST_DOWNLOADERS = {
    'gfs': (ensure_gfs_fcst200_for_period, ensure_gfs_olr_fcst_for_period, ensure_gfs_tmp850_fcst_for_period, ensure_gfs_hgt500_fcst_for_period, ensure_gfs_hgt250_fcst_for_period, ensure_gfs_uv250_fcst_for_period, ensure_gfs_uv850_fcst_for_period, ensure_gfs_t2m_fcst_for_period),
    'gefs': (ensure_gefs_fcst200_for_period, ensure_gefs_olr_fcst_for_period, ensure_gefs_tmp850_fcst_for_period, ensure_gefs_hgt500_fcst_for_period, ensure_gefs_hgt250_fcst_for_period, ensure_gefs_uv250_fcst_for_period, ensure_gefs_uv850_fcst_for_period, ensure_gefs_t2m_fcst_for_period),
    'ecmwf': (ensure_ecmwf_fcst200_for_period, ensure_ecmwf_olr_fcst_for_period, ensure_ecmwf_tmp850_fcst_for_period, ensure_ecmwf_hgt500_fcst_for_period, ensure_ecmwf_hgt250_fcst_for_period, ensure_ecmwf_uv250_fcst_for_period, ensure_ecmwf_uv850_fcst_for_period, ensure_ecmwf_t2m_fcst_for_period),
    'ecmwf_ens': (ensure_ecmwf_ens_fcst200_for_period, ensure_ecmwf_ens_olr_fcst_for_period, ensure_ecmwf_ens_tmp850_fcst_for_period, ensure_ecmwf_ens_hgt500_fcst_for_period, ensure_ecmwf_ens_hgt250_fcst_for_period, ensure_ecmwf_ens_uv250_fcst_for_period, ensure_ecmwf_ens_uv850_fcst_for_period, ensure_ecmwf_ens_t2m_fcst_for_period),
    # AIFS (IA do ECMWF): olr_fn=None -> sem mapa de OLR (o AIFS nao emite radiacao no topo).
    'aifs': (ensure_aifs_fcst200_for_period, None, ensure_aifs_tmp850_fcst_for_period, ensure_aifs_hgt500_fcst_for_period, ensure_aifs_hgt250_fcst_for_period, ensure_aifs_uv250_fcst_for_period, ensure_aifs_uv850_fcst_for_period, ensure_aifs_t2m_fcst_for_period),
    'aifs_ens': (ensure_aifs_ens_fcst200_for_period, None, ensure_aifs_ens_tmp850_fcst_for_period, ensure_aifs_ens_hgt500_fcst_for_period, ensure_aifs_ens_hgt250_fcst_for_period, ensure_aifs_ens_uv250_fcst_for_period, ensure_aifs_ens_uv850_fcst_for_period, ensure_aifs_ens_t2m_fcst_for_period),
    # AIGFS/AIGEFS (IA do NCEP, NOMADS byte-range): tambem SEM OLR. AIGEFS usa a media `avg`.
    'aigfs': (ensure_aigfs_fcst200_for_period, None, ensure_aigfs_tmp850_fcst_for_period, ensure_aigfs_hgt500_fcst_for_period, ensure_aigfs_hgt250_fcst_for_period, ensure_aigfs_uv250_fcst_for_period, ensure_aigfs_uv850_fcst_for_period, ensure_aigfs_t2m_fcst_for_period),
    'aigefs': (ensure_aigefs_fcst200_for_period, None, ensure_aigefs_tmp850_fcst_for_period, ensure_aigefs_hgt500_fcst_for_period, ensure_aigefs_hgt250_fcst_for_period, ensure_aigefs_uv250_fcst_for_period, ensure_aigefs_uv850_fcst_for_period, ensure_aigefs_t2m_fcst_for_period),
    # CFS = pseudo-ensemble subsazonal (americano, ~45 dias). NAO tem z250 (hgt250) nem T850
    # (tmp850) -> esses 2 ficam None e os mapas correspondentes sao pulados (desacoplamento).
    'cfs': (ensure_cfs_fcst200_uvz_for_period, ensure_cfs_olr_for_period, None, ensure_cfs_hgt500_for_period, None, ensure_cfs_uv250_for_period, ensure_cfs_uv850_for_period, ensure_cfs_t2m_for_period),
}
# Pre-busca COMBINADA (1 requisicao/passo pega todos os niveis/variaveis do mesmo GRIB) — so para
# os modelos NOMADS via grib filter (GFS/GEFS). Roda ANTES dos downloaders por-variavel, que viram
# fallback/cache-hit. Reduz ~8x as requisicoes ao NOMADS (mata o throttle 302) e acelera. Os demais
# (ECMWF/AIFS open data, AIGFS/AIGEFS byte-range) nao entram aqui — usam seus proprios mecanismos.
_FCST_COMBO = {
    'gfs': ensure_gfs_combo_for_period,
    'gefs': ensure_gefs_combo_for_period,
}

# Flags (settings) que habilitam cada modelo em MODE='forecast'. Pode habilitar mais de um:
# o s34 roda o pipeline para CADA modelo habilitado, salvando em Saida/<MODELO>/. (default, flag)
# ECMWF_ENS = media dos 50 membros (pesado: ~50 downloads por campo/passo).
_MODEL_FLAGS = {
    'gfs': ('RUN_GFS', True), 'gefs': ('RUN_GEFS', False),
    'ecmwf': ('RUN_ECMWF', False), 'ecmwf_ens': ('RUN_ECMWF_ENS', False),
    'aifs': ('RUN_AIFS', False), 'aifs_ens': ('RUN_AIFS_ENS', False),
    'aigfs': ('RUN_AIGFS', False), 'aigefs': ('RUN_AIGEFS', False),
    'cfs': ('RUN_CFS', False),
}
# Continente do modelo, para separar a saida (americanos NCEP/NOAA vs europeus ECMWF).
_MODEL_CONTINENT = {
    'gfs': 'MODELOS_AMERICANOS', 'gefs': 'MODELOS_AMERICANOS',
    'aigfs': 'MODELOS_AMERICANOS', 'aigefs': 'MODELOS_AMERICANOS',
    'ecmwf': 'MODELOS_EUROPEUS', 'ecmwf_ens': 'MODELOS_EUROPEUS',
    'aifs': 'MODELOS_EUROPEUS', 'aifs_ens': 'MODELOS_EUROPEUS',
    'cfs': 'MODELOS_AMERICANOS',
}

# Areas de plotagem (mesmo padrao do s07/s04)
DEFAULT_AREAS = ['globo', 'psa', 'hemisferio_sul', 'hemisferio_norte', 'america_sul', 'globo_3d']

# Pentadas: N janelas FIXAS (nao moveis) de 5 dias a partir do dia SEGUINTE a data da
# rodada. Ex.: rodada dia 15 -> P1=16..20, P2=21..25, P3=26..30. Sufixo de arquivo pentadaX.
# Configuravel via setting (default 3; ate 7 com o GEFS 00Z de 35 dias). Precedencia:
# N_PENTADAS_FIXAS (override so do s34) -> N_PENTADAS (mesma do s31/s32) -> este default.
N_PENTADAS_FIXAS = 3
PENTADA_DIAS = 5
# PENTADA (janela FIXA) so e plotada se tiver >= este nº de dias com dado (4 de 5 ok; <4 pula).
# A janela MOVEL e mais estrita: so plota se COMPLETA (todos os dias presentes) — assim a media
# movel encerra exatamente no ultimo dia do modelo, sem rotulo ultrapassar o alcance (ver Etapa 5).
MIN_DIAS_PENTADA = 4

# Ks — niveis de isolinha/sombreado (numero de onda estacionario adimensional)
KS_LEVELS = [1, 2, 3, 4, 5, 6, 7, 8]

# Colormap do Ks: YlOrRd truncado em 0.78 para suavizar os tons de vinho do topo.
# O extend='max' usa a mesma cor do topo truncado (set_over), evitando reescurecer.
_ks_base = plt.get_cmap('YlOrRd')
KS_CMAP = ListedColormap(_ks_base(np.linspace(0.0, 0.78, 256)))
KS_CMAP.set_over(_ks_base(0.78))

# Z200 anomalia — isolinhas no mapa Ks (m); solido=positivo, tracejado=negativo
Z200_LEVELS = np.array([-200, -160, -120, -80, -40, 40, 80, 120, 160, 200], dtype=float)

# Z200 anomalia sombreada (mapa estilo s07: Z200 anom shaded + WAF) — paleta do projeto
Z200_SHADED_LEVELS = np.arange(-200, 220, 20)
Z200_SHADED_TICKS = np.arange(-200, 250, 50)

# Candidatos de nome de variavel nos arquivos
HGT_VARS = ('hgt', 'z', 'gh', 'geopotential')
OLR_VARS = ('olr', 'ulwrf', 'sulwrf')
T2M_VARS = ('t2m', '2t', 'air', 't', 'tmp')  # temperatura do ar a 2 m

# OLR anomalia (mapa estilo s08: OLR anom shaded BrBG_r + Z200 contornos + WAF)
OLR_LEVELS = np.arange(-40, 44, 4)        # W/m2
OLR_TICKS = np.arange(-40, 50, 10)

# T850 anomalia (mapa estilo s10: T850 anom shaded + Z200 contornos + WAF). Unidade °C.
TMP_VARS = ('t', 'tmp', 'air', 'temperature')
TMP850_LEVELS = np.arange(-8, 8.5, 0.5)
TMP850_TICKS = np.arange(-8, 9, 1)


def _to_celsius(arr: np.ndarray) -> np.ndarray:
    """Converte K->°C se o campo estiver em escala Kelvin (media > 100)."""
    return arr - 273.15 if np.nanmean(arr) > 100 else arr

# Hovmoller v'200 — niveis/ticks fixos e arredondados (sem decimais), paleta padrao do projeto
HOV_LEVELS = np.arange(-30, 33, 3)        # m/s — -30..30 (extend='both' captura |v'|>30)
HOV_TICKS = np.arange(-30, 31, 10)        # -30,-20,-10,0,10,20,30

# RWS (fonte de onda de Rossby) — anomala, em 10^-11 s^-2. Paleta do s05 (BrBG):
#   verde (RWS>0) = fonte anticiclonica no HS (lanca o trem); marrom (RWS<0) = ciclonica.
RWS_SCALE = 1e11
RWS_LEVELS = np.arange(-40, 44, 4)        # x10^-11 s^-2
RWS_TICKS = np.arange(-40, 41, 10)
RWS_MASK_LAT = 10.0                       # mascara |lat|<10° (f->0 deixa a RWS ruidosa)
RWS_MASK_LAT_POLE = 75.0                  # mascara |lat|>75° (singularidade polar ~1/cos(phi))
# Vento divergente: densidade media (campo ja suavizado evita o caos do bruto)
DIVWIND_QUIVER = {'step': 2, 'width': 0.0014, 'scale': 80.0, 'min_mag': 0.5}

# WAF (quiver) — normalizado pelo maximo, mascara fracos. Padrao do s07.
QUIVER_DEFAULTS = {
    'step': 2, 'width': 0.002, 'headwidth': 4.5, 'headlength': 6.0,
    'scale': None, 'scale_units': 'xy', 'min_amp_ratio': 0.05,
}
QUIVER_POR_AREA = {
    'globo': {'step': 2},
    'psa': {'step': 2},
    'hemisferio_sul': {'step': 2},
    'hemisferio_norte': {'step': 2},
    'america_sul': {'step': 1, 'width': 0.004, 'headwidth': 5.0, 'headlength': 7.0, 'scale': 0.12},
    'globo_3d': {'step': 2, 'width': 0.003, 'headwidth': 5.0, 'headlength': 7.0},
}

# ---------------------------------------------------------------------------
# Modalidades novas: hgt500 e psi_jato (PSI + magnitude do vento 200 + WAF)
# ---------------------------------------------------------------------------
# Z500 anomalia sombreada (mgp) + isolinhas da Z500 MEDIA do periodo (preto, sem rotulo).
Z500_ANOM_LEVELS = np.arange(-200, 220, 20)
Z500_ANOM_TICKS = np.arange(-200, 250, 50)
Z500_MEAN_LEVELS = np.arange(4800, 6060, 60)   # isolinhas de Z500 media (m), passo 60 m

# PSI200 anomalia sombreada — paleta vermelho->branco->azul do s04 (LST_ANOM_CORRETA_INVERTIDA_CHUVA).
PSI_SCALE = 1e6
PSI_PALETTE = [
    '#910e00', '#a71400', '#bd1c00', '#d32400', '#e92a00', '#ff4e18', '#ff6c30', '#ff8b48',
    '#ff9a55', '#ffb76c', '#ffe391', '#ffffa9', '#ffffff', '#ffffff', '#b4eff9', '#8adbf2',
    '#61c7eb', '#38b2e4', '#0faedf', '#00a0d2', '#0092c5', '#0084b8', '#0075ab', '#00679e',
    '#005890', '#004a83',
]


def _psi_levels_ticks_norm(psi_values):
    """Niveis/ticks/cmap/norm da anomalia de PSI (10^6 m^2/s), dinamico (igual ao s04)."""
    psi = np.asarray(psi_values)
    if np.all((psi >= -50) & (psi <= 50)):
        levels, ticks = np.arange(-50, 53, 3), np.arange(-50, 60, 10)
    elif np.all((psi >= -100) & (psi <= 100)):
        levels, ticks = np.arange(-100, 110, 10), np.arange(-100, 120, 20)
    else:
        levels, ticks = np.arange(-200, 220, 20), np.arange(-200, 250, 50)
    n_bins = len(levels) - 1
    cmap = LinearSegmentedColormap.from_list('psi200_red_white_blue', PSI_PALETTE, N=n_bins)
    norm = BoundaryNorm(levels, ncolors=n_bins, clip=False)
    return levels, ticks, cmap, norm


# Magnitude media do vento 200 hPa (m/s) sobreposta no psi_jato — paleta azul-rosa do s16.
# So plota o JATO: >= 35 m/s (abaixo do 1o nivel fica TRANSPARENTE via extend='max', deixando
# o PSI da base aparecer).
WIND_MAG_LEVELS = np.arange(35, 90, 5)
WIND_MAG_TICKS = np.arange(40, 90, 10)
WIND_MAG_PALETTE = [
    '#9cdafa', '#53bff7', '#5393f7', '#f2b0bf', '#f2849e', '#ee5278', '#f22457', '#c9c9c9', '#e7e7e7',
]

# Isolinhas de altura geopotencial 250 hPa (jato) sobrepostas no psi_jato — MESMA config do s16:
# azul nos niveis baixos (jato fraco) e vermelho nos altos (jato forte), com rotulo e placa branca.
Z250_LEVELS_BLUE = [9960.0, 10200.0]
Z250_LEVELS_RED = [10440.0, 10680.0]
Z250_FMT_BLUE = {9960.0: '9960', 10200.0: '10200'}
Z250_FMT_RED = {10440.0: '10440', 10680.0: '10680'}


# CHI200 (potencial de velocidade) anomalia sombreada — paleta verde-marrom do s03 (escala 1e5).
CHI_SCALE = 1e5
CHI200_LEVELS = np.arange(-60, 65, 5)
CHI200_TICKS = np.arange(-60, 65, 10)
CHI200_PALETTE = [
    '#005a45', '#0f7a6c', '#2e9b96', '#62bdb7', '#9dd8d2', '#dff3f1',
    '#f7f4eb', '#e7d9a9', '#d6b566', '#bd8a35', '#9a6313', '#6f4300',
]


def _style_contour_labels(txts) -> None:
    """Estilo 'placa branca' (bbox) nos rotulos das isolinhas (igual ao s16) + zorder alto.

    zorder 450: rotulo acima de tudo (shaded/streamlines/WAF), porem ABAIXO do logo (z=500)."""
    for txt in txts:
        txt.set_bbox(dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none', alpha=0.85))
        txt.set_zorder(450)


def _draw_z250_isolines(ax, z250_cyc, lon_z2, lat_z2, transform) -> None:
    """Isolinhas de altura geopotencial 250 hPa (jato) — MESMA config/valores do s16: azul (jato
    fraco) 9960/10200, vermelho (jato forte) 10440/10680, com rotulo e placa branca. zorder ALTO
    (acima do WAF e de tudo), porem ABAIXO do logo (z=500): linhas 400, rotulos 450. Usado no
    psi_jato e no chi_jato."""
    if z250_cyc is None:
        return
    cs_blue = ax.contour(lon_z2, lat_z2, z250_cyc, levels=Z250_LEVELS_BLUE, colors=['blue'],
                         linewidths=2.0, transform=transform, zorder=400)
    _style_contour_labels(ax.clabel(cs_blue, fmt=Z250_FMT_BLUE, inline=False, fontsize=12, colors=['blue']))
    cs_red = ax.contour(lon_z2, lat_z2, z250_cyc, levels=Z250_LEVELS_RED, colors=['red'],
                        linewidths=2.2, transform=transform, zorder=400)
    _style_contour_labels(ax.clabel(cs_red, fmt=Z250_FMT_RED, inline=False, fontsize=12, colors=['red']))

# Linhas de corrente (vento rotacional) do psi_jato — config por area (subconjunto do s04).
STREAMLINE_DEFAULTS = {'density': 2.0, 'linewidth': 0.5, 'arrowsize': 1.0, 'color': 'dimgray'}
STREAMLINE_POR_AREA = {
    'globo': {'density': 3},
    'psa': {'density': 3},
    'hemisferio_sul': {'density': 3},
    'hemisferio_norte': {'density': 3},
    'america_sul': {'density': 2},
}


def _get_streamline_config(area: str) -> dict:
    cfg = dict(STREAMLINE_DEFAULTS)
    if area in STREAMLINE_POR_AREA:
        cfg.update(STREAMLINE_POR_AREA[area])
    return cfg


_GEOP_THRESHOLD = 20000.0  # separa altura (m: ~5500@500, ~10400@250) de geopotencial (m^2/s^2: >5e4)


def _to_geop_height(arr: np.ndarray) -> np.ndarray:
    """Converte geopotencial (m^2/s^2) -> altura geopotencial (m), POR DIA (slice de tempo).

    Forecast/GDAS ja entregam altura (m); ERA5 entrega `z` (geopotencial). Na REANALISE a serie
    MISTURA fontes (ERA5 em m^2/s^2 + GDAS em m no mesmo array) — por isso a decisao ÷g e feita
    por slice de tempo (nao pela media global, que corromperia os dias da outra fonte)."""
    arr = np.asarray(arr, dtype='float64')
    if arr.ndim == 3:  # (time, lat, lon): normaliza cada dia conforme sua propria escala
        m = np.nanmean(arr, axis=(1, 2))
        factor = np.where(m > _GEOP_THRESHOLD, 1.0 / G, 1.0)
        return arr * factor[:, None, None]
    return arr / G if np.nanmean(arr) > _GEOP_THRESHOLD else arr


def _prepare_streamline_lonuv(lon_cyc, u_cyc, v_cyc, central_lon_mapa):
    """Desloca lon/u/v p/ a costura cair na borda do mapa (igual ao s04), evitando faixa vazia."""
    shifted = (lon_cyc - central_lon_mapa + 180) % 360 - 180
    order = np.argsort(shifted)
    lon_sorted = shifted[order]
    u_sorted, v_sorted = u_cyc[:, order], v_cyc[:, order]
    if len(lon_sorted) > 1 and np.isclose(lon_sorted[0], lon_sorted[1]):
        lon_sorted, u_sorted, v_sorted = lon_sorted[1:], u_sorted[:, 1:], v_sorted[:, 1:]
    elif len(lon_sorted) > 1 and np.isclose(lon_sorted[-1], lon_sorted[-2]):
        lon_sorted, u_sorted, v_sorted = lon_sorted[:-1], u_sorted[:, :-1], v_sorted[:, :-1]
    return lon_sorted, u_sorted, v_sorted


# Campo t2m_wnd850: anomalia de T2m (shaded, paleta de anomalia do projeto) + streamlines do
# vento 850 anomalo + isolinhas pretas da Z500 media (reusa Z500_MEAN_LEVELS do hgt500).
T2M_ANOM_LEVELS = np.arange(-10, 11, 1)   # °C
T2M_ANOM_TICKS = np.arange(-10, 12, 2)


def _psi_rotwind(u2d: np.ndarray, v2d: np.ndarray, lat: np.ndarray, lon: np.ndarray):
    """Streamfunction (m^2/s) e vento rotacional a partir de u/v 2D (reusa o solver do s04, sem logs)."""
    zeta = _compute_vorticity(u2d, v2d, lat, lon)
    psi = _solve_poisson_sphere(zeta, lat, lon)
    u_rot, v_rot = _compute_rotational_wind(psi, lat, lon)
    return psi, u_rot, v_rot


def _get_quiver_config(area: str) -> dict:
    cfg = dict(QUIVER_DEFAULTS)
    if area in QUIVER_POR_AREA:
        cfg.update(QUIVER_POR_AREA[area])
    return cfg


def _prep_cyclic_180(da: xr.DataArray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converte para -180..180, ordena, adiciona cyclic point. Retorna (vals, lon, lat)."""
    da = _to_180(da)
    vals, lon_c = _acp(da.values, coord=da['lon'].values)
    return vals, lon_c, da['lat'].values


# ---------------------------------------------------------------------------
# Utilitarios de grade (padrao s31/s33)
# ---------------------------------------------------------------------------
def _ensure_time_coord(obj):
    if hasattr(obj, 'dims') and 'time' not in obj.dims and 'valid_time' in obj.dims:
        obj = obj.rename({'valid_time': 'time'})
    elif hasattr(obj, 'coords') and 'time' not in obj.coords and 'valid_time' in obj.coords:
        obj = obj.rename({'valid_time': 'time'})
    if 'time' not in obj.coords:
        raise KeyError("Nem 'time' nem 'valid_time' encontrados.")
    return obj


def _rename_std_latlon(obj):
    ren = {}
    for name in list(obj.dims) + list(obj.coords):
        low = name.lower()
        if low == 'latitude' and 'lat' not in obj.dims:
            ren[name] = 'lat'
        elif low == 'longitude' and 'lon' not in obj.dims:
            ren[name] = 'lon'
    return obj.rename(ren) if ren else obj


def _drop_expver(ds: xr.Dataset) -> xr.Dataset:
    if 'expver' in ds.dims:
        ds = ds.bfill('expver').ffill('expver').isel(expver=0, drop=True)
    if 'number' in ds.dims:
        ds = ds.isel(number=0, drop=True)
    for c in ('expver', 'number'):
        if c in ds.coords and c not in ds.dims:
            try:
                ds = ds.drop_vars(c)
            except Exception:
                pass
    return ds


def _sort_dedup_time(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.sortby('time')
    t = pd.DatetimeIndex(pd.to_datetime(ds['time'].values))
    _, idx = np.unique(t.values, return_index=True)
    if len(idx) != ds.sizes.get('time', 0):
        ds = ds.isel(time=np.sort(idx))
    return ds


def _find_uv_vars(ds: xr.Dataset) -> Tuple[str, str]:
    u_name = v_name = None
    for name in ('u', 'u_component_of_wind', 'U_GRD_L100', 'uwnd'):
        if name in ds.data_vars:
            u_name = name
            break
    for name in ('v', 'v_component_of_wind', 'V_GRD_L100', 'vwnd'):
        if name in ds.data_vars:
            v_name = name
            break
    if u_name is None or v_name is None:
        raise KeyError(f'u/v nao encontrados. Disponiveis: {list(ds.data_vars)}')
    return u_name, v_name


def _to_180(da: xr.DataArray) -> xr.DataArray:
    """Converte coords de lon 0..360 para -180..180 ordenado (para plotagem)."""
    if np.any(da['lon'].values > 180):
        da = da.assign_coords(lon=(((da['lon'].values + 180) % 360) - 180)).sortby('lon')
    return da


# ---------------------------------------------------------------------------
# Selecao de fonte ERA5/GDAS
# ---------------------------------------------------------------------------
def _get_data_sources(
    dt_ini: datetime,
    dt_fim: datetime,
) -> Tuple[Optional[Tuple[datetime, datetime]], Optional[Tuple[datetime, datetime]]]:
    """Retorna (era5_period, gdas_period) — None quando nao ha dados dessa fonte."""
    cutoff = (datetime.now() - timedelta(days=ERA5_LATENCY_DAYS)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    if dt_fim < cutoff:
        return (dt_ini, dt_fim), None
    if dt_ini >= cutoff:
        return None, (dt_ini, dt_fim)
    return (dt_ini, cutoff - timedelta(days=1)), (cutoff, dt_fim)


def _daily_uv200_on_grid(
    files, dt_ini: datetime, dt_fim: datetime,
    target_lat: np.ndarray, target_lon: np.ndarray, logger,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Le ERA5/GDAS, filtra horas sinoticas, media diaria, regrida p/ a grade da LTM.

    Mantem lon 0..360 (igual a LTM NCEP) para a anomalia ser consistente. Retorna
    (u_da, v_da) em (time, lat, lon), lat ascendente.
    """
    t_ini = np.datetime64(dt_ini.date())
    t_fim = np.datetime64(dt_fim.date())
    req = set(DEFAULT_SYNOPTIC_HOURS)
    tgt_lat = xr.DataArray(target_lat, dims=['lat'])
    tgt_lon = xr.DataArray(target_lon, dims=['lon'])

    us, vs = [], []
    for fp in files:
        ds = xr.open_dataset(fp, engine='netcdf4')
        try:
            ds = _sort_dedup_time(_rename_std_latlon(_drop_expver(_ensure_time_coord(ds))))
            ds = ds.assign_coords(lon=(ds['lon'] % 360)).sortby('lon')
            u_var, v_var = _find_uv_vars(ds)
            da_u, da_v = ds[u_var], ds[v_var]
            for dim in ('pressure_level', 'isobaricInhPa', 'level'):
                if dim in da_u.dims:
                    da_u = da_u.isel({dim: 0}, drop=True)
                    da_v = da_v.isel({dim: 0}, drop=True)
            ti = pd.DatetimeIndex(pd.to_datetime(da_u['time'].values))
            mh = np.array([h in req for h in ti.hour], dtype=bool)
            da_u, da_v = da_u.isel(time=mh), da_v.isel(time=mh)
            da_u = da_u.sel(time=slice(t_ini, t_fim))
            da_v = da_v.sel(time=slice(t_ini, t_fim))
            if da_u.sizes.get('time', 0) == 0:
                continue
            da_u = da_u.resample(time='1D').mean()
            da_v = da_v.resample(time='1D').mean()
            keep = ~((da_u['time'].dt.month == 2) & (da_u['time'].dt.day == 29))
            da_u, da_v = da_u.isel(time=keep.values), da_v.isel(time=keep.values)
            da_u = da_u.interp(lat=tgt_lat, lon=tgt_lon, method='linear').reset_coords(drop=True)
            da_v = da_v.interp(lat=tgt_lat, lon=tgt_lon, method='linear').reset_coords(drop=True)
            us.append(da_u.load())
            vs.append(da_v.load())
            logger.info('Serie diaria: {} -> {} dias', fp.name, da_u.sizes['time'])
        finally:
            ds.close()

    if not us:
        raise RuntimeError('Nenhum dado u/v 200 valido no periodo.')

    u_da = xr.concat(us, dim='time', coords='minimal', compat='override').sortby('time')
    v_da = xr.concat(vs, dim='time', coords='minimal', compat='override').sortby('time')
    _, uniq = np.unique(u_da['time'].values, return_index=True)
    return u_da.isel(time=uniq), v_da.isel(time=uniq)


def _daily_scalar_on_grid(
    files, candidates, dt_ini: datetime, dt_fim: datetime,
    target_lat: np.ndarray, target_lon: np.ndarray, logger,
) -> xr.DataArray:
    """Serie diaria de um escalar (hgt/olr/tmp) regridada p/ a grade da LTM. lat ascendente.

    `candidates` = nomes possiveis da variavel no arquivo (1o encontrado e usado).
    """
    t_ini, t_fim = np.datetime64(dt_ini.date()), np.datetime64(dt_fim.date())
    req = set(DEFAULT_SYNOPTIC_HOURS)
    tgt_lat = xr.DataArray(target_lat, dims=['lat'])
    tgt_lon = xr.DataArray(target_lon, dims=['lon'])
    hs = []
    for fp in files:
        ds = xr.open_dataset(fp, engine='netcdf4')
        try:
            ds = _sort_dedup_time(_rename_std_latlon(_drop_expver(_ensure_time_coord(ds))))
            ds = ds.assign_coords(lon=(ds['lon'] % 360)).sortby('lon')
            hname = next((v for v in candidates if v in ds.data_vars), None)
            if hname is None:
                continue
            da = ds[hname]
            for dim in ('pressure_level', 'isobaricInhPa', 'level'):
                if dim in da.dims:
                    da = da.isel({dim: 0}, drop=True)
            ti = pd.DatetimeIndex(pd.to_datetime(da['time'].values))
            da = da.isel(time=np.array([h in req for h in ti.hour], dtype=bool))
            da = da.sel(time=slice(t_ini, t_fim))
            if da.sizes.get('time', 0) == 0:
                continue
            da = da.resample(time='1D').mean()
            keep = ~((da['time'].dt.month == 2) & (da['time'].dt.day == 29))
            da = da.isel(time=keep.values)
            da = da.interp(lat=tgt_lat, lon=tgt_lon, method='linear').reset_coords(drop=True)
            hs.append(da.load())
        finally:
            ds.close()
    if not hs:
        raise RuntimeError(f'Nenhum dado valido no periodo para {candidates}.')
    h_da = xr.concat(hs, dim='time', coords='minimal', compat='override').sortby('time')
    _, uniq = np.unique(h_da['time'].values, return_index=True)
    return h_da.isel(time=uniq)


def _parse_forecast_init(spec: str, rodada: int) -> datetime:
    """Interpreta a data da rodada de FORECAST_INIT na hora `rodada`.

    Formato preferido: data ISO 'YYYY-MM-DD' (ex.: 2026-06-10) — a hora vem da RODADA.
    Tambem aceita, por compatibilidade, o timestamp completo 'YYYYMMDDHH' (ex.: 2026061000).
    """
    s = spec.strip()
    try:
        d = datetime.fromisoformat(s)  # aceita 'YYYY-MM-DD' (e variantes ISO com hora)
        # data pura (sem hora) -> aplica a hora da RODADA
        if 'T' not in s and ':' not in s:
            d = d.replace(hour=rodada)
        return d
    except ValueError:
        pass
    return datetime.strptime(s, '%Y%m%d%H')  # compatibilidade YYYYMMDDHH


def _resolve_run_inits(rodada: int, num_rodada: int, forecast_init) -> list:
    """Lista de ciclos de inicializacao (mais recente primeiro), todos na hora `rodada`.

    forecast_init vazio/None/'latest' -> ciclo `rodada` mais recente disponivel (now-6h).
    Caso contrario, usa a data informada como init0 (ver `_parse_forecast_init`).
    """
    spec = str(forecast_init or '').strip().lower()
    if spec in ('', 'latest'):
        now = datetime.utcnow() - timedelta(hours=6)  # folga p/ latencia do GFS
        init0 = datetime(now.year, now.month, now.day, rodada)
        if init0 > now:
            init0 -= timedelta(days=1)
    else:
        init0 = _parse_forecast_init(spec, rodada)
    return [init0 - timedelta(days=k) for k in range(max(1, num_rodada))]


def _lagged_ensemble_mean(per_run: list) -> xr.DataArray:
    """Media de lagged ensemble: alinha as series diarias das rodadas por tempo valido
    e faz a media (mais membros nos leads curtos). `per_run` = lista de DataArrays (time,lat,lon)."""
    if len(per_run) == 1:
        return per_run[0]
    all_t = np.unique(np.concatenate([p['time'].values for p in per_run]))
    aligned = [p.reindex(time=all_t) for p in per_run]
    stacked = xr.concat(aligned, dim='run')
    return stacked.mean(dim='run', skipna=True)


def _windows(daily_dates: list, w: int) -> list:
    """Janelas moveis deslizantes de `w` dias sobre `daily_dates`. Retorna (start, end) dates.
    w<=0 ou w>=N -> uma janela unica (periodo inteiro)."""
    n = len(daily_dates)
    if w <= 0 or w >= n:
        return [(daily_dates[0], daily_dates[-1])]
    return [(daily_dates[i], daily_dates[i + w - 1]) for i in range(n - w + 1)]


def _pentad_windows(run_date, n_pentadas: int = N_PENTADAS_FIXAS, pentada_dias: int = PENTADA_DIAS) -> list:
    """Janelas FIXAS de pentada a partir do dia SEGUINTE a `run_date` (data da rodada).

    Pentada k (1-based) = [run_date + 1 + (k-1)*5, run_date + k*5]. NAO sao moveis: as
    janelas sao contiguas e nao se sobrepoem. Retorna lista de (start_date, end_date)."""
    wins = []
    for k in range(n_pentadas):
        ws = run_date + timedelta(days=1 + k * pentada_dias)
        we = run_date + timedelta(days=(k + 1) * pentada_dias)
        wins.append((ws, we))
    return wins


def _waf_from_means(hgt_mean_da, hgt_clim_da, u_clim, v_clim, lat, lon):
    """WAF (TN2001) a partir de medias ja calculadas (grade LTM). Retorna
    (hgt_anom_da[grade LTM], px, py, lat_waf, lon_waf[grade WAF 2.5°])."""
    hgt_anom = hgt_mean_da - hgt_clim_da
    lat_waf = np.arange(90, -90 - WAF_GRID_SPACING / 2, -WAF_GRID_SPACING)
    lon_waf = np.arange(-180, 180, WAF_GRID_SPACING)

    def _to_waf(arr2d):
        da = _to_180(xr.DataArray(arr2d, dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}))
        return da.interp(lat=lat_waf, lon=lon_waf, method='linear').values

    phi_obs = _to_waf(hgt_mean_da.values) * G
    phi_clim = _to_waf(hgt_clim_da.values) * G
    u_c, v_c = _to_waf(u_clim), _to_waf(v_clim)
    mask = (np.abs(lat_waf) < TROPICAL_MASK_LAT) | (np.abs(lat_waf) > POLAR_MASK_LAT)
    for a in (phi_obs, phi_clim, u_c, v_c):
        a[mask, :] = np.nan
    px, py = tnflux.tnf2d(u_c, v_c, phi_clim, phi_obs, lat_waf, lon_waf, 200.0)
    px[mask, :] = np.nan
    py[mask, :] = np.nan
    return hgt_anom, px, py, lat_waf, lon_waf


def _window_spatial_fields(u_w, v_w, hgt_w, uc_w, vc_w, hgtc_w, lat, lon, smooth_deg):
    """Calcula Ks, RWS anomala + vento divergente, anomalia Z200 e WAF para a media de UMA janela."""
    sw = stationary_wavenumber(u_w, lat, lon, smooth_deg=smooth_deg)
    rws_p, uchi_p, vchi_p = rossby_wave_source(u_w, v_w, lat, lon)
    rws_c, uchi_c, vchi_c = rossby_wave_source(uc_w, vc_w, lat, lon)
    rws_anom = rws_p - rws_c
    # Mascara a singularidade EQUATORIAL (f->0) e POLAR (operadores esfericos ~1/cos(phi)
    # divergem em |lat|->90, gerando spikes de ~1e12 que saturam o contourf — sobretudo na
    # ortografica centrada no polo, globo_3d, onde o disco inteiro vira a cor de extremo).
    rws_anom[(np.abs(lat) < RWS_MASK_LAT) | (np.abs(lat) > RWS_MASK_LAT_POLE), :] = np.nan
    hgt_mean_da = xr.DataArray(hgt_w, dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon})
    hgt_clim_da = xr.DataArray(hgtc_w, dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon})
    hgt_anom, px, py, lat_waf, lon_waf = _waf_from_means(hgt_mean_da, hgt_clim_da, uc_w, vc_w, lat, lon)
    # PSI200 (psi_jato): anomalia da streamfunction (10^6 m^2/s) + vento rotacional (streamlines) +
    # magnitude MEDIA do vento 200 (m/s). PSI da media do periodo - PSI da media climatologica.
    psi_w, urot, vrot = _psi_rotwind(u_w, v_w, lat, lon)
    psi_c, _, _ = _psi_rotwind(uc_w, vc_w, lat, lon)
    psi_anom = (psi_w - psi_c) / PSI_SCALE
    wind_mag = np.sqrt(u_w ** 2 + v_w ** 2)
    # CHI200 (chi_jato): anomalia do potencial de velocidade (escala 1e5) — chi da media do
    # periodo menos chi da media climatologica. Vento divergente anomalo = uchi/vchi (abaixo).
    chi_anom = (chi_from_wind(u_w, v_w, lat, lon) - chi_from_wind(uc_w, vc_w, lat, lon)) / CHI_SCALE
    return {
        'ks': sw.ks, 'u_mean': u_w, 'rws_anom': rws_anom,
        'uchi': uchi_p - uchi_c, 'vchi': vchi_p - vchi_c,
        'hgt_anom': hgt_anom, 'px': px, 'py': py, 'lat_waf': lat_waf, 'lon_waf': lon_waf,
        'psi_anom': psi_anom, 'urot': urot, 'vrot': vrot, 'wind_mag': wind_mag,
        'chi_anom': chi_anom,
    }


# ---------------------------------------------------------------------------
# Plotagem — utilitarios (padrao s07)
# ---------------------------------------------------------------------------
def _get_area_list():
    if hasattr(settings, 'LST_AREAS_S34'):
        return list(settings.LST_AREAS_S34)
    return list(DEFAULT_AREAS)


def _resolve_logo_path(entrada_dir):
    """Logo conforme o settings, com a MESMA precedencia do projeto (so um deve estar true):
    SEM_LOGO (prioridade -> nenhum logo) > LOGO_GREC (logo_grec.png) > LOGO_AMPERE (novo_logo.png).
    Se nenhum estiver habilitado, NAO plota logo. Retorna Path existente ou None. Usado por TODOS
    os mapas do s34 (todos os tipos, em forecast de qualquer modelo e na reanalise)."""
    if settings.get('SEM_LOGO', False):
        return None
    if settings.get('LOGO_GREC', False):
        candidate = entrada_dir / 'logo_grec.png'
    elif settings.get('LOGO_AMPERE', True):
        candidate = entrada_dir / 'novo_logo.png'
    else:
        return None
    return candidate if candidate.exists() else None


def _add_logo_to_map(ax, logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=500):
    logo = Image.open(logo_path).convert('RGBA')
    bbox = logo.getbbox()
    if bbox is not None:
        logo = logo.crop(bbox)
    imagebox = OffsetImage(np.array(logo), zoom=zoom)
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

    if area in {'hemisferio_sul', 'hemisferio_norte', 'psa'}:
        gl.xlocator = MultipleLocator(40)
        gl.ylocator = MultipleLocator(20)
    elif area == 'america_sul':
        gl.xlocator = MultipleLocator(20)
        gl.ylocator = MultipleLocator(20)
    elif area == 'globo':
        gl.xlocator = MultipleLocator(40)
        gl.ylocator = MultipleLocator(20)
        gl.xlabel_style = {'size': 15, 'color': 'black'}
        gl.ylabel_style = {'size': 15, 'color': 'black'}


def _fmt_lon(v: int) -> str:
    if v == 0 or abs(v) == 180:
        return f'{abs(v)}°'
    return f'{v}°E' if v > 0 else f'{-v}°W'


def _fmt_lat(v: float) -> str:
    if v == 0:
        return '0°'
    return f'{abs(v):g}°N' if v > 0 else f'{abs(v):g}°S'


def _add_map_border(ax, is_polar: bool) -> None:
    """Contorno fino preto (linewidth 0.5) ao redor da area de plotagem retangular.

    Nao se aplica ao globo_3d (ortografico), que ja tem borda circular propria (set_boundary)."""
    if is_polar:
        return
    try:
        spine = ax.spines['geo']
    except (KeyError, TypeError):
        spine = getattr(ax, 'outline_patch', None)  # cartopy < 0.18
    if spine is not None:
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(0.5)
        # zorder alto: o contourf (z=10) e as feicoes (z=100) desenham por cima do spine
        # padrao (z~2.5) e comem a borda em alguns lados (ex.: esquerda). Forca a borda no topo.
        spine.set_zorder(1000)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _run_once(mode: str, forecast_model, logger):
    """Executa o pipeline completo (download -> processa -> plota -> cache) para UM modelo.

    `mode` = 'reanalysis' | 'forecast'. Em forecast, `forecast_model` ('gfs'/'gefs') seleciona
    os downloaders e a pasta de saida (Saida/<MODELO>/); em reanalise e None. Chamado por main()
    uma vez por modelo habilitado (flags RUN_GFS/RUN_GEFS no settings)."""
    # ---- Parametros do settings ----
    smooth_deg = float(settings.get('WGUIDE_SMOOTH_DEG', 5.0))

    # Dois Hovmollers: um por corrente de jato do HS (subtropical ~30S, polar ~50S).
    # Cada jato e um waveguide distinto, com trens de onda em geral em fases diferentes.
    hov_bands = [
        {
            'slug': 'subtropical',
            'nome': 'Jato Subtropical HS',
            'lat_min': float(settings.get('WGUIDE_HOV_SUBTROPICAL_LAT_MIN', -45.0)),
            'lat_max': float(settings.get('WGUIDE_HOV_SUBTROPICAL_LAT_MAX', -20.0)),
        },
        {
            'slug': 'polar',
            'nome': 'Jato Polar HS',
            'lat_min': float(settings.get('WGUIDE_HOV_POLAR_LAT_MIN', -60.0)),
            'lat_max': float(settings.get('WGUIDE_HOV_POLAR_LAT_MAX', -40.0)),
        },
    ]
    for b in hov_bands:
        if b['lat_min'] >= b['lat_max']:
            raise ValueError(
                f"Faixa {b['slug']}: LAT_MIN ({b['lat_min']}) deve ser menor que "
                f"LAT_MAX ({b['lat_max']})."
            )
        b['band'] = f"{_fmt_lat(b['lat_min'])}–{_fmt_lat(b['lat_max'])}"

    lst_areas = _get_area_list()
    logger.info(
        f'Areas: {lst_areas} | Hovmollers: '
        f"{', '.join(b['slug'] + ' ' + b['band'] for b in hov_bands)} | suavizacao {smooth_deg}°"
    )

    # ---- Modo de dados + janela movel dos mapas espaciais ----
    mov_avg_days = int(settings.get('MOV_AVG_DAYS', 5))
    run_inits = None
    lead_hours = 0
    fcst200_fn = olr_fn = tmp_fn = hgt500_fn = hgt250_fn = uv250_fn = uv850_fn = t2m_fn = None
    if mode == 'forecast':
        (fcst200_fn, olr_fn, tmp_fn, hgt500_fn, hgt250_fn, uv250_fn, uv850_fn,
         t2m_fn) = _FCST_DOWNLOADERS[forecast_model]
        if forecast_model == 'cfs':
            # CFS = pseudo-ensemble subsazonal: 1 "init" = dia D (ensemble interno de 16 membros
            # lagged); RODADA/NUM_RODADA nao se aplicam. D = ONTEM por padrao (garante os ciclos
            # publicados) ou FORECAST_INIT. Horizonte limitado a CFS_LEAD_DAYS (~45 dias).
            spec = str(settings.get('FORECAST_INIT', '') or '').strip().lower()
            if spec in ('', 'latest'):
                D = (datetime.utcnow() - timedelta(days=1)).date()
            else:
                D = datetime.fromisoformat(spec[:10]).date()
            init0 = datetime(D.year, D.month, D.day)
            run_inits = [init0]
            lead_hours = min(int(settings.get('FORECAST_LEAD_DAYS', 45)), CFS_LEAD_DAYS) * 24
            ini_dt = init0
            fim_dt = init0 + timedelta(hours=lead_hours)
            logger.info(
                f'MODO PREVISAO (CFS pseudo-ensemble): dia D={init0:%Y-%m-%d} (16 membros lagged); '
                f'lead {lead_hours}h ate {fim_dt.date()}; janela movel {mov_avg_days}d'
            )
        else:
            rodada = int(settings.get('RODADA', 0))
            if rodada not in (0, 6, 12, 18):
                raise ValueError(f'RODADA deve ser "00", "06", "12" ou "18" (UTC). Recebido: {rodada:02d}')
            num_rodada = int(settings.get('NUM_RODADA', 1))
            lead_hours = int(settings.get('FORECAST_LEAD_DAYS', 10)) * 24
            run_inits = _resolve_run_inits(rodada, num_rodada, settings.get('FORECAST_INIT', 'latest'))
            init0 = run_inits[0]
            ini_dt = datetime(init0.year, init0.month, init0.day)
            fim_dt = init0 + timedelta(hours=lead_hours)
            logger.info(
                f'MODO PREVISAO ({forecast_model.upper()}): {num_rodada} rodada(s) {rodada:02d}Z '
                f'(init0 {init0:%Y-%m-%d}); lead {lead_hours}h ate {fim_dt.date()}; janela movel {mov_avg_days}d'
            )
    else:
        ini_dt = datetime.fromisoformat(str(settings.DATA_INICIAL))
        fim_dt = datetime.fromisoformat(str(settings.DATA_FINAL))
    ini_str, fim_str = ini_dt.date().isoformat(), fim_dt.date().isoformat()

    total_days = (fim_dt.date() - ini_dt.date()).days + 1
    if total_days < 2:
        raise ValueError(
            f'O Hovmoller precisa de pelo menos 2 dias (periodo atual: {total_days} dia). '
            f'Em reanalysis ajuste DATA_INICIAL/DATA_FINAL; em forecast, FORECAST_LEAD_DAYS.'
        )

    # Janelas moveis (nominais) sobre as datas validas
    daily_dates = [d.date() for d in pd.date_range(ini_dt.date(), fim_dt.date(), freq='1D')]
    windows = _windows(daily_dates, mov_avg_days)
    win_labels = [f'{ws:%Y%m%d}_{we:%Y%m%d}' for ws, we in windows]
    logger.info(f'Mapas espaciais: {len(windows)} janela(s) movel(eis) de {mov_avg_days}d')

    # Pentadas FIXAS: N janelas de 5 dias a partir do dia SEGUINTE a data da rodada (default 3;
    # ate 7 aproveitando o GEFS 00Z de 35 dias). Em forecast a rodada e o init0; em reanalise
    # usa-se DATA_INICIAL como referencia. Precedencia: N_PENTADAS_FIXAS (override so do s34) ->
    # N_PENTADAS (mesma usada no s31/s32) -> default. Assim o N_PENTADAS ja existente vale aqui tb.
    n_pentadas = max(1, int(settings.get('N_PENTADAS_FIXAS', settings.get('N_PENTADAS', N_PENTADAS_FIXAS))))
    run_date = run_inits[0].date() if mode == 'forecast' else ini_dt.date()
    pent_windows = _pentad_windows(run_date, n_pentadas)
    pent_labels = [f'{ws:%Y%m%d}_{we:%Y%m%d}_pentada{k + 1}' for k, (ws, we) in enumerate(pent_windows)]
    if mode == 'forecast':  # pentadas so no forecast (reanalise usa janela movel + MEDIA_PERIODO_TOTAL)
        logger.info(
            'Mapas por pentada (5d fixos): '
            + ', '.join(f'P{k + 1} {ws:%d/%m}-{we:%d/%m}' for k, (ws, we) in enumerate(pent_windows))
        )

    # Saida: base s34_TELECONEXAO/ contem o MODO aninhado:
    #   forecast:  Saida/s34_TELECONEXAO/FORECAST/<CONTINENTE>/<MODELO>/<tipo>/<area>/
    #   reanalise: Saida/s34_TELECONEXAO/REANALISE/<tipo>/<area>/ (modelo unico)
    model_tag = forecast_model.upper() if mode == 'forecast' else 'REANALISE'
    base_dir = (Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_TELECONEXAO'
                / ('FORECAST' if mode == 'forecast' else 'REANALISE'))
    if mode == 'forecast':
        continente = _MODEL_CONTINENT.get(forecast_model, 'MODELOS_OUTROS')
        output_dir = base_dir / continente / model_tag
    else:
        output_dir = base_dir
    dados_dir = Path(settings.DIR_DADOS)
    entrada_dir = Path(settings.DIR_INPUT)

    # Alguns modelos de IA (AIFS) NAO tem OLR -> pulam o mapa de OLR. Em forecast, olr_fn=None
    # sinaliza isso; em reanalise sempre ha OLR (observado CPC).
    has_olr = (mode != 'forecast') or (olr_fn is not None)
    # Disponibilidade ESTRUTURAL dos campos OPCIONAIS neste modelo/modo. Em forecast, um downloader
    # None (ex.: CFS sem T850/Z250) significa "o modelo nao tem essa variavel" -> nao criamos o
    # diretorio de saida, nem o entramos no manifesto de cache, nem geramos mapas para ela (evita
    # pasta vazia ocupando espaco). Em reanalise ERA5/GDAS sempre ha todas. Ks/RWS/WAF-Z200/psi_jato/
    # chi_jato derivam de u/v/hgt 200 (nucleo) -> sempre presentes.
    has_tmp = (mode != 'forecast') or (tmp_fn is not None)
    has_hgt500 = (mode != 'forecast') or (hgt500_fn is not None)
    has_t2m = (mode != 'forecast') or (t2m_fn is not None)

    # Subpastas por tipo de plotagem (evita centenas de mapas misturados numa pasta).
    # waf_olr so e criada se o modelo tiver OLR (AIFS nao tem -> nao cria pasta vazia).
    sub_ks = output_dir / 'ks_waveguide'
    sub_rws = output_dir / 'fontes_rws'
    sub_wafz = output_dir / 'waf_z200'
    sub_olr = output_dir / 'waf_olr'
    sub_tmp = output_dir / 'waf_tmp850'
    sub_hgt500 = output_dir / 'hgt500_anom'
    sub_psi = output_dir / 'psi_jato'
    sub_chi = output_dir / 'chi_jato'
    sub_t2m = output_dir / 't2m_wnd850'
    sub_hov = output_dir / 'hovmoller'

    # Dentro de cada subpasta de tipo, ainda ha uma subpasta por AREA: <tipo>/<area>/arquivo.png
    ks_files, rws_files, waf_files, olr_files, tmp_files = {}, {}, {}, {}, {}
    hgt_files, psi_files, chi_files, t2m_files = {}, {}, {}, {}
    for wi, wl in enumerate(win_labels):
        for area in lst_areas:
            ks_files[(wi, area)] = sub_ks / area / f'ks_waveguide_200hpa_{area}_{wl}.png'
            rws_files[(wi, area)] = sub_rws / area / f'fontes_rws_200hpa_{area}_{wl}.png'
            waf_files[(wi, area)] = sub_wafz / area / f'waf_z200_anom_{area}_{wl}.png'
            tmp_files[(wi, area)] = sub_tmp / area / f'waf_tmp850_anom_{area}_{wl}.png'
            hgt_files[(wi, area)] = sub_hgt500 / area / f'hgt500_anom_{area}_{wl}.png'
            psi_files[(wi, area)] = sub_psi / area / f'psi_jato_200hpa_{area}_{wl}.png'
            chi_files[(wi, area)] = sub_chi / area / f'chi_jato_{area}_{wl}.png'
            t2m_files[(wi, area)] = sub_t2m / area / f't2m_wnd850_anom_{area}_{wl}.png'
            if has_olr:
                olr_files[(wi, area)] = sub_olr / area / f'waf_olr_anom_{area}_{wl}.png'
    # Mesma colecao de mapas, agora por pentada fixa (sufixo pentadaX no nome via pent_labels)
    ks_files_p, rws_files_p, waf_files_p, olr_files_p, tmp_files_p = {}, {}, {}, {}, {}
    hgt_files_p, psi_files_p, chi_files_p, t2m_files_p = {}, {}, {}, {}
    for pi, pl in enumerate(pent_labels):
        for area in lst_areas:
            ks_files_p[(pi, area)] = sub_ks / area / f'ks_waveguide_200hpa_{area}_{pl}.png'
            rws_files_p[(pi, area)] = sub_rws / area / f'fontes_rws_200hpa_{area}_{pl}.png'
            waf_files_p[(pi, area)] = sub_wafz / area / f'waf_z200_anom_{area}_{pl}.png'
            tmp_files_p[(pi, area)] = sub_tmp / area / f'waf_tmp850_anom_{area}_{pl}.png'
            hgt_files_p[(pi, area)] = sub_hgt500 / area / f'hgt500_anom_{area}_{pl}.png'
            psi_files_p[(pi, area)] = sub_psi / area / f'psi_jato_200hpa_{area}_{pl}.png'
            chi_files_p[(pi, area)] = sub_chi / area / f'chi_jato_{area}_{pl}.png'
            t2m_files_p[(pi, area)] = sub_t2m / area / f't2m_wnd850_anom_{area}_{pl}.png'
            if has_olr:
                olr_files_p[(pi, area)] = sub_olr / area / f'waf_olr_anom_{area}_{pl}.png'
    # Media do periodo TOTAL [DATA_INICIAL, DATA_FINAL] — SO em reanalise — na MESMA pasta de cada
    # tipo/area, com sufixo "media_total" no nome (como as pentadas usam pentadaX).
    media_label = f'{ini_dt:%Y%m%d}_{fim_dt:%Y%m%d}_media_total'
    ks_files_m, rws_files_m, waf_files_m, olr_files_m, tmp_files_m = {}, {}, {}, {}, {}
    hgt_files_m, psi_files_m, chi_files_m, t2m_files_m = {}, {}, {}, {}
    for area in lst_areas:
        ks_files_m[area] = sub_ks / area / f'ks_waveguide_200hpa_{area}_{media_label}.png'
        rws_files_m[area] = sub_rws / area / f'fontes_rws_200hpa_{area}_{media_label}.png'
        waf_files_m[area] = sub_wafz / area / f'waf_z200_anom_{area}_{media_label}.png'
        tmp_files_m[area] = sub_tmp / area / f'waf_tmp850_anom_{area}_{media_label}.png'
        hgt_files_m[area] = sub_hgt500 / area / f'hgt500_anom_{area}_{media_label}.png'
        psi_files_m[area] = sub_psi / area / f'psi_jato_200hpa_{area}_{media_label}.png'
        chi_files_m[area] = sub_chi / area / f'chi_jato_{area}_{media_label}.png'
        t2m_files_m[area] = sub_t2m / area / f't2m_wnd850_anom_{area}_{media_label}.png'
        if has_olr:
            olr_files_m[area] = sub_olr / area / f'waf_olr_anom_{area}_{media_label}.png'
    for b in hov_bands:
        b['png'] = sub_hov / f"hovmoller_vprime200_{b['slug']}.png"
    nc_name = f'waveguide_s34_{ini_str}_to_{fim_str}.nc'
    # Manifesto do cache: SO inclui os campos que de fato serao gerados. Campos opcionais ausentes
    # neste modelo (has_tmp/has_hgt500/has_t2m/has_olr=False) ficam de fora -> o cache nao espera
    # arquivos que nunca existirao (e nao se cria diretorio vazio para eles, ver mais adiante).
    window_dicts = [ks_files, rws_files, waf_files, psi_files, chi_files]
    pent_dicts = [ks_files_p, rws_files_p, waf_files_p, psi_files_p, chi_files_p]
    media_dicts = [ks_files_m, rws_files_m, waf_files_m, psi_files_m, chi_files_m]
    if has_olr:
        window_dicts.append(olr_files); pent_dicts.append(olr_files_p); media_dicts.append(olr_files_m)
    if has_tmp:
        window_dicts.append(tmp_files); pent_dicts.append(tmp_files_p); media_dicts.append(tmp_files_m)
    if has_hgt500:
        window_dicts.append(hgt_files); pent_dicts.append(hgt_files_p); media_dicts.append(hgt_files_m)
    if has_t2m:
        window_dicts.append(t2m_files); pent_dicts.append(t2m_files_p); media_dicts.append(t2m_files_m)
    output_files = (
        [str(p) for d in window_dicts for p in d.values()]
        + [str(b['png']) for b in hov_bands]
    )
    if mode == 'forecast':  # pentadas (5d fixos) so no forecast
        output_files += [str(p) for d in pent_dicts for p in d.values()]
    else:  # media do periodo total so na reanalise
        output_files += [str(p) for d in media_dicts for p in d.values()]

    cache_params = {
        'mode': mode,
        'forecast_model': forecast_model,
        'run_inits': [r.strftime('%Y%m%d%H') for r in run_inits] if run_inits else None,
        'DATA_INICIAL': ini_str,
        'DATA_FINAL': fim_str,
        'areas': lst_areas,
        'mov_avg_days': mov_avg_days,
        'n_pentadas': n_pentadas,
        'hov_bands': [(b['slug'], b['lat_min'], b['lat_max']) for b in hov_bands],
        'smooth_deg': smooth_deg,
        'script_version': '4.16',  # campos opcionais ausentes nao criam diretorio/manifesto vazio
    }

    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('CACHE VALIDO! Execucao ja foi realizada com os mesmos parametros.')
        return

    start_time = time.time()
    logger.info(f'Periodo: {ini_str} a {fim_str} ({total_days} dias)')
    output_dir.mkdir(parents=True, exist_ok=True)
    sub_hov.mkdir(parents=True, exist_ok=True)  # Hovmoller e por faixa de jato, nao por area
    # NAO pre-criamos as pastas <tipo>/<area>: cada figura cria a propria pasta na hora de salvar
    # (mkdir lazy em _finish_map/_plot_*). Assim, campo que NAO gera figura — porque o modelo nao
    # tem a variavel (ex.: CFS sem T850) OU porque o dado nao chegou no download (ex.: AIGFS t2m) —
    # nunca deixa um diretorio vazio ocupando espaco.
    dados_dir.mkdir(parents=True, exist_ok=True)

    # ---- Etapa 1+2: Download + serie diaria u/v/hgt 200 na grade LTM (2.5°) ----
    dates_probe = np.array([np.datetime64(fim_dt.date())])
    _, _, ltm_lat, ltm_lon = clim_uv200_daily(dates_probe)
    order = np.argsort(ltm_lat)
    lat, lon = ltm_lat[order], ltm_lon  # lat ascendente, lon 0..360

    olr_da = None
    t_da = None
    h500_da = None
    z250_da = None
    u250_da = None
    v250_da = None
    u850_da = None
    v850_da = None
    t2m_da = None
    fcst_info = None
    if mode == 'forecast':
        is_cfs = forecast_model == 'cfs'
        logger.info(
            f'Etapa 1: {forecast_model.upper()} — '
            + (f'pseudo-ensemble dia {run_inits[0]:%d/%m}'
               if is_cfs else f'{len(run_inits)} rodada(s) {rodada:02d}Z (lagged ensemble)')
        )
        u_runs, v_runs, h_runs, olr_runs, t_runs, h500_runs, z250_runs = [], [], [], [], [], [], []
        u250_runs, v250_runs = [], []
        u850_runs, v850_runs, t2m_runs = [], [], []
        combo_fn = _FCST_COMBO.get(forecast_model)
        for k, init_k in enumerate(run_inits):
            logger.info(f'  Rodada {k + 1}/{len(run_inits)}: init {init_k:%Y-%m-%d %H}Z')
            # Pre-busca COMBINADA (NOMADS GFS/GEFS): 1 requisicao/passo grava todos os NetCDFs por
            # variavel de uma vez -> os downloaders por-variavel abaixo viram cache-hit. Best-effort:
            # se algo faltar, eles baixam individualmente (fallback).
            if combo_fn is not None:
                combo_fn(init=init_k, lead_hours=lead_hours, hours=list(DEFAULT_SYNOPTIC_HOURS))
            files_k = list(fcst200_fn(
                init=init_k, lead_hours=lead_hours, hours=list(DEFAULT_SYNOPTIC_HOURS),
            ))
            uk, vk = _daily_uv200_on_grid(files_k, ini_dt, fim_dt, lat, lon, logger)
            hk = _daily_scalar_on_grid(files_k, HGT_VARS, ini_dt, fim_dt, lat, lon, logger)
            u_runs.append(uk)
            v_runs.append(vk)
            h_runs.append(hk)
            # T850 (waf_tmp850): OPCIONAL/desacoplado — alguns modelos nao tem (ex.: CFS) -> pula o mapa.
            if tmp_fn is not None:
                t_files_k = list(tmp_fn(
                    init=init_k, lead_hours=lead_hours, hours=list(DEFAULT_SYNOPTIC_HOURS)))
                if t_files_k:
                    t_runs.append(_daily_scalar_on_grid(
                        t_files_k, TMP_VARS, ini_dt, fim_dt, lat, lon, logger))
            # Z500 (hgt500_anom + ref do psi_jato): tratado como o OLR — pode faltar dia(s) sem
            # derrubar os demais produtos (ver intersecao/reindex adiante).
            if hgt500_fn is not None:
                h500_files_k = list(hgt500_fn(
                    init=init_k, lead_hours=lead_hours, hours=list(DEFAULT_SYNOPTIC_HOURS)))
                if h500_files_k:
                    h500_runs.append(_daily_scalar_on_grid(
                        h500_files_k, HGT_VARS, ini_dt, fim_dt, lat, lon, logger))
            # Z250 (isolinhas de jato): tratado como o Z500 — pode faltar (ex.: CFS sem z250) sem derrubar.
            if hgt250_fn is not None:
                z250_files_k = list(hgt250_fn(
                    init=init_k, lead_hours=lead_hours, hours=list(DEFAULT_SYNOPTIC_HOURS)))
                if z250_files_k:
                    z250_runs.append(_daily_scalar_on_grid(
                        z250_files_k, HGT_VARS, ini_dt, fim_dt, lat, lon, logger))
            # u/v 250 (magnitude do jato no chi_jato): tratado como o Z500 — pode faltar sem derrubar.
            if uv250_fn is not None:
                uv250_files_k = list(uv250_fn(
                    init=init_k, lead_hours=lead_hours, hours=list(DEFAULT_SYNOPTIC_HOURS)))
                if uv250_files_k:
                    u250k, v250k = _daily_uv200_on_grid(uv250_files_k, ini_dt, fim_dt, lat, lon, logger)
                    u250_runs.append(u250k)
                    v250_runs.append(v250k)
            # u/v 850 (streamlines do vento anomalo) + T2m (anomalia) — campo t2m_wnd850.
            if uv850_fn is not None:
                uv850_files_k = list(uv850_fn(
                    init=init_k, lead_hours=lead_hours, hours=list(DEFAULT_SYNOPTIC_HOURS)))
                if uv850_files_k:
                    u850k, v850k = _daily_uv200_on_grid(uv850_files_k, ini_dt, fim_dt, lat, lon, logger)
                    u850_runs.append(u850k)
                    v850_runs.append(v850k)
            if t2m_fn is not None:
                t2m_files_k = list(t2m_fn(
                    init=init_k, lead_hours=lead_hours, hours=list(DEFAULT_SYNOPTIC_HOURS)))
                if t2m_files_k:
                    t2m_runs.append(_daily_scalar_on_grid(
                        t2m_files_k, T2M_VARS, ini_dt, fim_dt, lat, lon, logger))
            if has_olr:  # AIFS nao tem OLR -> pula
                olr_files_k = list(olr_fn(
                    init=init_k, lead_hours=lead_hours, hours=list(DEFAULT_SYNOPTIC_HOURS),
                ))
                olr_runs.append(_daily_scalar_on_grid(
                    olr_files_k, OLR_VARS, ini_dt, fim_dt, lat, lon, logger))
        u_da = _lagged_ensemble_mean(u_runs)   # mapas: media das rodadas
        v_da = _lagged_ensemble_mean(v_runs)
        h_da = _lagged_ensemble_mean(h_runs)
        t_da = _lagged_ensemble_mean(t_runs) if t_runs else None  # T850 opcional (CFS nao tem)
        h500_da = _lagged_ensemble_mean(h500_runs) if h500_runs else None
        z250_da = _lagged_ensemble_mean(z250_runs) if z250_runs else None
        u250_da = _lagged_ensemble_mean(u250_runs) if u250_runs else None
        v250_da = _lagged_ensemble_mean(v250_runs) if v250_runs else None
        u850_da = _lagged_ensemble_mean(u850_runs) if u850_runs else None
        v850_da = _lagged_ensemble_mean(v850_runs) if v850_runs else None
        t2m_da = _lagged_ensemble_mean(t2m_runs) if t2m_runs else None
        olr_da = _lagged_ensemble_mean(olr_runs) if has_olr else None
        v_da_hov = v_runs[0]                    # Hovmoller: rodada mais recente
        # Linha de titulo com modelo/rodada/datas das inicializacoes
        runs_str = ', '.join(r.strftime('%d/%m') for r in run_inits)
        if is_cfs:
            fcst_info = f'{model_tag} | pseudo-ensemble subsazonal (16 membros) de {run_inits[0]:%d/%m/%Y}'
        elif len(run_inits) == 1:
            fcst_info = f'{model_tag} | rodada {rodada:02d}Z de {run_inits[0]:%d/%m/%Y}'
        else:
            fcst_info = (
                f'{model_tag} | ensemble {rodada:02d}Z, {len(run_inits)} rodadas '
                f'({run_inits[-1]:%d/%m} a {run_inits[0]:%d/%m/%Y}): {runs_str}'
            )
    else:
        logger.info('Etapa 1: Download ERA5/GDAS u/v/hgt 200 hPa...')
        era5_period, gdas_period = _get_data_sources(ini_dt, fim_dt)
        files, hgt200_files = [], []   # hgt200_files: NAO confundir com o dict hgt_files (paths hgt500)
        if era5_period:
            files += list(ensure_era5_uv200_for_period(
                start=era5_period[0], end=era5_period[1], hours_utc=list(DEFAULT_SYNOPTIC_HOURS)))
            hgt200_files += list(ensure_era5_hgt200_for_period(
                start=era5_period[0], end=era5_period[1], hours_utc=list(DEFAULT_SYNOPTIC_HOURS)))
        if gdas_period:
            files += list(ensure_gdas_uv200_for_period(start=gdas_period[0], end=gdas_period[1]))
            hgt200_files += list(ensure_gdas_hgt200_for_period(start=gdas_period[0], end=gdas_period[1]))
        t_files = []
        if era5_period:
            t_files += list(ensure_era5_t850_for_period(
                start=era5_period[0], end=era5_period[1], hours_utc=list(DEFAULT_SYNOPTIC_HOURS)))
        if gdas_period:
            t_files += list(ensure_gdas_tmp850_for_period(start=gdas_period[0], end=gdas_period[1]))
        h500_files = []
        if era5_period:
            h500_files += list(ensure_era5_altura_geopotencial_500_global_for_period_grib(
                start=era5_period[0], end=era5_period[1], hours_utc=list(DEFAULT_SYNOPTIC_HOURS)))
        if gdas_period:
            h500_files += list(ensure_gdas_hgt500_for_period(start=gdas_period[0], end=gdas_period[1]))
        z250_files = []
        if era5_period:
            z250_files += list(ensure_era5_z250_for_period(
                start=era5_period[0], end=era5_period[1], hours_utc=list(DEFAULT_SYNOPTIC_HOURS)))
        if gdas_period:
            z250_files += list(ensure_gdas_hgt250_for_period(start=gdas_period[0], end=gdas_period[1]))
        uv250_files = []
        if era5_period:
            uv250_files += list(ensure_era5_uv250_for_period(
                start=era5_period[0], end=era5_period[1], hours_utc=list(DEFAULT_SYNOPTIC_HOURS)))
        if gdas_period:
            uv250_files += list(ensure_gdas_uv250_for_period(start=gdas_period[0], end=gdas_period[1]))
        uv850_files, t2m_files = [], []
        if era5_period:
            uv850_files += list(ensure_era5_uv850_for_period(
                start=era5_period[0], end=era5_period[1], hours_utc=list(DEFAULT_SYNOPTIC_HOURS)))
            t2m_files += list(ensure_era5_superficie_for_period(
                start=era5_period[0], end=era5_period[1], hours_utc=list(DEFAULT_SYNOPTIC_HOURS)))
        if gdas_period:
            uv850_files += list(ensure_gdas_uv850_for_period(start=gdas_period[0], end=gdas_period[1]))
            t2m_files += list(ensure_gdas_t2m_for_period(start=gdas_period[0], end=gdas_period[1]))
        u_da, v_da = _daily_uv200_on_grid(files, ini_dt, fim_dt, lat, lon, logger)
        h_da = _daily_scalar_on_grid(hgt200_files, HGT_VARS, ini_dt, fim_dt, lat, lon, logger)
        t_da = _daily_scalar_on_grid(t_files, TMP_VARS, ini_dt, fim_dt, lat, lon, logger)
        if h500_files:
            h500_da = _daily_scalar_on_grid(h500_files, HGT_VARS, ini_dt, fim_dt, lat, lon, logger)
        if z250_files:
            z250_da = _daily_scalar_on_grid(z250_files, HGT_VARS, ini_dt, fim_dt, lat, lon, logger)
        if uv250_files:
            u250_da, v250_da = _daily_uv200_on_grid(uv250_files, ini_dt, fim_dt, lat, lon, logger)
        if uv850_files:
            u850_da, v850_da = _daily_uv200_on_grid(uv850_files, ini_dt, fim_dt, lat, lon, logger)
        if t2m_files:
            t2m_da = _daily_scalar_on_grid(t2m_files, T2M_VARS, ini_dt, fim_dt, lat, lon, logger)
        v_da_hov = v_da

    # eixo de tempo comum so dos campos do NUCLEO: u/v/hgt @ 200 (Ks/RWS/WAF/Z200). Todo o resto
    # (OLR, T850, Z500, Z250, u/v 250/850, T2m) fica de FORA e e reindexado no eixo comum com NaN:
    # assim um dia faltante OU uma variavel ausente num modelo (ex.: CFS sem T850/Z250) afeta SO o
    # mapa daquele campo — nunca derruba o nucleo nem aborta a rodada.
    common_t = np.intersect1d(u_da['time'].values, v_da['time'].values)
    common_t = np.intersect1d(common_t, h_da['time'].values)
    u_da, v_da, h_da = u_da.sel(time=common_t), v_da.sel(time=common_t), h_da.sel(time=common_t)
    if t_da is not None:
        # T850 (waf_tmp850): fora da intersecao, reindex com NaN (degrada so o mapa T850; CFS nao tem).
        t_da = t_da.reindex(time=common_t)
    if olr_da is not None:
        # OLR pode ter MENOS dias que os campos dinamicos (passo nao publicado/throttle do NOMADS).
        # Reindexa no eixo comum preenchendo com NaN: a media da janela usa nanmean (ignora o NaN)
        # e SO o mapa de OLR daquela janela e afetado — z200/Ks/RWS/T850 saem normalmente.
        faltam = np.setdiff1d(common_t, olr_da['time'].values)
        if faltam.size:
            logger.warning(
                'OLR ausente em {} dia(s) ({}) — mapas de OLR usarao apenas os dias disponiveis; '
                'z200/Ks/RWS/T850 NAO sao afetados.',
                faltam.size, ', '.join(str(pd.Timestamp(t).date()) for t in faltam))
        olr_da = olr_da.reindex(time=common_t)
    if h500_da is not None:
        # Z500 (hgt500_anom + psi_jato): mesma logica do OLR — fora da intersecao, reindex com NaN.
        # Um dia de Z500 faltante afeta SO o mapa hgt500 daquela janela, nunca z200/Ks/RWS/T850/PSI.
        faltam500 = np.setdiff1d(common_t, h500_da['time'].values)
        if faltam500.size:
            logger.warning(
                'Z500 ausente em {} dia(s) ({}) — mapas de hgt500 usarao apenas os dias disponiveis; '
                'demais produtos NAO sao afetados.',
                faltam500.size, ', '.join(str(pd.Timestamp(t).date()) for t in faltam500))
        h500_da = h500_da.reindex(time=common_t)
    if z250_da is not None:
        # Z250 (isolinhas de jato no psi_jato): mesma logica do Z500 — fora da intersecao, reindex NaN.
        z250_da = z250_da.reindex(time=common_t)
    if u250_da is not None and v250_da is not None:
        # u/v 250 (magnitude do jato no chi_jato): fora da intersecao, reindex NaN (degrada so o jato).
        u250_da = u250_da.reindex(time=common_t)
        v250_da = v250_da.reindex(time=common_t)
    else:
        u250_da = v250_da = None
    if u850_da is not None and v850_da is not None:
        # u/v 850 (streamlines do vento anomalo no t2m_wnd850): fora da intersecao, reindex NaN.
        u850_da = u850_da.reindex(time=common_t)
        v850_da = v850_da.reindex(time=common_t)
    else:
        u850_da = v850_da = None
    if t2m_da is not None:
        # T2m (anomalia no t2m_wnd850): fora da intersecao, reindex NaN (degrada so o mapa de T2m).
        t2m_da = t2m_da.reindex(time=common_t)
    dates = np.array([np.datetime64(pd.Timestamp(t).date()) for t in common_t])
    logger.info(f'Etapa 2: serie diaria com {len(dates)} dias ({dates[0]} a {dates[-1]})')

    # ---- Etapa 3: Climatologia diaria (NCEP u/v/hgt/T850 + CPC OLR mean), por dia-do-ano ----
    logger.info('Etapa 3: Climatologia diaria (NCEP u/v/hgt/T850 + CPC OLR mean)...')
    u_clim_d, v_clim_d, _, _ = clim_uv200_daily(dates)
    h_clim_d, _, _ = clim_hgt200_daily(dates)
    u_clim_d, v_clim_d = u_clim_d[:, order, :], v_clim_d[:, order, :]
    h_clim_d = h_clim_d[:, order, :]
    if t_da is not None:  # T850 opcional (CFS nao tem)
        t_clim_d, _, _ = clim_t850_daily(dates)
        t_clim_d = t_clim_d[:, order, :]
    else:
        t_clim_d = None
    if h500_da is not None:
        h500_clim_d, _, _ = clim_hgt500_daily(dates)
        h500_clim_d = h500_clim_d[:, order, :]
    else:
        h500_clim_d = None
    # Clim diaria p/ o t2m_wnd850: vento 850 (anomalia das streamlines) e T2m (anomalia shaded).
    if u850_da is not None:
        u850_clim_d, _, _ = clim_u850_daily(dates)
        v850_clim_d, _, _ = clim_v850_daily(dates)
        u850_clim_d, v850_clim_d = u850_clim_d[:, order, :], v850_clim_d[:, order, :]
    else:
        u850_clim_d = v850_clim_d = None
    if t2m_da is not None:
        t2m_clim_d, _, _ = clim_t2m_daily(dates)
        t2m_clim_d = t2m_clim_d[:, order, :]
    else:
        t2m_clim_d = None
    # OLR: reanalise usa o observado (CPC mean); anomalia vs clim do MEAN (decisao do usuario).
    # Modelos sem OLR (AIFS) -> has_olr=False -> olr_anom=None (mapa de OLR e pulado).
    if has_olr:
        if olr_da is None:  # reanalise: cai no observado (CPC)
            olr_da = xr.DataArray(
                olr_obs_daily(dates, lat, lon), dims=('time', 'lat', 'lon'),
                coords={'time': common_t, 'lat': lat, 'lon': lon})
        olr_anom = olr_da.values - clim_olr_daily(dates, lat, lon)
    else:
        olr_anom = None
    # T850 (waf_tmp850): normaliza p/ °C antes da anomalia. Opcional — None se o modelo nao tem T850.
    t_anom = (_to_celsius(t_da.values) - _to_celsius(t_clim_d)) if t_da is not None else None

    # ---- Etapa 4: Hovmollers v'200 (rodada mais recente; SEM janela movel) ----
    logger.info('Etapa 4: Gerando Hovmollers de v\'200 (jato subtropical + polar)...')
    dates_hov = np.array([np.datetime64(pd.Timestamp(t).date()) for t in v_da_hov['time'].values])
    _, v_clim_hov, _, _ = clim_uv200_daily(dates_hov)
    v_clim_hov = v_clim_hov[:, order, :]
    v_anom_hov = xr.DataArray(
        v_da_hov.values - v_clim_hov, dims=('time', 'lat', 'lon'),
        coords={'time': v_da_hov['time'].values, 'lat': lat, 'lon': lon},
    )
    for b in hov_bands:
        b['hov'] = v_anom_hov.sel(lat=slice(b['lat_min'], b['lat_max'])).mean(dim='lat', skipna=True)
        _plot_hovmoller(
            hov_v=b['hov'], hov_band=b['band'], nome=b['nome'], ini_dt=ini_dt, fim_dt=fim_dt,
            total_days=len(dates_hov), entrada_dir=entrada_dir, out_path=b['png'],
            fcst_info=fcst_info, logger=logger,
        )

    # ---- Mapas espaciais: setup comum (arrays/clim) ----
    info_plot = settings['areas_plotagem']
    uv, vv, hv = u_da.values, v_da.values, h_da.values
    # Z500/Z250: normaliza geopotencial->altura (ERA5 vem em m^2/s^2; forecast/GDAS ja em m).
    hv500 = _to_geop_height(h500_da.values) if h500_da is not None else None
    hv250 = _to_geop_height(z250_da.values) if z250_da is not None else None
    u250v = u250_da.values if u250_da is not None else None
    v250v = v250_da.values if v250_da is not None else None
    u850v = u850_da.values if u850_da is not None else None
    v850v = v850_da.values if v850_da is not None else None
    t2mv = t2m_da.values if t2m_da is not None else None
    arrays = (uv, vv, hv, olr_anom, t_anom, hv500, hv250, u250v, v250v, u850v, v850v, t2mv)
    clim = (u_clim_d, v_clim_d, h_clim_d, h500_clim_d, u850_clim_d, v850_clim_d, t2m_clim_d)

    # ---- Etapa 4b: MEDIA do periodo TOTAL [DATA_INICIAL, DATA_FINAL] — PRIMEIRO (so reanalise) ----
    if mode != 'forecast':
        logger.info('Etapa 4b: Media do periodo total [%s a %s] (primeiro) — %s areas x %s mapas...',
                    ini_str, fim_str, len(lst_areas), 5 if has_olr else 4)
        sel_total = (dates >= np.datetime64(ini_dt.date())) & (dates <= np.datetime64(fim_dt.date()))
        if sel_total.any():
            _render_spatial_window(
                ini_dt.date(), fim_dt.date(), sel_total, arrays, clim, lat, lon, smooth_deg,
                lst_areas, info_plot, entrada_dir,
                (ks_files_m, rws_files_m, waf_files_m, olr_files_m if has_olr else {}, tmp_files_m,
                 hgt_files_m, psi_files_m, chi_files_m, t2m_files_m),
                fcst_info, logger,
            )

    # ---- Etapa 5: Mapas espaciais por JANELA MOVEL (Ks + RWS + WAF/Z200 + WAF/OLR) ----
    logger.info('Etapa 5: Mapas espaciais por janela movel (Ks/RWS/WAF-Z200/WAF-OLR)...')
    last_fields = None
    for wi, (ws, we) in enumerate(windows):
        sel = (dates >= np.datetime64(ws)) & (dates <= np.datetime64(we))
        n_dias = int(sel.sum())
        span = (we - ws).days + 1
        if n_dias < span:  # janela movel so plota se COMPLETA (encerra no ultimo dia do modelo)
            if n_dias > 0:
                logger.warning(
                    f'  Janela {win_labels[wi]} incompleta ({n_dias}/{span} dias, passa do alcance) — pulando')
            continue
        logger.info(f'  Janela {wi + 1}/{len(windows)} ({ws} a {we}) — {len(lst_areas)} areas')
        last_fields = _render_spatial_window(
            ws, we, sel, arrays, clim, lat, lon, smooth_deg, lst_areas, info_plot, entrada_dir,
            ({a: ks_files[(wi, a)] for a in lst_areas},
             {a: rws_files[(wi, a)] for a in lst_areas},
             {a: waf_files[(wi, a)] for a in lst_areas},
             {a: olr_files[(wi, a)] for a in lst_areas} if has_olr else {},
             {a: tmp_files[(wi, a)] for a in lst_areas},
             {a: hgt_files[(wi, a)] for a in lst_areas},
             {a: psi_files[(wi, a)] for a in lst_areas},
             {a: chi_files[(wi, a)] for a in lst_areas},
             {a: t2m_files[(wi, a)] for a in lst_areas}),
            fcst_info, logger,
        )

    # ---- Etapa 6: Mapas espaciais por PENTADA fixa — SO no forecast (na reanalise as pentadas
    # cairiam para frente, alem de DATA_FINAL; a reanalise usa janela movel + MEDIA_PERIODO_TOTAL).
    if mode == 'forecast':
        logger.info('Etapa 6: Mapas espaciais por pentada fixa (a partir do dia seguinte a rodada)...')
        for pi, (ws, we) in enumerate(pent_windows):
            sel = (dates >= np.datetime64(ws)) & (dates <= np.datetime64(we))
            n_dias = int(sel.sum())
            if n_dias < MIN_DIAS_PENTADA:  # exige >= 4 dos 5 dias (4 ok; <4 nao plota)
                logger.warning(
                    f'  Pentada {pi + 1} ({ws} a {we}) com {n_dias}/{PENTADA_DIAS} dias '
                    f'(< {MIN_DIAS_PENTADA}) — pulando. Em forecast, aumente FORECAST_LEAD_DAYS '
                    f'para >= {n_pentadas * PENTADA_DIAS}.'
                )
                continue
            if n_dias < PENTADA_DIAS:
                logger.warning(f'  Pentada {pi + 1} ({ws} a {we}) incompleta: {n_dias}/{PENTADA_DIAS} dias')
            logger.info(f'  Pentada {pi + 1}/{len(pent_windows)} ({ws} a {we}) — {len(lst_areas)} areas')
            _render_spatial_window(
                ws, we, sel, arrays, clim, lat, lon, smooth_deg, lst_areas, info_plot, entrada_dir,
                ({a: ks_files_p[(pi, a)] for a in lst_areas},
                 {a: rws_files_p[(pi, a)] for a in lst_areas},
                 {a: waf_files_p[(pi, a)] for a in lst_areas},
                 {a: olr_files_p[(pi, a)] for a in lst_areas} if has_olr else {},
                 {a: tmp_files_p[(pi, a)] for a in lst_areas},
                 {a: hgt_files_p[(pi, a)] for a in lst_areas},
                 {a: psi_files_p[(pi, a)] for a in lst_areas},
                 {a: chi_files_p[(pi, a)] for a in lst_areas},
                 {a: t2m_files_p[(pi, a)] for a in lst_areas}),
                fcst_info, logger,
            )

    # ---- NetCDF (Hovmollers + campos da ultima janela) ----
    ds_out = xr.Dataset()
    for b in hov_bands:
        ds_out[f"vprime_hov_{b['slug']}"] = b['hov'].assign_attrs(
            long_name=f"Hovmoller v'200 {b['nome']} ({b['band']})", units='m s-1',
        )
    if last_fields is not None:
        ds_out['ks'] = xr.DataArray(
            last_fields['ks'], dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon},
            attrs={'long_name': 'Ks (Hoskins & Ambrizzi 1993), ultima janela', 'units': '1'})
        ds_out['rws_anom_200'] = xr.DataArray(
            last_fields['rws_anom'], dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon},
            attrs={'long_name': 'RWS anomala (Sardeshmukh-Hoskins), ultima janela', 'units': 's-2'})
        ds_out['hgt_anom_200'] = last_fields['hgt_anom'].assign_attrs(
            long_name='Anomalia hgt 200 hPa, ultima janela', units='m')
    nc_path = output_dir / nc_name
    logger.info(f'Salvando NetCDF: {nc_path}')
    ds_out.to_netcdf(str(nc_path))

    execution_time = time.time() - start_time
    save_cache_metadata(SCRIPT_ID, cache_params, output_files, execution_time)
    logger.info('=' * 80)
    logger.info(f'Script {SCRIPT_ID.upper()} concluido com sucesso!')
    logger.info(f'Tempo de execucao: {execution_time:.1f}s ({execution_time / 60:.1f} min)')
    logger.info(
        f'{len(windows)} janela(s) movel(eis) + {len(pent_windows)} pentada(s) '
        f'x {len(lst_areas)} areas x {9 if has_olr else 8} mapas + {len(hov_bands)} Hovmollers '
        f'em: {output_dir}'
    )
    logger.info('=' * 80)


def _enabled_models() -> list:
    """Modelos de previsao habilitados pelas flags do settings (ordem de _FCST_DOWNLOADERS)."""
    return [m for m, (flag, default) in _MODEL_FLAGS.items()
            if bool(settings.get(flag, default))]


def main():
    """Entry point — chamado pelo CLI sem argumentos.

    Reanalise: roda uma vez (Saida/REANALISE/). Previsao: roda o pipeline completo para CADA
    modelo habilitado pelas flags RUN_GFS/RUN_GEFS, salvando em Saida/<MODELO>/."""
    logger = get_logger(SCRIPT_ID)
    logger.info('=' * 80)
    logger.info(f'SCRIPT {SCRIPT_ID.upper()}: {SCRIPT_DESC}')
    logger.info('=' * 80)

    mode = str(settings.get('MODE', 'reanalysis')).strip().lower()
    if mode != 'forecast':
        _run_once('reanalysis', None, logger)
        return

    models = _enabled_models()
    if not models:
        raise ValueError(
            'MODE=forecast, mas nenhum modelo habilitado. Defina ao menos um no settings: '
            'RUN_GFS = true e/ou RUN_GEFS = true.'
        )
    logger.info(f'Modelos de previsao habilitados: {[m.upper() for m in models]}')
    for fm in models:
        logger.info('#' * 80)
        logger.info(f'### MODELO {fm.upper()}')
        logger.info('#' * 80)
        _run_once('forecast', fm, logger)


def _plot_ks_area(
    area, info_plot, lon_cyc, lat, ks_cyc, u_cyc,
    hgt_cyc, lon_hgt, lat_hgt, px_cyc, py_cyc, lon_waf, lat_waf,
    ini_dt, fim_dt, entrada_dir, out_path, fcst_info, logger,
):
    """Mapa de UMA area (padrao s07): Ks sombreado + Z200 anomalia + WAF + U=0."""
    cfg = info_plot[area]
    is_polar = cfg.get('projection', '') == 'orthographic_south'

    if is_polar:
        proj = ccrs.Orthographic(
            central_longitude=settings.get(
                'ORTHO_CENTRAL_LONGITUDE', cfg.get('ortho_central_longitude', -71)),
            central_latitude=settings.get(
                'ORTHO_CENTRAL_LATITUDE', cfg.get('ortho_central_latitude', -84)),
        )
    else:
        proj = ccrs.PlateCarree(central_longitude=cfg['central_longitude_mapa'])

    transform = ccrs.PlateCarree(central_longitude=cfg['central_longitude_plot'])

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    if is_polar:
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

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
        ax.set_xlim([cfg['lon_esq'], cfg['lon_dir']])
        ax.set_ylim([cfg['lat_inf'], cfg['lat_sup']])

    # Features (mesma pilha do s07)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2)
    ax.add_feature(cfeature.LAND.with_scale('50m'), linewidth=0.5, facecolor='whitesmoke')
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1.2, zorder=100)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, zorder=100)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, zorder=100)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), linewidth=0.5, facecolor='white')
    _add_map_border(ax, is_polar)

    # Ks sombreado (apenas onde a onda estacionaria E possivel)
    im = ax.contourf(
        lon_cyc, lat, ks_cyc, levels=KS_LEVELS, cmap=KS_CMAP, extend='max',
        transform=transform, zorder=10,
    )
    # Regioes evanescentes (leste / beta_M<0) ficam em branco (Ks=NaN, sem mascara).
    # Linha critica U=0 (fronteira oeste/leste) — delimita o "duto" do waveguide
    ax.contour(
        lon_cyc, lat, u_cyc, levels=[0.0], colors='black',
        linewidths=4.5, transform=transform, zorder=40,
    )
    # Onda real: isolinhas de anomalia de Z200 (preto solido=positivo, tracejado=negativo)
    zpos = Z200_LEVELS[Z200_LEVELS > 0]
    zneg = Z200_LEVELS[Z200_LEVELS < 0]
    ax.contour(
        lon_hgt, lat_hgt, hgt_cyc, levels=zpos, colors='black',
        linewidths=1.1, transform=transform, zorder=50,
    )
    ax.contour(
        lon_hgt, lat_hgt, hgt_cyc, levels=zneg, colors='black', linestyles='dashed',
        linewidths=1.1, transform=transform, zorder=50,
    )
    # WAF (Takaya & Nakamura 2001): direcao da propagacao de energia da onda
    qcfg = _get_quiver_config(area)
    step = int(qcfg['step'])
    lon_q = lon_waf[::step]
    lat_q = lat_waf[::step]
    px_q = px_cyc[::step, ::step]
    py_q = py_cyc[::step, ::step]
    amp = np.sqrt(px_q ** 2 + py_q ** 2)
    max_amp = np.nanmax(amp)
    if max_amp and max_amp > 0:
        weak = amp < float(qcfg['min_amp_ratio']) * max_amp
        px_q = np.where(weak, np.nan, px_q / max_amp)
        py_q = np.where(weak, np.nan, py_q / max_amp)
    ax.quiver(
        lon_q, lat_q, px_q, py_q, transform=transform,
        scale=qcfg['scale'], scale_units=qcfg['scale_units'], width=float(qcfg['width']),
        headwidth=float(qcfg['headwidth']), headlength=float(qcfg['headlength']),
        color='black', zorder=60,
    )

    # Colorbar (posicao/tamanho de ticks conforme o s07)
    if is_polar and area != 'globo_3d':
        cbar = plt.colorbar(im, ax=ax, pad=0.05, fraction=0.04, ticks=KS_LEVELS)
        cbar.set_label(label='Ks', size=10)
        cbar.ax.tick_params(labelsize=10)
    else:
        divider = make_axes_locatable(ax)
        if area in {'america_sul', 'globo_3d'}:
            cax = divider.append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
            cbar = plt.colorbar(
                im, cax=cax, pad=0.02, fraction=0.02375, extend='max', ticks=KS_LEVELS,
            )
        else:
            cax = divider.append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)
            cbar = plt.colorbar(
                im, cax=cax, pad=0.02, fraction=0.02375, location='bottom',
                extend='max', orientation='horizontal', ticks=KS_LEVELS,
            )
        cbar.set_label(label='Numero de onda estacionario de Rossby (Ks)', size=18)
        cbar.ax.tick_params(labelsize=20)

    # Titulo
    dt_ini = ini_dt.strftime('%d-%m-%y')
    dt_fim = fim_dt.strftime('%d-%m-%y')
    titulo = (
        'Guia de Onda de Rossby — Ks 200 hPa + anomalia Z200 + WAF\n'
        'branco: sem onda estacionaria (leste); linha grossa: U=0'
    )
    ax.set_title(_title(titulo, f'De {dt_ini} a {dt_fim}', fcst_info),
                 fontsize=12 if is_polar else 16, loc='left')

    # Logo
    logo_path = _resolve_logo_path(entrada_dir)
    if logo_path is not None:
        _add_logo_to_map(ax=ax, logo_path=logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=500)

    logger.info(f'Salvando a figura {out_path}')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)  # cria <tipo>/<area>/ so quando ha figura
    plt.savefig(str(out_path), dpi=fig.dpi, bbox_inches='tight')
    plt.close('all')


def _plot_rws_area(
    area, info_plot, lon_rws, lat_rws, rws_cyc, uchi_cyc, vchi_cyc,
    ini_dt, fim_dt, entrada_dir, out_path, fcst_info, logger,
):
    """Mapa companheiro (padrao s07): fonte de onda de Rossby anomala + vento divergente.

    Paleta BrBG (s05): verde (RWS>0) = fonte anticiclonica no HS (lanca o trem);
    marrom (RWS<0) = fonte ciclonica.
    """
    cfg = info_plot[area]
    is_polar = cfg.get('projection', '') == 'orthographic_south'

    if is_polar:
        proj = ccrs.Orthographic(
            central_longitude=settings.get(
                'ORTHO_CENTRAL_LONGITUDE', cfg.get('ortho_central_longitude', -71)),
            central_latitude=settings.get(
                'ORTHO_CENTRAL_LATITUDE', cfg.get('ortho_central_latitude', -84)),
        )
    else:
        proj = ccrs.PlateCarree(central_longitude=cfg['central_longitude_mapa'])
    transform = ccrs.PlateCarree(central_longitude=cfg['central_longitude_plot'])

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    if is_polar:
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        ax.set_boundary(mpath.Path(verts * radius + center), transform=ax.transAxes)

    if is_polar:
        gl = ax.gridlines(draw_labels=False, linestyle='--', alpha=0.5)
        gl.xlocator = MultipleLocator(30)
        gl.ylocator = MultipleLocator(20)
    else:
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.0)
        _configure_gridlines(gl, area)

    if not is_polar:
        ax.set_xlim([cfg['lon_esq'], cfg['lon_dir']])
        ax.set_ylim([cfg['lat_inf'], cfg['lat_sup']])

    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2)
    ax.add_feature(cfeature.LAND.with_scale('50m'), linewidth=0.5, facecolor='whitesmoke')
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1.2, zorder=100)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, zorder=100)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, zorder=100)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), linewidth=0.5, facecolor='white')
    _add_map_border(ax, is_polar)

    # RWS anomala sombreada (BrBG: verde=positivo/anticiclonico HS, marrom=negativo/ciclonico)
    cmap = plt.get_cmap('BrBG')
    im = ax.contourf(
        lon_rws, lat_rws, rws_cyc, levels=RWS_LEVELS, cmap=cmap, extend='both',
        transform=transform, zorder=10,
    )

    # Vento divergente anomalo (o "sopro" que gera a fonte)
    step = DIVWIND_QUIVER['step']
    lon_q, lat_q = lon_rws[::step], lat_rws[::step]
    u_q = uchi_cyc[::step, ::step].copy()
    v_q = vchi_cyc[::step, ::step].copy()
    mag = np.sqrt(u_q ** 2 + v_q ** 2)
    weak = mag < DIVWIND_QUIVER['min_mag']
    u_q[weak] = np.nan
    v_q[weak] = np.nan
    ax.quiver(
        lon_q, lat_q, u_q, v_q, transform=transform,
        scale=DIVWIND_QUIVER['scale'], width=DIVWIND_QUIVER['width'],
        color='black', zorder=60,
    )

    # Colorbar (posicao/tamanho de ticks conforme o s07)
    if is_polar and area != 'globo_3d':
        cbar = plt.colorbar(im, ax=ax, pad=0.05, fraction=0.04, ticks=RWS_TICKS)
        cbar.set_label(label='RWS (x10^-11 s^-2)', size=10)
        cbar.ax.tick_params(labelsize=10)
    else:
        divider = make_axes_locatable(ax)
        if area in {'america_sul', 'globo_3d'}:
            cax = divider.append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
            cbar = plt.colorbar(im, cax=cax, pad=0.02, fraction=0.02375, extend='both', ticks=RWS_TICKS)
        else:
            cax = divider.append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)
            cbar = plt.colorbar(
                im, cax=cax, pad=0.02, fraction=0.02375, location='bottom',
                extend='both', orientation='horizontal', ticks=RWS_TICKS,
            )
        cbar.set_label(label='Fonte de onda de Rossby anomala (x10^-11 s^-2)', size=18)
        cbar.ax.tick_params(labelsize=20)

    dt_ini = ini_dt.strftime('%d-%m-%y')
    dt_fim = fim_dt.strftime('%d-%m-%y')
    titulo = (
        'Fontes de Onda de Rossby (anomalia) — 200 hPa + vento divergente\n'
        'verde: fonte anticiclonica (HS); marrom: ciclonica'
    )
    ax.set_title(_title(titulo, f'De {dt_ini} a {dt_fim}', fcst_info),
                 fontsize=12 if is_polar else 16, loc='left')

    logo_path = _resolve_logo_path(entrada_dir)
    if logo_path is not None:
        _add_logo_to_map(ax=ax, logo_path=logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=500)

    logger.info(f'Salvando a figura {out_path}')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)  # cria <tipo>/<area>/ so quando ha figura
    plt.savefig(str(out_path), dpi=fig.dpi, bbox_inches='tight')
    plt.close('all')


def _plot_waf_z200_area(
    area, info_plot, hgt_cyc, lon_hgt, lat_hgt, px_cyc, py_cyc, lon_waf, lat_waf,
    ini_dt, fim_dt, entrada_dir, out_path, fcst_info, logger,
    z250_cyc=None, lon_z2=None, lat_z2=None,
):
    """Mapa estilo s07 em 200 hPa: anomalia de Z200 sombreada + WAF + isolinhas Z250 (jato, estilo s16)."""
    cfg = info_plot[area]
    is_polar = cfg.get('projection', '') == 'orthographic_south'
    if is_polar:
        proj = ccrs.Orthographic(
            central_longitude=settings.get(
                'ORTHO_CENTRAL_LONGITUDE', cfg.get('ortho_central_longitude', -71)),
            central_latitude=settings.get(
                'ORTHO_CENTRAL_LATITUDE', cfg.get('ortho_central_latitude', -84)),
        )
    else:
        proj = ccrs.PlateCarree(central_longitude=cfg['central_longitude_mapa'])
    transform = ccrs.PlateCarree(central_longitude=cfg['central_longitude_plot'])

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    if is_polar:
        theta = np.linspace(0, 2 * np.pi, 100)
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        ax.set_boundary(mpath.Path(verts * 0.5 + [0.5, 0.5]), transform=ax.transAxes)
    if is_polar:
        gl = ax.gridlines(draw_labels=False, linestyle='--', alpha=0.5)
        gl.xlocator = MultipleLocator(30)
        gl.ylocator = MultipleLocator(20)
    else:
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.0)
        _configure_gridlines(gl, area)
        ax.set_xlim([cfg['lon_esq'], cfg['lon_dir']])
        ax.set_ylim([cfg['lat_inf'], cfg['lat_sup']])

    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2)
    ax.add_feature(cfeature.LAND.with_scale('50m'), linewidth=0.5, facecolor='whitesmoke')
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1.2, zorder=100)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, zorder=100)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, zorder=100)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), linewidth=0.5, facecolor='white')
    _add_map_border(ax, is_polar)

    cmap = LinearSegmentedColormap.from_list('anom', settings.LST_ANOM_CORRETA)
    im = ax.contourf(
        lon_hgt, lat_hgt, hgt_cyc, levels=Z200_SHADED_LEVELS, cmap=cmap, extend='both',
        transform=transform, zorder=10,
    )
    ax.contour(
        lon_hgt, lat_hgt, hgt_cyc, levels=Z200_SHADED_LEVELS, colors='white',
        linewidths=0.6, transform=transform, zorder=20,
    )
    # WAF (mesmo quiver normalizado do mapa de Ks)
    qcfg = _get_quiver_config(area)
    step = int(qcfg['step'])
    lon_q, lat_q = lon_waf[::step], lat_waf[::step]
    pxq, pyq = px_cyc[::step, ::step], py_cyc[::step, ::step]
    amp = np.sqrt(pxq ** 2 + pyq ** 2)
    mx = np.nanmax(amp)
    if mx and mx > 0:
        weak = amp < float(qcfg['min_amp_ratio']) * mx
        pxq = np.where(weak, np.nan, pxq / mx)
        pyq = np.where(weak, np.nan, pyq / mx)
    ax.quiver(
        lon_q, lat_q, pxq, pyq, transform=transform,
        scale=qcfg['scale'], scale_units=qcfg['scale_units'], width=float(qcfg['width']),
        headwidth=float(qcfg['headwidth']), headlength=float(qcfg['headlength']),
        color='black', zorder=60,
    )
    # isolinhas de altura geopotencial 250 hPa (jato) — mesmas do psi_jato (estilo s16, zorder alto)
    _draw_z250_isolines(ax, z250_cyc, lon_z2, lat_z2, transform)

    if is_polar and area != 'globo_3d':
        cbar = plt.colorbar(im, ax=ax, pad=0.05, fraction=0.04, ticks=Z200_SHADED_TICKS)
        cbar.set_label(label='mgp', size=10)
        cbar.ax.tick_params(labelsize=10)
    else:
        divider = make_axes_locatable(ax)
        if area in {'america_sul', 'globo_3d'}:
            cax = divider.append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
            cbar = plt.colorbar(im, cax=cax, pad=0.02, fraction=0.02375, extend='both',
                                ticks=Z200_SHADED_TICKS)
        else:
            cax = divider.append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)
            cbar = plt.colorbar(im, cax=cax, pad=0.02, fraction=0.02375, location='bottom',
                                extend='both', orientation='horizontal', ticks=Z200_SHADED_TICKS)
        cbar.set_label(label='Anomalia de altura geopotencial 200 hPa (mgp)', size=18)
        cbar.ax.tick_params(labelsize=20)

    dt_ini, dt_fim = ini_dt.strftime('%d-%m-%y'), fim_dt.strftime('%d-%m-%y')
    ax.set_title(
        _title('Anomalia Z200 + Fluxo de Atividade de Onda de Rossby (WAF)',
               f'De {dt_ini} a {dt_fim}', fcst_info),
        fontsize=12 if is_polar else 16, loc='left',
    )
    logo_path = _resolve_logo_path(entrada_dir)
    if logo_path is not None:
        _add_logo_to_map(ax=ax, logo_path=logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=500)
    logger.info(f'Salvando a figura {out_path}')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)  # cria <tipo>/<area>/ so quando ha figura
    plt.savefig(str(out_path), dpi=fig.dpi, bbox_inches='tight')
    plt.close('all')


def _title(base: str, date_str: str, fcst_info) -> str:
    """Monta o titulo: descricao -> [modelo/rodada em previsao] -> data (ultima linha)."""
    lines = [base]
    if fcst_info:
        lines.append(fcst_info)
    lines.append(date_str)
    return '\n'.join(lines)


def _plot_waf_field_area(
    area, info_plot, field_cyc, lon_f, lat_f, hgt_cyc, lon_hgt, lat_hgt,
    px_cyc, py_cyc, lon_waf, lat_waf, cmap, levels, ticks, cbar_label, titulo,
    ini_dt, fim_dt, entrada_dir, out_path, fcst_info, logger,
):
    """Mapa generico estilo s08/s10: campo sombreado + contornos pretos de Z200 + WAF."""
    cfg = info_plot[area]
    is_polar = cfg.get('projection', '') == 'orthographic_south'
    if is_polar:
        proj = ccrs.Orthographic(
            central_longitude=settings.get(
                'ORTHO_CENTRAL_LONGITUDE', cfg.get('ortho_central_longitude', -71)),
            central_latitude=settings.get(
                'ORTHO_CENTRAL_LATITUDE', cfg.get('ortho_central_latitude', -84)),
        )
    else:
        proj = ccrs.PlateCarree(central_longitude=cfg['central_longitude_mapa'])
    transform = ccrs.PlateCarree(central_longitude=cfg['central_longitude_plot'])

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    if is_polar:
        theta = np.linspace(0, 2 * np.pi, 100)
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        ax.set_boundary(mpath.Path(verts * 0.5 + [0.5, 0.5]), transform=ax.transAxes)
    if is_polar:
        gl = ax.gridlines(draw_labels=False, linestyle='--', alpha=0.5)
        gl.xlocator = MultipleLocator(30)
        gl.ylocator = MultipleLocator(20)
    else:
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.0)
        _configure_gridlines(gl, area)
        ax.set_xlim([cfg['lon_esq'], cfg['lon_dir']])
        ax.set_ylim([cfg['lat_inf'], cfg['lat_sup']])

    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2)
    ax.add_feature(cfeature.LAND.with_scale('50m'), linewidth=0.5, facecolor='whitesmoke')
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1.2, zorder=100)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, zorder=100)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, zorder=100)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), linewidth=0.5, facecolor='white')
    _add_map_border(ax, is_polar)

    im = ax.contourf(
        lon_f, lat_f, field_cyc, levels=levels, cmap=cmap, extend='both',
        transform=transform, zorder=10,
    )
    # contornos pretos de Z200 (preto solido=positivo, tracejado=negativo)
    zpos = Z200_LEVELS[Z200_LEVELS > 0]
    zneg = Z200_LEVELS[Z200_LEVELS < 0]
    ax.contour(lon_hgt, lat_hgt, hgt_cyc, levels=zpos, colors='black',
               linewidths=1.0, transform=transform, zorder=40)
    ax.contour(lon_hgt, lat_hgt, hgt_cyc, levels=zneg, colors='black', linestyles='dashed',
               linewidths=1.0, transform=transform, zorder=40)
    # WAF
    qcfg = _get_quiver_config(area)
    step = int(qcfg['step'])
    lon_q, lat_q = lon_waf[::step], lat_waf[::step]
    pxq, pyq = px_cyc[::step, ::step], py_cyc[::step, ::step]
    amp = np.sqrt(pxq ** 2 + pyq ** 2)
    mx = np.nanmax(amp)
    if mx and mx > 0:
        weak = amp < float(qcfg['min_amp_ratio']) * mx
        pxq = np.where(weak, np.nan, pxq / mx)
        pyq = np.where(weak, np.nan, pyq / mx)
    ax.quiver(lon_q, lat_q, pxq, pyq, transform=transform, scale=qcfg['scale'],
              scale_units=qcfg['scale_units'], width=float(qcfg['width']),
              headwidth=float(qcfg['headwidth']), headlength=float(qcfg['headlength']),
              color='black', zorder=60)

    if is_polar and area != 'globo_3d':
        cbar = plt.colorbar(im, ax=ax, pad=0.05, fraction=0.04, ticks=ticks)
        cbar.set_label(label=cbar_label.split('(')[-1].rstrip(')'), size=10)
        cbar.ax.tick_params(labelsize=10)
    else:
        divider = make_axes_locatable(ax)
        if area in {'america_sul', 'globo_3d'}:
            cax = divider.append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
            cbar = plt.colorbar(im, cax=cax, pad=0.02, fraction=0.02375, extend='both', ticks=ticks)
        else:
            cax = divider.append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)
            cbar = plt.colorbar(im, cax=cax, pad=0.02, fraction=0.02375, location='bottom',
                                extend='both', orientation='horizontal', ticks=ticks)
        cbar.set_label(label=cbar_label, size=18)
        cbar.ax.tick_params(labelsize=20)

    dt_ini, dt_fim = ini_dt.strftime('%d-%m-%y'), fim_dt.strftime('%d-%m-%y')
    ax.set_title(_title(titulo, f'De {dt_ini} a {dt_fim}', fcst_info),
                 fontsize=12 if is_polar else 16, loc='left')
    logo_path = _resolve_logo_path(entrada_dir)
    if logo_path is not None:
        _add_logo_to_map(ax=ax, logo_path=logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=500)
    logger.info(f'Salvando a figura {out_path}')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)  # cria <tipo>/<area>/ so quando ha figura
    plt.savefig(str(out_path), dpi=fig.dpi, bbox_inches='tight')
    plt.close('all')


def _render_spatial_window(
    ws, we, sel, arrays, clim, lat, lon, smooth_deg, lst_areas, info_plot,
    entrada_dir, paths, fcst_info, logger,
):
    """Calcula os campos da media da janela [ws, we] (`sel`) e plota os mapas por area.

    `arrays` = (uv, vv, hv, olr_anom, t_anom, hv500, hv250, u250v, v250v, u850v, v850v, t2mv);
    `clim` = (u_clim_d, v_clim_d, h_clim_d, h500_clim_d, u850_clim_d, v850_clim_d, t2m_clim_d);
    `paths` = (ks, rws, waf, olr, tmp, hgt500, psi, chi, t2m) dicts {area: Path}.
    Reutilizado pela janela movel, pentadas e media total. Retorna o dict de campos."""
    uv, vv, hv, olr_anom, t_anom, hv500, hv250, u250v, v250v, u850v, v850v, t2mv = arrays
    u_clim_d, v_clim_d, h_clim_d, h500_clim_d, u850_clim_d, v850_clim_d, t2m_clim_d = clim
    (ks_paths, rws_paths, waf_paths, olr_paths, tmp_paths, hgt_paths, psi_paths, chi_paths,
     t2m_paths) = paths
    fields = _window_spatial_fields(
        uv[sel].mean(0), vv[sel].mean(0), hv[sel].mean(0),
        u_clim_d[sel].mean(0), v_clim_d[sel].mean(0), h_clim_d[sel].mean(0),
        lat, lon, smooth_deg,
    )
    if olr_anom is not None:  # AIFS nao tem OLR -> mapa de OLR pulado
        olr_win = olr_anom[sel]
        # OLR pode ter dia(s) NaN (passo faltante reindexado no eixo comum): nanmean usa so os
        # dias com dado. Se a janela inteira estiver sem OLR, NAO cria 'olr_anom' -> o mapa de OLR
        # daquela janela e pulado (has_olr_map=False em _plot_spatial_window); os demais produtos
        # saem normalmente.
        if not np.isnan(olr_win).all():
            fields['olr_anom'] = np.nanmean(olr_win, axis=0)
    if t_anom is not None:  # T850 opcional (CFS nao tem) -> mapa waf_tmp850 pulado
        t_win = t_anom[sel]
        if not np.isnan(t_win).all():
            fields['t_anom'] = np.nanmean(t_win, axis=0)
    # Z500 (hgt500): media da janela (nanmean — dia(s) faltante(s) reindexados como NaN). Sem
    # Z500 na janela -> nao cria os campos -> mapa hgt500 pulado so nesta janela.
    if hv500 is not None:
        z500_win = hv500[sel]
        if not np.isnan(z500_win).all():
            z500_mean = np.nanmean(z500_win, axis=0)
            fields['z500_mean'] = z500_mean
            fields['z500_anom'] = z500_mean - h500_clim_d[sel].mean(0)
    # Z250 media (isolinhas de jato sobrepostas no psi_jato) — sem anomalia/clim.
    if hv250 is not None:
        z250_win = hv250[sel]
        if not np.isnan(z250_win).all():
            fields['z250_mean'] = np.nanmean(z250_win, axis=0)
    # |V250| media (magnitude do jato em 250 hPa, sobreposta no chi_jato) — magnitude do vento medio.
    if u250v is not None and v250v is not None:
        u2_win, v2_win = u250v[sel], v250v[sel]
        if not np.isnan(u2_win).all():
            fields['jet250_mag'] = np.sqrt(np.nanmean(u2_win, axis=0) ** 2 + np.nanmean(v2_win, axis=0) ** 2)
    # t2m_wnd850: anomalia de T2m (shaded) + anomalia do vento 850 (streamlines). Anomalias =
    # media da janela - media climatologica da janela (nanmean p/ dias faltantes).
    if t2mv is not None and t2m_clim_d is not None:
        t2m_win = t2mv[sel]
        if not np.isnan(t2m_win).all():
            fields['t2m_anom'] = _to_celsius(np.nanmean(t2m_win, axis=0)) - _to_celsius(t2m_clim_d[sel].mean(0))
    if u850v is not None and v850v is not None and u850_clim_d is not None:
        u8_win, v8_win = u850v[sel], v850v[sel]
        if not np.isnan(u8_win).all():
            fields['u850_anom'] = np.nanmean(u8_win, axis=0) - u850_clim_d[sel].mean(0)
            fields['v850_anom'] = np.nanmean(v8_win, axis=0) - v850_clim_d[sel].mean(0)
    _plot_spatial_window(
        fields, lat, lon, lst_areas, info_plot,
        datetime(ws.year, ws.month, ws.day), datetime(we.year, we.month, we.day),
        entrada_dir, ks_paths, rws_paths, waf_paths, olr_paths, tmp_paths, hgt_paths, psi_paths,
        chi_paths, t2m_paths, fcst_info, logger,
    )
    return fields


def _setup_map_ax(area, info_plot):
    """Cria fig/ax com projecao, gridlines e feicoes (boilerplate comum dos mapas do s34).

    Retorna (fig, ax, is_polar, transform)."""
    cfg = info_plot[area]
    is_polar = cfg.get('projection', '') == 'orthographic_south'
    if is_polar:
        proj = ccrs.Orthographic(
            central_longitude=settings.get(
                'ORTHO_CENTRAL_LONGITUDE', cfg.get('ortho_central_longitude', -71)),
            central_latitude=settings.get(
                'ORTHO_CENTRAL_LATITUDE', cfg.get('ortho_central_latitude', -84)),
        )
    else:
        proj = ccrs.PlateCarree(central_longitude=cfg['central_longitude_mapa'])
    transform = ccrs.PlateCarree(central_longitude=cfg['central_longitude_plot'])
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    if is_polar:
        theta = np.linspace(0, 2 * np.pi, 100)
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        ax.set_boundary(mpath.Path(verts * 0.5 + [0.5, 0.5]), transform=ax.transAxes)
        gl = ax.gridlines(draw_labels=False, linestyle='--', alpha=0.5)
        gl.xlocator = MultipleLocator(30)
        gl.ylocator = MultipleLocator(20)
    else:
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.0)
        _configure_gridlines(gl, area)
        ax.set_xlim([cfg['lon_esq'], cfg['lon_dir']])
        ax.set_ylim([cfg['lat_inf'], cfg['lat_sup']])
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2)
    ax.add_feature(cfeature.LAND.with_scale('50m'), linewidth=0.5, facecolor='whitesmoke')
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1.2, zorder=100)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, zorder=100)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.2, zorder=100)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), linewidth=0.5, facecolor='white')
    _add_map_border(ax, is_polar)
    return fig, ax, is_polar, transform


def _finish_map(fig, ax, is_polar, titulo, ini_dt, fim_dt, fcst_info, entrada_dir, out_path, logger):
    """Titulo + logo + salvar (boilerplate comum)."""
    dt_ini, dt_fim = ini_dt.strftime('%d-%m-%y'), fim_dt.strftime('%d-%m-%y')
    ax.set_title(_title(titulo, f'De {dt_ini} a {dt_fim}', fcst_info),
                 fontsize=12 if is_polar else 16, loc='left')
    logo_path = _resolve_logo_path(entrada_dir)
    if logo_path is not None:
        _add_logo_to_map(ax=ax, logo_path=logo_path, zoom=0.65, xoffset=0, yoffset=0, zorder=500)
    logger.info(f'Salvando a figura {out_path}')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)  # cria <tipo>/<area>/ so quando ha figura
    plt.savefig(str(out_path), dpi=fig.dpi, bbox_inches='tight')
    plt.close('all')


def _plot_hgt500_area(
    area, info_plot, z500_anom_cyc, z500_mean_cyc, lon_z, lat_z,
    ini_dt, fim_dt, entrada_dir, out_path, fcst_info, logger,
):
    """Mapa hgt500: anomalia Z500 (shaded) + isolinhas da Z500 MEDIA do periodo (preto, sem rotulo)."""
    fig, ax, is_polar, transform = _setup_map_ax(area, info_plot)
    cmap = LinearSegmentedColormap.from_list('anom', settings.LST_ANOM_CORRETA)
    im = ax.contourf(
        lon_z, lat_z, z500_anom_cyc, levels=Z500_ANOM_LEVELS, cmap=cmap, extend='both',
        transform=transform, zorder=10,
    )
    # isolinhas da Z500 media — pretas, finas, SEM rotulo/valor
    ax.contour(
        lon_z, lat_z, z500_mean_cyc, levels=Z500_MEAN_LEVELS, colors='black',
        linewidths=0.6, transform=transform, zorder=40,
    )
    if is_polar and area != 'globo_3d':
        cbar = plt.colorbar(im, ax=ax, pad=0.05, fraction=0.04, ticks=Z500_ANOM_TICKS)
        cbar.set_label(label='mgp', size=10)
        cbar.ax.tick_params(labelsize=10)
    else:
        divider = make_axes_locatable(ax)
        if area in {'america_sul', 'globo_3d'}:
            cax = divider.append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
            cbar = plt.colorbar(im, cax=cax, pad=0.02, fraction=0.02375, extend='both',
                                ticks=Z500_ANOM_TICKS)
        else:
            cax = divider.append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)
            cbar = plt.colorbar(im, cax=cax, pad=0.02, fraction=0.02375, location='bottom',
                                extend='both', orientation='horizontal', ticks=Z500_ANOM_TICKS)
        cbar.set_label(label='Anomalia de altura geopotencial 500 hPa (mgp)', size=18)
        cbar.ax.tick_params(labelsize=20)
    _finish_map(fig, ax, is_polar, 'Anomalia Z500 + Z500 media (isolinhas pretas)',
                ini_dt, fim_dt, fcst_info, entrada_dir, out_path, logger)


def _plot_psi_jet_area(
    area, info_plot, psi_cyc, lon_psi, lat_psi, mag_cyc, lon_mag, lat_mag,
    urot_cyc, vrot_cyc, lon_rot_native, lat_rot, px_cyc, py_cyc, lon_waf, lat_waf,
    ini_dt, fim_dt, entrada_dir, out_path, fcst_info, logger,
    z250_cyc=None, lon_z2=None, lat_z2=None,
):
    """Mapa psi_jato: anomalia PSI200 (shaded s04) + |V200| media (shaded s16, fracos transparentes)
    + linhas de corrente (vento rotacional) + isolinhas Z250 (jato, estilo s16) + WAF por cima."""
    cfg = info_plot[area]
    fig, ax, is_polar, transform = _setup_map_ax(area, info_plot)
    # 1) base: anomalia de PSI (shaded vermelho-branco-azul)
    levels, ticks, cmap_psi, norm_psi = _psi_levels_ticks_norm(psi_cyc[np.isfinite(psi_cyc)])
    im_psi = ax.contourf(
        lon_psi, lat_psi, psi_cyc, levels=levels, cmap=cmap_psi, norm=norm_psi, extend='both',
        transform=transform, zorder=10,
    )
    ax.contour(lon_psi, lat_psi, psi_cyc, levels=levels, colors='white', linewidths=0.6,
               alpha=0.55, transform=transform, zorder=15)
    # 2) magnitude media do vento 200 sobreposta (paleta s16; SO o jato >= 35 m/s; abaixo, transparente)
    cmap_mag = LinearSegmentedColormap.from_list('wnd_speed', WIND_MAG_PALETTE)
    mag_present = np.isfinite(mag_cyc).any() and float(np.nanmax(mag_cyc)) >= WIND_MAG_LEVELS[0]
    im_mag = None
    if mag_present:
        im_mag = ax.contourf(
            lon_mag, lat_mag, mag_cyc, levels=WIND_MAG_LEVELS, cmap=cmap_mag, extend='max',
            transform=transform, zorder=20,
        )
    # 3) linhas de corrente do vento rotacional (estilo s04) — inclusive na ortografica (globo_3d).
    slcfg = _get_streamline_config(area)
    central_lon = 0 if is_polar else cfg['central_longitude_mapa']
    lon_sl, u_sl, v_sl = _prepare_streamline_lonuv(lon_rot_native, urot_cyc, vrot_cyc, central_lon)
    try:
        ax.streamplot(
            lon_sl, lat_rot, u_sl, v_sl, transform=ccrs.PlateCarree(central_longitude=central_lon),
            density=float(slcfg['density']), linewidth=float(slcfg['linewidth']),
            arrowsize=float(slcfg['arrowsize']), color=slcfg['color'], zorder=30,
        )
    except Exception as exc:
        logger.warning('  streamplot rotacional falhou em {} ({}) — mapa sem linhas', area, exc)
    # 4) isolinhas de altura geopotencial 250 hPa (jato) — estilo s16, zorder alto (abaixo do logo)
    _draw_z250_isolines(ax, z250_cyc, lon_z2, lat_z2, transform)
    # 5) WAF por cima de tudo (mesmo quiver normalizado dos demais mapas)
    qcfg = _get_quiver_config(area)
    step = int(qcfg['step'])
    lon_q, lat_q = lon_waf[::step], lat_waf[::step]
    pxq, pyq = px_cyc[::step, ::step], py_cyc[::step, ::step]
    amp = np.sqrt(pxq ** 2 + pyq ** 2)
    mx = np.nanmax(amp)
    if mx and mx > 0:
        weak = amp < float(qcfg['min_amp_ratio']) * mx
        pxq = np.where(weak, np.nan, pxq / mx)
        pyq = np.where(weak, np.nan, pyq / mx)
    ax.quiver(lon_q, lat_q, pxq, pyq, transform=transform, scale=qcfg['scale'],
              scale_units=qcfg['scale_units'], width=float(qcfg['width']),
              headwidth=float(qcfg['headwidth']), headlength=float(qcfg['headlength']),
              color='black', zorder=60)
    # colorbars: PSI (base) e magnitude (sobreposta). Polar: ambas presas ao ax; senao bottom+right.
    if is_polar and area != 'globo_3d':
        cb_psi = plt.colorbar(im_psi, ax=ax, pad=0.05, fraction=0.04, ticks=ticks)
        cb_psi.set_label(label=r'PSI 10$^6$ m$^2$ s$^{-1}$', size=10)
        cb_psi.ax.tick_params(labelsize=10)
        if im_mag is not None:
            cb_mag = plt.colorbar(im_mag, ax=ax, pad=0.12, fraction=0.04, ticks=WIND_MAG_TICKS)
            cb_mag.set_label(label='|V200| m/s', size=10)
            cb_mag.ax.tick_params(labelsize=10)
    else:
        divider = make_axes_locatable(ax)
        cax_psi = divider.append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)
        cb_psi = plt.colorbar(im_psi, cax=cax_psi, extend='both', location='bottom',
                              orientation='horizontal', ticks=ticks)
        cb_psi.set_label(label=r'Anomalia PSI200 (10$^6$ m$^2$ s$^{-1}$)', size=16)
        cb_psi.ax.tick_params(labelsize=16)
        if im_mag is not None:
            cax_mag = divider.append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
            cb_mag = plt.colorbar(im_mag, cax=cax_mag, extend='max', ticks=WIND_MAG_TICKS)
            cb_mag.set_label(label='Magnitude media |V200| (m/s)', size=16)
            cb_mag.ax.tick_params(labelsize=16)
    _finish_map(fig, ax, is_polar,
                'Anomalia PSI200 + |V200| media + linhas de corrente + Z250 + WAF',
                ini_dt, fim_dt, fcst_info, entrada_dir, out_path, logger)


def _plot_chi_jet_area(
    area, info_plot, chi_cyc, lon_chi, lat_chi, uchi_cyc, vchi_cyc, lon_dv, lat_dv,
    jet_cyc, lon_j, lat_j, ini_dt, fim_dt, entrada_dir, out_path, fcst_info, logger,
    z250_cyc=None, lon_z2=None, lat_z2=None,
):
    """Mapa chi_jato: anomalia CHI200 (shaded verde-marrom s03) + vento divergente 200 anomalo
    (vetores) + magnitude media do jato em 250 hPa (shaded s16, transparente <35) + isolinhas
    Z250 (jato, estilo s16). Sem WAF."""
    fig, ax, is_polar, transform = _setup_map_ax(area, info_plot)
    # 1) base: anomalia de CHI200 (shaded)
    n_bins = len(CHI200_LEVELS) - 1
    cmap_chi = LinearSegmentedColormap.from_list('chi200_green_brown', CHI200_PALETTE, N=n_bins)
    norm_chi = BoundaryNorm(CHI200_LEVELS, ncolors=n_bins, clip=False)
    im_chi = ax.contourf(
        lon_chi, lat_chi, chi_cyc, levels=CHI200_LEVELS, cmap=cmap_chi, norm=norm_chi, extend='both',
        transform=transform, zorder=10,
    )
    # 2) magnitude media do jato 250 sobreposta (paleta s16; SO o jato >= 35 m/s; abaixo transparente)
    cmap_jet = LinearSegmentedColormap.from_list('wnd_speed', WIND_MAG_PALETTE)
    jet_present = jet_cyc is not None and np.isfinite(jet_cyc).any() and float(np.nanmax(jet_cyc)) >= WIND_MAG_LEVELS[0]
    im_jet = None
    if jet_present:
        im_jet = ax.contourf(
            lon_j, lat_j, jet_cyc, levels=WIND_MAG_LEVELS, cmap=cmap_jet, extend='max',
            transform=transform, zorder=20,
        )
    # 3) vento divergente 200 anomalo (vetores, mesmo estilo do mapa de RWS)
    step = DIVWIND_QUIVER['step']
    lon_q, lat_q = lon_dv[::step], lat_dv[::step]
    u_q = uchi_cyc[::step, ::step].copy()
    v_q = vchi_cyc[::step, ::step].copy()
    mag = np.sqrt(u_q ** 2 + v_q ** 2)
    weak = mag < DIVWIND_QUIVER['min_mag']
    u_q[weak] = np.nan
    v_q[weak] = np.nan
    ax.quiver(lon_q, lat_q, u_q, v_q, transform=transform, scale=DIVWIND_QUIVER['scale'],
              width=DIVWIND_QUIVER['width'], color='black', zorder=60)
    # 4) isolinhas de altura geopotencial 250 hPa (jato) — mesmas do psi_jato (estilo s16, zorder alto)
    _draw_z250_isolines(ax, z250_cyc, lon_z2, lat_z2, transform)
    # colorbars: CHI (base) e |V250| (sobreposta). Polar: presas ao ax; senao bottom+right.
    if is_polar and area != 'globo_3d':
        cb_chi = plt.colorbar(im_chi, ax=ax, pad=0.05, fraction=0.04, ticks=CHI200_TICKS)
        cb_chi.set_label(label=r'CHI200 10$^5$ m$^2$ s$^{-1}$', size=10)
        cb_chi.ax.tick_params(labelsize=10)
        if im_jet is not None:
            cb_jet = plt.colorbar(im_jet, ax=ax, pad=0.12, fraction=0.04, ticks=WIND_MAG_TICKS)
            cb_jet.set_label(label='|V250| m/s', size=10)
            cb_jet.ax.tick_params(labelsize=10)
    else:
        divider = make_axes_locatable(ax)
        cax_chi = divider.append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)
        cb_chi = plt.colorbar(im_chi, cax=cax_chi, extend='both', location='bottom',
                              orientation='horizontal', ticks=CHI200_TICKS)
        cb_chi.set_label(label=r'Anomalia CHI200 (10$^5$ m$^2$ s$^{-1}$)', size=16)
        cb_chi.ax.tick_params(labelsize=16)
        if im_jet is not None:
            cax_jet = divider.append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
            cb_jet = plt.colorbar(im_jet, cax=cax_jet, extend='max', ticks=WIND_MAG_TICKS)
            cb_jet.set_label(label='Magnitude media do jato |V250| (m/s)', size=16)
            cb_jet.ax.tick_params(labelsize=16)
    _finish_map(fig, ax, is_polar,
                'Anomalia CHI200 + vento divergente 200 + jato |V250| + Z250',
                ini_dt, fim_dt, fcst_info, entrada_dir, out_path, logger)


def _plot_t2m_wnd850_area(
    area, info_plot, t2m_cyc, lon_t2, lat_t2, u850_cyc, v850_cyc, lon_w_native, lat_w,
    z5m_cyc, lon_z5, lat_z5, ini_dt, fim_dt, entrada_dir, out_path, fcst_info, logger,
):
    """Mapa t2m_wnd850: anomalia de T2m (shaded) + linhas de corrente do vento 850 ANOMALO +
    isolinhas pretas da Z500 media do periodo (sem rotulo). Sem WAF."""
    cfg = info_plot[area]
    fig, ax, is_polar, transform = _setup_map_ax(area, info_plot)
    cmap_t2m = LinearSegmentedColormap.from_list('anom', settings.LST_ANOM_CORRETA)
    im = ax.contourf(
        lon_t2, lat_t2, t2m_cyc, levels=T2M_ANOM_LEVELS, cmap=cmap_t2m, extend='both',
        transform=transform, zorder=10,
    )
    # linhas de corrente do vento 850 ANOMALO (estilo s04) — inclusive na ortografica (globo_3d).
    if u850_cyc is not None:
        slcfg = _get_streamline_config(area)
        central_lon = 0 if is_polar else cfg['central_longitude_mapa']
        lon_sl, u_sl, v_sl = _prepare_streamline_lonuv(lon_w_native, u850_cyc, v850_cyc, central_lon)
        try:
            ax.streamplot(
                lon_sl, lat_w, u_sl, v_sl, transform=ccrs.PlateCarree(central_longitude=central_lon),
                density=float(slcfg['density']), linewidth=float(slcfg['linewidth']),
                arrowsize=float(slcfg['arrowsize']), color=slcfg['color'], zorder=30,
            )
        except Exception as exc:
            logger.warning('  streamplot vento850 falhou em {} ({}) — mapa sem linhas', area, exc)
    # isolinhas pretas da Z500 media (sem rotulo), grossas, por cima (abaixo do logo z=500)
    if z5m_cyc is not None:
        ax.contour(lon_z5, lat_z5, z5m_cyc, levels=Z500_MEAN_LEVELS, colors='black',
                   linewidths=1.6, transform=transform, zorder=400)
    if is_polar and area != 'globo_3d':
        cbar = plt.colorbar(im, ax=ax, pad=0.05, fraction=0.04, ticks=T2M_ANOM_TICKS)
        cbar.set_label(label='°C', size=10)
        cbar.ax.tick_params(labelsize=10)
    else:
        divider = make_axes_locatable(ax)
        if area in {'america_sul', 'globo_3d'}:
            cax = divider.append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
            cbar = plt.colorbar(im, cax=cax, extend='both', ticks=T2M_ANOM_TICKS)
        else:
            cax = divider.append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)
            cbar = plt.colorbar(im, cax=cax, extend='both', location='bottom',
                                orientation='horizontal', ticks=T2M_ANOM_TICKS)
        cbar.set_label(label='Anomalia de temperatura do ar a 2 m (°C)', size=16)
        cbar.ax.tick_params(labelsize=16)
    _finish_map(fig, ax, is_polar,
                'Anomalia T2m + vento 850 anomalo (linhas de corrente) + Z500 media',
                ini_dt, fim_dt, fcst_info, entrada_dir, out_path, logger)


def _plot_spatial_window(
    fields, lat, lon, lst_areas, info_plot, win_ini_dt, win_fim_dt,
    entrada_dir, ks_paths, rws_paths, waf_paths, olr_paths, tmp_paths, hgt_paths, psi_paths,
    chi_paths, t2m_paths, fcst_info, logger,
):
    """Prepara os campos da janela (cyclic) e plota Ks + RWS + WAF/Z200 + WAF/OLR + WAF/T850
    + hgt500 + psi_jato + chi_jato."""
    ks_da = _to_180(xr.DataArray(fields['ks'], dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}))
    u180 = _to_180(xr.DataArray(fields['u_mean'], dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}))
    plon = ks_da['lon'].values
    ks_cyc, lon_cyc = _acp(ks_da.values, coord=plon)
    u_cyc, _ = _acp(u180.values, coord=plon)
    hgt_cyc, lon_hgt, lat_hgt = _prep_cyclic_180(fields['hgt_anom'])
    waf_coords = {'lat': fields['lat_waf'], 'lon': fields['lon_waf']}
    px_cyc, lon_waf_c, lat_waf_c = _prep_cyclic_180(
        xr.DataArray(fields['px'], dims=('lat', 'lon'), coords=waf_coords))
    py_cyc, _, _ = _prep_cyclic_180(xr.DataArray(fields['py'], dims=('lat', 'lon'), coords=waf_coords))
    rws_cyc, lon_rws, lat_rws = _prep_cyclic_180(
        xr.DataArray(
            fields['rws_anom'] * RWS_SCALE, dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}))
    uchi_cyc, _, _ = _prep_cyclic_180(
        xr.DataArray(fields['uchi'], dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}))
    vchi_cyc, _, _ = _prep_cyclic_180(
        xr.DataArray(fields['vchi'], dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}))
    has_olr_map = 'olr_anom' in fields and bool(olr_paths)  # AIFS nao tem OLR
    if has_olr_map:
        olr_cyc, lon_olr, lat_olr = _prep_cyclic_180(
            xr.DataArray(fields['olr_anom'], dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}))
    has_t_map = 't_anom' in fields and bool(tmp_paths)  # T850 opcional (CFS nao tem)
    if has_t_map:
        t_cyc, lon_t, lat_t = _prep_cyclic_180(
            xr.DataArray(fields['t_anom'], dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}))

    has_hgt500 = 'z500_anom' in fields and bool(hgt_paths)
    if has_hgt500:
        z5a_cyc, lon_z5, lat_z5 = _prep_cyclic_180(
            xr.DataArray(fields['z500_anom'], dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}))
        z5m_cyc, _, _ = _prep_cyclic_180(
            xr.DataArray(fields['z500_mean'], dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}))
    # Z250 media (isolinhas de jato) — compartilhado por psi_jato E chi_jato; opcional (pode faltar)
    z250_cyc = lon_z2 = lat_z2 = None
    if 'z250_mean' in fields:
        z250_cyc, lon_z2, lat_z2 = _prep_cyclic_180(
            xr.DataArray(fields['z250_mean'], dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}))
    has_psi = 'psi_anom' in fields and bool(psi_paths)
    if has_psi:
        psi_cyc, lon_psi, lat_psi = _prep_cyclic_180(
            xr.DataArray(fields['psi_anom'], dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}))
        mag_cyc, lon_mag, lat_mag = _prep_cyclic_180(
            xr.DataArray(fields['wind_mag'], dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}))
        # vento rotacional p/ streamlines: cyclic na grade NATIVA 0..360 (deslocado por area depois)
        urot_cyc, lon_rot_native = _acp(fields['urot'], coord=lon)
        vrot_cyc, _ = _acp(fields['vrot'], coord=lon)

    # chi_jato: CHI200 anom (shaded) + vento divergente 200 anom (vetores) + |V250| media (shaded s16)
    has_chi = 'chi_anom' in fields and bool(chi_paths)
    if has_chi:
        chi_cyc, lon_chi, lat_chi = _prep_cyclic_180(
            xr.DataArray(fields['chi_anom'], dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}))
        uchi_cyc2, lon_dv, lat_dv = _prep_cyclic_180(
            xr.DataArray(fields['uchi'], dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}))
        vchi_cyc2, _, _ = _prep_cyclic_180(
            xr.DataArray(fields['vchi'], dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}))
        jet250_cyc = lon_j2 = lat_j2 = None
        if 'jet250_mag' in fields:
            jet250_cyc, lon_j2, lat_j2 = _prep_cyclic_180(
                xr.DataArray(fields['jet250_mag'], dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}))

    # t2m_wnd850: T2m anom (shaded) + streamlines do vento 850 anom + isolinhas pretas Z500 media
    has_t2m = 't2m_anom' in fields and bool(t2m_paths)
    if has_t2m:
        t2m_cyc, lon_t2, lat_t2 = _prep_cyclic_180(
            xr.DataArray(fields['t2m_anom'], dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}))
        u850a_cyc = v850a_cyc = lon_w_native = None
        if 'u850_anom' in fields:
            u850a_cyc, lon_w_native = _acp(fields['u850_anom'], coord=lon)
            v850a_cyc, _ = _acp(fields['v850_anom'], coord=lon)
        z5m_t_cyc = lon_z5t = lat_z5t = None
        if 'z500_mean' in fields:
            z5m_t_cyc, lon_z5t, lat_z5t = _prep_cyclic_180(
                xr.DataArray(fields['z500_mean'], dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}))

    cmap_olr = plt.get_cmap('BrBG_r')
    cmap_t = LinearSegmentedColormap.from_list('anom', settings.LST_ANOM_CORRETA)
    for area in lst_areas:
        _plot_ks_area(
            area=area, info_plot=info_plot, lon_cyc=lon_cyc, lat=lat, ks_cyc=ks_cyc, u_cyc=u_cyc,
            hgt_cyc=hgt_cyc, lon_hgt=lon_hgt, lat_hgt=lat_hgt,
            px_cyc=px_cyc, py_cyc=py_cyc, lon_waf=lon_waf_c, lat_waf=lat_waf_c,
            ini_dt=win_ini_dt, fim_dt=win_fim_dt, entrada_dir=entrada_dir,
            out_path=ks_paths[area], fcst_info=fcst_info, logger=logger,
        )
        _plot_rws_area(
            area=area, info_plot=info_plot, lon_rws=lon_rws, lat_rws=lat_rws,
            rws_cyc=rws_cyc, uchi_cyc=uchi_cyc, vchi_cyc=vchi_cyc,
            ini_dt=win_ini_dt, fim_dt=win_fim_dt, entrada_dir=entrada_dir,
            out_path=rws_paths[area], fcst_info=fcst_info, logger=logger,
        )
        _plot_waf_z200_area(
            area=area, info_plot=info_plot, hgt_cyc=hgt_cyc, lon_hgt=lon_hgt, lat_hgt=lat_hgt,
            px_cyc=px_cyc, py_cyc=py_cyc, lon_waf=lon_waf_c, lat_waf=lat_waf_c,
            z250_cyc=z250_cyc, lon_z2=lon_z2, lat_z2=lat_z2,
            ini_dt=win_ini_dt, fim_dt=win_fim_dt, entrada_dir=entrada_dir,
            out_path=waf_paths[area], fcst_info=fcst_info, logger=logger,
        )
        if has_olr_map:
            _plot_waf_field_area(
                area=area, info_plot=info_plot, field_cyc=olr_cyc, lon_f=lon_olr, lat_f=lat_olr,
                hgt_cyc=hgt_cyc, lon_hgt=lon_hgt, lat_hgt=lat_hgt,
                px_cyc=px_cyc, py_cyc=py_cyc, lon_waf=lon_waf_c, lat_waf=lat_waf_c,
                cmap=cmap_olr, levels=OLR_LEVELS, ticks=OLR_TICKS, cbar_label='Anomalia OLR (W/m2)',
                titulo='Anomalia OLR + WAF (200 hPa) + Z200',
                ini_dt=win_ini_dt, fim_dt=win_fim_dt, entrada_dir=entrada_dir,
                out_path=olr_paths[area], fcst_info=fcst_info, logger=logger,
            )
        if has_t_map:
            _plot_waf_field_area(
                area=area, info_plot=info_plot, field_cyc=t_cyc, lon_f=lon_t, lat_f=lat_t,
                hgt_cyc=hgt_cyc, lon_hgt=lon_hgt, lat_hgt=lat_hgt,
                px_cyc=px_cyc, py_cyc=py_cyc, lon_waf=lon_waf_c, lat_waf=lat_waf_c,
                cmap=cmap_t, levels=TMP850_LEVELS, ticks=TMP850_TICKS,
                cbar_label='Anomalia T850 (°C)', titulo='Anomalia T850 + WAF (200 hPa) + Z200',
                ini_dt=win_ini_dt, fim_dt=win_fim_dt, entrada_dir=entrada_dir,
                out_path=tmp_paths[area], fcst_info=fcst_info, logger=logger,
            )
        if has_hgt500:
            _plot_hgt500_area(
                area=area, info_plot=info_plot, z500_anom_cyc=z5a_cyc, z500_mean_cyc=z5m_cyc,
                lon_z=lon_z5, lat_z=lat_z5, ini_dt=win_ini_dt, fim_dt=win_fim_dt,
                entrada_dir=entrada_dir, out_path=hgt_paths[area], fcst_info=fcst_info, logger=logger,
            )
        if has_psi:
            _plot_psi_jet_area(
                area=area, info_plot=info_plot, psi_cyc=psi_cyc, lon_psi=lon_psi, lat_psi=lat_psi,
                mag_cyc=mag_cyc, lon_mag=lon_mag, lat_mag=lat_mag,
                urot_cyc=urot_cyc, vrot_cyc=vrot_cyc, lon_rot_native=lon_rot_native, lat_rot=lat,
                px_cyc=px_cyc, py_cyc=py_cyc, lon_waf=lon_waf_c, lat_waf=lat_waf_c,
                z250_cyc=z250_cyc, lon_z2=lon_z2, lat_z2=lat_z2,
                ini_dt=win_ini_dt, fim_dt=win_fim_dt, entrada_dir=entrada_dir,
                out_path=psi_paths[area], fcst_info=fcst_info, logger=logger,
            )
        if has_chi:
            _plot_chi_jet_area(
                area=area, info_plot=info_plot, chi_cyc=chi_cyc, lon_chi=lon_chi, lat_chi=lat_chi,
                uchi_cyc=uchi_cyc2, vchi_cyc=vchi_cyc2, lon_dv=lon_dv, lat_dv=lat_dv,
                jet_cyc=jet250_cyc, lon_j=lon_j2, lat_j=lat_j2,
                z250_cyc=z250_cyc, lon_z2=lon_z2, lat_z2=lat_z2,
                ini_dt=win_ini_dt, fim_dt=win_fim_dt, entrada_dir=entrada_dir,
                out_path=chi_paths[area], fcst_info=fcst_info, logger=logger,
            )
        if has_t2m:
            _plot_t2m_wnd850_area(
                area=area, info_plot=info_plot, t2m_cyc=t2m_cyc, lon_t2=lon_t2, lat_t2=lat_t2,
                u850_cyc=u850a_cyc, v850_cyc=v850a_cyc, lon_w_native=lon_w_native, lat_w=lat,
                z5m_cyc=z5m_t_cyc, lon_z5=lon_z5t, lat_z5=lat_z5t,
                ini_dt=win_ini_dt, fim_dt=win_fim_dt, entrada_dir=entrada_dir,
                out_path=t2m_paths[area], fcst_info=fcst_info, logger=logger,
            )


def _plot_hovmoller(
    hov_v, hov_band, nome, ini_dt, fim_dt, total_days, entrada_dir, out_path, fcst_info, logger,
):
    """Hovmoller de v'200 na faixa do jato: bandas verticais=estacionaria, inclinadas=propagante.

    Eixo X em 0..360 (Pacifico inteiro no centro, data line em 180°), com rotulos W/E.
    """
    # cyclic point: fecha a costura repetindo lon=0 em lon=360 (faixa global continua)
    hov_v = xr.concat(
        [hov_v, hov_v.isel(lon=0).assign_coords(lon=360.0)], dim='lon',
    )
    lons_plot = hov_v['lon'].values  # 0..360
    times = hov_v['time'].values
    ytick_interval = 5 if total_days > 31 else 1

    cmap = LinearSegmentedColormap.from_list('anom', settings.LST_ANOM_CORRETA)

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.contourf(
        lons_plot, mdates.date2num(times), hov_v.values,
        levels=HOV_LEVELS, cmap=cmap, extend='both',
    )

    ax.yaxis.set_major_locator(mdates.DayLocator(interval=ytick_interval))
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    ax.invert_yaxis()

    # Ticks absolutos 0..360, rotulados em W/E (180 = data line, no centro)
    raw_ticks = np.arange(0, 361, 40)
    labels = [_fmt_lon(int(((t + 180) % 360) - 180)) for t in raw_ticks]
    ax.set_xticks(raw_ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(0, 360)
    ax.tick_params(axis='both', labelsize=15)

    ax.set_xlabel('Longitude', fontsize=16, labelpad=6)
    ax.set_ylabel('Data', fontsize=16, labelpad=6)

    cax = make_axes_locatable(ax).append_axes('right', size='3%', pad=0.15)
    cbar = fig.colorbar(im, cax=cax, ticks=HOV_TICKS)
    cbar.set_label("v'200 (m/s)", fontsize=14)
    cbar.ax.tick_params(labelsize=13)

    ini_fmt = ini_dt.strftime('%d/%m/%Y')
    fim_fmt = fim_dt.strftime('%d/%m/%Y')
    ax.set_title(
        _title(
            f'Hovmöller — Anomalia de Vento Meridional 200 hPa ({hov_band})\n'
            f'{nome} (bandas verticais = onda estacionaria; inclinadas = propagante)',
            f'De {ini_fmt} a {fim_fmt}', fcst_info),
        fontsize=12, loc='left',
    )

    logo_path = _resolve_logo_path(entrada_dir)
    if logo_path is not None:
        _add_logo_to_map(ax=ax, logo_path=logo_path, zoom=0.55)

    logger.info(f'Salvando Hovmoller: {out_path}')
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
