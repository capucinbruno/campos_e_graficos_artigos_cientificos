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

from scipy.ndimage import gaussian_filter as _gaussian_filter

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.util import add_cyclic_point
from matplotlib import patheffects as path_effects
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle

# Modulos locais
from app.shared.logger import get_logger
from app.src.uteis.forecast_daily import (
    daily_mslp_on_grid as _daily_mslp_on_grid,
    daily_scalar_on_grid as _daily_scalar_on_grid,
    daily_uv200_on_grid as _daily_uv200_on_grid,
    daily_wind_speed_on_grid as _daily_wind_speed_on_grid,
    lagged_ensemble_mean as _lagged_ensemble_mean,
    resolve_forecast_lead_init as _resolve_forecast_lead_init,
)
from app.shared.settings_factory import settings
from app.src.uteis.clim_diaria_uv200_ltm import (
    clim_hgt250_daily,
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


def _fmt_data_pentada(d0, dias: int, *, com_ano: bool) -> str:
    """Rotulo de DATA da pentada movel: intervalo [d0, d0+dias-1] em ingles US.

    Ex.: 'July 20–24, 2026' (mesmo mes) ou 'July 29 – August 2, 2026' (cruza o mes).
    `com_ano=False` omite o ano (rotulo curto). dias<=1 cai p/ data unica.
    """
    d0 = pd.Timestamp(d0)
    if dias <= 1:
        return d0.strftime('%B %-d, %Y') if com_ano else d0.strftime('%B %-d')
    d1 = d0 + pd.Timedelta(days=dias - 1)
    ano = f', {d1.year}' if com_ano else ''
    if d0.month == d1.month:
        return f"{d0.strftime('%B %-d')}–{d1.strftime('%-d')}{ano}"  # July 20–24, 2026
    return f"{d0.strftime('%B %-d')} – {d1.strftime('%B %-d')}{ano}"  # July 29 – August 2, 2026


# Modelos de forecast suportados para z250 (tem downloader de hgt250 dedicado).
_FCST_FLAGS = {'gfs': 'RUN_GFS', 'gefs': 'RUN_GEFS', 'ecmwf': 'RUN_ECMWF', 'aifs': 'RUN_AIFS'}


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
        # u/v 200 hPa (PSI200) — um arquivo diario com u, v e hgt em 200 hPa por modelo.
        ('gfs', 'fcst200'): ('downloaders_gfs_fcst200', 'ensure_gfs_fcst200_for_period'),
        ('gefs', 'fcst200'): ('downloaders_gefs_fcst200', 'ensure_gefs_fcst200_for_period'),
        ('ecmwf', 'fcst200'): ('downloaders_ecmwf_fcst200', 'ensure_ecmwf_fcst200_for_period'),
        ('aifs', 'fcst200'): ('downloaders_aifs_fcst200', 'ensure_aifs_fcst200_for_period'),
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
    """Anomalia diaria de PSI200 (campo='psi') ou CHI200 (campo='chi') das series u/v 200.

    Subtrai a climatologia diaria de u/v (LTM NCEP 1991-2020, mesma grade 2.5°) e inverte
    a equacao de Poisson (vorticidade->psi ou divergencia->chi) do vento de anomalia, dia a dia.
    """
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
    nome = _UV200_DERIVADO[campo]['nome']
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


def _era5_mslp(start, end, force):
    from app.src.uteis.downloaders_wind100m_ERA5 import ensure_era5_mslp_global_for_period as fn
    return fn(start=start, end=end, hours_utc=list(DEFAULT_SYNOPTIC_HOURS), force_redownload=force)


def _build_mslp_series(dt_ini: datetime, dt_fim: datetime) -> xr.DataArray | None:
    """Serie diaria de MSLP (hPa) regridada — ERA5 apenas (GDAS nao tem downloader MSLP)."""
    force = bool(getattr(settings, 'FORCE_DOWNLOAD', False))
    era5_period, _ = _get_data_sources(dt_ini, dt_fim)
    if not era5_period:
        return None
    era5_fim = min(dt_fim, era5_period[1])
    logger.info('Download ERA5 MSLP: {} -> {}', era5_period[0].date(), era5_fim.date())
    files = _era5_mslp(era5_period[0], era5_fim, force)
    tgt_lat, tgt_lon = _target_grid()
    return _daily_mslp_on_grid(files, dt_ini, era5_fim, tgt_lat, tgt_lon, logger)


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
    """Campo ABSOLUTO observado (ERA5/GDAS) — sem subtracao de climatologia."""
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
    da = _daily_wind_speed_on_grid(files, dt_ini, dt_fim, tgt_lat, tgt_lon, logger)
    logger.info('{}: {} dias | min={:.1f} max={:.1f} {}',
                spec['nome'], da.sizes['time'], float(da.min()), float(da.max()), spec['unidade'])
    return da


def _absolute_forecast_series(ficha: dict, model: str, dt_ini: datetime, dt_fim: datetime) -> xr.DataArray | None:
    """Campo ABSOLUTO previsto (lagged ensemble) — sem subtracao de climatologia."""
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
    logger.info('FORECAST {} [{}]: init {:%Y-%m-%d %H}Z, lead {}h | janela {} a {}',
                model.upper(), spec['nome'], init0, lead_hours, win_ini.date(), win_fim.date())
    fn = _fcst_downloader(model, spec['kind'])
    tgt_lat, tgt_lon = _target_grid()
    per_run: list[xr.DataArray] = []
    for init_k in run_inits:
        files_k = list(fn(init=init_k, lead_hours=lead_hours, hours=list(DEFAULT_SYNOPTIC_HOURS)))
        if files_k:
            per_run.append(_daily_wind_speed_on_grid(files_k, win_ini, win_fim, tgt_lat, tgt_lon, logger))
    if not per_run:
        raise RuntimeError(f'Sem dados de {spec["nome"]} do modelo {model.upper()} no horizonte.')
    da = _lagged_ensemble_mean(per_run)
    da.attrs['run_init'] = init0.strftime('%Y-%m-%d %H')
    return da


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
    """
    h = max(2, px // 2)
    fig = plt.figure(figsize=(px / 100.0, h / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(float(lon.min()), float(lon.max()))
    ax.set_ylim(float(lat.min()), float(lat.max()))
    ax.contourf(lon, lat, campo, levels=levels, cmap=cmap, extend=extend, antialiased=True)
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


def _build_var_series(ficha: dict, model: str | None,
                      dt_ini: datetime, dt_fim: datetime) -> xr.DataArray:
    """Serie diaria na janela [dt_ini, dt_fim], decidida PELAS DATAS:
    passado -> reanalise; futuro -> previsao (modelo); janela que cruza hoje ->
    EMENDA observado + previsao num unico vetor temporal continuo.
    Variaveis com 'absoluto': True retornam o campo bruto sem subtracao de climatologia.
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
            return abs_partes[0]
        abs_serie = xr.concat(abs_partes, dim='time').sortby('time')
        _, abs_idx = np.unique(abs_serie['time'].values, return_index=True)
        abs_serie = abs_serie.isel(time=np.sort(abs_idx))
        for p in abs_partes:
            if 'run_init' in p.attrs:
                abs_serie.attrs['run_init'] = p.attrs['run_init']
        return abs_serie

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
        return _aplicar_pentada_movel(_validar_dias_degenerados(partes[0], ficha, logger), ficha)

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
    return _aplicar_pentada_movel(_validar_dias_degenerados(serie, ficha, logger), ficha)


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
    'tmp850_mslp': {
        'titulo': 'Anomalia de Temperatura do Ar em 850 hPa + PNMM',
        'titulo_en': '850-hPa air temperature',
        'rotulo_box': 'Air Temperature Anomaly',
        'subtitulo_dir': 'Air temperature at 850 hPa + MSLP',
        'unidade': '°C',
        'isolinha_abs_0': True,
        'isolinha_mslp': True,   # isolinhas de PNMM em hPa (ERA5 apenas)
        'cmap_colors': [
            '#a30d92', '#720669', '#3c0654', '#17022b', '#021323',
            '#043462', '#104b87', '#256aab', '#3d89bd', '#5ea3cc',
            '#bad9eb', '#e1ebf4', '#f7f7f7', '#f8ede7', '#f9dac8',
            '#e58366', '#cd5147', '#b72532', '#9c1526', '#821220',
            '#5f0d19', '#3c0711', '#25060b', '#0d0707', '#3d3d3d',
        ],
        'niveis': 128,
        'simetrico': True,
        'vmax': 15.0,
        'spec': {
            'nome': 'tmp850_mslp', 'unidade': '°C', 'celsius': True,
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
        'spec': {
            'nome': 'tsm_anom', 'unidade': '°C', 'celsius': False,
            'var_candidates': ('sst',), 'clim_fn': None, 'kind': 'sst',
            'reanalise_fn': _tsm_reanalise_series,
            'forecast_fn': _tsm_forecast_series,  # SST nao tem previsao -> erro claro no forecast
            'era5_fn': None, 'gdas_fn': None,
        },
    },
    'wind250_abs': {
        'titulo': 'Magnitude do Vento em 250 hPa (Jet Stream)',
        'titulo_en': '250-hPa wind speed (m/s)',  # unidade no titulo (s38/s39)
        'rotulo_box': 'Jet Stream',               # caixa do s39
        'subtitulo_dir': 'Wind speed at 250 hPa',
        'unidade': 'ms⁻¹',
        'absoluto': True,                     # campo absoluto — sem subtracao de climatologia
        'simetrico': False,
        'vmin': float(settings.get('GLOBO_3D_VMIN_WIND250_ABS', 30.0)),  # abaixo = transparente
        'vmax': float(settings.get('GLOBO_3D_VMAX_WIND250_ABS', 90.0)),
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
            'nome': 'wind250_abs', 'unidade': 'ms⁻¹', 'kind': 'uv250',
            'era5_fn': _era5_uv250, 'gdas_fn': _gdas_uv250,
            'hgt_clim_fn': clim_hgt250_daily,  # Z250 clim p/ isolinhas no jet stream
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
_OV_CACHE_KEY: int | None = None   # último idx_dia do overlay Guillaume (por worker)
_OV_CACHE_VAL: np.ndarray | None = None  # overlay correspondente


def _init_frame_worker(ctx: dict) -> None:
    """Inicializa cada processo worker com o contexto do clipe (1x por worker)."""
    global _FRAME_CTX
    _FRAME_CTX = ctx


def _numeric_legend_ticks(vmin: float, vmax: float, step: float) -> list[float]:
    """Ticks numericos simetricos em torno de 0, de `step` em `step`, dentro de [vmin, vmax]."""
    if step <= 0:
        step = 0.5
    n = int(np.floor(max(abs(vmin), abs(vmax)) / step + 1e-9))
    vals = [round(k * step, 6) for k in range(-n, n + 1)]
    return [v for v in vals if vmin - 1e-9 <= v <= vmax + 1e-9]


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
        facecolor='#9a9a9a', alpha=1.0, edgecolor='none', zorder=20))
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
        # Unidade centralizada abaixo dos labels (ex.: 'm/s' para wind250_abs)
        if ctx.get('legenda_unidade'):
            fig.text(gl + gw / 2, gb - 0.026, ctx['legenda_unidade'], color='#c0c0c0',
                     fontsize=7.5, ha='center', va='top', family=ctx['font_legenda'], zorder=22)

    # ── Rodape: apenas o credito (dir.). O modelo/rodada subiu p/ baixo da data. ──
    fig.text(0.98, 0.028, ctx['credito'], color='#cfcfcf', fontsize=8.5,
             ha='right', va='center', family=FONT_SANS, zorder=21)


def _build_frame(f: int, ctx: dict) -> np.ndarray:
    """Renderiza o frame `f` e devolve um array RGB uint8 (HxWx3)."""
    vals_cyc = ctx['vals_cyc']
    lon_cyc, lat, levels = ctx['lon_cyc'], ctx['lat'], ctx['levels']
    n_dias, fpd, vel = ctx['n_dias'], ctx['frames_por_dia'], ctx['vel_var']
    # cmap_plot/cmap_legend herdados do pai via CoW (criados 1x em _render_clip).
    cmap_plot   = ctx['cmap_plot']
    cmap_legend = ctx['cmap_legend']
    paleta      = ctx['paleta']   # necessário para detecção de transparência no contorno
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
    data_wapo = ctx['dates_wapo'][idx_dia] if ctx.get('dates_wapo') else data_en
    guillaume = ctx.get('estilo') == 'guillaume'

    proj = _make_projection(float(ctx['lons'][f]), float(ctx['lats'][f]))
    fig = plt.figure(figsize=(8, 8), dpi=ctx['dpi'])
    fig.patch.set_facecolor('black')
    # Guillaume: globo grande (disco raio ~0.43, topo y~0.92), preenchendo o quadro
    # como na referencia; o canto sup-esq fica livre p/ a caixa compacta do nome/data
    # (no x=0.235 da caixa, o topo do globo cai p/ ~0.83, abaixo da data).
    # WaPo: rect [0.01, 0.10, 0.98, 0.83] -> r=0.415, cy=0.515, disc_top=0.930.
    # O disco desce até y=0.10 (dentro da tarja), mas o Rectangle preto (zorder=15)
    # mascara a parte do globo abaixo de barra_h=0.17; o bar_backup elimina halo/estrelas.
    rect = [0.07, 0.06, 0.86, 0.86] if guillaume else [0.01, 0.10, 0.98, 0.83]
    ax = fig.add_axes(rect, projection=proj)
    ax.patch.set_facecolor(ctx.get('cor_fundo_globo', 'black'))
    ax.set_global()
    if guillaume:
        ax.spines['geo'].set_linewidth(0)  # remove o anel preto; o halo azul define a borda

    # Continentes coloridos (para variaveis absolutas onde abaixo do vmin = transparente)
    if ctx.get('cor_continente'):
        ax.add_feature(cfeature.LAND.with_scale('50m'),
                       facecolor=ctx['cor_continente'], zorder=1)

    if ctx.get('usar_pcolormesh'):
        # Gradiente CONTINUO (gouraud) — liso, sem bandas.
        ax.pcolormesh(lon_cyc, lat, campo, norm=ctx['norm_fn'], cmap=cmap_plot,
                      transform=data_transform, shading='gouraud', zorder=2)
    else:
        # BANDAS suaves do contourf (estilo WaPo) SEM o bug do cartopy: o contourf e renderizado
        # num raster PLANO (_contourf_raster) e composto no globo via imshow (reprojecao de raster).
        # Desenhar ax.contourf direto no globo aciona o bug cartopy 0.25+mpl 3.10 (geometrias ->
        # TypeError -> _safe_transform_path devolve path vazio -> poligonos descartados -> washout).
        _shade_px = int(ctx.get('shade_px', 3600))
        _regrid = int(ctx.get('shade_regrid', 2048))  # resolucao da REPROJECAO (cartopy default=750=mole)
        flat_rgba = _contourf_raster(lon_cyc, lat, campo, levels, cmap_plot,
                                     ctx.get('extend_contourf', 'both'), px=_shade_px)
        ax.imshow(flat_rgba, origin='upper', transform=data_transform, zorder=2,
                  extent=[float(lon_cyc.min()), float(lon_cyc.max()),
                          float(lat.min()), float(lat.max())],
                  interpolation='bilinear', regrid_shape=_regrid)

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
    # Isolinhas fixas coloridas de Z250 absoluto (ex.: 10080/10200/10680 mgp no jet stream)
    if ctx.get('isolinhas_fixas_hgt') and ctx.get('hgt_z250_abs_cyc') is not None:
        _hgt_z250 = (1.0 - w) * ctx['hgt_z250_abs_cyc'][i0] + w * ctx['hgt_z250_abs_cyc'][i1]
        for _nivel, _cor, _lw in ctx['isolinhas_fixas_hgt']:
            ax.contour(lon_cyc, lat, _hgt_z250, levels=[float(_nivel)],
                       colors=[_cor], linewidths=_lw, transform=data_transform, zorder=7)
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

    # ── Caixa do Niño 3.4 (170°W–120°W, 5°S–5°N) + rotulo — so fichas com box_nino34 (tsm_anom) ──
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
        _t34 = ax.text(_lon_c, _lat_c, 'Niño 3.4', transform=data_transform, rotation=_ang,
                       rotation_mode='anchor', color='white', fontsize=13, fontweight='bold',
                       ha='center', va='center', family=ctx.get('font_legenda', FONT_SANS),
                       zorder=9)
        _t34.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'),
                               path_effects.Normal()])
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
        fig.text(0.5, 0.136, f"{ctx['titulo_en']}{_titulo_sufixo} on {data_wapo}", color='white',
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
    if ctx.get('usar_atmosfera_estrelas', False):
        _cx = ctx.get('atm_cx', 0.50)
        _cy = ctx.get('atm_cy', 0.49)
        _r  = ctx.get('atm_r',  0.433)
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

        # Passo 3: estrelas no fundo escuro
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
    # Renderizado numa figura transparente separada e composto por último,
    # garantindo que a caixa cinza e o cbar fiquem acima de estrelas e halo.
    if guillaume:
        # Cache de última entrada por worker: o overlay só muda quando idx_dia muda.
        # Dict acumulativo em ctx explodia RAM (N_dias × 19 MB por worker).
        global _OV_CACHE_KEY, _OV_CACHE_VAL
        if _OV_CACHE_KEY != idx_dia:
            fig_ov = plt.figure(figsize=(8, 8), dpi=ctx['dpi'])
            fig_ov.patch.set_alpha(0)
            fig_ov.canvas.draw()  # inicializa renderer para cálculo do extent do texto
            _overlay_guillaume(fig_ov, ctx, cmap_legend, data_full)
            fig_ov.canvas.draw()
            _OV_CACHE_VAL = np.asarray(fig_ov.canvas.buffer_rgba()).copy().astype(np.float32)
            plt.close(fig_ov)
            _OV_CACHE_KEY = idx_dia
        ov = _OV_CACHE_VAL
        ov_a = ov[..., 3:4] / 255.0
        arr = np.clip(arr * (1.0 - ov_a) + ov[..., :3] * ov_a, 0.0, 255.0)

    return arr.astype(np.uint8)


def _render_one_frame(f: int) -> np.ndarray:
    """Wrapper picklavel p/ o process pool (usa o contexto global do worker)."""
    return _build_frame(f, _FRAME_CTX)


# ---------------------------------------------------------------------------
# Renderizacao de um clipe MP4 a partir de uma serie diaria de anomalia
# ---------------------------------------------------------------------------
def _render_clip(anom: xr.DataArray, ficha: dict, variavel_key: str,
                 output_dir: Path, fonte_label: str, script_id: str = 's38',
                 hgt_anom_serie: xr.DataArray | None = None,
                 mslp_serie: xr.DataArray | None = None,
                 olr_serie: xr.DataArray | None = None,
                 contour_serie: xr.DataArray | None = None) -> Path:
    frames_por_dia = int(getattr(settings, 'GLOBO_3D_FRAMES_POR_DIA', 4))
    fps = int(getattr(settings, 'GLOBO_3D_FPS', 20))
    coarsen = int(getattr(settings, 'GLOBO_3D_COARSEN', 1))
    # Janela da pentada movel (0/1 = diario) — capturada ANTES do coarsen, que descarta attrs.
    _pent_dias = int(anom.attrs.get('pentada_dias', 0))

    if coarsen and coarsen > 1:
        anom = anom.coarsen(lat=coarsen, lon=coarsen, boundary='trim').mean()

    lat = anom['lat'].values
    # Suavizacao gaussiana opcional por variavel (ex.: GLOBO_3D_SIGMA_OLR_ANOM = 1.5).
    # Aplicada antes do add_cyclic_point com mode='wrap' em lon para evitar seam.
    _sigma_var = float(settings.get(f'GLOBO_3D_SIGMA_{variavel_key.upper()}', 0.0))
    _anom_vals = anom.values
    if _sigma_var > 0:
        _anom_vals = np.stack([
            _gaussian_filter(_anom_vals[t], sigma=_sigma_var, mode=['reflect', 'wrap'])
            for t in range(_anom_vals.shape[0])
        ])
    vals_cyc, lon_cyc = add_cyclic_point(_anom_vals, coord=anom['lon'].values)
    dates = pd.DatetimeIndex(pd.to_datetime(anom['time'].values))
    n_dias = vals_cyc.shape[0]

    # Série de Z250 anomalia para isolinhas no campo absoluto (wind250_abs):
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
    _isol_hgt_flag = ('GLOBO_3D_ISOL_HGT_WIND250' if variavel_key == 'wind250_abs'
                      else 'GLOBO_3D_ISOL_HGT250_ABS')
    _isol_flag_on = bool(settings.get(_isol_hgt_flag, False))
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

    # MSLP isolinhas (só para variáveis com isolinha_mslp=True, ex.: tmp850_mslp)
    mslp_cyc: np.ndarray | None = None
    mslp_levels: np.ndarray | None = None
    if mslp_serie is not None and ficha.get('isolinha_mslp'):
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

    # Escala de cores fixa (estavel durante todo o clipe)
    vmax = ficha.get('vmax')
    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(vals_cyc), 98))
        vmax = max(10.0, round(vmax / 10.0) * 10.0)
    if ficha.get('vmin') is not None:
        vmin = float(ficha['vmin'])
    elif ficha.get('simetrico', True):
        vmin = -vmax
    else:
        vmin = float(np.nanmin(vals_cyc))
    # Paleta POR VARIAVEL: settings GLOBO_3D_PALETA_<VAR> sobrescreve a da ficha
    # (ex.: GLOBO_3D_PALETA_TMP850_ANOM). Sem override, usa a paleta da ficha.
    override = settings.get(f'GLOBO_3D_PALETA_{variavel_key.upper()}', None)
    paleta = override or ficha['cmap_colors']
    # Nº de bandas POR VARIAVEL: settings GLOBO_3D_NIVEIS_<VAR> > ficha['niveis'] >
    # GLOBO_3D_NIVEIS global. Mais niveis = shaded mais suave (degrade mais fino).
    # NIVEIS EXPLICITOS (ficha['levels']): grade possivelmente NAO-UNIFORME (ex.: tsm_anom usa
    # LST_SSTA_NEW_GREC do s11, com refino de ±0.2 no zero). Override de GLOBO_3D_NIVEIS_<VAR>
    # tem prioridade (volta p/ grade uniforme). Sem nenhum dos dois -> linspace uniforme.
    explicit_levels = ficha.get('levels')
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

    # Colormap e norm construidos 1x aqui (no pai) e herdados por todos os workers via CoW.
    # Antes eram recriados a cada frame (N_frames × from_list() = chamadas desnecessárias).
    paleta_opaca = [c[:7] if isinstance(c, str) and len(c) == 9 else c for c in paleta]
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
        # Niveis nao-uniformes (explicit_levels) -> BoundaryNorm garante a cor certa por banda no
        # caminho pcolormesh; o contourf (default) ja usa `levels` direto.
        norm_fn = (BoundaryNorm(levels, ncolors=256) if explicit_levels is not None
                   else plt.Normalize(vmin=vmin, vmax=vmax))

    total_frames = (n_dias - 1) * frames_por_dia + 1 if n_dias > 1 else max(frames_por_dia, 1)
    lons, lats = _camera_path(total_frames)
    vel_var = max(0.05, float(getattr(settings, 'GLOBO_3D_VELOCIDADE_VAR', 1.0)))

    # Resolucao de saida (px). dpi tal que 8in * dpi = px (figsize fixo 8).
    px = int(getattr(settings, 'GLOBO_3D_TAMANHO_PX', 1080))
    dpi = px / 8.0

    # Fontes: titulo e legenda (ex.: "Aptos Display" se o .ttf existir em Entrada/fonts).
    font_titulo = _resolve_family(getattr(settings, 'GLOBO_3D_FONTE_TITULO', ''), FONT_SERIF)
    font_legenda = _resolve_family(getattr(settings, 'GLOBO_3D_FONTE_LEGENDA', ''), FONT_SANS)

    # Workers: GLOBO_3D_WORKERS (0 = auto = min(nucleos, 8)).
    # Cap original de 4 foi removido: o OOM vinha do pickle de ctx via initargs (~150 MB × N),
    # resolvido com fork+CoW (_FRAME_CTX global herdado sem cópia). Agora cada worker
    # aloca apenas ~20 MB de estado local por frame; 8 workers ≈ 160 MB adicionais.
    workers = int(getattr(settings, 'GLOBO_3D_WORKERS', 0)) or min(os.cpu_count() or 1, 8)
    workers = max(1, min(workers, total_frames))

    # Estilo de layout: s39 -> 'guillaume' (caixa do nome + barra de gradiente); demais -> WaPo.
    estilo = 'guillaume' if script_id == 's39' else 'wapo'
    # Centro e raio do disco para atmosfera/estrelas — derivado do rect de cada estilo:
    #   guillaume rect [0.07, 0.06, 0.86, 0.86] -> cy=0.49, r=0.433
    #   wapo     rect [0.01, 0.10, 0.98, 0.83] -> cy=0.515, r=0.415, disc_top=0.930
    if estilo == 'guillaume':
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
    _centro_opaco = (_centro_paleta[:7] if isinstance(_centro_paleta, str) and len(_centro_paleta) > 7
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

    ctx = {
        'vals_cyc': vals_cyc.astype(np.float32),
        'lon_cyc': lon_cyc, 'lat': lat, 'levels': levels,
        'paleta': list(paleta),
        'cmap_plot': cmap_plot, 'cmap_legend': cmap_legend, 'norm_fn': norm_fn,
        'shade_px': int(settings.get('GLOBO_3D_SHADE_PX', 3600)),      # resolucao do raster do contourf
        'shade_regrid': int(settings.get('GLOBO_3D_SHADE_REGRID', 2048)),  # resolucao da reprojecao imshow
        'cor_fundo_globo': cor_fundo_globo,
        'lons': lons, 'lats': lats,
        'frames_por_dia': frames_por_dia, 'vel_var': vel_var, 'n_dias': n_dias,
        # Pentada movel: a data vira o INTERVALO da janela [d, d+dias-1] (ex.: 'July 20–24, 2026')
        # em vez de um dia unico — senao o rotulo engana (o campo e media de `dias` dias).
        'dates_en': [_fmt_data_pentada(d, _pent_dias, com_ano=False) for d in dates],
        'dates_full': [_fmt_data_pentada(d, _pent_dias, com_ano=True) for d in dates],  # s39
        'dates_wapo': [_fmt_data_pentada(d, _pent_dias, com_ano=True) for d in dates],  # s38
        'titulo_en': ficha.get('titulo_en', ficha['titulo']) + _olr_suf_en,
        'olr_overlay': olr_overlay,
        'fonte_label': fonte_label,
        'credito': str(getattr(settings, 'GLOBO_3D_CREDITO', 'Bruno Capucin')).upper(),
        'font_titulo': font_titulo, 'font_legenda': font_legenda,
        'usar_vinheta': bool(getattr(settings, 'GLOBO_3D_VINHETA', True)),
        'usar_atmosfera_estrelas': bool(settings.get('GLOBO_3D_ATMOSFERA_ESTRELAS', False)),
        'atm_cx': _atm_cx, 'atm_cy': _atm_cy, 'atm_r': _atm_r,
        'barra_h': 0.17 if estilo == 'wapo' else 0.0,
        'clim_abs_cyc': clim_abs_cyc,
        'hgt_abs_cyc': hgt_abs_cyc,
        'hgt_abs_levels': hgt_abs_levels,
        'hgt_anom_vals_cyc': hgt_anom_vals_cyc,
        'hgt_z250_abs_cyc': hgt_z250_abs_cyc,
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
        'box_nino34': bool(ficha.get('box_nino34', False)),
        'cor_continente': ficha.get('cor_continente'),
        'cor_fronteiras': ficha.get('cor_fronteiras'),
        'extend_contourf': ficha.get('extend_contourf', 'both'),
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
        'lw_coast':  float(settings.get('GLOBO_3D_COASTLINE_LW', 0.5)),
        'lw_border': float(settings.get('GLOBO_3D_BORDERS_LW',  0.35)),
        'lw_states': float(settings.get('GLOBO_3D_STATES_LW',   0.2)),
        'usar_pcolormesh': bool(settings.get('GLOBO_3D_PCOLORMESH', False)),
    }

    logger.info('{} dias -> {} frames | {} fps | {}px | {} workers | {}',
                n_dias, total_frames, fps, px, workers, fonte_label)

    output_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = output_dir / f'{script_id}_{variavel_key}.mp4'
    writer = imageio.get_writer(
        str(mp4_path), fps=fps, codec='libx264', quality=8, macro_block_size=8,
    )
    t0 = _time.time()

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
            for f in range(total_frames):
                writer.append_data(_build_frame(f, ctx))
                if (f + 1) % 20 == 0 or f == total_frames - 1:
                    logger.info('  frame {}/{}', f + 1, total_frames)
    finally:
        writer.close()
        _FRAME_CTX = None  # libera referência após renderização

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
        # Para wind250_abs: carrega z250_anom somente quando GLOBO_3D_ISOL_HGT_WIND250=true
        # (necessário para isolinhas fixas coloridas e whitesmoke — ambas controladas pela flag).
        hgt_anom_serie = None
        if item['var'] == 'wind250_abs' and bool(settings.get('GLOBO_3D_ISOL_HGT_WIND250', False)):
            logger.info('wind250_abs: carregando z250_anom para Z250 absoluto (isolinhas)')
            hgt_anom_serie = _build_var_series(VARIAVEIS['z250_anom'], item['model'], dt_ini, dt_fim)
        mslp_serie = None
        if ficha.get('isolinha_mslp'):
            try:
                mslp_serie = _build_mslp_series(dt_ini, dt_fim)
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
            contour_serie = _build_var_series(VARIAVEIS[_cs_var], item['model'], dt_ini, dt_fim)
            # Pentada movel do CONTORNO seguindo a ficha-PAI: campos como z250_anom nao tem pentada
            # propria, mas como isolinha sobre chi200/olr devem acompanhar a media movel do shaded.
            # So aplica se a serie ainda nao veio pentada'da (ex.: psi200_anom ja tem pentada propria).
            _cs_pent = int(ficha.get('contorno_serie_pentada', 0) or 0)
            if _cs_pent > 1 and 'pentada_dias' not in contour_serie.attrs:
                contour_serie = _pentada_movel_serie(contour_serie, _cs_pent, _cs_var)
        outputs.append(_render_clip(serie, ficha, item['var'], item['dir'], item['label'], script_id,
                                    hgt_anom_serie, mslp_serie, olr_serie, contour_serie))
    return outputs
