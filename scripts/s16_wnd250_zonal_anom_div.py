"""s16 - Anomalia de Vento Zonal 250 hPa + Vento Divergente 200 hPa (reanalise OU previsao).

Diagnostico (5 modos, 1 PNG por modo/area/janela):
    - ''/full : shaded anomalia de vento zonal 250 hPa + quiver do vento divergente anomalo 200 hPa
    - '_pos'  : shaded so anomalia positiva + quiver divergente
    - '_nodiv': shaded, sem quiver
    - '_mag'  : magnitude do vento 250 (jato) + divergente + (OLR<0 se disponivel) + isolinhas Z250
    - '_waf'  : shaded anom vento zonal 250 + anomalia de Z250 (contornos) + vetores WAF de Rossby 250
    - 'z250_anom' : (SO forecast) shaded anomalia de Z250 250 hPa + isolinhas da altura geopotencial media

Modos de execucao (setting MODE):
    - reanalysis : ERA5/GDAS no periodo [DATA_INICIAL, DATA_FINAL] — janelas moveis + media do total.
    - forecast   : roda para CADA modelo habilitado (RUN_GFS/RUN_GEFS/...) — janelas moveis + pentadas
                   FIXAS. Saida separada por modelo. Espelha a arquitetura do s34.

Em ambos os modos os campos sao calculados por JANELA (media movel deslizante) usando climatologia
DIARIA NCEP (clim_uv250_daily/clim_uv200_daily/clim_hgt250_daily + CPC OLR), nao mais a climatologia
PSL por MM-DD dos helpers plot_* (que so faziam um periodo unico).

Saida:
    Saida/s16_WND250_ZONAL_ANOM/
        REANALISE/<tipo>/<area>/wnd250_<tipo>_<area>_<label>.png
        FORECAST/<CONTINENTE>/<MODELO>/<tipo>/<area>/wnd250_<tipo>_<area>_<label>.png
    onde <tipo> ∈ {full, pos, nodiv, mag, waf, u850_anom, v850_anom, wind_zonal_psi_waf,
    olr_wnd850, olr_psi200_div, z250_anom}; <label> = janela (YYYYMMDD_YYYYMMDD),
    pentadaN ou media_total.

Criado em: 2026-06-05 (reanalise) | Modo previsao + janelas/pentadas: 2026-06-29
"""

# Bibliotecas padrão
import time
from datetime import datetime
from pathlib import Path

# Bibliotecas de terceiros
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
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
from app.common.dataset_utils import area_display_name
from app.common.logo_helper import proportional_logo_zoom, resolve_logo_path
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.src.uteis.chi200_intrasazonal import chi_from_wind
from app.src.uteis.clim_diaria_olr import clim_olr_daily, olr_obs_daily
from app.src.uteis.clim_diaria_uv200_ltm import (
    clim_hgt250_daily,
    clim_u850_daily,
    clim_uv200_daily,
    clim_uv250_daily,
    clim_v850_daily,
)
from app.src.uteis.forecast_daily import (
    daily_scalar_on_grid,
    daily_uv200_on_grid,
    lagged_ensemble_mean,
    resolve_forecast_lead_init,
)
from app.src.uteis.forecast_models import (
    DEFAULT_SYNOPTIC_HOURS,
    MIN_DIAS_PENTADA,
    MODEL_CONTINENT,
    PENTADA_DIAS,
    enabled_models,
    get_data_sources,
    pentad_windows,
    windows,
)
from app.src.uteis.plot_psi200 import (
    _compute_rotational_wind,
    _compute_vorticity,
    _solve_poisson_sphere,
)
from app.src.uteis.plot_rossby_waf import waf_from_means
from app.src.uteis.rossby_wave_source import rossby_wave_source

# Downloaders de previsao por modelo (mesmos do s34)
from app.src.uteis.downloaders_ai_nomads import (
    ensure_aigefs_fcst200_for_period,
    ensure_aigefs_hgt250_fcst_for_period,
    ensure_aigefs_uv250_fcst_for_period,
    ensure_aigefs_uv850_fcst_for_period,
    ensure_aigfs_fcst200_for_period,
    ensure_aigfs_hgt250_fcst_for_period,
    ensure_aigfs_uv250_fcst_for_period,
    ensure_aigfs_uv850_fcst_for_period,
)
from app.src.uteis.downloaders_aifs_ens import (
    ensure_aifs_ens_fcst200_for_period,
    ensure_aifs_ens_hgt250_fcst_for_period,
    ensure_aifs_ens_uv250_fcst_for_period,
    ensure_aifs_ens_uv850_fcst_for_period,
)
from app.src.uteis.downloaders_aifs_fcst200 import ensure_aifs_fcst200_for_period
from app.src.uteis.downloaders_aifs_hgt250 import ensure_aifs_hgt250_fcst_for_period
from app.src.uteis.downloaders_aifs_uv250 import ensure_aifs_uv250_fcst_for_period
from app.src.uteis.downloaders_aifs_uv850 import ensure_aifs_uv850_fcst_for_period
from app.src.uteis.downloaders_cfs_ensemble import (
    CFS_LEAD_DAYS,
    ensure_cfs_fcst200_uvz_for_period,
    ensure_cfs_olr_for_period,
    ensure_cfs_uv250_for_period,
    ensure_cfs_uv850_for_period,
)
from app.src.uteis.downloaders_ecmwf_ens import (
    ensure_ecmwf_ens_fcst200_for_period,
    ensure_ecmwf_ens_hgt250_fcst_for_period,
    ensure_ecmwf_ens_olr_fcst_for_period,
    ensure_ecmwf_ens_uv250_fcst_for_period,
    ensure_ecmwf_ens_uv850_fcst_for_period,
)
from app.src.uteis.downloaders_ecmwf_fcst200 import ensure_ecmwf_fcst200_for_period
from app.src.uteis.downloaders_ecmwf_hgt250 import ensure_ecmwf_hgt250_fcst_for_period
from app.src.uteis.downloaders_ecmwf_olr import ensure_ecmwf_olr_fcst_for_period
from app.src.uteis.downloaders_ecmwf_uv250 import ensure_ecmwf_uv250_fcst_for_period
from app.src.uteis.downloaders_ecmwf_uv850 import ensure_ecmwf_uv850_fcst_for_period
from app.src.uteis.downloaders_gdas_hgt250 import ensure_gdas_hgt250_for_period
from app.src.uteis.downloaders_gdas_uv200 import ensure_gdas_uv200_for_period
from app.src.uteis.downloaders_gdas_uv250 import ensure_gdas_uv250_for_period
from app.src.uteis.downloaders_gdas_uv850 import ensure_gdas_uv850_for_period
from app.src.uteis.downloaders_gefs_fcst200 import ensure_gefs_fcst200_for_period
from app.src.uteis.downloaders_gefs_hgt250 import ensure_gefs_hgt250_fcst_for_period
from app.src.uteis.downloaders_gefs_olr import ensure_gefs_olr_fcst_for_period
from app.src.uteis.downloaders_gefs_uv250 import ensure_gefs_uv250_fcst_for_period
from app.src.uteis.downloaders_gefs_uv850 import ensure_gefs_uv850_fcst_for_period
from app.src.uteis.downloaders_gfs_fcst200 import ensure_gfs_fcst200_for_period
from app.src.uteis.downloaders_gfs_hgt250 import ensure_gfs_hgt250_fcst_for_period
from app.src.uteis.downloaders_gfs_olr import ensure_gfs_olr_fcst_for_period
from app.src.uteis.downloaders_gfs_uv250 import ensure_gfs_uv250_fcst_for_period
from app.src.uteis.downloaders_gfs_uv850 import ensure_gfs_uv850_fcst_for_period
from app.src.uteis.downloaders_nomads_combo import (
    ensure_gefs_combo_for_period,
    ensure_gfs_combo_for_period,
)
from app.src.uteis.downloaders_wind200 import ensure_era5_uv200_for_period
from app.src.uteis.downloaders_wind250 import ensure_era5_uv250_for_period
from app.src.uteis.downloaders_wind850 import ensure_era5_uv850_for_period
from app.src.uteis.downloaders_z250_era5 import ensure_era5_z250_for_period

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
# Downloaders de previsao por modelo: nome -> (fcst200, uv250, hgt250, olr, uv850).
#   - fcst200: u/v 200 hPa (vento divergente anomalo)
#   - uv250  : u/v 250 hPa (anomalia zonal + magnitude do jato + escoamento basico do WAF)
#   - hgt250 : Z 250 hPa (isolinhas do 'mag' + anomalia de Z250 do 'waf')
#   - olr    : OLR (modo 'mag'); None nos modelos sem OLR (AIFS/AIGFS/AIGEFS) -> camada pulada.
#   - uv850  : u/v 850 hPa (anomalias de vento zonal/meridional 850 — tipos u850_anom/v850_anom)
# CFS nao tem hgt250 -> z250=None: os contornos do 'mag' e o modo 'waf' sao pulados nele.
# ---------------------------------------------------------------------------
_FCST_DOWNLOADERS = {
    'gfs': (ensure_gfs_fcst200_for_period, ensure_gfs_uv250_fcst_for_period, ensure_gfs_hgt250_fcst_for_period, ensure_gfs_olr_fcst_for_period, ensure_gfs_uv850_fcst_for_period),
    'gefs': (ensure_gefs_fcst200_for_period, ensure_gefs_uv250_fcst_for_period, ensure_gefs_hgt250_fcst_for_period, ensure_gefs_olr_fcst_for_period, ensure_gefs_uv850_fcst_for_period),
    'ecmwf': (ensure_ecmwf_fcst200_for_period, ensure_ecmwf_uv250_fcst_for_period, ensure_ecmwf_hgt250_fcst_for_period, ensure_ecmwf_olr_fcst_for_period, ensure_ecmwf_uv850_fcst_for_period),
    'ecmwf_ens': (ensure_ecmwf_ens_fcst200_for_period, ensure_ecmwf_ens_uv250_fcst_for_period, ensure_ecmwf_ens_hgt250_fcst_for_period, ensure_ecmwf_ens_olr_fcst_for_period, ensure_ecmwf_ens_uv850_fcst_for_period),
    'aifs': (ensure_aifs_fcst200_for_period, ensure_aifs_uv250_fcst_for_period, ensure_aifs_hgt250_fcst_for_period, None, ensure_aifs_uv850_fcst_for_period),
    'aifs_ens': (ensure_aifs_ens_fcst200_for_period, ensure_aifs_ens_uv250_fcst_for_period, ensure_aifs_ens_hgt250_fcst_for_period, None, ensure_aifs_ens_uv850_fcst_for_period),
    'aigfs': (ensure_aigfs_fcst200_for_period, ensure_aigfs_uv250_fcst_for_period, ensure_aigfs_hgt250_fcst_for_period, None, ensure_aigfs_uv850_fcst_for_period),
    'aigefs': (ensure_aigefs_fcst200_for_period, ensure_aigefs_uv250_fcst_for_period, ensure_aigefs_hgt250_fcst_for_period, None, ensure_aigefs_uv850_fcst_for_period),
    'cfs': (ensure_cfs_fcst200_uvz_for_period, ensure_cfs_uv250_for_period, None, ensure_cfs_olr_for_period, ensure_cfs_uv850_for_period),
}
# Pre-busca COMBINADA (NOMADS GFS/GEFS): 1 requisicao/passo grava todos os NetCDFs por variavel.
_FCST_COMBO = {
    'gfs': ensure_gfs_combo_for_period,
    'gefs': ensure_gefs_combo_for_period,
}

# Variaveis candidatas nos GRIBs (para daily_scalar_on_grid)
HGT_VARS = ('hgt', 'z', 'gh', 'geopotential')
OLR_VARS = ('olr', 'ulwrf', 'sulwrf')

# Geopotencial (m^2/s^2) -> altura (m): ERA5 entrega `z`; forecast/GDAS ja entregam altura.
G = 9.80665
_GEOP_THRESHOLD = 20000.0  # ~10400 m @250 hPa vs >5e4 m^2/s^2

# Subpastas por tipo de plotagem (a saida fica em <base>/<tipo>/<area>/...).
#   - olr_wnd850: anomalia de OLR (shaded) + anomalia do vento 850 hPa (linhas de corrente).
#   - olr_psi200_div: vento divergente 200 anomalo (so onde chi<0) + anomalia de OLR (shaded) +
#     anomalia de PSI 200 (linhas de corrente).
#   Esses dois sao SO no modo forecast (e so em modelos com OLR — AIFS/AIGFS/AIGEFS nao tem).
#   - z250_anom: anomalia de Z250 (shaded) + isolinhas da altura geopotencial MEDIA 250 hPa.
#     SO no modo forecast (somado aos campos existentes) e so em modelos com hgt250 (CFS nao tem).
_TIPOS = ('full', 'pos', 'nodiv', 'mag', 'waf', 'u850_anom', 'v850_anom', 'wind_zonal_psi_waf',
          'olr_wnd850', 'olr_psi200_div', 'z250_anom')

# Tipos que dependem de Z250 (anomalia Z250 + WAF de Rossby) — pulados quando o modelo nao tem
# hgt250 (ex.: CFS). 'wind_zonal_psi_waf' = anom vento zonal 250 (shaded) + streamlines da anomalia
# rotacional/PSI 250 + vetores WAF de Rossby 250.
_TIPOS_COM_Z250 = {'waf', 'wind_zonal_psi_waf'}

# Tipos que so existem no forecast e exigem OLR (pulados na reanalise e em modelos sem OLR).
_TIPOS_OLR_FORECAST = {'olr_wnd850', 'olr_psi200_div'}

# Tipo que so existe no forecast e exige Z250 (anomalia de Z250 shaded + isolinhas da Z250 media).
# Pulado na reanalise e em modelos sem hgt250 (ex.: CFS).
_TIPOS_Z250_FORECAST = {'z250_anom'}

# Streamlines da anomalia do vento rotacional (PSI) — modo 'wind_zonal_psi_waf'.
STREAMLINE_DEFAULTS = {'density': 2.0, 'linewidth': 0.5, 'arrowsize': 1.0, 'color': 'dimgray'}
STREAMLINE_POR_AREA = {
    'globo': {'density': 3}, 'psa': {'density': 3}, 'hemisferio_sul': {'density': 3},
    'tropico': {'density': 3}, 'america_sul': {'density': 2},
}

# Areas reduzidas no modo previsao (override via setting LST_AREAS_S16_FORECAST).
_FORECAST_AREAS = ['enso', 'mjo', 'pacifico_leste_america_sul', 'hemisferio_sul', 'globo', 'psa',
                   'america_sul', 'tropico']

# ---------------------------------------------------------------------------
# Constantes de plotagem
# ---------------------------------------------------------------------------
LEVELS = np.arange(-40, 42, 2)
TICKS = np.arange(-40, 44, 8)

LEVELS_POS = np.arange(2, 42, 2)
TICKS_POS = np.arange(10, 42, 10)

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

# Anomalia de OLR (W m-2) — campo olr_wnd850 (shaded diverging BrBG_r: verde=convecção, marrom=inibição).
OLR_ANOM_LEVELS = np.arange(-40, 44, 4)
OLR_ANOM_TICKS = np.arange(-40, 48, 8)

# Anomalia de vento 850 hPa (tipos u850_anom / v850_anom) — paleta de anomalia do projeto.
LEVELS_850 = np.arange(-10, 11, 1)
TICKS_850 = np.arange(-10, 11, 2)

# Anomalia de altura geopotencial 250 hPa (tipo z250_anom) — shaded ±200 m geopotenciais (20 bandas).
LEVELS_Z250_ANOM = np.arange(-200, 210, 20)
TICKS_Z250_ANOM = np.arange(-200, 240, 40)
# Isolinhas da altura geopotencial MEDIA 250 hPa (m) sobrepostas ao shaded — intervalo padrao.
Z250_MEAN_CONTOUR_INTERVAL = 120.0

# Vetores do vento ANOMALO 850 hPa na FAIXA TROPICAL (|lat| <= lat_band), em verde escuro —
# desenhados por cima do campo wind_zonal_psi_waf. `step` = densidade em longitude (grade 2.5°;
# a latitude nunca e subamostrada p/ garantir as poucas linhas da faixa fina), `scale` = comprimento.
WND850_TROPICO_QUIVER = {
    'step': 2, 'scale': 150, 'width': 0.0016, 'headwidth': 3.0, 'headlength': 4.0,
    'lat_band': 5.0, 'color': 'darkgreen',
}

# Quiver do WAF (modo 'waf') — espelha o s10: vetores normalizados pelo maximo, scale_units='xy'.
WAF_QUIVER_DEFAULTS = {
    'step': 2, 'width': 0.002, 'headwidth': 4.5, 'headlength': 6.0,
    'scale': None, 'scale_units': 'xy', 'min_amp_ratio': 0.05,
}
WAF_QUIVER_POR_AREA = {
    'psa': {'step': 2}, 'hemisferio_sul': {'step': 2}, 'hemisferio_norte': {'step': 2},
    'globo': {'step': 2},
    'america_sul': {'step': 1, 'width': 0.004, 'headwidth': 5.0, 'headlength': 7.0, 'scale': 0.12},
    'globo_3d': {'step': 2, 'width': 0.003, 'headwidth': 5.0, 'headlength': 7.0},
}

QUIVER_DEFAULTS = {
    'scale': 100,
    'width': 0.0013,
    'step': 1,
    'min_mag': 0.5,
    'headwidth': 3.2,
    'headlength': 4.2,
    'headaxislength': 3.8,
    'color': 'black',
}

# Vetores do vento divergente 200 hPa (modos full/pos/mag). `step` menor = MAIS denso. Como o s16
# agora opera na grade LTM de 2.5° (e nao mais 0.25°), step=1 = 1 vetor a cada 2,5° (densidade
# maxima da grade); so as areas globais/hemisfericas usam step=2 (a cada 5°) p/ nao saturar.
QUIVER_POR_AREA = {
    'brasil': {'step': 1, 'scale': 30, 'width': 0.004},
    'america_sul': {'step': 1, 'scale': 50, 'width': 0.003},
    'america_sul_zom_out': {'step': 1, 'scale': 80, 'width': 0.002},
    'argentina': {'step': 1, 'scale': 20, 'width': 0.004},
    'costa_brasil': {'step': 1, 'scale': 20, 'width': 0.003},
    'hemisferio_sul': {'step': 2, 'scale': 100, 'width': 0.0013},
    'psa': {'step': 2, 'scale': 95, 'width': 0.0013},
    'globo': {'step': 2, 'scale': 95, 'width': 0.0017},
    'tropico': {'step': 2, 'scale': 100, 'width': 0.0013},
    'enso': {'step': 1, 'scale': 100, 'width': 0.0013},
    'mjo': {'step': 1, 'scale': 80, 'width': 0.002},
    'atlantico_tropical': {'step': 1, 'scale': 50, 'width': 0.0018},
    'africa': {'step': 1, 'scale': 50, 'width': 0.003},
    'africa_monsoon': {'step': 1, 'scale': 50, 'width': 0.003},
    'china': {'step': 1, 'scale': 100, 'width': 0.0016},
    'estados_unidos': {'step': 1, 'scale': 50, 'width': 0.003},
    'estados_unidos_zoom': {'step': 1, 'scale': 50, 'width': 0.002},
    'tsa': {'step': 1, 'scale': 50, 'width': 0.002},
    'tna': {'step': 1, 'scale': 50, 'width': 0.002},
    'iod': {'step': 1, 'scale': 50, 'width': 0.002},
    'pdo': {'step': 1, 'scale': 50, 'width': 0.002},
    'sad': {'step': 1, 'scale': 60, 'width': 0.002},
    'amo': {'step': 1, 'scale': 60, 'width': 0.002},
    'MDR': {'step': 1, 'scale': 50, 'width': 0.002},
    'pacific_chile': {'step': 1, 'scale': 50, 'width': 0.002},
    'pacifico_leste_america_sul': {'step': 1, 'scale': 100, 'width': 0.002},
    'zona_zcit_atlantico': {'step': 1, 'scale': 60, 'width': 0.002},
    'globo_3d': {'step': 2, 'scale': 70, 'width': 0.002},
}


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


def _get_waf_quiver_config(area: str) -> dict:
    cfg = dict(WAF_QUIVER_DEFAULTS)
    if area in WAF_QUIVER_POR_AREA:
        cfg.update(WAF_QUIVER_POR_AREA[area])
    return cfg


def _get_streamline_config(area: str) -> dict:
    cfg = dict(STREAMLINE_DEFAULTS)
    if area in STREAMLINE_POR_AREA:
        cfg.update(STREAMLINE_POR_AREA[area])
    return cfg


def _psi_rotwind(u2d, v2d, lat, lon):
    """Streamfunction (m^2/s) e vento rotacional a partir de u/v 2D (reusa o solver do s04)."""
    zeta = _compute_vorticity(u2d, v2d, lat, lon)
    psi = _solve_poisson_sphere(zeta, lat, lon)
    u_rot, v_rot = _compute_rotational_wind(psi, lat, lon)
    return psi, u_rot, v_rot


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


def _draw_waf_quiver(ax, area, px_waf_cyc, py_waf_cyc, lon_waf_cyc, lat_waf, transform):
    """Vetores WAF de Rossby normalizados pelo maximo (igual ao s10), mascarando os fracos."""
    wcfg = _get_waf_quiver_config(area)
    wstep = int(wcfg['step'])
    lon_wq, lat_wq = lon_waf_cyc[::wstep], lat_waf[::wstep]
    px_wq = px_waf_cyc[::wstep, ::wstep]
    py_wq = py_waf_cyc[::wstep, ::wstep]
    amp = np.sqrt(px_wq**2 + py_wq**2)
    max_amp = np.nanmax(amp)
    if max_amp > 0:
        weak = amp < float(wcfg['min_amp_ratio']) * max_amp
        px_wq = np.where(weak, np.nan, px_wq / max_amp)
        py_wq = np.where(weak, np.nan, py_wq / max_amp)
    ax.quiver(
        lon_wq, lat_wq, px_wq, py_wq, transform=transform,
        scale=wcfg['scale'], scale_units=wcfg['scale_units'],
        width=float(wcfg['width']), headwidth=float(wcfg['headwidth']),
        headlength=float(wcfg['headlength']), zorder=200, color='black',
    )


def _draw_div_quiver(ax, lon_q, lat_q, u_q_mask, v_q_mask, qcfg):
    """Vetores do vento divergente 200 hPa anomalo (ja mascarados onde chi>=0)."""
    ax.quiver(
        lon_q, lat_q, u_q_mask, v_q_mask, transform=ccrs.PlateCarree(),
        color=qcfg['color'], pivot='mid', scale=float(qcfg['scale']), width=float(qcfg['width']),
        headwidth=float(qcfg['headwidth']), headlength=float(qcfg['headlength']),
        headaxislength=float(qcfg['headaxislength']), zorder=5,
    )


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


def _to_geop_height(arr: np.ndarray) -> np.ndarray:
    """Converte geopotencial (m^2/s^2) -> altura (m) POR DIA (a serie de reanalise mistura fontes)."""
    arr = np.asarray(arr, dtype='float64')
    if arr.ndim == 3:
        m = np.nanmean(arr, axis=(1, 2))
        factor = np.where(m > _GEOP_THRESHOLD, 1.0 / G, 1.0)
        return arr * factor[:, None, None]
    return arr / G if np.nanmean(arr) > _GEOP_THRESHOLD else arr


# ---------------------------------------------------------------------------
# Pipeline de dados (serie diaria na grade LTM 2.5°)
# ---------------------------------------------------------------------------
def _build_daily_series(mode, forecast_model, ini_dt, fim_dt, run_inits, lead_hours, dl, lat, lon, logger):
    """Monta as series diarias de u200/v200, u250/v250, z250, OLR e u850/v850 na grade LTM.

    Retorna (u200, v200, u250, v250, z250, olr, u850, v850, fcst_info) como DataArrays
    (time, lat, lon) lat ascendente (z250/olr/u850/v850 podem ser None). Em forecast faz lagged
    ensemble das rodadas; em reanalise mistura ERA5 + GDAS pela latencia."""
    hours = list(DEFAULT_SYNOPTIC_HOURS)
    fcst_info = None
    z250 = olr = u850 = v850 = None
    if mode == 'forecast':
        fcst200_fn, uv250_fn, hgt250_fn, olr_fn, uv850_fn = dl
        combo_fn = _FCST_COMBO.get(forecast_model)
        rodada = int(settings.get('RODADA', 0))
        u2_runs, v2_runs, u25_runs, v25_runs, z25_runs, olr_runs = [], [], [], [], [], []
        u85_runs, v85_runs = [], []
        for k, init_k in enumerate(run_inits):
            logger.info(f'  Rodada {k + 1}/{len(run_inits)}: init {init_k:%Y-%m-%d %H}Z')
            if combo_fn is not None:  # pre-busca combinada (best-effort) — acelera GFS/GEFS
                combo_fn(init=init_k, lead_hours=lead_hours, hours=hours)
            f200 = list(fcst200_fn(init=init_k, lead_hours=lead_hours, hours=hours))
            u2k, v2k = daily_uv200_on_grid(f200, ini_dt, fim_dt, lat, lon, logger)
            u2_runs.append(u2k)
            v2_runs.append(v2k)
            if uv250_fn is not None:
                f250 = list(uv250_fn(init=init_k, lead_hours=lead_hours, hours=hours))
                if f250:
                    u25k, v25k = daily_uv200_on_grid(f250, ini_dt, fim_dt, lat, lon, logger)
                    u25_runs.append(u25k)
                    v25_runs.append(v25k)
            if hgt250_fn is not None:
                fz = list(hgt250_fn(init=init_k, lead_hours=lead_hours, hours=hours))
                if fz:
                    z25_runs.append(daily_scalar_on_grid(fz, HGT_VARS, ini_dt, fim_dt, lat, lon, logger))
            if olr_fn is not None:
                fo = list(olr_fn(init=init_k, lead_hours=lead_hours, hours=hours))
                if fo:
                    olr_runs.append(daily_scalar_on_grid(fo, OLR_VARS, ini_dt, fim_dt, lat, lon, logger))
            if uv850_fn is not None:
                f850 = list(uv850_fn(init=init_k, lead_hours=lead_hours, hours=hours))
                if f850:
                    u85k, v85k = daily_uv200_on_grid(f850, ini_dt, fim_dt, lat, lon, logger)
                    u85_runs.append(u85k)
                    v85_runs.append(v85k)
        u200 = lagged_ensemble_mean(u2_runs)
        v200 = lagged_ensemble_mean(v2_runs)
        u250 = lagged_ensemble_mean(u25_runs) if u25_runs else None
        v250 = lagged_ensemble_mean(v25_runs) if v25_runs else None
        z250 = lagged_ensemble_mean(z25_runs) if z25_runs else None
        olr = lagged_ensemble_mean(olr_runs) if olr_runs else None
        u850 = lagged_ensemble_mean(u85_runs) if u85_runs else None
        v850 = lagged_ensemble_mean(v85_runs) if v85_runs else None
        model_tag = forecast_model.upper()
        if forecast_model == 'cfs':
            fcst_info = f'{model_tag} | pseudo-ensemble subsazonal (16 membros) de {run_inits[0]:%d/%m/%Y}'
        elif len(run_inits) == 1:
            fcst_info = f'{model_tag} | rodada {rodada:02d}Z de {run_inits[0]:%d/%m/%Y}'
        else:
            runs_str = ', '.join(r.strftime('%d/%m') for r in run_inits)
            fcst_info = (f'{model_tag} | ensemble {rodada:02d}Z, {len(run_inits)} rodadas '
                         f'({run_inits[-1]:%d/%m} a {run_inits[0]:%d/%m/%Y}): {runs_str}')
    else:
        era5_period, gdas_period = get_data_sources(ini_dt, fim_dt)
        f200, f250, fz, f850 = [], [], [], []
        if era5_period:
            f200 += list(ensure_era5_uv200_for_period(start=era5_period[0], end=era5_period[1], hours_utc=hours))
            f250 += list(ensure_era5_uv250_for_period(start=era5_period[0], end=era5_period[1], hours_utc=hours))
            fz += list(ensure_era5_z250_for_period(start=era5_period[0], end=era5_period[1], hours_utc=hours))
            f850 += list(ensure_era5_uv850_for_period(start=era5_period[0], end=era5_period[1], hours_utc=hours))
        if gdas_period:
            f200 += list(ensure_gdas_uv200_for_period(start=gdas_period[0], end=gdas_period[1]))
            f250 += list(ensure_gdas_uv250_for_period(start=gdas_period[0], end=gdas_period[1]))
            fz += list(ensure_gdas_hgt250_for_period(start=gdas_period[0], end=gdas_period[1]))
            f850 += list(ensure_gdas_uv850_for_period(start=gdas_period[0], end=gdas_period[1]))
        u200, v200 = daily_uv200_on_grid(f200, ini_dt, fim_dt, lat, lon, logger)
        u250, v250 = daily_uv200_on_grid(f250, ini_dt, fim_dt, lat, lon, logger)
        if fz:
            z250 = daily_scalar_on_grid(fz, HGT_VARS, ini_dt, fim_dt, lat, lon, logger)
        if f850:
            u850, v850 = daily_uv200_on_grid(f850, ini_dt, fim_dt, lat, lon, logger)
    return u200, v200, u250, v250, z250, olr, u850, v850, fcst_info


def _window_fields(sel, arrays, clims, lat, lon):
    """Campos da media da janela (`sel`) — DataArrays prontos para a plotagem (lat descendente).

    `arrays` = (u200v, v200v, u250v, v250v, z250v, olr_anom, u850v, v850v) — z250v/olr_anom/u850v/
    v850v podem ser None; todos numpy (n, lat, lon) na grade LTM (lat ascendente). `clims` =
    (uc200, vc200, uc250, vc250, zc250, uc850, vc850). Retorna dict com da_wnd/da_spd/da_uchi/
    da_vchi/da_chi/da_z250/waf/da_olr/da_u850/da_v850/psi."""
    u200v, v200v, u250v, v250v, z250v, olr_anom, u850v, v850v = arrays
    uc200, vc200, uc250, vc250, zc250, uc850, vc850 = clims

    u2, v2 = u200v[sel].mean(0), v200v[sel].mean(0)
    uc2, vc2 = uc200[sel].mean(0), vc200[sel].mean(0)
    u25, v25 = np.nanmean(u250v[sel], 0), np.nanmean(v250v[sel], 0)
    uc25, vc25 = uc250[sel].mean(0), vc250[sel].mean(0)

    def _std(arr):
        return _standardize_coords(xr.DataArray(arr, dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}))

    # Anomalia de vento zonal 250 hPa (shaded full/pos/nodiv)
    da_wnd = _std(u25 - uc25)
    # Magnitude do vento 250 hPa (jato; shaded 'mag')
    da_spd = _std(np.sqrt(u25 ** 2 + v25 ** 2))
    # Vento divergente ANOMALO 200 hPa + potencial de velocidade anomalo (quiver + mascara chi<0)
    _, uchi_p, vchi_p = rossby_wave_source(u2, v2, lat, lon)
    _, uchi_c, vchi_c = rossby_wave_source(uc2, vc2, lat, lon)
    da_uchi, da_vchi = _std(uchi_p - uchi_c), _std(vchi_p - vchi_c)
    da_chi = _std(chi_from_wind(u2, v2, lat, lon) - chi_from_wind(uc2, vc2, lat, lon))

    # Anomalia do vento ROTACIONAL (streamlines da anomalia de PSI). Mantida na grade nativa
    # (lat ascendente, lon 0..360) — o streamplot exige eixos crescentes.
    #   - psi   : PSI 250 (campo wind_zonal_psi_waf)
    #   - psi200: PSI 200 (campo olr_psi200_div)
    _, urot_p, vrot_p = _psi_rotwind(u25, v25, lat, lon)
    _, urot_c, vrot_c = _psi_rotwind(uc25, vc25, lat, lon)
    psi_rot = {'urot': urot_p - urot_c, 'vrot': vrot_p - vrot_c, 'lat': lat, 'lon': lon}
    _, urot200_p, vrot200_p = _psi_rotwind(u2, v2, lat, lon)
    _, urot200_c, vrot200_c = _psi_rotwind(uc2, vc2, lat, lon)
    psi200_rot = {'urot': urot200_p - urot200_c, 'vrot': vrot200_p - vrot200_c, 'lat': lat, 'lon': lon}

    out = {'da_wnd': da_wnd, 'da_spd': da_spd, 'da_uchi': da_uchi, 'da_vchi': da_vchi,
           'da_chi': da_chi, 'da_z250': None, 'da_z250_anom': None, 'waf': None, 'da_olr': None,
           'da_u850': None, 'da_v850': None, 'psi': psi_rot, 'psi200': psi200_rot, 'wnd850': None}

    # Anomalias de vento 850 hPa: zonal/meridional shaded (tipos u850_anom/v850_anom) + o vetor
    # (u,v) cru em grade ascendente para as streamlines do campo olr_wnd850.
    u85a = v85a = None
    if u850v is not None and uc850 is not None:
        u85 = np.nanmean(u850v[sel], 0)
        if not np.isnan(u85).all():
            u85a = u85 - uc850[sel].mean(0)
            out['da_u850'] = _std(u85a)
    if v850v is not None and vc850 is not None:
        v85 = np.nanmean(v850v[sel], 0)
        if not np.isnan(v85).all():
            v85a = v85 - vc850[sel].mean(0)
            out['da_v850'] = _std(v85a)
    if u85a is not None and v85a is not None:
        out['wnd850'] = {'u': u85a, 'v': v85a, 'lat': lat, 'lon': lon}

    if z250v is not None:
        z25 = np.nanmean(z250v[sel], 0)
        if not np.isnan(z25).all():
            zc25 = zc250[sel].mean(0)
            out['da_z250'] = _std(z25)              # altura geopot. media 250 hPa (m): isolinhas (mag, z250_anom)
            out['da_z250_anom'] = _std(z25 - zc25)  # anomalia de Z250 (m): shaded do tipo z250_anom
            # WAF de Rossby 250 hPa: anomalia de Z250 + fluxo (Takaya-Nakamura)
            hgt_mean_da = xr.DataArray(z25, dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon})
            hgt_clim_da = xr.DataArray(zc25, dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon})
            hgt_anom, px, py, lat_waf, lon_waf = waf_from_means(
                hgt_mean_da, hgt_clim_da, uc25, vc25, lat, lon, pressure=250.0)
            out['waf'] = {
                'hgt_anom': hgt_anom,
                'waf_x': xr.DataArray(px, dims=('lat_waf', 'lon_waf'),
                                      coords={'lat_waf': lat_waf, 'lon_waf': lon_waf}),
                'waf_y': xr.DataArray(py, dims=('lat_waf', 'lon_waf'),
                                      coords={'lat_waf': lat_waf, 'lon_waf': lon_waf}),
            }

    if olr_anom is not None:
        olr_win = olr_anom[sel]
        if not np.isnan(olr_win).all():
            out['da_olr'] = _std(np.nanmean(olr_win, 0))
    return out


# ---------------------------------------------------------------------------
# Plotagem por janela (5 modos x areas)
# ---------------------------------------------------------------------------
def _render_window(fields, tipos, lst_areas, info_plot, ini_dt, fim_dt, out_base, label, fcst_info,
                   input_dir, logger):
    """Plota os tipos de `tipos` para cada area a partir do dict de campos da janela.

    `tipos` = subconjunto de _TIPOS habilitado para este modo/modelo (ja exclui waf/wind_zonal_psi_waf
    sem Z250 e olr_wnd850 fora do forecast). `out_base` = pasta do modo (REANALISE ou
    FORECAST/<cont>/<modelo>); cada figura vai em <out_base>/<tipo>/<area>/wnd250_<tipo>_<area>_<label>.png."""
    # Cyclic point + standardize (igual ao pipeline antigo, agora por janela)
    da_wnd = fields['da_wnd']
    wnd_cyc, lon_wnd_cyc = _add_cyclic_2d(da_wnd)
    lat_wnd = da_wnd['lat'].values

    da_spd = fields['da_spd']
    spd_cyc, lon_spd_cyc = _add_cyclic_2d(da_spd)
    lat_spd = da_spd['lat'].values

    da_uchi, da_vchi, da_chi_scalar = fields['da_uchi'], fields['da_vchi'], fields['da_chi']
    uchi_cyc, vchi_cyc, lon_chi_cyc = _add_cyclic_uv(da_uchi, da_vchi)
    chi_scalar_cyc, _ = _add_cyclic_2d(da_chi_scalar)
    lat_chi = da_uchi['lat'].values

    da_z250 = fields['da_z250']
    if da_z250 is not None:
        z250_cyc, lon_z250_cyc = _add_cyclic_2d(da_z250)
        lat_z250 = da_z250['lat'].values

    da_z250_anom = fields['da_z250_anom']
    if da_z250_anom is not None:
        z250_anom_cyc, lon_z250_anom_cyc = _add_cyclic_2d(da_z250_anom)
        lat_z250_anom = da_z250_anom['lat'].values

    waf = fields['waf']
    if waf is not None:
        hgt_anom = waf['hgt_anom']
        lat_hgt_waf = hgt_anom['lat'].values
        lon_hgt = hgt_anom['lon'].values
        lon_hgt_shift = ((lon_hgt + 180) % 360) - 180
        sort_hgt = np.argsort(lon_hgt_shift)
        hgt_waf_cyc, lon_hgt_waf_cyc = add_cyclic_point(
            hgt_anom.isel(lon=sort_hgt).values, coord=lon_hgt_shift[sort_hgt])
        if np.all((hgt_waf_cyc >= -50) & (hgt_waf_cyc <= 50)):
            hgt_waf_levels = np.arange(-50, 53, 3)
        elif np.all((hgt_waf_cyc >= -100) & (hgt_waf_cyc <= 100)):
            hgt_waf_levels = np.arange(-100, 120, 20)
        else:
            hgt_waf_levels = np.arange(-200, 220, 20)
        waf_x, waf_y = waf['waf_x'], waf['waf_y']
        lat_waf = waf_x['lat_waf'].values
        lon_waf = waf_x['lon_waf'].values
        lon_waf_shift = ((lon_waf + 180) % 360) - 180
        sort_waf = np.argsort(lon_waf_shift)
        px_waf_cyc, lon_waf_cyc = add_cyclic_point(waf_x.values[:, sort_waf], coord=lon_waf_shift[sort_waf])
        py_waf_cyc, _ = add_cyclic_point(waf_y.values[:, sort_waf], coord=lon_waf_shift[sort_waf])

    da_olr = fields['da_olr']
    if da_olr is not None:
        if float(da_olr['lon'].min()) >= 0:
            da_olr = da_olr.assign_coords(lon=((da_olr['lon'].values + 180) % 360) - 180).sortby('lon')
        olr_cyc, lon_olr_cyc = add_cyclic_point(da_olr.values, coord=da_olr['lon'].values)
        lat_olr = da_olr['lat'].values
    else:
        olr_cyc = lon_olr_cyc = lat_olr = None

    da_u850 = fields['da_u850']
    if da_u850 is not None:
        u850_cyc, lon_u850_cyc = _add_cyclic_2d(da_u850)
        lat_u850 = da_u850['lat'].values
    da_v850 = fields['da_v850']
    if da_v850 is not None:
        v850_cyc, lon_v850_cyc = _add_cyclic_2d(da_v850)
        lat_v850 = da_v850['lat'].values

    # Anomalia do vento rotacional (PSI) — grade nativa lat ascendente p/ o streamplot.
    psi_rot = fields['psi']
    lat_rot = psi_rot['lat']
    urot_cyc, lon_rot_cyc = add_cyclic_point(psi_rot['urot'], coord=psi_rot['lon'])
    vrot_cyc, _ = add_cyclic_point(psi_rot['vrot'], coord=psi_rot['lon'])

    psi200_rot = fields['psi200']
    lat_rot200 = psi200_rot['lat']
    urot200_cyc, lon_rot200_cyc = add_cyclic_point(psi200_rot['urot'], coord=psi200_rot['lon'])
    vrot200_cyc, _ = add_cyclic_point(psi200_rot['vrot'], coord=psi200_rot['lon'])

    # Anomalia do vento 850 hPa (streamlines do campo olr_wnd850) — grade nativa lat ascendente.
    wnd850 = fields['wnd850']
    if wnd850 is not None:
        lat_w850 = wnd850['lat']
        u850a_cyc, lon_w850_cyc = add_cyclic_point(wnd850['u'], coord=wnd850['lon'])
        v850a_cyc, _ = add_cyclic_point(wnd850['v'], coord=wnd850['lon'])

    cmap_full = LinearSegmentedColormap.from_list('anom', settings.LST_ANOM_CORRETA)
    cmap_pos = LinearSegmentedColormap.from_list('anom_pos', CMAP_POS_COLORS)
    cmap_mag = LinearSegmentedColormap.from_list('wnd_speed', CMAP_MAG_COLORS)
    cmap_olr_neg = LinearSegmentedColormap.from_list('olr_neg', CMAP_OLR_NEG_COLORS)
    cmap_olr_anom = plt.get_cmap('BrBG_r')  # OLR anom: verde=convecção, marrom=inibição

    dt_ini_str = _to_str_date(ini_dt)
    dt_fim_str = _to_str_date(fim_dt)
    logo_path = resolve_logo_path(input_dir)

    for area in lst_areas:
        logger.info('  Area {} ({})', area_display_name(area), label)

        is_polar = info_plot[area].get('projection', '') == 'orthographic_south'
        central_lon = 0 if is_polar else info_plot[area]['central_longitude_mapa']
        data_transform = ccrs.PlateCarree(
            central_longitude=info_plot[area]['central_longitude_plot']
        )
        qcfg = _get_quiver_config(area)
        _AREAS_COM_CORTE_BORDA = {'globo', 'psa', 'hemisferio_sul'}

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
        chi_q = chi_scalar_cyc[::step, ::step]
        u_q_mask = np.ma.masked_where(chi_q >= 0, u_q_mask)
        v_q_mask = np.ma.masked_where(chi_q >= 0, v_q_mask)

        for mode in tipos:
            if mode in _TIPOS_COM_Z250 and waf is None:
                continue  # sem z250 (ex.: CFS) -> sem WAF (modos 'waf' e 'wind_zonal_psi_waf')
            if mode == 'u850_anom' and da_u850 is None:
                continue  # sem u850 -> sem mapa de anomalia de vento zonal 850
            if mode == 'v850_anom' and da_v850 is None:
                continue  # sem v850 -> sem mapa de anomalia de vento meridional 850
            if mode == 'olr_wnd850' and (olr_cyc is None or wnd850 is None):
                continue  # precisa de OLR (shaded) + vento 850 (streamlines)
            if mode == 'olr_psi200_div' and olr_cyc is None:
                continue  # precisa de OLR (shaded); PSI200 e divergente sao do nucleo 200
            if mode == 'z250_anom' and (da_z250_anom is None or da_z250 is None):
                continue  # precisa da anomalia de Z250 (shaded) + Z250 media (isolinhas)
            if mode == 'pos':
                use_levels, use_ticks, use_extend, use_cmap = LEVELS_POS, TICKS_POS, 'max', cmap_pos
                use_data, use_lon, use_lat = wnd_cyc, lon_wnd_cyc, lat_wnd
            elif mode == 'mag':
                use_levels, use_ticks, use_extend, use_cmap = LEVELS_MAG, TICKS_MAG, 'max', cmap_mag
                use_data, use_lon, use_lat = spd_cyc, lon_spd_cyc, lat_spd
            elif mode == 'u850_anom':
                use_levels, use_ticks, use_extend, use_cmap = LEVELS_850, TICKS_850, 'both', cmap_full
                use_data, use_lon, use_lat = u850_cyc, lon_u850_cyc, lat_u850
            elif mode == 'v850_anom':
                use_levels, use_ticks, use_extend, use_cmap = LEVELS_850, TICKS_850, 'both', cmap_full
                use_data, use_lon, use_lat = v850_cyc, lon_v850_cyc, lat_v850
            elif mode in ('olr_wnd850', 'olr_psi200_div'):
                use_levels, use_ticks, use_extend, use_cmap = OLR_ANOM_LEVELS, OLR_ANOM_TICKS, 'both', cmap_olr_anom
                use_data, use_lon, use_lat = olr_cyc, lon_olr_cyc, lat_olr
            elif mode == 'z250_anom':
                use_levels, use_ticks, use_extend, use_cmap = LEVELS_Z250_ANOM, TICKS_Z250_ANOM, 'both', cmap_full
                use_data, use_lon, use_lat = z250_anom_cyc, lon_z250_anom_cyc, lat_z250_anom
            else:
                use_levels, use_ticks, use_extend, use_cmap = LEVELS, TICKS, 'both', cmap_full
                use_data, use_lon, use_lat = wnd_cyc, lon_wnd_cyc, lat_wnd

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
                for x, y, label_n, color, stroke_color in [
                    (66.25, -13.64, 'Niño 1+2', 'red', 'black'),
                    (34.1, 8.45, 'Niño 3', 'blue', 'white'),
                    (8.6, -9.45, 'Niño 3.4', 'black', 'white'),
                    (-22.5, 8.45, 'Niño 4', 'm', 'black'),
                ]:
                    t = plt.text(x, y, label_n, fontsize=14, color=color, weight='bold', zorder=500)
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

            # Z250: isolinhas de altura geopotencial (apenas modo mag, se disponivel)
            if mode == 'mag' and da_z250 is not None:
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

            # Overlays vetoriais:
            #  - 'waf':  contornos da anomalia de Z250 (preto) + vetores WAF de Rossby
            #  - 'wind_zonal_psi_waf': streamlines da anomalia rotacional/PSI 250 + vetores WAF
            #  - full/pos/mag: vento divergente 200 hPa
            #  - 'nodiv'/u850_anom/v850_anom: nenhum
            if mode == 'waf':
                ax.contour(
                    lon_hgt_waf_cyc, lat_hgt_waf, hgt_waf_cyc,
                    levels=hgt_waf_levels, colors='black', linewidths=1.0,
                    transform=data_transform, zorder=110,
                )
                _draw_waf_quiver(ax, area, px_waf_cyc, py_waf_cyc, lon_waf_cyc, lat_waf, data_transform)
            elif mode == 'wind_zonal_psi_waf':
                slcfg = _get_streamline_config(area)
                lon_sl, u_sl, v_sl = _prepare_streamline_lonuv(lon_rot_cyc, urot_cyc, vrot_cyc, central_lon)
                try:
                    ax.streamplot(
                        lon_sl, lat_rot, u_sl, v_sl,
                        transform=ccrs.PlateCarree(central_longitude=central_lon),
                        density=float(slcfg['density']), linewidth=float(slcfg['linewidth']),
                        arrowsize=float(slcfg['arrowsize']), color=slcfg['color'], zorder=30,
                    )
                except Exception as exc:  # streamplot pode falhar em projecao polar — segue sem linhas
                    logger.warning('  streamplot rotacional falhou em {} ({}) — mapa sem linhas', area, exc)
                _draw_waf_quiver(ax, area, px_waf_cyc, py_waf_cyc, lon_waf_cyc, lat_waf, data_transform)
                # Vento anomalo 850 hPa (verde escuro) restrito a faixa tropical |lat| <= lat_band.
                # So a longitude e subamostrada (tstep); a latitude e mantida inteira para nao perder
                # as poucas linhas da faixa fina (ex.: ±5° = ~5 pontos na grade 2.5°).
                if da_u850 is not None and da_v850 is not None:
                    tcfg = WND850_TROPICO_QUIVER
                    tstep = int(tcfg['step'])
                    lat_t = lat_u850
                    lon_t = lon_u850_cyc[::tstep]
                    u_t = u850_cyc[:, ::tstep]
                    v_t = v850_cyc[:, ::tstep]
                    fora2d = np.broadcast_to((np.abs(lat_t) > float(tcfg['lat_band']))[:, None], u_t.shape)
                    u_tm = np.ma.masked_where(fora2d, u_t)
                    v_tm = np.ma.masked_where(fora2d, v_t)
                    ax.quiver(
                        lon_t, lat_t, u_tm, v_tm, transform=ccrs.PlateCarree(),
                        color=tcfg['color'], pivot='mid', scale=float(tcfg['scale']),
                        width=float(tcfg['width']), headwidth=float(tcfg['headwidth']),
                        headlength=float(tcfg['headlength']), zorder=210,
                    )
            elif mode == 'olr_wnd850':
                # Streamlines da anomalia do vento 850 hPa (campo cru, lat ascendente) sobre o OLR.
                slcfg = _get_streamline_config(area)
                lon_sl, u_sl, v_sl = _prepare_streamline_lonuv(lon_w850_cyc, u850a_cyc, v850a_cyc, central_lon)
                try:
                    ax.streamplot(
                        lon_sl, lat_w850, u_sl, v_sl,
                        transform=ccrs.PlateCarree(central_longitude=central_lon),
                        density=float(slcfg['density']), linewidth=float(slcfg['linewidth']),
                        arrowsize=float(slcfg['arrowsize']), color=slcfg['color'], zorder=30,
                    )
                except Exception as exc:  # streamplot pode falhar em projecao polar — segue sem linhas
                    logger.warning('  streamplot vento850 falhou em {} ({}) — mapa sem linhas', area, exc)
            elif mode == 'olr_psi200_div':
                # Streamlines da anomalia de PSI 200 + vento divergente 200 anomalo (so onde chi<0).
                slcfg = _get_streamline_config(area)
                lon_sl, u_sl, v_sl = _prepare_streamline_lonuv(lon_rot200_cyc, urot200_cyc, vrot200_cyc, central_lon)
                try:
                    ax.streamplot(
                        lon_sl, lat_rot200, u_sl, v_sl,
                        transform=ccrs.PlateCarree(central_longitude=central_lon),
                        density=float(slcfg['density']), linewidth=float(slcfg['linewidth']),
                        arrowsize=float(slcfg['arrowsize']), color=slcfg['color'], zorder=30,
                    )
                except Exception as exc:  # streamplot pode falhar em projecao polar — segue sem linhas
                    logger.warning('  streamplot psi200 falhou em {} ({}) — mapa sem linhas', area, exc)
                _draw_div_quiver(ax, lon_q, lat_q, u_q_mask, v_q_mask, qcfg)
            elif mode == 'z250_anom':
                # Isolinhas da altura geopotencial MEDIA 250 hPa (m) sobre a anomalia sombreada.
                zfloor = np.floor(np.nanmin(z250_cyc) / Z250_MEAN_CONTOUR_INTERVAL) * Z250_MEAN_CONTOUR_INTERVAL
                zceil = np.ceil(np.nanmax(z250_cyc) / Z250_MEAN_CONTOUR_INTERVAL) * Z250_MEAN_CONTOUR_INTERVAL
                mean_levels = np.arange(zfloor, zceil + Z250_MEAN_CONTOUR_INTERVAL, Z250_MEAN_CONTOUR_INTERVAL)
                cs_mean = ax.contour(
                    lon_z250_cyc, lat_z250, z250_cyc,
                    levels=mean_levels, colors='#333333', linewidths=1.0,
                    transform=data_transform, zorder=900,
                )
                txts_mean = ax.clabel(cs_mean, inline=True, fontsize=9, fmt='%d')
                _style_contour_labels(txts_mean)
            elif mode in ('full', 'pos', 'mag'):  # so estes recebem o vento divergente 200 hPa
                _draw_div_quiver(ax, lon_q, lat_q, u_q_mask, v_q_mask, qcfg)

            # Colorbar (unidade depende da variavel sombreada: OLR em W m⁻², Z250 em m, resto em m s⁻¹)
            if mode in ('olr_wnd850', 'olr_psi200_div'):
                cbar_label = 'W m⁻²'
            elif mode == 'z250_anom':
                cbar_label = 'm'
            else:
                cbar_label = 'm s⁻¹'
            if is_polar and area != 'globo_3d':
                cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.05, ticks=use_ticks)
                cbar.set_label(label=cbar_label, size=10)
                cbar.ax.tick_params(labelsize=10)
            elif area in {'america_sul', 'globo_3d'}:
                cax = make_axes_locatable(ax).append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
                cbar = plt.colorbar(im, cax=cax, pad=0.02, fraction=0.02375, extend=use_extend, ticks=use_ticks)
                cbar.set_label(label=cbar_label, size=18)
                cbar.ax.tick_params(labelsize=20)
            elif area in {'enso', 'tropico', 'MDR', 'hemisferio_sul', 'psa'}:
                cax = make_axes_locatable(ax).append_axes('bottom', size='6%', pad=0.50, axes_class=plt.Axes)
                cbar = plt.colorbar(
                    im, cax=cax, pad=0.02, fraction=0.02375,
                    location='bottom', extend=use_extend, orientation='horizontal', ticks=use_ticks,
                )
                cbar.set_label(label=cbar_label, size=18)
                cbar.ax.tick_params(labelsize=20)
            else:
                cax = make_axes_locatable(ax).append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
                cbar = plt.colorbar(im, cax=cax, pad=0.02, fraction=0.02375, extend=use_extend, ticks=use_ticks)
                cbar.set_label(label=cbar_label, size=18)
                cbar.ax.tick_params(labelsize=20)

            # Título — a 1a linha (descrição) é quebrada em DUAS para não estourar a largura.
            # Formato: <desc linha 1>\n<desc linha 2>\nDe <ini> a <fim>[\n<modelo/rodada>].
            fcst_line = f'\n{fcst_info}' if fcst_info else ''
            data_line = f'\nDe {dt_ini_str} a {dt_fim_str}{fcst_line}'
            if mode == 'mag':
                olr_txt = ' + Anom OLR<0' if olr_cyc is not None else ''
                z_txt = ' + Z250' if da_z250 is not None else ''
                titulo = (
                    f'Magnitude Vento 250hPa\n'
                    f'+ Vento Divergente 200hPa (chi<0){olr_txt}{z_txt}{data_line}'
                )
            elif mode == 'waf':
                titulo = (
                    f'Anomalia Vento Zonal 250hPa + Anomalia Z250\n'
                    f'+ WAF de Rossby 250hPa{data_line}'
                )
            elif mode == 'u850_anom':
                titulo = f'Anomalia Vento\nZonal 850hPa{data_line}'
            elif mode == 'v850_anom':
                titulo = f'Anomalia Vento\nMeridional 850hPa{data_line}'
            elif mode == 'wind_zonal_psi_waf':
                v850_txt = ' + Vento 850hPa trop. (verde)' if da_u850 is not None else ''
                titulo = (
                    f'Anomalia Vento Zonal 250hPa + Anomalia PSI (linhas de corrente)\n'
                    f'+ WAF de Rossby 250hPa{v850_txt}{data_line}'
                )
            elif mode == 'olr_wnd850':
                titulo = (
                    f'Anomalia de OLR\n+ Anomalia Vento 850hPa (linhas de corrente){data_line}'
                )
            elif mode == 'olr_psi200_div':
                titulo = (
                    f'Anomalia de OLR + Vento Divergente 200hPa (chi<0)\n'
                    f'+ Anomalia PSI200 (linhas de corrente){data_line}'
                )
            elif mode == 'z250_anom':
                titulo = (
                    f'Anomalia de Altura Geopotencial 250hPa (sombreado)\n'
                    f'+ Altura Geopotencial Média 250hPa (isolinhas){data_line}'
                )
            else:
                div_txt = '' if mode == 'nodiv' else ' + Vento Divergente 200hPa'
                titulo = f'Anomalia Vento\nZonal 250hPa{div_txt}{data_line}'
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

            out_path = out_base / mode / area / f'wnd250_{mode}_{area}_{label}.png'
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(out_path), dpi=fig.dpi, bbox_inches='tight')
            plt.close('all')


# ---------------------------------------------------------------------------
# Execucao de um modo/modelo
# ---------------------------------------------------------------------------
def _run_once(mode, forecast_model, logger):
    """Pipeline completo para UM modo (reanalise) ou UM modelo de previsao."""
    # Forecast usa um conjunto reduzido de areas (override via LST_AREAS_S16_FORECAST).
    if mode == 'forecast':
        lst_areas = list(settings.get('LST_AREAS_S16_FORECAST', _FORECAST_AREAS))
    else:
        lst_areas = _get_area_list()
    input_dir = Path(settings.DIR_INPUT)
    info_plot = settings['areas_plotagem']
    mov_avg_days = int(settings.get('MOV_AVG_DAYS', 5))

    # ---- Datas + horizonte ----
    run_inits = None
    lead_hours = 0
    dl = None
    if mode == 'forecast':
        dl = _FCST_DOWNLOADERS[forecast_model]
        if forecast_model != 'cfs':
            rodada = int(settings.get('RODADA', 0))
            if rodada not in (0, 6, 12, 18):
                raise ValueError(f'RODADA deve ser "00", "06", "12" ou "18" (UTC). Recebido: {rodada:02d}')
        run_inits, lead_hours = resolve_forecast_lead_init(
            forecast_model,
            rodada=int(settings.get('RODADA', 0)),
            num_rodada=int(settings.get('NUM_RODADA', 1)),
            forecast_init=settings.get('FORECAST_INIT', 'latest'),
            gefs_lead_days=int(settings.get('GEFS_FORECAST_LEAD_DAYS', settings.get('FORECAST_LEAD_DAYS', 35))),
            cfs_lead_days=CFS_LEAD_DAYS,
        )
        init0 = run_inits[0]
        # Descarta o dia do init (passo f000): a OLR acumulada sai degenerada no lead 0 e os campos
        # instantaneos perdem so o 1o dia. Comeca em init+1 — onde as pentadas tambem comecam.
        ini_dt = datetime(init0.year, init0.month, init0.day) + pd.Timedelta(days=1)
        ini_dt = datetime(ini_dt.year, ini_dt.month, ini_dt.day)
        fim_dt = init0 + pd.Timedelta(hours=lead_hours)
        fim_dt = datetime(fim_dt.year, fim_dt.month, fim_dt.day)
        model_tag = forecast_model.upper()
    else:
        ini_dt = datetime.fromisoformat(str(settings.DATA_INICIAL))
        fim_dt = datetime.fromisoformat(str(settings.DATA_FINAL))
        model_tag = 'REANALISE'
    ini_str, fim_str = ini_dt.date().isoformat(), fim_dt.date().isoformat()

    total_days = (fim_dt.date() - ini_dt.date()).days + 1
    if total_days < mov_avg_days:
        raise ValueError(
            f'Periodo de {total_days} dia(s) menor que a janela movel MOV_AVG_DAYS={mov_avg_days}. '
            f'Ajuste as datas (reanalise) ou o horizonte do modelo (forecast).')

    # ---- Pasta de saida (por modo/continente/modelo) ----
    base_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_WND250_ZONAL_ANOM'
    if mode == 'forecast':
        out_base = base_dir / 'FORECAST' / MODEL_CONTINENT.get(forecast_model, 'MODELOS_OUTROS') / model_tag
    else:
        out_base = base_dir / 'REANALISE'

    has_olr = (mode != 'forecast') or (dl[3] is not None)

    # ---- Janelas + pentadas + labels ----
    daily_dates = [d.date() for d in pd.date_range(ini_dt.date(), fim_dt.date(), freq='1D')]
    win_list = windows(daily_dates, mov_avg_days)
    win_labels = [f'{ws:%Y%m%d}_{we:%Y%m%d}' for ws, we in win_list]
    if mode == 'forecast':
        n_pentadas = max(1, (lead_hours // 24) // PENTADA_DIAS)
        run_date = run_inits[0].date()
        pent_list = pentad_windows(run_date, n_pentadas)
        pent_labels = [f'{ws:%Y%m%d}_{we:%Y%m%d}_pentada{k + 1}' for k, (ws, we) in enumerate(pent_list)]
    else:
        n_pentadas = 0
        pent_list, pent_labels = [], []
    media_label = f'{ini_dt:%Y%m%d}_{fim_dt:%Y%m%d}_media_total'

    # ---- Lista de arquivos esperados (cache) ----
    # Filtra tipos que nao se aplicam a este modo/modelo (cache nao espera arquivos inexistentes):
    #   - waf / wind_zonal_psi_waf: precisam de Z250 (fora quando o modelo nao tem hgt250, ex.: CFS);
    #   - olr_wnd850 / olr_psi200_div: SO no forecast e SO em modelos com OLR (AIFS/AIGFS/AIGEFS nao tem).
    tipos = []
    for t in _TIPOS:
        if t in _TIPOS_COM_Z250 and mode == 'forecast' and dl[2] is None:
            continue
        if t in _TIPOS_OLR_FORECAST and (mode != 'forecast' or dl[3] is None):
            continue
        if t in _TIPOS_Z250_FORECAST and (mode != 'forecast' or dl[2] is None):
            continue  # z250_anom: so forecast e so em modelos com hgt250 (CFS nao tem)
        tipos.append(t)
    output_files = []
    label_sets = list(win_labels)
    if mode == 'forecast':
        label_sets += pent_labels
    else:
        label_sets += [media_label]
    for lbl in label_sets:
        for t in tipos:
            for area in lst_areas:
                output_files.append(str(out_base / t / area / f'wnd250_{t}_{area}_{lbl}.png'))

    cache_id = f'{SCRIPT_ID}_{forecast_model}' if mode == 'forecast' else SCRIPT_ID
    cache_params = {
        'mode': mode,
        'forecast_model': forecast_model,
        'run_inits': [r.strftime('%Y%m%d%H') for r in run_inits] if run_inits else None,
        'DATA_INICIAL': ini_str,
        'DATA_FINAL': fim_str,
        'areas': lst_areas,
        'mov_avg_days': mov_avg_days,
        'n_pentadas': n_pentadas,
        'script_version': '3.5',  # + campo z250_anom (forecast): anom Z250 shaded + isolinhas Z250 media
    }
    if check_cache_valid(cache_id, cache_params, output_files):
        logger.info('🎯 CACHE VÁLIDO ({}): {} mapas já existem em {}', model_tag, len(output_files), out_base)
        return

    start_time = time.time()
    logger.info('📅 Período: {} a {} ({} dias) | janela móvel {}d | modo {}',
                ini_str, fim_str, total_days, mov_avg_days, model_tag)

    # ---- Etapa 1: serie diaria na grade LTM (2.5°) ----
    logger.info('Etapa 1: série diária u/v 200, u/v 250, Z250, OLR e u/v 850 ({})', model_tag)
    dates_probe = np.array([np.datetime64(fim_dt.date())])
    _, _, ltm_lat, ltm_lon = clim_uv200_daily(dates_probe)
    order = np.argsort(ltm_lat)
    lat, lon = ltm_lat[order], ltm_lon  # lat ascendente, lon 0..360

    u200_da, v200_da, u250_da, v250_da, z250_da, olr_da, u850_da, v850_da, fcst_info = (
        _build_daily_series(mode, forecast_model, ini_dt, fim_dt, run_inits, lead_hours, dl,
                            lat, lon, logger))

    if u250_da is None or v250_da is None:
        raise RuntimeError(
            f'{model_tag}: vento 250 hPa indisponivel — sem ele nao ha anomalia de vento zonal 250 '
            f'(produto principal do s16).')

    # Eixo de tempo comum do NUCLEO (u/v 200 + u/v 250). Z250/OLR ficam de fora e sao reindexados
    # com NaN: dia/variavel faltante afeta SO o mapa daquele campo, nunca o nucleo.
    common_t = np.intersect1d(u200_da['time'].values, v200_da['time'].values)
    common_t = np.intersect1d(common_t, np.intersect1d(u250_da['time'].values, v250_da['time'].values))
    if common_t.size < mov_avg_days:
        raise RuntimeError(
            f'{model_tag}: só {common_t.size} dia(s) com u/v 200+250 simultaneos (< janela '
            f'{mov_avg_days}d). Verifique o download.')
    u200_da = u200_da.sel(time=common_t)
    v200_da = v200_da.sel(time=common_t)
    u250_da = u250_da.sel(time=common_t)
    v250_da = v250_da.sel(time=common_t)
    if z250_da is not None:
        z250_da = z250_da.reindex(time=common_t)
    if olr_da is not None:
        olr_da = olr_da.reindex(time=common_t)
    if u850_da is not None:
        u850_da = u850_da.reindex(time=common_t)
    if v850_da is not None:
        v850_da = v850_da.reindex(time=common_t)

    dates = np.array([np.datetime64(pd.Timestamp(t).date()) for t in common_t])
    logger.info('Etapa 2: série diária com {} dias ({} a {})', len(dates), dates[0], dates[-1])

    # ---- Etapa 3: Climatologia diaria (NCEP u/v 200/250 + hgt 250 + CPC OLR) ----
    logger.info('Etapa 3: Climatologia diária (NCEP u/v200, u/v250, Z250 + CPC OLR)...')
    uc200, vc200, _, _ = clim_uv200_daily(dates)
    uc250, vc250, _, _ = clim_uv250_daily(dates)
    uc200, vc200 = uc200[:, order, :], vc200[:, order, :]
    uc250, vc250 = uc250[:, order, :], vc250[:, order, :]
    if z250_da is not None:
        zc250, _, _ = clim_hgt250_daily(dates)
        zc250 = zc250[:, order, :]
    else:
        zc250 = None
    if u850_da is not None or v850_da is not None:
        uc850, _, _ = clim_u850_daily(dates)
        vc850, _, _ = clim_v850_daily(dates)
        uc850, vc850 = uc850[:, order, :], vc850[:, order, :]
    else:
        uc850 = vc850 = None

    olr_anom = None
    if has_olr:
        if olr_da is None:  # reanalise: cai no observado (CPC mean), anomalia vs clim do mean
            olr_da = xr.DataArray(
                olr_obs_daily(dates, lat, lon), dims=('time', 'lat', 'lon'),
                coords={'time': common_t, 'lat': lat, 'lon': lon})
        olr_anom = olr_da.values - clim_olr_daily(dates, lat, lon)

    # numpy arrays (lat ascendente) para o fatiamento por janela
    u200v, v200v = u200_da.values, v200_da.values
    u250v, v250v = u250_da.values, v250_da.values
    z250v = _to_geop_height(z250_da.values) if z250_da is not None else None
    u850v = u850_da.values if u850_da is not None else None
    v850v = v850_da.values if v850_da is not None else None
    arrays = (u200v, v200v, u250v, v250v, z250v, olr_anom, u850v, v850v)
    clims = (uc200, vc200, uc250, vc250, zc250, uc850, vc850)

    n_maps_area = len(tipos)

    # ---- Etapa 4: MEDIA do periodo TOTAL — so reanalise ----
    if mode != 'forecast':
        sel_total = (dates >= np.datetime64(ini_dt.date())) & (dates <= np.datetime64(fim_dt.date()))
        if sel_total.any():
            logger.info('Etapa 4: média do período total [{} a {}]...', ini_str, fim_str)
            fields = _window_fields(sel_total, arrays, clims, lat, lon)
            _render_window(fields, tipos, lst_areas, info_plot, ini_dt, fim_dt, out_base, media_label,
                           fcst_info, input_dir, logger)

    # ---- Etapa 5: janelas moveis ----
    logger.info('Etapa 5: {} janela(s) móvel(eis) de {}d x {} áreas x {} mapas...',
                len(win_list), mov_avg_days, len(lst_areas), n_maps_area)
    for wi, (ws, we) in enumerate(win_list):
        sel = (dates >= np.datetime64(ws)) & (dates <= np.datetime64(we))
        span = (we - ws).days + 1
        if int(sel.sum()) < span:  # janela movel so plota se COMPLETA
            if int(sel.sum()) > 0:
                logger.warning('  Janela {} incompleta ({}/{} dias) — pulando', win_labels[wi], int(sel.sum()), span)
            continue
        fields = _window_fields(sel, arrays, clims, lat, lon)
        _render_window(fields, tipos, lst_areas, info_plot, datetime(ws.year, ws.month, ws.day),
                       datetime(we.year, we.month, we.day), out_base, win_labels[wi], fcst_info,
                       input_dir, logger)

    # ---- Etapa 6: pentadas fixas — so forecast ----
    if mode == 'forecast':
        logger.info('Etapa 6: {} pentada(s) fixa(s) (a partir do dia seguinte à rodada)...', len(pent_list))
        for pi, (ws, we) in enumerate(pent_list):
            sel = (dates >= np.datetime64(ws)) & (dates <= np.datetime64(we))
            n_dias = int(sel.sum())
            if n_dias < MIN_DIAS_PENTADA:
                logger.warning('  Pentada {} ({} a {}) com {}/{} dias (< {}) — pulando. Aumente '
                               'GEFS_FORECAST_LEAD_DAYS se for GEFS.',
                               pi + 1, ws, we, n_dias, PENTADA_DIAS, MIN_DIAS_PENTADA)
                continue
            if n_dias < PENTADA_DIAS:
                logger.warning('  Pentada {} ({} a {}) incompleta: {}/{} dias', pi + 1, ws, we, n_dias, PENTADA_DIAS)
            fields = _window_fields(sel, arrays, clims, lat, lon)
            _render_window(fields, tipos, lst_areas, info_plot, datetime(ws.year, ws.month, ws.day),
                           datetime(we.year, we.month, we.day), out_base, pent_labels[pi], fcst_info,
                           input_dir, logger)

    execution_time = time.time() - start_time
    save_cache_metadata(cache_id, cache_params, output_files, execution_time)
    logger.info('=' * 80)
    logger.info('✅ {} concluído ({})! {:.1f}s ({:.1f} min)', SCRIPT_ID.upper(), model_tag,
                execution_time, execution_time / 60)
    logger.info('📊 {} janela(s) + {} pentada(s) em: {}', len(win_list), len(pent_list), out_base)
    logger.info('=' * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Entry point — chamado pelo CLI sem argumentos.

    reanalysis: roda 1× (Saida/.../REANALISE/). forecast: roda para CADA modelo habilitado pelas
    flags RUN_GFS/RUN_GEFS/..., salvando em Saida/.../FORECAST/<CONTINENTE>/<MODELO>/."""
    logger = get_logger(SCRIPT_ID)
    logger.info('=' * 80)
    logger.info('📊 SCRIPT {}: {}', SCRIPT_ID.upper(), SCRIPT_DESC)
    logger.info('=' * 80)

    mode = str(settings.get('MODE', 'reanalysis')).strip().lower()
    if mode != 'forecast':
        _run_once('reanalysis', None, logger)
        return

    models = enabled_models()
    if not models:
        raise ValueError(
            'MODE=forecast, mas nenhum modelo habilitado. Defina ao menos um no settings: '
            'RUN_GFS = true e/ou RUN_GEFS = true.')
    logger.info('Modelos de previsão habilitados: {}', [m.upper() for m in models])
    for fm in models:
        logger.info('#' * 80)
        logger.info('### MODELO {}', fm.upper())
        logger.info('#' * 80)
        _run_once('forecast', fm, logger)


if __name__ == '__main__':
    main()
