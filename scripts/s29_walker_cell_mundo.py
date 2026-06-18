"""s29 - Célula de Walker: anomalia de movimento vertical (omega) 30°S–30°N.

Visualização 3D interativa com Plotly:
  - Piso: superfície global (oceano = SSTA OISSTv2, continentes = verde)
  - Volume: anomalia de omega (lon × nível de pressão) com efeito de colunas
    luminosas (go.Volume: isosuperfícies semi-transparentes empilhadas)
  - Linhas de costa como go.Scatter3d no piso

Convenção omega:  ω < 0  →  ascendente  (azul)
                  ω > 0  →  descendente (âmbar)

Saída:
  - PNG estático via kaleido
  - HTML interativo (sempre gerado)

Dados (híbrido ERA5/GDAS):
  - ERA5 CDS: vertical_velocity para datas > 7 dias
  - GDAS NOMADS: omega para últimos 7 dias
  - Climatologia PSL (NCEP R1): omega multi-nível
  - OISSTv2: anomalia de TSM para o piso

Criado em: 2026-06-06
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import cartopy.feature as cfeature
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
from scipy.ndimage import binary_dilation, gaussian_filter

from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.common.dataset_utils import arquivo_cobre_periodo
from app.common.download_helper import DownloadEngine, download_with_progress
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.src.uteis.clim_PSL_omega_multilevel import get_clim_omega_multilevel
from app.src.uteis.clim_PSL_wnd850 import get_clim_wnd850_paths
from app.src.uteis.downloaders_gdas_omega import ensure_gdas_omega_for_period
from app.src.uteis.downloaders_omega_era5 import ensure_era5_omega_for_period
from app.src.uteis.downloaders_wind200 import ensure_era5_uv200_for_period
from app.src.uteis.downloaders_wind850 import ensure_era5_uv850_for_period
from app.src.uteis.ssta_climatologia import clim_mean_array

# ---------------------------------------------------------------------------
# Identidade
# ---------------------------------------------------------------------------
SCRIPT_ID = Path(__file__).stem.split('_')[0]
SCRIPT_NAME = Path(__file__).stem
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_NAME

# ---------------------------------------------------------------------------
# Domínio e níveis
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = -30.0, 30.0   # cobertura de download ERA5/PSL
LAT_MEAN      = 10.0              # ±LAT_MEAN° para a média de omega (PSL usa 5°S-5°N)
LEVELS_HPA: List[int] = [100, 250, 300, 500, 850]
ERA5_LATENCY_DAYS = 7

# ---------------------------------------------------------------------------
# Colormap omega (Plotly: lista de [posição 0-1, cor])
# ---------------------------------------------------------------------------
OMEGA_VMAX = 0.10
OMEGA_THRESHOLD = 0.005  # limiar mínimo para isosuperfície (0.005 Pa/s ≈ sinal fraco)

PLOTLY_CMAP_OMEGA = [
    [0.00, '#0d47a1'],  # azul escuro (ascendente forte)
    [0.38, '#64b5f6'],  # azul claro
    [0.50, '#f5f5f5'],  # neutro
    [0.62, '#ffb74d'],  # âmbar claro
    [1.00, '#e65100'],  # âmbar escuro (descendente forte)
]

# ---------------------------------------------------------------------------
# Parâmetros de renderização
# ---------------------------------------------------------------------------
SIGMA_LEV   = 0.3    # suavização vertical mínima (1.5 destruía estrutura com só 5 níveis)
SIGMA_LON   = 3.0    # suavização horizontal leve (antes era 5.0)
N_LEV_FINE  = 25     # níveis interpolados (resolução vertical do volume)
N_LAT_THIN  = 13     # fatias latitudinais do volume
LAT_THIN    = 25.0   # extensão ±LAT_THIN° para o volume omega
FLOOR_Z     = 1050.0  # hPa (piso) — z=+1050; eixo log+reversed exibe embaixo
FIG_W       = 2400
FIG_H       = 1350

# Vetores de vento 850 hPa (go.Mesh3d quiver — polígonos planos)
WIND_VEC_STEP_LON  = 2.5    # espaçamento entre vetores (longitude, graus)
WIND_VEC_STEP_LAT  = 2.5    # espaçamento entre vetores (latitude, graus)
WIND_VEC_LAT_MAX   = 30.0   # extensão ±lat para os vetores
WIND_VEC_SCALE     = 0.8    # 1 m/s → 0.8° comprimento total da seta
WIND_VEC_SHAFT_W   = 0.20   # meia-largura da haste (graus) — controla espessura
WIND_VEC_HEAD_W    = 0.55   # meia-largura da cabeça (graus)
WIND_VEC_HEAD_FRAC = 0.38   # fração do comprimento total que é a cabeça
WIND_VEC_THRESHOLD = 0.5    # m/s mínimo para plotar um vetor
WIND_VEC_Z         = 950.0  # hPa — ligeiramente acima do piso (1050 hPa) p/ evitar z-fighting

# Vento divergente 200 hPa (go.Mesh3d quiver, branco com contorno preto)
DIV200_STEP_LON    = 2.5    # espaçamento entre vetores (longitude, graus)
DIV200_STEP_LAT    = 2.5    # espaçamento entre vetores (latitude, graus)
DIV200_LAT_MAX     = 30.0   # cobertura ±30°
DIV200_SCALE       = 0.8    # 1 m/s → 0.8° comprimento total da seta
DIV200_SHAFT_W     = 0.20   # meia-largura da haste branca (graus)
DIV200_HEAD_W      = 0.55   # meia-largura da cabeça branca (graus)
DIV200_HEAD_FRAC   = 0.38   # fração do comprimento que é a cabeça
DIV200_THRESHOLD   = 0.3    # m/s mínimo para plotar vetor divergente
DIV200_Z           = 200.0  # hPa — superfície de plotagem
DIV200_OUTLINE_EXP = 0.12   # expansão extra da meia-largura para o contorno preto (graus)

# OISSTv2 — SST absoluta (a anomalia e calculada subtraindo a climatologia diaria)
OISST_URL_TPL = (
    'https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.day.mean.{year}.nc'
)


# ---------------------------------------------------------------------------
# Seleção de fonte ERA5/GDAS
# ---------------------------------------------------------------------------
def _get_data_sources(
    dt_ini: datetime,
    dt_fim: datetime,
) -> Tuple[Optional[Tuple[datetime, datetime]], Optional[Tuple[datetime, datetime]]]:
    cutoff = (datetime.now() - timedelta(days=ERA5_LATENCY_DAYS)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    if dt_fim < cutoff:
        return (dt_ini, dt_fim), None
    if dt_ini >= cutoff:
        return None, (dt_ini, dt_fim)
    return (dt_ini, cutoff - timedelta(days=1)), (cutoff, dt_fim)


# ---------------------------------------------------------------------------
# Utilitários de grade
# ---------------------------------------------------------------------------
def _ensure_lon360(da: xr.DataArray) -> xr.DataArray:
    if 'lon' in da.coords and np.any(da['lon'].values < 0):
        da = da.assign_coords(lon=(da['lon'].values + 360) % 360).sortby('lon')
    return da


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


def _normalize_pressure_dim(da: xr.DataArray) -> xr.DataArray:
    for lname in ('isobaricInhPa', 'level'):
        if lname in da.dims:
            return da.rename({lname: 'pressure_level'})
    return da


def _ensure_time_coord(ds: xr.Dataset) -> xr.Dataset:
    if 'time' not in ds.dims and 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
    elif 'time' not in ds.coords and 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
    return ds


def _sort_dedup_time(ds: xr.Dataset) -> xr.Dataset:
    import pandas as pd

    ds = ds.sortby('time')
    t = pd.DatetimeIndex(pd.to_datetime(ds['time'].values))
    _, idx = np.unique(t.values, return_index=True)
    if len(idx) != ds.sizes.get('time', 0):
        ds = ds.isel(time=np.sort(idx))
    return ds


# ---------------------------------------------------------------------------
# SST (OISSTv2)
# ---------------------------------------------------------------------------
def _download_sst_anos(dados_dir: Path, ini_dt: datetime, fim_dt: datetime, logger) -> List[Path]:
    anos = list(range(ini_dt.year, fim_dt.year + 1))
    paths = []
    for year in anos:
        url = OISST_URL_TPL.format(year=year)
        sst_path = dados_dir / f'sst.day.mean.{year}.nc'
        t0 = np.datetime64(f'{year}-01-01')
        t1 = np.datetime64(f'{year}-12-31' if year < fim_dt.year else str(fim_dt.date()))
        if sst_path.exists() and arquivo_cobre_periodo(sst_path, t0, t1):
            logger.info('SST {} já existe — pulando', year)
        else:
            download_with_progress(
                url=url,
                output_path=str(sst_path),
                description=f'OISSTv2 anom {year}',
                max_retries=5,
                force=sst_path.exists(),
                engine=DownloadEngine.AUTO,
            )
        paths.append(sst_path)
    return paths


def _load_sst_mean(sst_paths: List[Path], ini_dt: datetime, fim_dt: datetime) -> xr.DataArray:
    datasets = [xr.open_dataset(str(p)) for p in sst_paths]
    ds = datasets[0] if len(datasets) == 1 else xr.concat(datasets, dim='time').sortby('time')
    da = ds['sst'].sel(time=slice(str(ini_dt.date()), str(fim_dt.date())))
    da = _rename_std_latlon(da)
    if np.any(da['lon'].values < 0):
        da = _ensure_lon360(da)
    da_mean = da.mean(dim='time', skipna=True)
    for ds_ in datasets:
        ds_.close()
    return da_mean


# ---------------------------------------------------------------------------
# Omega ERA5 / GDAS
# ---------------------------------------------------------------------------
def _load_era5_omega(files: List[Path], ini_dt: datetime, fim_dt: datetime) -> xr.DataArray:
    parts = []
    for fp in files:
        ds = xr.open_dataset(str(fp), engine='netcdf4')
        ds = _ensure_time_coord(ds)
        ds = _drop_expver(ds)
        ds = _rename_std_latlon(ds)
        ds = ds.sortby('lat')
        for vname in ('w', 'vertical_velocity', 'omega'):
            if vname in ds.data_vars:
                da = ds[vname]
                break
        else:
            da = next(iter(ds.data_vars.values()))
        da = _normalize_pressure_dim(da)
        da = _ensure_lon360(da)
        parts.append(da)
        ds.close()
    da_all = xr.concat(parts, dim='time').sortby('time')
    return da_all.sel(time=slice(np.datetime64(ini_dt.date()), np.datetime64(fim_dt.date())))


def _load_gdas_omega(files: List[Path], ini_dt: datetime, fim_dt: datetime) -> xr.DataArray:
    parts = []
    for fp in files:
        ds = xr.open_dataset(str(fp), engine='netcdf4')
        ds = _rename_std_latlon(ds)
        ds = ds.sortby('lat')
        for vname in ('omega', 'w', 'vvel', 'VVEL'):
            if vname in ds.data_vars:
                da = ds[vname]
                break
        else:
            da = next(iter(ds.data_vars.values()))
        da = _normalize_pressure_dim(da)
        da = _ensure_lon360(da)
        parts.append(da)
        ds.close()
    da_all = xr.concat(parts, dim='time').sortby('time')
    return da_all.sel(time=slice(np.datetime64(ini_dt.date()), np.datetime64(fim_dt.date())))


def _compute_omega_mean(
    era5_files: List[Path],
    gdas_files: List[Path],
    ini_dt: datetime,
    fim_dt: datetime,
) -> xr.DataArray:
    da_era5_mean, n_era5, era5_lons = None, 0, None
    if era5_files:
        da = _load_era5_omega(era5_files, ini_dt, fim_dt)
        if da.sizes['time']:
            n_era5 = da.sizes['time']
            era5_lons = da['lon'].values
            da_era5_mean = (
                da.sel(lat=slice(-LAT_MEAN, LAT_MEAN)).mean(dim=['time', 'lat'], skipna=True)
            )

    da_gdas_mean, n_gdas = None, 0
    if gdas_files:
        da = _load_gdas_omega(gdas_files, ini_dt, fim_dt)
        if da.sizes['time']:
            n_gdas = da.sizes['time']
            if era5_lons is not None:
                da = da.interp(lon=era5_lons, method='linear')
            da_gdas_mean = (
                da.sel(lat=slice(-LAT_MEAN, LAT_MEAN)).mean(dim=['time', 'lat'], skipna=True)
            )

    if da_era5_mean is not None and da_gdas_mean is not None:
        da_gdas_mean = da_gdas_mean.interp(
            lon=da_era5_mean['lon'],
            pressure_level=da_era5_mean['pressure_level'],
            method='linear',
        )
        total = n_era5 + n_gdas
        return (da_era5_mean * n_era5 + da_gdas_mean * n_gdas) / total
    return da_era5_mean if da_era5_mean is not None else da_gdas_mean


# ---------------------------------------------------------------------------
# Climatologia PSL omega
# ---------------------------------------------------------------------------
def _load_psl_omega_mean(psl_path: Path, era5_lons: np.ndarray) -> xr.DataArray:
    ds = xr.open_dataset(str(psl_path), engine='netcdf4')
    da = ds['omega']
    ds.close()
    da = _rename_std_latlon(da)
    da = da.sortby('lat')
    da = _ensure_lon360(da)
    return (
        da.sel(lat=slice(-LAT_MEAN, LAT_MEAN))
        .mean(dim='lat', skipna=True)
        .interp(lon=era5_lons, method='linear')
    )


# ---------------------------------------------------------------------------
# U/V 850 hPa — carregamento e anomalia
# ---------------------------------------------------------------------------
def _load_era5_uv850_mean(
    files: List[Path],
    ini_dt: datetime,
    fim_dt: datetime,
) -> Tuple[Optional[xr.DataArray], Optional[xr.DataArray]]:
    """Média temporal U/V 850 hPa ERA5 sobre o período → DataArrays 2D (lat, lon)."""
    u_parts, v_parts = [], []
    for fp in files:
        ds = xr.open_dataset(str(fp), engine='netcdf4')
        ds = _ensure_time_coord(ds)
        ds = _drop_expver(ds)
        ds = _rename_std_latlon(ds)
        ds = ds.sortby('lat')
        if 'pressure_level' in ds.dims and ds.sizes['pressure_level'] == 1:
            ds = ds.isel(pressure_level=0, drop=True)
        u_da, v_da = None, None
        for vn in ('u', 'uwnd', 'u10', 'u_component_of_wind'):
            if vn in ds.data_vars:
                u_da = _ensure_lon360(ds[vn])
                break
        for vn in ('v', 'vwnd', 'v10', 'v_component_of_wind'):
            if vn in ds.data_vars:
                v_da = _ensure_lon360(ds[vn])
                break
        if u_da is not None and v_da is not None:
            u_parts.append(u_da)
            v_parts.append(v_da)
        ds.close()

    if not u_parts:
        return None, None

    u_all = xr.concat(u_parts, dim='time').sortby('time')
    v_all = xr.concat(v_parts, dim='time').sortby('time')
    t0, t1 = np.datetime64(ini_dt.date()), np.datetime64(fim_dt.date())
    u_m = u_all.sel(time=slice(t0, t1)).mean(dim='time', skipna=True)
    v_m = v_all.sel(time=slice(t0, t1)).mean(dim='time', skipna=True)
    return u_m, v_m


def _load_psl_uv850(
    path_u: Path,
    path_v: Path,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Carrega climatologia PSL U/V 850mb → DataArrays 2D normalizados."""
    das = []
    for path, candidates in [
        (path_u, ('uwnd', 'u', 'uwnd850', 'U_wind')),
        (path_v, ('vwnd', 'v', 'vwnd850', 'V_wind')),
    ]:
        ds = xr.open_dataset(str(path), engine='netcdf4')
        da = None
        for vn in candidates:
            if vn in ds.data_vars:
                da = ds[vn]
                break
        if da is None:
            da = next(iter(ds.data_vars.values()))
        for dim in ('time', 'level'):
            if dim in da.dims:
                da = da.isel(**{dim: 0}, drop=True)
        da = _rename_std_latlon(da)
        da = _ensure_lon360(da)
        ds.close()
        das.append(da)
    return das[0], das[1]


def _build_uv850_cone_grid(
    u_era5: xr.DataArray,
    v_era5: xr.DataArray,
    u_psl: xr.DataArray,
    v_psl: xr.DataArray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sub-amostra anomalia U/V 850 em grade regular para go.Cone."""
    lon_vec = np.arange(0.0, 360.0, WIND_VEC_STEP_LON)
    lat_vec = np.arange(-WIND_VEC_LAT_MAX, WIND_VEC_LAT_MAX + 0.1, WIND_VEC_STEP_LAT)

    def _interp(da, lons, lats):
        return da.interp(lon=lons, lat=lats, method='linear').values

    u_anom = np.nan_to_num(_interp(u_era5, lon_vec, lat_vec) - _interp(u_psl, lon_vec, lat_vec), nan=0.0)
    v_anom = np.nan_to_num(_interp(v_era5, lon_vec, lat_vec) - _interp(v_psl, lon_vec, lat_vec), nan=0.0)
    lon_2d, lat_2d = np.meshgrid(lon_vec, lat_vec)
    return lon_2d, lat_2d, u_anom, v_anom


# ---------------------------------------------------------------------------
# Preparação dos dados Plotly
# ---------------------------------------------------------------------------
def _build_omega_volume(
    lons_era5: np.ndarray,
    levels_hpa: List[int],
    omega_anom: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cria o volume 3D de omega para go.Volume.

    Retorna (x, y, z, value) flatten:
      x = longitude (0–360)
      y = latitude (banda ±LAT_THIN° com taper gaussiano)
      z = nível de pressão (hPa)
    """
    levels_sorted = sorted(levels_hpa)

    # NaN → 0 antes do filtro para evitar propagação (regiões sem dado = anomalia zero)
    omega_clean = np.where(np.isnan(omega_anom), 0.0, omega_anom)

    # Suavização Gaussian
    omega_s = gaussian_filter(omega_clean, sigma=[SIGMA_LEV, SIGMA_LON])

    # Interpolação vertical
    lev_fine = np.linspace(levels_sorted[0], levels_sorted[-1], N_LEV_FINE)
    interp_fn = interp1d(levels_sorted, omega_s, axis=0, kind='linear', fill_value='extrapolate')
    omega_fine = np.clip(interp_fn(lev_fine), -OMEGA_VMAX * 1.05, OMEGA_VMAX * 1.05)

    # Subamostrar longitude para 1°
    lons_1deg = np.arange(0.0, 360.0, 1.0)
    omega_1deg = np.empty((N_LEV_FINE, len(lons_1deg)))
    for k in range(N_LEV_FINE):
        omega_1deg[k] = np.interp(lons_1deg, lons_era5, omega_fine[k], period=360)

    # Banda latitudinal com taper gaussiano (cria forma de "coluna" no eixo Y)
    lat_thin = np.linspace(-LAT_THIN, LAT_THIN, N_LAT_THIN)
    sigma_lat = LAT_THIN * 0.5
    lat_weights = np.exp(-(lat_thin**2) / (2 * sigma_lat**2))

    # 3D: (N_LEV_FINE, N_LAT_THIN, n_lon)
    omega_3d = omega_1deg[:, np.newaxis, :] * lat_weights[np.newaxis, :, np.newaxis]
    omega_3d = np.nan_to_num(omega_3d, nan=0.0)  # garantia extra antes do flatten

    # Usa pressão POSITIVA como z (mesmo padrão do script de referência).
    # O eixo z usa type='log' + autorange='reversed' para exibir 1000 embaixo e 100 em cima.
    # IMPORTANTE: lev_fine vai de 100 → 850 (crescente) para que z seja ascendente na grade.
    lev_g, lat_g, lon_g = np.meshgrid(lev_fine, lat_thin, lons_1deg, indexing='ij')

    val_flat = omega_3d.flatten()
    val_desc = np.maximum(val_flat, 0.0)   # descendente (ω > 0): âmbar
    val_asc  = np.maximum(-val_flat, 0.0)  # ascendente (|ω < 0|): azul

    return lon_g.flatten(), lat_g.flatten(), lev_g.flatten(), val_flat, val_desc, val_asc


def _build_land_mesh3d(z_level: float):
    """Preenche continentes com go.Mesh3d usando Delaunay com pontos interiores.

    Grade interna de 1.5° dentro de cada polígono garante triângulos ~5:1 aspect
    ratio (vs 300:1 da fan triangulation anterior) — sem slivers visíveis.
    Borda segue exatamente os polígonos Natural Earth 50m (sem escada raster).
    """
    import plotly.graph_objects as go
    import cartopy.io.shapereader as shpreader
    from matplotlib.path import Path as MplPath
    from scipy.spatial import Delaunay

    STEP = 1.5  # graus entre pontos interiores da grade

    shp = shpreader.natural_earth(resolution='50m', category='physical', name='land')
    geoms = list(shpreader.Reader(shp).geometries())

    vx: list = []
    vy: list = []
    vz: list = []
    ti: list = []
    tj: list = []
    tk: list = []

    for geom in geoms:
        parts = list(geom.geoms) if hasattr(geom, 'geoms') else [geom]
        for part in parts:
            if part.area < 0.05:
                continue
            try:
                # buffer(1.5) expande 1.5° além da costa para cobrir a faixa de NaN
                # que o OISST deixa próximo à costa (células sem dado válido que
                # aparecem como LAND_SENTINEL entre o dado de TSM e o continente).
                simple = part.buffer(1.5).simplify(0.3)
            except Exception:
                continue
            coords = np.array(simple.exterior.coords)[:-1]
            if len(coords) < 3:
                continue

            x_min, y_min, x_max, y_max = simple.bounds
            if x_max - x_min > 350:  # pula Antártida (polígono circumpolar)
                continue

            poly_path = MplPath(np.array(simple.exterior.coords))

            # Grade interna: pontos a STEP° dentro do polígono.
            # Previne slivers de alta razão de aspecto no Delaunay.
            x_int = np.arange(
                np.ceil(x_min / STEP) * STEP, x_max + 1e-9, STEP
            )
            y_int = np.arange(
                np.ceil(y_min / STEP) * STEP, y_max + 1e-9, STEP
            )
            if len(x_int) and len(y_int):
                xm, ym = np.meshgrid(x_int, y_int)
                cands = np.column_stack([xm.ravel(), ym.ravel()])
                interior = cands[poly_path.contains_points(cands)]
            else:
                interior = np.empty((0, 2))

            all_pts = np.vstack([coords, interior]) if len(interior) else coords

            try:
                tri = Delaunay(all_pts)
            except Exception:
                continue

            for simplex in tri.simplices:
                pts = all_pts[simplex]
                # Filtro antimeridiano — com pontos interiores a 1.5°,
                # qualquer triângulo legítimo tem span < 5° no máximo.
                lons_360 = pts[:, 0] % 360
                if lons_360.max() - lons_360.min() > 90:
                    continue
                # Centróide deve estar dentro do polígono
                cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
                if not poly_path.contains_point((cx, cy)):
                    continue
                off = len(vx)
                for lon, lat in pts:
                    vx.append(float(lon % 360))
                    vy.append(float(lat))
                    vz.append(float(z_level))
                ti.append(off)
                tj.append(off + 1)
                tk.append(off + 2)

    return go.Mesh3d(
        x=vx, y=vy, z=vz,
        i=ti, j=tj, k=tk,
        color='#f5f5f5',
        opacity=1.0,
        flatshading=True,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
        name='Continentes',
        hoverinfo='skip',
        showlegend=False,
    )


def _build_cartopy_land_mask(lon_2d: np.ndarray, lat_2d: np.ndarray) -> np.ndarray:
    """Máscara booleana de terra (True=terra) usando polígonos Natural Earth 50m.

    Usa a mesma resolução das linhas de costa/fronteiras do Scatter3d para que
    a borda do preenchimento LAND_SENTINEL coincida com as linhas pretas.
    """
    import cartopy.io.shapereader as shpreader
    from matplotlib.path import Path as MplPath

    shp = shpreader.natural_earth(resolution='50m', category='physical', name='land')
    geoms = list(shpreader.Reader(shp).geometries())

    lon_geo = np.where(lon_2d > 180, lon_2d - 360, lon_2d)
    pts = np.column_stack([lon_geo.ravel(), lat_2d.ravel()])

    mask = np.zeros(pts.shape[0], dtype=bool)
    for geom in geoms:
        parts = list(geom.geoms) if hasattr(geom, 'geoms') else [geom]
        for part in parts:
            coords = np.array(part.exterior.coords)
            mask |= MplPath(coords).contains_points(pts)

    return mask.reshape(lon_2d.shape)


def _build_floor_arrays(
    da_sst_mean: Optional[xr.DataArray],
    logger,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepara o piso: lon_2d, lat_2d, ocean_sst, land_mask, land_mask_bool."""
    lon_floor = np.arange(0.0, 360.25, 0.25)
    lat_floor = np.arange(-90.0, 90.25, 0.25)
    lon_2d, lat_2d = np.meshgrid(lon_floor, lat_floor)

    sst_2d = np.full((len(lat_floor), len(lon_floor)), np.nan)
    if da_sst_mean is not None:
        try:
            sst_2d = da_sst_mean.interp(
                lon=lon_floor, lat=lat_floor, method='linear'
            ).values
            sst_2d = np.where(np.abs(sst_2d) > 50.0, np.nan, sst_2d)
        except Exception as exc:
            logger.warning('Falha ao interpolar SST para o piso: {}', exc)

    logger.debug('  Construindo máscara de terra (Natural Earth 50m)...')
    land_mask_bool = _build_cartopy_land_mask(lon_2d, lat_2d)

    n_land = int(np.sum(land_mask_bool))
    logger.debug('  Máscara de terra: {} células de {} total', n_land, lon_2d.size)

    ocean_sst = np.where(land_mask_bool, np.nan, sst_2d)
    land_mask = np.where(land_mask_bool, 1.0, np.nan)

    return lon_2d, lat_2d, ocean_sst, land_mask, land_mask_bool


def _extract_lines_360(geoms_iter) -> Tuple[list, list]:
    """Extrai coordenadas de geometrias lineares em 0–360 para go.Scatter3d.

    Insere None onde pontos consecutivos saltam > 180° no espaço 0–360 (cruzamento
    do meridiano 0°: ex. -1°W=359° → +1°E=1° apareceria como linha de 358° cruzando
    todo o mapa sem esse corte).
    """
    lons_list: list = []
    lats_list: list = []
    for geom in geoms_iter:
        parts = list(geom.geoms) if hasattr(geom, 'geoms') else [geom]
        for part in parts:
            try:
                if hasattr(part, 'exterior'):
                    coords = np.array(part.exterior.coords)
                elif hasattr(part, 'coords'):
                    coords = np.array(part.coords)
                else:
                    continue
                lons_c = coords[:, 0] % 360
                lats_c = coords[:, 1]
                seg_lons: list = [float(lons_c[0])]
                seg_lats: list = [float(lats_c[0])]
                for k in range(1, len(lons_c)):
                    if abs(lons_c[k] - lons_c[k - 1]) > 180:
                        seg_lons.append(None)
                        seg_lats.append(None)
                    seg_lons.append(float(lons_c[k]))
                    seg_lats.append(float(lats_c[k]))
                lons_list.extend(seg_lons + [None])
                lats_list.extend(seg_lats + [None])
            except Exception:
                continue
    return lons_list, lats_list


def _extract_coastlines_360() -> Tuple[list, list]:
    """Linhas de costa 50m Natural Earth em 0–360."""
    return _extract_lines_360(cfeature.COASTLINE.with_scale('50m').geometries())


def _extract_borders_360() -> Tuple[list, list]:
    """Fronteiras de países 50m Natural Earth em 0–360."""
    feat = cfeature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', '50m')
    return _extract_lines_360(feat.geometries())


def _extract_brazil_states_360() -> Tuple[list, list]:
    """Limites estaduais do Brasil 10m Natural Earth em 0–360."""
    import cartopy.io.shapereader as shpreader
    shp = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_1_states_provinces_lines')
    reader = shpreader.Reader(shp)
    bra_geoms = [
        r.geometry for r in reader.records()
        if r.attributes.get('ADM0_A3') == 'BRA'
    ]
    return _extract_lines_360(bra_geoms)


def _load_era5_uv200_mean(
    files: List[Path],
    ini_dt: datetime,
    fim_dt: datetime,
) -> Tuple[Optional[xr.DataArray], Optional[xr.DataArray]]:
    """Média temporal U/V 200 hPa ERA5 → DataArrays 2D (lat, lon)."""
    u_parts, v_parts = [], []
    for fp in files:
        ds = xr.open_dataset(str(fp), engine='netcdf4')
        ds = _ensure_time_coord(ds)
        ds = _drop_expver(ds)
        ds = _rename_std_latlon(ds)
        ds = ds.sortby('lat')
        if 'pressure_level' in ds.dims and ds.sizes['pressure_level'] == 1:
            ds = ds.isel(pressure_level=0, drop=True)
        u_da, v_da = None, None
        for vn in ('u', 'uwnd', 'u_component_of_wind'):
            if vn in ds.data_vars:
                u_da = _ensure_lon360(ds[vn])
                break
        for vn in ('v', 'vwnd', 'v_component_of_wind'):
            if vn in ds.data_vars:
                v_da = _ensure_lon360(ds[vn])
                break
        if u_da is not None and v_da is not None:
            u_parts.append(u_da)
            v_parts.append(v_da)
        ds.close()
    if not u_parts:
        return None, None
    u_all = xr.concat(u_parts, dim='time').sortby('time')
    v_all = xr.concat(v_parts, dim='time').sortby('time')
    t0, t1 = np.datetime64(ini_dt.date()), np.datetime64(fim_dt.date())
    return (
        u_all.sel(time=slice(t0, t1)).mean(dim='time', skipna=True),
        v_all.sel(time=slice(t0, t1)).mean(dim='time', skipna=True),
    )


def _compute_divergent_wind(
    u_mean: xr.DataArray,
    v_mean: xr.DataArray,
    lat_max: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vento divergente via Poisson espectral (FFT, aprox. Cartesiana para trópicos).

    Resolve ∇²χ = ∇·V → u_div = ∂χ/∂x, v_div = ∂χ/∂y.
    Retorna (lat, lon, u_div, v_div) com u_div/v_div em m/s.
    """
    from scipy.fft import rfft2, irfft2, rfftfreq, fftfreq

    # Restringe à faixa tropical para o cálculo e plotagem
    lat_all = u_mean['lat'].values
    lon_all = u_mean['lon'].values
    mask_lat = (lat_all >= -lat_max) & (lat_all <= lat_max)
    lat = lat_all[mask_lat]

    u = u_mean.sel(lat=slice(-lat_max, lat_max)).values.astype(np.float64)
    v = v_mean.sel(lat=slice(-lat_max, lat_max)).values.astype(np.float64)

    a = 6371000.0
    lat0_r = np.deg2rad(0.0)                       # referência equatorial para aprox. Cartesiana
    dlon_r = np.deg2rad(np.mean(np.diff(lon_all)))
    dlat_r = np.deg2rad(np.mean(np.diff(lat)))
    dx = a * np.cos(lat0_r) * dlon_r               # espaçamento físico x (m)
    dy = a * dlat_r                                 # espaçamento físico y (m)

    ny, nx = u.shape

    # Divergência ∇·V = ∂u/∂x + ∂v/∂y (Cartesiana)
    du_dx = np.gradient(u, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)
    div = du_dx + dv_dy  # s⁻¹

    # Solver espectral Poisson: ∇²χ = D  →  χ_hat = -D_hat / k²
    kx = rfftfreq(nx, d=dx) * 2.0 * np.pi         # rad/m
    ky = fftfreq(ny,  d=dy) * 2.0 * np.pi
    kx_2d, ky_2d = np.meshgrid(kx, ky)
    k2 = kx_2d**2 + ky_2d**2
    k2[0, 0] = 1.0                                 # evita /0 no modo DC
    div_hat = rfft2(div)
    chi_hat = -div_hat / k2
    chi_hat[0, 0] = 0.0                            # média zero (gauge)
    chi = irfft2(chi_hat, s=(ny, nx))              # m²/s

    # Vento divergente = grad(χ)
    u_div = np.gradient(chi, dx, axis=1)           # m/s
    v_div = np.gradient(chi, dy, axis=0)           # m/s

    return lat, lon_all, u_div, v_div


def _build_div200_grid(
    lat: np.ndarray,
    lon: np.ndarray,
    u_div: np.ndarray,
    v_div: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sub-amostra vento divergente 200 hPa em grade regular para plotagem."""
    lon_vec = np.arange(0.0, 360.0, DIV200_STEP_LON)
    lat_vec = np.arange(-DIV200_LAT_MAX, DIV200_LAT_MAX + 0.1, DIV200_STEP_LAT)

    # Cria DataArrays para interpolação
    u_da = xr.DataArray(u_div, coords={'lat': lat, 'lon': lon}, dims=['lat', 'lon'])
    v_da = xr.DataArray(v_div, coords={'lat': lat, 'lon': lon}, dims=['lat', 'lon'])

    u_sub = np.nan_to_num(u_da.interp(lon=lon_vec, lat=lat_vec, method='linear').values, nan=0.0)
    v_sub = np.nan_to_num(v_da.interp(lon=lon_vec, lat=lat_vec, method='linear').values, nan=0.0)

    lon_2d, lat_2d = np.meshgrid(lon_vec, lat_vec)
    return lon_2d, lat_2d, u_sub, v_sub


def _build_quiver_mesh3d(
    lon_arr: np.ndarray,
    lat_arr: np.ndarray,
    u_unit: np.ndarray,
    v_unit: np.ndarray,
    spd_deg: np.ndarray,
    head_frac: float,
    shaft_hw: float,
    head_hw: float,
    z_val: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Geometria go.Mesh3d para vetores quiver como polígonos planos.

    Cada seta = retângulo (haste, 2 triângulos) + triângulo (cabeça).
    Retorna (x, y, z, i, j, k) prontos para go.Mesh3d.
    """
    n = len(lon_arr)
    vx = np.empty(7 * n)
    vy = np.empty(7 * n)
    vz = np.full(7 * n, z_val)
    ti = np.empty(3 * n, dtype=np.int32)
    tj = np.empty(3 * n, dtype=np.int32)
    tk = np.empty(3 * n, dtype=np.int32)

    for k in range(n):
        x0, y0   = lon_arr[k], lat_arr[k]
        ux, uv   = u_unit[k], v_unit[k]
        total    = spd_deg[k]
        shaft_l  = total * (1.0 - head_frac)

        # perpendicular (sentido horário → esquerda da direção)
        px, py = -uv, ux  # já é unitário porque (ux,uv) é vetor unitário

        sx = x0 + ux * shaft_l    # início da cabeça
        sy = y0 + uv * shaft_l
        tx = x0 + ux * total      # ponta
        ty = y0 + uv * total

        b = 7 * k
        # vértices 0-3: haste (retângulo)
        vx[b],   vy[b]   = x0 + px*shaft_hw,  y0 + py*shaft_hw   # cauda esq
        vx[b+1], vy[b+1] = x0 - px*shaft_hw,  y0 - py*shaft_hw   # cauda dir
        vx[b+2], vy[b+2] = sx - px*shaft_hw,  sy - py*shaft_hw   # junção dir
        vx[b+3], vy[b+3] = sx + px*shaft_hw,  sy + py*shaft_hw   # junção esq
        # vértices 4-6: cabeça (triângulo)
        vx[b+4], vy[b+4] = sx + px*head_hw,   sy + py*head_hw    # cabeça esq
        vx[b+5], vy[b+5] = sx - px*head_hw,   sy - py*head_hw    # cabeça dir
        vx[b+6], vy[b+6] = tx,                ty                  # ponta

        bt = 3 * k
        # haste: 2 triângulos
        ti[bt],   tj[bt],   tk[bt]   = b,   b+1, b+2
        ti[bt+1], tj[bt+1], tk[bt+1] = b,   b+2, b+3
        # cabeça: 1 triângulo
        ti[bt+2], tj[bt+2], tk[bt+2] = b+4, b+5, b+6

    return vx, vy, vz, ti, tj, tk


def _extended_colorscale(colors: list, cap_frac: float) -> list:
    """Colorscale com caps sólidos nas extremidades — efeito extend='both'.

    cap_frac: fração do range TOTAL estendido ocupada por cada cap.
    Exemplo: cap_frac=0.08 → 8% inferior = cor mínima sólida, 8% superior = cor máxima sólida.
    """
    n = len(colors)
    inner = 1.0 - 2 * cap_frac
    cs: list = [[0.0, colors[0]], [cap_frac, colors[0]]]
    for i, c in enumerate(colors):
        p = round(cap_frac + i * inner / max(n - 1, 1), 6)
        cs.append([p, c])
    if cs[-1][0] < 1.0 - 1e-6:
        cs.append([round(1.0 - cap_frac, 6), colors[-1]])
    cs.append([1.0, colors[-1]])
    return cs


# ---------------------------------------------------------------------------
# Renderização Plotly
# ---------------------------------------------------------------------------
def _windows_desktop() -> Optional[Path]:
    """Área de Trabalho do Windows no WSL: usa DESKTOP_DIR (settings) se definido; senão
    auto-detecta o usuário real em /mnt/c/Users (pula pastas de sistema; tenta 'Desktop' e
    'Área de Trabalho'). Antes o caminho era fixo em 'Pichau' e não funcionava em outra máquina."""
    cfg = str(settings.get('DESKTOP_DIR', '') or '').strip()
    if cfg:
        p = Path(cfg)
        return p if p.is_dir() else None
    users = Path('/mnt/c/Users')
    if not users.is_dir():
        return None
    skip = {'all users', 'default', 'default user', 'public',
            'todos os usuários', 'usuário padrão', 'desktop.ini'}
    for u in sorted(users.iterdir()):
        if not u.is_dir() or u.name.lower() in skip:
            continue
        for name in ('Desktop', 'Área de Trabalho', 'Area de Trabalho'):
            d = u / name
            if d.is_dir():
                return d
    return None


def _plot_walker_cell_plotly(
    lons: np.ndarray,
    levels_hpa: List[int],
    omega_anom: np.ndarray,
    da_sst_mean: Optional[xr.DataArray],
    ini_str: str,
    fim_str: str,
    era5_period: Optional[Tuple[datetime, datetime]],
    gdas_period: Optional[Tuple[datetime, datetime]],
    output_path: Path,
    logger,
    uv850_grid: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    div200_grid: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
) -> None:
    import plotly.graph_objects as go

    ini_fmt = datetime.fromisoformat(ini_str).strftime('%d/%m/%Y')
    fim_fmt = datetime.fromisoformat(fim_str).strftime('%d/%m/%Y')
    fonte_parts = (['ERA5'] if era5_period else []) + (['GDAS'] if gdas_period else [])
    fonte_str = ' + '.join(fonte_parts) if fonte_parts else 'ERA5'

    fig = go.Figure()

    # ── Piso ─────────────────────────────────────────────────────────────
    logger.debug('  Preparando piso (SST + continentes)...')
    lon_2d, lat_2d, ocean_sst, land_mask, land_mask_bool = _build_floor_arrays(da_sst_mean, logger)

    try:
        sst_colors = list(settings.LST_ANOM_CORRETA)
        n_c = len(sst_colors)
    except Exception:
        sst_colors = ['#004a83','#0084b8','#38b2e4','#b4eff9','#ffffff',
                      '#ffffa9','#ffb76c','#ff6c30','#e92a00','#bd1c00','#910e00']
        n_c = len(sst_colors)
    sst_cmin, sst_cmax = -5.0, 5.0

    # ── Superfície piso (oceano + terra sentinela) ────────────────────────
    # z=FLOOR_Z para todas as células: evita artefatos de borda (z=NaN cria
    # borda ciano/escalonada). Terra recebe LAND_SENTINEL → whitesmoke via
    # colorscale; o Mesh3d vetorial cobre essas células com bordas precisas.
    sst_range = sst_cmax - sst_cmin
    LAND_SENTINEL = sst_cmin - sst_range
    full_range = sst_cmax - LAND_SENTINEL
    sst_min_pos = sst_range / full_range  # = 0.5

    # Dilata 1 célula em torno de TODA a máscara NaN (terra + gelo/sem dado OISST).
    # Sem isso, a aresta WebGL entre uma célula whitesmoke e uma célula azul fria
    # (anomalia negativa adjacente) renderiza com contorno azul visível.
    full_nan_mask = land_mask_bool | np.isnan(ocean_sst)
    full_nan_mask = binary_dilation(full_nan_mask, iterations=1)
    combined = np.where(full_nan_mask, LAND_SENTINEL, ocean_sst)
    combined_cscale = [
        [0.0, '#f5f5f5'],
        [round(sst_min_pos - 1e-6, 7), '#f5f5f5'],
    ]
    for i, c in enumerate(sst_colors):
        pos = sst_min_pos + (1.0 - sst_min_pos) * i / (n_c - 1)
        combined_cscale.append([round(pos, 7), c])

    # Colorscale identico ao piso: mapeia sst_colors linearmente de 0→1
    # para que cmin=-5 / cmax=+5 do dummy bata exatamente com o combined_cscale
    _sst_cscale = [[round(i / max(n_c - 1, 1), 6), c] for i, c in enumerate(sst_colors)]

    # Colorbar SSTA: trace dummy invisível
    fig.add_trace(go.Scatter3d(
        x=[0.0], y=[0.0], z=[FLOOR_Z + 200],
        mode='markers',
        marker=dict(
            size=0.001,
            color=[0.0],
            colorscale=_sst_cscale,
            cmin=sst_cmin, cmax=sst_cmax,
            showscale=True,
            colorbar=dict(
                title=dict(text='SSTA (°C)', font=dict(color='white', size=11)),
                x=-0.02, len=0.45, y=0.25, thickness=14,
                tickfont=dict(color='white', size=10),
                tickvals=[-5, -2.5, 0, 2.5, 5],
                ticktext=['-5', '-2.5', '0', '+2.5', '+5'],
            ),
        ),
        hoverinfo='skip',
        showlegend=False,
        name='_sst_cbar',
    ))

    fig.add_trace(go.Surface(
        x=lon_2d, y=lat_2d,
        z=np.full_like(lon_2d, FLOOR_Z, dtype=float),
        surfacecolor=combined,
        colorscale=combined_cscale,
        cmin=LAND_SENTINEL, cmax=sst_cmax,
        showscale=False,
        name='SSTA Oceano',
        opacity=1.0,
        lighting=dict(ambient=0.9, diffuse=0.1),
        hovertemplate=(
            'Lon: %{x:.1f}°<br>Lat: %{y:.1f}°<br>'
            'SSTA: %{surfacecolor:.2f} °C<extra></extra>'
        ),
    ))

    # ── Continentes vetoriais (Mesh3d) ────────────────────────────────────
    # Delaunay com grade interna 1.5° → triângulos ~5:1 aspect ratio (sem slivers).
    # z=FLOOR_Z-10 (1040 hPa) garante renderização acima do ocean Surface (1050 hPa).
    logger.debug('  Construindo mesh3d dos continentes (Delaunay + grade interna)...')
    land_mesh = _build_land_mesh3d(FLOOR_Z - 10.0)
    n_tri = len(land_mesh.i) if land_mesh.i else 0
    logger.info('  Continentes mesh3d: {} triângulos gerados', n_tri)
    fig.add_trace(land_mesh)

    # ── Linhas de costa, fronteiras e estados ─────────────────────────────
    LINE_Z = FLOOR_Z - 20.0

    logger.debug('  Extraindo linhas de costa (50m)...')
    coast_lons, coast_lats = _extract_coastlines_360()
    coast_z = [LINE_Z if v is not None else None for v in coast_lons]
    fig.add_trace(go.Scatter3d(
        x=coast_lons, y=coast_lats, z=coast_z,
        mode='lines',
        line=dict(color='rgba(0,0,0,1.0)', width=6),
        name='Costa',
        hoverinfo='skip',
    ))

    logger.debug('  Extraindo fronteiras de países (50m)...')
    bord_lons, bord_lats = _extract_borders_360()
    bord_z = [LINE_Z if v is not None else None for v in bord_lons]
    fig.add_trace(go.Scatter3d(
        x=bord_lons, y=bord_lats, z=bord_z,
        mode='lines',
        line=dict(color='rgba(0,0,0,1.0)', width=4),
        name='Fronteiras',
        hoverinfo='skip',
    ))

    logger.debug('  Extraindo estados do Brasil (10m)...')
    bra_lons, bra_lats = _extract_brazil_states_360()
    bra_z = [LINE_Z if v is not None else None for v in bra_lons]
    fig.add_trace(go.Scatter3d(
        x=bra_lons, y=bra_lats, z=bra_z,
        mode='lines',
        line=dict(color='rgba(0,0,0,1.0)', width=3),
        name='Estados BR',
        hoverinfo='skip',
    ))

    # ── Isosuperfícies omega (dois traces: descend=âmbar, ascend=azul) ──────
    # go.Isosurface (mesmo traço do script de referência) — mais robusto que
    # go.Volume em grades esparsas. z = pressão positiva (hPa); o eixo z usa
    # type='log' + autorange='reversed' para exibir 1000 hPa embaixo.
    logger.debug('  Construindo isosuperfícies omega 3D...')
    x_vol, y_vol, z_vol, val_flat_omega, val_desc, val_asc = _build_omega_volume(lons, levels_hpa, omega_anom)

    pos_max = float(np.nanmax(val_desc))
    neg_max = float(np.nanmax(val_asc))
    logger.debug(
        '  Omega iso: max_desc={:.4f} Pa/s, max_asc={:.4f} Pa/s', pos_max, neg_max
    )

    omega_range = max(pos_max, neg_max, OMEGA_THRESHOLD * 2)

    # Colorbar omega: trace dummy invisível com escala divergente (-range a +range)
    fig.add_trace(go.Scatter3d(
        x=[0.0], y=[0.0], z=[FLOOR_Z + 200],
        mode='markers',
        marker=dict(
            size=0.001,
            color=[0.0],
            colorscale=_extended_colorscale(
                ['#0d47a1', '#64b5f6', '#f5f5f5', '#ffb74d', '#e65100'],
                cap_frac=0.08,   # 8% do range total = cap sólido nas extremidades
            ),
            cmin=-omega_range / (1.0 - 2 * 0.08),  # range estendido para acomodar caps
            cmax=omega_range / (1.0 - 2 * 0.08),
            showscale=True,
            colorbar=dict(
                title=dict(text='ω anom. (Pa/s)', font=dict(color='white', size=11)),
                x=1.02, len=0.55, y=0.28, thickness=14,
                tickfont=dict(color='white', size=10),
                tickvals=[-omega_range, -omega_range / 2, 0, omega_range / 2, omega_range],
                ticktext=[
                    f'{-omega_range:.3f}', f'{-omega_range/2:.3f}', '0',
                    f'+{omega_range/2:.3f}', f'+{omega_range:.3f}',
                ],
            ),
        ),
        hoverinfo='skip',
        showlegend=False,
        name='_omega_cbar',
    ))

    # Descendente (ω > 0) → âmbar
    if pos_max > OMEGA_THRESHOLD:
        fig.add_trace(go.Isosurface(
            x=x_vol, y=y_vol, z=z_vol,
            value=val_desc,
            isomin=OMEGA_THRESHOLD,
            isomax=pos_max,
            opacity=0.45,
            surface_count=15,
            colorscale=[[0.0, '#ffcc80'], [0.5, '#ffb300'], [1.0, '#e65100']],
            showscale=False,
            name='ω descend. (âmbar)',
            caps=dict(x_show=False, y_show=False, z_show=False),
            hoverinfo='skip',
        ))
        logger.info('  Isosuperfície descendente adicionada (max={:.4f} Pa/s)', pos_max)
    else:
        logger.warning('  Sem anomalias de descendência acima do limiar.')

    # Ascendente (ω < 0) → azul
    if neg_max > OMEGA_THRESHOLD:
        fig.add_trace(go.Isosurface(
            x=x_vol, y=y_vol, z=z_vol,
            value=val_asc,
            isomin=OMEGA_THRESHOLD,
            isomax=neg_max,
            opacity=0.45,
            surface_count=15,
            colorscale=[[0.0, '#90caf9'], [0.5, '#42a5f5'], [1.0, '#0d47a1']],
            showscale=False,
            name='ω ascend. (azul)',
            caps=dict(x_show=False, y_show=False, z_show=False),
            hoverinfo='skip',
        ))
        logger.info('  Isosuperfície ascendente adicionada (max={:.4f} Pa/s)', neg_max)
    else:
        logger.warning('  Sem anomalias de ascendência acima do limiar.')

    # ── Vetores de vento 850 hPa (go.Mesh3d quiver) ──────────────────────
    # go.Mesh3d: polígono plano (haste retangular + cabeça triangular) por vetor.
    # Evita limitação WebGL de line width (~2px) do go.Scatter3d.
    # z=WIND_VEC_Z (950 hPa) ≠ FLOOR_Z (1050 hPa) → sem z-fighting.
    if uv850_grid is not None:
        logger.debug('  Adicionando vetores de vento U/V 850 hPa (Mesh3d quiver)...')
        lon_2d_w, lat_2d_w, u_anom, v_anom = uv850_grid
        lon_f = lon_2d_w.flatten()
        lat_f = lat_2d_w.flatten()
        u_f   = u_anom.flatten() * WIND_VEC_SCALE
        v_f   = v_anom.flatten() * WIND_VEC_SCALE
        spd   = np.sqrt(u_f**2 + v_f**2)
        mask  = (spd >= WIND_VEC_THRESHOLD * WIND_VEC_SCALE) & np.isfinite(spd)

        if mask.any():
            lx, ly   = lon_f[mask], lat_f[mask]
            spd_m    = np.where(spd[mask] > 0, spd[mask], 1e-6)
            cu_n     = u_f[mask] / spd_m
            cv_n     = v_f[mask] / spd_m

            qx, qy, qz, qi, qj, qk = _build_quiver_mesh3d(
                lon_arr=lx, lat_arr=ly,
                u_unit=cu_n, v_unit=cv_n,
                spd_deg=spd_m,
                head_frac=WIND_VEC_HEAD_FRAC,
                shaft_hw=WIND_VEC_SHAFT_W,
                head_hw=WIND_VEC_HEAD_W,
                z_val=WIND_VEC_Z,
            )
            fig.add_trace(go.Mesh3d(
                x=qx, y=qy, z=qz,
                i=qi, j=qj, k=qk,
                color='black',
                opacity=1.0,
                flatshading=True,
                lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0),
                hoverinfo='skip',
                name='U/V 850 anom',
            ))
            logger.info(
                '  Vetores U/V 850 (Mesh3d): {} setas, {} triângulos (z={} hPa)',
                int(mask.sum()), 3 * int(mask.sum()), WIND_VEC_Z,
            )
        else:
            logger.warning('  Nenhum vetor U/V 850 acima do limiar.')

    # ── Vento divergente 200 hPa (Mesh3d quiver, branco + contorno preto) ──
    # Dois layers go.Mesh3d: (1) preto ligeiramente maior em z=DIV200_Z+0.5
    # (2) branco normal em z=DIV200_Z — efeito "negrito preto envolta".
    if div200_grid is not None:
        logger.debug('  Adicionando vento divergente 200 hPa...')
        lon_2d_d, lat_2d_d, u_div, v_div = div200_grid
        lon_fd = lon_2d_d.flatten()
        lat_fd = lat_2d_d.flatten()
        u_fd   = u_div.flatten() * DIV200_SCALE
        v_fd   = v_div.flatten() * DIV200_SCALE
        spd_d  = np.sqrt(u_fd**2 + v_fd**2)
        mask_d = (spd_d >= DIV200_THRESHOLD * DIV200_SCALE) & np.isfinite(spd_d)

        if mask_d.any():
            lxd, lyd = lon_fd[mask_d], lat_fd[mask_d]
            spd_md   = np.where(spd_d[mask_d] > 0, spd_d[mask_d], 1e-6)
            cu_d     = u_fd[mask_d] / spd_md
            cv_d     = v_fd[mask_d] / spd_md

            # Camada 1: contorno preto (shaft_hw e head_hw maiores, z ligeiramente abaixo)
            exp = DIV200_OUTLINE_EXP
            qx_b, qy_b, qz_b, qi_b, qj_b, qk_b = _build_quiver_mesh3d(
                lon_arr=lxd, lat_arr=lyd,
                u_unit=cu_d, v_unit=cv_d,
                spd_deg=spd_md,
                head_frac=DIV200_HEAD_FRAC,
                shaft_hw=DIV200_SHAFT_W + exp,
                head_hw=DIV200_HEAD_W + exp,
                z_val=DIV200_Z + 0.5,   # 0.5 hPa abaixo → depth buffer o coloca atrás do branco
            )
            fig.add_trace(go.Mesh3d(
                x=qx_b, y=qy_b, z=qz_b,
                i=qi_b, j=qj_b, k=qk_b,
                color='black',
                opacity=1.0,
                flatshading=True,
                lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0),
                hoverinfo='skip',
                name='Vdiv 200 contorno',
            ))

            # Camada 2: preenchimento branco (tamanho normal, z=DIV200_Z)
            qx_w, qy_w, qz_w, qi_w, qj_w, qk_w = _build_quiver_mesh3d(
                lon_arr=lxd, lat_arr=lyd,
                u_unit=cu_d, v_unit=cv_d,
                spd_deg=spd_md,
                head_frac=DIV200_HEAD_FRAC,
                shaft_hw=DIV200_SHAFT_W,
                head_hw=DIV200_HEAD_W,
                z_val=DIV200_Z,
            )
            fig.add_trace(go.Mesh3d(
                x=qx_w, y=qy_w, z=qz_w,
                i=qi_w, j=qj_w, k=qk_w,
                color='white',
                opacity=1.0,
                flatshading=True,
                lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0),
                hoverinfo='skip',
                name='Vdiv 200 hPa',
            ))
            logger.info(
                '  Vento div 200 hPa: {} vetores (z={} hPa, ±{}°)',
                int(mask_d.sum()), DIV200_Z, DIV200_LAT_MAX,
            )
        else:
            logger.warning('  Nenhum vetor divergente 200 hPa acima do limiar.')

    # ── Layout ────────────────────────────────────────────────────────────
    bg = '#0d1117'
    fig.update_layout(
        title=dict(
            text=(
                f'Célula de Walker — Anomalia de ω (Pa/s) 30°S–30°N<br>'
                f'<sup>{ini_fmt} a {fim_fmt}  |  {fonte_str}  |  '
                f'Anomalia relativa à Climatologia PSL (NCEP R1)</sup>'
            ),
            font=dict(color='white', size=15),
            x=0.5,
            xanchor='center',
        ),
        paper_bgcolor=bg,
        scene=dict(
            bgcolor=bg,
            xaxis=dict(
                title='Longitude',
                range=[0, 360],
                tickvals=[0, 60, 120, 180, 240, 300, 360],
                ticktext=['0°', '60°E', '120°E', '180°', '120°W', '60°W', '0°'],
                color='white',
                gridcolor='rgba(255,255,255,0.12)',
                zerolinecolor='rgba(255,255,255,0.2)',
            ),
            yaxis=dict(
                title='Latitude',
                range=[-90, 90],
                tickvals=[-90, -60, -30, 0, 30, 60, 90],
                color='white',
                gridcolor='rgba(255,255,255,0.12)',
                zerolinecolor='rgba(255,255,255,0.2)',
            ),
            zaxis=dict(
                title='Pressão (hPa)',
                # range em log10: [log10(piso+margem), log10(topo)] com piso > topo
                # → eixo invertido sem depender de autorange (que quebra com go.Cone/Scatter3d)
                type='log',
                range=[np.log10(FLOOR_Z * 1.02), np.log10(85)],
                tickvals=[100, 200, 300, 500, 700, 850, 1000],
                ticktext=['100', '200', '300', '500', '700', '850', '1000'],
                color='white',
                gridcolor='rgba(255,255,255,0.12)',
                zerolinecolor='rgba(255,255,255,0.2)',
            ),
            aspectratio=dict(x=2.0, y=1.0, z=0.55),
            aspectmode='manual',
            camera=dict(
                eye=dict(x=0.0, y=-2.6, z=1.4),
                center=dict(x=0.0, y=0.18, z=-0.15),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        width=FIG_W,
        height=FIG_H,
        margin=dict(l=0, r=0, t=80, b=0),
    )

    # ── Salvar HTML (sempre) ──────────────────────────────────────────────
    html_path = output_path.with_suffix('.html')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_path), include_plotlyjs='cdn')
    logger.info('HTML interativo: {}', html_path)

    # ── Salvar PNG via kaleido ────────────────────────────────────────────
    try:
        import kaleido  # noqa: F401

        fig.write_image(str(output_path), width=FIG_W, height=FIG_H, scale=2)
        logger.info('PNG salvo: {}', output_path)
    except Exception as exc:
        logger.warning(
            'PNG não gerado ({}). Abra o HTML no browser: {}', exc, html_path
        )

    # ── Cópia automática para a Área de Trabalho do Windows (WSL) ────────
    desktop = _windows_desktop()
    if desktop is not None:
        dest = desktop / 'walker_cell_s29.html'
        import shutil
        shutil.copy2(str(html_path), str(dest))
        logger.info('HTML copiado para a Área de Trabalho: {}', dest)
    else:
        logger.info('Área de Trabalho do Windows não encontrada (defina DESKTOP_DIR no settings '
                    'se quiser a cópia automática). HTML disponível em: {}', html_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print('>>> S29 VERSAO 5.22 RODANDO <<<')
    logger = get_logger(SCRIPT_ID)

    logger.info('=' * 80)
    logger.info('SCRIPT {}: {}', SCRIPT_ID.upper(), SCRIPT_DESC)
    logger.info('=' * 80)

    ini_str = str(settings.DATA_INICIAL)
    fim_str = str(settings.DATA_FINAL)
    ini_dt = datetime.fromisoformat(ini_str)
    fim_dt = datetime.fromisoformat(fim_str)

    output_dir = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_WALKER_CELL_MUNDO'
    png_name = f'walker_cell_omega_anom_{ini_str}_to_{fim_str}.png'
    output_files = [str(output_dir / png_name)]

    cache_params = {
        'DATA_INICIAL': ini_str,
        'DATA_FINAL': fim_str,
        'lat_min': LAT_MIN,
        'lat_max': LAT_MAX,
        'levels_hpa': sorted(LEVELS_HPA),
        'anom_source': 'sst.day.mean - ltm.1991-2020',
        'script_version': '5.50',
    }

    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('CACHE VÁLIDO! Pulando execução.')
        return

    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    dados_dir = Path(settings.DIR_DADOS)
    dados_dir.mkdir(parents=True, exist_ok=True)

    era5_period, gdas_period = _get_data_sources(ini_dt, fim_dt)
    if era5_period:
        logger.info('ERA5: {} a {}', era5_period[0].date(), era5_period[1].date())
    if gdas_period:
        logger.info('GDAS: {} a {}', gdas_period[0].date(), gdas_period[1].date())

    # ── Etapa 1: ERA5 omega ───────────────────────────────────────────────
    era5_omega_files: List[Path] = []
    if era5_period:
        logger.info('Etapa 1: ERA5 omega {} hPa ...', LEVELS_HPA)
        era5_omega_files = ensure_era5_omega_for_period(
            start=era5_period[0], end=era5_period[1], levels_hpa=LEVELS_HPA,
        )

    # ── Etapa 2: GDAS omega ───────────────────────────────────────────────
    gdas_omega_files: List[Path] = []
    if gdas_period:
        logger.info('Etapa 2: GDAS omega {} hPa ...', LEVELS_HPA)
        gdas_omega_files = ensure_gdas_omega_for_period(
            start=gdas_period[0], end=gdas_period[1], levels_hpa=LEVELS_HPA,
        )

    # ── Etapa 3: Média omega ──────────────────────────────────────────────
    logger.info('Etapa 3: Calculando média omega (30°S–30°N) ...')
    da_omega_mean = _compute_omega_mean(era5_omega_files, gdas_omega_files, ini_dt, fim_dt)
    if da_omega_mean is None:
        raise ValueError('Nenhum dado de omega disponível para o período.')
    era5_lons = da_omega_mean['lon'].values

    # ── Etapa 4: Climatologia PSL omega ───────────────────────────────────
    logger.info('Etapa 4: Climatologia PSL omega {} hPa ...', LEVELS_HPA)
    psl_path = get_clim_omega_multilevel(ini_str, fim_str, LEVELS_HPA)

    # ── Etapa 5: Anomalia omega ───────────────────────────────────────────
    logger.info('Etapa 5: Anomalia omega ...')
    da_psl = _load_psl_omega_mean(psl_path, era5_lons)
    levels_sorted = sorted(LEVELS_HPA)
    omega_anom = np.full((len(levels_sorted), len(era5_lons)), np.nan)
    for i, lev in enumerate(levels_sorted):
        omega_anom[i, :] = (
            da_omega_mean.sel(pressure_level=lev, method='nearest').values
            - da_psl.sel(level=lev, method='nearest').values
        )
    nan_frac = np.isnan(omega_anom).mean()
    if nan_frac > 0.3:
        raise ValueError(f'Anomalia omega com {nan_frac:.0%} de NaN.')
    logger.info(
        'Omega anom: min={:.4f} Pa/s, max={:.4f} Pa/s',
        float(np.nanmin(omega_anom)),
        float(np.nanmax(omega_anom)),
    )

    # ── Etapa 6: OISSTv2 ──────────────────────────────────────────────────
    logger.info('Etapa 6: OISSTv2 ...')
    da_sst_mean: Optional[xr.DataArray] = None
    try:
        sst_paths = _download_sst_anos(dados_dir, ini_dt, fim_dt, logger)
        da_sst_mean = _load_sst_mean(sst_paths, ini_dt, fim_dt)
        # Anomalia = SST absoluta media - climatologia diaria recortada no periodo
        clim_arr = clim_mean_array(
            np.datetime64(ini_dt.date(), 'D'), np.datetime64(fim_dt.date(), 'D'),
            da_sst_mean['lat'].values, da_sst_mean['lon'].values, logger,
        )
        da_sst_mean = da_sst_mean.copy(data=da_sst_mean.values - clim_arr)
    except Exception as exc:
        logger.warning('SST indisponível ({}). Piso sem SSTA.', exc)

    # ── Etapa 8: ERA5 U/V 850 hPa ────────────────────────────────────────
    era5_uv850_files: List[Path] = []
    if era5_period:
        logger.info('Etapa 8: ERA5 U/V 850 hPa {} → {} ...', era5_period[0].date(), era5_period[1].date())
        era5_uv850_files = ensure_era5_uv850_for_period(
            start=era5_period[0], end=era5_period[1],
        )

    # ── Etapas 9–10: climatologia PSL + anomalia U/V 850 ─────────────────
    uv850_grid = None
    if era5_uv850_files:
        logger.info('Etapa 9: Climatologia PSL U/V 850 hPa ...')
        psl_u850_path, psl_v850_path = get_clim_wnd850_paths(ini_str, fim_str)

        logger.info('Etapa 10: Anomalia U/V 850 hPa ...')
        u_era5_m, v_era5_m = _load_era5_uv850_mean(era5_uv850_files, ini_dt, fim_dt)
        if u_era5_m is not None:
            u_psl_clim, v_psl_clim = _load_psl_uv850(psl_u850_path, psl_v850_path)
            lon_2d_w, lat_2d_w, u_anom_2d, v_anom_2d = _build_uv850_cone_grid(
                u_era5_m, v_era5_m, u_psl_clim, v_psl_clim,
            )
            uv850_grid = (lon_2d_w, lat_2d_w, u_anom_2d, v_anom_2d)
            logger.info(
                'U850 anom: min={:.2f} m/s, max={:.2f} m/s | '
                'V850 anom: min={:.2f} m/s, max={:.2f} m/s',
                float(np.nanmin(u_anom_2d)), float(np.nanmax(u_anom_2d)),
                float(np.nanmin(v_anom_2d)), float(np.nanmax(v_anom_2d)),
            )
        else:
            logger.warning('U/V 850 ERA5 não carregado — vetores omitidos.')
    else:
        logger.warning('Período ERA5 indisponível para U/V 850 — vetores omitidos.')

    # ── Etapa 12: ERA5 U/V 200 hPa (vento divergente) ────────────────────
    div200_grid = None
    era5_uv200_files: List[Path] = []
    if era5_period:
        logger.info('Etapa 12: ERA5 U/V 200 hPa {} → {} ...', era5_period[0].date(), era5_period[1].date())
        era5_uv200_files = ensure_era5_uv200_for_period(
            start=era5_period[0], end=era5_period[1],
        )

    if era5_uv200_files:
        logger.info('Etapa 13: Calculando vento divergente 200 hPa (Poisson espectral) ...')
        u200_m, v200_m = _load_era5_uv200_mean(era5_uv200_files, ini_dt, fim_dt)
        if u200_m is not None:
            lat_d, lon_d, u_div, v_div = _compute_divergent_wind(u200_m, v200_m, lat_max=DIV200_LAT_MAX)
            lon_2d_d, lat_2d_d, u_div_sub, v_div_sub = _build_div200_grid(lat_d, lon_d, u_div, v_div)
            div200_grid = (lon_2d_d, lat_2d_d, u_div_sub, v_div_sub)
            logger.info(
                'Vdiv 200: u=[{:.2f},{:.2f}] m/s | v=[{:.2f},{:.2f}] m/s',
                float(np.nanmin(u_div)), float(np.nanmax(u_div)),
                float(np.nanmin(v_div)), float(np.nanmax(v_div)),
            )
        else:
            logger.warning('U/V 200 hPa ERA5 não carregado — vento divergente omitido.')
    else:
        logger.warning('Período ERA5 indisponível para U/V 200 hPa — vento divergente omitido.')

    # ── Etapa 14: Plotly 3D ───────────────────────────────────────────────
    logger.info('Etapa 14: Renderizando figura Plotly 3D ...')
    out_path = output_dir / png_name
    _plot_walker_cell_plotly(
        lons=era5_lons,
        levels_hpa=levels_sorted,
        omega_anom=omega_anom,
        da_sst_mean=da_sst_mean,
        ini_str=ini_str,
        fim_str=fim_str,
        era5_period=era5_period,
        gdas_period=gdas_period,
        output_path=out_path,
        logger=logger,
        uv850_grid=uv850_grid,
        div200_grid=div200_grid,
    )

    elapsed = time.time() - start_time
    save_cache_metadata(SCRIPT_ID, cache_params, output_files)
    logger.info('=' * 80)
    logger.info('Concluído em {:.1f}s.', elapsed)
    logger.info('=' * 80)
