# app/src/uteis/plot_chi200.py
# -*- coding: utf-8 -*-
"""
Download e processamento de vento 250 hPa (ERA5) para anomalia de CHI200.

Pipeline:
1. Baixa dados ERA5 de u/v em 250 hPa via CDS (NetCDF)
2. Concatena arquivos mensais e calcula media diaria
3. Carrega climatologias de u/v 250 hPa (uwnd250_clim, vwnd250_clim)
4. Interpola climatologias para o grid do ERA5 se necessario
5. Calcula anomalia = media_periodo - media_climatologica
6. Calcula velocity potential (chi) e vento divergente (uchi, vchi) via scipy
7. Salva resultado em dados/chi200.nc

Chamado por: scripts/s02_chi200_anom.py via `from app.src.uteis.plot_chi200 import main`
"""

from __future__ import annotations

# Bibliotecas padrão
# Bibliotecas padrao
import logging
from datetime import datetime
from pathlib import Path
from typing import Sequence

# Bibliotecas de terceiros
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.util import add_cyclic_point as _acp
from scipy import fft

# Módulos locais
# Modulos locais
from app.src.uteis.downloaders_wind250 import ensure_era5_uv250_for_period

# -----------------------------------------------------------------------------
# Integracao com settings
# -----------------------------------------------------------------------------
try:
    # Módulos locais
    from app.shared.settings_factory import settings  # type: ignore

    DIR_DADOS_BASE = Path(settings.DIR_DADOS)
except Exception:
    DIR_DADOS_BASE = Path('dados')

# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
LOGGER = logging.getLogger('PLOT_CHI200')
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)

# Horas sinoticas padrao
DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)

# Constantes fisicas
EARTH_RADIUS = 6.371e6  # metros
G = 9.80665


# -----------------------------------------------------------------------------
# Utilitarios (mesmos padroes de plot_geop250.py)
# -----------------------------------------------------------------------------
def _ensure_time_coord(obj):
    """Normaliza valid_time -> time."""
    if hasattr(obj, 'dims') and 'time' not in obj.dims and 'valid_time' in obj.dims:
        obj = obj.rename({'valid_time': 'time'})
    elif hasattr(obj, 'coords') and 'time' not in obj.coords and 'valid_time' in obj.coords:
        obj = obj.rename({'valid_time': 'time'})
    if 'time' not in obj.coords:
        raise KeyError("Nem 'time' nem 'valid_time' encontrados.")
    return obj


def _drop_or_collapse_expver(ds: xr.Dataset) -> xr.Dataset:
    """Colapsa dim expver/number se existirem (ERA5/ERA5T)."""
    rename_dims = {}
    for d in ds.dims:
        dl = d.lower()
        if dl == 'expver' and d != 'expver':
            rename_dims[d] = 'expver'
        elif dl == 'number' and d != 'number':
            rename_dims[d] = 'number'

    if rename_dims:
        ds = ds.rename(rename_dims)

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


def _normalize_latlon_names(ds):
    """Renomeia latitude/longitude -> lat/lon se necessario."""
    rename = {}
    for name in ds.dims:
        low = name.lower()
        if low in {'latitude'} and 'lat' not in ds.dims:
            rename[name] = 'lat'
        elif low in {'longitude'} and 'lon' not in ds.dims:
            rename[name] = 'lon'
    if rename:
        ds = ds.rename(rename)
    return ds


def _normalize_lon(ds):
    """Converte longitude de 0..360 para -180..180 se necessario."""
    if 'lon' not in ds.coords:
        return ds
    lon_vals = ds['lon'].values
    if np.any(lon_vals > 180):
        ds = ds.assign_coords(lon=(ds['lon'].values + 180) % 360 - 180)
        ds = ds.sortby('lon')
    return ds


def _sort_and_dedup_time(ds: xr.Dataset) -> xr.Dataset:
    """Ordena e remove timestamps duplicados."""
    ds = ds.sortby('time')
    t = pd.DatetimeIndex(pd.to_datetime(ds['time'].values))
    _, idx = np.unique(t.values, return_index=True)
    idx = np.sort(idx)
    if len(idx) != ds.sizes.get('time', 0):
        LOGGER.warning('Removendo %d timestamps duplicados.', ds.sizes['time'] - len(idx))
        ds = ds.isel(time=idx)
    return ds


def _drop_feb29(ds: xr.Dataset) -> xr.Dataset:
    """Remove 29/02 para compatibilizar com climatologia 365d."""
    ds = _ensure_time_coord(ds)
    t = xr.DataArray(ds['time'].values, dims=['time'])
    mask = ~((t.dt.month == 2) & (t.dt.day == 29))
    n_before = ds.sizes['time']
    ds2 = ds.isel(time=mask.values)
    n_after = ds2.sizes['time']
    if n_after < n_before:
        LOGGER.warning(
            'Removendo %d registro(s) de 29/02 para compatibilizar com climatologia 365d.',
            n_before - n_after,
        )
    return ds2


# -----------------------------------------------------------------------------
# Abertura e concatenacao de arquivos mensais
# -----------------------------------------------------------------------------
def _open_and_merge_uv_files(files: Sequence[Path]) -> xr.Dataset:
    """Abre e concatena arquivos mensais de u/v 250 hPa em NetCDF."""
    if not files:
        raise ValueError('Lista de arquivos de vento 250 hPa vazia.')

    dsets = []
    for fp in files:
        LOGGER.info('Abrindo: %s', fp)
        ds = xr.open_dataset(fp, engine='netcdf4')
        ds = _ensure_time_coord(ds)
        ds = _drop_or_collapse_expver(ds)
        ds = _normalize_latlon_names(ds)
        dsets.append(ds)

    if len(dsets) == 1:
        ds_all = dsets[0]
    else:
        ds_all = xr.concat(dsets, dim='time', data_vars='minimal', coords='minimal', compat='override')

    ds_all = _ensure_time_coord(ds_all)
    ds_all = _sort_and_dedup_time(ds_all)

    LOGGER.info(
        'Dataset combinado: %d timestamps | %s -> %s',
        ds_all.sizes.get('time', 0),
        str(pd.to_datetime(ds_all.time.values[0])) if ds_all.sizes.get('time', 0) else 'NA',
        str(pd.to_datetime(ds_all.time.values[-1])) if ds_all.sizes.get('time', 0) else 'NA',
    )
    return ds_all


# -----------------------------------------------------------------------------
# Media diaria a partir de horas sinoticas
# -----------------------------------------------------------------------------
def _compute_daily_mean(
    ds: xr.Dataset,
    required_hours: Sequence[int] = DEFAULT_SYNOPTIC_HOURS,
) -> xr.Dataset:
    """Calcula media diaria filtrando apenas horas sinoticas requeridas."""
    ds = _ensure_time_coord(ds)
    ds = _sort_and_dedup_time(ds)

    required_set = set(int(h) for h in required_hours)
    t_idx = pd.DatetimeIndex(pd.to_datetime(ds['time'].values))

    mask = np.array([h in required_set for h in t_idx.hour], dtype=bool)
    ds = ds.isel(time=mask)

    if ds.sizes.get('time', 0) == 0:
        raise ValueError('Dataset sem timestamps apos filtrar horas sinoticas.')

    ds_daily = ds.resample(time='1D').mean(keep_attrs=True)

    # Remove dias vazios criados pelo resample
    for vn in ds_daily.data_vars:
        da = ds_daily[vn]
        other_dims = [d for d in da.dims if d != 'time']
        if other_dims:
            valid = da.notnull().any(dim=other_dims)
        else:
            valid = da.notnull()
        ds_daily = ds_daily.isel(time=valid.values)
        break

    LOGGER.info(
        'Media diaria: %d dias | %s -> %s',
        ds_daily.sizes.get('time', 0),
        str(pd.to_datetime(ds_daily.time.values[0])) if ds_daily.sizes.get('time', 0) else 'NA',
        str(pd.to_datetime(ds_daily.time.values[-1])) if ds_daily.sizes.get('time', 0) else 'NA',
    )
    return ds_daily


# -----------------------------------------------------------------------------
# Climatologia
# -----------------------------------------------------------------------------
def _load_wind_climatology(path_clim: Path, component: str) -> xr.DataArray:
    """Carrega climatologia de componente de vento 250 hPa (u ou v).

    Parameters
    ----------
    path_clim : Path
        Caminho para o arquivo NetCDF da climatologia.
    component : str
        'u' ou 'v' — usado para identificar a variavel no dataset.
    """
    if not path_clim.exists():
        raise FileNotFoundError(f'Climatologia vento 250 hPa ({component}) nao encontrada: {path_clim}')

    LOGGER.info('Abrindo climatologia %s 250hPa: %s', component, path_clim)
    ds = xr.open_dataset(path_clim, engine='netcdf4')
    ds = _normalize_latlon_names(ds)

    LOGGER.info('Variaveis na climatologia %s: %s', component, list(ds.data_vars))

    # Procura variavel de vento
    u_names = ('uwnd', 'u', 'u_component_of_wind', 'U_GRD_L100')
    v_names = ('vwnd', 'v', 'v_component_of_wind', 'V_GRD_L100')
    candidates = u_names if component == 'u' else v_names

    for vname in candidates:
        if vname in ds.data_vars:
            return ds[vname]

    # Fallback: primeira variavel numerica 3D
    for vname in ds.data_vars:
        if np.issubdtype(ds[vname].dtype, np.number) and len(ds[vname].dims) >= 2:
            LOGGER.warning('Usando variavel %s como %s-wind da climatologia.', vname, component)
            return ds[vname]

    raise KeyError(
        f'Nao encontrei variavel de vento ({component}) na climatologia: {list(ds.data_vars)}'
    )


def _select_climatology_same_days(
    clim: xr.DataArray,
    daily_dates: xr.DataArray,
) -> xr.DataArray:
    """Seleciona na climatologia os mesmos dias do periodo.

    Suporta 3 formatos de coordenada temporal na climatologia:
    - time com inteiros 1-365 (dayofyear como inteiro)
    - time com datetimes (selecao por month-day)
    - dayofyear como coordenada separada
    """
    target_dates = pd.DatetimeIndex(pd.to_datetime(daily_dates.values))

    # Caso 1: coord 'time' com inteiros 1-365 (dayofyear)
    if 'time' in clim.coords:
        time_vals = clim['time'].values
        if np.issubdtype(type(time_vals[0]), np.integer) or (
            hasattr(time_vals[0], 'dtype') and np.issubdtype(time_vals[0].dtype, np.integer)
        ):
            LOGGER.info('Climatologia com coord time inteira (dayofyear 1-365).')
            target_doy = target_dates.dayofyear.to_numpy()
            clim_sel = clim.sel(time=xr.DataArray(target_doy, dims=['time']))
            clim_sel = clim_sel.assign_coords(time=daily_dates.values)
            LOGGER.info('Climatologia selecionada por dayofyear (inteiro): %d dias.', len(target_doy))
            return clim_sel

        # Caso 2: coord 'time' com datetimes
        try:
            clim_dates = pd.DatetimeIndex(pd.to_datetime(time_vals))
        except Exception:
            raise KeyError(
                f"Coord 'time' da climatologia nao e datetime nem inteiro: {type(time_vals[0])}"
            )

        clim_md = pd.Index(
            [f'{t.month:02d}-{t.day:02d}' for t in clim_dates],
            name='monthday',
        )
        target_md = [f'{t.month:02d}-{t.day:02d}' for t in target_dates]

        md_to_idx = {}
        for i, md in enumerate(clim_md.tolist()):
            md_to_idx[md] = i

        missing = [md for md in target_md if md not in md_to_idx]
        if missing:
            LOGGER.warning(
                'Dias ausentes na climatologia (serao ignorados na anomalia): %s', missing[:10]
            )
            valid_pairs = [(md, i) for i, md in enumerate(target_md) if md in md_to_idx]
            if not valid_pairs:
                raise ValueError('Nenhum dia do periodo encontrado na climatologia.')
            valid_md, valid_target_idx = zip(*valid_pairs)
            idxs = [md_to_idx[md] for md in valid_md]
            clim_sel = clim.isel(time=idxs)
            target_times = daily_dates.values[list(valid_target_idx)]
            clim_sel = clim_sel.assign_coords(time=target_times)
        else:
            idxs = [md_to_idx[md] for md in target_md]
            clim_sel = clim.isel(time=idxs)
            clim_sel = clim_sel.assign_coords(time=daily_dates.values)

        LOGGER.info("Climatologia selecionada por month-day usando coord 'time'.")
        return clim_sel

    # Caso 3: coord 'dayofyear' separada
    if 'dayofyear' in clim.coords:
        target_doy = target_dates.dayofyear.to_numpy()
        clim_sel = clim.sel(dayofyear=xr.DataArray(target_doy, dims=['time']))
        clim_sel = clim_sel.assign_coords(time=daily_dates.values)
        LOGGER.info("Climatologia selecionada por coord 'dayofyear'.")
        return clim_sel

    raise KeyError("Climatologia precisa ter coord 'time' (datetime ou dayofyear int) ou 'dayofyear'.")


def _add_cyclic_and_interp_clim(clim_period: xr.DataArray, target_da: xr.DataArray) -> xr.DataArray:
    """Adiciona cyclic point e interpola climatologia para o grid do ERA5."""
    # Cyclic point para fechar gap longitudinal
    if 'lon' in clim_period.coords:
        clim_vals, clim_lon = _acp(clim_period.values, coord=clim_period['lon'].values)
        clim_period = xr.DataArray(
            clim_vals,
            dims=clim_period.dims,
            coords={d: clim_period.coords[d] for d in clim_period.dims if d != 'lon'},
        )
        clim_period = clim_period.assign_coords(lon=('lon', clim_lon))

    # Interpolar para o grid do ERA5 se necessario
    if (
        'lat' in target_da.coords
        and 'lon' in target_da.coords
        and 'lat' in clim_period.coords
        and 'lon' in clim_period.coords
    ):
        if (
            target_da.sizes.get('lat') != clim_period.sizes.get('lat')
            or target_da.sizes.get('lon') != clim_period.sizes.get('lon')
            or not np.array_equal(target_da['lat'].values, clim_period['lat'].values)
            or not np.array_equal(target_da['lon'].values, clim_period['lon'].values)
        ):
            LOGGER.warning('Grid da climatologia difere do ERA5. Interpolando climatologia.')
            clim_period = clim_period.interp_like(target_da)

    return clim_period


# -----------------------------------------------------------------------------
# Calculo de velocity potential (chi) e vento divergente via scipy
# -----------------------------------------------------------------------------
def _compute_divergence(u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Calcula divergencia do vento em coordenadas esfericas.

    D = (1/(a*cos(phi))) * [du/dlambda + d(v*cos(phi))/dphi]

    Parameters
    ----------
    u, v : np.ndarray (nlat, nlon)
        Componentes zonal e meridional do vento.
    lat, lon : np.ndarray (1D, em graus)
        Vetores de latitude e longitude.

    Returns
    -------
    div : np.ndarray (nlat, nlon)
        Campo de divergencia.
    """
    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon)
    cosphi = np.cos(phi)

    dlam = np.mean(np.diff(lam))
    dphi = np.mean(np.diff(phi))

    # du/dlambda (derivada zonal)
    du_dlam = np.gradient(u, dlam, axis=1)

    # d(v*cosphi)/dphi (derivada meridional)
    v_cosphi = v * cosphi[:, np.newaxis]
    dvcosphi_dphi = np.gradient(v_cosphi, dphi, axis=0)

    # Evita divisao por zero nos polos
    cosphi_safe = np.where(np.abs(cosphi) < 1e-10, 1e-10, cosphi)

    div = (1.0 / (EARTH_RADIUS * cosphi_safe[:, np.newaxis])) * (du_dlam + dvcosphi_dphi)

    return div


def _solve_poisson_sphere(
    div: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    """Resolve a equacao de Poisson na esfera: nabla^2(chi) = div.

    Usa metodo espectral com FFT na direcao zonal e diferencias finitas
    na direcao meridional, resolvendo o sistema tridiagonal resultante.

    Parameters
    ----------
    div : np.ndarray (nlat, nlon)
        Campo de divergencia.
    lat, lon : np.ndarray (1D, em graus)
        Vetores de latitude e longitude.

    Returns
    -------
    chi : np.ndarray (nlat, nlon)
        Velocity potential.
    """
    nlat, nlon = div.shape
    phi = np.deg2rad(lat)
    dphi = np.mean(np.diff(phi))
    cosphi = np.cos(phi)

    # FFT na direcao zonal
    div_hat = fft.rfft(div, axis=1)
    chi_hat = np.zeros_like(div_hat)

    # Numero de onda zonal
    m = np.arange(div_hat.shape[1])

    for k in range(div_hat.shape[1]):
        # Montar sistema tridiagonal para cada numero de onda
        # (1/a^2) * [d/dphi(cos(phi) * dchi/dphi) / cos(phi)] - m^2/(a*cos(phi))^2 * chi = div
        #
        # Discretizacao:
        # chi_{j+1} * A_j + chi_j * B_j + chi_{j-1} * C_j = div_j
        A = np.zeros(nlat)
        B = np.zeros(nlat)
        C = np.zeros(nlat)
        rhs = np.zeros(nlat, dtype=complex)

        for j in range(1, nlat - 1):
            cos_jp = np.cos(phi[j] + dphi / 2)
            cos_jm = np.cos(phi[j] - dphi / 2)
            cos_j = cosphi[j]
            cos_j_safe = max(abs(cos_j), 1e-10)

            A[j] = cos_jp / (EARTH_RADIUS**2 * dphi**2 * cos_j_safe)
            C[j] = cos_jm / (EARTH_RADIUS**2 * dphi**2 * cos_j_safe)
            B[j] = -(A[j] + C[j]) - (m[k] ** 2) / (EARTH_RADIUS * cos_j_safe) ** 2
            rhs[j] = div_hat[j, k]

        # Condicoes de contorno (polos): chi fixo = 0
        B[0] = 1.0
        rhs[0] = 0.0
        B[-1] = 1.0
        rhs[-1] = 0.0

        # Para m=0 o sistema e singular (solucao definida a menos de constante)
        # Fixar chi[0]=0 resolve a ambiguidade
        if k == 0:
            B[0] = 1.0
            rhs[0] = 0.0

        # Resolver sistema tridiagonal (Thomas algorithm)
        chi_col = _solve_tridiag(C, B, A, rhs)
        chi_hat[:, k] = chi_col

    # Transformada inversa
    chi = fft.irfft(chi_hat, n=nlon, axis=1)

    return chi


def _solve_tridiag(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """Resolve sistema tridiagonal Ax = d pelo algoritmo de Thomas.

    a = subdiagonal, b = diagonal principal, c = superdiagonal.
    """
    n = len(d)
    c_ = np.zeros(n, dtype=complex)
    d_ = np.zeros(n, dtype=complex)
    x = np.zeros(n, dtype=complex)

    c_[0] = c[0] / b[0] if b[0] != 0 else 0
    d_[0] = d[0] / b[0] if b[0] != 0 else 0

    for i in range(1, n):
        denom = b[i] - a[i] * c_[i - 1]
        if abs(denom) < 1e-30:
            denom = 1e-30
        c_[i] = c[i] / denom
        d_[i] = (d[i] - a[i] * d_[i - 1]) / denom

    x[-1] = d_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i + 1]

    return x


def _compute_divergent_wind(
    chi: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula vento divergente a partir do velocity potential.

    u_div = (1/(a*cos(phi))) * dchi/dlambda
    v_div = (1/a) * dchi/dphi

    Parameters
    ----------
    chi : np.ndarray (nlat, nlon)
    lat, lon : np.ndarray (1D, em graus)

    Returns
    -------
    u_div, v_div : np.ndarray (nlat, nlon)
    """
    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon)
    cosphi = np.cos(phi)
    cosphi_safe = np.where(np.abs(cosphi) < 1e-10, 1e-10, cosphi)

    dlam = np.mean(np.diff(lam))
    dphi = np.mean(np.diff(phi))

    dchi_dlam = np.gradient(chi, dlam, axis=1)
    dchi_dphi = np.gradient(chi, dphi, axis=0)

    u_div = (1.0 / (EARTH_RADIUS * cosphi_safe[:, np.newaxis])) * dchi_dlam
    v_div = (1.0 / EARTH_RADIUS) * dchi_dphi

    return u_div, v_div


def _helmholtz_chi(
    u_anom: np.ndarray,
    v_anom: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula velocity potential e vento divergente via decomposicao de Helmholtz.

    Parameters
    ----------
    u_anom, v_anom : np.ndarray (nlat, nlon)
        Anomalia das componentes u e v do vento.
    lat, lon : np.ndarray (1D, em graus)

    Returns
    -------
    chi : np.ndarray (nlat, nlon)
        Velocity potential (m^2/s).
    u_div : np.ndarray (nlat, nlon)
        Componente divergente de u.
    v_div : np.ndarray (nlat, nlon)
        Componente divergente de v.
    """
    LOGGER.info('Calculando divergencia do vento...')
    div = _compute_divergence(u_anom, v_anom, lat, lon)

    LOGGER.info('Resolvendo equacao de Poisson para velocity potential (chi)...')
    chi = _solve_poisson_sphere(div, lat, lon)

    LOGGER.info('Calculando vento divergente a partir de chi...')
    u_div, v_div = _compute_divergent_wind(chi, lat, lon)

    LOGGER.info(
        'Chi calculado: min=%.2e, max=%.2e m2/s',
        float(np.nanmin(chi)),
        float(np.nanmax(chi)),
    )

    return chi, u_div, v_div


# -----------------------------------------------------------------------------
# Identificacao de variaveis no dataset ERA5
# -----------------------------------------------------------------------------
def _find_uv_vars(ds: xr.Dataset) -> tuple[str, str]:
    """Identifica nomes das variaveis u e v no dataset."""
    u_candidates = ('u', 'u_component_of_wind', 'U_GRD_L100', 'uwnd')
    v_candidates = ('v', 'v_component_of_wind', 'V_GRD_L100', 'vwnd')

    u_var = None
    v_var = None
    for vn in u_candidates:
        if vn in ds.data_vars:
            u_var = vn
            break
    for vn in v_candidates:
        if vn in ds.data_vars:
            v_var = vn
            break

    if u_var is None or v_var is None:
        raise KeyError(f'Nao encontrei variaveis u/v no dataset. Disponiveis: {list(ds.data_vars)}')

    return u_var, v_var


# -----------------------------------------------------------------------------
# Pipeline principal
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Download, processamento e calculo de anomalia CHI200 (velocity potential).

    Resultado: dados/chi200.nc com variaveis:
    - chi_anom_mean_scaled: velocity potential anomaly (10^5 m^2/s)
    - uchi_anom_mean: componente u do vento divergente (m/s)
    - vchi_anom_mean: componente v do vento divergente (m/s)
    """
    dt_ini = datetime.strptime(settings.DATA_INICIAL, '%Y-%m-%d')
    dt_fim = datetime.strptime(settings.DATA_FINAL, '%Y-%m-%d')

    LOGGER.info('=' * 70)
    LOGGER.info('PLOT_CHI200: Download e anomalia velocity potential 250 hPa')
    LOGGER.info('Periodo: %s a %s', settings.DATA_INICIAL, settings.DATA_FINAL)
    LOGGER.info('=' * 70)

    # 1. Download ERA5 u/v 250 hPa (NetCDF)
    LOGGER.info('Etapa 1: Download ERA5 u/v 250 hPa (NetCDF)')
    uv_files = ensure_era5_uv250_for_period(
        start=dt_ini,
        end=dt_fim,
        hours_utc=list(DEFAULT_SYNOPTIC_HOURS),
        force_redownload=getattr(settings, 'FORCE_DOWNLOAD', False),
    )
    LOGGER.info('Arquivos de vento 250 hPa: %d', len(uv_files))
    for f in uv_files:
        LOGGER.info('  - %s', f)

    # 2. Abrir e concatenar arquivos mensais
    LOGGER.info('Etapa 2: Concatenando arquivos mensais')
    ds_uv = _open_and_merge_uv_files(uv_files)
    ds_uv = _normalize_latlon_names(ds_uv)
    ds_uv = _normalize_lon(ds_uv)

    # 3. Media diaria (00/06/12/18 UTC)
    LOGGER.info('Etapa 3: Calculando media diaria')
    ds_daily = _compute_daily_mean(ds_uv)
    ds_daily = _drop_feb29(ds_daily)

    # Identificar variaveis u/v
    u_var, v_var = _find_uv_vars(ds_daily)
    u_daily = ds_daily[u_var]
    v_daily = ds_daily[v_var]

    # Dropar dimensao pressure_level se existir (unico nivel 250 hPa)
    for dim_name in ('pressure_level', 'isobaricInhPa', 'level'):
        if dim_name in u_daily.dims:
            u_daily = u_daily.isel({dim_name: 0}, drop=True)
            v_daily = v_daily.isel({dim_name: 0}, drop=True)
            LOGGER.info('Dimensao %s removida do ERA5 (nivel unico).', dim_name)

    # 4. Carregar climatologias u/v e calcular anomalias
    LOGGER.info('Etapa 4: Carregando climatologias e calculando anomalias')

    u_clim_path = Path(settings.FILE_CLIMATOLOGIA_UWND250)
    v_clim_path = Path(settings.FILE_CLIMATOLOGIA_VWND250)

    u_clim_da = _load_wind_climatology(u_clim_path, 'u')
    v_clim_da = _load_wind_climatology(v_clim_path, 'v')

    # Dropar dimensao 'level' se existir
    for clim_da in (u_clim_da, v_clim_da):
        if 'level' in clim_da.dims:
            clim_da = clim_da.isel(level=0, drop=True)
    # Re-assign apos possiveis drops
    if 'level' in u_clim_da.dims:
        u_clim_da = u_clim_da.isel(level=0, drop=True)
    if 'level' in v_clim_da.dims:
        v_clim_da = v_clim_da.isel(level=0, drop=True)

    # Normalizar lon das climatologias
    for name, clim in [('u_clim', u_clim_da), ('v_clim', v_clim_da)]:
        ds_tmp = _normalize_latlon_names(clim.to_dataset(name=clim.name or name))
        ds_tmp = _normalize_lon(ds_tmp)
        if name == 'u_clim':
            u_clim_da = list(ds_tmp.data_vars.values())[0]
        else:
            v_clim_da = list(ds_tmp.data_vars.values())[0]

    # Selecionar mesmos dias na climatologia
    u_clim_period = _select_climatology_same_days(u_clim_da, u_daily['time'])
    v_clim_period = _select_climatology_same_days(v_clim_da, v_daily['time'])

    # Cyclic point + interpolacao para grid ERA5
    u_clim_period = _add_cyclic_and_interp_clim(u_clim_period, u_daily)
    v_clim_period = _add_cyclic_and_interp_clim(v_clim_period, v_daily)

    # Medias do periodo e climatologica
    u_period_mean = u_daily.mean(dim='time')
    v_period_mean = v_daily.mean(dim='time')
    u_clim_mean = u_clim_period.mean(dim='time')
    v_clim_mean = v_clim_period.mean(dim='time')

    # Anomalias
    u_anom = u_period_mean - u_clim_mean
    v_anom = v_period_mean - v_clim_mean

    LOGGER.info(
        'Anomalia u: min=%.2f, max=%.2f m/s',
        float(u_anom.min()),
        float(u_anom.max()),
    )
    LOGGER.info(
        'Anomalia v: min=%.2f, max=%.2f m/s',
        float(v_anom.min()),
        float(v_anom.max()),
    )

    # 5. Calcular velocity potential e vento divergente
    LOGGER.info('Etapa 5: Calculando velocity potential (chi) e vento divergente')
    lat = u_anom['lat'].values
    lon = u_anom['lon'].values

    # Garantir lat decrescente (N->S) para calculo correto
    if lat[0] < lat[-1]:
        u_anom = u_anom.sortby('lat', ascending=False)
        v_anom = v_anom.sortby('lat', ascending=False)
        lat = u_anom['lat'].values

    chi, u_div, v_div = _helmholtz_chi(
        u_anom.values,
        v_anom.values,
        lat,
        lon,
    )

    # Escalar chi para 10^5 m^2/s
    CHI_SCALE = 1e5
    chi_scaled = chi / CHI_SCALE

    LOGGER.info(
        'Chi escalado (10^5 m2/s): min=%.1f, max=%.1f',
        float(np.nanmin(chi_scaled)),
        float(np.nanmax(chi_scaled)),
    )

    # 6. Salvar como dados/chi200.nc
    LOGGER.info('Etapa 6: Salvando chi200.nc')

    chi_da = xr.DataArray(
        chi_scaled,
        dims=['lat', 'lon'],
        coords={'lat': lat, 'lon': lon},
        name='chi_anom_mean_scaled',
        attrs={
            'long_name': 'velocity potential anomaly (scaled)',
            'units': '1e5 m2 s-1',
        },
    )

    uchi_da = xr.DataArray(
        u_div,
        dims=['lat', 'lon'],
        coords={'lat': lat, 'lon': lon},
        name='uchi_anom_mean',
        attrs={'long_name': 'divergent u-wind anomaly', 'units': 'm s-1'},
    )

    vchi_da = xr.DataArray(
        v_div,
        dims=['lat', 'lon'],
        coords={'lat': lat, 'lon': lon},
        name='vchi_anom_mean',
        attrs={'long_name': 'divergent v-wind anomaly', 'units': 'm s-1'},
    )

    ds_out = xr.Dataset({
        'chi_anom_mean_scaled': chi_da,
        'uchi_anom_mean': uchi_da,
        'vchi_anom_mean': vchi_da,
    })

    output_path = DIR_DADOS_BASE / 'chi200.nc'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    ds_out.to_netcdf(output_path, engine='netcdf4')
    LOGGER.info('Chi200 salvo em: %s', output_path)
    LOGGER.info('=' * 70)
    LOGGER.info('PLOT_CHI200: Concluido com sucesso')
    LOGGER.info('=' * 70)


if __name__ == '__main__':
    main()
