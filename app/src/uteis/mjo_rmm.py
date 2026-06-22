# -*- coding: utf-8 -*-
"""
Indice RMM da MJO (Wheeler & Hendon 2004) com EOF COMBINADO proprio.

Reproduz o metodo do RMM com dados na MESMA combinacao original do WH04 (e a mesma base das
climatologias do projeto, garantindo consistencia dos fatores de normalizacao):

  - OLR  : CPC Blended OLR 2.5 (satelite) — olr.day.mean.nc (ver clim_diaria_olr).
  - U850 : NCEP Reanalysis 2.5 — OPeNDAP dailyavgs (ncep.reanalysis.dailyavgs/pressure/uwnd.YYYY.nc).
  - U200 : idem NCEP.

Pre-processamento (IDENTICO no treino do EOF e na projecao de obs/previsao):
  1. media meridional ponderada por cos(lat) na faixa 15S-15N -> serie (time, lon);
  2. anomalia = campo - climatologia do dia-do-ano (calculada da propria base e CACHEADA);
  3. remove a media dos 120 dias anteriores (trailing) -> tira ENSO/baixa frequencia;
  4. normaliza cada variavel pelo seu desvio (fator de normalizacao do treino);
  5. concatena [OLR(lon), U850(lon), U200(lon)] e projeta nos 2 primeiros EOFs combinados;
  6. RMMk = PCk / std(PCk do treino)  ->  RMM1, RMM2 com variancia unitaria.

Tudo cacheado em Entrada/arquivos_nc/mjo_rmm_eof_cpc_ncep_<ini>_<fim>.nc (construido 1x).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from app.shared.logger import get_logger
from app.shared.settings_factory import settings

logger = get_logger('mjo_rmm')

BASE_INI, BASE_FIM = 1991, 2020
LAT_BAND = 15.0          # media meridional 15S..15N
RUNNING_DAYS = 120       # remocao da media dos 120 dias anteriores (WH04)
N_MODES = 2              # RMM1, RMM2

OLR_MEAN_URL = 'https://downloads.psl.noaa.gov/Datasets/cpc_blended_olr-2.5deg/olr.day.mean.nc'
NCEP_UWND_OPENDAP = (
    'https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs/pressure/uwnd.{year}.nc'
)
VARS = ('olr', 'u850', 'u200')


def _cache_path(ini: int = BASE_INI, fim: int = BASE_FIM) -> Path:
    return Path(settings.DIR_INPUT) / 'arquivos_nc' / f'mjo_rmm_eof_cpc_ncep_{ini}_{fim}.nc'


# ---------------------------------------------------------------------------
# Leitura das fontes -> serie diaria (time, lon) media na faixa 15S-15N
# ---------------------------------------------------------------------------
def _std_latlon(da: xr.DataArray) -> xr.DataArray:
    ren = {}
    for name in list(da.dims) + list(da.coords):
        low = name.lower()
        if low == 'latitude' and 'lat' not in da.dims:
            ren[name] = 'lat'
        elif low == 'longitude' and 'lon' not in da.dims:
            ren[name] = 'lon'
    if ren:
        da = da.rename(ren)
    return da.assign_coords(lon=(da['lon'] % 360)).sortby('lon').sortby('lat')


def _band_mean(da: xr.DataArray) -> xr.DataArray:
    """Media meridional ponderada por cos(lat) na faixa |lat|<=LAT_BAND -> (time, lon)."""
    da = da.sel(lat=slice(-LAT_BAND, LAT_BAND)) if float(da['lat'][0]) < float(da['lat'][-1]) \
        else da.sel(lat=slice(LAT_BAND, -LAT_BAND))
    w = np.cos(np.deg2rad(da['lat']))
    return (da * w).sum('lat') / w.sum('lat')


def _olr_mean_file() -> Path:
    """Garante o olr.day.mean.nc local (reusa o cache do clim_diaria_olr)."""
    from app.src.uteis.clim_diaria_olr import _ensure_olr_mean_file
    return _ensure_olr_mean_file()


def _load_olr_band(dates_index: pd.DatetimeIndex | None = None) -> xr.DataArray:
    """OLR CPC (time, lon) na faixa tropical. Se `dates_index` for None, usa o periodo base."""
    ds = xr.open_dataset(str(_olr_mean_file()))
    try:
        da = ds['olr'] if 'olr' in ds.data_vars else next(iter(ds.data_vars.values()))
        da = _std_latlon(da)
        band = _band_mean(da).load()
    finally:
        ds.close()
    return band


def _load_uwnd_band(level: int, years: range) -> xr.DataArray:
    """U(level) NCEP diario (time, lon) na faixa tropical, concatenando os anos via OPeNDAP."""
    parts = []
    for y in years:
        url = NCEP_UWND_OPENDAP.format(year=y)
        with xr.open_dataset(url, decode_times=True) as ds:
            da = _std_latlon(ds['uwnd'])
            ilev = int(np.argmin(np.abs(da['level'].values - level)))
            da = da.isel(level=ilev, drop=True)
            da = da.sel(lat=slice(-LAT_BAND - 2.5, LAT_BAND + 2.5)) if float(da['lat'][0]) < float(da['lat'][-1]) \
                else da.sel(lat=slice(LAT_BAND + 2.5, -LAT_BAND - 2.5))
            parts.append(_band_mean(da).load())
        logger.info('NCEP u{} {}: {} dias', level, y, parts[-1].sizes.get('time', 0))
    return xr.concat(parts, dim='time').sortby('time')


def _align_to_lon(da: xr.DataArray, lon: np.ndarray) -> xr.DataArray:
    """Garante a grade de longitude `lon` (interp linear ciclica se necessario)."""
    if np.array_equal(np.round(da['lon'].values, 4), np.round(lon, 4)):
        return da
    return da.interp(lon=xr.DataArray(lon, dims=['lon']), method='linear')


# ---------------------------------------------------------------------------
# Pre-processamento comum (anomalia DOY -> remove 120d -> normaliza)
# ---------------------------------------------------------------------------
def _doy(t: pd.DatetimeIndex) -> np.ndarray:
    """Dia-do-ano com 29/fev mapeado p/ 28/fev (1..365)."""
    doy = np.asarray(t.dayofyear, dtype=int)
    after_leap = np.asarray(t.is_leap_year) & (doy >= 60)
    doy = np.where(after_leap, doy - 1, doy)
    return np.clip(doy, 1, 365)


def _seasonal_clim(band: np.ndarray, t: pd.DatetimeIndex) -> np.ndarray:
    """Climatologia por dia-do-ano (365, lon) a partir da serie (time, lon)."""
    doy = _doy(t)
    nlon = band.shape[1]
    clim = np.full((365, nlon), np.nan)
    for d in range(1, 366):
        sel = doy == d
        if sel.any():
            clim[d - 1] = np.nanmean(band[sel], axis=0)
    # preenche dias-do-ano sem amostra por interpolacao circular
    return clim


def _anomaly_from_clim(band: np.ndarray, t: pd.DatetimeIndex, clim_doy: np.ndarray) -> np.ndarray:
    """Anomalia = band - clim_doy[dia-do-ano]."""
    return band - clim_doy[_doy(t) - 1]


def _remove_running(anom: np.ndarray, days: int = RUNNING_DAYS) -> np.ndarray:
    """Subtrai a media dos `days` dias anteriores (trailing, causal). NaN nos primeiros `days`."""
    df = pd.DataFrame(anom)
    trailing = df.rolling(window=days, min_periods=days).mean().shift(1)
    return (df - trailing).values


# ---------------------------------------------------------------------------
# Construcao do EOF combinado (treino)
# ---------------------------------------------------------------------------
def _build_eof(ini: int = BASE_INI, fim: int = BASE_FIM) -> xr.Dataset:
    """Constroi o EOF combinado RMM (2 modos) + climatologias DOY + fatores de normalizacao."""
    years = range(ini, fim + 1)

    olr = _load_olr_band()
    lon = olr['lon'].values
    olr = olr.sel(time=slice(f'{ini}-01-01', f'{fim}-12-31'))
    u850 = _align_to_lon(_load_uwnd_band(850, years), lon)
    u200 = _align_to_lon(_load_uwnd_band(200, years), lon)

    # datas comuns (intersecao das 3 series, sem 29/fev)
    t_all = pd.DatetimeIndex(np.sort(np.array(
        list(set(pd.to_datetime(olr['time'].values).normalize())
             & set(pd.to_datetime(u850['time'].values).normalize())
             & set(pd.to_datetime(u200['time'].values).normalize())))))
    t_all = t_all[~((t_all.month == 2) & (t_all.day == 29))]

    def _on(da):
        d = da.copy()
        d = d.assign_coords(time=pd.to_datetime(d['time'].values).normalize())
        d = d.sel(time=t_all)
        return d.values

    raw = {'olr': _on(olr), 'u850': _on(u850), 'u200': _on(u200)}

    clim_doy = {v: _seasonal_clim(raw[v], t_all) for v in VARS}
    anom = {v: _anomaly_from_clim(raw[v], t_all, clim_doy[v]) for v in VARS}
    intra = {v: _remove_running(anom[v]) for v in VARS}

    valid = np.all([np.isfinite(intra[v]).all(axis=1) for v in VARS], axis=0)
    norm = {v: float(np.nanstd(intra[v][valid], ddof=1)) for v in VARS}

    # matriz combinada (time x 3*nlon), normalizada por variavel
    cols = [intra[v][valid] / norm[v] for v in VARS]
    M = np.concatenate(cols, axis=1)

    # EOF via SVD: M = U S Wt -> EOFk = Wt[k]; PCk = U[:,k]*S[k]
    U, S, Wt = np.linalg.svd(M, full_matrices=False)
    eofs = Wt[:N_MODES]                       # (2, 3*nlon)
    pcs = U[:, :N_MODES] * S[:N_MODES]        # (n, 2)
    pcstd = np.std(pcs, axis=0, ddof=1)       # (2,)
    explained = (S[:N_MODES] ** 2) / np.sum(S ** 2)

    eofs, pcstd = _apply_convention(eofs, pcstd, lon)

    logger.info('EOF RMM construido: var. explicada {:.1f}% / {:.1f}% | norm OLR/U850/U200 = '
                '{:.2f}/{:.2f}/{:.2f} | {} dias',
                explained[0] * 100, explained[1] * 100, norm['olr'], norm['u850'], norm['u200'],
                int(valid.sum()))

    nlon = len(lon)
    out = xr.Dataset(
        {
            'eofs': (('mode', 'feature'), eofs.astype('float64')),
            'pcstd': (('mode',), pcstd.astype('float64')),
            'clim_olr': (('doy', 'lon'), clim_doy['olr'].astype('float64')),
            'clim_u850': (('doy', 'lon'), clim_doy['u850'].astype('float64')),
            'clim_u200': (('doy', 'lon'), clim_doy['u200'].astype('float64')),
            'explained': (('mode',), explained.astype('float64')),
        },
        coords={'mode': [1, 2], 'lon': lon, 'doy': np.arange(1, 366),
                'feature': np.arange(3 * nlon)},
        attrs={
            'descricao': 'EOF combinado RMM (OLR+U850+U200, 15S-15N) — indice MJO Wheeler-Hendon',
            'fonte': 'CPC Blended OLR + NCEP Reanalysis (u850,u200)',
            'periodo_base': f'{ini}-{fim}',
            'lat_band': LAT_BAND,
            'running_days': RUNNING_DAYS,
            'norm_olr': norm['olr'], 'norm_u850': norm['u850'], 'norm_u200': norm['u200'],
        },
    )
    return out


def _apply_convention(eofs: np.ndarray, pcstd: np.ndarray, lon: np.ndarray) -> tuple:
    """Fixa o SINAL dos 2 EOFs pela carga de OLR (convencao WH04, aproximada).

    RMM1+ ~ conveccao (OLR negativo) sobre o Continente Maritimo (~120E); RMM2+ ~ conveccao
    sobre o Hemisferio Oeste/Africa-Oceano Indico (~60-90E). O sinal pode precisar de ajuste
    fino apos comparacao visual com o CPC (ver nota no s36)."""
    nlon = len(lon)
    olr_eof = eofs[:, :nlon]  # parte de OLR de cada modo
    mc = (lon >= 100) & (lon <= 150)       # Continente Maritimo
    io = (lon >= 60) & (lon <= 90)         # Oceano Indico
    if np.nanmean(olr_eof[0, mc]) > 0:     # RMM1+ deve ter OLR NEGATIVO no MC
        eofs[0] = -eofs[0]
    if np.nanmean(olr_eof[1, io]) > 0:     # RMM2+ deve ter OLR NEGATIVO no Indico
        eofs[1] = -eofs[1]
    return eofs, pcstd


def ensure_mjo_eof(force_rebuild: bool = False, ini: int = BASE_INI, fim: int = BASE_FIM) -> xr.Dataset:
    """Retorna o Dataset do EOF RMM (eofs, pcstd, clim_*, norm_*), construindo/cacheando se preciso."""
    path = _cache_path(ini, fim)
    if not force_rebuild and path.exists():
        try:
            return xr.load_dataset(path)
        except Exception:
            logger.warning('Cache do EOF RMM invalido ({}). Reconstruindo.', path.name)
    ds = _build_eof(ini, fim)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    logger.info('EOF RMM salvo em {}', path)
    return ds


# ---------------------------------------------------------------------------
# Projecao: series (time, lon) de obs/previsao -> RMM1, RMM2
# ---------------------------------------------------------------------------
def rmm_from_bands(
    olr_band: xr.DataArray, u850_band: xr.DataArray, u200_band: xr.DataArray,
    eof_ds: xr.Dataset, logger_=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Projeta series diarias (time, lon) ja media-meridional (15S-15N) no EOF RMM.

    Aplica o MESMO pre-processamento do treino: anomalia (clim DOY do cache) -> remove 120d ->
    normaliza -> projeta. Retorna (dates, rmm1, rmm2). `dates` sao as datas com janela de 120d
    completa (as primeiras sao descartadas por NaN no trailing mean).
    """
    lg = logger_ or logger
    lon = eof_ds['lon'].values
    clim = {'olr': eof_ds['clim_olr'].values, 'u850': eof_ds['clim_u850'].values,
            'u200': eof_ds['clim_u200'].values}
    norm = {'olr': float(eof_ds.attrs['norm_olr']), 'u850': float(eof_ds.attrs['norm_u850']),
            'u200': float(eof_ds.attrs['norm_u200'])}
    eofs = eof_ds['eofs'].values          # (2, 3*nlon)
    pcstd = eof_ds['pcstd'].values        # (2,)

    bands = {'olr': _align_to_lon(olr_band, lon), 'u850': _align_to_lon(u850_band, lon),
             'u200': _align_to_lon(u200_band, lon)}
    for v in VARS:
        bands[v] = bands[v].assign_coords(time=pd.DatetimeIndex(
            pd.to_datetime(bands[v]['time'].values)).normalize())
        bands[v] = bands[v].isel(time=np.unique(bands[v]['time'].values, return_index=True)[1])

    # eixo diario CONTINUO cobrindo as 3 series; interpola pequenas lacunas (emenda obs<->fcst)
    t0 = min(pd.DatetimeIndex(bands[v]['time'].values).min() for v in VARS)
    t1 = max(pd.DatetimeIndex(bands[v]['time'].values).max() for v in VARS)
    t_common = pd.date_range(t0, t1, freq='D')
    raw = {}
    for v in VARS:
        da = bands[v].reindex(time=t_common).interpolate_na('time', limit=7)
        raw[v] = da.values

    intra = {}
    for v in VARS:
        a = _anomaly_from_clim(raw[v], t_common, clim[v])
        intra[v] = _remove_running(a) / norm[v]

    valid = np.all([np.isfinite(intra[v]).all(axis=1) for v in VARS], axis=0)
    M = np.concatenate([intra[v] for v in VARS], axis=1)
    rmm = np.full((len(t_common), N_MODES), np.nan)
    rmm[valid] = (M[valid] @ eofs.T) / pcstd
    if not valid.any():
        lg.warning('RMM: nenhuma data com janela de 120 dias completa.')
    return t_common.values[valid], rmm[valid, 0], rmm[valid, 1]


def band_mean_on_grid(da: xr.DataArray) -> xr.DataArray:
    """Helper publico: padroniza lat/lon e tira a media meridional 15S-15N de um campo (time,lat,lon)."""
    return _band_mean(_std_latlon(da))
