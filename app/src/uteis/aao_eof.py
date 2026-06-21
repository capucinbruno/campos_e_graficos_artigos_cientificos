# -*- coding: utf-8 -*-
"""
Padrao de carga (EOF1) e indice da AAO (Antarctic Oscillation), metodologia CPC.

Replica o metodo do CPC (https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/
history/method.shtml) com dados PROPRIOS:

- EOF1 das anomalias MENSAIS de altura geopotencial em 700 hPa no HS (poleward de 20S),
  ERA5 1991-2020, com peso de area `sqrt(cos(lat))` (matriz de covariancia via SVD).
- Indice diario = projecao das anomalias DIARIAS de Z700 no EOF1, normalizada pelo
  desvio-padrao do indice MENSAL do periodo base.

O padrao e construido na MESMA grade da climatologia diaria NCEP (2.5, `clim_hgt700_daily`),
subconjunto do HS (lat <= -20, ascendente), garantindo alinhamento perfeito com as anomalias
diarias calculadas no s35.

Cacheado em Entrada/arquivos_nc/aao_eof_era5_1991_2020.nc (construido 1x).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.src.uteis.clim_diaria_uv200_ltm import clim_hgt700_daily
from app.src.uteis.downloaders_era5_hgt700_monthly import ensure_era5_hgt700_monthly_sh

logger = get_logger('aao_eof')

G = 9.80665
GEOP_THRESHOLD = 20000.0  # acima disso o campo esta em geopotencial (m^2/s^2) -> dividir por g
LAT_NORTH = -20.0  # poleward de 20S
BASE_INI, BASE_FIM = 1991, 2020


def _cache_path() -> Path:
    return Path(settings.DIR_INPUT) / 'arquivos_nc' / f'aao_eof_era5_{BASE_INI}_{BASE_FIM}.nc'


def _to_geop_height(arr: np.ndarray) -> np.ndarray:
    """Geopotencial (m^2/s^2) -> altura (m), decidindo POR DIA (slice de tempo).

    O observado MISTURA fontes (ERA5 em m^2/s^2 + GDAS ja em m no mesmo array); por isso a decisao
    de dividir por g e feita por slice de tempo — uma media global corromperia os dias da outra
    fonte (mesma licao do s34)."""
    arr = np.asarray(arr, dtype='float64')
    if arr.ndim == 3:  # (time, lat, lon): normaliza cada dia conforme a propria escala
        m = np.nanmean(np.abs(arr), axis=(1, 2))
        factor = np.where(m > GEOP_THRESHOLD, 1.0 / G, 1.0)
        return arr * factor[:, None, None]
    return arr / G if np.nanmean(np.abs(arr)) > GEOP_THRESHOLD else arr


def _sh_grid():
    """Grade do HS (lat<=-20, ascendente; lon 0..360) identica a da climatologia diaria NCEP."""
    probe = np.array([np.datetime64('2020-01-15')])
    _, lat_raw, lon = clim_hgt700_daily(probe)
    order = np.argsort(lat_raw)
    lat_full = lat_raw[order]
    mask = lat_full <= LAT_NORTH
    return lat_full[mask], np.asarray(lon), order, mask


def _weights(lat_sh: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Peso de area sqrt(cos(lat)) replicado em (lat, lon)."""
    w_lat = np.sqrt(np.clip(np.cos(np.deg2rad(lat_sh)), 0.0, None))
    return np.repeat(w_lat[:, None], len(lon), axis=1)


def _build_loading_pattern() -> xr.Dataset:
    """Constroi o EOF1 (espaco ponderado, vetor unitario) + pesos + norm_std a partir do ERA5 mensal."""
    lat_sh, lon, _, _ = _sh_grid()
    w = _weights(lat_sh, lon)  # (lat, lon)

    # ERA5 mensal z700 HS -> altura, regrade p/ a grade SH da LTM, anomalias mensais (sem ciclo sazonal)
    monthly_path = ensure_era5_hgt700_monthly_sh(BASE_INI, BASE_FIM)
    ds = xr.open_dataset(monthly_path, engine='netcdf4')
    try:
        ren = {}
        for n in list(ds.dims) + list(ds.coords):
            low = n.lower()
            if low in ('latitude',) and 'lat' not in ds.dims:
                ren[n] = 'lat'
            elif low in ('longitude',) and 'lon' not in ds.dims:
                ren[n] = 'lon'
            elif low in ('valid_time',) and 'time' not in ds.dims:
                ren[n] = 'time'
        ds = ds.rename(ren) if ren else ds
        zname = next((v for v in ('z', 'geopotential') if v in ds.data_vars), None)
        if zname is None:
            raise KeyError(f'z/geopotential nao encontrado em {monthly_path}')
        da = ds[zname]
        for d in ('pressure_level', 'isobaricInhPa', 'level'):
            if d in da.dims:
                da = da.isel({d: 0}, drop=True)
        da = da.assign_coords(lon=(da['lon'] % 360)).sortby('lon').sortby('lat')
        da = da.interp(lat=xr.DataArray(lat_sh, dims=['lat']),
                       lon=xr.DataArray(lon, dims=['lon']), method='linear')
        hgt = _to_geop_height(da.values)  # (time, lat, lon) em metros
        months = da['time'].dt.month.values
    finally:
        ds.close()

    # Remover ciclo sazonal: subtrair a media de cada mes-calendario (1991-2020)
    anom = np.empty_like(hgt)
    for m in range(1, 13):
        sel = months == m
        anom[sel] = hgt[sel] - np.nanmean(hgt[sel], axis=0, keepdims=True)

    nt = anom.shape[0]
    A = (anom * w[None, :, :]).reshape(nt, -1)  # espaco ponderado (time x pixel)
    A = np.nan_to_num(A, nan=0.0)

    # EOF via SVD: A = U S Vt ; EOF1 = Vt[0] (unitario), PC1 = U[:,0]*S[0]
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    eof1 = Vt[0]                      # (pixel,) vetor unitario no espaco ponderado
    pc1 = U[:, 0] * S[0]             # indice mensal (nao normalizado)
    explained = float(S[0] ** 2 / np.sum(S ** 2))

    eof1_2d = eof1.reshape(len(lat_sh), len(lon))

    # Convencao de sinal: AAO+ => anomalia de altura NEGATIVA sobre a calota polar.
    band = (lat_sh <= -65) & (lat_sh >= -85)  # evita exatamente -90 (cos~0)
    pole_mean = float(np.nanmean(eof1_2d[band, :])) if band.any() else float(np.nanmean(eof1_2d))
    if pole_mean > 0:
        eof1_2d = -eof1_2d
        pc1 = -pc1

    norm_std = float(np.std(pc1, ddof=1))
    logger.info(
        'EOF1 AAO construido: variancia explicada {:.1f}% | norm_std={:.2f} | grade {}x{} ({} meses)',
        explained * 100, norm_std, len(lat_sh), len(lon), nt,
    )

    out = xr.Dataset(
        {
            'eof1': (('lat', 'lon'), eof1_2d.astype('float64')),
            'weights': (('lat', 'lon'), w.astype('float64')),
        },
        coords={'lat': lat_sh, 'lon': lon},
        attrs={
            'descricao': 'EOF1 (espaco ponderado, unitario) da anomalia mensal Z700 HS - indice AAO',
            'fonte': 'ERA5 mensal 700 hPa',
            'periodo_base': f'{BASE_INI}-{BASE_FIM}',
            'norm_std': norm_std,
            'explained_var': explained,
            'lat_north': LAT_NORTH,
        },
    )
    return out


def ensure_aao_loading_pattern(force_rebuild: bool = False) -> xr.Dataset:
    """Retorna o Dataset do EOF1 (eof1, weights, attrs.norm_std), construindo/cacheando se preciso."""
    path = _cache_path()
    if not force_rebuild and path.exists():
        try:
            return xr.load_dataset(path)
        except Exception:
            logger.warning('Cache do EOF AAO invalido ({}). Reconstruindo.', path.name)
    ds = _build_loading_pattern()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    logger.info('EOF1 AAO salvo em {}', path)
    return ds


def project_anomaly_to_index(anom_sh: np.ndarray, eof_ds: xr.Dataset) -> np.ndarray:
    """Projeta anomalias diarias de Z700 (HS, grade do EOF) no EOF1 -> indice AAO normalizado.

    `anom_sh`: (time, lat, lon) na MESMA grade SH do EOF (lat<=-20 ascendente, lon 0..360).
    Retorna array (time,) do indice (adimensional, normalizado pelo std mensal do periodo base).
    """
    eof1 = eof_ds['eof1'].values          # (lat, lon)
    w = eof_ds['weights'].values          # (lat, lon)
    norm_std = float(eof_ds.attrs['norm_std'])
    anom_sh = np.asarray(anom_sh, dtype='float64')
    if anom_sh.ndim == 2:
        anom_sh = anom_sh[None, :, :]
    proj = np.nansum(anom_sh * w[None, :, :] * eof1[None, :, :], axis=(1, 2))
    return proj / norm_std


def aao_index_from_height(height_da: xr.DataArray, logger_=None) -> tuple[np.ndarray, np.ndarray]:
    """Calcula a serie do indice AAO a partir de um campo DIARIO de Z700 na grade da LTM.

    `height_da`: (time, lat, lon) na grade COMPLETA da LTM (lat ascendente, lon 0..360),
    como produzido por `daily_scalar_on_grid`. Pode estar em altura (m) ou geopotencial (m^2/s^2).

    Retorna (index (time,), dates (time,)). Forma a anomalia com `clim_hgt700_daily`, recorta o HS
    e projeta no EOF1.
    """
    lg = logger_ or logger
    eof_ds = ensure_aao_loading_pattern()
    lat_eof = eof_ds['lat'].values
    lon_eof = eof_ds['lon'].values

    dates = np.array([np.datetime64(np.datetime64(t, 'D')) for t in height_da['time'].values])
    clim, lat_raw, lon_raw = clim_hgt700_daily(dates)  # (n, lat, lon) grade raw da LTM
    order = np.argsort(lat_raw)
    clim = clim[:, order, :]  # lat ascendente, alinhado a height_da

    hgt = _to_geop_height(height_da.values)  # (n, lat, lon) em metros
    anom_full = hgt - clim                    # grade completa LTM (ascendente)

    lat_full = lat_raw[order]
    mask = np.isin(np.round(lat_full, 5), np.round(lat_eof, 5))
    anom_sh = anom_full[:, mask, :]
    if anom_sh.shape[1] != len(lat_eof):
        lg.warning('AAO: grade do HS ({}) difere do EOF ({}); usando recorte por lat<=-20.',
                   anom_sh.shape[1], len(lat_eof))
        m2 = lat_full <= LAT_NORTH
        anom_sh = anom_full[:, m2, :]

    idx = project_anomaly_to_index(anom_sh, eof_ds)
    return idx, dates
