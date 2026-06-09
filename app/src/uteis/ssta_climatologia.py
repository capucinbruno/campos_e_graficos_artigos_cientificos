# -*- coding: utf-8 -*-
"""
Anomalia de TSM a partir da climatologia diaria OISST (LTM 1991-2020).

Centraliza a logica usada pelos scripts que plotam anomalia de TSM (s11, s12, s14,
s24, s29). Em vez de baixar o `sst.day.anom.{ano}.nc` ja pronto, baixa-se a SST
absoluta (`sst.day.mean.{ano}.nc`) e subtrai-se a climatologia diaria recortada
no mesmo periodo:

    anomalia = media(SST no periodo) - clim_mean_array(periodo)

A climatologia tem 365 valores (um por dia-do-ano). Para cada dia do periodo,
mapeia (mes, dia) -> indice do dia-do-ano e faz a media PONDERADA pela quantidade
de vezes que cada dia-do-ano aparece no periodo. Assim o resultado reproduz
exatamente a media das anomalias diarias `mean_d[ SST(d) - clim(doy(d)) ]`.
29/fev (ano bissexto) usa 28/fev, pois a climatologia tem 365 dias.

Caminho da climatologia: setting `FILE_CLIMATOLOGIA_SST`
(default `Entrada/sst.day.mean.ltm.1991-2020.nc`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from app.shared.settings_factory import settings

# Climatologia diaria OISST (LTM 1991-2020) — base da anomalia de TSM
SST_CLIM_FILE_DEFAULT = 'Entrada/sst.day.mean.ltm.1991-2020.nc'

# SST absoluta diaria (OISSTv2) — substitui o sst.day.anom nos scripts de anomalia
SST_MEAN_URL_TEMPLATE = (
    'https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.day.mean.{year}.nc'
)
SST_MEAN_FILE_TEMPLATE = 'sst.day.mean.{year}.nc'


def get_clim_path() -> Path:
    """Caminho da climatologia diaria de TSM (configuravel via FILE_CLIMATOLOGIA_SST)."""
    return Path(settings.get('FILE_CLIMATOLOGIA_SST', SST_CLIM_FILE_DEFAULT))


def _build_clim_mean_da(
    start_date: np.datetime64,
    end_date: np.datetime64,
    logger,
    clim_path: Path | None = None,
) -> xr.DataArray:
    """Media da climatologia diaria no periodo, como DataArray (lat, lon) na grade nativa."""
    clim_path = clim_path or get_clim_path()
    if not clim_path.exists():
        raise FileNotFoundError(
            f'Climatologia SST nao encontrada: {clim_path}. '
            f'Configure FILE_CLIMATOLOGIA_SST ou coloque o arquivo em Entrada/.'
        )

    import cftime

    with xr.open_dataset(str(clim_path), decode_times=False) as clim:
        clat = clim['lat'].values
        clon = clim['lon'].values

        # Decodifica o tempo (dias-desde-1800, ano ~1 — fora do range do pandas) via cftime
        tvar = clim['time']
        tdates = cftime.num2date(
            tvar.values, tvar.attrs['units'], tvar.attrs.get('calendar', 'standard')
        )
        idx_by_md = {(int(d.month), int(d.day)): i for i, d in enumerate(tdates)}

        # Conta quantas vezes cada dia-do-ano aparece no periodo
        dates = np.arange(start_date, end_date + np.timedelta64(1, 'D'), dtype='datetime64[D]')
        counts: dict[int, int] = {}
        for dt64 in dates:
            d = dt64.astype(object)  # datetime.date
            md = (d.month, d.day)
            if md == (2, 29) and md not in idx_by_md:
                md = (2, 28)  # climatologia tem 365 dias — 29/fev usa 28/fev
            idx = idx_by_md[md]
            counts[idx] = counts.get(idx, 0) + 1

        # Media ponderada das fatias de climatologia (cada dia-do-ano lido uma vez)
        clim_sst = clim['sst']
        sum_2d = np.zeros((len(clat), len(clon)), dtype=np.float64)
        count_2d = np.zeros_like(sum_2d)
        for idx, cnt in counts.items():
            arr = clim_sst.isel(time=idx).values.astype(np.float64)
            valid = ~np.isnan(arr)
            sum_2d += np.where(valid, arr, 0.0) * cnt
            count_2d += valid * cnt

    with np.errstate(invalid='ignore'):
        mean = np.where(count_2d > 0, sum_2d / count_2d, np.nan)

    total = int(sum(counts.values()))
    logger.info(
        f'Climatologia SST: media sobre {total} dias do periodo '
        f'({len(counts)} dias-do-ano distintos)'
    )
    return xr.DataArray(mean, dims=('lat', 'lon'), coords={'lat': clat, 'lon': clon})


def clim_mean_array(
    start_date: np.datetime64,
    end_date: np.datetime64,
    expected_lat: np.ndarray,
    expected_lon: np.ndarray,
    logger,
    clim_path: Path | None = None,
) -> np.ndarray:
    """
    Media da climatologia no periodo, alinhada a (expected_lat, expected_lon) como numpy 2D.

    Quando a grade da SST coincide exatamente com a da climatologia (caso normal —
    ambos sao OISSTv2 0.25°), retorna os valores diretamente. Se a ordem das
    coordenadas diferir (ex: longitude rolada), reindexa por rotulo (nearest).
    """
    da = _build_clim_mean_da(start_date, end_date, logger, clim_path)
    clat = da['lat'].values
    clon = da['lon'].values
    expected_lat = np.asarray(expected_lat)
    expected_lon = np.asarray(expected_lon)

    if (clat.shape == expected_lat.shape and clon.shape == expected_lon.shape
            and np.allclose(clat, expected_lat) and np.allclose(clon, expected_lon)):
        return da.values

    aligned = da.reindex(lat=expected_lat, lon=expected_lon, method='nearest')
    return aligned.values
