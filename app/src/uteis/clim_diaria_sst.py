# app/src/uteis/clim_diaria_sst.py
# -*- coding: utf-8 -*-
"""TSM diaria (OISSTv2 0.25°) — SST observada e climatologia diaria (LTM 1991-2020).

Fonte da SST absoluta: sst.day.mean.{ano}.nc (PSL/NOAA, um arquivo por ano).
Climatologia diaria: Entrada/sst.day.mean.ltm.1991-2020.nc (365 dias-do-ano),
caminho configuravel via FILE_CLIMATOLOGIA_SST (reusa ssta_climatologia.get_clim_path).

Para o globo 3D (s38/s39) a grade nativa 0.25° e subamostrada para ~0.5°
(GLOBO_3D_SST_COARSEN, default 2) — controla RAM/tempo mantendo bom detalhe.

  - sst_obs_daily(dt_ini, dt_fim)   -> SST observada (°C) diaria, grade ~0.5°
  - clim_sst_daily_for_anim(dates)  -> climatologia diaria (°C) na MESMA grade
    (assinatura compativel com _anom_from_clim de globo_3d_anim.py)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from app.common.dataset_utils import arquivo_cobre_periodo
from app.common.download_helper import DownloadEngine, download_with_progress
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.src.uteis.ssta_climatologia import get_clim_path

logger = get_logger(__name__)

SST_MEAN_URL_TEMPLATE = (
    'https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.day.mean.{year}.nc'
)
SST_MEAN_FILE_TEMPLATE = 'sst.day.mean.{year}.nc'


def _coarsen_factor() -> int:
    """Fator de subamostragem da grade OISST 0.25° p/ o globo (default 2 -> ~0.5°)."""
    return max(1, int(settings.get('GLOBO_3D_SST_COARSEN', 2)))


def _coarsen(da: xr.DataArray) -> xr.DataArray:
    f = _coarsen_factor()
    if f <= 1:
        return da
    return da.coarsen(lat=f, lon=f, boundary='trim').mean(skipna=True)


def _sst_dir() -> Path:
    d = Path(settings.DIR_DADOS) / 'OISST_SST'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download_sst_years(dt_ini: datetime, dt_fim: datetime) -> list[Path]:
    """Baixa os arquivos anuais OISST (sst.day.mean.{ano}) cobrindo [dt_ini, dt_fim]."""
    dados_dir = _sst_dir()
    current_year = datetime.now().year
    start_d = np.datetime64(dt_ini.date(), 'D')
    end_d = np.datetime64(dt_fim.date(), 'D')
    paths: list[Path] = []
    for year in range(dt_ini.year, dt_fim.year + 1):
        url = SST_MEAN_URL_TEMPLATE.format(year=year)
        path = dados_dir / SST_MEAN_FILE_TEMPLATE.format(year=year)
        if year < current_year and path.exists():
            paths.append(path)
            continue
        ys = max(np.datetime64(f'{year}-01-01', 'D'), start_d)
        ye = min(end_d, np.datetime64(f'{year}-12-31', 'D'))
        if arquivo_cobre_periodo(path, ys, ye):
            logger.info('OISST {} ja cobre o periodo — pulando download', year)
            paths.append(path)
            continue
        download_with_progress(
            url=url, output_path=str(path), description=f'OISST SST media {year}',
            max_retries=5, force=path.exists(), prefer_ftp=False,
            engine=DownloadEngine.AUTO, timeout=600,
        )
        paths.append(path)
    return paths


def _standardize(da: xr.DataArray) -> xr.DataArray:
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


def sst_obs_daily(dt_ini: datetime, dt_fim: datetime) -> xr.DataArray:
    """SST observada (°C) diaria no periodo, grade ~0.5° (OISST 0.25° subamostrada).

    lat ascendente, lon 0..360. Land = NaN (mascarado no globo)."""
    paths = _download_sst_years(dt_ini, dt_fim)
    das = []
    for p in paths:
        ds = xr.open_dataset(str(p))
        das.append(ds['sst'] if 'sst' in ds.data_vars else next(iter(ds.data_vars.values())))
    da = xr.concat(das, dim='time') if len(das) > 1 else das[0]
    da = _standardize(da)
    da = da.sel(time=slice(np.datetime64(dt_ini.date()), np.datetime64(dt_fim.date())))
    da = da.assign_coords(time=da['time'].dt.floor('D'))
    return _coarsen(da).load()


def clim_sst_daily_for_anim(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Climatologia diaria de TSM (OISST LTM 1991-2020) na grade ~0.5°. Retorna (arr, lat, lon).

    Assinatura compativel com _anom_from_clim de globo_3d_anim.py. Le o arquivo LTM
    (365 dias-do-ano; 29/fev usa 28/fev) por (mes, dia) e subamostra igual a SST observada."""
    import cftime

    clim_path = get_clim_path()
    if not clim_path.exists():
        raise FileNotFoundError(
            f'Climatologia SST nao encontrada: {clim_path}. '
            f'Configure FILE_CLIMATOLOGIA_SST ou coloque o arquivo em Entrada/.')
    with xr.open_dataset(str(clim_path), decode_times=False) as clim:
        tvar = clim['time']
        tdates = cftime.num2date(
            tvar.values, tvar.attrs['units'], tvar.attrs.get('calendar', 'standard'))
        idx_by_md = {(int(d.month), int(d.day)): i for i, d in enumerate(tdates)}
        sst = clim['sst']
        slices = []
        for dt64 in dates:
            d = pd.Timestamp(dt64)
            md = (int(d.month), int(d.day))
            if md == (2, 29) and md not in idx_by_md:
                md = (2, 28)  # climatologia tem 365 dias
            slices.append(sst.isel(time=idx_by_md[md]))
        clim_da = xr.concat(slices, dim='t')
    clim_da = _standardize(clim_da)
    clim_da = _coarsen(clim_da)
    return clim_da.values, clim_da['lat'].values, clim_da['lon'].values
