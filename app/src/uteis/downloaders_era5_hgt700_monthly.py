# -*- coding: utf-8 -*-
"""
Downloader do ERA5 *mensal* de geopotencial em 700 hPa no Hemisferio Sul, usado UMA vez
para construir o padrao de carga (EOF1) do indice AAO (ver app/src/uteis/aao_eof.py).

- Dataset CDS: 'reanalysis-era5-pressure-levels-monthly-means'
  (product_type 'monthly_averaged_reanalysis').
- Variavel: geopotential (z, m^2/s^2) -> a altura (m) sai dividindo por g no aao_eof.
- Periodo base: 1991-2020 (todas as 12 mensalidades).
- Dominio: HS poleward de 20S (area [-20, -180, -90, 180]); grade 2.5 (regridada de novo no
  aao_eof para a grade exata da LTM NCEP, garantindo alinhamento com a climatologia diaria).
- Tudo numa unica requisicao -> um unico NetCDF cacheado em dados/.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cdsapi
import xarray as xr

try:
    from app.shared.settings_factory import settings  # type: ignore

    DIR_DADOS_BASE = Path(settings.DIR_DADOS)
except Exception:
    DIR_DADOS_BASE = Path('dados')

LOGGER = logging.getLogger('ERA5_HGT700_MONTHLY')
if not LOGGER.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'))
    LOGGER.addHandler(_h)
LOGGER.setLevel(logging.INFO)

URL_API_COPERNICUS = 'https://cds.climate.copernicus.eu/api'
try:
    KEY_CDS = settings.KEY_CDS  # type: ignore
except Exception:
    KEY_CDS = os.environ.get('CDSAPI_KEY', '')

DATASET = 'reanalysis-era5-pressure-levels-monthly-means'
DIR_OUT = DIR_DADOS_BASE / 'ERA5_HGT700_MONTHLY'

BASE_YEAR_INI = 1991
BASE_YEAR_FIM = 2020
PRESSURE_LEVEL = '700'
AREA_SH = [-20, -180, -90, 180]  # N, W, S, E (poleward de 20S)
GRID = [2.5, 2.5]
MIN_BYTES = 50_000


def _get_cds_client() -> cdsapi.Client:
    url = os.environ.get('CDSAPI_URL', URL_API_COPERNICUS)
    key = os.environ.get('CDSAPI_KEY', KEY_CDS)
    return cdsapi.Client(url=url, key=key, debug=False, progress=True,
                         retry_max=8, sleep_max=120, delete=False)


def _file_ok(path: Path) -> bool:
    try:
        if path.stat().st_size < MIN_BYTES:
            return False
        with xr.open_dataset(path, engine='netcdf4') as ds:
            return any(v in ds.variables for v in ('z', 'geopotential'))
    except Exception:
        return False


def ensure_era5_hgt700_monthly_sh(
    year_ini: int = BASE_YEAR_INI,
    year_fim: int = BASE_YEAR_FIM,
    force_redownload: bool = False,
) -> Path:
    """Garante o NetCDF mensal de z700 (HS, 1991-2020) e retorna o caminho.

    Cacheado: se o arquivo existir e for valido, nao rebaixa.
    """
    DIR_OUT.mkdir(parents=True, exist_ok=True)
    out = DIR_OUT / f'era5_z700_monthly_{year_ini}_{year_fim}_sh25.nc'
    if not force_redownload and _file_ok(out):
        LOGGER.info('z700 mensal HS ja existe e e valido: %s', out.name)
        return out

    request = {
        'product_type': ['monthly_averaged_reanalysis'],
        'variable': ['geopotential'],
        'pressure_level': [PRESSURE_LEVEL],
        'year': [f'{y:04d}' for y in range(year_ini, year_fim + 1)],
        'month': [f'{m:02d}' for m in range(1, 13)],
        'time': ['00:00'],
        'area': AREA_SH,
        'grid': GRID,
        'data_format': 'netcdf',
        'download_format': 'unarchived',
    }
    tmp = out.with_suffix('.nc.part')
    if tmp.exists():
        tmp.unlink()
    LOGGER.info('Baixando ERA5 z700 mensal HS %d-%d (todas as mensalidades) -> %s',
                year_ini, year_fim, out.name)
    client = _get_cds_client()
    client.retrieve(DATASET, request, str(tmp))
    if tmp.stat().st_size < MIN_BYTES:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f'[ERA5_HGT700_MONTHLY] Download invalido (muito pequeno): {tmp}')
    out.unlink(missing_ok=True)
    tmp.rename(out)
    LOGGER.info('[OK] z700 mensal HS salvo: %s', out)
    return out
