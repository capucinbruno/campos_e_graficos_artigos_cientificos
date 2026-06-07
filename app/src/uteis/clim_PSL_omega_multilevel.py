"""Downloader de climatologia PSL de velocidade vertical (omega) para múltiplos níveis.

Usa PSL Daily Composites (NCEP/NCAR R1) para omega em qualquer nível entre 100 e 850 hPa.

Fluxo:
  1. Por nível: baixa um .nc temporário via Playwright (PSL só permite um nível por vez)
  2. Mescla todos os níveis em um único arquivo com dim 'level'

Cache primário (reutilizado entre scripts):
  Entrada/arquivos_nc/clim_omega_ML_{100-250-300-500-850}_{MM}{DD}_{MM}{DD}.nc

Cache intermediário (um por nível):
  Entrada/arquivos_nc/clim_omega_{level}hpa_{MM}{DD}_{MM}{DD}.nc

Nota: a climatologia é do NCEP R1 — combinar com ERA5 introduz viés sistemático.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import httpx
import xarray as xr
from playwright.sync_api import Browser, sync_playwright

from app.shared.logger import get_logger
from app.shared.settings_factory import settings

logger = get_logger(__name__)

DEFAULT_LEVELS: List[int] = [100, 250, 300, 500, 850]

_LEVEL_LABEL: Dict[int, str] = {
    100: '100mb', 150: '150mb', 200: '200mb',
    250: '250mb', 300: '300mb', 400: '400mb',
    500: '500mb', 700: '700mb', 850: '850mb', 925: '925mb',
}


def get_clim_omega_multilevel(
    data_inicial: str | object,
    data_final: str | object,
    levels_hpa: List[int] | None = None,
) -> Path:
    """Retorna Path para netCDF com todos os níveis de omega PSL (dim 'level').

    Se o arquivo multi-nível já existir, retorna imediatamente sem acessar a rede.
    Caso contrário, baixa cada nível via Playwright (uma sessão, N páginas) e
    mescla em um único arquivo.
    """
    if levels_hpa is None:
        levels_hpa = DEFAULT_LEVELS

    data_inicial, data_final = str(data_inicial), str(data_final)

    for lev in levels_hpa:
        if lev not in _LEVEL_LABEL:
            raise ValueError(f'Nível {lev} hPa inválido. Disponíveis: {sorted(_LEVEL_LABEL)}')

    out_dir = Path(settings.DIR_FILE_NC)
    out_dir.mkdir(parents=True, exist_ok=True)

    mes_ini, dia_ini = data_inicial.split('-')[1], data_inicial.split('-')[2]
    mes_fim, dia_fim = data_final.split('-')[1], data_final.split('-')[2]
    ano = data_final.split('-')[0]

    levels_tag = '-'.join(str(l) for l in sorted(levels_hpa))
    merged_path = out_dir / f'clim_omega_ML_{levels_tag}_{mes_ini}{dia_ini}_{mes_fim}{dia_fim}.nc'

    if merged_path.exists():
        logger.info('Climatologia omega multi-nível já existe: {}', merged_path.name)
        return merged_path

    # Baixar níveis que ainda não têm arquivo individual
    level_paths = _ensure_individual_levels(
        data_inicial, data_final, levels_hpa, out_dir, mes_ini, dia_ini, mes_fim, dia_fim, ano
    )

    # Mesclar todos os níveis em um único netCDF
    logger.info('Mesclando {} níveis em um único arquivo ...', len(levels_hpa))
    das = []
    for lev in sorted(levels_hpa):
        ds = xr.open_dataset(str(level_paths[lev]), engine='netcdf4')
        da = _extract_omega_da(ds)
        da = da.expand_dims({'level': [lev]})
        das.append(da)
        ds.close()

    da_merged = xr.concat(das, dim='level')
    da_merged.name = 'omega'
    xr.Dataset({'omega': da_merged}).to_netcdf(str(merged_path))

    logger.info('Omega multi-nível salvo: {}', merged_path.name)
    return merged_path


# ---------------------------------------------------------------------------
# Funções auxiliares (internas)
# ---------------------------------------------------------------------------
def _ensure_individual_levels(
    data_inicial: str,
    data_final: str,
    levels_hpa: List[int],
    out_dir: Path,
    mes_ini: str, dia_ini: str,
    mes_fim: str, dia_fim: str,
    ano: str,
) -> Dict[int, Path]:
    """Garante que cada nível tem seu arquivo individual. Retorna {level: Path}."""
    paths: Dict[int, Path] = {}
    pending: List[int] = []

    for lev in levels_hpa:
        fname = out_dir / f'clim_omega_{lev}hpa_{mes_ini}{dia_ini}_{mes_fim}{dia_fim}.nc'
        if fname.exists():
            logger.info('  [SKIP] omega {}hPa: {}', lev, fname.name)
            paths[lev] = fname
        else:
            pending.append(lev)

    if not pending:
        return paths

    logger.info('Baixando climatologia PSL omega — {} níveis via Playwright ...', len(pending))

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        for lev in pending:
            fname = out_dir / f'clim_omega_{lev}hpa_{mes_ini}{dia_ini}_{mes_fim}{dia_fim}.nc'
            paths[lev] = _fetch_one_level(browser, lev, mes_ini, dia_ini, mes_fim, dia_fim, ano, fname)
        browser.close()

    return paths


def _fetch_one_level(
    browser: Browser,
    level_hpa: int,
    mes_ini: str, dia_ini: str,
    mes_fim: str, dia_fim: str,
    ano: str,
    out_path: Path,
) -> Path:
    context = browser.new_context()
    page = context.new_page()
    page.goto('https://psl.noaa.gov/data/composites/day/')

    page.locator('select[name="var"]').select_option('Omega (to 100mb only)')
    page.locator('select[name="level"]').select_option(_LEVEL_LABEL[level_hpa])
    page.locator('select[name="monr1"]').select_option(str(int(mes_ini)))
    page.locator('select[name="monr2"]').select_option(str(int(mes_fim)))
    page.locator('select[name="dayr1"]').select_option(str(int(dia_ini)))
    page.locator('select[name="dayr2"]').select_option(str(int(dia_fim)))
    page.locator('input[name="iyr\\[1\\]"]').fill(ano)
    page.locator('input[name="type"]').nth(2).check()  # 2 = climatologia
    page.get_by_role('button', name='Create Plot').click()

    with page.expect_download() as dl_info:
        page.get_by_role('link', name='Get a copy of the netCDF data file used for the plot').click()
    url = dl_info.value.url
    context.close()

    resp = httpx.get(url, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(
            f'Falha no download da climatologia PSL omega {level_hpa}hPa: HTTP {resp.status_code}'
        )
    out_path.write_bytes(resp.content)
    logger.info('  [OK] omega {}hPa: {}', level_hpa, out_path.name)
    return out_path


def _extract_omega_da(ds: xr.Dataset) -> xr.DataArray:
    """Extrai a variável omega de um dataset PSL, normalizando nomes de dims."""
    for vname in ('omega', 'air', 'w', 'var39'):
        if vname in ds.data_vars:
            da = ds[vname]
            break
    else:
        da = next(iter(ds.data_vars.values()))

    # Remove dims singleton extras (time, level)
    for dim in ('time', 'level', 'isobaricInhPa', 'pressure_level'):
        if dim in da.dims and da.sizes[dim] == 1:
            da = da.isel({dim: 0}, drop=True)

    rename = {}
    for n in list(da.dims):
        if n.lower() == 'latitude' and 'lat' not in da.dims:
            rename[n] = 'lat'
        elif n.lower() == 'longitude' and 'lon' not in da.dims:
            rename[n] = 'lon'
    if rename:
        da = da.rename(rename)

    return da
