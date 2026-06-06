from __future__ import annotations

from pathlib import Path

import httpx
from playwright.sync_api import Playwright, sync_playwright

from app.shared.logger import get_logger
from app.shared.settings_factory import settings

logger = get_logger(__name__)


def get_clim_wnd_meridional_250_path(data_inicial, data_final) -> Path:
    """Garante e retorna path da climatologia PSL de vento meridional (v) em 250mb (com cache local)."""
    data_inicial = str(data_inicial)
    data_final = str(data_final)
    with sync_playwright() as playwright:
        return _run_wnd_meridional_250(playwright, data_inicial, data_final)


def _run_wnd_meridional_250(playwright: Playwright, data_inicial: str, data_final: str) -> Path:
    output_dir = Path(settings.DIR_FILE_NC)
    output_dir.mkdir(exist_ok=True, parents=True)

    mes_ini, dia_ini = data_inicial.split('-')[1], data_inicial.split('-')[2]
    mes_fim, dia_fim = data_final.split('-')[1], data_final.split('-')[2]

    # Nome codifica MM-DD para cache cross-year: mesmo período calendário = mesmo arquivo
    file_v = output_dir / f'clim_vmeridional250_{mes_ini}{dia_ini}_{mes_fim}{dia_fim}.nc'

    if file_v.exists():
        logger.info('Climatologia PSL v-meridional 250mb já existe, pulando download: {}', file_v)
        return file_v

    logger.info('Baixando climatologia PSL v-meridional 250mb ({}/{} → {}/{})', dia_ini, mes_ini, dia_fim, mes_fim)

    ano = data_final.split('-')[0]
    tipo_var = 2  # 0=media, 1=anomalia, 2=climatologia

    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()
    page.goto('https://psl.noaa.gov/data/composites/day/')
    page.locator('select[name="var"]').select_option('Meridional Wind')
    page.locator('select[name="level"]').select_option('250mb')
    page.locator('select[name="monr1"]').select_option(str(int(mes_ini)))
    page.locator('select[name="monr2"]').select_option(str(int(mes_fim)))
    page.locator('select[name="dayr1"]').select_option(str(int(dia_ini)))
    page.locator('select[name="dayr2"]').select_option(str(int(dia_fim)))
    page.locator('input[name="iyr\\[1\\]"]').fill(ano)
    page.locator('input[name="type"]').nth(tipo_var).check()
    page.get_by_role('button', name='Create Plot').click()

    with page.expect_download() as dl_info:
        page.get_by_role('link', name='Get a copy of the netCDF data file used for the plot').click()
    url_v = dl_info.value.url

    context.close()
    browser.close()

    resp = httpx.get(url_v)
    if resp.status_code != 200:
        raise RuntimeError(f'Falha no download da climatologia PSL v-meridional 250mb: HTTP {resp.status_code}')
    file_v.write_bytes(resp.content)
    logger.info('Climatologia v-meridional 250mb salva: {}', file_v)

    return file_v
