from __future__ import annotations

# Bibliotecas padrão
from pathlib import Path

# Bibliotecas de terceiros
import httpx
from playwright.sync_api import Playwright, sync_playwright

# Módulos locais
from app.shared.logger import get_logger
from app.shared.settings_factory import settings

logger = get_logger(__name__)


def get_clim_psi200_path(data_inicial, data_final) -> Path:
    """Garante e retorna path da climatologia PSL streamfunction 200mb (com cache local)."""
    data_inicial = str(data_inicial)
    data_final = str(data_final)
    with sync_playwright() as playwright:
        return _run_psi200(playwright, data_inicial, data_final)


def _run_psi200(playwright: Playwright, data_inicial: str, data_final: str) -> Path:
    output_dir = Path(settings.DIR_FILE_NC)
    output_dir.mkdir(exist_ok=True, parents=True)

    mes_ini, dia_ini = data_inicial.split('-')[1], data_inicial.split('-')[2]
    mes_fim, dia_fim = data_final.split('-')[1], data_final.split('-')[2]

    # Nome codifica MM-DD para cache cross-year: mesmo período calendário = mesmo arquivo
    file_name = f'clim_psi200_{mes_ini}{dia_ini}_{mes_fim}{dia_fim}.nc'
    file_path = output_dir / file_name

    if file_path.exists():
        logger.info('Climatologia PSL psi200 já existe, pulando download: {}', file_path)
        return file_path

    logger.info('Baixando climatologia PSL psi200 ({}/{} → {}/{})', dia_ini, mes_ini, dia_fim, mes_fim)

    ano = data_final.split('-')[0]
    tipo_var = 2  # 0=media, 1=anomalia, 2=climatologia

    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()
    page.goto('https://psl.noaa.gov/data/composites/day/')
    page.locator('select[name="var"]').select_option('Streamfunction')
    page.locator('select[name="level"]').select_option('.2582 sigma')
    page.locator('select[name="monr1"]').select_option(str(int(mes_ini)))
    page.locator('select[name="monr2"]').select_option(str(int(mes_fim)))
    page.locator('select[name="dayr1"]').select_option(str(int(dia_ini)))
    page.locator('select[name="dayr2"]').select_option(str(int(dia_fim)))
    page.locator('input[name="iyr\\[1\\]"]').fill(ano)
    page.locator('input[name="type"]').nth(tipo_var).check()
    page.get_by_role('button', name='Create Plot').click()

    with page.expect_download() as download_info:
        page.get_by_role('link', name='Get a copy of the netCDF data file used for the plot').click()
    url = download_info.value.url

    context.close()
    browser.close()

    resp = httpx.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f'Falha no download da climatologia PSL psi200: HTTP {resp.status_code}')

    file_path.write_bytes(resp.content)
    logger.info('Climatologia PSL psi200 salva: {}', file_path)
    return file_path
