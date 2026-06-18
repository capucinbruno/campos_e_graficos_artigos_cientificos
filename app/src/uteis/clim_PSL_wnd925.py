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


def get_clim_wnd925_paths(data_inicial, data_final) -> tuple[Path, Path]:
    """Garante e retorna (path_u, path_v) da climatologia PSL vento 925mb (com cache local)."""
    data_inicial = str(data_inicial)
    data_final = str(data_final)
    with sync_playwright() as playwright:
        return _run_wnd925(playwright, data_inicial, data_final)


def _run_wnd925(playwright: Playwright, data_inicial: str, data_final: str) -> tuple[Path, Path]:
    output_dir = Path(settings.DIR_FILE_NC)
    output_dir.mkdir(exist_ok=True, parents=True)

    mes_ini, dia_ini = data_inicial.split('-')[1], data_inicial.split('-')[2]
    mes_fim, dia_fim = data_final.split('-')[1], data_final.split('-')[2]

    # Nome codifica MM-DD para cache cross-year: mesmo período calendário = mesmo arquivo
    file_u = output_dir / f'clim_u925_{mes_ini}{dia_ini}_{mes_fim}{dia_fim}.nc'
    file_v = output_dir / f'clim_v925_{mes_ini}{dia_ini}_{mes_fim}{dia_fim}.nc'

    if file_u.exists() and file_v.exists():
        logger.info('Climatologia PSL u/v 925mb já existe, pulando download: {}', output_dir)
        return file_u, file_v

    logger.info('Baixando climatologia PSL u/v 925mb ({}/{} → {}/{})', dia_ini, mes_ini, dia_fim, mes_fim)

    ano = data_final.split('-')[0]
    tipo_var = 2  # 0=media, 1=anomalia, 2=climatologia

    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()
    page.goto('https://psl.noaa.gov/data/composites/day/')
    page.locator('select[name="var"]').select_option('Vector Wind')
    page.locator('select[name="level"]').select_option('925mb')
    page.locator('select[name="monr1"]').select_option(str(int(mes_ini)))
    page.locator('select[name="monr2"]').select_option(str(int(mes_fim)))
    page.locator('select[name="dayr1"]').select_option(str(int(dia_ini)))
    page.locator('select[name="dayr2"]').select_option(str(int(dia_fim)))
    page.locator('input[name="iyr\\[1\\]"]').fill(ano)
    page.locator('input[name="type"]').nth(tipo_var).check()
    page.get_by_role('button', name='Create Plot').click()

    # Obter URLs antes de fechar o browser
    with page.expect_download() as dl_u_info:
        page.get_by_role('link', name='Get a copy of the u wind').click()
    url_u = dl_u_info.value.url

    with page.expect_download() as dl_v_info:
        page.get_by_role('link', name='Get a copy of the v wind').click()
    url_v = dl_v_info.value.url

    context.close()
    browser.close()

    for url, path, comp in [(url_u, file_u, 'u'), (url_v, file_v, 'v')]:
        resp = httpx.get(url)
        if resp.status_code != 200:
            raise RuntimeError(f'Falha no download da climatologia PSL {comp}925: HTTP {resp.status_code}')
        path.write_bytes(resp.content)
        logger.info('Climatologia {} 925mb salva: {}', comp, path)

    return file_u, file_v
