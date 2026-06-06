import httpx
from playwright.sync_api import Playwright, sync_playwright
from pathlib import Path
from app.config import settings
import os

def main():
    data_inicial = settings.DATA_INICIAL
    data_final = settings.DATA_FINAL
    with sync_playwright() as playwright:
        run_omega(playwright, data_inicial, data_final)


def run_omega(playwright: Playwright, data_inicial: str, data_final: str) -> None:
    DIR_OUTPUT = "Entrada/arquivos_nc"
    Path(DIR_OUTPUT).mkdir(exist_ok=True, parents=True)

    mes_inicial = str(int(data_inicial.split("-")[1]))
    mes_final = str(int(data_final.split("-")[1]))

    dia_inicial = str(int(data_inicial.split("-")[2]))
    dia_final = str(int(data_final.split("-")[2]))

    ano = str(int(data_final.split("-")[0]))

    tipo_var = 2 #(0=media, 1=anomalia, 2=climatologia)
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://psl.noaa.gov/data/composites/day/")
    page.locator('select[name="var"]').select_option("Omega (to 100mb only)")
    page.locator("select[name=\"level\"]").select_option("500mb")
    page.locator('select[name="monr1"]').select_option(mes_inicial)
    page.locator('select[name="monr2"]').select_option(mes_final)
    page.locator('select[name="dayr1"]').select_option(dia_inicial)
    page.locator('select[name="dayr2"]').select_option(dia_final)
    page.locator('input[name="iyr\\[1\\]"]').fill(ano)
    page.locator('input[name="type"]').nth(tipo_var).check()
    page.get_by_role("button", name="Create Plot").click()

    with page.expect_download() as download01_info:
        page.get_by_role("link", name="Get a copy of the netCDF data file used for the plot").click()
    download01 = download01_info.value

    url_request = (download01.url, f"{DIR_OUTPUT}/omega500.nc"),

    file_name = f"omega500.nc"
    file_path = os.path.join(DIR_OUTPUT, file_name)

    # ---------------------
    context.close()
    browser.close()

    url_request = download01.url
    filename_output = os.path.join(DIR_OUTPUT, file_name)
    response = httpx.get(url_request)
    if response.status_code != 200:
        raise Exception("Falha no download")
    with open(filename_output, "wb") as f:
        f.write(response.content)
    print(f"Sucesso: {url_request}")


if __name__ == "__main__":
    main()
