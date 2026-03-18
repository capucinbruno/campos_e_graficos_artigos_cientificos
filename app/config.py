"""
Description:

Centraliza a configuração do Dynaconf em um único lugar.

Author:           @Palin
Created:          2022-07-10
Copyright:        (c) Ampere Consultoria Ltda
"""

# Bibliotecas padrão
import logging
from functools import lru_cache
from pathlib import Path
from random import sample

# Bibliotecas de terceiros
from dynaconf import Dynaconf
from fake_headers import Headers

# Módulos locais
from app.common.logger import get_logger

# Evita logs verbosos de bibliotecas externas
logging.getLogger("paramiko").setLevel(logging.ERROR)
logging.getLogger("cdsapi").setLevel(logging.WARNING)

# Logger geral da aplicação (grava em logs/campos_observados.log)
logger = get_logger()

# Geração de headers aleatórios para web scraping
header_factory = Headers(
    browser="chrome",  # Generate only Chrome UA
    os="win",  # Generate ony Windows platform
    headers=True,  # generate misc headers
)

lst_headers = []
total_headers = 50
for i in range(total_headers):
    lst_headers.append(header_factory.generate())

header = sample(lst_headers, 1)[0]


@lru_cache()
def get_settings():
    """
    Carrega configurações do Dynaconf.

    Returns:
        Dynaconf: Objeto de configurações carregadas
    """
    settings = Dynaconf(
        envvar_prefix="AMPERE",
        settings_files=[
            "app/settings/settings.toml",
            "settings.local.toml",
            "app/settings/settings.json",
        ],
        environments=True,
        load_dotenv=True,
    )
    return settings


settings = get_settings()
