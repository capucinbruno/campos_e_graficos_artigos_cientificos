---
description: Convencoes de estilo de codigo do projeto campos_e_graficos_artigos_cientificos
paths: ["app/**/*.py", "artigos/**/*.py", "tests/**/*.py"]
---

# Convencoes de Estilo - Campos e Graficos para Artigos Cientificos

## Imports

Ordem obrigatoria (isort profile=black):

```python
# Bibliotecas padrão
import os
import sys
from datetime import datetime
from pathlib import Path

# Bibliotecas de terceiros
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from loguru import logger

# Módulos locais
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.common.cache_manager import check_cache_valid, save_cache_metadata
```

- Prefixo canonico: `app.` para todos os modulos locais
- isort config: `profile = "black"`, `known_first_party = ["app"]`

## Nomenclatura

- **Modulos**: snake_case (`sNN_descricao`, `downloaders_<variavel>_ERA5`)
- **Classes**: PascalCase (`SFTPClient`, `Singleton`)
- **Funcoes/metodos**: snake_case (`get_logger`, `ensure_era5_<variavel>_for_period`)
- **Constantes**: UPPER_SNAKE_CASE (`DIR_OUTPUT`, `DIR_DADOS`)
- **Variaveis de dominio**: portugues (`data_inicial`, `data_final`, `anomalia_clim_file`)
- **Variaveis tecnicas**: ingles (`logger`, `settings`, `cache_params`, `output_dir`)

## Formatacao

- **Ruff format**: single quotes, preview mode, line-length=105
- **Indentacao**: 4 espacos
- **Strings**: aspas simples (enforced pelo ruff)
- **f-strings**: preferidas para interpolacao

## Type Hints

- **Funcoes publicas**: obrigatorios em parametros e retorno
- **Funcoes internas**: recomendados mas opcionais
- Usar `from __future__ import annotations` para sintaxe moderna (`list[str]` em vez de `List[str]`)

## Docstrings

- Portugues para funcoes de dominio meteorologico
- Ingles para infraestrutura e utilitarios
- Formato:

```python
def main():
    """Entry point - chamado pelo CLI sem argumentos."""
```

## Tratamento de Erro

**NAO** usar try/except generico nos scripts. O decorator `@friendly_errors` no entry point captura tudo:

```python
# CERTO - deixa subir
ds = xr.open_dataset("arquivo.nc")

# CERTO - adiciona contexto e re-lanca
try:
    ds = xr.open_dataset(path)
except OSError:
    raise RuntimeError(f"Falha ao abrir climatologia: {path}") from None

# ERRADO - engole o erro
try:
    ds = xr.open_dataset("arquivo.nc")
except Exception as e:
    print(f"Erro: {e}")
    return
```

Para erros especificos, adicione em `_ERROR_HINTS` no `app/shared/error_handler.py`.

## Logging

```python
from app.shared.logger import get_logger

logger = get_logger(__name__)  # ou get_logger("s00") para scripts

logger.info("Mensagem informativa")
logger.warning("Atencao")
logger.error("Erro grave")
logger.info("=" * 80)  # separador visual
```

## Patterns

```python
# Singleton via metaclass
from app.shared.singleton import Singleton
class MyService(metaclass=Singleton): ...

# Context Manager para SFTP
from app.shared.sftp_client import SFTPClient
with SFTPClient() as sftp:
    sftp.download(remote, local)

# Settings access (dot notation)
from app.shared.settings_factory import settings
settings.DIR_OUTPUT
settings.DATA_INICIAL
settings.get("RUN_S00", True)  # com default
```
