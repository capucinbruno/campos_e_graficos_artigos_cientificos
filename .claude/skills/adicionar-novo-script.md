---
name: adicionar-novo-script
description: Guia passo a passo para adicionar um novo script meteorologico a um artigo do projeto campos_e_graficos_artigos_cientificos, cobrindo registro no CLI, criacao do script, settings e arquivos de suporte.
---

# Adicionar Novo Script Meteorologico

Skill para adicionar um novo script a um artigo do projeto `campos_e_graficos_artigos_cientificos`.
Cada artigo cientifico tem sua propria pasta em `artigos/` (ex: `artigos/artigo_JBN_AS_17_07_2026/`).
Siga os passos na ordem — cada um depende do anterior.

## Checklist

1. Escolher o artigo (pasta em `artigos/`) e o identificador (sNN) e descricao
2. Criar `artigos/<artigo>/__init__.py` (vazio), se for o primeiro script do artigo
3. Criar o script em `artigos/<artigo>/sNN_descricao.py` com funcao `main()`
4. Criar downloader em `app/src/uteis/` (se necessario)
5. Criar processador em `app/src/uteis/` (se necessario)
6. Registrar no dicionario SCRIPTS (em `_build_scripts_dict()`, `app/cli/run_script.py`), sob a chave do artigo
7. Adicionar flag RUN_SNN em `app/settings/settings.toml`
8. Atualizar `settings.local.example.toml` com a nova flag comentada
9. Atualizar CHANGELOG.md
10. Adicionar erros especificos em `_ERROR_HINTS` (se necessario)
11. Testar com `uv run python run_script.py --list` e `uv run python run_script.py <artigo> sNN`

---

## Passo 1 — Definir artigo, identificador e descricao

Convencao de nomes:
- Artigo: pasta em `artigos/` (ex: `artigo_JBN_AS_17_07_2026`)
- Identificador: `sNN` (sequencial dentro do artigo: s00, s01, s02...)
- Arquivo: `artigos/<artigo>/sNN_descricao_curta.py`
- Logger name: `sNN` ou `sNN_descricao`
- Cache name: `<artigo>_sNN_descricao` (prefixado pelo artigo, evita colisao entre artigos)
- Flag: `RUN_SNN`
- Diretorio de saida: `Saida/<artigo>/sNN_DESCRICAO/`

Exemplo: s00 para precipitacao no artigo JBN → `artigos/artigo_JBN_AS_17_07_2026/s00_precipitacao_era5.py`, flag `RUN_S00`

---

## Passo 2 — Criar a pasta do artigo (se ainda nao existir)

```bash
mkdir -p artigos/<artigo>
touch artigos/<artigo>/__init__.py
```

---

## Passo 3 — Criar o script principal

Criar `artigos/<artigo>/sNN_descricao.py` seguindo o template:

```python
"""sNN: Descricao do que o script faz."""

import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.shared.logger import get_logger
from app.shared.settings_factory import settings

logger = get_logger("sNN")


def main():
    """Entry point — chamado pelo CLI sem argumentos."""
    start_time = time.time()

    logger.info("=" * 80)
    logger.info("SCRIPT sNN: DESCRICAO")
    logger.info("=" * 80)

    # --- 1. Definir areas/parametros ---
    output_dir = Path(settings.DIR_OUTPUT) / "<artigo>" / "sNN_DESCRICAO"
    output_dir.mkdir(parents=True, exist_ok=True)

    lst_areas = ["brasil", "nordeste"]  # areas de settings["areas_plotagem"]

    # --- 2. Cache check ---
    cache_params = {
        "DATA_INICIAL": settings.DATA_INICIAL,
        "DATA_FINAL": settings.DATA_FINAL,
        "areas": lst_areas,
        "script_version": "1.0",
    }
    output_files = [str(output_dir / f"sNN_{area}.png") for area in lst_areas]

    if check_cache_valid("<artigo>_sNN", cache_params, output_files):
        logger.info("CACHE VALIDO! Pulando execucao.")
        return

    # --- 3. Parse de datas ---
    dt_ini = datetime.strptime(settings.DATA_INICIAL, "%Y-%m-%d")
    dt_fim = datetime.strptime(settings.DATA_FINAL, "%Y-%m-%d")

    # --- 4. Download de dados ---
    # from app.src.uteis.downloaders_variavel_ERA5 import ensure_variavel_for_period
    # files = ensure_variavel_for_period(start=dt_ini, end=dt_fim, ...)

    # --- 5. Processamento ---
    # ds = xr.open_dataset(...)
    # ... calculos ...

    # --- 6. Plotagem por area ---
    for area in lst_areas:
        # fig, ax = ...
        # plt.savefig(output_dir / f"sNN_{area}.png", dpi=150, bbox_inches="tight")
        # plt.close(fig)
        pass

    # --- 7. Salvar cache ---
    execution_time = time.time() - start_time
    save_cache_metadata("<artigo>_sNN", cache_params, output_files, execution_time)

    logger.info(f"Script sNN concluido em {execution_time:.1f}s")
```

**Regras obrigatorias:**
- A funcao DEVE se chamar `main()` sem parametros
- Use `from app.shared.settings_factory import settings` (nao `app.config`)
- Use `from app.shared.logger import get_logger` (nao `app.common.logger`)
- Use cache_manager para evitar reprocessamento — prefixe o nome do cache com o artigo para nao colidir com scripts de mesmo `sNN` em outro artigo

---

## Passo 4 — Criar downloader (se necessario)

Se o script precisa baixar dados de uma API (ex: Copernicus CDS), criar:

```
app/src/uteis/downloaders_<variavel>_<dataset>.py
```

Convencao:
- Nome: `downloaders_<variavel>_<fonte>.py` (ex: `downloaders_precip_ERA5.py`)
- Funcao principal: `ensure_<variavel>_for_period(start, end, area, ...)`
- Se for ERA5/CDS, prefira usar/estender `downloaders_era5_generico.py` (autenticacao ja resolve
  a `KEY_CDS_<PESQUISADOR>` certa sozinha — veja secao 3 do GUIA-NOVOS-SCRIPTS.md)
- Salvar arquivos em `Entrada/arquivos_nc/<subdir>/`
- Suportar `force_redownload` parameter

`app/src/uteis/` e compartilhado entre todos os artigos — se o downloader for reutilizavel, deixe-o generico (nao acople ao nome do artigo).

---

## Passo 5 — Criar processador (se necessario)

Se o script precisa processar dados brutos antes de plotar:

```
app/src/uteis/processa_<variavel>.py
```

Convencao:
- Funcao principal retorna dataset processado
- Exemplo: `build_daily_mean_dataset(files, required_hours_utc, ...)`

---

## Passo 6 — Registrar no SCRIPTS dict

Editar `_build_scripts_dict()` em `app/cli/run_script.py`, adicionar entrada sob a chave do artigo (crie a chave se for o primeiro script dele):

```python
def _build_scripts_dict() -> dict:
    return {
        "<artigo>": {
            # ... scripts existentes do artigo ...
            "sNN": {
                "module": "artigos.<artigo>.sNN_descricao",
                "description": "Descricao curta (max ~40 chars)",
                "setting_flag": "RUN_SNN",
                "support_files": [
                    # Arquivos que podem ser baixados via SFTP (climatologias, etc.)
                    # {
                    #     "local": "Entrada/arquivos_nc/arquivo.nc",
                    #     "remote": f"{REMOTE_BASE}/Entrada/arquivos_nc/arquivo.nc",
                    #     "description": "Descricao do arquivo",
                    # },
                ],
                "required_files": [
                    # Arquivos obrigatorios SEM download automatico (legendas, etc.)
                    # {
                    #     "local": "Entrada/arquivo.png",
                    #     "description": "Descricao do arquivo",
                    # },
                ],
            },
        },
        # "outro_artigo": { ... },
    }
```

**Diferenca entre `support_files` e `required_files`:**
- `support_files`: tem path `remote`, pode ser baixado via SFTP quando SFTP_ENABLED=true
- `required_files`: so tem path `local`, levanta `FileNotFoundError` se nao existir

`list_scripts()` (`--list`) e o menu de comandos sao gerados automaticamente a partir desse dicionario — nao precisa editar mais nada em `app/cli/run_script.py`.

---

## Passo 7 — Adicionar flag no settings.toml

Editar `app/settings/settings.toml`, adicionar na secao `[default]`:

```toml
[default]
# ... flags existentes ...
RUN_SNN = true
```

Se o script precisa de configuracoes especificas por ambiente:

```toml
[development]
RUN_SNN = true

[production]
RUN_SNN = true
```

---

## Passo 8 — Atualizar settings.local.example.toml

Adicionar a flag comentada no template:

```toml
# Flags: descomente para desativar scripts no --all
# RUN_SNN = false
```

---

## Passo 9 — Atualizar CHANGELOG.md

Adicionar na secao `[Unreleased]`:

```markdown
### Adicionado
- Script <artigo>/sNN: descricao do que faz
```

---

## Passo 10 — Tratamento de erros

O projeto usa um handler global (`@friendly_errors` em `app/shared/error_handler.py`).
NAO coloque try/except generico nos scripts — deixe os erros subirem ate o handler.

Se o novo script pode gerar erros especificos, adicione entradas em `_ERROR_HINTS`:

```python
# Em app/shared/error_handler.py
_ERROR_HINTS = [
    # ...
    (
        Exception,
        "substring do erro",
        "Mensagem amigavel com solucao.",
    ),
]
```

Ver skill `tratamento-de-erros` para detalhes completos.

---

## Passo 11 — Testar

```bash
# Verificar se aparece no menu
uv run python run_script.py --list

# Executar isoladamente
uv run python run_script.py <artigo> sNN

# Executar com --all (deve respeitar flag RUN_SNN)
uv run python run_script.py --all

# Testar com verbose
uv run python run_script.py <artigo> sNN --verbose
```

---

## Resumo de arquivos tocados

| Acao    | Arquivo                                     | Obrigatorio?              |
|---------|----------------------------------------------|--------------------------|
| Criar   | `artigos/<artigo>/__init__.py`               | Se for o 1o script do artigo |
| Criar   | `artigos/<artigo>/sNN_descricao.py`          | Sim          |
| Criar   | `app/src/uteis/downloaders_*.py`             | Se usa API   |
| Criar   | `app/src/uteis/processa_*.py`                | Se processa  |
| Editar  | `app/cli/run_script.py` (`_build_scripts_dict()`) | Sim     |
| Editar  | `app/settings/settings.toml` (RUN_SNN)       | Sim          |
| Editar  | `settings.local.example.toml`                | Sim          |
| Editar  | `CHANGELOG.md`                               | Sim          |
| Editar  | `app/settings/.secrets.toml`                 | Se nova API  |
