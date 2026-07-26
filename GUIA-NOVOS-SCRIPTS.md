# Guia: Adicionando Novos Scripts

Como adicionar um novo script meteorologico ao projeto `campos_e_graficos_artigos_cientificos`.

---

## Visao geral

Cada script do projeto segue um padrao:

1. Baixa dados de uma fonte (ERA5/CDS, SFTP, etc.)
2. Processa os dados (medias, anomalias, etc.)
3. Gera mapas/graficos por area geografica
4. Salva resultados em `Saida/`

O CLI (`run_script.py`) descobre e executa scripts a partir do dicionario `SCRIPTS` em `app/cli/run_script.py`. Para adicionar um novo, basta criar o script e registra-lo.

---

## Passo a passo

### 1. Escolher identificador

O identificador segue o padrao `sNN` (sequencial):

| Existente | Descricao |
|-----------|-----------|
| `s00` | Vento 100m + MSLP (ERA5) |
| `s01` | Anomalia Geopotencial 250hPa |
| `s02` | *(proximo disponivel)* |

**Convencao de nomes:**

```
Identificador:    s02
Arquivo:          scripts/s02_precipitacao_era5.py
Flag:             RUN_S02
Diretorio saida:  Saida/s02_PRECIPITACAO/
Logger:           get_logger("s02")
Cache:            "s02_precipitacao"
```

### 2. Criar o script

Crie o arquivo `scripts/sNN_descricao.py` com a seguinte estrutura:

```python
"""sNN: Descricao breve do que o script faz."""

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
    """Entry point - chamado pelo CLI sem argumentos."""
    start_time = time.time()

    logger.info("=" * 80)
    logger.info("SCRIPT sNN: DESCRICAO")
    logger.info("=" * 80)

    # 1. Configurar saida
    output_dir = Path(settings.DIR_OUTPUT) / "sNN_DESCRICAO"
    output_dir.mkdir(parents=True, exist_ok=True)

    lst_areas = ["brasil", "nordeste"]

    # 2. Verificar cache (evita reprocessamento)
    cache_params = {
        "DATA_INICIAL": settings.DATA_INICIAL,
        "DATA_FINAL": settings.DATA_FINAL,
        "areas": lst_areas,
        "script_version": "1.0",
    }
    output_files = [str(output_dir / f"sNN_{area}.png") for area in lst_areas]

    if check_cache_valid("sNN", cache_params, output_files):
        logger.info("CACHE VALIDO - pulando execucao")
        return

    # 3. Parse de datas do settings
    dt_ini = datetime.strptime(settings.DATA_INICIAL, "%Y-%m-%d")
    dt_fim = datetime.strptime(settings.DATA_FINAL, "%Y-%m-%d")

    # 4. Download de dados
    # from app.src.uteis.downloaders_variavel_ERA5 import ensure_variavel_for_period
    # files = ensure_variavel_for_period(start=dt_ini, end=dt_fim, ...)

    # 5. Processamento
    # ds = xr.open_dataset(...)

    # 6. Plotagem por area
    for area in lst_areas:
        area_cfg = settings["areas_plotagem"][area]
        # fig = plt.figure(figsize=(15, 10))
        # ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        # ... contourf, colorbar, gridlines ...
        # plt.savefig(output_dir / f"sNN_{area}.png", dpi=150, bbox_inches="tight")
        # plt.close(fig)
        pass

    # 7. Salvar cache
    execution_time = time.time() - start_time
    save_cache_metadata("sNN", cache_params, output_files, execution_time)
    logger.info(f"Script sNN concluido em {execution_time:.1f}s")
```

> **Importante:** A funcao DEVE se chamar `main()` sem parametros. O CLI chama `module.main()` via `importlib`.

### 3. Criar downloader (se precisar baixar dados)

Se o script baixa dados de uma API (ex: Copernicus CDS), crie um modulo em:

```
app/src/uteis/downloaders_<variavel>_<fonte>.py
```

Convencao de nome da funcao principal: `ensure_era5_<variavel>_for_period(...)`.

Padrao:
- Use `settings.KEY_CDS` para autenticacao no CDS
- Salve arquivos em `dados/<subdiretorio>/` (NAO em `Entrada/` — este e para arquivos fixos)
- Aceite parametro `force_redownload` para forcar re-download

### 4. Criar processador (se precisar processar dados brutos)

Se precisa transformar os dados brutos antes de plotar:

```
app/src/uteis/processa_<variavel>.py
```

Exemplo de uso tipico: concatenar arquivos mensais baixados e calcular a media diaria.

### 5. Registrar no CLI

Edite `app/cli/run_script.py` e adicione ao dicionario `SCRIPTS`:

```python
SCRIPTS = {
    "s00": { ... },
    "s01": { ... },
    # Novo script:
    "sNN": {
        "module": "scripts.sNN_descricao",
        "description": "Descricao curta (max ~40 chars)",
        "setting_flag": "RUN_SNN",
        "support_files": [],
        "required_files": [],
    },
}
```

#### Tipos de arquivos de dependencia

O projeto diferencia dois tipos de arquivos que um script pode precisar:

**`support_files`** — Arquivos que podem ser baixados via SFTP do servidor Oracle:

```python
"support_files": [
    {
        "local": "Entrada/arquivos_nc/climatologia.nc",
        "remote": "/home/ubuntu/resources/meteorologia/campos-observados/climatologia/climatologia.nc",
        "description": "Climatologia 1991-2020",
    },
],
```

- Quando `SFTP_ENABLED=true` (development): baixa automaticamente antes de executar
- Quando `SFTP_ENABLED=false` (production): avisa mas nao bloqueia

**`required_files`** — Arquivos que devem existir localmente (sem SFTP):

```python
"required_files": [
    {
        "local": "Entrada/legenda.png",
        "description": "Legenda do mapa",
    },
],
```

- Se nao existir, levanta `FileNotFoundError` com mensagem clara
- Use para arquivos que nao estao no servidor remoto

### 6. Adicionar flag no settings

Edite `app/settings/settings.toml` na secao `[default]`:

```toml
[default]
# ... flags existentes ...
RUN_S00 = true
RUN_S01 = true
RUN_SNN = true    # <-- novo
```

Edite `settings.local.example.toml`:

```toml
# Flags: descomente para desativar scripts no --all
# RUN_S00 = false
# RUN_S01 = false
# RUN_SNN = false    # <-- novo
```

### 7. Adicionar descricao no menu

Na funcao `list_scripts()` em `app/cli/run_script.py`, adicione a descricao do comando na secao "Comandos:":

```python
print(f"  {GREEN}uv run python run_script.py sNN{RESET}")
print(f"    {DIM}Baixa dados X via API CDS, salva .nc em dados/{RESET}")
print(f"    {DIM}e gera mapas de Y em Saida/sNN_DESCRICAO/{RESET}")
print(f"    {YELLOW}Requisito: descricao (se houver){RESET}")
print()
```

### 8. Atualizar documentacao

Adicione no `CHANGELOG.md`:

```markdown
## [Unreleased]

### Adicionado
- Script sNN: descricao do que faz
```

---

## Resumo de arquivos

| Acao   | Arquivo | Obrigatorio? |
|--------|---------|:------------:|
| Criar  | `scripts/sNN_descricao.py` | Sim |
| Criar  | `app/src/uteis/downloaders_*.py` | Se usa API |
| Criar  | `app/src/uteis/processa_*.py` | Se processa |
| Editar | `app/cli/run_script.py` — SCRIPTS dict | Sim |
| Editar | `app/cli/run_script.py` — list_scripts() | Sim |
| Editar | `app/settings/settings.toml` — RUN_SNN | Sim |
| Editar | `settings.local.example.toml` | Sim |
| Editar | `CHANGELOG.md` | Sim |

---

## Diagrama de fluxo

```mermaid
flowchart TD
    A[uv run python run_script.py sNN] --> B[parse_args]
    B --> C[_apply_overrides]
    C --> D{required_files?}
    D -->|Sim| E[_check_required_files]
    E -->|Faltando| F[FileNotFoundError]
    E -->|OK| G{support_files?}
    D -->|Nao| G
    G -->|Sim + SFTP| H[_ensure_support_files / SFTP download]
    G -->|Nao| I[importlib.import_module]
    H --> I
    I --> J[module.main]
    J --> K[Cache check]
    K -->|Valido| L[Return - pula execucao]
    K -->|Invalido| M[Download dados]
    M --> N[Processar]
    N --> O[Plotar mapas]
    O --> P[Salvar cache]
    P --> Q[Concluido]
```

---

## Fluxo de configuracoes (prioridade)

```mermaid
flowchart LR
    A[settings.toml] --> B[settings.local.toml]
    B --> C[.secrets.toml]
    C --> D[settings.json]
    D --> E[.env]
    E --> F[CLI args]
    F --> G[Settings finais]

    style F fill:#2d6,color:#fff
    style A fill:#666,color:#fff
```

Arquivos a direita tem **maior prioridade** e sobrescrevem os da esquerda.

---

## Imports padrao

```python
# Infra do projeto (use SEMPRE estes, nao os antigos)
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.common.cache_manager import check_cache_valid, save_cache_metadata

# Cartografia (se gera mapas)
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

# Dados
import numpy as np
import xarray as xr
```

> **Nao use** `from app.config import settings` nem `from app.common.logger import ...` — estes sao legados.

---

## Tratamento de erros

O projeto usa um **handler global de excecoes** no entry point do CLI (`@friendly_errors` em `app/cli/run_script.py`). Isso significa que voce **nao precisa colocar try/except** em cada pedaco de codigo.

### Como funciona

Toda excecao nao tratada em Python sobe automaticamente ate o topo da pilha de chamadas. O decorator `@friendly_errors` fica no `main()` e intercepta tudo ali, traduzindo erros tecnicos em mensagens amigaveis com solucao.

```
Seu script (nivel 3)  ──┐
  Downloader (nivel 2)  ─┤  excecao sobe automaticamente
    Paramiko (nivel 1)  ──┘
                          ▼
    @friendly_errors     ← intercepta e mostra mensagem amigavel
```

Em vez de um traceback de 50 linhas, o usuario ve:

```
ERRO: Arquivo nao encontrado no servidor SFTP

Solucao:
  Verifique se o caminho remoto esta correto
  ou copie o arquivo manualmente.

Dica: use --verbose para ver o traceback completo
```

### O que fazer no seu script

| Situacao | O que fazer |
|----------|-------------|
| Erro generico (IO, rede, parse) | Deixe subir. O handler trata. |
| Precisa adicionar contexto | `raise RuntimeError("mensagem clara") from None` |
| Erro recuperavel (retry, fallback) | `try/except` local e OK |
| Validacao de entrada | `raise ValueError("mensagem clara")` |

**Errado** — engole o erro:

```python
try:
    ds = xr.open_dataset("arquivo.nc")
except Exception as e:
    print(f"Erro: {e}")  # handler global nunca ve o erro
    return
```

**Certo** — deixa subir:

```python
ds = xr.open_dataset("arquivo.nc")  # se falhar, o handler trata
```

**Aceitavel** — adiciona contexto e re-lanca:

```python
try:
    ds = xr.open_dataset(path)
except OSError:
    raise RuntimeError(f"Falha ao abrir climatologia: {path}") from None
```

### Adicionando erros conhecidos ao seu script

Se o seu script pode gerar um erro especifico que merece uma dica amigavel, adicione uma entrada no mapa `_ERROR_HINTS` em `app/shared/error_handler.py`:

```python
(
    Exception,       # tipo da excecao
    "503",           # substring na mensagem (ou None)
    "API indisponivel. Tente novamente em alguns minutos.",
),
```

Erros ja mapeados: SFTP, SSH, CDS API, NetCDF corrompido, imports faltando. Veja a lista completa em `app/shared/error_handler.py`.

---

## Dicas

- **Cache**: Inclua `script_version` nos `cache_params`. Incremente quando mudar a logica de plotagem para forcar reprocessamento.
- **Areas**: Use `settings["areas_plotagem"]` para configuracoes geograficas. As areas ja estao definidas no `settings.toml` (brasil, nordeste, atlantico_tropical, etc.).
- **Dados baixados**: Downloads do CDS vao para `dados/`, NAO para `Entrada/`.
- **Teste rapido**: Use `--data-inicial` e `--data-final` com periodo curto (1-2 dias) para testar sem baixar muitos dados.
- **ERA5**: Os dados tem ~5 dias de atraso. Use `DATA_FINAL` de pelo menos 5 dias atras.
