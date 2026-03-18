# Changelog

Todas as mudancas notaveis neste projeto serao documentadas neste arquivo.

O formato segue o [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/).

---

## [Unreleased]

### Adicionado

- **Refatoracao completa de infraestrutura** seguindo padroes dos projetos `usando_api_ampere` e `windx-automatico`
- Migracao de Poetry para **UV** como package manager
- `setup.sh` — setup automatizado com menu de ambiente (development/production/qa), copia configs, cria diretorios
- `setup.sh` — copia automatica de `.vscode/settings_exemplo.json` para `.vscode/settings.json`
- `.python-version` — Python 3.12.9 fixo
- `.env.example` — template para `ENV_FOR_DYNACONF`
- `.vscode/settings_exemplo.json` — template VSCode com icones e cores do projeto
- Modulo `app/shared/` com infraestrutura reutilizavel:
  - `singleton.py` — Metaclass Singleton
  - `settings_factory.py` — Dynaconf factory com Singleton
  - `logger.py` — LoggerService com Loguru (Singleton, silencia cdsapi/paramiko)
  - `sftp_client.py` — Cliente SFTP enxuto com context manager
- Modulo `app/cli/run_script.py` — CLI com argparse e dicionario SCRIPTS
- `run_script.py` — entry point wrapper na raiz
- Download automatico de arquivos de suporte (`support_files`) via SFTP antes da execucao de cada script
- Verificacao de arquivos obrigatorios (`required_files`) com `FileNotFoundError` claro antes da execucao
- Saida colorida no `--list` com status de cada script e seus requisitos (ANSI colors)
- Suporte a 3 ambientes: development (SFTP + DEBUG), qa, production
- `app/settings/.secrets_example.toml` — template para credenciais (KEY_CDS + SSH)
- `settings.local.example.toml` — template simplificado para datas e flags
- Secao `[qa]` no `settings.toml`
- Configuracoes de logging por ambiente (LEVEL_LOGGING, LOGGER_BACKTRACE, LOGGER_DIAGNOSE)
- README reescrito com diagramas Mermaid (fluxo geral, s00, s01, prioridade de settings)
- QUICKSTART atualizado para UV e novo CLI
- `GUIA-NOVOS-SCRIPTS.md` — guia completo para adicionar novos scripts ao projeto
- Skill Claude Code `.claude/skills/adicionar-novo-script.md` para assistencia ao adicionar scripts
- `app/shared/error_handler.py` — Handler global de excecoes com `@friendly_errors` e mapa `_ERROR_HINTS`
- Skill Claude Code `.claude/skills/tratamento-de-erros.md` para guia de tratamento de erros
- Secao "Tratamento de Erros" no README e GUIA-NOVOS-SCRIPTS.md

### Alterado

- `pyproject.toml` migrado de Poetry para UV (hatchling)
- Imports em todos os modulos atualizados: `app.config` → `app.shared.settings_factory`, `app.common.logger` → `app.shared.logger`
- `settings.toml` refatorado com ambientes (development/qa/production) e configuracoes de logging
- CLI simplificado: `uv run python run_script.py s00` em vez de `poetry run python main.py --script s00`
- Credenciais movidas de `settings.local.toml` para `app/settings/.secrets.toml`
- `.gitignore` atualizado para UV (uv.lock, .secrets.toml, poetry.lock)

### Removido

- `app/logging_config.json` (substituido por LoggerService)
- Dependencia de Poetry (substituido por UV)

### Corrigido

- Logging excessivo do `cdsapi` (debug=False nos downloaders + logging.WARNING)
- Logging excessivo do `paramiko` (logging.ERROR)

---

## [0.1.0] - 2026-03-13

Commit: `6bba154`

### Adicionado

- Estrutura inicial do projeto
- Script s00: download e plotagem de vento a 100m + MSLP do ERA5
- Script s01: download e plotagem de anomalia de geopotencial em 250hPa
- Modulo `app/common/` com infraestrutura reutilizavel:
  - `logger.py` - Sistema de logging com Loguru (rotacao, retencao, thread-safe)
  - `cache_manager.py` - Cache inteligente com hash MD5 de parametros
  - `connections.py` - Conexao SSH/SFTP via Paramiko
  - `download_helper.py` - Download multi-engine com retry e barra de progresso
  - `file_loader.py` - Carregamento transparente local/remoto de arquivos
  - `dataset_utils.py` - Utilitarios xarray para NetCDF
  - `parallel_helper.py` - Processamento paralelo com ThreadPoolExecutor
- Modulo `app/src/uteis/` com downloaders e processadores ERA5
- Configuracao via Dynaconf com suporte a ambientes (development/production)
- Suporte a SSH/SFTP para acesso remoto a dados (modo development)
- `main.py` com CLI para execucao de scripts (--script, --force-download, --force-rerun, --clear-cache)
- Configuracoes de 30+ regioes geograficas para plotagem (settings.json)
- Template de configuracao local (settings.local.example.toml)
- `.gitignore` configurado para excluir dados pesados (.nc, .grb), saidas e credenciais
