# Campos Observados ERA5

Download e plotagem de campos meteorologicos observados a partir da reanalise ERA5 (Copernicus CDS). Gera mapas de vento, pressao, geopotencial e anomalias para diversas regioes geograficas.

## Tech Stack

Python 3.12.9 | UV | Dynaconf | Loguru | Paramiko/SFTP | cdsapi (Copernicus) | xarray | cartopy | matplotlib | scipy | NumPy | Pandas

## Comandos

```bash
uv run python run_script.py --list                                   # Listar scripts
uv run python run_script.py s00                                      # Executar script
uv run python run_script.py s00 --verbose                            # Com traceback completo
uv run python run_script.py --all                                    # Todos os habilitados
uv run python run_script.py s00 --data-inicial 2026-03-01 --data-final 2026-03-12
bash setup.sh                                                        # Setup interativo
```

## Estrutura de Pastas

```
Scripts/                  # Scripts do meteorologista (s00, s01, ...)
app/
  cli/                    # CLI (run_script.py com SCRIPTS dict)
  shared/                 # Infraestrutura (singleton, settings, logger, sftp, error_handler)
  common/                 # Utilitarios (cache, download, dataset_utils, parallel)
  settings/               # TOML configs (Dynaconf) + settings.json (regioes)
  src/uteis/              # Downloaders e processadores ERA5
Entrada/                  # Dados baixados (.nc, .grb)
Saida/                    # Mapas gerados (.png)
logs/                     # Logs da aplicacao
```

## Regras Universais

1. **CHANGELOG**: Atualizar `CHANGELOG.md` em CADA commit (secao `[Unreleased]`)
2. **Configs sincronizadas**: `settings.local.toml` <-> `settings.local.example.toml` | `.secrets.toml` <-> `.secrets_example.toml`
3. **Imports**: Prefixo `app.` canonico. SEMPRE usar `app.shared.settings_factory` e `app.shared.logger` (NAO `app.config` ou `app.common.logger`)
4. **Secrets**: NUNCA commitar `.secrets.toml` ou `.env` com valores reais. NUNCA logar chaves/tokens
5. **Tratamento de erros**: NAO usar try/except generico nos scripts. O decorator `@friendly_errors` no entry point captura tudo. Para erros especificos, adicionar em `_ERROR_HINTS` no `app/shared/error_handler.py`
6. **Scripts**: Funcao `main()` sem parametros. Registrar no dict `SCRIPTS` em `app/cli/run_script.py`
7. **Cache**: Usar `check_cache_valid()` / `save_cache_metadata()` em todo script. Incluir `script_version` nos params
8. **Arquivos de suporte**: `support_files` (SFTP) vs `required_files` (local com FileNotFoundError)

## Ambientes

| Ambiente | SFTP | Logging | Uso |
|----------|------|---------|-----|
| development | Habilitado | DEBUG | Maquina local |
| qa | Desabilitado | DEBUG | Testes |
| production | Desabilitado | INFO | Servidor Oracle |

## Gatilhos Contextuais

Ao realizar cada tipo de tarefa, leia o arquivo indicado ANTES de comecar:

| Tarefa | Ler primeiro |
|--------|-------------|
| Adicionar novo script | [GUIA-NOVOS-SCRIPTS.md](GUIA-NOVOS-SCRIPTS.md) |
| Mexer em tratamento de erros | [.claude/skills/tratamento-de-erros.md](.claude/skills/tratamento-de-erros.md) |
| Mexer em SFTP, download de suporte | [app/shared/sftp_client.py](app/shared/sftp_client.py) + [app/cli/run_script.py](app/cli/run_script.py) |
| Mexer em settings/configs | [app/shared/settings_factory.py](app/shared/settings_factory.py) |
| Mexer em logging | [app/shared/logger.py](app/shared/logger.py) |
| Mexer em downloaders ERA5 | [app/src/uteis/](app/src/uteis/) |
| Mexer em cache | [app/common/cache_manager.py](app/common/cache_manager.py) |
| Mexer em secrets, SSH | [app/settings/.secrets_example.toml](app/settings/.secrets_example.toml) |

## Skills Disponiveis

| Skill | Descricao |
|-------|-----------|
| `adicionar-novo-script` | Checklist completo para adicionar scripts ao projeto |
| `tratamento-de-erros` | Como funciona o handler global e como adicionar erros conhecidos |
