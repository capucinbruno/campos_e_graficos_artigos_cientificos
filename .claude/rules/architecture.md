---
description: Arquitetura do projeto campos_e_graficos_artigos_cientificos
---

# Arquitetura - Campos e Graficos para Artigos Cientificos

## Estrutura de Diretorios

```
campos_e_graficos_artigos_cientificos/
├── run_script.py                 # Entry point (wrapper)
├── setup.sh                      # Setup automatizado
├── artigos/                      # Um artigo cientifico por pasta
│   └── artigo_JBN_AS_17_07_2026/     # Scripts (sNN) especificos deste artigo
├── app/
│   ├── cli/
│   │   └── run_script.py         # CLI (argparse + SCRIPTS dict + @friendly_errors)
│   ├── shared/                   # Cross-cutting concerns (NAO importar de outras camadas)
│   │   ├── singleton.py          # Metaclass Singleton
│   │   ├── settings_factory.py   # Dynaconf factory (Singleton)
│   │   ├── logger.py             # LoggerService (Loguru, Singleton)
│   │   ├── sftp_client.py        # Cliente SFTP (Paramiko, Context Manager)
│   │   └── error_handler.py      # @friendly_errors + _ERROR_HINTS
│   ├── common/                   # Utilitarios reutilizaveis
│   │   ├── cache_manager.py      # Cache por hash de params
│   │   ├── dataset_utils.py      # Utilitarios xarray
│   │   ├── download_helper.py    # Download multi-engine
│   │   └── parallel_helper.py    # ThreadPoolExecutor
│   ├── settings/                 # Configs TOML (Dynaconf)
│   │   ├── settings.toml         # Config master (ambientes)
│   │   ├── settings.json         # Regioes de plotagem
│   │   ├── .secrets.toml         # Credenciais (git-ignored)
│   │   └── .secrets_example.toml # Template credenciais
│   └── src/uteis/                # Downloaders ERA5/GDAS genericos (variaveis_meteorologicas.py) + processadores
├── Entrada/                      # Arquivos fixos (legendas, climatologias)
├── dados/                        # Dados baixados do CDS (.nc, .grb) — gitignored
├── Saida/                        # Mapas gerados (.png)
└── logs/                         # Logs da aplicacao
```

## Fluxo de Execucao

```
run_script.py → app/cli/run_script.py
    → _check_required_files()     (arquivos locais obrigatorios)
    → _ensure_support_files()     (SFTP download se necessario)
    → importlib.import_module()   (artigos/<artigo>/sNN_*.py)
    → module.main()               (download CDS → processa → plota → salva)
```

## Separacao de Diretorios

| Diretorio | Tipo | Gitignored | Conteudo |
|-----------|------|:----------:|----------|
| `Entrada/` | Fixo/estatico | Parcial (.nc sim, imagens nao) | Legendas, climatologias |
| `dados/` | Baixado do CDS | Sim | Arquivos .nc/.grb por variavel |
| `Saida/` | Gerado | Sim | Mapas PNG por script |
| `logs/` | Gerado | Sim | Logs Loguru |

## Padroes Arquiteturais

| Padrao | Onde | Exemplo |
|--------|------|---------|
| **Singleton** (metaclass) | shared/ | `SettingsFactory`, `LoggerService` |
| **Context Manager** | shared/ | `SFTPClient` |
| **Factory** | shared/ | `get_settings()`, `get_logger()` |
| **Decorator** | shared/ | `@friendly_errors` |

## Regras de Dependencia

```
cli/ ──→ artigos/<artigo>/ ──→ app/src/uteis/
  │              │                   │
  │              ├──→ common/        │
  │              │                   │
  └──→ shared/ ←─────────────────────┘
```

- `cli/` pode importar de qualquer camada
- `artigos/<artigo>/` importa de `shared/`, `common/`, `src/uteis/`
- `src/uteis/` importa de `shared/`
- `shared/` NAO importa de nenhuma outra camada da app

## Ambientes

| Ambiente | SFTP | Logging | DIR_DADOS |
|----------|------|---------|-----------|
| **development** | Habilitado | DEBUG | `dados/` (relativo) |
| **qa** | Desabilitado | DEBUG | `dados/` (relativo) |
| **production** | Desabilitado | INFO | Caminho absoluto no servidor |
