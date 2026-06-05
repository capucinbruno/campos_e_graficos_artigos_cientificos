# Changelog

Todas as mudancas notaveis neste projeto serao documentadas neste arquivo.

O formato segue o [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/).

---

## [Unreleased]

### Modificado

- `app/src/uteis/plot_geop250.py`: substituída concatenação em memória por acumulador streaming (`_compute_period_mean_streaming`) — processa um arquivo mensal por vez, evitando estouro de RAM no WSL para períodos longos (3–4 meses de dados globais ERA5/GDAS)
- `app/src/uteis/clim_PSL_psi200.py`: reescrito com cache por período MM-DD, logger, path via settings e bug de tupla morta corrigido (não utilizado no pipeline — s03 reutiliza `clim_PSL_wnd200.py`)
- `app/src/uteis/clim_PSL_wnd850.py`: criado pelo usuário; corrigido erro na mensagem de exceção (200→850)
- `app/src/uteis/plot_olr_wind250_anom.py`: migrado para pipeline híbrido ERA5/GDAS (250 hPa) + PSL clim + streaming; removidas importações quebradas de plot_chi200.py
- `app/src/uteis/plot_olr_wind850_anom.py`: criado seguindo mesmo padrão do 250 hPa, para anomalia de vento 850 hPa
- `scripts/s06_olr_wind_250_850_anom.py`: renomeado de s06_olr_wind250_anom.py; adicionado pipeline 850 hPa com figuras com sufixo `_250hPa.png` e `_850hPa.png`; OLR download com aria2c; try/except removido
- `app/cli/run_script.py`: s06 atualizado para novo módulo e support_files limpos

### Adicionado (s06/850hPa)

- `app/src/uteis/downloaders_wind850.py`: downloader ERA5 u/v em 850 hPa
- `app/src/uteis/downloaders_gdas_uv250.py`: downloader GDAS u/v 250mb via NOMADS
- `app/src/uteis/downloaders_gdas_uv850.py`: downloader GDAS u/v 850mb via NOMADS
- `app/src/uteis/clim_PSL_wnd250.py`: criado pelo usuário; corrigido erro na mensagem de exceção (200→250)
- `app/src/uteis/plot_rossby_waf.py`: migrado para pipeline híbrido ERA5/GDAS (hgt 250 hPa) + PSL hgt+uv 250mb + streaming accumulator; removida dependência de funções do plot_psi200.py
- `scripts/s04_fluxo_rossby_wave.py`: `script_version` incrementado para 2.0
- `app/cli/run_script.py`: removidas as 3 climatologias locais dos `support_files` do s04
- `app/src/uteis/plot_psi200.py`: migrado para pipeline híbrido ERA5 (200 hPa)/GDAS + climatologia PSL u/v 200mb (via `clim_PSL_wnd200.py`) + streaming accumulator; anomalia calculada de u/v primeiro, depois psi — mesmo padrão do s02
- `scripts/s03_psi200_anom.py`: `script_version` incrementado para 2.0
- `app/cli/run_script.py`: removidas climatologias uwnd/vwnd250 dos `support_files` do s03
- `app/src/uteis/clim_PSL_wnd200.py`: reescrito com cache por período MM-DD, logger, path via settings e uma única sessão Playwright para u e v
- `app/src/uteis/plot_chi200.py`: migrado para pipeline híbrido ERA5 (200 hPa)/GDAS + climatologia PSL + streaming accumulator (mesmo padrão do s01)
- `scripts/s02_chi200_anom.py`: removido try/except genérico em torno de `plot_chi200()`; `script_version` incrementado para 2.0

### Adicionado (s02)

- `app/src/uteis/downloaders_wind200.py`: downloader ERA5 u/v em **200 hPa** (separado do wind250 que continua em uso por s03/s04/s06)
- `app/src/uteis/downloaders_gdas_uv200.py`: downloader GDAS u/v 200mb via NOMADS Grib Filter (padrão do downloaders_gdas_hgt250)

### Removido

- Scripts de teste legados da raiz: `teste.py`, `teste00_dados_vento_SEMOP.py`, `teste01_dados_anom_geop.py`, `teste02_dados_anom_chi200.py`
- Módulos legados da API Ampere: `app/common/ampere_api_client.py`, `app/config.py`, `app/src/reanalise.py`, `app/src/prompt_data.py`, `app/src/gridlines.py`
- `AMPERE_API_BASE_URL` do `settings.toml` e bloco `AMPERE_API_USERNAME/PASSWORD` do `settings.local.example.toml`

### Adicionado

- `--force-rerun` no `run_script.py` — invalida o cache de um script especifico antes de executar (ex: `uv run python run_script.py s05 --force-rerun`), forcando o reprocessamento sem mexer no cache dos demais. Portado do `main.py` legado
- Script s06: Anomalia de OLR + linhas de corrente da anomalia do vento em 250 hPa (OLR do PSL/NOAA + vento ERA5/CDS)
- `app/src/uteis/plot_olr_wind250_anom.py` — modulo de processamento que reutiliza downloader de vento 250 hPa e climatologias uwnd/vwnd para calcular anomalia
- `RUN_S06` — flag de execucao do s06 no settings.toml
- Script s05: Anomalia de OLR (Outgoing Longwave Radiation) a partir de dados CPC Blended OLR do PSL/NOAA
- Download integrado no s05 (HTTP simples via `download_with_progress`, sem necessidade de downloader separado)
- `RUN_S05` — flag de execucao do s05 no settings.toml
- Script s04: Rossby Wave Activity Flux (Takaya & Nakamura 2001) com vetores WAF sobre anomalia de hgt 250 hPa
- `app/src/uteis/plot_rossby_waf.py` — modulo de processamento que reutiliza downloaders existentes (hgt250 + climatologias uwnd/vwnd) e calcula WAF via pacote `tnflux`
- Dependencia `tnflux` adicionada ao projeto
- `RUN_S04` — flag de execucao do s04 no settings.toml
- Script s02: anomalia de CHI200 (velocity potential) com vetores de vento divergente
- `app/src/uteis/plot_chi200.py` — modulo de download, processamento e calculo de anomalia CHI200 via scipy (Helmholtz decomposition)
- `app/src/uteis/downloaders_wind250.py` — downloader de u/v 250 hPa (ERA5) em formato NetCDF
- `FILE_CLIMATOLOGIA_UWND250` e `FILE_CLIMATOLOGIA_VWND250` — settings para climatologias de vento 250 hPa
- Climatologias uwnd250/vwnd250 adicionadas como `support_files` do s02 (download automatico via SFTP)
- `RUN_S02` — flag de execucao do s02 no settings.toml
- Interpolacao automatica da climatologia de vento para o grid do ERA5 (mesmo padrao do s01/geop250)
- Configuracao de quiver por area via `CHI200_QUIVER_POR_AREA` no settings (scale, width, step, min_mag, headwidth, headlength, color)
- Boxes ENSO (Nino 1+2, 3, 3.4, 4), IOD e TSA/TNA no s02

### Alterado

- Mensagem de interrupcao por Ctrl+C (`KeyboardInterrupt` no `@friendly_errors`) agora exibe um bloco mais visivel informando que o script foi cancelado
- `downloaders_wind250.py`: download de u/v 250 hPa migrado de GRIB para **NetCDF** — elimina dependencia de ecCodes C library (`cfgrib`)
- `downloaders_wind250.py`: removidos parametros `area` e `grid` — download global sem filtro (padrao hgt250)
- `downloaders_wind250.py`: download de u e v agora em **requisicoes CDS separadas em paralelo** — cada variavel baixa independentemente via ThreadPoolExecutor, depois mescla com `xr.merge()` (melhora significativa de velocidade)

### Removido

- `main.py` — entry point legado removido. So conhecia o s00 e usava imports proibidos pelo CLAUDE.md (`app.config`, `app.common.logger`). O entry point unico passa a ser `run_script.py` → `app/cli/run_script.py`, que ja registra s00–s06 no dict `SCRIPTS`. As funcionalidades uteis do `main.py` foram portadas (`--force-rerun`); o tratamento de Ctrl+C ja existia no `@friendly_errors`

### Corrigido

- s05/s06: arquivo `olr.day.anom.nc` (PSL/NOAA, atualizado diariamente) nao era re-baixado quando ja existia localmente, gerando mapas em branco ou com dados incorretos quando o periodo solicitado nao estava coberto pelo arquivo local cacheado. Agora o script consulta o periodo solicitado (DATA_INICIAL/DATA_FINAL) contra o arquivo local e so re-baixa se necessario; se mesmo apos download o periodo nao estiver disponivel, aborta com mensagem clara mostrando a primeira/ultima data disponivel no arquivo
- `dataset_utils.arquivo_cobre_periodo`: novo helper que verifica se um arquivo NetCDF existe e cobre completamente um intervalo `[start_date, end_date]`. Util para evitar re-downloads desnecessarios de datasets atualizados periodicamente (PSL/NOAA OLR diario, etc.)
- `dataset_utils.validar_cobertura_temporal`: novo helper que valida cobertura temporal de um dataset e levanta `RuntimeError` com mensagem clara contendo as datas disponiveis se o periodo solicitado nao estiver coberto ou tiver gaps
- `download_helper.download_with_progress`: novo parametro `max_age_hours` (Optional[float]) — checa mtime do arquivo local e re-baixa se exceder o limite. Util para datasets atualizados diariamente
- Logo do s02 deslocado para cima — substituido `fig.add_axes` + `get_position` por `AnchoredOffsetbox` (ancorado ao axes, sobrevive a `bbox_inches='tight'`)
- Faixa branca na plotagem do s01 na divisa dos hemisferios — removida conversao 0-360 e sort manual de lon, usando `sortby` + `add_cyclic_point` direto no DataArray (mesmo padrao do s00)

### Adicionado

- `app/src/uteis/plot_geop250.py` — modulo de download, processamento e calculo de anomalia geopotencial 250 hPa (ERA5)
- `FILE_CLIMATOLOGIA_GEOP250` e `REMOTE_CLIMATOLOGIA_GEOP250` — settings para climatologia geopotencial 250 hPa
- Climatologia geop250 adicionada como `support_files` do s01 (download automatico via SFTP)
- Docstrings descritivas nos scripts (`s00`, `s01`) com finalidade, dados de entrada/saida, datas de criacao e atualizacao
- `SCRIPT_ID`, `SCRIPT_NAME`, `SCRIPT_DESC` derivados de `__file__` e `__doc__` — logger, cache e header dinamicos
- `LST_AREAS_S01` no `settings.local.toml` — lista de areas de plotagem do s01 configuravel sem mexer no git
- `.hooks/` — diretorio para shell scripts de desenvolvimento (changelog-reminder.sh)
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
- `dados/` — diretorio dedicado para dados baixados do CDS (separado de `Entrada/` que mantem arquivos fixos)
- `DIR_DADOS` — nova setting para configurar diretorio de dados baixados por ambiente
- `.claude/rules/` — regras adaptadas do windx-automatico (code-style, security, architecture, gotchas, review, testing)
- Diagrama mermaid de separacao de diretorios (Entrada/ vs dados/ vs Saida/) no README
- `pyproject.toml` com ruff, isort, taskipy e pytest configurados (alinhado com windx-automatico)
- `.pre-commit-config.yaml` com hooks de lint, format, sort, fix e changelog-reminder
- `scripts/changelog-reminder.sh` — lembrete de atualizar CHANGELOG em cada commit
- Detalhes de conexao SFTP (IP, porta, chave) nas mensagens de log e erro
- Validacao de integridade de arquivos .nc/.grb com retry automatico (apaga + re-download 1x)
- `FILE_CLIMATOLOGIA_VENTO100M` e `REMOTE_CLIMATOLOGIA_VENTO100M` configuraveis via settings

### Alterado

- `downloaders_hgt250_ERA5.py`: download de geopotencial 250 hPa migrado de GRIB para **NetCDF** — elimina dependencia de ecCodes C library (`cfgrib`)
- `pyproject.toml` migrado de Poetry para UV (hatchling)
- Imports em todos os modulos atualizados: `app.config` → `app.shared.settings_factory`, `app.common.logger` → `app.shared.logger`
- `settings.toml` refatorado com ambientes (development/qa/production) e configuracoes de logging
- CLI simplificado: `uv run python run_script.py s00` em vez de `poetry run python main.py --script s00`
- Credenciais movidas de `settings.local.toml` para `app/settings/.secrets.toml`
- `.gitignore` atualizado para UV (uv.lock, .secrets.toml, poetry.lock)
- Downloaders e processadores agora salvam em `dados/` (DIR_DADOS) em vez de `Entrada/arquivos_nc/`
- Climatologia vento 100m agora lida do settings (`FILE_CLIMATOLOGIA_VENTO100M`) em vez de path hardcoded
- Scripts `s00` e `s01` refatorados: paths via `Path(settings.DIR_OUTPUT)`, `Path(settings.DIR_INPUT)`, `Path(settings.DIR_DADOS)`
- `s01`: diretorio de saida renomeado de `s04_GEOP250` para `s01_GEOP250`
- `s01`: `os.path` substituido por `pathlib.Path`, `raise Exception` por `raise RuntimeError from err`
- Logging nos scripts: f-strings substituidas por `%s` (lazy formatting do loguru)
- `settings.json` reformatado (whitespace consistente) e limpo (`lst_emails` removido — vinha de outro projeto)
- `s01`: `lst_areas` hardcoded movido para `settings.LST_AREAS_S01`
- `Scripts/` renomeado para `scripts/` (convencao Python lowercase)
- `s01`: referencias `s04_GEOP250` corrigidas para `s01_GEOP250` no README, QUICKSTART e list_scripts
- `s01`: descricao de requisitos no CLI atualizada (climatologia + legenda + dados em `dados/`)
- `scripts/changelog-reminder.sh` movido para `.hooks/changelog-reminder.sh`
- `REMOTE_CLIMATOLOGIA_VENTO100M`: path corrigido para `.../climatologia/` em vez de `.../Entrada/arquivos_nc/`

### Corrigido

- `s01`: logger usava `'s04'`, cache key `'s04'` e mensagens `'S04'` — corrigido para `'s01'`
- `s01`: path da legenda Atlantico usava `Path.cwd()/Entrada/` hardcoded — agora usa `settings.DIR_INPUT`

### Removido

- `database.md` das rules (nao ha banco de dados neste projeto)

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
