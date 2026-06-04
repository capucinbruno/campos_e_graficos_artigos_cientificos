# Quickstart - Campos Observados ERA5

Guia rapido para baixar e gerar mapas em **5 minutos**.

---

## 1. Clonar e instalar dependencias

```bash
git clone <url-do-repositorio>
cd campos_observados_era5

# Criar e ativar o ambiente virtual
python -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -r <(poetry export --without-hashes)

# Instalar browser do Playwright
playwright install chromium
```

## 2. Obter sua chave do Copernicus CDS

1. Crie conta em https://cds.climate.copernicus.eu/
2. Login → clique no seu nome → perfil
3. Copie o valor do campo **key** do `.cdsapirc`

## 3. Configurar credenciais

Edite `app/settings/.secrets.toml`:

```toml
[default]
KEY_CDS = "cole-sua-chave-aqui"
```

## 4. Configurar datas

Copie e edite o arquivo de configuracao local:

```bash
cp settings.local.example.toml settings.local.toml
```

Edite `settings.local.toml`:

```toml
DATA_INICIAL = "2026-05-20"
DATA_FINAL   = "2026-06-03"
```

> **Dica:** O ERA5 tem ~7 dias de atraso. Para períodos recentes o script usa automaticamente o GDAS (NOMADS).
> Se `DATA_FINAL` for hoje, o script ajusta para ontem automaticamente e avisa no terminal.

## 5. Ativar o ambiente e rodar

```bash
source .venv/bin/activate

# Listar scripts disponíveis
python run_script.py --list

# Geopotencial 250hPa (ERA5/GDAS + Climatologia PSL)
DYNACONF_SFTP_ENABLED=false python run_script.py s01

# Vento 100m + MSLP
DYNACONF_SFTP_ENABLED=false python run_script.py s00
```

> **Nota s01:** Climatologia baixada automaticamente do PSL/NOAA via Playwright. Precisa da legenda (`legenda_atlantic.png`) em `Entrada/`.
> **Nota s00:** Precisa da climatologia de vento (`climatologia_1991_2020_vento100m_ERA5.nc`). Se `SFTP_ENABLED=true`, sera baixada automaticamente. Caso contrario, copie manualmente para `Entrada/arquivos_nc/`.

## 6. Ver resultados

- **Mapas s01:** `Saida/s01_GEOP250/`
- **Mapas s00:** `Saida/s00_VENTO_EOLICAS_SEMOP/`
- **Dados baixados:** `dados/`
- **Logs:** `logs/campos_observados.log`

---

## Comandos uteis

```bash
# Sobrescrever datas via CLI (ignora settings.local.toml)
DYNACONF_SFTP_ENABLED=false python run_script.py s01 --data-inicial 2026-05-20 --data-final 2026-06-03

# Forcar re-download dos dados
DYNACONF_SFTP_ENABLED=false python run_script.py s01 --force-download

# Logging detalhado (traceback completo)
DYNACONF_SFTP_ENABLED=false python run_script.py s01 --verbose

# Executar todos os scripts habilitados
DYNACONF_SFTP_ENABLED=false python run_script.py --all
```

## Problemas comuns

| Problema | Solucao |
|----------|---------|
| `ImportError: jinja2` | Execute `pip install jinja2` no `.venv` |
| `ModuleNotFoundError` | Verifique se o `.venv` esta ativado (`source .venv/bin/activate`) |
| `FileNotFoundError: legenda_atlantic.png` | Copie manualmente para `Entrada/` |
| Erro de autenticacao CDS | Verifique `KEY_CDS` no `app/settings/.secrets.toml` |
| `DATA_FINAL` muito recente | O script ajusta automaticamente para ontem e avisa no terminal |
| `'Settings' object has no attribute 'LST_AREAS_S01'` | Crie `settings.local.toml` a partir do exemplo |

---

Para documentacao completa, veja o [README.md](README.md).
