# Quickstart - Campos Observados ERA5

Guia rapido para baixar e gerar mapas de vento ERA5 em **5 minutos**.

---

## 1. Setup automatizado

```bash
git clone <url-do-repositorio>
cd campos_observados_era5
bash setup.sh
```

O `setup.sh` instala o UV, pergunta qual ambiente (development/production/qa), copia templates de configuracao, configura VSCode e instala dependencias.

## 2. Obter sua chave do Copernicus CDS

1. Crie conta em https://cds.climate.copernicus.eu/
2. Login -> clique no seu nome (canto superior direito) -> perfil
3. Copie a **Personal Access Token** (API Key)

## 3. Configurar credenciais

Edite `app/settings/.secrets.toml`:

```toml
[default]
KEY_CDS = "cole-sua-chave-aqui"
```

## 4. Configurar datas

Edite `settings.local.toml`:

```toml
[development]
DATA_INICIAL = "2026-05-01"
DATA_FINAL = "2026-05-17"
```

> **Dica:** O ERA5 tem ~5 dias de atraso. Use uma `DATA_FINAL` de pelo menos 5 dias atras.

## 5. Listar scripts disponiveis

```bash
uv run python run_script.py --list
```

## 6. Rodar

```bash
# Vento 100m + MSLP
uv run python run_script.py s00

# Geopotencial 250hPa
uv run python run_script.py s01
```

> **Nota s00:** Precisa da climatologia (`climatologia_1991_2020_vento100m_ERA5.nc`).
> Se `SFTP_ENABLED=true`, sera baixado automaticamente do servidor Oracle.
> Caso contrario, copie manualmente para `Entrada/arquivos_nc/`.

> **Nota s01:** Precisa da climatologia geop250 (`climatologia_1991_2020_geop250_ERA5.nc`).
> Se `SFTP_ENABLED=true`, sera baixado automaticamente do servidor Oracle.
> Caso contrario, copie manualmente para `Entrada/arquivos_nc/`.
> Tambem precisa da legenda (`legenda_atlantic.png`) em `Entrada/` — copie manualmente.

## 7. Ver resultados

- **Mapas s00:** `Saida/s00_VENTO_EOLICAS_SEMOP/`
- **Mapas s01:** `Saida/s01_GEOP250/`
- **Dados baixados:** `dados/`
- **Logs:** `logs/campos_observados.log`

---

## Comandos uteis

```bash
# Forcar re-download dos dados
uv run python run_script.py s00 --force-download

# Sobrescrever datas via CLI
uv run python run_script.py s00 --data-inicial 2026-03-01 --data-final 2026-03-12

# Logging detalhado
uv run python run_script.py s00 --verbose

# Executar todos os scripts habilitados
uv run python run_script.py --all

# Limpar cache e re-processar
uv run python run_script.py --clear-cache
```

### Alternativa: ativar virtualenv

```bash
source .venv/bin/activate
python run_script.py --list
python run_script.py s00
```

## Problemas comuns

| Problema | Solucao |
|----------|---------|
| `ImportError: jinja2` | Execute `uv sync` |
| `FileNotFoundError: climatologia_...nc` | Configure SFTP ou copie manualmente (ver README) |
| `FileNotFoundError: legenda_atlantic.png` | Copie manualmente para `Entrada/` |
| Erro de autenticacao CDS | Verifique `KEY_CDS` no `.secrets.toml` |
| Datas indisponiveis | Ajuste `DATA_FINAL` para ~5 dias antes de hoje |
| `poetry shell` nao encontrado | Use `uv run python` ou `source .venv/bin/activate` |

---

Para documentacao completa, veja o [README.md](README.md).
