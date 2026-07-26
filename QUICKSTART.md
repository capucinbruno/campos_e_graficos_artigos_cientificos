# Quickstart - Campos e Graficos para Artigos Cientificos

Guia rapido para baixar e gerar mapas em **5 minutos**.

---

## 1. Clonar e instalar dependencias

```bash
git clone <url-do-repositorio>
cd campos_e_graficos_artigos_cientificos

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

Projeto compartilhado entre pesquisadores — cada um usa a sua propria chave. Edite
`app/settings/.secrets.toml` (copie de `.secrets_example.toml`) e preencha SO a sua entrada:

```toml
[default]
KEY_CDS_CAPUCIN = "cole-sua-chave-aqui"   # troque CAPUCIN pelo seu sobrenome
```

E diga ao projeto quem voce e — no `settings.local.toml` (proximo passo):

```toml
PESQUISADOR = "capucin"   # capucin | reboita | gozzo | vemado
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

> **Dica:** O ERA5 tem ~5-7 dias de atraso — se `DATA_FINAL` for muito recente, o CDS retorna erro.
> Ajuste para pelo menos 5 dias atras (veja [.claude/rules/gotchas.md](.claude/rules/gotchas.md)).

## 5. Ativar o ambiente e rodar

```bash
source .venv/bin/activate

# Listar scripts disponíveis
python run_script.py --list
```

> Nenhum script vem registrado por padrão neste projeto — siga o [GUIA-NOVOS-SCRIPTS.md](GUIA-NOVOS-SCRIPTS.md)
> para criar o primeiro. Depois de registrado (ex.: `s00`), rode com `python run_script.py s00`.

## 6. Ver resultados

- **Mapas:** `Saida/<script>_<descricao>/`
- **Dados baixados:** `dados/`
- **Logs:** `logs/campos_observados.log`

---

## Comandos uteis

```bash
# Sobrescrever datas via CLI (ignora settings.local.toml)
DYNACONF_SFTP_ENABLED=false python run_script.py <script> --data-inicial 2026-05-20 --data-final 2026-06-03

# Forcar re-download dos dados
DYNACONF_SFTP_ENABLED=false python run_script.py <script> --force-download

# Logging detalhado (traceback completo)
DYNACONF_SFTP_ENABLED=false python run_script.py <script> --verbose

# Executar todos os scripts habilitados
DYNACONF_SFTP_ENABLED=false python run_script.py --all
```

## Problemas comuns

| Problema | Solucao |
|----------|---------|
| `ImportError: jinja2` | Execute `pip install jinja2` no `.venv` |
| `ModuleNotFoundError` | Verifique se o `.venv` esta ativado (`source .venv/bin/activate`) |
| Erro de autenticacao CDS | Verifique `PESQUISADOR` (settings.local.toml) e `KEY_CDS_<NOME>` no `app/settings/.secrets.toml` |
| `DATA_FINAL` muito recente | O script ajusta automaticamente para ontem e avisa no terminal |
| `'Settings' object has no attribute '...'` | Crie `settings.local.toml` a partir do exemplo |

---

Para documentacao completa, veja o [README.md](README.md).
