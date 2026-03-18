---
name: tratamento-de-erros
description: Como funciona o tratamento de erros amigaveis no projeto campos_observados_era5. Explica o decorator @friendly_errors, o mapa _ERROR_HINTS e como adicionar novos erros conhecidos.
---

# Tratamento de Erros Amigaveis

O projeto usa um **handler global de excecoes** no entry point do CLI. Nao precisa de try/except espalhado pelo codigo.

## Problema que resolve

Tracebacks longos do Python assustam desenvolvedores juniors e nao ajudam meteorologistas. Em vez de 50 linhas de pilha de chamadas, o usuario ve:

```
ERRO: Arquivo nao encontrado no servidor SFTP

Solucao:
  Verifique se o caminho remoto esta correto no SCRIPTS dict
  ou copie o arquivo manualmente para o caminho local esperado.

Dica: use --verbose para ver o traceback completo
```

## Como funciona

### Principio: funil de excecoes

Em Python, toda excecao nao tratada sobe ate o topo da pilha de chamadas. Nao importa se o erro aconteceu 10 niveis abaixo — ele vai subir ate encontrar um `try/except` ou chegar ao topo e matar o processo.

O decorator `@friendly_errors` fica no entry point (`main()`) e intercepta TUDO ali. E como um detector de fumaca no corredor do predio: nao precisa de um em cada gaveta.

```
Script s00 (nivel 3)  ──┐
  Downloader (nivel 2)  ─┤  excecao sobe automaticamente
    Paramiko (nivel 1)  ──┘
                          ▼
    @friendly_errors (nivel 0)  ← intercepta aqui
```

### Arquitetura

```
app/shared/error_handler.py    ← modulo com decorator e mapa de erros
app/cli/run_script.py          ← @friendly_errors no main()
```

Tres componentes:

1. **`_ERROR_HINTS`** — Lista de tuplas `(tipo_excecao, substring, dica)`. O "cerebro" que traduz erros tecnicos em mensagens uteis.

2. **`_find_hint(exc)`** — Percorre a lista procurando match por tipo + substring na mensagem.

3. **`@friendly_errors`** — Decorator que envolve `main()`. Captura excecoes, busca hint, formata saida colorida.

### Fluxo de decisao

```
Excecao sobe ate main()
    │
    ├─ SystemExit? → re-raise (argparse usa isso)
    ├─ KeyboardInterrupt? → "Execucao interrompida" + exit 130
    └─ Exception?
        │
        ├─ Encontrou hint em _ERROR_HINTS?
        │   ├─ Sim → mostra ERRO + Solucao
        │   └─ Nao → mostra ERRO generico
        │
        └─ --verbose ativo?
            ├─ Sim → mostra traceback completo
            └─ Nao → "use --verbose para traceback"
```

## Como usar no codigo

### No entry point (ja configurado)

```python
from app.shared.error_handler import friendly_errors

@friendly_errors
def main():
    ...
```

Pronto. Nada mais precisa ser feito nos scripts.

### Nos scripts: NAO faca try/except generico

**Errado** — engole o erro e impede o handler global de atuar:

```python
def main():
    try:
        ds = xr.open_dataset("arquivo.nc")
    except Exception as e:
        print(f"Erro: {e}")  # mensagem ruim, sem solucao
        return  # handler global nunca ve o erro
```

**Certo** — deixe o erro subir naturalmente:

```python
def main():
    ds = xr.open_dataset("arquivo.nc")  # se falhar, sobe pro handler
```

**Aceitavel** — captura para adicionar contexto, depois re-lanca:

```python
def main():
    try:
        ds = xr.open_dataset(path)
    except OSError:
        raise RuntimeError(f"Falha ao abrir climatologia: {path}") from None
```

O `raise` garante que o erro continua subindo ate o handler global.

### Regra geral

| Situacao | O que fazer |
|----------|-------------|
| Erro generico (IO, rede, parse) | Deixe subir. O handler trata. |
| Precisa adicionar contexto | `raise NovaMensagem from None` |
| Erro recuperavel (retry, fallback) | `try/except` local e OK |
| Validacao de entrada | `raise ValueError("mensagem clara")` |

## Como adicionar novos erros conhecidos

Edite `app/shared/error_handler.py`, adicione uma tupla em `_ERROR_HINTS`:

```python
_ERROR_HINTS: list[tuple[type, str | None, str]] = [
    # ... erros existentes ...

    # Novo erro: quando o CDS retorna 503 (indisponivel)
    (
        Exception,                    # tipo da excecao (ou subclasse)
        "503",                        # substring na mensagem (ou None para qualquer)
        "API do Copernicus CDS indisponivel (HTTP 503).\n"
        "  Tente novamente em alguns minutos.\n"
        "  Status: https://cds.climate.copernicus.eu/",
    ),
]
```

**Campos:**
- `tipo_excecao`: classe da excecao (use `Exception` se nao souber o tipo exato)
- `substring`: texto que deve existir na mensagem do erro (case-sensitive). Use `None` para capturar qualquer mensagem desse tipo.
- `dica`: mensagem amigavel com solucao. Use `\n` para quebras de linha.

**Ordem importa:** a primeira tupla que fizer match e usada. Coloque erros mais especificos antes dos genericos.

## Erros ja mapeados

| Categoria | Substring | Solucao resumida |
|-----------|-----------|------------------|
| SFTP | `No such file` | Caminho remoto incorreto ou arquivo inexistente |
| SFTP | `Arquivo obrigatorio` | Copiar arquivo manualmente |
| SSH | `ConnectionRefused` | Verificar host/porta |
| SSH | `Authentication failed` | Verificar usuario/chave |
| SSH | `No such file or directory` | Chave .pem nao encontrada |
| SSH | `TimeoutError` | Servidor offline |
| CDS | `KEY_CDS` | Preencher chave no .secrets.toml |
| CDS | `403` | Chave invalida/expirada |
| NetCDF | `Unknown file format` | Arquivo corrompido, re-baixar |
| NetCDF | `climatologia` | Climatologia corrompida |
| Import | `jinja2` | Executar `uv sync` |
| Import | `ModuleNotFoundError` | Executar `uv sync` ou ativar venv |

## Testando

```bash
# Saida amigavel (padrao)
uv run python run_script.py s00

# Traceback completo para debug
uv run python run_script.py s00 --verbose
```
