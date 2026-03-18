---
description: Armadilhas e comportamentos nao-obvios do projeto campos_observados_era5
---

# Gotchas - Campos Observados ERA5

## Datetime Naive

O projeto NAO usa timezone. Tudo e tratado como horario de Brasilia (UTC-3) implicitamente. NAO adicionar `tzinfo` aos datetimes existentes.

## SFTP Rate Limit

O servidor Oracle limita a 4 conexoes SFTP simultaneas novas (iptables). Nao abrir multiplas conexoes em paralelo.

## CDS API Latencia

O ERA5 tem ~5 dias de atraso. Se pedir dados de ontem, o CDS retornara erro. Ajustar `DATA_FINAL` para pelo menos 5 dias antes de hoje.

## Climatologia via Settings

O caminho da climatologia e configuravel via `FILE_CLIMATOLOGIA_VENTO100M` no settings.toml. NAO usar path hardcoded no script. Sempre ler do settings:

```python
# CERTO
clim_path = Path(settings.FILE_CLIMATOLOGIA_VENTO100M)

# ERRADO
clim_path = Path("Entrada/arquivos_nc/climatologia_1991_2020_vento100m_ERA5.nc")
```

## Arquivos .nc/.grb Corrompidos

O `run_script.py` tem logica de retry para arquivos de suporte corrompidos:
1. Valida com `xr.open_dataset()` antes de executar
2. Se corrompido + SFTP habilitado → apaga e re-baixa 1x
3. Se ainda corrompido → erro claro com instrucoes

NAO adicionar validacao extra nos scripts — o CLI ja faz.

## @friendly_errors Captura Tudo

O decorator `@friendly_errors` no `main()` do CLI captura TODAS as excecoes nao tratadas. NAO adicionar try/except generico nos scripts — deixe os erros subirem naturalmente.

Se precisar de mensagem amigavel para um erro especifico, adicione em `_ERROR_HINTS` no `app/shared/error_handler.py`.

## Entrada/ vs dados/

- `Entrada/` e para arquivos fixos (logos, legendas, climatologias)
- `dados/` e para dados baixados do CDS (arquivos .nc/.grb)

NAO salvar downloads do CDS em `Entrada/`. Os downloaders usam `settings.DIR_DADOS` como base.

## settings.local.toml Nao E Versionado

O `settings.local.toml` e gitignored. Ao adicionar nova setting:
1. Adicione o valor default no `settings.toml`
2. Adicione comentario no `settings.local.example.toml`
3. Atualize o `settings.local.toml` do desenvolvedor (ou avise)

## Cache Invalido

Se mudar a logica de processamento de um script, incremente o `script_version` nos `cache_params` para forcar reprocessamento:

```python
cache_params = {
    ...
    "script_version": "1.1",  # incrementar quando mudar logica
}
```
