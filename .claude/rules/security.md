---
description: Regras de seguranca do projeto campos_e_graficos_artigos_cientificos
---

# Seguranca - Campos e Graficos para Artigos Cientificos

## Gestao de Secrets

### Arquivos Sensiveis (git-ignored)

| Arquivo | Conteudo | Exemplo |
|---------|----------|---------|
| `app/settings/.secrets.toml` | KEY_CDS_<NOME> (uma por pesquisador), SSH credentials | `.secrets_example.toml` |
| `.env` | `ENV_FOR_DYNACONF` | `.env.example` |
| `settings.local.toml` | PESQUISADOR, datas, flags | `settings.local.example.toml` |

### Credenciais

Projeto compartilhado entre pesquisadores — cada um tem sua propria `KEY_CDS_<NOME>` em
`.secrets.toml`; a setting `PESQUISADOR` (`settings.local.toml`) escolhe qual usar. Nomes
aceitos: `capucin`, `reboita`, `gozzo`, `vemado` (ver `PESQUISADORES_VALIDOS` em
`app/src/uteis/downloaders_era5_generico.py`).

| Credencial | Onde configurar | Para que |
|------------|----------------|----------|
| `PESQUISADOR` | `settings.local.toml` | Seleciona qual `KEY_CDS_<NOME>` usar |
| `KEY_CDS_CAPUCIN` / `_REBOITA` / `_GOZZO` / `_VEMADO` | `.secrets.toml` | API Copernicus CDS (uma por pesquisador) |
| `SSH_HOST` | `.secrets.toml` | Servidor SFTP Oracle |
| `SSH_USERNAME` | `.secrets.toml` | Usuario SSH |
| `SSH_KEY_PATH` | `.secrets.toml` | Chave SSH (.pem) |
| `SSH_PORT` | `.secrets.toml` | Porta SSH (default 22) |

### Chaves SSH

| Chave | Uso | Permissao |
|-------|-----|-----------|
| `~/.ssh/meteorologia-oracle-sp.pem` | SFTP para servidor de dados | 600 ou 400 |

## Conexoes de Rede

| Servico | Host | Porta | Protocolo |
|---------|------|-------|-----------|
| SFTP Files | Configurado em SSH_HOST | 22 | SSH/SFTP |
| Copernicus CDS (ERA5) | cds.climate.copernicus.eu | 443 | HTTPS |
| NOMADS (GDAS) | nomads.ncep.noaa.gov | 443 | HTTPS |

## Regras ao Modificar

1. **NUNCA** commitar `.secrets.toml` ou `.env` com valores reais
2. **SEMPRE** atualizar arquivo de exemplo ao adicionar nova secret
3. **NUNCA** logar senhas, tokens ou chaves SSH em nenhum nivel de log
4. Manter `check-added-large-files` no pre-commit (10MB max)
5. Arquivos `.nc`, `.grb` e dados binarios estao no `.gitignore` — nao remover
6. `KEY_CDS_<NOME>` e pessoal — cada pesquisador deve obter a sua em https://cds.climate.copernicus.eu/ e preencher so a sua entrada em `.secrets.toml`
