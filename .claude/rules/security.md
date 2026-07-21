---
description: Regras de seguranca do projeto campos_observados_era5
---

# Seguranca - Campos Observados ERA5

## Gestao de Secrets

### Arquivos Sensiveis (git-ignored)

| Arquivo | Conteudo | Exemplo |
|---------|----------|---------|
| `app/settings/.secrets.toml` | KEY_CDS, SSH credentials | `.secrets_example.toml` |
| `.env` | `ENV_FOR_DYNACONF` | `.env.example` |
| `settings.local.toml` | Datas, flags — pode conter KEY_CDS | `settings.local.example.toml` |

### Credenciais

| Credencial | Onde configurar | Para que |
|------------|----------------|----------|
| `KEY_CDS` | `.secrets.toml` ou `settings.local.toml` | API Copernicus CDS |
| `SSH_HOST` | `.secrets.toml` | Servidor SFTP Oracle |
| `SSH_USERNAME` | `.secrets.toml` | Usuario SSH |
| `SSH_KEY_PATH` | `.secrets.toml` | Chave SSH (.pem) |
| `SSH_PORT` | `.secrets.toml` | Porta SSH (default 22) |
| `EARTHDATA_TOKEN` | `.secrets.toml` | NASA Earthdata (IMERG-GPM via earthaccess, s49) |

### Chaves SSH

| Chave | Uso | Permissao |
|-------|-----|-----------|
| `~/.ssh/meteorologia-oracle-sp.pem` | SFTP para servidor de dados | 600 ou 400 |

## Conexoes de Rede

| Servico | Host | Porta | Protocolo |
|---------|------|-------|-----------|
| SFTP Files | Configurado em SSH_HOST | 22 | SSH/SFTP |
| Copernicus CDS | cds.climate.copernicus.eu | 443 | HTTPS |
| CPTEC MERGE (FTP-HTTPS) | ftp.cptec.inpe.br | 443 | HTTPS |
| NASA GES DISC (IMERG) | gesdisc.eosdis.nasa.gov | 443 | HTTPS (Bearer token) |

## Regras ao Modificar

1. **NUNCA** commitar `.secrets.toml` ou `.env` com valores reais
2. **SEMPRE** atualizar arquivo de exemplo ao adicionar nova secret
3. **NUNCA** logar senhas, tokens ou chaves SSH em nenhum nivel de log
4. Manter `check-added-large-files` no pre-commit (10MB max)
5. Arquivos `.nc`, `.grb` e dados binarios estao no `.gitignore` — nao remover
6. `KEY_CDS` e pessoal — cada desenvolvedor deve obter a sua em https://cds.climate.copernicus.eu/
