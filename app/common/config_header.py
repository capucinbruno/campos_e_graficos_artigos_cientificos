# app/common/config_header.py
# -*- coding: utf-8 -*-
"""Aplica a config do "topo do script" (bloco TOML embutido no proprio .py) no Dynaconf, em runtime.

Motivacao: os scripts de MIDIA (s38 em diante) tem MUITAS features; espalhar centenas de chaves
`GLOBO_3D_*` no settings.local virou um caos. A partir do s38, cada script carrega a SUA config num
bloco TOML no topo do arquivo (co-localizada, sem sufixo `_S<NN>` -- cada script injeta as chaves
BASE com os SEUS valores). O motor continua lendo `settings.GLOBO_3D_X` normalmente.

Precedencia (do mais forte pro mais fraco):
  1. env `AMPERE_<KEY>`  (override do usuario / smoke test; a CLI --data-inicial tambem seta esse sinal)
  2. config do HEADER do script (este helper)
  3. defaults do settings.toml (mae) / da ficha

-> por isso o helper NAO sobrescreve nenhuma chave que ja tenha o sinal `AMPERE_<KEY>` no ambiente.
"""

from __future__ import annotations

import os
import tomllib

from app.shared.logger import get_logger
from app.shared.settings_factory import settings

logger = get_logger(__name__)


def aplicar_config_header(toml_str: str) -> None:
    """Parseia o bloco TOML `toml_str` e injeta cada chave no `settings`, exceto as que vieram de
    FORA (env `AMPERE_<KEY>` ou CLI, que seta o mesmo sinal) -- essas mantem prioridade."""
    cfg = tomllib.loads(toml_str)
    aplicadas = 0
    puladas: list[str] = []
    for key, val in cfg.items():
        if os.environ.get(f'AMPERE_{key}') is not None:   # override externo vence
            puladas.append(key)
            continue
        settings.set(key, val)
        aplicadas += 1
    if puladas:
        logger.info('Config do header: {} chave(s) aplicada(s); {} mantida(s) por override externo: {}',
                    aplicadas, len(puladas), puladas)
    else:
        logger.debug('Config do header aplicada: {} chave(s).', aplicadas)
