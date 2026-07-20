# app/common/config_header.py
# -*- coding: utf-8 -*-
"""Aplica a config dedicada de cada script de MIDIA (s38 em diante) no Dynaconf, em runtime.

Motivacao: esses scripts tem MUITAS features; espalhar centenas de chaves `GLOBO_3D_*` no
settings.local virou um caos. Cada script carrega a SUA config de um TOML dedicado em
`scripts/config_local/<nome-do-script>.toml` (gitignored -- eh parametro editorial, muda a cada
pauta, entao nao deve gerar commit no .py). Sem `.example.toml`/template: mesmo padrao do
settings.local.toml, que tambem parou de manter exemplo sincronizado. Chaves BASE, sem sufixo
`_S<NN>` -- cada script injeta as SUAS. O motor continua lendo `settings.GLOBO_3D_X` normalmente.

Precedencia (do mais forte pro mais fraco):
  1. env `AMPERE_<KEY>`  (override do usuario / smoke test; a CLI --data-inicial tambem seta esse sinal)
  2. `scripts/config_local/<nome-do-script>.toml` (este helper)
  3. defaults do settings.toml (mae) -- NUNCA o settings.local.toml (ver `_neutralizar_settings_local`)

-> por isso o helper NAO sobrescreve nenhuma chave que ja tenha o sinal `AMPERE_<KEY>` no ambiente.

HERMETICO em relacao ao settings.local.toml: um script com config dedicada so deve obedecer a
MAE (identidade fixa: paletas, presets, ENQUADRAMENTOS...) + a SUA PROPRIA config -- nunca uma
chave solta que sobrou ligada no settings.local.toml de outro fluxo/script. Sem isso, ex.:
GLOBO_3D_JATO=true no settings.local vazava pro s46 mesmo sem estar no scripts/config_local/s46_*
(o jato so devia aparecer se o PROPRIO script pedisse). `aplicar_config_header` por isso comeca
resetando pra MAE toda chave que o settings.local.toml toca (exceto as com sinal `AMPERE_<KEY>`,
que continuam vencendo -- CLI/smoke test).
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from dynaconf import Dynaconf

from app.shared.logger import get_logger
from app.shared.settings_factory import settings

logger = get_logger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_MAE_PATH = _ROOT / 'app' / 'settings' / 'settings.toml'
_LOCAL_PATH = _ROOT / 'settings.local.toml'
_SEM_VALOR = object()   # sentinela p/ distinguir "chave ausente na mae" de valor None de verdade

_mae_settings: Dynaconf | None = None   # cache preguicoso (1x por processo)


def _get_mae_settings() -> Dynaconf:
    """Dynaconf carregando SO a mae (settings.toml), mesmo ambiente/prefixo do settings principal
    -- usado pra saber "qual seria o valor sem o settings.local.toml", sem reimplementar o merge
    [default]/[ambiente] do Dynaconf na mao."""
    global _mae_settings
    if _mae_settings is None:
        _mae_settings = Dynaconf(
            envvar_prefix='AMPERE', settings_files=[str(_MAE_PATH)],
            environments=True, load_dotenv=True, root_path=str(_ROOT),
        )
    return _mae_settings


def _chaves_settings_local() -> set[str]:
    """Chaves que settings.local.toml define (qualquer secao -- [default] ou [ambiente])."""
    if not _LOCAL_PATH.exists():
        return set()
    cfg = tomllib.loads(_LOCAL_PATH.read_text(encoding='utf-8'))
    chaves: set[str] = set()
    for nome_secao, valor in cfg.items():
        if isinstance(valor, dict):
            chaves |= set(valor.keys())
    return chaves


def _neutralizar_settings_local() -> None:
    """Reseta pro valor da MAE toda chave que o settings.local.toml sobrescreve -- ANTES de
    aplicar a config do proprio script. Chaves sem sinal `AMPERE_<KEY>` (CLI/env, que continua
    vencendo) e presentes na mae voltam pro default da mae; chaves do settings.local que a mae
    NAO define (raro -- convencao do projeto exige default na mae) ficam como estao, sem reset."""
    mae = _get_mae_settings()
    resetadas: list[str] = []
    for key in _chaves_settings_local():
        if os.environ.get(f'AMPERE_{key}') is not None:
            continue
        valor_mae = mae.get(key, default=_SEM_VALOR)
        if valor_mae is _SEM_VALOR:
            continue
        settings.set(key, valor_mae)
        resetadas.append(key)
    if resetadas:
        logger.debug('Config do script: {} chave(s) do settings.local neutralizada(s) (voltaram '
                     'pro default da mae antes da config propria): {}', len(resetadas), resetadas)


def aplicar_config_header(toml_str: str) -> None:
    """Parseia o bloco TOML `toml_str` e injeta cada chave no `settings`, exceto as que vieram de
    FORA (env `AMPERE_<KEY>` ou CLI, que seta o mesmo sinal) -- essas mantem prioridade.

    Antes de aplicar, neutraliza o settings.local.toml (ver `_neutralizar_settings_local`) -- o
    script fica hermetico: mae + a propria config, nunca o settings.local de outro fluxo."""
    _neutralizar_settings_local()
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


def carregar_config_script(script_path: Path) -> str:
    """Le scripts/config_local/<nome>.toml (gitignored) -- config dedicada do script.
    Sem fallback: se faltar, erro claro pedindo pra criar o arquivo."""
    nome = script_path.stem
    diretorio = script_path.parent / 'config_local'
    local = diretorio / f'{nome}.toml'
    if not local.exists():
        raise FileNotFoundError(
            f'Config do script nao encontrada: {local}\n'
            f'Crie o arquivo (bloco TOML com as chaves GLOBO_3D_*) em {diretorio}.'
        )
    return local.read_text(encoding='utf-8')


def aplicar_config_script(script_path: Path) -> None:
    """Le e aplica a config dedicada do script (scripts/config_local/<nome>.toml)."""
    aplicar_config_header(carregar_config_script(script_path))
