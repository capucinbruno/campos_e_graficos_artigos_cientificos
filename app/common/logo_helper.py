# -*- coding: utf-8 -*-
"""Resolucao centralizada do logo dos graficos do projeto.

Prioridade (primeira flag verdadeira vence):
    LOGO_CAPUCIN (logo_capucin.png) > LOGO_GREC (logo_grec.png) > LOGO_AMPERE (novo_logo.png)

A antiga chave SEM_LOGO foi REMOVIDA. Para nao plotar nenhum logo, deixe as tres flags como false
(a funcao retorna None). Centralizar aqui evita repetir a logica nos ~36 scripts.
"""

from __future__ import annotations

from pathlib import Path

from app.shared.settings_factory import settings


def resolve_logo_path(input_dir) -> Path | None:
    """Caminho do logo conforme as flags do settings (ou None = sem logo)."""
    d = Path(input_dir)
    if settings.get('LOGO_CAPUCIN', False):
        return d / 'logo_capucin.png'
    if settings.get('LOGO_GREC', False):
        return d / 'logo_grec.png'
    if settings.get('LOGO_AMPERE', True):
        return d / 'novo_logo.png'
    return None


# Fracao da LARGURA dos eixos que o logo deve ocupar (proporcional a cada area/mapa).
LOGO_WIDTH_FRAC = float(settings.get('LOGO_WIDTH_FRAC', 0.16))


def proportional_logo_zoom(ax, img_w_px, frac: float = LOGO_WIDTH_FRAC) -> float:
    """Zoom do OffsetImage p/ o logo ocupar `frac` da LARGURA dos eixos (independe de dpi).

    Assim a caixa fica do mesmo tamanho RELATIVO em todas as areas de plotagem, em vez de um
    tamanho fixo em pixels (que parece enorme em mapa pequeno e minusculo em mapa grande).
    largura_exibida_px = zoom * img_w_px * dpi/72 ; queremos = frac * largura_eixos_px."""
    fig = ax.figure
    ax_w_in = ax.get_position().width * fig.get_size_inches()[0]
    return max(1e-3, frac * ax_w_in * 72.0 / float(img_w_px))
