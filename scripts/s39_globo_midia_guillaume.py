# -*- coding: utf-8 -*-
"""s39 - Globo 3D animado (midia) estilo Guillaume: voo da camera + evolucao temporal.

Gera um video MP4 de uma variavel meteorologica sobre um globo flutuante que
gira/voa em torno do eixo enquanto o campo evolui no tempo. A variavel e o modo
(reanalise/forecast) sao escolhidos via settings — o motor e generico e o
registro de variaveis vive em app/src/uteis/globo_3d_anim.py.

  - Variavel:  VARIAVEL_GLOBO_3D (ex.: 'z250_anom')
  - Modo:      AUTOMATICO pelas datas (passado=reanalise, futuro=previsao, cruza hoje=emenda)
  - Voo:       GLOBO_3D_LON/LAT_INICIAL -> GLOBO_3D_LON/LAT_FINAL (+ VOLTAS_EXTRA)

Saida:
  - Reanalise: Saida/s39_GLOBO_MIDIA/REANALISE/s39_<variavel>.mp4
  - Forecast:  Saida/s39_GLOBO_MIDIA/FORECAST/<MODELO>/s39_<variavel>.mp4

Criado em: 2026-06-26 (copia do s38)
"""

# Bibliotecas padrao
import time
from pathlib import Path

# Modulos locais
from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.src.uteis.globo_3d_anim import (
    _enabled_forecast_models,
    _output_plan,
    VARIAVEIS,
    gerar_animacao,
)

SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's39'
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_ID


def _get_variaveis() -> list[str]:
    """Lista de variaveis a animar. VARIAVEIS_GLOBO_3D (lista) > VARIAVEL_GLOBO_3D
    (singular, compat) > todas as registradas."""
    lst = settings.get('VARIAVEIS_GLOBO_3D', None)
    if lst:
        variaveis = [str(v) for v in lst]
    elif getattr(settings, 'VARIAVEL_GLOBO_3D', None):
        variaveis = [str(settings.VARIAVEL_GLOBO_3D)]
    else:
        variaveis = list(VARIAVEIS.keys())
    invalidas = [v for v in variaveis if v not in VARIAVEIS]
    if invalidas:
        raise ValueError(
            f'Variaveis nao registradas: {invalidas}. Disponiveis: {list(VARIAVEIS.keys())}'
        )
    return variaveis


def main():
    logger = get_logger(SCRIPT_ID)
    logger.info('=' * 80)
    logger.info('SCRIPT {}: {}', SCRIPT_ID.upper(), SCRIPT_DESC)
    logger.info('=' * 80)

    variaveis = _get_variaveis()
    output_base = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_GLOBO_MIDIA'
    # Plano (modo decidido pelas datas): caminhos esperados p/ validar o cache.
    plano, _, _ = _output_plan(variaveis, output_base)
    output_files = [str(item['dir'] / f"{SCRIPT_ID}_{item['var']}.mp4") for item in plano]

    cache_params = {
        'variaveis': variaveis,
        'DATA_INICIAL': str(settings.DATA_INICIAL),
        'DATA_FINAL': str(settings.DATA_FINAL),
        'forecast_init': str(settings.get('FORECAST_INIT', 'latest')),
        'rodada': int(settings.get('RODADA', 0)),
        'modelos': _enabled_forecast_models(),
        'camera': [
            float(getattr(settings, 'GLOBO_3D_LON_INICIAL', -150.0)),
            float(getattr(settings, 'GLOBO_3D_LAT_INICIAL', 0.0)),
            float(getattr(settings, 'GLOBO_3D_LON_FINAL', -45.0)),
            float(getattr(settings, 'GLOBO_3D_LAT_FINAL', -15.0)),
            float(getattr(settings, 'GLOBO_3D_VOLTAS_EXTRA', 0.0)),
        ],
        'inclinacao': str(getattr(settings, 'GLOBO_3D_INCLINACAO', '')),
        'easing': str(getattr(settings, 'GLOBO_3D_EASING', 'linear')),
        'velocidade_var': float(getattr(settings, 'GLOBO_3D_VELOCIDADE_VAR', 1.0)),
        'frames_por_dia': int(getattr(settings, 'GLOBO_3D_FRAMES_POR_DIA', 4)),
        'fps': int(getattr(settings, 'GLOBO_3D_FPS', 20)),
        'grid_deg': float(getattr(settings, 'GLOBO_3D_GRID_DEG', 0.5)),
        'niveis': int(getattr(settings, 'GLOBO_3D_NIVEIS', 16)),
        'niveis_var': {v: settings.get(f'GLOBO_3D_NIVEIS_{v.upper()}', None) for v in variaveis},
        'coarsen': int(getattr(settings, 'GLOBO_3D_COARSEN', 1)),
        'projection': str(getattr(settings, 'GLOBO_3D_PROJECTION', 'nearside')),
        'credito': str(getattr(settings, 'GLOBO_3D_CREDITO', 'Bruno Capucin')),
        'vinheta': bool(getattr(settings, 'GLOBO_3D_VINHETA', True)),
        'atmosfera': bool(getattr(settings, 'GLOBO_3D_ATMOSFERA', True)),
        'paletas': {v: list(settings.get(f'GLOBO_3D_PALETA_{v.upper()}', []) or []) for v in variaveis},
        'fonte_titulo': str(getattr(settings, 'GLOBO_3D_FONTE_TITULO', '')),
        'fonte_legenda': str(getattr(settings, 'GLOBO_3D_FONTE_LEGENDA', '')),
        'tamanho_px': int(getattr(settings, 'GLOBO_3D_TAMANHO_PX', 1080)),
        'script_version': '3.70-guillaume',  # olr_anom: sem periodo de climatologia no canto sup-dir
    }

    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('CACHE VALIDO — pulando execucao ({} arquivo(s))', len(output_files))
        return

    start_time = time.time()
    gerados = gerar_animacao(variaveis, output_base, SCRIPT_ID)

    execution_time = time.time() - start_time
    save_cache_metadata(SCRIPT_ID, cache_params, [str(p) for p in gerados], execution_time)
    logger.info('=' * 80)
    logger.info('Script {} concluido! {} MP4(s) em {:.1f}s', SCRIPT_ID.upper(), len(gerados),
                execution_time)
    for p in gerados:
        logger.info('  {}', p)
    logger.info('=' * 80)


if __name__ == '__main__':
    main()
