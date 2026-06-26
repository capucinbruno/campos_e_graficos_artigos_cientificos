# -*- coding: utf-8 -*-
"""s38 - Globo 3D animado (midia): voo da camera + evolucao temporal.

Gera um video MP4 de uma variavel meteorologica sobre um globo flutuante que
gira/voa em torno do eixo enquanto o campo evolui no tempo. A variavel e o modo
(reanalise/forecast) sao escolhidos via settings — o motor e generico e o
registro de variaveis vive em app/src/uteis/globo_3d_anim.py.

  - Variavel:  VARIAVEL_GLOBO_3D (ex.: 'z250_anom')
  - Modo:      GLOBO_3D_MODO ('reanalise' | 'forecast')
  - Voo:       GLOBO_3D_LON/LAT_INICIAL -> GLOBO_3D_LON/LAT_FINAL (+ VOLTAS_EXTRA)

Saida:
  - Reanalise: Saida/s38_GLOBO_MIDIA/REANALISE/s38_<variavel>.mp4
  - Forecast:  Saida/s38_GLOBO_MIDIA/FORECAST/<MODELO>/s38_<variavel>.mp4

Criado em: 2026-06-25
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
    VARIAVEIS,
    gerar_animacao,
)

SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's38'
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_ID


def _expected_outputs(variavel: str, modo: str, output_base: Path) -> list[str]:
    """Caminhos esperados dos MP4 — usados para validar o cache antes de rodar."""
    if modo.startswith('rean'):
        return [str(output_base / 'REANALISE' / f's38_{variavel}.mp4')]
    return [
        str(output_base / 'FORECAST' / m.upper() / f's38_{variavel}.mp4')
        for m in _enabled_forecast_models()
    ]


def main():
    logger = get_logger(SCRIPT_ID)
    logger.info('=' * 80)
    logger.info('SCRIPT {}: {}', SCRIPT_ID.upper(), SCRIPT_DESC)
    logger.info('=' * 80)

    variavel = str(getattr(settings, 'VARIAVEL_GLOBO_3D', 'z250_anom'))
    if variavel not in VARIAVEIS:
        raise ValueError(
            f"VARIAVEL_GLOBO_3D='{variavel}' nao registrada. "
            f'Disponiveis: {list(VARIAVEIS.keys())}'
        )
    modo = str(getattr(settings, 'GLOBO_3D_MODO', 'reanalise')).lower()

    output_base = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_GLOBO_MIDIA'
    output_files = _expected_outputs(variavel, modo, output_base)

    cache_params = {
        'variavel': variavel,
        'modo': modo,
        'DATA_INICIAL': str(settings.DATA_INICIAL),
        'DATA_FINAL': str(settings.DATA_FINAL),
        'forecast_init': str(settings.get('FORECAST_INIT', 'latest')),
        'rodada': int(settings.get('RODADA', 0)),
        'modelos': _enabled_forecast_models() if modo.startswith('fore') else [],
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
        'coarsen': int(getattr(settings, 'GLOBO_3D_COARSEN', 1)),
        'projection': str(getattr(settings, 'GLOBO_3D_PROJECTION', 'nearside')),
        'credito': str(getattr(settings, 'GLOBO_3D_CREDITO', 'Bruno Capucin')),
        'vinheta': bool(getattr(settings, 'GLOBO_3D_VINHETA', True)),
        'paleta': list(settings.get('GLOBO_3D_PALETA', []) or []),
        'fonte_titulo': str(getattr(settings, 'GLOBO_3D_FONTE_TITULO', '')),
        'fonte_legenda': str(getattr(settings, 'GLOBO_3D_FONTE_LEGENDA', '')),
        'tamanho_px': int(getattr(settings, 'GLOBO_3D_TAMANHO_PX', 1080)),
        'script_version': '1.6',  # forecast respeita DATA_INICIAL/DATA_FINAL (janela)
    }

    if check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('CACHE VALIDO — pulando execucao ({} arquivo(s))', len(output_files))
        return

    start_time = time.time()
    gerados = gerar_animacao(variavel, output_base)

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
