# -*- coding: utf-8 -*-
"""s41 - Globo 3D animado (midia) padrao "Google Earth": voo da camera + evolucao temporal.

Copia FIEL do s39 (estilo Guillaume): mesmo motor, mesmas variaveis, mesma saida MP4 para
reanalise E previsao (todos os modelos habilitados). A UNICA diferenca e a PROJECAO 3D: em vez
do globo flutuante geoestacionario, usa a projecao "Google Earth" (NearsidePerspective com a
camera mais perto -> regiao ampliada e com mais curvatura). Atmosfera/estrelas/vinheta ficam
desligadas nesse modo (assumem o disco flutuante centralizado).

  - Variavel:  VARIAVEIS_GLOBO_3D (mesma lista do s38/s39)
  - Modo:      AUTOMATICO pelas datas (passado=reanalise, futuro=previsao, cruza hoje=emenda)
  - Projecao:  GLOBO_3D_PROJECTION_S41 (default 'google_earth') + GLOBO_3D_GE_ALTURA (zoom)
  - Voo:       GLOBO_3D_LON/LAT_INICIAL -> GLOBO_3D_LON/LAT_FINAL (+ VOLTAS_EXTRA)

Saida:
  - Reanalise: Saida/s41_GLOBO_GOOGLE_EARTH/REANALISE/s41_<variavel>.mp4
  - Forecast:  Saida/s41_GLOBO_GOOGLE_EARTH/FORECAST/<MODELO>/s41_<variavel>.mp4

Criado em: 2026-07-01 (copia do s39, muda a projecao 3D)
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
    expandir_variaveis,
    VARIAVEIS,
    gerar_animacao,
)

SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's41'
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
    # Variantes automaticas: z250_anom gera tambem a media movel de 5 dias (dois MP4s).
    return expandir_variaveis(variaveis)


def main():
    logger = get_logger(SCRIPT_ID)
    logger.info('=' * 80)
    logger.info('SCRIPT {}: {}', SCRIPT_ID.upper(), SCRIPT_DESC)
    logger.info('=' * 80)

    variaveis = _get_variaveis()
    output_base = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_GLOBO_GOOGLE_EARTH'
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
        # Projecao "Google Earth" (o que distingue o s41 do s39).
        'projection': str(settings.get('GLOBO_3D_PROJECTION_S41', 'google_earth')),
        'ge_altura': float(settings.get('GLOBO_3D_GE_ALTURA', 5_000_000.0)),
        'ge_aspect': float(settings.get('GLOBO_3D_GE_ASPECT', 0.62)),
        'ge_globo_frac': float(settings.get('GLOBO_3D_GE_GLOBO_FRAC', 1.02)),
        'ge_globo_cy': float(settings.get('GLOBO_3D_GE_GLOBO_CY', 0.29)),
        'credito': str(getattr(settings, 'GLOBO_3D_CREDITO', 'Bruno Capucin')),
        'vinheta': bool(getattr(settings, 'GLOBO_3D_VINHETA', True)),
        'atmosfera': bool(getattr(settings, 'GLOBO_3D_ATMOSFERA', True)),
        'paletas': {v: list(settings.get(f'GLOBO_3D_PALETA_{v.upper()}', []) or []) for v in variaveis},
        'fonte_titulo': str(getattr(settings, 'GLOBO_3D_FONTE_TITULO', '')),
        'fonte_legenda': str(getattr(settings, 'GLOBO_3D_FONTE_LEGENDA', '')),
        'tamanho_px': int(getattr(settings, 'GLOBO_3D_TAMANHO_PX', 1080)),
        'script_version': '1.1-ge-landscape',  # projecao Google Earth calibrada + quadro paisagem
    }

    # Por padrao SEMPRE regenera o MP4 (GLOBO_3D_SEMPRE_REGERAR=true): saida de midia iterada
    # visualmente, e features de aparencia (projecao, box, camera, cores...) NAO entram na chave
    # de cache. Os DOWNLOADS seguem em cache (so o render e refeito). false p/ reativar o skip.
    if not bool(settings.get('GLOBO_3D_SEMPRE_REGERAR', True)) \
            and check_cache_valid(SCRIPT_ID, cache_params, output_files):
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
