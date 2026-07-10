# -*- coding: utf-8 -*-
"""s42 - Globo 3D animado (midia) estilo "The Weather Channel".

Copia do s41 (Google Earth) como base (mesmo motor/projecao), mas plota campos ABSOLUTOS (nao
anomalia) — comecando por z250_abs (altura geopotencial 250 hPa). O MP4 anda em HORA SINOTICA
(00/06/12/18 UTC) em vez de 1 passo por dia, pra reanalise E previsao; o GIF/PNG continuam
media do periodo, como sempre. Visual ainda sera editado iterativamente ("The Weather Channel").

  - Variavel:  GLOBO_3D_VARIAVEIS_S42 (proprio do s42) > VARIAVEIS_GLOBO_3D (compartilhado)
  - Modo:      AUTOMATICO pelas datas (passado=reanalise, futuro=previsao, cruza hoje=emenda)
  - Projecao:  GLOBO_3D_PROJECTION_S41 (default 'google_earth') + GLOBO_3D_GE_ALTURA (zoom)
  - Voo:       GLOBO_3D_LON/LAT_INICIAL -> GLOBO_3D_LON/LAT_FINAL (+ VOLTAS_EXTRA)

Saida (TRES arquivos por variavel, em REANALISE/ ou FORECAST/<MODELO>/):
  - s42_<variavel>.mp4        : video do periodo em HORA SINOTICA (fichas com sinotico_mp4=True)
                                ou dia a dia (demais fichas) + jato FLUINDO
  - s42_<variavel>_media.png  : figura da MEDIA do periodo + corrente de jato PARADA (estatica)
  - s42_<variavel>_media.gif  : MEDIA do periodo (campo fixo) + 'JET STREAM'/setas animando W->E
  (com GLOBO_3D_JATO=true — master unico dos globos — o jato e plotado em QUALQUER campo e nas TRES
   saidas: fluindo no MP4, parado no PNG, animado no GIF. Precisa do Z250; se o modelo nao tiver, avisa e segue sem jato.
   Para fichas Z250-absoluto, o campo-guia do jato reusa o proprio campo shaded, ja suavizado.)

Criado em: 2026-07-03 (copia do s41). Atualizado 2026-07-03: z250_abs + MP4 em hora sinotica.
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

SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's42'
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_ID


def _get_variaveis() -> list[str]:
    """Lista de variaveis a animar. GLOBO_3D_VARIAVEIS_S42 (proprio do s42) > VARIAVEIS_GLOBO_3D
    (compartilhado com s38-s41) > VARIAVEL_GLOBO_3D (singular, compat) > todas as registradas.

    O s42 tem lista PROPRIA porque plota campos ABSOLUTOS (ex.: z250_abs) — fisicamente
    diferente do que s38-s41 plotam (anomalia); forcar a mesma lista global impediria configurar
    variaveis diferentes por script.
    """
    lst = settings.get('GLOBO_3D_VARIAVEIS_S42', None) or settings.get('VARIAVEIS_GLOBO_3D', None)
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
    output_base = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_GLOBO_WEATHER_CHANNEL'
    # Plano (modo decidido pelas datas): caminhos esperados p/ validar o cache.
    plano, _, _ = _output_plan(variaveis, output_base)
    # Tres saidas por variavel: MP4 do periodo + PNG da media + GIF da media.
    output_files = []
    for item in plano:
        _base = item['dir'] / f"{SCRIPT_ID}_{item['var']}"
        output_files += [f'{_base}.mp4', f'{_base}_media.png', f'{_base}_media.gif']

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
        'vmin_var': {v: settings.get(f'GLOBO_3D_VMIN_{v.upper()}', None) for v in variaveis},
        'vmax_var': {v: settings.get(f'GLOBO_3D_VMAX_{v.upper()}', None) for v in variaveis},
        'alpha_var': {v: settings.get(f'GLOBO_3D_ALPHA_{v.upper()}', None) for v in variaveis},
        'coarsen': int(getattr(settings, 'GLOBO_3D_COARSEN', 1)),
        # Projecao "Google Earth" (herdada do s41 por enquanto).
        'projection': str(settings.get('GLOBO_3D_PROJECTION_S41', 'google_earth')),
        'ge_altura': float(settings.get('GLOBO_3D_GE_ALTURA', 5_000_000.0)),
        'ge_aspect': float(settings.get('GLOBO_3D_GE_ASPECT', 0.62)),
        'ge_globo_frac': float(settings.get('GLOBO_3D_GE_GLOBO_FRAC', 1.02)),
        'ge_globo_cy': float(settings.get('GLOBO_3D_GE_GLOBO_CY', 0.29)),
        # Tres saidas + corrente de jato (nas TRES: fluindo no MP4, parada no PNG, animada no GIF).
        # Inclui as sub-flags (nao so o master) -- sem isso, mudar p.ex. SUBTROPICAL_JET_NIVEL nao
        # invalidava o cache e o script reaproveitava a saida antiga sem re-renderizar o jato.
        'jato': bool(settings.get('GLOBO_3D_JATO', False)),
        'jato_jet_stream': bool(settings.get('GLOBO_3D_JET_STREAM', True)),
        'jato_jet_stream_nivel': float(settings.get('GLOBO_3D_JET_STREAM_NIVEL', 10200.0)),
        'jato_jet_stream_cor': str(settings.get('GLOBO_3D_JET_STREAM_COR', '#1787ad')),
        'jato_jet_stream_texto': str(settings.get('GLOBO_3D_JET_STREAM_TEXTO', 'JET STREAM')),
        'jato_jet_stream_velocidade': float(settings.get('GLOBO_3D_JET_STREAM_VELOCIDADE', 1.0)),
        'jato_subtropical': bool(settings.get('GLOBO_3D_SUBTROPICAL_JET', False)),
        'jato_subtropical_nivel': float(settings.get('GLOBO_3D_SUBTROPICAL_JET_NIVEL', 10600.0)),
        'jato_subtropical_cor': str(settings.get('GLOBO_3D_SUBTROPICAL_JET_COR', '#2e8b57')),
        'jato_subtropical_texto': str(settings.get('GLOBO_3D_SUBTROPICAL_JET_TEXTO', 'SUBTROPICAL JET')),
        'jato_subtropical_velocidade': float(settings.get('GLOBO_3D_SUBTROPICAL_JET_VELOCIDADE', 1.0)),
        'jato_hemisferio_sul': bool(settings.get('GLOBO_3D_JATO_HEMISFERIO_SUL', True)),
        'jato_hemisferio_norte': bool(settings.get('GLOBO_3D_JATO_HEMISFERIO_NORTE', True)),
        'jato_lat_min': float(settings.get('GLOBO_3D_JATO_LAT_MIN', 15.0)),
        'jato_lat_max': float(settings.get('GLOBO_3D_JATO_LAT_MAX', 60.0)),
        'jato_drape': bool(settings.get('GLOBO_3D_JATO_DRAPE', False)),
        'gif_frames': int(settings.get('GLOBO_3D_GE_GIF_FRAMES', 48)),
        'gif_fps': int(settings.get('GLOBO_3D_GE_GIF_FPS', 12)),
        'credito': str(getattr(settings, 'GLOBO_3D_CREDITO', 'Bruno Capucin')),
        'vinheta': bool(getattr(settings, 'GLOBO_3D_VINHETA', True)),
        'atmosfera': bool(getattr(settings, 'GLOBO_3D_ATMOSFERA', True)),
        'paletas': {v: list(settings.get(f'GLOBO_3D_PALETA_{v.upper()}', []) or []) for v in variaveis},
        'fonte_titulo': str(getattr(settings, 'GLOBO_3D_FONTE_TITULO', '')),
        'fonte_legenda': str(getattr(settings, 'GLOBO_3D_FONTE_LEGENDA', '')),
        'tamanho_px': int(getattr(settings, 'GLOBO_3D_TAMANHO_PX', 1080)),
        'script_version': '1.2-jato-cache-completo',  # cache_params do jato completo (nivel/cor/hemisferio)
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
    logger.info('Script {} concluido! {} arquivo(s) [mp4 + png + gif por variavel] em {:.1f}s',
                SCRIPT_ID.upper(), len(gerados), execution_time)
    for p in gerados:
        logger.info('  {}', p)
    logger.info('=' * 80)


if __name__ == '__main__':
    main()
