# -*- coding: utf-8 -*-
"""s43 - Mapa 2D PLANO estilo "The Weather Channel" (irmao do s42, sem globo).

Mesma "marca" visual e mesmos recursos do s42 (caixa The Weather Channel, corrente de jato,
icones de pressao animados, caixa de texto livre, credito), mas plotado numa projecao
RETANGULAR FIXA (PlateCarree, tipo carta sinotica do GFS) em vez do globo 3D giratorio com
voo de camera. Motor proprio, mais simples: `app/src/uteis/mapa_2d_anim.py`.

  - Variavel:  GLOBO_3D_VARIAVEIS_S43 (proprio do s43) > VARIAVEIS_GLOBO_3D (compartilhado)
  - Modo:      AUTOMATICO pelas datas (passado=reanalise, futuro=previsao, cruza hoje=emenda)
  - Regiao:    GLOBO_2D_AREA (chave de settings["areas_plotagem"]; default "europa") — FIXA,
               sem voo de camera
  - Transicao: crossfade suave entre passos sinoticos (GLOBO_2D_FRAMES_POR_PASSO)

Saida (TRES arquivos por variavel, em REANALISE/ ou FORECAST/<MODELO>/):
  - s43_<variavel>.mp4        : video do periodo (crossfade entre passos) + jato FLUINDO
  - s43_<variavel>_media.png  : figura da MEDIA do periodo + corrente de jato PARADA (estatica)
  - s43_<variavel>_media.gif  : MEDIA do periodo (campo fixo) + 'JET STREAM'/setas animando W->E
  (mesmas settings globais GLOBO_3D_JATO*/ICONES_PRESSAO/CAIXA_LIVRE do s38-s42 — master unico
   "para todos os globos", agora tambem para o mapa 2D. v1: jato so plota quando a variavel e
   uma altura geopotencial absoluta propria, ex. z250_abs/z500_abs — ver mapa_2d_anim.py.)

Criado em: 2026-07-04 (mapa 2D irmao do s42).
"""

# Bibliotecas padrao
import time
from pathlib import Path

# Modulos locais
from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.src.uteis.globo_3d_anim import VARIAVEIS, expandir_variaveis
from app.src.uteis.mapa_2d_anim import _output_plan, gerar_animacao_2d

SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's43'
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_ID


def _get_variaveis() -> list[str]:
    """Lista de variaveis a plotar, a partir de GLOBO_3D_VARIAVEIS_S43 (lista PROPRIA do s43).

    Lista VAZIA `[]` (default do master): modo "SEM VARIAVEL" — plota o mapa 2D SO com as
    features habilitadas (jato, icones de pressao, caixas de texto, credito), sem nenhum campo
    shaded (ficha sintetica 'sem_variavel' em globo_3d_anim.py, mesma usada pelo s42).
    """
    lst = settings.get('GLOBO_3D_VARIAVEIS_S43', None)
    if not lst:
        return ['sem_variavel']
    variaveis = [str(v) for v in lst]
    invalidas = [v for v in variaveis if v not in VARIAVEIS]
    if invalidas:
        raise ValueError(
            f'Variaveis nao registradas: {invalidas}. Disponiveis: {list(VARIAVEIS.keys())}'
        )
    return expandir_variaveis(variaveis)


def main():
    logger = get_logger(SCRIPT_ID)
    logger.info('=' * 80)
    logger.info('SCRIPT {}: {}', SCRIPT_ID.upper(), SCRIPT_DESC)
    logger.info('=' * 80)

    variaveis = _get_variaveis()
    output_base = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_GLOBO_WEATHER_CHANNEL_2D'
    plano, _, _ = _output_plan(variaveis, output_base)
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
        'area': str(settings.get('GLOBO_2D_AREA', 'europa')),
        'frames_por_passo': int(settings.get('GLOBO_2D_FRAMES_POR_PASSO', 5)),
        'velocidade_var': float(settings.get('GLOBO_2D_VELOCIDADE_VAR', 1.0)),
        'fps': int(settings.get('GLOBO_2D_FPS', 8)),
        'figsize': [float(settings.get('GLOBO_2D_FIGSIZE_W', 15.0)),
                    float(settings.get('GLOBO_2D_FIGSIZE_H', 10.0))],
        'dpi': int(settings.get('GLOBO_2D_DPI', 150)),
        'cauda_seg': float(settings.get('GLOBO_2D_JATO_MP4_CAUDA_SEG', 0.0)),
        'niveis_var': {v: settings.get(f'GLOBO_3D_NIVEIS_{v.upper()}', None) for v in variaveis},
        'vmin_var': {v: settings.get(f'GLOBO_3D_VMIN_{v.upper()}', None) for v in variaveis},
        'vmax_var': {v: settings.get(f'GLOBO_3D_VMAX_{v.upper()}', None) for v in variaveis},
        'jato': bool(settings.get('GLOBO_3D_JATO', False)),
        'icones_pressao': list(settings.get('GLOBO_3D_ICONES_PRESSAO', []) or []),
        'caixa_livre': bool(settings.get('GLOBO_3D_CAIXA_LIVRE', False)),
        'credito': str(settings.get('GLOBO_3D_CREDITO', 'Bruno Capucin')),
        'titulo_twc': str(settings.get('TITULO_THE_WEATHER_CHANNEL', '')),
        'script_version': '1.0-2d-flat',
    }

    if not bool(settings.get('GLOBO_3D_SEMPRE_REGERAR', True)) \
            and check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('CACHE VALIDO — pulando execucao ({} arquivo(s))', len(output_files))
        return

    start_time = time.time()
    gerados = gerar_animacao_2d(variaveis, output_base, SCRIPT_ID)

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
