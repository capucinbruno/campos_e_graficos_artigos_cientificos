# -*- coding: utf-8 -*-
"""s44 - Globo 3D INCLINADO ("deitado", estilo Google Earth com pitch): voo + evolucao temporal.

Copia fiel do s39 (estilo Guillaume): MESMO motor, MESMAS variaveis e MESMA saida (MP4 + PNG da
media + GIF) para reanalise E previsao (todos os modelos habilitados), com o mesmo pipeline de
dados do s39/s42 (reanalise, previsao, emenda observado+previsao). A UNICA diferenca e o
ENQUADRAMENTO: em vez do globo flutuante visto de cima, a camera fica "deitada"/inclinada -- o
horizonte curva so no topo do quadro e o chao preenche o resto (efeito Google Earth quando se
inclina a camera). Tecnica: janela de recorte descentralizada na NearsidePerspective (ver
`_aplicar_janela_inclinada` em globo_3d_anim.py).

  - Variavel:  UNIAO de VARIAVEIS_GLOBO_3D (s38/s39) + GLOBO_3D_VARIAVEIS_S42 (s42), sem duplicar
  - Modo:      AUTOMATICO pelas datas (passado=reanalise, futuro=previsao, cruza hoje=emenda)
  - Voo:       GLOBO_3D_LON/LAT_INICIAL -> GLOBO_3D_LON/LAT_FINAL (herdado do s39), inclinado

Config PROPRIA (namespace GLOBO_3D_INC_*, NAO afeta s38-s43):
  - GLOBO_3D_INC_ALTURA      : altura da camera em metros (curvatura; menor = mais perto)
  - GLOBO_3D_INC_INCLINACAO  : o quanto "deita" (desloca a janela p/ baixo do nadir; fracao do raio)
  - GLOBO_3D_INC_JANELA_FRAC : zoom da janela (largura / diametro do disco)
  - GLOBO_3D_INC_ASPECT      : altura/largura do quadro (paisagem; 0.5625 = 16:9)
  Qualquer GLOBO_3D_INC_<X> tambem sobrepoe o GLOBO_3D_<X> compartilhado SO no s44 (ex.:
  GLOBO_3D_INC_LON_INICIAL muda o voo do s44 sem tocar no s39).

Saida (TRES arquivos por variavel, em REANALISE/ ou FORECAST/<MODELO>/):
  - s44_<variavel>.mp4        : video do periodo (voo inclinado + evolucao dia a dia) + jato FLUINDO
  - s44_<variavel>_media.png  : figura da MEDIA do periodo + corrente de jato PARADA
  - s44_<variavel>_media.gif  : MEDIA do periodo (campo fixo) + 'JET STREAM'/setas animando W->E

Criado em: 2026-07-13 (copia do s39/s41, muda so o enquadramento p/ inclinado).
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

SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's44'
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_ID


def _get_variaveis() -> list[str]:
    """UNIAO das DUAS listas: VARIAVEIS_GLOBO_3D (compartilhada com s38-s41) + GLOBO_3D_VARIAVEIS_S42
    (propria do s42) — sem duplicar e preservando a ordem (primeiro as do s39, depois as do s42).
    Assim o s44 gera tudo o que o s39 E o s42 gerariam num unico run.

    UMA lista vazia (ausente, [], [""], so espacos) simplesmente NAO contribui — o s44 roda normal
    com o que houver na OUTRA lista. Se as DUAS ficarem vazias, levanta erro claro pedindo pra por
    ao menos uma variavel numa das listas. 'sem_variavel' (globo so com features) pode entrar
    explicitamente em qualquer uma."""
    variaveis: list[str] = []
    for _lst in (settings.get('VARIAVEIS_GLOBO_3D', None),
                 settings.get('GLOBO_3D_VARIAVEIS_S42', None)):
        for v in (_lst or []):
            _v = str(v).strip()
            if _v and _v not in variaveis:   # ignora strings vazias (ex.: lista [""] = "nao usar esta lista")
                variaveis.append(_v)
    if not variaveis:
        raise ValueError(
            'Nenhuma variavel definida para o s44. Preencha VARIAVEIS_GLOBO_3D e/ou '
            'GLOBO_3D_VARIAVEIS_S42 no settings.local.toml — pelo menos UMA variavel numa das listas '
            f'(disponiveis: {list(VARIAVEIS.keys())}).'
        )
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
    output_base = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_GLOBO_INCLINADO'
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
        # Voo herdado do s39, mas o s44 pode sobrepor via GLOBO_3D_INC_* (fallback ao compartilhado).
        'camera': [
            float(settings.get('GLOBO_3D_INC_LON_INICIAL', getattr(settings, 'GLOBO_3D_LON_INICIAL', -150.0))),
            float(settings.get('GLOBO_3D_INC_LAT_INICIAL', getattr(settings, 'GLOBO_3D_LAT_INICIAL', 0.0))),
            float(settings.get('GLOBO_3D_INC_LON_FINAL', getattr(settings, 'GLOBO_3D_LON_FINAL', -45.0))),
            float(settings.get('GLOBO_3D_INC_LAT_FINAL', getattr(settings, 'GLOBO_3D_LAT_FINAL', -15.0))),
            float(settings.get('GLOBO_3D_INC_VOLTAS_EXTRA', getattr(settings, 'GLOBO_3D_VOLTAS_EXTRA', 0.0))),
        ],
        'easing': str(settings.get('GLOBO_3D_INC_EASING', getattr(settings, 'GLOBO_3D_EASING', 'linear'))),
        'velocidade_var': float(getattr(settings, 'GLOBO_3D_VELOCIDADE_VAR', 1.0)),
        'frames_por_dia': int(getattr(settings, 'GLOBO_3D_FRAMES_POR_DIA', 4)),
        'fps': int(getattr(settings, 'GLOBO_3D_FPS', 20)),
        'grid_deg': float(getattr(settings, 'GLOBO_3D_GRID_DEG', 0.5)),
        'niveis': int(getattr(settings, 'GLOBO_3D_NIVEIS', 16)),
        'niveis_var': {v: settings.get(f'GLOBO_3D_NIVEIS_{v.upper()}', None) for v in variaveis},
        'coarsen': int(getattr(settings, 'GLOBO_3D_COARSEN', 1)),
        # Enquadramento INCLINADO (o que distingue o s44 do s39).
        'inc_altura': float(settings.get('GLOBO_3D_INC_ALTURA', 9_000_000.0)),
        'inc_deitar': float(settings.get('GLOBO_3D_INC_DEITAR', 0.555)),
        'inc_inclinacao': str(settings.get('GLOBO_3D_INC_INCLINACAO', '')),   # corte HS/HN (latitude fixa)
        'inc_janela_frac': float(settings.get('GLOBO_3D_INC_JANELA_FRAC', 0.858)),
        'inc_aspect': float(settings.get('GLOBO_3D_INC_ASPECT', 0.5625)),
        'atmosfera_estrelas': bool(settings.get('GLOBO_3D_ATMOSFERA_ESTRELAS', False)),
        # Tres saidas + corrente de jato (nas TRES: fluindo no MP4, parada no PNG, animada no GIF).
        'jato': bool(settings.get('GLOBO_3D_JATO', False)),
        'gif_frames': int(settings.get('GLOBO_3D_GE_GIF_FRAMES', 48)),
        'gif_fps': int(settings.get('GLOBO_3D_GE_GIF_FPS', 12)),
        'credito': str(getattr(settings, 'GLOBO_3D_CREDITO', 'Bruno Capucin')),
        'paletas': {v: list(settings.get(f'GLOBO_3D_PALETA_{v.upper()}', []) or []) for v in variaveis},
        'tamanho_px': int(getattr(settings, 'GLOBO_3D_TAMANHO_PX', 1080)),
        'script_version': '1.0-inclinado',
    }

    # Por padrao SEMPRE regenera o MP4 (GLOBO_3D_SEMPRE_REGERAR=true): saida de midia iterada
    # visualmente, e features de aparencia (enquadramento, box, camera, cores...) NAO entram na chave
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
