# -*- coding: utf-8 -*-
"""s45 - Mapa global 2D (midia) estilo s39, SEM globo 3D nem voo de camera.

Irmao 2D do s39: MESMAS variaveis, MESMO pipeline de dados (reanalise/previsao/emenda) e MESMAS
features de aparencia do s39 (overlay guillaume com caixa do nome da variavel + subtitulo + data +
legenda, corrente de jato, icones de pressao, escoamento de baixos niveis, caixa de texto livre,
credito, LOGO e fade-in/out do MP4). A UNICA diferenca e a PROJECAO: em vez do globo flutuante 3D
com camera voando, plota um MAPA GLOBAL 2D fixo (PlateCarree) respeitando a area de plotagem do
settings.json — por default a area "globo" (global, com o Pacifico no centro do quadro via
`central_longitude_mapa=180`). Sem controle de voo/camera: so a animacao da variavel no tempo.

Motor: `app/src/uteis/mapa_2d_anim.py` (o mesmo do s43), agora `script_id`-aware.

  - Variavel:  UNIAO de VARIAVEIS_GLOBO_3D (s38-s41) + GLOBO_3D_VARIAVEIS_S42 (s42), sem duplicar
  - Modo:      AUTOMATICO pelas datas (passado=reanalise, futuro=previsao, cruza hoje=emenda)
  - Regiao:    GLOBO_2D_S45_AREA (namespace proprio, default "globo") — FIXA, sem voo de camera

Config PROPRIA (namespace GLOBO_2D_S45_*, NAO afeta o s43): _AREA, _FPS, _FIGSIZE_W, _DPI,
_FRAMES_POR_PASSO, _VELOCIDADE_VAR, _GIF_FRAMES, _GIF_FPS, _JATO_MP4_CAUDA_SEG, _FADE_CAUDA_DUR_SEG
(cada GLOBO_2D_S45_<X> sobrepoe o GLOBO_2D_<X> compartilhado so no s45; ausente = herda o do s43).
Fade-in/out do MP4 usa o mesmo GLOBO_3D_FADE_IN_SEG/_OUT_SEG dos globos.

Saida (TRES arquivos por variavel, em REANALISE/ ou FORECAST/<MODELO>/):
  - s45_<variavel>.mp4        : video do periodo (evolucao da variavel) + jato FLUINDO + fade
  - s45_<variavel>_media.png  : figura da MEDIA do periodo + corrente de jato PARADA
  - s45_<variavel>_media.gif  : MEDIA do periodo (campo fixo) + 'JET STREAM'/setas animando W->E

Criado em: 2026-07-14 (mapa 2D irmao do s39, via motor do s43).
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

SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's45'
SCRIPT_DESC = __doc__.strip().split('\n')[0] if __doc__ else SCRIPT_ID


def _get_variaveis() -> list[str]:
    """UNIAO das DUAS listas: VARIAVEIS_GLOBO_3D (compartilhada com s38-s41) + GLOBO_3D_VARIAVEIS_S42
    (propria do s42) — sem duplicar e preservando a ordem (primeiro as do s39, depois as do s42).
    Assim o s45 gera tudo o que o s39 E o s42 gerariam num unico run (espelha o _get_variaveis do s44).

    UMA lista vazia (ausente, [], [""], so espacos) simplesmente NAO contribui — o s45 roda normal com
    o que houver na OUTRA lista. Se as DUAS ficarem vazias, levanta erro claro pedindo pra por ao menos
    uma variavel numa das listas."""
    variaveis: list[str] = []
    for _lst in (settings.get('VARIAVEIS_GLOBO_3D', None),
                 settings.get('GLOBO_3D_VARIAVEIS_S42', None)):
        for v in (_lst or []):
            _v = str(v).strip()
            if _v and _v not in variaveis:   # ignora strings vazias (ex.: lista [""] = "nao usar esta lista")
                variaveis.append(_v)
    if not variaveis:
        raise ValueError(
            'Nenhuma variavel definida para o s45. Preencha VARIAVEIS_GLOBO_3D e/ou '
            'GLOBO_3D_VARIAVEIS_S42 no settings.local.toml — pelo menos UMA variavel numa das listas '
            f'(disponiveis: {list(VARIAVEIS.keys())}).'
        )
    invalidas = [v for v in variaveis if v not in VARIAVEIS]
    if invalidas:
        raise ValueError(
            f'Variaveis nao registradas: {invalidas}. Disponiveis: {list(VARIAVEIS.keys())}'
        )
    # Variantes automaticas: z250_anom gera tambem a media movel de 5 dias (dois clipes).
    return expandir_variaveis(variaveis)


def main():
    logger = get_logger(SCRIPT_ID)
    logger.info('=' * 80)
    logger.info('SCRIPT {}: {}', SCRIPT_ID.upper(), SCRIPT_DESC)
    logger.info('=' * 80)

    variaveis = _get_variaveis()
    output_base = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_GLOBO_MIDIA_2D'
    # Plano (modo decidido pelas datas): caminhos esperados p/ validar o cache.
    plano, _, _ = _output_plan(variaveis, output_base)
    # Tres saidas por variavel: MP4 do periodo + PNG da media + GIF da media.
    output_files = []
    for item in plano:
        _base = item['dir'] / f"{SCRIPT_ID}_{item['var']}"
        output_files += [f'{_base}.mp4', f'{_base}_media.png', f'{_base}_media.gif']

    # Namespace proprio GLOBO_2D_S45_* com fallback ao GLOBO_2D_* do s43 (mesma logica de
    # _script_setting_2d no motor). Assim o cache reflete o que o s45 realmente usa.
    def _s45(suffix, default):
        return settings.get(f'GLOBO_2D_S45_{suffix}', settings.get(f'GLOBO_2D_{suffix}', default))

    cache_params = {
        'variaveis': variaveis,
        'DATA_INICIAL': str(settings.DATA_INICIAL),
        'DATA_FINAL': str(settings.DATA_FINAL),
        'forecast_init': str(settings.get('FORECAST_INIT', 'latest')),
        'rodada': int(settings.get('RODADA', 0)),
        'area': str(_s45('AREA', 'globo')),
        'frames_por_passo': int(_s45('FRAMES_POR_PASSO', 5)),
        'velocidade_var': float(_s45('VELOCIDADE_VAR', 1.0)),
        'fps': int(_s45('FPS', 8)),
        'figsize_w': float(_s45('FIGSIZE_W', 15.0)),
        'dpi': int(_s45('DPI', 150)),
        'cauda_seg': float(_s45('JATO_MP4_CAUDA_SEG', 0.0)),
        'fade_in': float(settings.get('GLOBO_3D_FADE_IN_SEG', 0.0)),
        'fade_out': float(settings.get('GLOBO_3D_FADE_OUT_SEG', 0.0)),
        'niveis_var': {v: settings.get(f'GLOBO_3D_NIVEIS_{v.upper()}', None) for v in variaveis},
        'vmin_var': {v: settings.get(f'GLOBO_3D_VMIN_{v.upper()}', None) for v in variaveis},
        'vmax_var': {v: settings.get(f'GLOBO_3D_VMAX_{v.upper()}', None) for v in variaveis},
        'jato': bool(settings.get('GLOBO_3D_JATO', False)),
        'icones_pressao': list(settings.get('GLOBO_3D_ICONES_PRESSAO', []) or []),
        'caixa_livre': bool(settings.get('GLOBO_3D_CAIXA_LIVRE', False)),
        'credito': str(settings.get('GLOBO_3D_CREDITO', 'Bruno Capucin')),
        'paletas': {v: list(settings.get(f'GLOBO_3D_PALETA_{v.upper()}', []) or []) for v in variaveis},
        'script_version': '1.0-2d-global-s39',
    }

    # Por padrao SEMPRE regenera (GLOBO_3D_SEMPRE_REGERAR=true): saida de midia iterada visualmente,
    # e features de aparencia NAO entram na chave de cache. Os DOWNLOADS seguem em cache.
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
