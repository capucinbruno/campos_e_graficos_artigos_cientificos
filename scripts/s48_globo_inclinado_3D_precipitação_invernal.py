
# ── s48 - Globo 3D INCLINADO de PRECIPITAÇÃO INVERNAL: cópia do s46 ─────────────────────────────────
# Idêntico ao s46 (globo "deitado"/inclinado estilo Google Earth com pitch, chuva+neve por ptype):
# MESMO motor, MESMO enquadramento inclinado e MESMO pipeline de dados (reanálise/previsão/emenda).
# A finalidade é separar os vídeos de PRECIPITAÇÃO INVERNAL (chuva/neve em eventos de inverno) do
# s46, cada um com sua própria config (scripts/config_local/), cache e pasta de saída.
#   - Variável: VARIAVEIS_GLOBO_3D em scripts/config_local/s48_globo_inclinado_3D_precipitação_invernal.toml
#   - Modo:     AUTOMÁTICO pelas datas (passado=reanálise, futuro=previsão, cruza hoje=emenda)
#   - Voo/enquadramento: namespace INC_ (o motor trata s48 igual s44/s46: `_inclinado`, regrid leve,
#     PLOTAR_SOMENTE, estilo Guillaume/TWC). Aparência por VARIÁVEL/SCRIPT -> o s48 pode ter seu
#     próprio GLOBO_3D_PCOLORMESH_S48 sem tocar no s44/s46.
# Saída (dois arquivos por variável, em REANALISE/ ou FORECAST/<MODELO>/):
#   - s48_<variavel>.mp4        : vídeo do período (voo inclinado + evolução dia a dia) + jato FLUINDO
#   - s48_<variavel>_total.png  : com ACUMULAR_NO_TEMPO, a chuva TOTAL do período (= último frame do
#                                 vídeo) + jato PARADO. Sem ele, vira _media.png (média do período)
#   (o 3º arquivo do s44, s48_<variavel>_total.gif — campo fixo + 'JET STREAM'/setas animando W->E —
#    NÃO sai aqui; religue com GLOBO_3D_GIF_MEDIA = true na config do script se precisar.)
# Criado em: 2026-07-19 (cópia do s46, finalidade = precipitação invernal).

# Bibliotecas padrao
import os
import time
from pathlib import Path

# Modulos locais
from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.common.config_header import aplicar_config_script
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.src.uteis.globo_3d_anim import (
    _enabled_forecast_models,
    _output_plan,
    expandir_variaveis,
    VARIAVEIS,
    gerar_animacao,
)

SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's48'
SCRIPT_DESC = 's48 - Globo 3D INCLINADO de PRECIPITACAO INVERNAL (copia do s46)'


# Campos do preset de ENQUADRAMENTOS (settings.toml) -> chaves GLOBO_3D_INC_* que o motor le.
# A `inclinacao` tambem preenche LAT_INICIAL/FINAL: o motor as ignora enquanto INCLINACAO != '',
# mas deixa-las com outro valor faria a config/log mentirem sobre a latitude da camera.
_ENQUADRAMENTO_MAPA = {
    'altura':      ('GLOBO_3D_INC_ALTURA',),
    'inclinacao':  ('GLOBO_3D_INC_INCLINACAO', 'GLOBO_3D_INC_LAT_INICIAL', 'GLOBO_3D_INC_LAT_FINAL'),
    'lon':         ('GLOBO_3D_INC_LON_INICIAL', 'GLOBO_3D_INC_LON_FINAL'),
    'deitar':      ('GLOBO_3D_INC_DEITAR',),
    'janela_frac': ('GLOBO_3D_INC_JANELA_FRAC',),
    'aspect':      ('GLOBO_3D_INC_ASPECT',),
}


def _aplicar_enquadramento() -> None:
    """Expande o preset escolhido em ENQUADRAMENTO (config local do script) nas chaves GLOBO_3D_INC_*.

    A lista ENQUADRAMENTOS vive no settings.toml (mae) por ser identidade fixa, nao config de rodada.
    ENQUADRAMENTO = '' -> no-op (valem os GLOBO_3D_INC_* soltos). Mesma precedencia da config local: quem
    veio de FORA (env AMPERE_<KEY> / CLI) continua vencendo."""
    nome = str(settings.get('ENQUADRAMENTO', '') or '').strip()
    if not nome:
        return
    presets = {str(e.get('nome', '')).strip(): e
               for e in (settings.get('ENQUADRAMENTOS', None) or [])}
    if nome not in presets:
        raise ValueError(
            f"ENQUADRAMENTO '{nome}' nao existe na lista ENQUADRAMENTOS do settings.toml. "
            f'Disponiveis: {sorted(presets)}'
        )
    logger = get_logger(SCRIPT_ID)
    aplicadas, puladas = 0, []
    for campo, chaves in _ENQUADRAMENTO_MAPA.items():
        if campo not in presets[nome]:
            continue
        for chave in chaves:
            if os.environ.get(f'AMPERE_{chave}') is not None:   # override externo vence
                puladas.append(chave)
                continue
            settings.set(chave, presets[nome][campo])
            aplicadas += 1
    logger.info("Enquadramento '{}': {} chave(s) GLOBO_3D_INC_* aplicada(s){}", nome, aplicadas,
                f'; {len(puladas)} mantida(s) por override externo: {puladas}' if puladas else '')


def _get_variaveis() -> list[str]:
    """Lista de variaveis do s48, vinda do VARIAVEIS_GLOBO_3D da config local do script (que sobrepoe
    o settings.local). Strings vazias sao ignoradas; globo so com features = 'sem_variavel' explicito."""
    variaveis: list[str] = []
    for v in (settings.get('VARIAVEIS_GLOBO_3D', None) or []):
        _v = str(v).strip()
        if _v and _v not in variaveis:
            variaveis.append(_v)
    if not variaveis:
        raise ValueError(
            'Nenhuma variavel definida para o s48. Preencha VARIAVEIS_GLOBO_3D no arquivo '
            'scripts/config_local/s48_globo_inclinado_3D_precipitação_invernal.toml — pelo menos UMA variavel '
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
    aplicar_config_script(Path(__file__))   # injeta a config dedicada do script no settings (env/CLI ainda vencem)
    logger = get_logger(SCRIPT_ID)
    logger.info('=' * 80)
    logger.info('SCRIPT {}: {}', SCRIPT_ID.upper(), SCRIPT_DESC)
    logger.info('=' * 80)
    _aplicar_enquadramento()                # preset de camera (ENQUADRAMENTO) -> GLOBO_3D_INC_*

    variaveis = _get_variaveis()
    output_base = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_GLOBO_INCLINADO_PRECIPITACAO_INVERNAL'
    # Plano (modo decidido pelas datas): caminhos esperados p/ validar o cache.
    plano, _, _ = _output_plan(variaveis, output_base)
    # Saidas por variavel: MP4 do periodo + PNG do resumo (+ GIF do resumo so se GLOBO_3D_GIF_MEDIA,
    # que a config deste script desliga). O sufixo do resumo acompanha o que a figura mostra: '_total'
    # com ACUMULAR_NO_TEMPO (chuva somada do periodo inteiro), '_media' no resto (media do periodo) --
    # mesma regra do motor em `gerar_animacao`. Manter em sincronia com o gate do motor: um arquivo
    # listado aqui e nao gerado la faria o cache pedir regeracao eterna.
    _sfx = 'total' if bool(settings.get('ACUMULAR_NO_TEMPO', False)) else 'media'
    _gif_on = bool(settings.get('GLOBO_3D_GIF_MEDIA', True))
    output_files = []
    for item in plano:
        _base = item['dir'] / f"{SCRIPT_ID}_{item['var']}"
        output_files += [f'{_base}.mp4', f'{_base}_{_sfx}.png']
        if _gif_on:
            output_files.append(f'{_base}_{_sfx}.gif')

    cache_params = {
        'variaveis': variaveis,
        'enquadramento': str(settings.get('ENQUADRAMENTO', '')),   # preset de camera (valores logo abaixo)
        'DATA_INICIAL': str(settings.DATA_INICIAL),
        'DATA_FINAL': str(settings.DATA_FINAL),
        'forecast_init': str(settings.get('FORECAST_INIT', 'latest')),
        'rodada': int(settings.get('RODADA', 0)),
        'modelos': _enabled_forecast_models(),
        # Voo: namespace INC_ (o motor le INC_ com precedencia no s48), fallback ao compartilhado.
        'camera': [
            float(settings.get('GLOBO_3D_INC_LON_INICIAL', getattr(settings, 'GLOBO_3D_LON_INICIAL', -150.0))),
            float(settings.get('GLOBO_3D_INC_LAT_INICIAL', getattr(settings, 'GLOBO_3D_LAT_INICIAL', 0.0))),
            float(settings.get('GLOBO_3D_INC_LON_FINAL', getattr(settings, 'GLOBO_3D_LON_FINAL', -45.0))),
            float(settings.get('GLOBO_3D_INC_LAT_FINAL', getattr(settings, 'GLOBO_3D_LAT_FINAL', -15.0))),
            float(settings.get('GLOBO_3D_INC_VOLTAS_EXTRA', getattr(settings, 'GLOBO_3D_VOLTAS_EXTRA', 0.0))),
        ],
        'easing': str(settings.get('GLOBO_3D_INC_EASING', getattr(settings, 'GLOBO_3D_EASING', 'linear'))),
        'velocidade_var': float(settings.get('GLOBO_3D_INC_VELOCIDADE_VAR',
                                             getattr(settings, 'GLOBO_3D_VELOCIDADE_VAR', 1.0))),
        'frames_por_dia': int(settings.get('GLOBO_3D_INC_FRAMES_POR_DIA',
                                           getattr(settings, 'GLOBO_3D_FRAMES_POR_DIA', 4))),
        'fps': int(settings.get('GLOBO_3D_INC_FPS', getattr(settings, 'GLOBO_3D_FPS', 20))),
        'grid_deg': float(getattr(settings, 'GLOBO_3D_GRID_DEG', 0.5)),
        'niveis': int(getattr(settings, 'GLOBO_3D_NIVEIS', 16)),
        'niveis_var': {v: settings.get(f'GLOBO_3D_NIVEIS_{v.upper()}', None) for v in variaveis},
        'coarsen': int(getattr(settings, 'GLOBO_3D_COARSEN', 1)),
        # Enquadramento INCLINADO (o mesmo do s44/s46).
        'inc_altura': float(settings.get('GLOBO_3D_INC_ALTURA', 9_000_000.0)),
        'inc_deitar': float(settings.get('GLOBO_3D_INC_DEITAR', 0.555)),
        'inc_inclinacao': str(settings.get('GLOBO_3D_INC_INCLINACAO', '')),   # corte HS/HN (latitude fixa)
        'inc_janela_frac': float(settings.get('GLOBO_3D_INC_JANELA_FRAC', 0.858)),
        'inc_aspect': float(settings.get('GLOBO_3D_INC_ASPECT', 0.5625)),
        'atmosfera_estrelas': bool(settings.get('GLOBO_3D_ATMOSFERA_ESTRELAS', False)),
        # Corrente de jato nas duas saidas: fluindo no MP4, parada no PNG (e animada no GIF, se religado).
        'jato': bool(settings.get('GLOBO_3D_JATO', False)),
        'gif_frames': int(settings.get('GLOBO_3D_INC_GIF_FRAMES',
                                       getattr(settings, 'GLOBO_3D_GIF_FRAMES', 48))),
        'gif_fps': int(settings.get('GLOBO_3D_INC_GIF_FPS', getattr(settings, 'GLOBO_3D_GIF_FPS', 12))),
        'credito': str(getattr(settings, 'GLOBO_3D_CREDITO', 'Bruno Capucin')),
        'paletas': {v: list(settings.get(f'GLOBO_3D_PALETA_{v.upper()}', []) or []) for v in variaveis},
        'tamanho_px': int(getattr(settings, 'GLOBO_3D_TAMANHO_PX', 1080)),
        'script_version': '1.0-inclinado-precipitacao-invernal',  # copia do s46; identidade propria
    }

    # Por padrao SEMPRE regenera o MP4 (GLOBO_3D_SEMPRE_REGERAR=true): features de aparencia nao
    # entram na chave de cache. Os DOWNLOADS seguem em cache (so o render e refeito).
    if not bool(settings.get('GLOBO_3D_SEMPRE_REGERAR', True)) \
            and check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('CACHE VALIDO — pulando execucao ({} arquivo(s))', len(output_files))
        return

    start_time = time.time()
    gerados = gerar_animacao(variaveis, output_base, SCRIPT_ID)

    execution_time = time.time() - start_time
    save_cache_metadata(SCRIPT_ID, cache_params, [str(p) for p in gerados], execution_time)
    logger.info('=' * 80)
    logger.info('Script {} concluido! {} arquivo(s) [{} por variavel] em {:.1f}s',
                SCRIPT_ID.upper(), len(gerados),
                'mp4 + png + gif' if _gif_on else 'mp4 + png', execution_time)
    for p in gerados:
        logger.info('  {}', p)
    logger.info('=' * 80)


if __name__ == '__main__':
    main()
