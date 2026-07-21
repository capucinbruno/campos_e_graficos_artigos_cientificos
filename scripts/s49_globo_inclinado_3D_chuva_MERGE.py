# ── s49 - Globo 3D INCLINADO de CHUVA OBSERVADA (MERGE/CPTEC + IMERG/NASA) ──────────────────────
# Copia do s46 no formato/enquadramento (mesmo motor globo_3d_anim, mesma familia de presets
# ENQUADRAMENTOS, mesmo estilo TWC), mas a fonte de dados e OBSERVACAO, nao previsao de modelo:
#   - America do Sul: MERGE/CPTEC (satelite IMERG corrigido por estacoes pluviometricas)
#   - Resto do globo: IMERG-GPM Late Run/NASA puro (satelite, sem correcao — CPTEC nao mantem
#     produto global; ver app/src/uteis/clim_diaria_precip_merge.py pro detalhe da investigacao)
# Por ser observacao pura, DATA_FINAL pode ir no maximo ate HOJE (sem "previsao" possivel aqui) --
#   HOJE entra como dia PARCIAL, somando o que ja foi publicado (MERGE HOURLY_NOW + IMERG Early
#   meia-hora, ver _dia_parcial_hoje em clim_diaria_precip_merge.py); o log do terminal avisa ate
#   que horario cada fonte contabilizou (a caixa de data do video NAO mostra isso -- ficava grande
#   demais). DATA_FINAL alem de hoje ainda levanta erro claro (sem previsao de verdade).
#   - Modo:  MERGE_MODO na config do script escolhe "absoluto" (mm/dia) ou "anomalia" (mm/dia
#     vs. climatologia diaria do CPTEC 1998-2024, valida so na caixa lon -85/-30, lat -56/13 —
#     fora dai a anomalia sai transparente por falta de climatologia).
#   - Acumulado: ACUMULAR_NO_TEMPO (mesma flag do s46) soma a chuva ao longo do periodo (running
#     total, arquivo _total.png); false (default) = cada frame e o acumulado DAQUELE dia (_media.png).
# Saida (dois arquivos, em REANALISE/): s49_<variavel>.mp4 + s49_<variavel>_media.png (ou _total.png
# com ACUMULAR_NO_TEMPO=true).
# Criado em: 2026-07-20 (copia do s46, finalidade = chuva OBSERVADA em vez de prevista).

# Bibliotecas padrao
import os
import time
from pathlib import Path

# Modulos locais
from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.common.config_header import aplicar_config_script
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.src.uteis.globo_3d_anim import _output_plan, VARIAVEIS, gerar_animacao

SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's49'
SCRIPT_DESC = 's49 - Globo 3D INCLINADO de CHUVA OBSERVADA (MERGE/CPTEC + IMERG/NASA)'

_MERGE_MODO_VARIAVEL = {
    'absoluto': 'merge_precip_abs',
    'anomalia': 'merge_precip_anom',
}

# Mesmo mapa de enquadramento do s46 (generico do motor, nao especifico de chuva-modelo).
_ENQUADRAMENTO_MAPA = {
    'altura': ('GLOBO_3D_INC_ALTURA',),
    'inclinacao': ('GLOBO_3D_INC_INCLINACAO', 'GLOBO_3D_INC_LAT_INICIAL', 'GLOBO_3D_INC_LAT_FINAL'),
    'lon': ('GLOBO_3D_INC_LON_INICIAL', 'GLOBO_3D_INC_LON_FINAL'),
    'deitar': ('GLOBO_3D_INC_DEITAR',),
    'janela_frac': ('GLOBO_3D_INC_JANELA_FRAC',),
    'aspect': ('GLOBO_3D_INC_ASPECT',),
}


def _aplicar_enquadramento() -> None:
    """Expande o preset escolhido em ENQUADRAMENTO (config local do script) nas chaves GLOBO_3D_INC_*.
    Copia exata da mesma funcao do s46 — ver o comentario la para o racional completo."""
    nome = str(settings.get('ENQUADRAMENTO', '') or '').strip()
    if not nome:
        return
    presets = {str(e.get('nome', '')).strip(): e for e in (settings.get('ENQUADRAMENTOS', None) or [])}
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
            if os.environ.get(f'AMPERE_{chave}') is not None:  # override externo vence
                puladas.append(chave)
                continue
            settings.set(chave, presets[nome][campo])
            aplicadas += 1
    logger.info(
        "Enquadramento '{}': {} chave(s) GLOBO_3D_INC_* aplicada(s){}",
        nome,
        aplicadas,
        f'; {len(puladas)} mantida(s) por override externo: {puladas}' if puladas else '',
    )


def _get_variaveis() -> list[str]:
    """Traduz a flag amigavel MERGE_MODO ("absoluto"|"anomalia") da config local do script pra
    a variavel registrada em VARIAVEIS (merge_precip_abs / merge_precip_anom)."""
    modo = str(settings.get('MERGE_MODO', 'absoluto') or 'absoluto').strip().lower()
    if modo not in _MERGE_MODO_VARIAVEL:
        raise ValueError(
            f"MERGE_MODO invalido: '{modo}'. Use 'absoluto' ou 'anomalia' na config "
            f'scripts/config_local/s49_globo_inclinado_3D_chuva_MERGE.toml.'
        )
    variavel = _MERGE_MODO_VARIAVEL[modo]
    if variavel not in VARIAVEIS:
        raise ValueError(f'Variavel {variavel} nao registrada em VARIAVEIS (globo_3d_anim.py).')
    return [variavel]


def main():
    aplicar_config_script(
        Path(__file__)
    )  # injeta a config dedicada do script no settings (env/CLI ainda vencem)
    logger = get_logger(SCRIPT_ID)
    logger.info('=' * 80)
    logger.info('SCRIPT {}: {}', SCRIPT_ID.upper(), SCRIPT_DESC)
    logger.info('=' * 80)
    _aplicar_enquadramento()  # preset de camera (ENQUADRAMENTO) -> GLOBO_3D_INC_*

    variaveis = _get_variaveis()
    output_base = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_GLOBO_INCLINADO_CHUVA_OBSERVADA'
    # Plano (so REANALISE: chuva observada nao tem "previsao" — DATA_FINAL tem que ficar no
    # passado, senao _output_plan ja levanta erro claro pedindo modelo, que aqui nao existe).
    plano, _, _ = _output_plan(variaveis, output_base)
    _gif_on = bool(settings.get('GLOBO_3D_GIF_MEDIA', False))
    # Mesma regra do s46: com ACUMULAR_NO_TEMPO o PNG do resumo e o TOTAL do periodo (running total,
    # sufixo _total); sem isso, e a media diaria de sempre (sufixo _media). So faz sentido pro modo
    # absoluto (merge_precip_anom nao acumula -- ver `_merge_precip_abs_reanalise_series`).
    _sfx = 'total' if bool(settings.get('ACUMULAR_NO_TEMPO', False)) else 'media'
    output_files = []
    for item in plano:
        _base = item['dir'] / f'{SCRIPT_ID}_{item["var"]}'
        output_files += [f'{_base}.mp4', f'{_base}_{_sfx}.png']
        if _gif_on:
            output_files.append(f'{_base}_{_sfx}.gif')

    cache_params = {
        'variaveis': variaveis,
        'merge_modo': str(settings.get('MERGE_MODO', 'absoluto')),
        'acumular_no_tempo': bool(settings.get('ACUMULAR_NO_TEMPO', False)),
        'enquadramento': str(settings.get('ENQUADRAMENTO', '')),
        'DATA_INICIAL': str(settings.DATA_INICIAL),
        'DATA_FINAL': str(settings.DATA_FINAL),
        'merge_coarsen': int(settings.get('GLOBO_3D_MERGE_COARSEN', 1)),
        'camera': [
            float(
                settings.get(
                    'GLOBO_3D_INC_LON_INICIAL', getattr(settings, 'GLOBO_3D_LON_INICIAL', -150.0)
                )
            ),
            float(
                settings.get('GLOBO_3D_INC_LAT_INICIAL', getattr(settings, 'GLOBO_3D_LAT_INICIAL', 0.0))
            ),
            float(
                settings.get('GLOBO_3D_INC_LON_FINAL', getattr(settings, 'GLOBO_3D_LON_FINAL', -45.0))
            ),
            float(
                settings.get('GLOBO_3D_INC_LAT_FINAL', getattr(settings, 'GLOBO_3D_LAT_FINAL', -15.0))
            ),
            float(
                settings.get(
                    'GLOBO_3D_INC_VOLTAS_EXTRA', getattr(settings, 'GLOBO_3D_VOLTAS_EXTRA', 0.0)
                )
            ),
        ],
        'easing': str(
            settings.get('GLOBO_3D_INC_EASING', getattr(settings, 'GLOBO_3D_EASING', 'linear'))
        ),
        'velocidade_var': float(
            settings.get(
                'GLOBO_3D_INC_VELOCIDADE_VAR', getattr(settings, 'GLOBO_3D_VELOCIDADE_VAR', 1.0)
            )
        ),
        'frames_por_dia': int(
            settings.get('GLOBO_3D_INC_FRAMES_POR_DIA', getattr(settings, 'GLOBO_3D_FRAMES_POR_DIA', 4))
        ),
        'fps': int(settings.get('GLOBO_3D_INC_FPS', getattr(settings, 'GLOBO_3D_FPS', 20))),
        'grid_deg': float(getattr(settings, 'GLOBO_3D_GRID_DEG', 0.5)),
        'niveis': int(getattr(settings, 'GLOBO_3D_NIVEIS', 16)),
        'niveis_var': {v: settings.get(f'GLOBO_3D_NIVEIS_{v.upper()}', None) for v in variaveis},
        'coarsen': int(getattr(settings, 'GLOBO_3D_COARSEN', 1)),
        'inc_altura': float(settings.get('GLOBO_3D_INC_ALTURA', 9_000_000.0)),
        'inc_deitar': float(settings.get('GLOBO_3D_INC_DEITAR', 0.555)),
        'inc_inclinacao': str(settings.get('GLOBO_3D_INC_INCLINACAO', '')),
        'inc_janela_frac': float(settings.get('GLOBO_3D_INC_JANELA_FRAC', 0.858)),
        'inc_aspect': float(settings.get('GLOBO_3D_INC_ASPECT', 0.5625)),
        'atmosfera_estrelas': bool(settings.get('GLOBO_3D_ATMOSFERA_ESTRELAS', False)),
        'jato': bool(settings.get('GLOBO_3D_JATO', False)),
        'credito': str(getattr(settings, 'GLOBO_3D_CREDITO', 'Bruno Capucin')),
        'paletas': {v: list(settings.get(f'GLOBO_3D_PALETA_{v.upper()}', []) or []) for v in variaveis},
        'tamanho_px': int(getattr(settings, 'GLOBO_3D_TAMANHO_PX', 1080)),
        'script_version': '1.0-merge-precip-observada',
    }

    if not bool(settings.get('GLOBO_3D_SEMPRE_REGERAR', True)) and check_cache_valid(
        SCRIPT_ID, cache_params, output_files
    ):
        logger.info('CACHE VALIDO — pulando execucao ({} arquivo(s))', len(output_files))
        return

    start_time = time.time()
    gerados = gerar_animacao(variaveis, output_base, SCRIPT_ID)

    execution_time = time.time() - start_time
    save_cache_metadata(SCRIPT_ID, cache_params, [str(p) for p in gerados], execution_time)
    logger.info('=' * 80)
    logger.info(
        'Script {} concluido! {} arquivo(s) [{} por variavel] em {:.1f}s',
        SCRIPT_ID.upper(),
        len(gerados),
        'mp4 + png + gif' if _gif_on else 'mp4 + png',
        execution_time,
    )
    for p in gerados:
        logger.info('  {}', p)
    logger.info('=' * 80)


if __name__ == '__main__':
    main()
