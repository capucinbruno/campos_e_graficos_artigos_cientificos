# -*- coding: utf-8 -*-
# ══════════════════════════════════════════════════════════════════════════════════════════════════
# CONFIG DO SCRIPT s40 — EDITE em scripts/config_local/s40_globo_midia_estatico.toml (gitignored,
# não no settings.local). Aplicada no início do main() via aplicar_config_script(). Precedência:
# env AMPERE_<KEY> / CLI --data-inicial > config local do script > settings.toml.
# Datas PASSADAS => reanálise (ERA5); FUTURAS => previsão; cruzando hoje => emenda observado+previsão.
# s40 = ESTÁTICO: figuras PNG (câmera FIXA, sem voo/vídeo). Coleções: diario, media_movel, pentadas_fixas, media_total.
# ══════════════════════════════════════════════════════════════════════════════════════════════════

# ── s40 - Globo 3D ESTÁTICO (mídia): figuras PNG por agregação (padrão do s34) ─────────────────────
# Cópia do s39 (mesmo motor/variáveis/estilo Guillaume), mas a saída são FIGURAS estáticas do globo
# (câmera FIXA), não um MP4 — por isso a config NÃO tem voo de câmera nem frames/fps (isso é do s39).
# Por variável e modo: diario/, media_movel/, pentadas_fixas/, media_total/.
# media_movel e pentadas_fixas só saem com >= GLOBO_3D_ESTATICO_MIN_DIAS dias; diario e media_total sempre.
# Saída: Saida/s40_GLOBO_ESTATICO/{REANALISE|FORECAST/<MODELO>}/<var>/<coleção>/*.png. Criado em: 2026-07-01.

# Bibliotecas padrao
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
    gerar_figuras_estaticas,
)

SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's40'
SCRIPT_DESC = 's40 - Globo 3D ESTATICO (midia): figuras PNG por agregacao'


def _get_variaveis() -> list[str]:
    """Lista de variaveis a plotar. VARIAVEIS_GLOBO_3D (lista) > VARIAVEL_GLOBO_3D
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
    # Variantes automaticas: z250_anom gera tambem a media movel de 5 dias.
    return expandir_variaveis(variaveis)


def main():
    aplicar_config_script(Path(__file__))   # injeta a config dedicada do script no settings (env/CLI ainda vencem)
    logger = get_logger(SCRIPT_ID)
    logger.info('=' * 80)
    logger.info('SCRIPT {}: {}', SCRIPT_ID.upper(), SCRIPT_DESC)
    logger.info('=' * 80)

    variaveis = _get_variaveis()
    output_base = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_GLOBO_ESTATICO'
    # Plano (modo decidido pelas datas): sentinela da media_total (sempre gerada) p/ o cache.
    plano, dt_ini, dt_fim = _output_plan(variaveis, output_base)
    output_files = [
        str(item['dir'] / item['var'] / 'media_total' /
            f"{SCRIPT_ID}_{item['var']}_{dt_ini:%Y%m%d}_{dt_fim:%Y%m%d}_media_total.png")
        for item in plano
    ]

    cache_params = {
        'variaveis': variaveis,
        'DATA_INICIAL': str(settings.DATA_INICIAL),
        'DATA_FINAL': str(settings.DATA_FINAL),
        'forecast_init': str(settings.get('FORECAST_INIT', 'latest')),
        'rodada': int(settings.get('RODADA', 0)),
        'modelos': _enabled_forecast_models(),
        'camera': [
            float(settings.get('GLOBO_3D_ESTATICO_LON', getattr(settings, 'ORTHO_CENTRAL_LONGITUDE', -45.0))),
            float(settings.get('GLOBO_3D_ESTATICO_LAT', getattr(settings, 'ORTHO_CENTRAL_LATITUDE', -15.0))),
        ],
        'mov_avg_days': int(settings.get('MOV_AVG_DAYS', 5)),
        'min_dias': int(settings.get('GLOBO_3D_ESTATICO_MIN_DIAS', 5)),
        'grid_deg': float(getattr(settings, 'GLOBO_3D_GRID_DEG', 0.5)),
        'coarsen': int(getattr(settings, 'GLOBO_3D_COARSEN', 1)),
        'projection': str(getattr(settings, 'GLOBO_3D_PROJECTION', 'nearside')),
        'niveis_var': {v: settings.get(f'GLOBO_3D_NIVEIS_{v.upper()}', None) for v in variaveis},
        'jato': bool(settings.get('GLOBO_3D_JATO', False)),  # master unico das correntes de jato (s38/s39/s40/s41)
        'paletas': {v: list(settings.get(f'GLOBO_3D_PALETA_{v.upper()}', []) or []) for v in variaveis},
        'tamanho_px': int(getattr(settings, 'GLOBO_3D_TAMANHO_PX', 1080)),
        'script_version': '1.1',  # corrente de jato disponivel em qualquer campo (master unico GLOBO_3D_JATO)
    }

    # Por padrao SEMPRE regenera (GLOBO_3D_SEMPRE_REGERAR=true): features de aparencia (camera,
    # cores, box...) NAO entram no cache. Os DOWNLOADS seguem em cache (so o render e refeito).
    if not bool(settings.get('GLOBO_3D_SEMPRE_REGERAR', True)) \
            and check_cache_valid(SCRIPT_ID, cache_params, output_files):
        logger.info('CACHE VALIDO — pulando execucao ({} colecao(oes))', len(plano))
        return

    start_time = time.time()
    gerados = gerar_figuras_estaticas(variaveis, output_base, SCRIPT_ID)

    execution_time = time.time() - start_time
    save_cache_metadata(SCRIPT_ID, cache_params, [str(p) for p in gerados], execution_time)
    logger.info('=' * 80)
    logger.info('Script {} concluido! {} PNG(s) em {:.1f}s', SCRIPT_ID.upper(), len(gerados),
                execution_time)
    logger.info('=' * 80)


if __name__ == '__main__':
    main()
