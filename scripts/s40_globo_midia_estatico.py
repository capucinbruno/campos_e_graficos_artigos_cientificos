# -*- coding: utf-8 -*-
# ══════════════════════════════════════════════════════════════════════════════════════════════════
# CONFIG DO SCRIPT s40 — EDITE AQUI (não no settings.local). Aplicada no início do main() via
# aplicar_config_header(). Precedência: env AMPERE_<KEY> / CLI --data-inicial > este header > settings.toml.
# Datas PASSADAS => reanálise (ERA5); FUTURAS => previsão; cruzando hoje => emenda observado+previsão.
# s40 = ESTÁTICO: figuras PNG (câmera FIXA, sem voo/vídeo). Coleções: diario, media_movel, pentadas_fixas, media_total.
# ══════════════════════════════════════════════════════════════════════════════════════════════════
_CONFIG_HEADER = """#toml
DATA_INICIAL = "2026-07-14"
DATA_FINAL   = "2026-07-14"

RUN_GFS       = false
RUN_GEFS      = false
RUN_ECMWF     = false
RUN_ECMWF_ENS = false
RUN_AIFS      = false
RUN_AIFS_ENS  = false
RUN_AIGFS     = false
RUN_AIGEFS    = false
RUN_CFS       = false

FORECAST_INIT = ""             # "" = rodada mais recente
RODADA        = "00"           # "00" | "06" | "12" | "18" (UTC)
NUM_RODADA    = 1              # últimas N rodadas (lagged ensemble)
GEFS_FORECAST_LEAD_DAYS = 35
ECMWF_ENS_MEMBERS = 30
ECMWF_ENS_WORKERS = 64

# Variáveis a plotar (1 coleção de PNGs por variável) — copie o nome (com aspas e vírgula) e cole em VARIAVEIS_GLOBO_3D:

# ── ABSOLUTAS ──
#   "z250_abs", "z500_abs", "z500_contour", "pwat_abs", "pwat_abs_alicia", "pwat_cores_z500_contornos",
#   "precip_abs", "tsm_abs", "jet_stream", "jet_stream_psi200_contour",

# ── ANOMALIA (inclui os combinados *_cores_*: shaded em cores + 2ª variável em isolinhas) ──
#   "z250_anom", "z500_anom", "psi200_anom", "chi200_anom", "tmp850_anom", "tmp850_mslp", "olr_anom",
#   "wnd250_zonal_anom", "wnd850_zonal_anom", "wnd850_meridional_anom", "tsm_anom",
#   "chi200_cores_psi200_contornos", "chi200_cores_z250_contornos",
#   "olr_cores_psi200_contornos", "olr_cores_z250_contornos", "olr_cores_z500_contornos",
#   "tmp850_cores_psi200_contornos", "tmp850_cores_z500_contornos",
#   "wnd200_zonal_cores_psi200_contornos", "wnd200_meridional_cores_psi200_contornos",

# NOTA: precip_abs = só FORECAST; tsm_abs e tsm_anom = só REANÁLISE. O resto roda nos dois.

VARIAVEIS_GLOBO_3D = ["tsm_anom"]

GLOBO_3D_VARIANTES_AUTO = false   # true = ao plotar z250_anom, gera TB a z250_anom_5d (média móvel 5d); false = só a diária. Só afeta z250_anom

# Câmera FIXA (s40 não voa). Sem valor => cai em ORTHO_CENTRAL_LON/LAT.
GLOBO_3D_ESTATICO_LON = -139.0
GLOBO_3D_ESTATICO_LAT =  0.0

# Agregação das coleções
MOV_AVG_DAYS = 5                    # dias da média móvel (coleção media_movel)
GLOBO_3D_ESTATICO_MIN_DIAS = 5      # mínimo de dias p/ gerar media_movel e pentadas_fixas (senão pula)

# Figura PNG (tamanho, resolução da grade, projeção, paralelismo) — NÃO é vídeo, é como o PNG é gerado
GLOBO_3D_TAMANHO_PX = 1080          # tamanho do PNG (px)
GLOBO_3D_WORKERS = 3               # figuras renderizadas em paralelo (0 = todas as CPUs)
GLOBO_3D_GRID_DEG = 0.5            # resolução da grade do dado (0.25 detalhe/lento; 0.5 rápido; 1.0 suave)
GLOBO_3D_COARSEN = 1               # subamostragem extra da grade (1 = nenhuma)
GLOBO_3D_PROJECTION = "nearside"    # projeção do globo: "nearside" | "orthographic"
GLOBO_3D_OLR_OVERLAY = false        # camada extra de OLR equatorial sobre o campo (pesado)

# Aparência
GLOBO_3D_CREDITO = ""
GLOBO_3D_VINHETA = false
GLOBO_3D_ATMOSFERA_ESTRELAS = true
GLOBO_3D_SOMENTE_ESTRELAS = false
GLOBO_3D_FONTE_TITULO  = "Ubuntu Sans"
GLOBO_3D_FONTE_LEGENDA = "Ubuntu Sans"
GLOBO_3D_SEMPRE_REGERAR = true
GLOBO_3D_CONTORNO_Z250_ANOM = false
GLOBO_3D_ISOTERMA_0C        = false
GLOBO_3D_BOX_NINO34         = false
GLOBO_3D_BOX_NINO34_TSM_ABS = true
GLOBO_3D_BOX_NINO34_TSM_ANOM = true

# Isolinhas de PNMM (só a variável tmp850_mslp desenha)
GLOBO_3D_MSLP_INTERVALO = 3.0
GLOBO_3D_MSLP_COR       = "black"
GLOBO_3D_MSLP_LW        = 0.5
GLOBO_3D_MSLP_SIGMA     = 2.0

"""

# ── s40 - Globo 3D ESTÁTICO (mídia): figuras PNG por agregação (padrão do s34) ─────────────────────
# Cópia do s39 (mesmo motor/variáveis/estilo Guillaume), mas a saída são FIGURAS estáticas do globo
# (câmera FIXA), não um MP4. Por variável e modo: diario/, media_movel/, pentadas_fixas/, media_total/.
# media_movel e pentadas_fixas só saem com >= GLOBO_3D_ESTATICO_MIN_DIAS dias; diario e media_total sempre.
# Saída: Saida/s40_GLOBO_ESTATICO/{REANALISE|FORECAST/<MODELO>}/<var>/<coleção>/*.png. Criado em: 2026-07-01.

# Bibliotecas padrao
import time
from pathlib import Path

# Modulos locais
from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.common.config_header import aplicar_config_header
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
    aplicar_config_header(_CONFIG_HEADER)   # injeta a config do topo no settings (env/CLI ainda vencem)
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
