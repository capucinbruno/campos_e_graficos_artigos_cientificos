# -*- coding: utf-8 -*-
# ══════════════════════════════════════════════════════════════════════════════════════════════════
# CONFIG DO SCRIPT s39 — EDITE AQUI (não no settings.local). Aplicada no início do main() via
# aplicar_config_header(). Precedência: env AMPERE_<KEY> / CLI --data-inicial > este header > settings.toml.
# Datas PASSADAS => reanálise (ERA5); FUTURAS => previsão; cruzando hoje => emenda observado+previsão.
# ══════════════════════════════════════════════════════════════════════════════════════════════════
_CONFIG_HEADER = """#toml
DATA_INICIAL = "2026-07-16"
DATA_FINAL   = "2026-08-16"

RUN_GFS       = false
RUN_GEFS      = true
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

# Variáveis a plotar (1 MP4 por variável) — copie o nome (com aspas e vírgula) e cole em VARIAVEIS_GLOBO_3D:

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

VARIAVEIS_GLOBO_3D = ["olr_cores_psi200_contornos","psi200_anom","wnd200_zonal_cores_psi200_contornos"]

GLOBO_3D_VARIANTES_AUTO = false   # true = ao plotar z250_anom, gera TB a z250_anom_5d (média móvel 5d); false = só a diária. Só afeta z250_anom

# Voo da câmera (lon negativa = Oeste)
GLOBO_3D_LON_INICIAL = -80.0
GLOBO_3D_LAT_INICIAL = -23.0
GLOBO_3D_LON_FINAL   = -80.0
GLOBO_3D_LAT_FINAL   = -23.0
GLOBO_3D_INCLINACAO  = ""
GLOBO_3D_VOLTAS_EXTRA = 0.0
GLOBO_3D_EASING = "ease_in_out"
GLOBO_3D_VELOCIDADE_VAR = 1.0
GLOBO_3D_FADE_IN_SEG  = 0
GLOBO_3D_FADE_OUT_SEG = 1.5

# Render
GLOBO_3D_FRAMES_POR_DIA = 5
GLOBO_3D_FPS = 8
GLOBO_3D_TAMANHO_PX = 1080
GLOBO_3D_WORKERS = 3
GLOBO_3D_RENDER_BUFFER = 2
GLOBO_3D_BG_CACHE = true
GLOBO_3D_BG_CACHE_CHECK = true
GLOBO_3D_GRID_DEG = 0.5
GLOBO_3D_COARSEN = 1
GLOBO_3D_PROJECTION = "nearside"    # "nearside" | "orthographic"
GLOBO_3D_OLR_OVERLAY = false

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

# Isolinhas de PNMM (só a variável tmp850_mslp desenha)
GLOBO_3D_MSLP_INTERVALO = 3.0
GLOBO_3D_MSLP_COR       = "black"
GLOBO_3D_MSLP_LW        = 0.5
GLOBO_3D_MSLP_SIGMA     = 2.0

# Caixa de texto LIVRE (anotação ancorada em lat/lon)
GLOBO_3D_CAIXA_LIVRE          = false
GLOBO_3D_CAIXA_LIVRE_LAT      = -30
GLOBO_3D_CAIXA_LIVRE_LON      = -110
GLOBO_3D_CAIXA_LIVRE_TEXTO    = "Powerful subtropical jet"
GLOBO_3D_CAIXA_LIVRE_COR_BOX  = "#158c28"
GLOBO_3D_CAIXA_LIVRE_COR_TEXTO = "white"
GLOBO_3D_CAIXA_LIVRE_CONTORNO_COR = "white"
GLOBO_3D_CAIXA_LIVRE_CONTORNO_LW  = 0.8
GLOBO_3D_CAIXA_LIVRE_PAD      = 0.30
GLOBO_3D_CAIXA_LIVRE_FONTSIZE = 12.0
GLOBO_3D_CAIXA_LIVRE_LARGURA  = 22
GLOBO_3D_CAIXA_LIVRE_FIXA     = true
GLOBO_3D_CAIXA_LIVRE_INICIO   = 0.0
GLOBO_3D_CAIXA_LIVRE_FADE     = 0.12
GLOBO_3D_CAIXA_LIVRE_ALPHA_MAX = 1.0
GLOBO_3D_CAIXA_LIVRE_SOMBRA   = false
"""

# ── s39 - Globo 3D animado (mídia) estilo Guillaume: voo da câmera + evolução temporal ─────────────
# Cópia do s38 (mesmo motor/pipeline), estilo Guillaume: caixa do nome da variável + barra de gradiente.
# Modo (reanálise/forecast) decidido pelas datas. Motor genérico em app/src/uteis/globo_3d_anim.py.
# Saída: Saida/s39_GLOBO_MIDIA/{REANALISE|FORECAST/<MODELO>}/s39_<var>.mp4. Criado em: 2026-06-26.

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
    gerar_animacao,
)

SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's39'
SCRIPT_DESC = 's39 - Globo 3D animado (midia) estilo Guillaume'


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
    aplicar_config_header(_CONFIG_HEADER)   # injeta a config do topo no settings (env/CLI ainda vencem)
    logger = get_logger(SCRIPT_ID)
    logger.info('=' * 80)
    logger.info('SCRIPT {}: {}', SCRIPT_ID.upper(), SCRIPT_DESC)
    logger.info('=' * 80)

    variaveis = _get_variaveis()
    output_base = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_GLOBO_MIDIA'
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
        'projection': str(getattr(settings, 'GLOBO_3D_PROJECTION', 'nearside')),
        'credito': str(getattr(settings, 'GLOBO_3D_CREDITO', 'Bruno Capucin')),
        'vinheta': bool(getattr(settings, 'GLOBO_3D_VINHETA', True)),
        'atmosfera': bool(getattr(settings, 'GLOBO_3D_ATMOSFERA', True)),
        'jato': bool(settings.get('GLOBO_3D_JATO', False)),  # master unico das correntes de jato (s38/s39/s40/s41)
        'paletas': {v: list(settings.get(f'GLOBO_3D_PALETA_{v.upper()}', []) or []) for v in variaveis},
        'fonte_titulo': str(getattr(settings, 'GLOBO_3D_FONTE_TITULO', '')),
        'fonte_legenda': str(getattr(settings, 'GLOBO_3D_FONTE_LEGENDA', '')),
        'tamanho_px': int(getattr(settings, 'GLOBO_3D_TAMANHO_PX', 1080)),
        'script_version': '3.92-guillaume',  # corrente de jato disponivel em qualquer campo (master unico GLOBO_3D_JATO)
    }

    # Por padrao SEMPRE regenera o MP4 (GLOBO_3D_SEMPRE_REGERAR=true): saida de midia iterada
    # visualmente, e features de aparencia (box, caixa livre, camera, cores...) NAO entram na
    # chave de cache — sem isso, mudar uma dessas daria cache-hit e traria um video velho. Os
    # DOWNLOADS seguem em cache (so o render e refeito). Ponha false p/ reativar o skip por cache.
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
