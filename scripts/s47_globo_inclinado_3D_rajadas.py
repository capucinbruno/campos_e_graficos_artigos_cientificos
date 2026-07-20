_CONFIG_HEADER = """#toml

############## DATAS E MODELOS ############
# ATENÇÃO: rajada é FORECAST-ONLY (só ECMWF-HRES). DATA_INICIAL no PASSADO faz o motor pedir
# reanálise e abortar — reveja esta data a cada dia que for rodar.
DATA_INICIAL = "2026-07-19"
DATA_FINAL   = "2026-07-21"
# Rajada NÃO acumula no tempo: cada frame é o MÁXIMO daquele dia (cada dia o seu). false = diário.
ACUMULAR_NO_TEMPO = false

RUN_GFS       = false
RUN_GEFS      = false
RUN_ECMWF     = true    # rajada (10fg) só existe no ECMWF-HRES
RUN_ECMWF_ENS = false
RUN_AIFS      = false
RUN_AIFS_ENS  = false
RUN_AIGFS     = false
RUN_AIGEFS    = false
RUN_CFS       = false

FORECAST_INIT = "" #lançar datas antigas, para rodadas antigas
RODADA        = "00" #hora da rodada           
GEFS_FORECAST_LEAD_DAYS = 35
ECMWF_ENS_MEMBERS = 30
ECMWF_ENS_WORKERS = 64

############## LISTA DE VARIÁVEIS ##############
#   "rajada_abs"      – RAJADA de vento a 10 m (máx diária, km/h), SÓ forecast ECMWF-HRES; escala/paleta no fim deste header

VARIAVEIS_GLOBO_3D = ["rajada_abs"]

############## ENQUADRAMENTO DA CAMERA ##############
# Escolha UM preset da lista ENQUADRAMENTOS do settings.toml (a mãe) pelo nome. Ele carrega os 5
# knobs de câmera de uma vez (altura/zoom, nadir, longitude, deitar, proporção do globo) — eles são
# ACOPLADOS, então viajam juntos: mexer na altura sem mexer no nadir joga o alvo p/ fora do quadro.
#
#   "sul_brasil"          -> RS/Uruguai no centro-direita, Pampa pela esquerda, span ~13° (PADRÃO)
#   "sul_brasil_centrado" -> mesmo zoom, Sul centrado no quadro
#   "sul_brasil_amplo"    -> RS até SP/MS, span ~22° (o mais parecido com o Google Earth)
#
# Para criar um novo: copie uma linha da lista em app/settings/settings.toml (lá tem o que cada
# campo faz, a fórmula da curvatura e o limite de nitidez do satélite) e dê um nome novo.
# "" = ignora a lista e usa os GLOBO_3D_INC_* soltos (da mãe, ou os que você puser aqui).
ENQUADRAMENTO = "sul_brasil"

GLOBO_3D_INC_EASING      = ""      # linear | ease_out | ease_in | ease_in_out ("" = linear)
GLOBO_3D_INC_VOLTAS_EXTRA   = 0.0  # giros completos (360°) extras antes de assentar
GLOBO_3D_INC_VELOCIDADE_VAR = 1.0  # velocidade da variável animada (1=normal, 2=2x) — desacopla o campo do voo

############## RENDERIZAÇÃO (FRAMES ETC) ##############
GLOBO_3D_INC_FRAMES_POR_DIA = 10   # + frames = voo MAIS LENTO e suave (+ render/RAM). Principal regulador da lentidão
GLOBO_3D_INC_FPS            = 8    # fps do MP4
GLOBO_3D_GIF_MEDIA          = false  # s46 NÃO entrega GIF: só MP4 + PNG. true = 3ª saída (resumo animado)
# (GIF_FRAMES/GIF_FPS abaixo só valem se você religar o GLOBO_3D_GIF_MEDIA acima.)
GLOBO_3D_INC_GIF_FRAMES     = 48   # frames do loop do GIF (base do fluxo do jato)
GLOBO_3D_INC_GIF_FPS        = 12   # fps do GIF
GLOBO_3D_TAMANHO_PX     = 1080     # resolução (px)
GLOBO_3D_WORKERS        = 2        # CPUs no render paralelo (0 = todas). O pico de RAM é dominado pela
# CONCORRÊNCIA (medido, 40 frames: 1 worker = 1,4 GB | 2 = 3,5 GB | 3 = 4,7 GB), mais ~18 MB por frame
# de acumulação. Com 3, o clipe de 100 frames projeta ~5,8 GB e a máquina tem ~7 livres — sem margem.
# Com 2, ~4,6 GB. Suba para 3 só se a máquina tiver folga (o WSL já caiu 2x por isso).
GLOBO_3D_RENDER_BUFFER  = 2        # frames em voo ALÉM dos workers (pico de RAM = workers+este)
GLOBO_3D_BG_CACHE       = true
GLOBO_3D_BG_CACHE_CHECK = true
# Grade das séries (`_target_grid`). O ECMWF/GFS de chuva é 0.25° NATIVO: com 0.5 o motor
# reamostrava pra metade da resolução ANTES de plotar, e no zoom do inclinado isso é o degrau que
# mais aparece. 0.25 = usa o dado como veio (não inventa nada).
GLOBO_3D_GRID_DEG       = 0.25
GLOBO_3D_COARSEN        = 1
GLOBO_3D_NIVEIS         = 16
GLOBO_3D_OLR_OVERLAY    = false
GLOBO_3D_SEMPRE_REGERAR = true

############## SATÉLITE (BLUE MARBLE) E LINHAS ##############
# Resolução da FONTE do satélite. 4096 = default histórico (9,8 km/px) e BORRA no zoom do inclinado;
# 8192 = nitidez nativa do Entrada/blue_marble.png (4,9 km/px).
# CUSTO REAL: não é o array de 100 MB (esse sim é alocado 1x e herdado por CoW) — é o WARP. O cartopy
# reprojeta com uma KDTree sobre TODOS os pixels da fonte, POR FRAME E POR WORKER: em 8192 isso media
# 2,8 GB por warp (vs 0,87 GB em 4096) e derrubava a máquina. O que viabiliza o 8192 é o recorte no
# bbox visível (GLOBO_3D_BLUE_MARBLE_CROP, default true): ~0,36 GB por warp com a MESMA nitidez.
GLOBO_3D_BLUE_MARBLE_MAX_PX = 8192
GLOBO_3D_BLUE_MARBLE_CROP   = true
# (a malha da reprojeção por frame é o GLOBO_3D_INC_BLUE_MARBLE_REGRID, no bloco de regrids abaixo;
#  são coisas diferentes: subir aquele NÃO tirava o bloco, porque o gargalo era a fonte.)

# Espessura das linhas pretas (por variável; _S46 sobrepõe só neste script). A vista rasante do
# inclinado engole linha fina — estes valores são ~2x o default da ficha (1.0 / 0.8 / 0.6).
GLOBO_3D_LW_COAST_RAJADA_ABS_S47  = 2.0   # costas/litoral
GLOBO_3D_LW_BORDER_RAJADA_ABS_S47 = 1.6   # divisas de país
GLOBO_3D_LW_STATES_RAJADA_ABS_S47 = 1.2   # divisas estaduais
# Idem p/ o globo SEM variável (usado no smoke test de câmera):
GLOBO_3D_LW_COAST_SEM_VARIAVEL_S46  = 2.0
GLOBO_3D_LW_BORDER_SEM_VARIAVEL_S46 = 1.6
GLOBO_3D_LW_STATES_SEM_VARIAVEL_S46 = 1.2

GLOBO_3D_INC_BLUE_MARBLE = true   # (s44/s46) satélite leve no continente; false = continente CINZA
GLOBO_3D_COR_OCEANO_SEM_VARIAVEL    = "#0e426d"
GLOBO_3D_COR_CONTINENTE_SEM_VARIAVEL = "#d9d9d9"   # cor da terra quando o satélite está OFF
PLOTAR_SOMENTE_BRASIL     = true
PLOTAR_SOMENTE_CONTINENTE = true

############## CAIXAS DE TEXTO DE TÍTULO DO VÍDEO ##############
PADRAO_TWC = true
TITULO_THE_WEATHER_CHANNEL_COR  = "#0077a7" # azul padrão #0077a7 | vermelho #d3012f                           # true = só crédito + caixa azul do título + caixa da data
TITULO_THE_WEATHER_CHANNEL      = "Rajada de vento"
# "" = a caixa cinza mostra a DATA do frame. Com ACUMULAR_NO_TEMPO=true ela vira a JANELA ACUMULADA,
# ancorada na DATA_INICIAL e crescendo com o vídeo ("Jul 17" -> "Jul 17–18" -> ... -> "Jul 17–25") —
# que é o que o campo mostra (cada frame é o total desde o início). A figura estática (_total.png)
# recebe o intervalo fechado do período. Texto fixo aqui sobrescreve tudo isso.
TITULO_THE_WEATHER_CHANNEL_DATA = ""
GLOBO_3D_FONTE_TWC = "Liberation Sans"
# Caixa "Modelo <MODELO>" colada sob a azul, alinhada à esquerda com ela. A sigla vem do modelo
# habilitado (RUN_GFS/RUN_GEFS/RUN_ECMWF...) — não precisa repetir o nome aqui.
TITULO_THE_WEATHER_CHANNEL_MODELO          = true
TITULO_THE_WEATHER_CHANNEL_MODELO_COR      = "#14223d"
TITULO_THE_WEATHER_CHANNEL_MODELO_PREFIXO  = "Modelo"
TITULO_THE_WEATHER_CHANNEL_MODELO_FONTSIZE = 12.0
# Legenda de faixas ao lado da caixa do modelo. Só desenha as faixas que APARECEM pintadas no quadro
# (máximo do período dentro do enquadramento e do recorte do Brasil) — se a rajada não passa de
# 90-100 km/h, a legenda para ali. Rótulos e cores saem dos GLOBO_3D_LEVELS/PALETA_RAJADA_ABS abaixo.
TITULO_THE_WEATHER_CHANNEL_LEGENDA          = true
TITULO_THE_WEATHER_CHANNEL_LEGENDA_FONTSIZE = 8.0
TITULO_THE_WEATHER_CHANNEL_LEGENDA_BORDA_LW = 1.2

############## FEATURES COMO ESTRELAS, ATMOSFERA, CRÉDITO ##############
LOGO_CAPUCIN = true
LOGO_AMPERE  = false
GLOBO_3D_CREDITO = ""
GLOBO_3D_ATMOSFERA_ESTRELAS = false   # halo + estrelas (habilitados no inclinado; geometria já ajustada)
GLOBO_3D_SOMENTE_ESTRELAS   = false  # só vale se ATMOSFERA_ESTRELAS = false
GLOBO_3D_FONTE_LEGENDA = "Ubuntu Sans"

############## EFEITOS DE FADE-IN & FADE-OUT ##############
GLOBO_3D_FADE_IN_SEG  = 0     # MP4: sai do preto nos primeiros N seg (0 = off)
# O último frame (= acumulado TOTAL do período) fica CONGELADO 6 s em brilho cheio e só então começa
# o fade-out, que escurece até o preto ao longo de 2 s. Sem o congelamento, o fade morderia os
# últimos frames renderizados e o total já apareceria escurecendo.
GLOBO_3D_ULTIMO_FRAME_SEG = 6.0
GLOBO_3D_FADE_OUT_SEG     = 2.0   # MP4: escurece até o preto nos últimos N seg (0 = off)
GLOBO_3D_FADE_OUT_HOLD_SEG = 0.4  # segundos no preto no fim, p/ o vídeo não cortar seco

############### CAIXA DE TEXTO LIVRE ##########################
GLOBO_3D_CAIXA_LIVRE          = false
GLOBO_3D_CAIXA_LIVRE_LAT      = -30
GLOBO_3D_CAIXA_LIVRE_LON      = -110
GLOBO_3D_CAIXA_LIVRE_TEXTO    = "Heavy rainfall"
GLOBO_3D_CAIXA_LIVRE_COR_BOX  = "#158c28"
GLOBO_3D_CAIXA_LIVRE_COR_TEXTO = "white"
GLOBO_3D_CAIXA_LIVRE_CONTORNO_COR = "white"
GLOBO_3D_CAIXA_LIVRE_CONTORNO_LW  = 1.5
GLOBO_3D_CAIXA_LIVRE_PAD      = 0.30
GLOBO_3D_CAIXA_LIVRE_FONTSIZE = 11.0
GLOBO_3D_CAIXA_LIVRE_LARGURA  = 22
GLOBO_3D_CAIXA_LIVRE_FIXA     = true
GLOBO_3D_CAIXA_LIVRE_INICIO   = 0.0
GLOBO_3D_CAIXA_LIVRE_FADE     = 0.12
GLOBO_3D_CAIXA_LIVRE_ALPHA_MAX = 1.0
GLOBO_3D_CAIXA_LIVRE_SOMBRA   = false

# Paleta/escala da RAJADA (km/h). Levels em km/h; abaixo do 1º nível (40) = transparente.
# 9 faixas: <50, 50-60, 60-70, 70-80, 80-90, 90-100, 100-120, 120-140, 140-160 km/h (9 cores)
GLOBO_3D_LEVELS_RAJADA_ABS = [40, 50, 60, 70, 80, 90, 100, 120, 140, 160]
GLOBO_3D_PALETA_RAJADA_ABS = ["#3ac5ff", "#0037fe", "#fd38e8", "#8b0100", "#b81310", "#e31302", "#eea5bd", "#ffe6ff", "#ceffff"]
# (GLOBO_3D_COR_OCEANO_RAJADA_ABS não entra: no globo INCLINADO a cor do oceano vem SEMPRE do
#  GLOBO_3D_COR_OCEANO_SEM_VARIAVEL, que é a base do disco — a cor por variável é ignorada aqui.)
GLOBO_3D_PCOLORMESH_RAJADA_ABS = false    # bandas discretas via contourf (não pcolormesh)
# Bandas LISAS sem perder o pico: interpolação CÚBICA do campo (6x) antes de rasterizar (mesmo motivo
# da chuva: a célula de 0.25° vira ~21 px no zoom e as facetas do contourf viram degraus). Cúbica passa
# pelos pontos da grade -> valores intactos (NÃO usar GLOBO_3D_SIGMA, que achataria o pico da rajada).
GLOBO_3D_SHADE_UPSAMPLE_RAJADA_ABS = 6

# PNMM (pressão ao nível médio do mar) em ISOLINHAS por cima da rajada. true = plota; false = só a
# rajada. Média diária de `msl` do ECMWF (o mesmo modelo). Controla se você quer o mapa de tempestade.
GLOBO_3D_ISOLINHA_MSLP_RAJADA_ABS = false

# Cor e espessura das isolinhas de PNMM (só valem quando a flag acima está true). Aceita qualquer cor
# do matplotlib: nome ("white", "black", "red") ou hex "#rrggbb". Lidos pelo motor em globo_3d_anim.py.
GLOBO_3D_MSLP_COR = "white"
GLOBO_3D_MSLP_LW  = 0.5

# Sombreado CONTÍNUO (gouraud) nas DEMAIS variáveis do s47 (a rajada usa o contourf acima).
GLOBO_3D_PCOLORMESH_S47 = true

# Regrid das camadas que cobrem o globo INTEIRO: a janela 16:9 do inclinado estica o warp (~1.8x de
# RAM); a saída é ~1080px, então 2048 é oversampling invisível. Só o s44/s46 leem estes INC_.
GLOBO_3D_INC_SHADE_REGRID            = 1280   # sombreado
GLOBO_3D_INC_JATO_DRAPE_REGRID       = 1280   # corrente de jato
GLOBO_3D_INC_ESCOAMENTO_DRAPE_REGRID = 1280   # setas do escoamento de baixos níveis
GLOBO_3D_INC_BLUE_MARBLE_REGRID      = 1280   # satélite do continente
GLOBO_3D_ESCOAMENTO_DRAPE_MAX_PX     = 3000   # teto na maior dimensão do raster do escoamento

GLOBO_3D_INC_LOGO_WIDTH_FRAC    = 0.24   # tamanho do logo (MAIOR = logo maior)
GLOBO_3D_INC_LOGO_MARGEM_X_FRAC = 0.025  # folga até a borda DIREITA
GLOBO_3D_INC_LOGO_MARGEM_Y_FRAC = 0.03   # folga até a borda INFERIOR

# (GLOBO_3D_FADE_CAUDA/_DUR_SEG não entram: o motor as lê só quando script_id == 's42'.)
GLOBO_3D_MP4_MEDIA_FIXA      = false     # true = MP4 usa a MÉDIA do período + câmera parada (igual ao GIF)
GLOBO_3D_SUAVIZACAO_REGIONAL_HGT       = true
GLOBO_3D_SUAVIZACAO_REGIONAL_HGT_SIGMA = 10.0

GLOBO_3D_JATO_PNG              = true    # PNG da média sai COM jato (parado, fase 0)
GLOBO_3D_JATO_MP4_CAUDA_SEG    = 15.0
GLOBO_3D_JATO_STRIPE_N         = 3
GLOBO_3D_JATO_STRIPE_LARGURA_DEG = 0.5
GLOBO_3D_JATO_DRAPE            = true    # jato drapejado na superfície (perspectiva 3D correta)
"""

# ── s47 - Globo 3D INCLINADO de RAJADAS: cópia do s46, rajada de vento a 10 m (só ECMWF) ───────────
# Idêntico ao s44 (globo "deitado"/inclinado estilo Google Earth com pitch): MESMO motor, MESMO
# enquadramento inclinado e MESMO pipeline de dados (reanálise/previsão/emenda). A finalidade é separar
# os vídeos de RAJADA de vento (10fg, ECMWF-HRES) do s46, cada um com sua própria config (header), cache
# e pasta de saída. Difere do s44 nas saídas: aqui são DUAS (MP4 + PNG), sem o GIF do resumo
# (GLOBO_3D_GIF_MEDIA = false no header) — o s44 segue com as três.
#   - Variável: VARIAVEIS_GLOBO_3D do header acima (não mais GLOBO_3D_VARIAVEIS_S46 do settings.local)
#   - Modo:     AUTOMÁTICO pelas datas (passado=reanálise, futuro=previsão, cruza hoje=emenda)
#   - Voo/enquadramento: namespace INC_ (o motor trata s47 igual s44: `_inclinado`, regrid leve,
#     PLOTAR_SOMENTE, estilo Guillaume/TWC). Aparência por VARIÁVEL/SCRIPT -> o s47 pode ter seu
#     próprio GLOBO_3D_PCOLORMESH_S47 sem tocar no s44/s46.
# Saída (dois arquivos por variável, em FORECAST/ECMWF/):
#   - s47_<variavel>.mp4        : vídeo do período (voo inclinado + evolução dia a dia da rajada máx)
#   - s47_<variavel>_media.png  : rajada MÉDIA do período (ACUMULAR_NO_TEMPO=false; rajada não acumula)
#   (sem GIF: GLOBO_3D_GIF_MEDIA = false no header. Religue com true se precisar.)
# Criado em: 2026-07-17 (cópia do s46, finalidade = rajada de vento a 10 m; só ECMWF-HRES).

# Bibliotecas padrao
import os
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

SCRIPT_ID = Path(__file__).stem.split('_')[0]  # 's46'
SCRIPT_DESC = 's47 - Globo 3D INCLINADO de RAJADAS de vento (copia do s46; so ECMWF)'


# Campos do preset de ENQUADRAMENTOS (settings.toml) -> chaves GLOBO_3D_INC_* que o motor le.
# A `inclinacao` tambem preenche LAT_INICIAL/FINAL: o motor as ignora enquanto INCLINACAO != '',
# mas deixa-las com outro valor faria o header/log mentirem sobre a latitude da camera.
_ENQUADRAMENTO_MAPA = {
    'altura':      ('GLOBO_3D_INC_ALTURA',),
    'inclinacao':  ('GLOBO_3D_INC_INCLINACAO', 'GLOBO_3D_INC_LAT_INICIAL', 'GLOBO_3D_INC_LAT_FINAL'),
    'lon':         ('GLOBO_3D_INC_LON_INICIAL', 'GLOBO_3D_INC_LON_FINAL'),
    'deitar':      ('GLOBO_3D_INC_DEITAR',),
    'janela_frac': ('GLOBO_3D_INC_JANELA_FRAC',),
    'aspect':      ('GLOBO_3D_INC_ASPECT',),
}


def _aplicar_enquadramento() -> None:
    """Expande o preset escolhido em ENQUADRAMENTO (topo deste script) nas chaves GLOBO_3D_INC_*.

    A lista ENQUADRAMENTOS vive no settings.toml (mae) por ser identidade fixa, nao config de rodada.
    ENQUADRAMENTO = '' -> no-op (valem os GLOBO_3D_INC_* soltos). Mesma precedencia do header: quem
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
    """Lista de variaveis do s46, vinda do VARIAVEIS_GLOBO_3D do header (que sobrepoe o
    settings.local). Strings vazias sao ignoradas; globo so com features = 'sem_variavel' explicito."""
    variaveis: list[str] = []
    for v in (settings.get('VARIAVEIS_GLOBO_3D', None) or []):
        _v = str(v).strip()
        if _v and _v not in variaveis:
            variaveis.append(_v)
    if not variaveis:
        raise ValueError(
            'Nenhuma variavel definida para o s46. Preencha VARIAVEIS_GLOBO_3D no bloco '
            '_CONFIG_HEADER no topo deste arquivo — pelo menos UMA variavel '
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
    aplicar_config_header(_CONFIG_HEADER)   # injeta a config do topo no settings (env/CLI ainda vencem)
    logger = get_logger(SCRIPT_ID)
    logger.info('=' * 80)
    logger.info('SCRIPT {}: {}', SCRIPT_ID.upper(), SCRIPT_DESC)
    logger.info('=' * 80)
    _aplicar_enquadramento()                # preset de camera (ENQUADRAMENTO) -> GLOBO_3D_INC_*

    variaveis = _get_variaveis()
    output_base = Path(settings.DIR_OUTPUT) / f'{SCRIPT_ID}_GLOBO_INCLINADO_RAJADAS'
    # Plano (modo decidido pelas datas): caminhos esperados p/ validar o cache.
    plano, _, _ = _output_plan(variaveis, output_base)
    # Saidas por variavel: MP4 do periodo + PNG do resumo (+ GIF do resumo so se GLOBO_3D_GIF_MEDIA,
    # que o header deste script desliga). O sufixo do resumo acompanha o que a figura mostra: '_total'
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
        # Voo: namespace INC_ (o motor le INC_ com precedencia no s46), fallback ao compartilhado.
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
        # Enquadramento INCLINADO (o mesmo do s44).
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
        'script_version': '1.0-inclinado-rajadas',  # copia do s46 p/ rajada de vento
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
