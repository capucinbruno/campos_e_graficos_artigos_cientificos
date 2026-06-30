"""Pecas compartilhadas do modo previsao (forecast) entre os scripts dinamicos.

Concentra o que e GENERICO e independe das variaveis de cada script:
  - flags de modelos habilitados (RUN_GFS/RUN_GEFS/...) e seu continente de saida;
  - janelas moveis (media deslizante) e pentadas fixas;
  - escolha da fonte de reanalise (ERA5 vs GDAS) pela latencia do ERA5.

O dicionario de DOWNLOADERS por modelo NAO mora aqui de proposito: cada script precisa de um
conjunto diferente de variaveis (o s34 baixa 8 campos; o s16, 4), entao cada um mantem o seu
proprio mapeamento modelo -> downloaders. Aqui ficam so as partes realmente comuns.

Originado da fatoracao do s34 (scripts/s34_wave_guide_rossby_wave.py); reutilizado pelo s16.
"""

from datetime import datetime, timedelta

from app.shared.settings_factory import settings

# Latencia tipica do ERA5 (dias): antes disso a data ainda nao esta disponivel na reanalise.
ERA5_LATENCY_DAYS = 7
# Horas sinoticas usadas para a media diaria (reduz aliasing do ciclo diurno).
DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)

# Pentadas: N janelas FIXAS (nao moveis) de 5 dias a partir do dia SEGUINTE a data da rodada.
# Ex.: rodada dia 15 -> P1=16..20, P2=21..25, P3=26..30. Sufixo de arquivo pentadaX.
N_PENTADAS_FIXAS = 3
PENTADA_DIAS = 5
# PENTADA (janela FIXA) so e plotada se tiver >= este nº de dias com dado (4 de 5 ok; <4 pula).
MIN_DIAS_PENTADA = 4

# Flags (settings) que habilitam cada modelo em MODE='forecast'. Pode habilitar mais de um:
# o script roda o pipeline para CADA modelo habilitado, salvando em pasta propria. (default, flag)
# ECMWF_ENS = media dos 50 membros; AIFS/AIGFS/AIGEFS = modelos de IA; CFS = pseudo-ensemble.
MODEL_FLAGS = {
    'gfs': ('RUN_GFS', True), 'gefs': ('RUN_GEFS', False),
    'ecmwf': ('RUN_ECMWF', False), 'ecmwf_ens': ('RUN_ECMWF_ENS', False),
    'aifs': ('RUN_AIFS', False), 'aifs_ens': ('RUN_AIFS_ENS', False),
    'aigfs': ('RUN_AIGFS', False), 'aigefs': ('RUN_AIGEFS', False),
    'cfs': ('RUN_CFS', False),
}

# Continente do modelo, para separar a saida (americanos NCEP/NOAA vs europeus ECMWF).
MODEL_CONTINENT = {
    'gfs': 'MODELOS_AMERICANOS', 'gefs': 'MODELOS_AMERICANOS',
    'aigfs': 'MODELOS_AMERICANOS', 'aigefs': 'MODELOS_AMERICANOS',
    'ecmwf': 'MODELOS_EUROPEUS', 'ecmwf_ens': 'MODELOS_EUROPEUS',
    'aifs': 'MODELOS_EUROPEUS', 'aifs_ens': 'MODELOS_EUROPEUS',
    'cfs': 'MODELOS_AMERICANOS',
}


def enabled_models() -> list:
    """Modelos de previsao habilitados pelas flags do settings (ordem de MODEL_FLAGS)."""
    return [m for m, (flag, default) in MODEL_FLAGS.items()
            if bool(settings.get(flag, default))]


def get_data_sources(dt_ini: datetime, dt_fim: datetime):
    """Retorna (era5_period, gdas_period) — None quando nao ha dados dessa fonte.

    Datas anteriores ao cutoff (hoje - ERA5_LATENCY_DAYS) vem do ERA5; o restante do GDAS.
    """
    cutoff = (datetime.now() - timedelta(days=ERA5_LATENCY_DAYS)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    if dt_fim < cutoff:
        return (dt_ini, dt_fim), None
    if dt_ini >= cutoff:
        return None, (dt_ini, dt_fim)
    return (dt_ini, cutoff - timedelta(days=1)), (cutoff, dt_fim)


def windows(daily_dates: list, w: int) -> list:
    """Janelas moveis deslizantes de `w` dias sobre `daily_dates`. Retorna (start, end) dates.
    w<=0 ou w>=N -> uma janela unica (periodo inteiro)."""
    n = len(daily_dates)
    if w <= 0 or w >= n:
        return [(daily_dates[0], daily_dates[-1])]
    return [(daily_dates[i], daily_dates[i + w - 1]) for i in range(n - w + 1)]


def pentad_windows(run_date, n_pentadas: int = N_PENTADAS_FIXAS, pentada_dias: int = PENTADA_DIAS) -> list:
    """Janelas FIXAS de pentada a partir do dia SEGUINTE a `run_date` (data da rodada).

    Pentada k (1-based) = [run_date + 1 + (k-1)*5, run_date + k*5]. NAO sao moveis: as
    janelas sao contiguas e nao se sobrepoem. Retorna lista de (start_date, end_date)."""
    wins = []
    for k in range(n_pentadas):
        ws = run_date + timedelta(days=1 + k * pentada_dias)
        we = run_date + timedelta(days=(k + 1) * pentada_dias)
        wins.append((ws, we))
    return wins
