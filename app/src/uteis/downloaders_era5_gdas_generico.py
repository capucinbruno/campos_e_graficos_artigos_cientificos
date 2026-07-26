"""Downloader "emenda" ERA5 + GDAS: tenta o ERA5 primeiro e completa com GDAS SO o que o
proprio CDS disser que falta -- sem assumir um numero fixo de dias de atraso.

O ERA5 (reanalise) tem uma latencia de publicacao que VARIA (perto de hoje, mas nao e sempre o
mesmo numero de dias -- pode estar disponivel ate mais perto de hoje num dia e menos noutro).
Em vez de cortar num numero fixo (ex. "sempre os ultimos 7 dias sao GDAS"), esta funcao:

    1. Pede o periodo inteiro ao ERA5.
    2. Se o CDS recusar por o periodo ainda nao estar completo, ele informa a data real ate onde
       tem dado (`Era5PeriodoIncompleto.ultimo_dia_disponivel`, extraida da propria mensagem de
       erro do CDS) -- usamos ESSA data, nao uma estimativa.
    3. So entao busca GDAS para o que sobrou (do dia seguinte ao ultimo disponivel ate `end`).

Se o CDS nao informar a data exata (mensagem de erro em formato inesperado), cai para uma
margem fixa de seguranca (`DIAS_GDAS_RECENTE`) como ultimo recurso.

Uso tipico (dentro de um script em artigos/<artigo>/) -- troque `ensure_era5_for_period` por
esta funcao quando o periodo pedido pelo usuario pode chegar proximo de hoje:

    from app.src.uteis.downloaders_era5_gdas_generico import ensure_observado_for_period

    arquivos = ensure_observado_for_period(
        artigo='artigo_JBN_AS_17_07_2026',
        variavel='geopotencial',
        nivel=500,
        start=dt_ini,
        end=dt_fim,
    )
"""

from __future__ import annotations

# Bibliotecas padrão
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

# Módulos locais
from app.shared.logger import get_logger
from app.src.uteis.downloaders_era5_generico import Era5PeriodoIncompleto, ensure_era5_for_period
from app.src.uteis.downloaders_gdas_generico import ensure_gdas_for_period

logger = get_logger(__name__)

# Usado SO quando o CDS recusa o periodo mas nao informa a data exata ate onde tem dado (formato
# de erro inesperado) -- ultimo recurso, nao a regra. Nesse caso assumimos essa margem conservadora
# (ver .claude/rules/gotchas.md: ERA5 tem ~5 dias de atraso na pratica; 7 da folga extra).
DIAS_GDAS_RECENTE = 7


def ensure_observado_for_period(
    artigo: str,
    variavel: str,
    start: datetime,
    end: datetime,
    nivel: int | None = None,
    area: tuple[float, float, float, float] | None = None,
    hours_utc: Sequence[int] | None = None,
    force_redownload: bool = False,
) -> list[Path]:
    """Garante os arquivos de `variavel` para [start, end], usando ERA5 e completando com GDAS
    apenas o trecho que o CDS confirmar estar indisponivel ainda.

    Args:
        artigo: Pasta do artigo em artigos/ -- ver `ensure_era5_for_period`/`ensure_gdas_for_period`.
        variavel: Chave cadastrada em `variaveis_meteorologicas.VARIAVEIS`.
        start: Data inicial (inclusive).
        end: Data final (inclusive).
        nivel: Nivel de pressao em hPa, se a variavel exigir.
        area: Bounding box `(N, W, S, E)` em graus. None = dominio global/grade cheia.
        hours_utc: Horas sinoticas (default 00/06/12/18 UTC nas duas fontes).
        force_redownload: Se True, ignora arquivos ja baixados.

    Returns:
        Lista de paths (mensais do ERA5 seguidos dos diarios do GDAS, quando houver os dois
        trechos). Mesma variavel de saida nas duas fontes -- da pra concatenar direto, ex.:
        `xr.open_mfdataset(sorted(arquivos))`.
    """
    try:
        arquivos = ensure_era5_for_period(
            artigo, variavel, start, end, nivel, area, hours_utc, force_redownload
        )
        logger.info(f'{variavel}: ERA5 cobriu o periodo inteiro ({start.date()} -> {end.date()})')
        return arquivos
    except Era5PeriodoIncompleto as e:
        ultimo_dia_era5 = e.ultimo_dia_disponivel
        if ultimo_dia_era5 is None:
            ultimo_dia_era5 = datetime.now(timezone.utc).date() - timedelta(days=DIAS_GDAS_RECENTE + 1)
            logger.warning(
                f'{variavel}: CDS recusou o periodo mas nao informou ate onde tem dado -- '
                f'usando margem de seguranca de {DIAS_GDAS_RECENTE} dias (assumindo ate '
                f'{ultimo_dia_era5}).'
            )

    arquivos: list[Path] = []

    if ultimo_dia_era5 >= start.date():
        era5_end = datetime.combine(ultimo_dia_era5, datetime.min.time())
        logger.info(
            f'{variavel}: ERA5 disponivel ate {ultimo_dia_era5} -- baixando {start.date()} -> {ultimo_dia_era5}'
        )
        arquivos += ensure_era5_for_period(
            artigo, variavel, start, era5_end, nivel, area, hours_utc, force_redownload
        )

    gdas_start_date = ultimo_dia_era5 + timedelta(days=1)
    if gdas_start_date <= end.date():
        gdas_start = datetime.combine(max(gdas_start_date, start.date()), datetime.min.time())
        logger.info(
            f'{variavel}: completando com GDAS de {gdas_start.date()} -> {end.date()} '
            f'(ainda indisponivel no ERA5)'
        )
        arquivos += ensure_gdas_for_period(
            artigo, variavel, gdas_start, end, nivel, area, hours_utc, force_redownload
        )

    return arquivos
