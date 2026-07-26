"""Downloader generico de ERA5 (Copernicus CDS) por variavel/nivel/periodo.

Em vez de um downloader por variavel (padrao antigo do projeto), esta funcao aceita a chave da
variavel (ver `variaveis_meteorologicas.py`), o nivel de pressao (se a variavel exigir) e o
periodo, e resolve o dataset/request do CDS sozinha. Pensado para scripts de artigos que
precisam de combinacoes diferentes de variavel/nivel sob demanda.

Uso tipico (dentro de um script em artigos/<artigo>/) -- os arquivos vao para
dados/<artigo>/ERA5_<variavel>[_<nivel>hPa]/, a pasta dados/<artigo>/ ja existe (o CLI cria
automaticamente uma pasta em dados/ pra cada pasta em artigos/):

    from app.src.uteis.downloaders_era5_generico import ensure_era5_for_period

    arquivos = ensure_era5_for_period(
        artigo='artigo_JBN_AS_17_07_2026',
        variavel='geopotencial',
        nivel=500,
        start=dt_ini,
        end=dt_fim,
    )
"""

from __future__ import annotations

# Bibliotecas padrão
import calendar
import re
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

# Bibliotecas de terceiros
import cdsapi
import xarray as xr

# Módulos locais
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.src.uteis.variaveis_meteorologicas import VariavelSpec, get_variavel

logger = get_logger(__name__)

URL_API_COPERNICUS = 'https://cds.climate.copernicus.eu/api'
DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)
DATASET_PRESSURE_LEVELS = 'reanalysis-era5-pressure-levels'
DATASET_SINGLE_LEVELS = 'reanalysis-era5-single-levels'
MIN_BYTES_NC = 10_000

# Projeto compartilhado entre pesquisadores -- cada um tem sua propria KEY_CDS (ver
# .secrets_example.toml), selecionada pela setting PESQUISADOR (settings.local.toml).
PESQUISADORES_VALIDOS = ('capucin', 'reboita', 'gozzo', 'vemado')


class Era5PeriodoIncompleto(RuntimeError):
    """O CDS ainda nao tem o periodo pedido completo (fim do periodo bate perto de hoje).

    `ultimo_dia_disponivel` traz a data real informada pelo CDS ate onde ha dado, quando o
    proprio erro do CDS a menciona (None se nao foi possivel extrair). Quem chama pode usar essa
    data para saber exatamente ate onde confiar no ERA5 e a partir de quando completar com outra
    fonte (GDAS) -- em vez de assumir um numero fixo de dias de atraso.
    """

    def __init__(self, message: str, ultimo_dia_disponivel: date | None):
        super().__init__(message)
        self.ultimo_dia_disponivel = ultimo_dia_disponivel


def _is_cds_no_data_error(error_text: str) -> bool:
    t = error_text.lower()
    return (
        'none of the data you have requested is available yet' in t
        or 'multiadapternodataerror' in t
        or 'no matching data' in t
        or 'requested data is not available' in t
        or 'latest date available' in t
    )


def _extract_latest_date_from_cds_error(error_text: str) -> date | None:
    """Extrai a data de 'latest date available: YYYY-MM-DD HH:MM' da mensagem de erro do CDS."""
    m = re.search(
        r'latest date available.*?:\s*(\d{4}-\d{2}-\d{2})\s+(\d{2}):(\d{2})',
        error_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), '%Y-%m-%d').date()
    except ValueError:
        return None


def _resolve_key_cds() -> str:
    """Resolve a KEY_CDS do pesquisador atual (setting `PESQUISADOR`)."""
    pesquisador = (settings.get('PESQUISADOR') or '').strip().lower()
    if not pesquisador:
        raise RuntimeError(
            'PESQUISADOR nao definido. Adicione ao seu settings.local.toml, ex.:\n'
            '  PESQUISADOR = "capucin"\n'
            f'Valores aceitos: {", ".join(PESQUISADORES_VALIDOS)}'
        )
    if pesquisador not in PESQUISADORES_VALIDOS:
        raise RuntimeError(
            f"PESQUISADOR '{pesquisador}' nao reconhecido. "
            f'Valores aceitos: {", ".join(PESQUISADORES_VALIDOS)}'
        )

    chave_setting = f'KEY_CDS_{pesquisador.upper()}'
    key = settings.get(chave_setting)
    if not key:
        raise RuntimeError(
            f'{chave_setting} nao encontrada em app/settings/.secrets.toml.\n'
            f'  Copie de .secrets_example.toml e preencha com a chave do CDS de {pesquisador} '
            '(https://cds.climate.copernicus.eu/).'
        )
    return key


def _get_cds_client() -> cdsapi.Client:
    return cdsapi.Client(
        url=URL_API_COPERNICUS,
        key=_resolve_key_cds(),
        debug=False,
        progress=True,
        retry_max=8,
        sleep_max=120,
        delete=False,
    )


def _normalize_hours(hours_utc: Sequence[int] | None) -> tuple[int, ...]:
    hours = hours_utc if hours_utc is not None else DEFAULT_SYNOPTIC_HOURS
    uniq = sorted({int(h) for h in hours})
    if not uniq or any(h < 0 or h > 23 for h in uniq):
        raise ValueError(f'Horas UTC invalidas: {hours_utc}')
    return tuple(uniq)


def _day_list(year: int, month: int, end_day: int | None) -> list[str]:
    last_day = end_day if end_day is not None else calendar.monthrange(year, month)[1]
    return [f'{d:02d}' for d in range(1, last_day + 1)]


def _iter_year_month(start: datetime, end: datetime) -> list[tuple[int, int]]:
    if end < start:
        raise ValueError("Periodo invalido: 'end' e anterior a 'start'.")
    y, m = start.year, start.month
    out: list[tuple[int, int]] = []
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _output_dir(artigo: str, chave: str, nivel: int | None) -> Path:
    sufixo = f'{chave}_{nivel}hPa' if nivel is not None else chave
    return Path(settings.DIR_DADOS) / artigo / f'ERA5_{sufixo}'


def _download_month(
    spec: VariavelSpec,
    artigo: str,
    chave: str,
    nivel: int | None,
    year: int,
    month: int,
    end_day: int | None,
    hours_utc: tuple[int, ...],
    area: tuple[float, float, float, float] | None,
    force_redownload: bool,
) -> Path:
    out_dir = _output_dir(artigo, chave, nivel)
    out_dir.mkdir(parents=True, exist_ok=True)

    hours_tag = ''.join(f'{h:02d}' for h in hours_utc)
    fname = f'era5_{chave}_{year:04d}{month:02d}_h{hours_tag}.nc'
    target = out_dir / fname

    if target.exists() and target.stat().st_size >= MIN_BYTES_NC and not force_redownload:
        logger.info(f'Arquivo ja existe, pulando download: {target}')
        return target

    dataset = DATASET_PRESSURE_LEVELS if spec.requer_nivel else DATASET_SINGLE_LEVELS
    request: dict = {
        'product_type': ['reanalysis'],
        'variable': [spec.era5_nome],
        'year': [f'{year:04d}'],
        'month': [f'{month:02d}'],
        'day': _day_list(year, month, end_day),
        'time': [f'{h:02d}:00' for h in hours_utc],
        'data_format': 'netcdf',
        'download_format': 'unarchived',
    }
    if spec.requer_nivel:
        request['pressure_level'] = [str(nivel)]
    if area is not None:
        request['area'] = list(area)  # [N, W, S, E]

    tmp = target.with_suffix(target.suffix + '.part')
    if tmp.exists():
        tmp.unlink()

    logger.info(f'Baixando {dataset} -> {target.name} ({spec.descricao}, {year:04d}-{month:02d})')
    try:
        _get_cds_client().retrieve(dataset, request, str(tmp))
    except Exception as e:
        msg = str(e)
        if _is_cds_no_data_error(msg):
            if tmp.exists():
                tmp.unlink()
            ultimo_dia = _extract_latest_date_from_cds_error(msg)
            raise Era5PeriodoIncompleto(
                f'CDS ainda nao tem {spec.descricao} completo para {year:04d}-{month:02d} '
                f'(ultimo dia disponivel informado pelo CDS: {ultimo_dia or "nao informado"}).',
                ultimo_dia_disponivel=ultimo_dia,
            ) from e
        raise RuntimeError(
            f'Falha ao baixar {spec.descricao} do CDS ({year:04d}-{month:02d}): {e}'
        ) from e

    if not tmp.exists() or tmp.stat().st_size < MIN_BYTES_NC:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f'Arquivo baixado invalido (muito pequeno ou ausente): {target}')

    if target.exists():
        target.unlink()
    tmp.rename(target)

    _renomear_e_converter(target, spec)

    logger.info(f'[OK] {spec.descricao} {year:04d}-{month:02d} salvo em {target}')
    return target


def _renomear_e_converter(path: Path, spec: VariavelSpec) -> None:
    """Renomeia a variavel principal do NetCDF baixado para `spec.var_saida` e aplica o fator
    de conversao de unidade, in-place."""
    ds = xr.open_dataset(path, engine='netcdf4')
    try:
        candidatos = [v for v in ds.data_vars if v not in {'number', 'expver'}]
        if not candidatos:
            raise KeyError(f'Nenhuma variavel de dados encontrada em {path}')
        nome_bruto = candidatos[0]
        da = ds[nome_bruto]

        if spec.era5_fator_conversao != 1.0:
            da = da * spec.era5_fator_conversao

        da.name = spec.var_saida
        da.attrs['units'] = spec.unidade
        da.attrs['long_name'] = spec.descricao
        ds_out = da.to_dataset(name=spec.var_saida)
    finally:
        ds.close()

    tmp = path.with_suffix('.renamed.nc')
    ds_out.to_netcdf(tmp)
    ds_out.close()
    path.unlink()
    tmp.rename(path)


def ensure_era5_for_period(
    artigo: str,
    variavel: str,
    start: datetime,
    end: datetime,
    nivel: int | None = None,
    area: tuple[float, float, float, float] | None = None,
    hours_utc: Sequence[int] | None = None,
    force_redownload: bool = False,
) -> list[Path]:
    """Garante os arquivos mensais de `variavel` (ERA5/CDS) para o periodo [start, end].

    Args:
        artigo: Pasta do artigo em artigos/ (ex. 'artigo_JBN_AS_17_07_2026') -- os arquivos vao
            para dados/<artigo>/ERA5_<variavel>[_<nivel>hPa]/.
        variavel: Chave cadastrada em `variaveis_meteorologicas.VARIAVEIS` (ex. 'geopotencial').
        start: Data inicial (inclusive).
        end: Data final (inclusive).
        nivel: Nivel de pressao em hPa. Obrigatorio se a variavel exigir nivel (ver
            `VariavelSpec.requer_nivel`); deve ficar None caso contrario.
        area: Bounding box `(N, W, S, E)` em graus. None = dominio global.
        hours_utc: Horas sinoticas (default 00/06/12/18 UTC).
        force_redownload: Se True, ignora arquivos ja baixados.

    Returns:
        Lista de paths dos NetCDF mensais (variavel renomeada para `var_saida`, unidade
        convertida quando aplicavel).
    """
    spec = get_variavel(variavel)
    if spec.requer_nivel and nivel is None:
        raise ValueError(f"Variavel '{variavel}' exige o parametro 'nivel' (hPa).")
    if not spec.requer_nivel and nivel is not None:
        raise ValueError(f"Variavel '{variavel}' nao aceita 'nivel' (e de nivel fixo).")

    hours = _normalize_hours(hours_utc)

    arquivos: list[Path] = []
    for year, month in _iter_year_month(start, end):
        end_day = end.day if (year == end.year and month == end.month) else None
        arquivos.append(
            _download_month(
                spec, artigo, variavel, nivel, year, month, end_day, hours, area, force_redownload
            )
        )
    return arquivos
