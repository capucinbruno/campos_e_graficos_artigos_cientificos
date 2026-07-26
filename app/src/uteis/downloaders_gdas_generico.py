"""Downloader generico de GDAS (NOMADS Grib Filter, analise f000) por variavel/nivel/periodo.

Mesma ideia do `downloaders_era5_generico.py`, mas para o GDAS -- usado para complementar o ERA5
em periodos recentes (o ERA5 tem ~5 dias de atraso; o GDAS cobre o gap). Ver
`variaveis_meteorologicas.py` para as chaves de variavel disponiveis.

Uso tipico (dentro de um script em artigos/<artigo>/) -- os arquivos vao para
dados/<artigo>/GDAS_<variavel>[_<nivel>hPa]/, a pasta dados/<artigo>/ ja existe (o CLI cria
automaticamente uma pasta em dados/ pra cada pasta em artigos/):

    from app.src.uteis.downloaders_gdas_generico import ensure_gdas_for_period

    arquivos = ensure_gdas_for_period(
        artigo='artigo_JBN_AS_17_07_2026',
        variavel='geopotencial',
        nivel=500,
        start=dt_ini,
        end=dt_fim,
    )
"""

from __future__ import annotations

# Bibliotecas padrão
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Sequence
from urllib.parse import urlencode

# Bibliotecas de terceiros
import xarray as xr

# Módulos locais
from app.common.download_helper import download_with_progress
from app.shared.logger import get_logger
from app.shared.settings_factory import settings
from app.src.uteis.variaveis_meteorologicas import VariavelSpec, get_variavel

logger = get_logger(__name__)

NOMADS_FILTER_URL = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_gdas_0p25.pl'
DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)


def _nivel_key(spec: VariavelSpec, nivel: int | None) -> str:
    if spec.requer_nivel:
        return spec.gdas_nivel_template.format(nivel=nivel)
    return spec.gdas_nivel_fixo


def _build_filter_url(
    spec: VariavelSpec,
    nivel: int | None,
    date_str: str,
    hour: int,
    area: tuple[float, float, float, float] | None,
) -> str:
    hh = f'{hour:02d}'
    params = {
        'file': f'gdas.t{hh}z.pgrb2.0p25.f000',
        _nivel_key(spec, nivel): 'on',
        f'var_{spec.gdas_var}': 'on',
        'dir': f'/gdas.{date_str}/{hh}/atmos',
    }
    if area is not None:
        n, w, s, e = area
        params.update({'subregion': 'on', 'toplat': n, 'leftlon': w, 'bottomlat': s, 'rightlon': e})
    return f'{NOMADS_FILTER_URL}?{urlencode(params)}'


def _output_dir(artigo: str, chave: str, nivel: int | None) -> Path:
    sufixo = f'{chave}_{nivel}hPa' if nivel is not None else chave
    return Path(settings.DIR_DADOS) / artigo / f'GDAS_{sufixo}'


def _abrir_e_normalizar(path_grb2: Path, spec: VariavelSpec) -> xr.Dataset:
    """Abre o GRIB2 filtrado (1 campo/nivel/hora so) e normaliza nomes de dims/variavel.

    O filtro do NOMADS ja restringe a UMA variavel e UM nivel, entao o arquivo tem uma unica
    mensagem GRIB -- nao precisa de `filter_by_keys` para desambiguar.
    """
    ds = xr.open_dataset(path_grb2, engine='cfgrib', backend_kwargs={'indexpath': ''})

    rename = {}
    for name in list(ds.dims) + list(ds.coords):
        low = name.lower()
        if low == 'latitude' and 'lat' not in ds.dims:
            rename[name] = 'lat'
        elif low == 'longitude' and 'lon' not in ds.dims:
            rename[name] = 'lon'
    if rename:
        ds = ds.rename(rename)

    if 'time' not in ds.coords and 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
    if 'time' not in ds.dims and 'time' in ds.coords:
        ds = ds.expand_dims('time')

    for dim in ('isobaricInhPa', 'level', 'pressure_level', 'heightAboveGround', 'meanSea'):
        if dim in ds.dims:
            ds = ds.isel({dim: 0}, drop=True)
        elif dim in ds.coords:
            ds = ds.drop_vars(dim)

    candidatos = [v for v in ds.data_vars if v not in {'number', 'expver'}]
    if not candidatos:
        raise KeyError(f'Nenhuma variavel de dados encontrada em {path_grb2}')
    ds = ds.rename({candidatos[0]: spec.var_saida})
    ds[spec.var_saida].attrs['units'] = spec.unidade
    ds[spec.var_saida].attrs['long_name'] = spec.descricao

    return ds[[spec.var_saida]]


def _download_day(
    spec: VariavelSpec,
    artigo: str,
    chave: str,
    nivel: int | None,
    target_date: date,
    hours: tuple[int, ...],
    area: tuple[float, float, float, float] | None,
    force_redownload: bool,
) -> Path:
    out_dir = _output_dir(artigo, chave, nivel)
    out_dir.mkdir(parents=True, exist_ok=True)

    date_str = target_date.strftime('%Y%m%d')
    nc_path = out_dir / f'gdas_{chave}_{date_str}.nc'

    if nc_path.exists() and not force_redownload:
        logger.info(f'GDAS {spec.descricao} {date_str} ja existe, pulando download.')
        return nc_path

    logger.info(f'Baixando GDAS {spec.descricao} para {date_str} ({len(hours)} sinoticas)')

    datasets = []
    grb_paths = []
    for hour in hours:
        grb_path = out_dir / f'gdas_{chave}_{date_str}_{hour:02d}z.grb2'
        url = _build_filter_url(spec, nivel, date_str, hour, area)

        ok = download_with_progress(
            url,
            str(grb_path),
            description=f'GDAS {chave} {date_str} {hour:02d}Z',
            force=force_redownload,
        )
        if not ok:
            raise RuntimeError(f'Falha ao baixar GDAS {spec.descricao} {date_str} {hour:02d}Z')

        grb_paths.append(grb_path)
        datasets.append(_abrir_e_normalizar(grb_path, spec))

    ds_day = xr.concat(datasets, dim='time', coords='minimal', compat='override').sortby('time')

    if nc_path.exists():
        nc_path.unlink()
    ds_day.to_netcdf(nc_path, engine='netcdf4')
    ds_day.close()

    for grb_path in grb_paths:
        if grb_path.exists():
            grb_path.unlink()

    logger.info(f'[OK] GDAS {spec.descricao} {date_str} salvo em {nc_path}')
    return nc_path


def ensure_gdas_for_period(
    artigo: str,
    variavel: str,
    start: datetime,
    end: datetime,
    nivel: int | None = None,
    area: tuple[float, float, float, float] | None = None,
    hours_utc: Sequence[int] | None = None,
    force_redownload: bool = False,
) -> list[Path]:
    """Garante os NetCDF diarios de `variavel` (GDAS/NOMADS) para o periodo [start, end].

    Args:
        artigo: Pasta do artigo em artigos/ (ex. 'artigo_JBN_AS_17_07_2026') -- os arquivos vao
            para dados/<artigo>/GDAS_<variavel>[_<nivel>hPa]/.
        variavel: Chave cadastrada em `variaveis_meteorologicas.VARIAVEIS` (ex. 'geopotencial').
        start: Data inicial (inclusive).
        end: Data final (inclusive).
        nivel: Nivel de pressao em hPa. Obrigatorio se a variavel exigir nivel; deve ficar None
            caso contrario.
        area: Bounding box `(N, W, S, E)` em graus. None = grade global 0.25 graus.
        hours_utc: Horas sinoticas (default 00/06/12/18 UTC).
        force_redownload: Se True, ignora arquivos ja baixados.

    Returns:
        Lista de paths dos NetCDF diarios (variavel renomeada para `var_saida`, mesmo nome usado
        pelo `downloaders_era5_generico`).
    """
    spec = get_variavel(variavel)
    if spec.requer_nivel and nivel is None:
        raise ValueError(f"Variavel '{variavel}' exige o parametro 'nivel' (hPa).")
    if not spec.requer_nivel and nivel is not None:
        raise ValueError(f"Variavel '{variavel}' nao aceita 'nivel' (e de nivel fixo).")

    hours = tuple(sorted({int(h) for h in (hours_utc or DEFAULT_SYNOPTIC_HOURS)}))

    arquivos: list[Path] = []
    current = start.date()
    end_date = end.date()
    while current <= end_date:
        arquivos.append(
            _download_day(spec, artigo, variavel, nivel, current, hours, area, force_redownload)
        )
        current += timedelta(days=1)

    logger.info(
        f'GDAS {variavel}: {len(arquivos)} arquivo(s) para periodo {start.date()} -> {end.date()}'
    )
    return arquivos
