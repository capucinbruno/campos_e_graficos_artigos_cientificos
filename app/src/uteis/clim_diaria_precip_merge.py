# app/src/uteis/clim_diaria_precip_merge.py
# -*- coding: utf-8 -*-
"""Chuva OBSERVADA diaria: MERGE (CPTEC) na America do Sul + IMERG-GPM (NASA) no resto do globo.

MERGE cobre so a America do Sul (lon -120.05 a -20.05, lat -60.05 a 32.25, grade 0.1°) com
correcao de vies (satelite + estacoes pluviometricas). Fora dai NAO ha produto do CPTEC mantido
(a pasta DAILY/GLOBO/ e um teste abandonado, so maio/2025) -- por isso o resto do globo usa o
IMERG-GPM Late Run puro (mesma fonte de satelite que alimenta o proprio MERGE: o titulo dos
.ctl do CPTEC diz "MERGE GENERATED FROM GPM-IMERG-late"), buscado via `earthaccess` (NASA,
autenticado por token).

As duas grades sao 0.1° com a MESMA fase (centros em x.x5°) -- confirmado inspecionando um
GRIB2 real do MERGE (lon 239.95..339.95, lat -60.05..32.25). Por isso o mosaico usa
`reindex_like` (sem interpolar) com fallback pra `.interp` se algum dia a fase nao bater.

MERGE tem prioridade onde existe (America do Sul); IMERG preenche o resto -- corte DURO na
borda do dominio MERGE (sem feather). Ver `_mosaic_day`.

DIA DE HOJE (parcial): o produto DIARIO so fecha as 12Z e e publicado com atraso -- pra hoje,
soma-se o que ja foi publicado nas fontes de menor latencia: MERGE HOURLY_NOW (dominio MENOR que
o produto diario, so nucleo bem instrumentado) + IMERG Early Run meia-hora (GPM_3IMERGHHE, taxa
mm/h * 0.5h = mm no intervalo). Ver `_dia_parcial_hoje`.

  - merge_precip_obs_daily(dt_ini, dt_fim)   -> chuva observada (mm/dia), serie diaria global;
    se `dt_fim` for HOJE, o ultimo passo vem do acumulado PARCIAL (horas/meias-horas ja publicadas)
  - clim_merge_precip_daily_for_anim(dates)  -> climatologia diaria do CPTEC (mm/dia), assinatura
    compativel com `_anom_from_clim` de globo_3d_anim.py (so cobre lon -85.05/-30.05,
    lat -56.15/12.85 -- fora dai a anomalia sai NaN/transparente, por nao ter climatologia).
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import xarray as xr

from app.common.forecast_download import GRIB_NETCDF_LOCK
from app.shared.logger import get_logger
from app.shared.settings_factory import settings

logger = get_logger(__name__)

CPTEC_MERGE_BASE = 'https://ftp.cptec.inpe.br/modelos/tempo/MERGE/GPM/DAILY'
CPTEC_MERGE_HOURLY_NOW_BASE = 'https://ftp.cptec.inpe.br/modelos/tempo/MERGE/GPM/HOURLY_NOW'
CPTEC_CLIM_URL = (
    'https://ftp.cptec.inpe.br/modelos/tempo/MERGE/GPM/CLIMATOLOGY/DAILY_AVERAGE/'
    'climatologia_366_dias_1998_2024.nc'
)
IMERG_SHORT_NAME = 'GPM_3IMERGDL'
IMERG_VERSION = '07'
# Early Run (menor latencia, ~4h) -- mesmo espirito do "-now" do MERGE: e o que existe pra HOJE.
IMERG_HALFHOURLY_SHORT_NAME = 'GPM_3IMERGHHE'
IMERG_HALFHOURLY_VERSION = '07'
# Nomes candidatos p/ a variavel de precipitacao em cada fonte (mesmo padrao de `var_candidates`
# usado no resto do projeto p/ tolerar pequenas diferencas de nomenclatura entre versoes/produtos).
_MERGE_VAR_CANDIDATES = ('prec', 'PREC', 'tp')
_IMERG_VAR_CANDIDATES = ('precipitation', 'precipitationCal', 'precip', 'HQprecipitation')
_CLIM_VAR_CANDIDATES = ('pmed', 'prec', 'precipitation', 'tp')

_TENTATIVAS = 5
_MIN_BYTES_GRIB = 50_000  # MERGE diario valido tem centenas de KB; abaixo disso e erro/HTML
_MIN_BYTES_GRIB_HOURLY = 10_000  # MERGE horario (dominio menor, 1h so) e bem menor que o diario
_MIN_BYTES_CLIM = 200_000_000  # climatologia consolidada tem ~1 GB


def _dados_dir() -> Path:
    d = Path(settings.DIR_DADOS) / 'MERGE_PRECIP'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _file_ok(path: Path, min_bytes: int) -> bool:
    try:
        return path.stat().st_size >= min_bytes
    except FileNotFoundError:
        return False


def _espera_backoff(tentativa: int) -> float:
    return min(2.0 ** (tentativa - 1), 30.0)


def _snap_equally_spaced(coord: np.ndarray) -> np.ndarray:
    """Reconstroi o eixo com espacamento EXATO (mediana dos deltas), a partir do primeiro valor.

    MERGE e IMERG vem de arquivos/fontes distintas; cada um carrega lat/lon com o proprio ruido de
    ponto flutuante (ex.: passo nominal 0.1 chega como 0.09999999999999432 num, 0.1 exato noutro).
    Sem isso, `xr.concat` entre dias com arrays "quase iguais mas nao identicos" UNE os dois
    conjuntos de coordenadas em vez de alinhar (vira um eixo gigante, irregular, com quase-
    duplicatas) e o `add_cyclic_point` do cartopy (fecha a costura do globo em 360/0) rejeita
    qualquer eixo que nao seja estritamente uniforme."""
    step = float(np.median(np.diff(coord)))
    return float(coord[0]) + step * np.arange(len(coord))


def _standardize(da: xr.DataArray) -> xr.DataArray:
    """lat/lon (nao latitude/longitude), lat ascendente, lon 0..360, espacamento exato, SEM
    coordenadas escalares sobrando (`time`/`step`/`valid_time`/`surface` do cfgrib) -- essas
    colidem quando varias horas/dias (cada uma com seu PROPRIO valor escalar de `time`) sao
    empilhadas com `xr.concat`/`expand_dims(time=...)`: o `time` novo (a chave real da serie)
    entra em conflito com o `time` escalar antigo (residuo do arquivo original), e dependendo do
    calendario do GRIB pode nem ser `numpy.datetime64` (ja vimos `cftime.DatetimeJulian` no
    MERGE HOURLY_NOW), quebrando com um erro obscuro de tipo bem depois, longe da causa real."""
    ren = {}
    for name in list(da.dims) + list(da.coords):
        low = str(name).lower()
        if low == 'latitude' and 'lat' not in da.dims:
            ren[name] = 'lat'
        elif low == 'longitude' and 'lon' not in da.dims:
            ren[name] = 'lon'
    if ren:
        da = da.rename(ren)
    da = da.reset_coords(drop=True)  # fora tudo que nao seja dimensao (time/step/valid_time/surface/...)
    da = da.assign_coords(lon=(da['lon'] % 360)).sortby('lon').sortby('lat')
    return da.assign_coords(
        lat=_snap_equally_spaced(da['lat'].values), lon=_snap_equally_spaced(da['lon'].values))


def _pick_var(ds: xr.Dataset, candidates: tuple[str, ...], contexto: str) -> str:
    for c in candidates:
        if c in ds.data_vars:
            return c
    raise RuntimeError(
        f'{contexto}: nenhuma variavel de precipitacao encontrada (candidatas: {candidates}, '
        f'disponiveis: {list(ds.data_vars)})'
    )


# ---------------------------------------------------------------------------------------
# MERGE regional (CPTEC, HTTP simples, sem autenticacao)
# ---------------------------------------------------------------------------------------
def _merge_url(day: datetime) -> str:
    return f'{CPTEC_MERGE_BASE}/{day:%Y}/{day:%m}/MERGE_CPTEC_{day:%Y%m%d}.grib2'


def _download_merge_day(day: datetime, force: bool) -> Path | None:
    """Baixa o GRIB2 diario do MERGE regional. 404 = dia ainda nao publicado pelo CPTEC
    (latencia) -> None, nao fatal (o mosaico cai so no IMERG naquele dia)."""
    out_dir = _dados_dir() / 'regional'
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'MERGE_CPTEC_{day:%Y%m%d}.grib2'
    if path.exists() and not force and _file_ok(path, _MIN_BYTES_GRIB):
        return path
    url = _merge_url(day)
    for attempt in range(1, _TENTATIVAS + 1):
        try:
            r = httpx.get(url, timeout=120, follow_redirects=True)
            if r.status_code == 404:
                logger.warning(
                    'MERGE regional {}: nao publicado pelo CPTEC (404) — dia usa so IMERG', day.date()
                )
                return None
            r.raise_for_status()
            path.write_bytes(r.content)
            if not _file_ok(path, _MIN_BYTES_GRIB):
                path.unlink(missing_ok=True)
                raise RuntimeError(f'MERGE regional {day.date()}: arquivo baixado invalido/pequeno')
            return path
        except httpx.HTTPStatusError:
            raise
        except Exception as exc:
            if attempt == _TENTATIVAS:
                raise RuntimeError(
                    f'Falha ao baixar MERGE regional {day.date()} apos {_TENTATIVAS} tentativas'
                ) from exc
            _s = _espera_backoff(attempt)
            logger.warning(
                'Tentativa {}/{} falhou no MERGE regional {} ({}) — nova tentativa em {:.0f}s',
                attempt,
                _TENTATIVAS,
                day.date(),
                exc,
                _s,
            )
            time.sleep(_s)
    return None


def _open_merge_day(path: Path, day: datetime) -> xr.DataArray:
    """Abre o GRIB2 do MERGE regional. Variavel `prec` (kg/m² = mm acumulado no dia, valido 12Z)."""
    with GRIB_NETCDF_LOCK:
        ds = xr.open_dataset(str(path), engine='cfgrib', backend_kwargs={'indexpath': ''})
        varname = _pick_var(ds, _MERGE_VAR_CANDIDATES, f'MERGE regional {path.name}')
        da = ds[varname].astype('float32').load()
        ds.close()
    da = _standardize(da)
    da = da.expand_dims(time=[np.datetime64(day.date())])
    da.name = 'merge_precip'
    return da


# ---------------------------------------------------------------------------------------
# MERGE HOURLY_NOW (CPTEC, dominio MENOR que o produto diario -- so pra somar o dia de HOJE)
# ---------------------------------------------------------------------------------------
def _download_merge_hour(dt_hour: datetime, force: bool) -> Path | None:
    """Baixa o GRIB2 de UMA hora do MERGE HOURLY_NOW. 404 = hora ainda nao publicada -> None
    (nao fatal -- soma-se so as horas que ja existem)."""
    out_dir = _dados_dir() / 'hourly_now'
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'MERGE_CPTEC_{dt_hour:%Y%m%d%H}.grib2'
    if path.exists() and not force and _file_ok(path, _MIN_BYTES_GRIB_HOURLY):
        return path
    url = (f'{CPTEC_MERGE_HOURLY_NOW_BASE}/{dt_hour:%Y}/{dt_hour:%m}/{dt_hour:%d}/'
           f'MERGE_CPTEC_{dt_hour:%Y%m%d%H}.grib2')
    for attempt in range(1, _TENTATIVAS + 1):
        try:
            r = httpx.get(url, timeout=60, follow_redirects=True)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            path.write_bytes(r.content)
            if not _file_ok(path, _MIN_BYTES_GRIB_HOURLY):
                path.unlink(missing_ok=True)
                raise RuntimeError(f'MERGE horario {dt_hour:%Y-%m-%d %HZ}: arquivo invalido/pequeno')
            return path
        except httpx.HTTPStatusError:
            raise
        except Exception as exc:
            if attempt == _TENTATIVAS:
                logger.warning(
                    'MERGE horario {:%Y-%m-%d %HZ}: falha apos {} tentativas ({}) — hora pulada',
                    dt_hour, _TENTATIVAS, exc,
                )
                return None
            time.sleep(_espera_backoff(attempt))
    return None


def _open_merge_hour(path: Path) -> xr.DataArray:
    """Abre o GRIB2 horario. `prec` em mm/h com granularidade de 1h == mm daquela hora (numericamente
    equivalente; ver GRIB_units='kg m**-2' vs. o .ctl dizendo 'mm/hr' -- mesma coisa p/ dt=1h)."""
    with GRIB_NETCDF_LOCK:
        ds = xr.open_dataset(str(path), engine='cfgrib', backend_kwargs={'indexpath': ''})
        varname = _pick_var(ds, _MERGE_VAR_CANDIDATES, f'MERGE horario {path.name}')
        da = ds[varname].astype('float32').load()
        ds.close()
    return _standardize(da)


def _merge_hourly_now_sum(
    dt_ini_hour: datetime, dt_fim_hour: datetime,
) -> tuple[xr.DataArray | None, datetime | None]:
    """Soma as horas MERGE HOURLY_NOW ja publicadas em [dt_ini_hour, dt_fim_hour].

    Retorna (total, ultima_hora) -- `ultima_hora` e a hora mais recente efetivamente somada (None
    se nenhuma), pra quem chamar logar ate onde a chuva de hoje ja foi contabilizada."""
    force = bool(getattr(settings, 'FORCE_DOWNLOAD', False))
    horas = pd.date_range(dt_ini_hour.replace(minute=0, second=0, microsecond=0), dt_fim_hour, freq='1h')
    partes = []
    ultima_hora: datetime | None = None
    for h in horas:
        hpy = h.to_pydatetime()
        path = _download_merge_hour(hpy, force)
        if path is not None:
            partes.append(_open_merge_hour(path))
            ultima_hora = hpy
    if not partes:
        return None, None
    # join='override': mesma razao do concat diario (`merge_precip_obs_daily`) -- cada hora vem de
    # um arquivo proprio, o ruido de ponto flutuante entre eles nao pode virar uniao de eixos.
    empilhado = xr.concat(partes, dim='hora', join='override', coords='minimal', compat='override')
    total = empilhado.sum(dim='hora', skipna=True)
    total = total.expand_dims(time=[np.datetime64(dt_ini_hour.date())])
    total.name = 'merge_precip_hourly_sum'
    return total, ultima_hora


# ---------------------------------------------------------------------------------------
# IMERG-GPM Late Run (NASA GES DISC, via earthaccess -- autenticado por EARTHDATA_TOKEN)
# ---------------------------------------------------------------------------------------
def _earthdata_login() -> None:
    import earthaccess

    token = str(
        getattr(settings, 'EARTHDATA_TOKEN', '') or os.environ.get('EARTHDATA_TOKEN', '')
    ).strip()
    if not token:
        raise RuntimeError(
            'EARTHDATA_TOKEN nao configurado. Preencha em app/settings/.secrets.toml '
            '(gere o token em urs.earthdata.nasa.gov -> Applications -> Generate Token).'
        )
    os.environ['EARTHDATA_TOKEN'] = token
    auth = earthaccess.login(strategy='environment')
    if not getattr(auth, 'authenticated', False):
        raise RuntimeError('Falha ao autenticar no NASA Earthdata com o EARTHDATA_TOKEN configurado.')


def _download_imerg_day(day: datetime, force: bool) -> Path | None:
    """Busca e baixa o granulo diario IMERG Late Run (GPM_3IMERGDL.07) via earthaccess (CMR)."""
    import earthaccess

    out_dir = _dados_dir() / 'imerg_late'
    out_dir.mkdir(parents=True, exist_ok=True)
    existentes = list(out_dir.glob(f'*{day:%Y%m%d}*.nc4')) + list(out_dir.glob(f'*{day:%Y%m%d}*.nc'))
    if existentes and not force and _file_ok(existentes[0], 10_000):
        return existentes[0]

    _earthdata_login()
    dt_seguinte = day + timedelta(days=1)
    granulos = earthaccess.search_data(
        short_name=IMERG_SHORT_NAME,
        version=IMERG_VERSION,
        temporal=(day.strftime('%Y-%m-%d'), dt_seguinte.strftime('%Y-%m-%d')),
    )
    if not granulos:
        logger.warning(
            'IMERG Late {}: nenhum granulo encontrado no CMR — dia usa so MERGE (se houver)', day.date()
        )
        return None
    baixados = earthaccess.download(granulos[:1], local_path=str(out_dir))
    if not baixados:
        return None
    path = Path(baixados[0])
    if not _file_ok(path, 10_000):
        raise RuntimeError(f'IMERG Late {day.date()}: arquivo baixado invalido/pequeno')
    return path


def _open_imerg_day(path: Path, day: datetime) -> xr.DataArray:
    ds = xr.open_dataset(str(path))
    varname = _pick_var(ds, _IMERG_VAR_CANDIDATES, f'IMERG {path.name}')
    da = ds[varname].astype('float32').load()
    ds.close()
    # IMERG grava (lon, lat) em alguns produtos -- garante (lat, lon) apos padronizar.
    da = da.squeeze(drop=True)
    da = _standardize(da)
    if 'time' not in da.dims:
        da = da.expand_dims(time=[np.datetime64(day.date())])
    da.name = 'imerg_precip'
    return da


def _imerg_halfhourly_sum(
    dt_ini: datetime, dt_fim: datetime,
) -> tuple[xr.DataArray | None, datetime | None]:
    """Soma os granulos IMERG Early meia-hora ja publicados em [dt_ini, dt_fim].

    Cada granulo vem em grupo HDF5 'Grid' (diferente do .nc4 diario, que e plano) e a variavel
    `precipitation` e uma TAXA em mm/h -- multiplica por 0.5h pra virar mm acumulado no intervalo.
    Retorna (total, ultimo_inicio_meia_hora) -- None se nenhum granulo publicado ainda."""
    import earthaccess

    _earthdata_login()
    granulos = earthaccess.search_data(
        short_name=IMERG_HALFHOURLY_SHORT_NAME,
        version=IMERG_HALFHOURLY_VERSION,
        temporal=(dt_ini.strftime('%Y-%m-%dT%H:%M:%S'), dt_fim.strftime('%Y-%m-%dT%H:%M:%S')),
    )
    if not granulos:
        return None, None
    out_dir = _dados_dir() / 'imerg_early_hh'
    out_dir.mkdir(parents=True, exist_ok=True)
    baixados = earthaccess.download(granulos, local_path=str(out_dir))
    partes = []
    ultima_meia_hora: datetime | None = None
    for p in baixados:
        path = Path(p)
        if not _file_ok(path, 10_000):
            continue
        ds = xr.open_dataset(str(path), group='Grid')
        varname = _pick_var(ds, _IMERG_VAR_CANDIDATES, f'IMERG meia-hora {path.name}')
        # `time` do IMERG vem as vezes em calendario NAO-padrao (cftime.DatetimeJulian, nao
        # numpy.datetime64) -- pd.Timestamp() rejeita cftime direto; extrai os campos na mao.
        _t_raw = ds['time'].values[0]
        if isinstance(_t_raw, np.datetime64):
            _t = pd.Timestamp(_t_raw).to_pydatetime()
        else:
            _t = datetime(_t_raw.year, _t_raw.month, _t_raw.day, _t_raw.hour, _t_raw.minute, _t_raw.second)
        if ultima_meia_hora is None or _t > ultima_meia_hora:
            ultima_meia_hora = _t
        da = (ds[varname].astype('float32') * 0.5).squeeze(drop=True)  # mm/h * 0.5h = mm no intervalo
        ds.close()
        partes.append(_standardize(da))
    if not partes:
        return None, None
    empilhado = xr.concat(
        partes, dim='meia_hora', join='override', coords='minimal', compat='override')
    total = empilhado.sum(dim='meia_hora', skipna=True)
    total = total.expand_dims(time=[np.datetime64(dt_ini.date())])
    total.name = 'imerg_precip_halfhourly_sum'
    return total, ultima_meia_hora


# ---------------------------------------------------------------------------------------
# Mosaico + serie diaria
# ---------------------------------------------------------------------------------------
# Deteccao de pixel ISOLADO com valor muito acima da vizinhanca -- assinatura de erro pontual no
# dado bruto (satelite/estacao), nao de chuva real (que sempre tem area coerente ao redor). Achado
# num caso real: 675 mm num UNICO pixel/UNICO dia no litoral de SP, vizinhanca inteira em ~0 mm --
# nada a ver com dominio/mosaico (o ponto fica bem dentro da area do MERGE, longe da borda).
_OUTLIER_MIN_MM = 150.0            # so avalia pixels acima disso (chuva fraca/moderada nunca e "suspeita")
_OUTLIER_FATOR_VIZINHANCA = 4.0    # pixel precisa ser > 4x a mediana da vizinhanca p/ ser suspeito
_OUTLIER_JANELA = 5                # janela da mediana local (5x5 celulas ~ 0.5°, ~55 km)


def _filtra_pixels_isolados(da: xr.DataArray) -> xr.DataArray:
    """Substitui pela MEDIANA LOCAL (janela 5x5, ~0.5°) qualquer pixel > `_OUTLIER_MIN_MM` que seja
    > `_OUTLIER_FATOR_VIZINHANCA` vezes maior que essa mediana -- chuva real sempre tem area coerente
    ao redor (convectiva ou orografica cobre varios pixels vizinhos); um pico isolado cercado de
    quase-zero e erro pontual do dado bruto, nao meteorologia. Loga cada correcao (ate 10 por dia).

    `da` e sempre 1 UNICO passo de tempo aqui (1 dia, ou o parcial de hoje) -- trata a janela 2D
    (lat, lon) explicitamente (squeeze/expand do eixo `time`) em vez de deixar o `median_filter`
    correr nos 3 eixos, o que degeneraria (mas so por acidente, via padding) no eixo time=1."""
    from scipy.ndimage import median_filter

    has_time = 'time' in da.dims
    campo = da.isel(time=0) if has_time else da
    vals = campo.values
    mediana_local = median_filter(np.nan_to_num(vals, nan=0.0), size=_OUTLIER_JANELA, mode='nearest')
    suspeitos = (vals > _OUTLIER_MIN_MM) & (vals > _OUTLIER_FATOR_VIZINHANCA * np.maximum(mediana_local, 1e-6))
    n = int(suspeitos.sum())
    if n == 0:
        return da
    dia_str = pd.Timestamp(da['time'].values[0]).date() if has_time else '?'
    lat_idx, lon_idx = np.where(suspeitos)
    for i, j in list(zip(lat_idx, lon_idx))[:10]:
        _lon_disp = float(campo['lon'].values[j])
        _lon_disp = _lon_disp - 360 if _lon_disp > 180 else _lon_disp
        logger.warning(
            'Chuva {}: pixel isolado suspeito em lat={:.2f} lon={:.2f} — {:.1f}mm (vizinhanca '
            'mediana {:.1f}mm) — substituido pela mediana local',
            dia_str, float(campo['lat'].values[i]), _lon_disp, vals[i, j], mediana_local[i, j],
        )
    if n > 10:
        logger.warning('Chuva {}: +{} outro(s) pixel(s) isolado(s) suspeito(s) (nao listados)', dia_str, n - 10)
    vals_corrigidos = np.where(suspeitos, mediana_local, vals)
    novo_campo = campo.copy(data=vals_corrigidos)
    novo = novo_campo.expand_dims(time=da['time']) if has_time else novo_campo
    novo.name = da.name
    novo.attrs.update(da.attrs)
    return novo


# Grade global CANONICA (mesma convencao/fase do IMERG: 0.1°, centros em x.x5°, lat -89.95..89.95,
# lon 0.05..359.95, 1800x3600). Fixa e independente do dia -- necessaria pro fallback de
# `_mosaic_day` (dia sem IMERG, so MERGE): sem regridar pra essa grade, o dia sairia no tamanho
# REGIONAL do MERGE (~924x1001), diferente dos outros dias (globais, 1800x3600), e o
# `xr.concat(..., join='override')` de `merge_precip_obs_daily` quebra ("dimensions... don't have
# the same size") ao tentar empilhar dias com shapes diferentes.
_GRID_LAT_GLOBAL = np.round(np.arange(-89.95, 90.0, 0.1), 6)
_GRID_LON_GLOBAL = np.round(np.arange(0.05, 360.0, 0.1), 6)


def _para_grade_global(da: xr.DataArray) -> xr.DataArray:
    """Interpola pra grade global canonica (0.1°, mesma fase do IMERG). Fora do dominio original
    (ex.: MERGE regional fora da America do Sul) sai NaN -- esperado, o mosaico normal ja lida com
    NaN fora do dominio de cada fonte."""
    out = da.interp(lat=_GRID_LAT_GLOBAL, lon=_GRID_LON_GLOBAL, method='linear')
    out.name = da.name
    out.attrs.update(da.attrs)
    return out


def _mosaic_day(merge_da: xr.DataArray | None, imerg_da: xr.DataArray | None) -> xr.DataArray:
    """MERGE tem prioridade onde existe (America do Sul); IMERG preenche o resto do globo.

    Usa `.interp` (nao `reindex_like`) de propósito: MERGE e IMERG vem de arquivos distintos, entao
    seus eixos lat/lon — mesmo com o MESMO passo nominal 0.1° — nunca batem por igualdade EXATA de
    ponto flutuante. `reindex_like` casa por RÓTULO exato e, sem tolerância, um desalinhamento de
    1e-10 já preenche tudo com NaN (falha silenciosa: o composto pareceria 100% IMERG, sem erro
    nenhum). `.interp` casa pelo VALOR numérico (via scipy), robusto a essa diferença mínima.

    Quando falta uma das duas fontes, o resultado passa por `_para_grade_global` -- senao o dia sai
    na grade NATIVA da fonte que sobrou (regional se so MERGE, ou o tamanho daquele IMERG especifico
    se so IMERG), o que quebra o concat entre dias em `merge_precip_obs_daily` (ver docstring acima).

    Filtra pixels isolados suspeitos (`_filtra_pixels_isolados`) no resultado final -- roda uma vez
    so, ja no composto (nao em cada fonte separada), cobrindo os dois casos (merge_da/imerg_da None
    ou ambos presentes) com um unico ponto de chamada."""
    if merge_da is None:
        return _filtra_pixels_isolados(_para_grade_global(imerg_da))
    if imerg_da is None:
        return _filtra_pixels_isolados(_para_grade_global(merge_da))
    merge_on_imerg_grid = merge_da.interp(
        lat=imerg_da['lat'], lon=imerg_da['lon'], method='linear').assign_coords(time=imerg_da['time'])
    composite = merge_on_imerg_grid.combine_first(imerg_da)
    composite.name = 'merge_precip_abs'
    return _filtra_pixels_isolados(composite)


def _dia_parcial_hoje() -> xr.DataArray | None:
    """Chuva de HOJE somando o que ja foi publicado: MERGE HOURLY_NOW + IMERG Early meia-hora.
    None se nenhuma das duas fontes tiver nada publicado ainda (dia pulado por quem chamar).

    Loga no terminal ate que horario (UTC) cada fonte contabilizou -- a caixa de data do video NAO
    mostra isso (deixava a caixa grande demais), entao esse log e o unico lugar que informa."""
    hoje = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    agora = datetime.now()
    merge_total, merge_ultima = _merge_hourly_now_sum(hoje, agora)
    imerg_total, imerg_ultima = _imerg_halfhourly_sum(hoje, agora)
    if merge_total is None and imerg_total is None:
        return None
    logger.info(
        'Chuva PARCIAL de hoje ({}): MERGE HOURLY_NOW ate {} | IMERG Early ate {}',
        hoje.date(),
        f'{merge_ultima:%H}Z' if merge_ultima else 'sem dado',
        f'{imerg_ultima:%H:%M}Z' if imerg_ultima else 'sem dado',
    )
    return _mosaic_day(merge_total, imerg_total)


def merge_precip_obs_daily(dt_ini: datetime, dt_fim: datetime) -> xr.DataArray:
    """Chuva observada (mm/dia) no periodo [dt_ini, dt_fim]: MERGE (America do Sul, bias-corrected)
    + IMERG Late (resto do globo, satelite puro). lat ascendente, lon 0..360.

    Se `dt_fim` incluir HOJE, o ultimo passo da serie vem do acumulado PARCIAL (soma das horas/
    meias-horas ja publicadas ate agora) em vez do produto diario -- que so fecha as 12Z e ainda
    nao existe pra hoje. Ver `_dia_parcial_hoje`."""
    force = bool(getattr(settings, 'FORCE_DOWNLOAD', False))
    coarsen_f = max(1, int(settings.get('GLOBO_3D_MERGE_COARSEN', 1)))
    hoje = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    dias = pd.date_range(dt_ini.date(), dt_fim.date(), freq='1D')

    partes: list[xr.DataArray] = []
    for d in dias:
        day = d.to_pydatetime()
        if day.date() == hoje.date():
            parcial = _dia_parcial_hoje()
            if parcial is None:
                logger.warning('Chuva observada hoje ({}): nada publicado ainda — dia pulado', day.date())
                continue
            partes.append(parcial)
            continue
        merge_path = _download_merge_day(day, force)
        imerg_path = _download_imerg_day(day, force)
        merge_da = _open_merge_day(merge_path, day) if merge_path is not None else None
        imerg_da = _open_imerg_day(imerg_path, day) if imerg_path is not None else None
        if merge_da is None and imerg_da is None:
            logger.warning('Chuva observada {}: sem MERGE e sem IMERG — dia pulado', day.date())
            continue
        partes.append(_mosaic_day(merge_da, imerg_da))

    if not partes:
        raise RuntimeError(
            f'Sem dados de chuva observada (MERGE/IMERG) na janela {dt_ini.date()} a {dt_fim.date()}.'
        )

    # join='override'/compat='override': cada dia ja foi interpolado pro MESMO grid nominal
    # (`_mosaic_day`), mas o valor bruto de lat/lon pode variar por ruido de ponto flutuante entre
    # arquivos IMERG de dias diferentes -- sem 'override', o concat ALINHARIA por igualdade exata
    # e uniria os eixos quase-iguais num eixo gigante/irregular (mesma causa do bug do
    # add_cyclic_point). 'override' usa o lat/lon do PRIMEIRO dia pra todos, o que e seguro aqui
    # porque todos vem do mesmo produto/resolucao.
    daily = xr.concat(partes, dim='time', join='override', coords='minimal', compat='override')
    daily = daily.sortby('time')
    if coarsen_f > 1:
        daily = daily.coarsen(lat=coarsen_f, lon=coarsen_f, boundary='trim').mean(skipna=True)
    daily.name = 'merge_precip_abs'
    daily.attrs['units'] = 'mm'
    logger.info(
        'Chuva observada MERGE+IMERG: {} dias | min={:.1f} max={:.1f} mm',
        daily.sizes['time'],
        float(daily.min()),
        float(daily.max()),
    )
    return daily


# ---------------------------------------------------------------------------------------
# Climatologia diaria do CPTEC (so p/ anomalia)
# ---------------------------------------------------------------------------------------
def _clim_path() -> Path:
    d = _dados_dir() / 'climatologia'
    d.mkdir(parents=True, exist_ok=True)
    return d / 'climatologia_366_dias_1998_2024.nc'


def _download_clim_precip(force: bool = False) -> Path:
    path = _clim_path()
    if path.exists() and not force and _file_ok(path, _MIN_BYTES_CLIM):
        return path
    logger.info('Baixando climatologia diaria de chuva do CPTEC (~1 GB, unica vez)...')
    with httpx.stream('GET', CPTEC_CLIM_URL, timeout=600, follow_redirects=True) as r:
        r.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                f.write(chunk)
    if not _file_ok(path, _MIN_BYTES_CLIM):
        path.unlink(missing_ok=True)
        raise RuntimeError('Climatologia de chuva do CPTEC: download invalido/incompleto.')
    return path


def clim_merge_precip_daily_for_anim(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Climatologia diaria de chuva (CPTEC, 1998-2024) na grade nativa do MERGE (~0.1°, so
    America do Sul: lon -85.05/-30.05, lat -56.15/12.85). Assinatura compativel com
    `_anom_from_clim` de globo_3d_anim.py -> (arr, lat, lon)."""
    clim_path = _download_clim_precip()
    with xr.open_dataset(str(clim_path)) as clim:
        varname = _pick_var(clim, _CLIM_VAR_CANDIDATES, 'Climatologia de chuva CPTEC')
        clim_var = clim[varname]
        tvar = clim_var['time'] if 'time' in clim_var.coords else clim['time']
        tdates = pd.to_datetime(tvar.values)
        idx_by_md = {(int(d.month), int(d.day)): i for i, d in enumerate(tdates)}
        slices = []
        for dt64 in dates:
            d = pd.Timestamp(dt64)
            md = (int(d.month), int(d.day))
            if md == (2, 29) and md not in idx_by_md:
                md = (2, 28)  # climatologia so tem 365/366 dias-do-ano
            slices.append(clim_var.isel(time=idx_by_md[md]))
        clim_da = xr.concat(slices, dim='t').astype('float32').load()
    clim_da = _standardize(clim_da)
    return clim_da.values, clim_da['lat'].values, clim_da['lon'].values
