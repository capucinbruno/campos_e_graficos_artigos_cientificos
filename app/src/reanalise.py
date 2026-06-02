import logging
from datetime import date, timedelta

import numpy as np
import xarray as xr

from app.common.ampere_api_client import AmpereAPIClient
from app.shared.logger import get_logger
from app.shared.settings_factory import settings


def _validar_referencia(ano_referencia_inicio: int, ano_referencia_fim: int) -> None:
    if ano_referencia_inicio > ano_referencia_fim:
        raise ValueError(
            f"ano_referencia_inicio ({ano_referencia_inicio}) é posterior "
            f"a ano_referencia_fim ({ano_referencia_fim})"
        )


def _validar_filtros(
    data_inicial: str,
    data_final: str,
    filtro_anos: list[int],
    filtro_meses: list[int],
    filtro_dias: list[int],
    filtro_horas: list[int],
) -> None:
    dt_ini = date.fromisoformat(data_inicial)
    dt_fim = date.fromisoformat(data_final)

    if dt_ini > dt_fim:
        raise ValueError(f"data_inicial ({data_inicial}) é posterior a data_final ({data_final})")

    if filtro_anos:
        anos_validos = set(range(dt_ini.year, dt_fim.year + 1))
        invalidos = sorted(set(filtro_anos) - anos_validos)
        if invalidos:
            raise ValueError(f"filtro_anos contém anos fora do período {dt_ini.year}–{dt_fim.year}: {invalidos}")

    if filtro_meses:
        meses_validos: set[int] = set()
        ano, mes = dt_ini.year, dt_ini.month
        while (ano, mes) <= (dt_fim.year, dt_fim.month):
            meses_validos.add(mes)
            mes = mes % 12 + 1
            if mes == 1:
                ano += 1
        invalidos = sorted(set(filtro_meses) - meses_validos)
        if invalidos:
            raise ValueError(f"filtro_meses contém meses fora do período {data_inicial} → {data_final}: {invalidos}")

    if filtro_dias:
        dias_validos: set[int] = set()
        d = dt_ini
        while d <= dt_fim:
            dias_validos.add(d.day)
            if len(dias_validos) == 31:
                break
            d += timedelta(days=1)
        invalidos = sorted(set(filtro_dias) - dias_validos)
        if invalidos:
            raise ValueError(f"filtro_dias contém dias fora do período {data_inicial} → {data_final}: {invalidos}")

    # Horas: apenas 0, 6, 12 ou 18
    horas_validas = {0, 6, 12, 18}
    horas_invalidas = sorted(h for h in filtro_horas if h not in horas_validas)
    if horas_invalidas:
        raise ValueError(f"filtro_horas contém valores inválidos (esperado 0, 6, 12 ou 18): {horas_invalidas}")


class ReanaliseAPI:
    """
    Interface pública para a API de reanálise da Ampere Consultoria.

    Uso:
        with ReanaliseAPI() as api:
            datasets   = api.listar_datasets()
            variaveis  = api.listar_variaveis()
            niveis     = api.listar_niveis_pressao()
            tipos      = api.listar_tipos_agregacao()

            ds = api.agregacao_temporal(
                nm_dataset='reanalysis-era5-pressure-levels',
                nm_variavel='z',
                pressure_level_hpa=250,
                data_inicial='2020-01-01',
                data_final='2020-01-31',
                tipo_agregacao='media',
                filtro_anos=[2020],
                filtro_dias=[15],
                filtro_meses=[1],
                filtro_horas=[0, 12],
            )  # xr.Dataset
    """

    def __init__(self, logger: logging.Logger | None = None):
        self._logger = logger or get_logger('reanalise-api')
        self._client: AmpereAPIClient | None = None
        self._product_key: str | None = None

    def __enter__(self):
        self._client = AmpereAPIClient(
            settings.AMPERE_API_USERNAME,
            settings.AMPERE_API_PASSWORD,
            self._logger,
        ).__enter__()
        self._client.login()
        permission = self._client.check_permission(item='meteorologia')
        data = permission['data']
        self._product_key = (
            data if isinstance(data, str)
            else (data.get('product_key') or data.get('productKey') or data.get('key'))
        )
        if not self._product_key:
            raise ValueError('product_key não encontrado na resposta de permissão')
        return self

    def __exit__(self, *args):
        if self._client:
            self._client.__exit__(*args)

    # ── Discovery ────────────────────────────────────────────────────────────

    def listar_datasets(self) -> list[dict]:
        """Retorna os datasets de reanálise disponíveis."""
        return self._client.listar_datasets_reanalise(self._product_key)['data']['data']

    def listar_variaveis(self) -> list[dict]:
        """Retorna as variáveis disponíveis com unidade e nível de pressão."""
        return self._client.listar_variaveis_reanalise(self._product_key)['data']['data']

    def listar_niveis_pressao(self) -> list[dict]:
        """Retorna os níveis de pressão disponíveis em hPa."""
        return self._client.listar_niveis_pressao_reanalise(self._product_key)['data']['data']

    def listar_tipos_agregacao(self) -> list[dict]:
        """Retorna os tipos de agregação disponíveis com descrição."""
        return self._client.listar_tipos_agregacao_reanalise(self._product_key)['data']['data']

    # ── Cálculos ─────────────────────────────────────────────────────────────

    def agregacao_temporal(
        self,
        nm_dataset: str,
        nm_variavel: str,
        data_inicial: str,
        data_final: str,
        tipo_agregacao: str,
        filtro_anos: list[int],
        filtro_meses: list[int],
        filtro_dias: list[int],
        filtro_horas: list[int],
        pressure_level_hpa: int | None = None,
    ) -> xr.Dataset:
        """
        Executa a agregação temporal e retorna um xr.Dataset (lat x lon).

        Args:
            nm_dataset: Nome do dataset (ex: 'reanalysis-era5-pressure-levels')
            nm_variavel: Nome da variável (ex: 'z')
            data_inicial: Data inicial no formato 'YYYY-MM-DD'
            data_final: Data final no formato 'YYYY-MM-DD'
            tipo_agregacao: Tipo de agregação (ex: 'media', 'soma')
            filtro_anos: Lista de anos a incluir (ex: [2020])
            filtro_meses: Lista de meses a incluir (ex: [1])
            filtro_dias: Lista de dias do mês a incluir (ex: [15])
            filtro_horas: Lista de horas UTC a incluir (ex: [0, 12])
            pressure_level_hpa: Nível de pressão em hPa (apenas para pressure-levels)
        """
        _validar_filtros(data_inicial, data_final, filtro_anos, filtro_meses, filtro_dias, filtro_horas)

        resp = self._client.agregacao_temporal_reanalise(
            product_key=self._product_key,
            nm_dataset=nm_dataset,
            nm_variavel=nm_variavel,
            data_inicial=data_inicial,
            data_final=data_final,
            tipo_agregacao=tipo_agregacao,
            filtro_anos=filtro_anos,
            filtro_dias=filtro_dias,
            filtro_meses=filtro_meses,
            filtro_horas=filtro_horas,
            pressure_level_hpa=pressure_level_hpa,
        )

        inner = resp['data']['data']
        dados = inner['dados']
        nm_var = inner['nm_variavel']

        return xr.Dataset(
            {nm_var: xr.DataArray(
                data=np.array(dados['data']),
                dims=['lat', 'lon'],
                coords={
                    'lat': dados['coords']['lat'],
                    'lon': dados['coords']['lon'],
                },
                attrs=dados['attrs'],
            )}
        )

    def anomalia(
        self,
        nm_dataset: str,
        nm_variavel: str,
        data_inicial: str,
        data_final: str,
        ano_referencia_inicio: int,
        ano_referencia_fim: int,
        tipo_agregacao: str,
        pressure_level_hpa: int | None = None,
        filtro_anos: list[int] | None = None,
        filtro_anos_referencia: list[int] | None = None,
        filtro_meses: list[int] | None = None,
        filtro_dias: list[int] | None = None,
        filtro_horas: list[int] | None = None,
    ) -> xr.Dataset:
        """
        Calcula a anomalia em relação a um período de referência e retorna xr.Dataset (lat x lon).

        Args:
            nm_dataset: Nome do dataset
            nm_variavel: Código da variável
            data_inicial: Início do período de análise (YYYY-MM-DD)
            data_final: Fim do período de análise (YYYY-MM-DD)
            ano_referencia_inicio: Ano inicial da climatologia de referência
            ano_referencia_fim: Ano final da climatologia de referência
            tipo_agregacao: 'media' ou 'soma'
            pressure_level_hpa: Nível de pressão em hPa (apenas pressure-levels)
            filtro_anos: Anos do período de análise a incluir (None = todos)
            filtro_anos_referencia: Anos da referência a incluir (None = todos)
            filtro_meses: Meses a incluir (None = todos)
            filtro_dias: Dias a incluir (None = todos)
            filtro_horas: Horas UTC a incluir — 0, 6, 12 ou 18 (None = todas)
        """
        _validar_referencia(ano_referencia_inicio, ano_referencia_fim)

        dt_ini = date.fromisoformat(data_inicial)
        dt_fim = date.fromisoformat(data_final)

        if filtro_anos:
            anos_validos = set(range(dt_ini.year, dt_fim.year + 1))
            invalidos = sorted(set(filtro_anos) - anos_validos)
            if invalidos:
                raise ValueError(f"filtro_anos contém anos fora do período {dt_ini.year}–{dt_fim.year}: {invalidos}")

        if filtro_anos_referencia:
            invalidos = sorted(a for a in filtro_anos_referencia if not (ano_referencia_inicio <= a <= ano_referencia_fim))
            if invalidos:
                raise ValueError(
                    f"filtro_anos_referencia contém anos fora do período de referência "
                    f"{ano_referencia_inicio}–{ano_referencia_fim}: {invalidos}"
                )

        if filtro_meses:
            meses_validos: set[int] = set()
            ano, mes = dt_ini.year, dt_ini.month
            while (ano, mes) <= (dt_fim.year, dt_fim.month):
                meses_validos.add(mes)
                mes = mes % 12 + 1
                if mes == 1:
                    ano += 1
            invalidos = sorted(set(filtro_meses) - meses_validos)
            if invalidos:
                raise ValueError(f"filtro_meses contém meses fora do período {data_inicial} → {data_final}: {invalidos}")

        if filtro_dias:
            dias_validos: set[int] = set()
            d = dt_ini
            while d <= dt_fim:
                dias_validos.add(d.day)
                if len(dias_validos) == 31:
                    break
                d += timedelta(days=1)
            invalidos = sorted(set(filtro_dias) - dias_validos)
            if invalidos:
                raise ValueError(f"filtro_dias contém dias fora do período {data_inicial} → {data_final}: {invalidos}")

        if filtro_horas:
            invalidos = sorted(h for h in filtro_horas if h not in {0, 6, 12, 18})
            if invalidos:
                raise ValueError(f"filtro_horas contém valores inválidos (esperado 0, 6, 12 ou 18): {invalidos}")

        resp = self._client.anomalia_reanalise(
            product_key=self._product_key,
            nm_dataset=nm_dataset,
            nm_variavel=nm_variavel,
            data_inicial=data_inicial,
            data_final=data_final,
            ano_referencia_inicio=ano_referencia_inicio,
            ano_referencia_fim=ano_referencia_fim,
            tipo_agregacao=tipo_agregacao,
            pressure_level_hpa=pressure_level_hpa,
            filtro_anos=filtro_anos,
            filtro_anos_referencia=filtro_anos_referencia,
            filtro_meses=filtro_meses,
            filtro_dias=filtro_dias,
            filtro_horas=filtro_horas,
        )

        inner = resp['data']['data']
        dados = inner['dados']
        nm_var = inner['nm_variavel']

        return xr.Dataset(
            {nm_var: xr.DataArray(
                data=np.array(dados['data']),
                dims=['lat', 'lon'],
                coords={
                    'lat': dados['coords']['lat'],
                    'lon': dados['coords']['lon'],
                },
                attrs=dados['attrs'],
            )}
        )
