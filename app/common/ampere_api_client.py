import logging
from typing import Any

import httpx

from app.shared.settings_factory import settings



class AmpereAPIClient:
    """Cliente para interagir com a API da Ampere Consultoria."""

    def __init__(
        self,
        username: str,
        password: str,
        logger: logging.Logger,
        user_token: str | None = None,
    ):
        self.username = username
        self.password = password
        self.client = httpx.Client(base_url=settings.AMPERE_API_BASE_URL, timeout=30.0)
        self.access_token = None
        self.user_token = user_token
        self.logger = logger

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()

    def login(self) -> dict[str, Any]:
        """
        Realiza login na API e obtém os tokens de autenticação.

        Returns:
            Resposta do servidor com os dados de autenticação
        """
        self.logger.info('Realizando login...')

        response = self.client.put(
            '/login',
            headers={'Content-Type': 'application/json'},
            json={'username': self.username, 'password': self.password},
        )
        response.raise_for_status()

        data = response.json()

        if 'access_token' in data['data'] or 'accessToken' in data['data']:
            self.access_token = data['data'].get('access_token') or data['data'].get('accessToken')

        self.logger.info('Login realizado com sucesso!')
        self.logger.debug('Access Token: %s', self.access_token)

        return data

    def check_permission(self, item: str = 'meteorologia') -> dict[str, Any]:
        """
        Verifica se o usuário atual tem permissão para um item específico.

        Args:
            item: Nome do item para verificar permissão (padrão: "meteorologia")

        Returns:
            Resposta do servidor com os dados de permissão
        """
        self.logger.info("Verificando permissão para '%s'...", item)

        headers = {'Content-Type': 'application/json'}

        if self.access_token:
            headers['x-access-token'] = self.access_token
        if self.user_token:
            headers['x-user-token'] = self.user_token

        response = self.client.get(
            '/admin/contratos/current-user-has-permission/', headers=headers, params={'item': item}
        )
        response.raise_for_status()

        data = response.json()
        self.logger.info('Permissão verificada com sucesso')

        return data

    def listar_datasets_reanalise(self, product_key: str) -> dict[str, Any]:
        """
        Lista os datasets disponíveis para reanálise.

        Args:
            product_key: Chave do produto

        Returns:
            Resposta do servidor com a lista de datasets
        """
        self.logger.info('Listando datasets de reanálise...')
        data = self._get_reanalise('/produtos/reanalise/listar-datasets', product_key)
        self.logger.info('Datasets listados com sucesso!')
        return data

    def listar_variaveis_reanalise(self, product_key: str) -> dict[str, Any]:
        self.logger.info('Listando variáveis de reanálise...')
        data = self._get_reanalise('/produtos/reanalise/listar-variaveis', product_key)
        self.logger.info('Variáveis listadas com sucesso!')
        return data

    def listar_niveis_pressao_reanalise(self, product_key: str) -> dict[str, Any]:
        self.logger.info('Listando níveis de pressão de reanálise...')
        data = self._get_reanalise('/produtos/reanalise/listar-niveis-pressao', product_key)
        self.logger.info('Níveis de pressão listados com sucesso!')
        return data

    def listar_tipos_agregacao_reanalise(self, product_key: str) -> dict[str, Any]:
        self.logger.info('Listando tipos de agregação de reanálise...')
        data = self._get_reanalise('/produtos/reanalise/listar-tipos-agregacao', product_key)
        self.logger.info('Tipos de agregação listados com sucesso!')
        return data

    def agregacao_temporal_reanalise(
        self,
        product_key: str,
        nm_dataset: str,
        nm_variavel: str,
        data_inicial: str,
        data_final: str,
        tipo_agregacao: str,
        filtro_anos: list[int],
        filtro_dias: list[int],
        filtro_meses: list[int],
        filtro_horas: list[int],
        pressure_level_hpa: int | None = None,
    ) -> dict[str, Any]:
        self.logger.info('Executando agregação temporal de reanálise...')

        headers = {'Content-Type': 'application/json'}
        if self.access_token:
            headers['x-access-token'] = self.access_token
        if self.user_token:
            headers['x-user-token'] = self.user_token

        payload: dict[str, Any] = {
            'nm_dataset': nm_dataset,
            'nm_variavel': nm_variavel,
            'data_inicial': data_inicial,
            'data_final': data_final,
            'tipo_agregacao': tipo_agregacao,
            'filtro_anos': filtro_anos,
            'filtro_dias': filtro_dias,
            'filtro_meses': filtro_meses,
            'filtro_horas': filtro_horas,
        }
        if pressure_level_hpa is not None:
            payload['pressure_level_hpa'] = pressure_level_hpa

        response = self.client.post(
            '/produtos/reanalise/agregacao-temporal',
            headers=headers,
            params={'product_key': product_key},
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        self._checar_resposta(data, 'agregacao-temporal')
        self.logger.info('Agregação temporal executada com sucesso!')
        return data

    def anomalia_reanalise(
        self,
        product_key: str,
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
    ) -> dict[str, Any]:
        self.logger.info('Executando cálculo de anomalia de reanálise...')

        headers = {'Content-Type': 'application/json'}
        if self.access_token:
            headers['x-access-token'] = self.access_token
        if self.user_token:
            headers['x-user-token'] = self.user_token

        payload: dict[str, Any] = {
            'nm_dataset': nm_dataset,
            'nm_variavel': nm_variavel,
            'data_inicial': data_inicial,
            'data_final': data_final,
            'ano_referencia_inicio': ano_referencia_inicio,
            'ano_referencia_fim': ano_referencia_fim,
            'tipo_agregacao': tipo_agregacao,
            'filtro_anos': filtro_anos,
            'filtro_anos_referencia': filtro_anos_referencia,
            'filtro_meses': filtro_meses,
            'filtro_dias': filtro_dias,
            'filtro_horas': filtro_horas,
        }
        if pressure_level_hpa is not None:
            payload['pressure_level_hpa'] = pressure_level_hpa

        response = self.client.post(
            '/produtos/reanalise/anomalia',
            headers=headers,
            params={'product_key': product_key},
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        self._checar_resposta(data, 'anomalia')
        self.logger.info('Anomalia calculada com sucesso!')
        return data

    def _get_reanalise(self, path: str, product_key: str) -> dict[str, Any]:
        headers = {'Content-Type': 'application/json'}
        if self.access_token:
            headers['x-access-token'] = self.access_token
        if self.user_token:
            headers['x-user-token'] = self.user_token
        response = self.client.get(path, headers=headers, params={'product_key': product_key})
        response.raise_for_status()
        return response.json()
    
    def _checar_resposta(self, data: dict[str, Any], endpoint: str) -> None:
        """Levanta exceção se a resposta da API indicar falha no processamento."""
        inner = data.get('data')
        if not isinstance(inner, dict):
            raise Exception(f"[{endpoint}] Resposta inesperada da API: {data}")
        code = inner.get('code')
        if code is not None and code != 200:
            raise Exception(f"[{endpoint}] Erro {code}: {inner.get('message', 'sem mensagem')}")