"""Registro central de variaveis meteorologicas suportadas pelos downloaders genericos
(downloaders_era5_generico.py / downloaders_gdas_generico.py).

Cada variavel e um vocabulario CONTROLADO -- em vez de aceitar qualquer nome de variavel do
CDS/GDAS "na mao" (arriscado: erro de digitacao so estoura na chamada da API, e os nomes do GDAS
via NOMADS sao curtos e criticos, ex. 'HGT'/'UGRD', sem nenhuma relacao obvia com os nomes do
ERA5), cada script escolhe uma CHAVE amigavel (ex. 'geopotencial') e este modulo resolve os
parametros reais de cada fonte.

Para adicionar uma variavel nova, crie uma entrada em VARIAVEIS com os campos de VariavelSpec.
"""

from __future__ import annotations

# Bibliotecas padrão
from dataclasses import dataclass


@dataclass(frozen=True)
class VariavelSpec:
    """Especificacao de uma variavel meteorologica nas duas fontes suportadas.

    Attributes:
        era5_nome: Nome da variavel no CDS (`reanalysis-era5-pressure-levels` ou
            `reanalysis-era5-single-levels`, conforme `requer_nivel`).
        era5_fator_conversao: Fator multiplicativo aplicado ao valor bruto do ERA5 para bater
            com a unidade de `var_saida` (ex. geopotencial m2/s2 -> altura geopotencial m e
            1/9.80665). 1.0 = sem conversao.
        gdas_var: Nome curto da variavel no filtro GRIB2 do NOMADS (`var_<gdas_var>=on`).
        requer_nivel: Se True, a variavel existe em multiplos niveis de pressao (hPa) e o
            chamador DEVE informar `nivel`. Se False, e uma variavel de nivel fixo (superficie,
            2m, 10m, coluna inteira, etc.) e `nivel` deve ficar None.
        gdas_nivel_template: Usado quando `requer_nivel=True` -- template da chave de nivel do
            filtro NOMADS, com `{nivel}` substituido (ex. 'lev_{nivel}_mb').
        gdas_nivel_fixo: Usado quando `requer_nivel=False` -- chave de nivel fixa do filtro
            NOMADS (ex. 'lev_mean_sea_level', 'lev_2_m_above_ground').
        var_saida: Nome da variavel no NetCDF final (mesmo para as duas fontes, para o script
            que consome o dado nao precisar saber de qual fonte veio).
        unidade: Unidade de `var_saida` (documentacional -- nao aplicada automaticamente).
        descricao: Descricao curta para mensagens de erro/log.
    """

    era5_nome: str
    gdas_var: str
    requer_nivel: bool
    var_saida: str
    unidade: str
    descricao: str
    era5_fator_conversao: float = 1.0
    gdas_nivel_template: str | None = None
    gdas_nivel_fixo: str | None = None

    def __post_init__(self) -> None:
        if self.requer_nivel and not self.gdas_nivel_template:
            raise ValueError(f'{self.descricao}: requer_nivel=True precisa de gdas_nivel_template')
        if not self.requer_nivel and not self.gdas_nivel_fixo:
            raise ValueError(f'{self.descricao}: requer_nivel=False precisa de gdas_nivel_fixo')


VARIAVEIS: dict[str, VariavelSpec] = {
    'geopotencial': VariavelSpec(
        era5_nome='geopotential',
        era5_fator_conversao=1 / 9.80665,  # m2/s2 -> altura geopotencial (m)
        gdas_var='HGT',
        requer_nivel=True,
        gdas_nivel_template='lev_{nivel}_mb',
        var_saida='hgt',
        unidade='m',
        descricao='Altura geopotencial',
    ),
    'temperatura': VariavelSpec(
        era5_nome='temperature',
        gdas_var='TMP',
        requer_nivel=True,
        gdas_nivel_template='lev_{nivel}_mb',
        var_saida='tmp',
        unidade='K',
        descricao='Temperatura do ar (nivel de pressao)',
    ),
    'vento_zonal': VariavelSpec(
        era5_nome='u_component_of_wind',
        gdas_var='UGRD',
        requer_nivel=True,
        gdas_nivel_template='lev_{nivel}_mb',
        var_saida='uwnd',
        unidade='m/s',
        descricao='Componente zonal (u) do vento',
    ),
    'vento_meridional': VariavelSpec(
        era5_nome='v_component_of_wind',
        gdas_var='VGRD',
        requer_nivel=True,
        gdas_nivel_template='lev_{nivel}_mb',
        var_saida='vwnd',
        unidade='m/s',
        descricao='Componente meridional (v) do vento',
    ),
    'umidade_relativa': VariavelSpec(
        era5_nome='relative_humidity',
        gdas_var='RH',
        requer_nivel=True,
        gdas_nivel_template='lev_{nivel}_mb',
        var_saida='rh',
        unidade='%',
        descricao='Umidade relativa (nivel de pressao)',
    ),
    'pressao_nivel_mar': VariavelSpec(
        era5_nome='mean_sea_level_pressure',
        gdas_var='PRMSL',
        requer_nivel=False,
        gdas_nivel_fixo='lev_mean_sea_level',
        var_saida='msl',
        unidade='Pa',
        descricao='Pressao ao nivel medio do mar (MSLP)',
    ),
    'temperatura_2m': VariavelSpec(
        era5_nome='2m_temperature',
        gdas_var='TMP',
        requer_nivel=False,
        gdas_nivel_fixo='lev_2_m_above_ground',
        var_saida='t2m',
        unidade='K',
        descricao='Temperatura a 2 metros',
    ),
    'vento_zonal_10m': VariavelSpec(
        era5_nome='10m_u_component_of_wind',
        gdas_var='UGRD',
        requer_nivel=False,
        gdas_nivel_fixo='lev_10_m_above_ground',
        var_saida='u10',
        unidade='m/s',
        descricao='Componente zonal (u) do vento a 10 metros',
    ),
    'vento_meridional_10m': VariavelSpec(
        era5_nome='10m_v_component_of_wind',
        gdas_var='VGRD',
        requer_nivel=False,
        gdas_nivel_fixo='lev_10_m_above_ground',
        var_saida='v10',
        unidade='m/s',
        descricao='Componente meridional (v) do vento a 10 metros',
    ),
    'agua_precipitavel': VariavelSpec(
        era5_nome='total_column_water_vapour',
        gdas_var='PWAT',
        requer_nivel=False,
        gdas_nivel_fixo='lev_entire_atmosphere_(considered_as_a_single_layer)',
        var_saida='pwat',
        unidade='kg/m2',
        descricao='Agua precipitavel (coluna atmosferica inteira)',
    ),
}


def get_variavel(chave: str) -> VariavelSpec:
    """Resolve uma chave de variavel para sua especificacao.

    Raises:
        ValueError: chave desconhecida -- lista as chaves disponiveis na mensagem.
    """
    spec = VARIAVEIS.get(chave)
    if spec is None:
        disponiveis = ', '.join(sorted(VARIAVEIS))
        raise ValueError(f"Variavel '{chave}' nao cadastrada. Disponiveis: {disponiveis}")
    return spec
