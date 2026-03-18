# ERA5 - Dados Horarios em Niveis Unicos (Single Levels)

**Fonte:** [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview)

**Dataset:** `reanalysis-era5-single-levels`

---

## O que e o ERA5?

O ERA5 e a **quinta geracao de reanálise do ECMWF** (Centro Europeu de Previsao do Tempo de Medio Prazo) para o clima e tempo global das ultimas 8 decadas. Os dados estao disponiveis **a partir de 1940 ate o presente**.

### O que e uma reanálise?

Imagine que voce tem:
- **Observacoes reais** espalhadas pelo mundo (estacoes meteorologicas, satelites, boias, avioes)
- **Um modelo numerico** que simula a atmosfera baseado em leis da fisica

A **reanálise** combina os dois: o modelo faz uma previsao e, a cada 12 horas, as observacoes reais "corrigem" essa previsao. Esse processo se chama **assimilacao de dados**. O resultado e uma estimativa otima do estado da atmosfera — mais completa que as observacoes isoladas e mais realista que o modelo sozinho.

Diferente de uma previsao operacional (que precisa ser rapida), a reanálise pode rodar com calma, usar mais observacoes e aplicar tecnicas mais sofisticadas. Por isso, gera dados de altissima qualidade.

### Principais caracteristicas

- **Estimativas horarias** de um grande numero de variaveis atmosfericas, oceanicas e de superficie
- **Incerteza estimada** por um ensemble de 10 membros (a cada 3 horas)
- **Atualizado diariamente**, com latencia de aproximadamente **5 dias**
- Os dados apresentados no CDS foram **re-interpolados** para uma grade regular lat-lon de **0.25 graus** (reanálise) e **0.5 graus** (estimativas de incerteza)

---

## Descricao dos Dados

| Campo | Valor |
|-------|-------|
| **Tipo de dado** | Grade regular (gridded) |
| **Projecao** | Grade regular latitude-longitude |
| **Cobertura horizontal** | Global |
| **Resolucao horizontal** | Reanálise: **0.25° x 0.25°** (~31 km no equador) para atmosfera, 0.5° x 0.5° para ondas oceanicas |
| **Cobertura temporal** | 1940 ate o presente |
| **Resolucao temporal** | **Horaria** |
| **Formato de arquivo** | GRIB (nativo), NetCDF (conversao disponivel) |
| **Frequencia de atualizacao** | Diaria (~5 dias de atraso) |

> **Nota para o projeto:** Nosso sistema baixa os dados em formato **NetCDF** (nao GRIB), selecionando a opcao `data_format: "netcdf"` na requisicao da API.

---

## Variaveis Utilizadas no Projeto

Das centenas de variaveis disponiveis no ERA5 single-levels, este projeto utiliza **3 variaveis** para os mapas de vento e pressao:

### 1. Pressao ao Nivel Medio do Mar (Mean Sea Level Pressure - MSLP)

| Propriedade | Valor |
|-------------|-------|
| **Nome em ingles** | Mean sea level pressure |
| **Nome curto (short name)** | `msl` |
| **Unidade** | Pa (Pascal) |
| **Nome no codigo** | `mean_sea_level_pressure` |

**O que e:** A pressao da atmosfera ao nivel medio do mar. E uma medida do peso de toda a coluna de ar acima de um determinado ponto, ajustada para o nivel do mar.

**Por que e importante:** A pressao ao nivel do mar e fundamental para identificar sistemas meteorologicos como **ciclones** (baixa pressao) e **anticiclones** (alta pressao). As linhas de igual pressao nos mapas (isobaras) revelam a direcao e intensidade dos ventos.

**Analogia:** Pense na pressao como o "peso" da atmosfera. Onde a atmosfera e mais leve (baixa pressao), o ar tende a subir e formar nuvens. Onde e mais pesada (alta pressao), o ar desce e o tempo fica mais estavel.

> **Conversao comum:** 1 hPa = 100 Pa. Os valores tipicos variam entre ~980 hPa (ciclones intensos) e ~1040 hPa (anticiclones fortes).

---

### 2. Componente U do Vento a 100m (100m U-component of wind)

| Propriedade | Valor |
|-------------|-------|
| **Nome em ingles** | 100m u-component of wind |
| **Nome curto (short name)** | `100u` |
| **Unidade** | m/s (metros por segundo) |
| **Nome no codigo** | `100m_u_component_of_wind` |

**O que e:** A componente **leste-oeste** (zonal) do vento a 100 metros de altura acima da superficie da Terra. Valores positivos indicam vento soprando **para leste**, valores negativos indicam vento soprando **para oeste**.

**Por que 100 metros?** A altura de 100m e o padrao da industria de **energia eolica**, pois corresponde a altura tipica do hub (centro do rotor) de turbinas eolicas modernas.

**Cuidado:** Ao comparar dados do modelo com observacoes, lembre-se que observacoes sao pontuais no espaco e no tempo, enquanto os dados do modelo representam medias sobre uma celula de grade.

---

### 3. Componente V do Vento a 100m (100m V-component of wind)

| Propriedade | Valor |
|-------------|-------|
| **Nome em ingles** | 100m v-component of wind |
| **Nome curto (short name)** | `100v` |
| **Unidade** | m/s (metros por segundo) |
| **Nome no codigo** | `100m_v_component_of_wind` |

**O que e:** A componente **norte-sul** (meridional) do vento a 100 metros de altura acima da superficie da Terra. Valores positivos indicam vento soprando **para norte**, valores negativos indicam vento soprando **para sul**.

**Mesma nota sobre comparacao** com observacoes se aplica aqui.

---

### Como U e V se combinam para dar velocidade e direcao

As componentes U e V sao como coordenadas cartesianas do vetor vento. Juntas, permitem calcular:

```
Velocidade do vento = sqrt(u^2 + v^2)

Direcao do vento = arctan2(u, v)  (com ajustes para convencao meteorologica)
```

**Exemplo visual:**

```
         Norte (V+)
           |
           |   /  Vento resultante
           |  /     = sqrt(u^2 + v^2)
           | /
           |/
Oeste -----+--------> Leste (U+)
(U-)       |
           |
           |
         Sul (V-)
```

Se U = 5 m/s e V = 5 m/s, o vento sopra para **nordeste** com velocidade de ~7.07 m/s.

---

## Horarios Sinoticos Utilizados

O projeto baixa dados em **4 horarios sinoticos** por dia:

| Horario UTC | Horario Brasilia (BRT, UTC-3) | Descricao |
|-------------|-------------------------------|-----------|
| 00:00 UTC | 21:00 BRT (dia anterior) | Meia-noite UTC |
| 06:00 UTC | 03:00 BRT | Madrugada |
| 12:00 UTC | 09:00 BRT | Manha |
| 18:00 UTC | 15:00 BRT | Tarde |

Esses 4 horarios sao os mais utilizados em meteorologia operacional, pois coincidem com os horarios padrao de lancamento de radiossondas e observacoes sinoticas mundiais.

> **No codigo:** Definido em `DEFAULT_SYNOPTIC_HOURS = (0, 6, 12, 18)` no arquivo `app/src/uteis/downloaders_wind100m_ERA5.py`.

---

## Area de Download do Projeto

O projeto baixa dados para a seguinte regiao:

| Limite | Valor | Descricao |
|--------|-------|-----------|
| **Norte** | 10°N | Norte da Venezuela |
| **Sul** | 40°S | Sul da Argentina |
| **Oeste** | 80°W | Pacifico leste (costa do Equador/Peru) |
| **Leste** | 15°E | Costa oeste da Africa |

Essa area cobre todo o Brasil, Atlantico Sul tropical e subtropical, e parte do Atlantico Norte tropical — ideal para analise de padroes de vento e pressao que afetam o clima brasileiro.

> **No codigo:** Definido como `area = [10.0, -80.0, -40.0, 15.0]` (formato: [N, W, S, E]).

---

## Referencia

- **Dataset oficial:** https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
- **Documentacao da API:** https://cds.climate.copernicus.eu/how-to-api
- **Documentacao ERA5 completa (ECMWF):** https://confluence.ecmwf.int/display/CKB/ERA5
- **Biblioteca Python:** `cdsapi` >= 0.7.7 (`pip install cdsapi`)
