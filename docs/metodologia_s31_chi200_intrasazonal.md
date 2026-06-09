# Metodologia — s31: CHI200 Intrasazonal

## Referência operacional

A metodologia do s31 é idêntica ao método operacional do **CPC/NOAA** para monitoramento em tempo real do sinal intrasazonal no potencial de velocidade em 200 hPa (CHI200). Esse método é descrito em:

> **Gottschalck et al. (2010)** — *A Framework for Assessing Operational Madden-Julian Oscillation Forecasts: A CLIVAR MJO Working Group Project*, BAMS.

e é o mesmo utilizado nos produtos de monitoramento da MJO disponíveis em:

- https://www.cpc.ncep.noaa.gov/products/precip/CWlink/MJO/mjo.shtml

---

## Passo a passo do método

### Etapa 1 — Download dos dados de vento

São baixados os dados diários de vento zonal (U) e meridional (V) em 200 hPa a partir de duas fontes, dependendo da disponibilidade:

- **ERA5** (ECMWF/Copernicus): cobre o período histórico até ~5 dias antes de hoje (latência do ERA5)
- **GDAS** (NCEP/NOAA): cobre os últimos dias até o dia corrente

Os dois datasets são concatenados em uma série temporal contígua de médias diárias.

---

### Etapa 2 — Remoção do ciclo sazonal (LTM diária)

Para cada dia `t` da série, subtrai-se a **climatologia diária** (Long-Term Mean — LTM) correspondente ao dia do ano:

```
anom(t) = vento(t) - LTM(dia_do_ano(t))
```

A LTM utilizada é a climatologia diária NCEP do período 1991–2020 (arquivo fixo em `Entrada/`), interpolada para a grade 2,5° do ERA5. Isso remove o **ciclo sazonal** (variação anual media), isolando a anomalia interanual + intrasazonal.

---

### Etapa 3 — Remoção do sinal interanual (média móvel de 120 dias)

Da anomalia calculada na etapa anterior, subtrai-se a **média móvel trailing de 120 dias**:

```
intra(t) = anom(t) - média( anom[t-120 : t-1] )
```

Esse passo remove a variabilidade de **baixa frequência** (interanual, tendências de longo prazo — ENOS, ODP, etc.), isolando apenas a banda **intrasazonal (20–90 dias)**, que corresponde à MJO e ondas equatoriais associadas.

> O valor de 120 dias é o padrão operacional do CPC. Equivale a dizer: "o sinal de hoje menos o que foi a tendência dos últimos 4 meses."

---

### Etapa 4 — Potencial de velocidade CHI200 (equação de Poisson)

Para cada dia `t`, o campo de vento intrasazonal `(u_intra, v_intra)` é decomposto em sua parte **divergente**, resolvendo a equação de Poisson na esfera:

```
∇²χ(t) = ∇ · V_intra(t)
```

onde `∇ · V_intra` é a divergência do vento intrasazonal. A solução `χ` é o **potencial de velocidade CHI200 intrasazonal**.

- Regiões de **CHI200 negativo** (verde nos mapas): **divergência** em 200 hPa → subsidência → convecção suprimida na superfície
- Regiões de **CHI200 positivo** (marrom nos mapas): **convergência** em 200 hPa → ascendência → convecção ativa na superfície

---

### Etapa 5 — Vento divergente associado ao CHI200

O vento divergente associado ao potencial de velocidade é calculado pelo gradiente esférico de χ:

```
u_div = (1 / a·cos φ) · ∂χ/∂λ
v_div = (1 / a)        · ∂χ/∂φ
```

onde `a` é o raio da Terra, `φ` é a latitude e `λ` é a longitude. Esses vetores são sobrepostos nos mapas como setas (quiver).

---

### Etapa 6 — Produtos gerados

| Produto | Descrição |
|---|---|
| **Mapas de pentada** | Média do CHI200 intrasazonal em blocos de 5 dias, das últimas N pentadas |
| **Hovmöller** | Evolução temporal do CHI200 médio na faixa equatorial (5°S–5°N), ao longo dos últimos 120 dias |
| **Mapa do período** | Média do CHI200 intrasazonal entre `DATA_INICIAL` e `DATA_FINAL` |

---

## Diferença em relação ao método Wheeler-Weickmann (WW)

O método do s31 isola a **banda intrasazonal total** (todos os modos juntos). O método WW (Wheeler & Kiladis 1999; Wheeler & Weickmann 2001) usa filtragem espectral 2D (número de onda × frequência) para separar os modos individualmente: MJO, ondas de Kelvin, ondas de Rossby equatorial (ER), etc. — como as isolinhas coloridas nos produtos do NCICS.

| | s31 (CPC running-mean) | Wheeler-Weickmann |
|---|---|---|
| Método | Média móvel no tempo | FFT 2D (espaço-tempo) |
| Sinal isolado | Intrasazonal total | Cada modo separado |
| Complexidade | Baixa | Alta |
| Uso operacional | CPC/NOAA tempo real | NCICS, pesquisa |
| Dados necessários | ~120+ dias | ~180–365 dias |
