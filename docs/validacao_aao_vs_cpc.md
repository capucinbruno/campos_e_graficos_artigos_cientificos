# Validação do índice AAO (s35) contra o CPC/NOAA

Registro de evidência de que o índice da Oscilação Antártica (AAO) calculado pelo
`s35` está **calibrado** em relação ao produto oficial do CPC/NOAA. Documento gerado
em 2026-07-15.

## Contexto

Surgiu a dúvida: a previsão do **GEFS** no `s35` alcançava ~−1 em julho/2026,
enquanto o gráfico oficial do CPS (`aao.gefs.sprd2.png`) mostrava o ensemble
chegando a ~−2. A suspeita era erro de normalização/escala no nosso índice.

Duas comparações resolveram a questão.

## 1. Média do ensemble vs membros individuais (linearidade)

O índice AAO do `s35` é uma **projeção linear** da anomalia de Z700 no EOF1, com
normalização por constante fixa (`norm_std`). Não há passo não-linear (sem
quadrado, módulo, ou normalização por membro). Logo, matematicamente:

```
índice( média dos campos dos membros )  ==  média dos índices de cada membro
```

Teste numérico com a função real `aao_index_from_height` (30 membros sintéticos):
diferença máxima entre as duas formas = **6,3 × 10⁻¹⁶** (ruído de arredondamento).

**Consequência:** baixar apenas o campo médio do ensemble (`geavg` do GEFS) e
projetá-lo no EOF dá **exatamente** a mesma média que projetar cada membro e mediar.
A escolha do `s35` (usar `geavg`) é correta e não perde nada para a **média**. O que
`geavg` não fornece é o **espalhamento** (spread) — para isso seriam necessários os
membros individuais (`gep01…gepNN`).

## 2. Observado do s35 vs CPC oficial (calibração de amplitude)

O CPC publica o índice AAO **mensal** em:
`https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/aao/monthly.aao.index.b79.current.ascii`
(o diário só é disponibilizado como PNG). Comparando a **média mensal** do nosso
observado (`dados/s35_verif/obs_archive.csv`) com o mensal do CPC, em 2026:

| Mês   | s35 (obs) | CPC oficial | Diferença | Dias |
|-------|:---------:|:-----------:|:---------:|:----:|
| Abril | +0,054    | −0,125      | +0,18     | 28   |
| Maio  | +0,261    | +0,544      | −0,28     | 31   |
| **Junho** | **+2,573** | **+2,506** | **+0,07** | 30 |

- **Junho** (mês de **sinal forte**, AAO+ ~2,5): diferença de **0,07 (3%)**, razão de
  amplitude **1,027**. Se houvesse erro de escala (índice ~½ do correto), Junho teria
  dado ~1,25 — não deu. **Amplitude confirmada.**
- Abril/Maio (sinal fraco, perto de zero): diferenças de ~0,2–0,3, esperadas por
  usarmos **EOF do ERA5 + climatologia diária do PSL/NCEP**, ligeiramente diferente do
  "tudo NCEP" do CPC. Viés médio (Mai+Jun) = **−0,11**, desprezível.

## Conclusão

A diferença observada no GEFS (nosso ~−1,2 vs tracejado do CPC ~−1,8) **não** é erro de
cálculo nem de normalização. É explicada por:

1. **Rodadas (init) diferentes** — os gráficos comparados eram de dias diferentes;
   perto de um vale o GEFS varia de init para init.
2. **Lagged ensemble mean** — o `s35` mistura init D + D−1
   (`_forecast_index` em `scripts/s35_aao_index.py`), o que amortece um pouco mais que
   a rodada única mostrada pelo CPC.
3. As linhas finas do gráfico do CPC (~−3,5) são **membros individuais**, não a média —
   comparando média-com-média a distância é pequena.

## Como reproduzir

- Observado salvo: `dados/s35_verif/obs_archive.csv` (coluna `obs_idx`).
- CPC mensal: baixar o `.ascii` acima e filtrar `^ 2026`.
- Comparar as médias mensais (meses completos: usar Maio e Junho).
