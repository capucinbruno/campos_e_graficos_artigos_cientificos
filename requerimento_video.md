# Requerimento de Vídeo — Globo 3D e Mapa 2D (s38 a s43)

Registro dos pedidos de vídeo do globo 3D (`s38` Ben Noll, `s39` Guillaume, `s40` estático,
`s41` Google Earth, `s42` The Weather Channel) e do mapa 2D plano (`s43` The Weather Channel
2D — mesma marca visual do s42, região fixa retangular, sem globo/câmera).

## Como funciona

1. Você descreve o vídeo em linguagem natural (foco geográfico, variável, câmera, ícones,
   caixa de texto, textos, tempos de fade/cauda etc.) — não precisa saber os nomes das
   settings nem preencher nada.
2. Eu traduzo o pedido pros nomes reais de setting (catálogo na seção 1-6 abaixo) e registro
   o resultado numa entrada nova em **"Pedidos"**, com data e os valores resolvidos.
3. Eu aplico esses valores no `settings.local.toml` só na hora de rodar, disparo o render em
   background e, ao terminar, **devolvo o `settings.local.toml` ao estado anterior** (o que
   estava ativo antes do pedido). Assim o `settings.local.toml` não fica acumulando lixo de
   pedido em pedido — o histórico de "o que foi pedido e com quais valores" vive só aqui.
4. Antes de disparar a rodada eu sempre mostro os valores resolvidos pra você confirmar.

O catálogo de settings abaixo (seções 1 a 6) é só referência — a MESMA setting listada como
"global (s38–s43)" vale pros 6 scripts ao mesmo tempo (não tem versão por script).

## Cuidados conhecidos (bugs já corrigidos — NÃO reintroduzir)

- **Corrente de jato "cortada" no meridiano de Greenwich (lon 0°/360°).** Já aconteceu DUAS
  vezes: (1) no globo 3D (`globo_3d_anim.py`), corrigido com `np.unwrap` em `_jet_segments` +
  diferença centrada PERIÓDICA em `_offset_polyline` (isolinhas circumpolares fecham dando a
  volta no globo). (2) no mapa 2D (`mapa_2d_anim.py`, s43), reaberto em 2026-07-04: a região
  "europa" cruza lon 0°, e o jato estava sendo desenhado DIRETO nos eixos reais via
  `_draw_jet_flat` (que chama `ax.plot`/`ax.text` **sem** `transform=`) — a grade global vem em
  0..360°, mas os eixos usam a janela -30..60°; sem `transform`, matplotlib trata os valores
  como coordenada já-projetada e corta tudo que cai fora da janela numérica (mesmo estando
  dentro da região visível fisicamente). **Regra permanente**: qualquer desenho DIRETO num eixo
  cartopy (`ax.plot`, `ax.text`, `ax.fill`) com dados de lon/lat precisa de `transform=`, OU
  (caso a função não aceite `transform`, como `_draw_jet_flat`/`_jato_flat_overlay`) renderizar
  num raster plano à parte (`_jato_raster`) e colar com `ax.imshow(..., transform=...,
  extent=[...])` — nunca chamar essas duas funções direto num `GeoAxes` real. Fix aplicado em
  `mapa_2d_anim.py::_build_frame_2d` (raster + imshow, igual ao mecanismo do globo). Resolve o
  corte GRANDE (faixa central inteira sumindo); confirmado visualmente.

- **Quebra fina nas faixas translúcidas na costura do anel fechado (2026-07-04) — tentativa de
  fix pontual REVERTIDA, causa raiz resolvida de outra forma (ver abaixo).** Uma primeira rodada
  desta investigação tentou 4 correções cirúrgicas dentro do próprio desenho do jato (`_tan` em
  `_jato_flat_overlay`, padding+compensação de DPI em `_jato_raster`, deslocamento de fase em
  `_jet_flow_sequence`, fechamento explícito do laço em `_draw_jet_flat`/`_draw_jet_stream`).
  Reduziu bastante o problema (texto "JET STREAM" foi de ilegível a legível), mas o usuário achou
  o resultado visual final pior que antes e pediu pra **desfazer tudo** — as 4 correções foram
  revertidas por completo (ficam só no histórico do git).

  **Causa raiz real, resolvida na ORIGEM em vez de no desenho**: `_jet_segments`/`_jato_raster`
  tratam o array de longitude como um retângulo comum — contourpy não sabe que a borda esquerda
  e a direita do array são o MESMO lugar físico no globo. Qualquer efeito de borda dessa
  reconstrução (isolinha circumpolar fragmentada e remontada, raster+imshow do jato) acontece
  exatamente onde o array COMEÇA/TERMINA — e isso é sempre 0°/360° por padrão, porque
  `add_cyclic_point` sempre fecha o array ali. Ou seja: o "corte no Meridiano de Greenwich" nunca
  foi um bug de UM desenho específico, e sim uma propriedade estrutural de qualquer array
  cíclico fechado em 0°.

  **Fix arquitetural (2026-07-04, aplicado em `globo_3d_anim.py`, vale para s38–s42 automaticamente
  — código compartilhado em `_render_clip`)**: em vez de caçar cada efeito colateral no desenho,
  a grade é GIRADA (só a ORDEM de armazenamento, nenhum ponto físico muda) para que a borda do
  array fique em ~160°E (meio do Pacífico) sempre que a câmera do clipe inteiro (fixa ou em voo)
  chega perto de Greenwich em algum frame — ver `_lon_seam_alvo`/`_lon_seam_roll_index`/
  `_lon_seam_roll_lon`/`_lon_seam_roll`, logo após `_camera_path`. Automático, sem nenhuma
  setting nova, sem custo quando a câmera nunca passa perto de Greenwich (roll é pulado nesse
  caso). Testado (GIF isolado, câmera fixa em cima da Europa): "JET STREAM" sai perfeitamente
  legível em TODAS as repetições do anel, sem nenhum resíduo nas faixas translúcidas — o
  artefato, que antes caía sempre sobre a Europa/Atlântico, agora cai no meio do Pacífico
  (~160°E), fora de qualquer enquadramento típico deste projeto.

  **Se um dia o enquadramento típico mudar pra incluir o Pacífico central** (ex.: um script novo
  com câmera fixa perto de 160°E), o mesmo artefato reapareceria ali — nesse caso, ajustar o
  `alvo` de `_lon_seam_alvo` (hoje fixo em 160.0) pra outra longitude livre, ou generalizar pra
  calcular a longitude MENOS visitada pela câmera durante o clipe em vez de um valor fixo.

  **Confirmado em produção 2026-07-04** (`s42`, z250_abs, reanálise 04-08/06, câmera fixa GE
  15°E/50°N — mesmo cenário do "Heat Dome"): MP4 completo (144 frames + 6s de cauda), PNG e GIF
  gerados em `Saida/s42_GLOBO_WEATHER_CHANNEL/REANALISE/`. Inspecionado frame a frame (início,
  início da cauda, meio da cauda, último frame): "JET STREAM" perfeitamente legível em TODAS as
  repetições do anel, sem nenhum resíduo na costura; caixa "Heat Dome persiste na região" fixa
  (opacidade total) desde o frame 0 até o último, sem esmaecer em nenhum momento — as duas
  correções desta rodada (deslocamento da costura + `caixa_fixa`) validadas visualmente.

- **Faixas translúcidas "abrindo leque"/se sobrepondo perto de um cavado/crista MUITO apertado
  (onda curta real, não o meridiano de Greenwich) — 3 redesenhos tentados e DESCARTADOS em
  2026-07-04, `_draw_jet_flat` (V1, produção) mantida intocada.** Esse é um problema DIFERENTE do
  corte no meridiano acima: acontece em QUALQUER longitude onde duas cristas do próprio
  escoamento ficam muito próximas (visto num caso real: GEFS Z500, pêntada 05-10/07, cavado sobre
  a Escandinávia/Rússia). As faixas finas são hoje construídas por `_offset_polyline` (V1):
  desloca cada ponto do traço pela normal LOCAL — isso nunca pula nem desenha errado, mas perto de
  um cavado/crista apertado a distância "abre leque" (as faixas se afastam da faixa central mais
  do que deveriam) ou se sobrepõem no lado côncavo. Três alternativas foram tentadas pra eliminar
  isso, todas descartadas por trazerem um problema NOVO, geralmente pior:
  1. **`shapely.offset_curve` (junção em arco/round join)**: matematicamente uma curva paralela
     "correta" a distância constante — mas quando duas cristas ficam mais próximas entre si do
     que a distância do deslocamento, a curva paralela verdadeira precisaria se autointersectar,
     e o shapely PODA esse trecho inteiro, ligando direto de um lado a outro — virou uma linha
     reta cruzando o mapa (bem mais chamativo que o problema original).
  2. **Campo de distância (`_jet_distance_field`, k-d tree) + raster (`ax.imshow`)**: colorir cada
     pixel pela distância até o traço mais próximo, sem construir uma segunda curva — geometria
     comprovadamente livre de artefatos (sem corte, sem salto, testado pixel a pixel). Mas as
     faixas saíram visualmente MUITO mais fracas que a V1 mesmo com a MESMA opacidade nominal
     (0,35, confirmado via percentis do canal alpha) — testado com 3 valores de "feather"
     (1,5px, 0,5px, 0,15px) e nenhum resolveu; aumentar a opacidade (2,2x) fez as faixas colarem
     numa massa só, perdendo a separação pedida. Pior ainda: no cavado apertado, a FAIXA OPACA
     também "incha" num borrão largo — porque o campo de distância não distingue "perto de que
     PARTE do traço", então o vão estreito entre duas cristas próximas fica todo dentro do
     raio da faixa, sendo preenchido como se fosse parte dela.
  3. **Isolinha do campo de distância (`contourpy.contour_generator(...).lines(dc)`) desenhada via
     `ax.plot` (mesmo mecanismo de desenho da V1)**: no trecho normal do traço, ficou visualmente
     IDÊNTICO à V1 (confirmado lado a lado). Mas bem no ponto mais apertado do cavado/crista, a
     isolinha (que passa perto do "eixo medial" — o lugar onde um ponto está igualmente perto de
     DUAS cristas diferentes) sofre uma mudança abrupta de qual trecho do traço original está
     "mais perto", produzindo um zigue-zague espúrio ali.

  **Causa raiz comum às 3 tentativas**: qualquer técnica que pergunte "quais pontos estão a X
  graus do traço" (curva deslocada explícita, campo de distância ou isolinha desse campo) sofre
  da mesma ambiguidade matemática exatamente onde duas partes do traço ficam mais próximas entre
  si do que a distância do deslocamento — é uma propriedade topológica de curvas perto de se
  autotocarem, não um bug de implementação. A V1 nunca sofre disso porque desloca cada ponto pela
  normal LOCAL (parametricamente ancorada ao próprio ponto de origem), nunca faz uma busca de
  "qual é o ponto mais próximo em todo o traço" — o preço é abrir um pouco o leque nesses cavados
  apertados, mas nunca pular/inchar/cruzar.

  **Também testado e descartado**: "glow" nativo (múltiplas larguras da MESMA linha, sem nenhum
  cálculo de deslocamento) — geometricamente perfeito (é literalmente a mesma curva), mas produz
  um gradiente suave sem separação nítida em faixas distintas, mudando a estética pedida.

  **Estado atual**: V1 (`_offset_polyline`) é a única versão em produção; nenhum código dos 3
  redesenhos ficou no arquivo (removido por completo, incluindo `_jet_distance_field`,
  `_draw_jet_flat_v2` e a setting de teste `GLOBO_3D_JATO_V2`). Se retomar essa investigação no
  futuro, o próximo passo mais promissor provavelmente é podar/suavizar especificamente os ramos
  espúrios da tentativa 3 (isolinha) perto do eixo medial, em vez de mais uma abordagem do zero.

- **Caixa de texto livre nunca aparecia nos frames de fundo CONGELADO (cauda do MP4 + todo o
  GIF) com `GLOBO_3D_BG_CACHE` ligado (default).** O fast-path do cache de fundo
  (`_render_overlay_rgba`) só compunha jato + ícones, nunca a caixa — ela só existia no caminho
  "completo" de `_build_frame`. Corrigido fatorando o desenho em `_draw_caixa_livre` (chamada
  tanto no caminho completo quanto de dentro de `_render_overlay_rgba`); a construção do fundo
  cacheado (`skip_overlay=True`) pula a caixa explicitamente, já que ela é sempre redesenhada por
  frame agora. **Regra permanente**: qualquer elemento novo que precise aparecer nos frames
  congelados do MP4/GIF tem que ser desenhado dentro de `_render_overlay_rgba`, não só no
  caminho completo de `_build_frame` — senão some quando `GLOBO_3D_BG_CACHE=true`.
- **Caixa de texto livre esmaecendo (fade-in) mesmo quando o conteúdo pedia um rótulo FIXO.**
  Nova flag `caixa_fixa` (`GLOBO_3D_FADE_CAUDA=true` no s42, vale em QUALQUER saída — MP4, GIF,
  PNG): força `alpha_max` sempre, sem rampa. Sem essa flag, mantém o fade-in configurável de
  sempre (`GLOBO_3D_CAIXA_LIVRE_INICIO`/`FADE`).

---

## 1. Script(s) e variável

- Script(s) a rodar: `s38` / `s39` / `s40` / `s41` / `s42` / `s43` (2D plano)
- Período: `DATA_INICIAL` = _____ / `DATA_FINAL` = _____
- Modo: `MODE` = `reanalysis` (ERA5/GDAS) ou `forecast` (qual modelo: GFS/ECMWF/...)
- Variável(is):
  - s38/s39/s40/s41 → `VARIAVEIS_GLOBO_3D` (lista compartilhada; ex.: `z250_anom`, `tmp850_anom`, `jet_stream`...)
  - s42 (plota ABSOLUTO, não anomalia) → `GLOBO_3D_VARIAVEIS_S42` (`z250_abs`, `z500_abs`)
  - s43 (mapa 2D, mesmas variáveis absolutas do s42) → `GLOBO_3D_VARIAVEIS_S43`
- Se `z250_abs`/`z500_abs` (s42/s43): confirmar escala/paleta em `GLOBO_3D_VMIN/VMAX/NIVEIS/PALETA/ALPHA_<VAR>` e cor do oceano/fronteiras/suavização (`GLOBO_3D_COR_OCEANO_<VAR>`, `GLOBO_3D_COR_FRONTEIRAS_<VAR>`, `GLOBO_3D_LW_COAST/BORDER/STATES_<VAR>`, `GLOBO_3D_SIGMA_<VAR>`) — `<VAR>` = `Z250_ABS` ou `Z500_ABS`. Z500 já vem calibrado pela carta de referência GFS (468–600 dam = 4680–6000 mgp).
- **s43 (mapa 2D): jato só funciona (v1) com variável hgt absoluta própria (`z250_abs`/`z500_abs`)** — outras variáveis rodam sem jato (aviso no log).

## 2. Mapa / projeção e voo da câmera

**Projeção** (fixa por script; não muda por rodada, só documento aqui):
- s38/s39/s40 → `GLOBO_3D_PROJECTION` = `nearside` (globo 3D flutuante) ou `orthographic`
- s41/s42 → `GLOBO_3D_PROJECTION_S41` = `google_earth` (satélite/paisagem) — compartilhada entre os dois

**Câmera s38/s39/s40** (setting `GLOBO_3D_*`, sem sufixo):
- Ponto inicial: lat = _____ / lon = _____ (`GLOBO_3D_LAT_INICIAL` / `GLOBO_3D_LON_INICIAL`)
- Ponto final: lat = _____ / lon = _____ (`GLOBO_3D_LAT_FINAL` / `GLOBO_3D_LON_FINAL`)
- Inclinação fixa (`GLOBO_3D_INCLINACAO`, "" = usa lat inicial/final)
- Voltas extra (`GLOBO_3D_VOLTAS_EXTRA`, giros completos antes de assentar)
- Easing (`GLOBO_3D_EASING`: linear / ease_in / ease_out / ease_in_out)
- Velocidade da variável (`GLOBO_3D_VELOCIDADE_VAR`)
- Frames/dia e fps (`GLOBO_3D_FRAMES_POR_DIA`, `GLOBO_3D_FPS`)

**Câmera s41/s42** (setting `GLOBO_3D_GE_*` — compartilhada entre os dois; sem elas, herdam os valores acima):
- Ponto inicial: lat = _____ / lon = _____ (`GLOBO_3D_GE_LAT_INICIAL` / `GLOBO_3D_GE_LON_INICIAL`)
- Ponto final: lat = _____ / lon = _____ (`GLOBO_3D_GE_LAT_FINAL` / `GLOBO_3D_GE_LON_FINAL`) — igual ao inicial = câmera fixa, sem voo
- Altura da câmera / zoom (`GLOBO_3D_GE_ALTURA`, metros — menor = mais perto)
- Enquadramento: `GLOBO_3D_GE_ASPECT` (proporção do quadro), `GLOBO_3D_GE_GLOBO_FRAC` (zoom do disco), `GLOBO_3D_GE_GLOBO_CY` (sobe/desce o globo no quadro)
- Inclinação (`GLOBO_3D_GE_INCLINACAO`), voltas extra (`GLOBO_3D_GE_VOLTAS_EXTRA`)
- Easing/frames/fps próprios (opcionais: `GLOBO_3D_GE_EASING`, `GLOBO_3D_GE_FRAMES_POR_DIA`, `GLOBO_3D_GE_FPS`, `GLOBO_3D_GE_VELOCIDADE_VAR`)

**Mapa s43 (2D plano — SEM câmera, região fixa retangular tipo carta sinótica)**:
- Região: `GLOBO_2D_AREA` (chave de `settings["areas_plotagem"]`; default `"europa"`)
- Transição entre horas sinóticas: crossfade suave, `GLOBO_2D_FRAMES_POR_PASSO` (frames por transição), `GLOBO_2D_VELOCIDADE_VAR`
- `GLOBO_2D_FPS`, `GLOBO_2D_FIGSIZE_W/H` (polegadas), `GLOBO_2D_DPI`
- Cauda do MP4 (campo congela, jato continua): `GLOBO_2D_JATO_MP4_CAUDA_SEG` + `GLOBO_2D_FADE_CAUDA_DUR_SEG`
- GIF: `GLOBO_2D_GIF_FRAMES` / `GLOBO_2D_GIF_FPS`

## 3. Corrente(s) de jato — global (s38–s43)

- Ligar? `GLOBO_3D_JATO` = true/false
- Hemisférios: `GLOBO_3D_JATO_HEMISFERIO_NORTE` / `_SUL`
- Drape (colado na superfície): `GLOBO_3D_JATO_DRAPE`
- Cauda do MP4 (campo congela, jato continua fluindo): `GLOBO_3D_JATO_MP4_CAUDA_SEG` (segundos; 0 = sem cauda)
- Fade-in sincronizado no início da cauda (**só s42**): `GLOBO_3D_FADE_CAUDA` + `GLOBO_3D_FADE_CAUDA_DUR_SEG`

**Jato 1 — JET STREAM:**
- `GLOBO_3D_JET_STREAM` = true/false
- Nível/posição (Z250 mgp): `GLOBO_3D_JET_STREAM_NIVEL`
- Cor: `GLOBO_3D_JET_STREAM_COR`
- Texto animado: `GLOBO_3D_JET_STREAM_TEXTO`
- Velocidade do fluxo: `GLOBO_3D_JET_STREAM_VELOCIDADE`

**Jato 2 — SUBTROPICAL JET:**
- `GLOBO_3D_SUBTROPICAL_JET` = true/false
- Nível: `GLOBO_3D_SUBTROPICAL_JET_NIVEL`
- Cor: `GLOBO_3D_SUBTROPICAL_JET_COR`
- Texto: `GLOBO_3D_SUBTROPICAL_JET_TEXTO`
- Velocidade: `GLOBO_3D_SUBTROPICAL_JET_VELOCIDADE`

**Estilo compartilhado (ambos os jatos):**
- Espessura faixa central: `GLOBO_3D_JATO_LARGURA` (screen) / `GLOBO_3D_JATO_LARGURA_DEG` (drape)
- Faixas finas translúcidas: `GLOBO_3D_JATO_STRIPE_N` (nº), `GLOBO_3D_JATO_STRIPE_LARGURA`/`_DEG`, `GLOBO_3D_JATO_STRIPE_ALPHA`
- Setas: `GLOBO_3D_JATO_SETAS_ENTRE`, `GLOBO_3D_JATO_SETAS_PASSO`, `GLOBO_3D_JATO_ARROW_TAM`/`_DEG`, `GLOBO_3D_JATO_ARROW_COR`
- Texto: `GLOBO_3D_JATO_TEXTO_TAM`/`_DEG`, `GLOBO_3D_JATO_TEXTO_COR`
- Banda de latitude: `GLOBO_3D_JATO_LAT_MIN` / `_MAX`

## 4. Ícones de pressão animados — global (s38–s43)

Lista `GLOBO_3D_ICONES_PRESSAO` (um item por ícone; vazio = nenhum):

```toml
GLOBO_3D_ICONES_PRESSAO = [
    {tipo = "____", lat = ____, lon = ____, velocidade = ____, tamanho_deg = ____,
     fade_in = false, fade_inicio = 0.0, fade_duracao = 0.0},
]
```

- `tipo`: qual ícone (GIF em `Entrada/icones_pressao/`, ex. `HIGH_HN`, `BAIXA_HS`)
- `lat` / `lon`: posição no globo
- `velocidade`: velocidade de rotação do GIF
- `tamanho_deg`: diâmetro em graus (default 8.0)
- `fade_in` / `fade_inicio` / `fade_duracao`: fade-in opcional (default: aparece direto)
- Resolução de reprojeção (raramente precisa mudar): `GLOBO_3D_ICONE_PRESSAO_REGRID`

## 5. Caixa de texto livre — global (s38–s43)

- Ligar? `GLOBO_3D_CAIXA_LIVRE` = true/false
- Posição: `GLOBO_3D_CAIXA_LIVRE_LAT` / `GLOBO_3D_CAIXA_LIVRE_LON`
- Texto: `GLOBO_3D_CAIXA_LIVRE_TEXTO` (`\n` quebra linha manual; senão quebra automática por `_LARGURA`)
- Cores: `GLOBO_3D_CAIXA_LIVRE_COR_BOX`, `_COR_TEXTO`, `_CONTORNO_COR`, `_CONTORNO_LW` (espessura do contorno; 0 = sem)
- Tamanho da caixa: `GLOBO_3D_CAIXA_LIVRE_FONTSIZE` (tamanho da letra — a caixa acompanha) + `GLOBO_3D_CAIXA_LIVRE_PAD` (espaço texto→borda) + `GLOBO_3D_CAIXA_LIVRE_LARGURA` (caracteres antes de quebrar linha)
- Opacidade final: `GLOBO_3D_CAIXA_LIVRE_ALPHA_MAX`
- Sombra esfumaçada atrás da caixa: `GLOBO_3D_CAIXA_LIVRE_SOMBRA`
- Fade-in: `GLOBO_3D_CAIXA_LIVRE_INICIO` (fração do clipe onde começa) + `GLOBO_3D_CAIXA_LIVRE_FADE` (duração, fração do clipe) — **exceção**: se `GLOBO_3D_FADE_CAUDA=true` (só s42), o fade da caixa passa a seguir o início da cauda do jato, ignorando `_INICIO`/`_FADE`

## 6. Aparência geral

- Crédito: `GLOBO_3D_CREDITO` (nome no rodapé)
- Vinheta (escurece cantos): `GLOBO_3D_VINHETA`
- Atmosfera + estrelas: `GLOBO_3D_ATMOSFERA_ESTRELAS` (true = halo azul + estrelas, tem prioridade) / `GLOBO_3D_SOMENTE_ESTRELAS` (só vale se a de cima = false)
- Fonte do título/legenda: `GLOBO_3D_FONTE_TITULO`, `GLOBO_3D_FONTE_LEGENDA`
- Espessura das linhas de costa/países/estados: `GLOBO_3D_COASTLINE_LW`, `GLOBO_3D_BORDERS_LW`, `GLOBO_3D_STATES_LW`

**Só s42:**
- Fundo satélite (blue marble): `GLOBO_3D_BLUE_MARBLE` (só aparece onde o shaded estiver semi-transparente — precisa `_ALPHA_<VAR>` < 1.0 pra ver por baixo)
- Modo minimalista (some título/data/legenda, só fica o crédito): `GLOBO_3D_SO_CREDITO`
- Caixa "The Weather Channel" (título + data BR, exige `GLOBO_3D_SO_CREDITO=true`): `TITULO_THE_WEATHER_CHANNEL` (texto), `GLOBO_3D_FONTE_TWC` (fonte das duas caixas, "" = usa a legenda)

## 7. Execução

- Rodar em background: `uv run python run_script.py <sNN> --data-inicial <ini> --data-final <fim> --force-rerun`

---

## Pedidos

Cada pedido feito em linguagem natural vira uma entrada aqui, com data, o texto original
(resumido) e a tabela de settings resolvidas que foi de fato aplicada naquela rodada.

### 2026-07-04 — s42, z500_abs, jato na isolinha 5760 mgp, ícone França/Alemanha — APLICADO

> "quero um vídeo do s42 com foco no continente europeu, câmera de voo iniciando devagar no
> leste do Atlântico Norte e terminando lentamente sobre a Europa; variável z500 animada em
> velocidade 3, período 05 a 10/07/2026; caixa The Weather Channel com o texto 'Heat dome
> persiste'; no último tempo da variável, esmaecer e aparecer a corrente de jato contornando a
> isolinha da altura geopotencial 5760 mgp (576 dam), oculta, + ícone de alta pressão do HN
> entre a França e a Alemanha + caixa de texto livre abaixo do ícone (mesma cor do jato)
> escrito 'Calor Extremo continua'; esses três elementos ficam visíveis por 10 segundos até
> terminar o vídeo."

| Setting | Valor |
|---|---|
| Script | `s42` |
| Período (via CLI, não settings) | `--data-inicial 2026-07-05 --data-final 2026-07-10` |
| `GLOBO_3D_VARIAVEIS_S42` | `["z500_abs"]` |
| `GLOBO_3D_GE_LAT_INICIAL` / `GLOBO_3D_GE_LON_INICIAL` | `48.0` / `-25.0` (leste do Atlântico Norte) |
| `GLOBO_3D_GE_LAT_FINAL` / `GLOBO_3D_GE_LON_FINAL` | `50.0` / `15.0` (Europa continental — já era o valor ativo) |
| `GLOBO_3D_GE_EASING` | não setado → herda `GLOBO_3D_EASING = ease_in_out` (início E fim lentos) |
| `GLOBO_3D_GE_VELOCIDADE_VAR` | `3.0` |
| `TITULO_THE_WEATHER_CHANNEL` | `"Heat dome persiste"` |
| `GLOBO_3D_JET_STREAM_NIVEL` | `5760` (mgp — isolinha do PRÓPRIO campo shaded z500_abs, não Z250) |
| `GLOBO_3D_FADE_CAUDA` / `_DUR_SEG` | `true` / `1.5` (já ativo, sem mudança) |
| `GLOBO_3D_JATO_MP4_CAUDA_SEG` | `10.0` (era `6.0`) |
| `GLOBO_3D_ICONES_PRESSAO` | `[{tipo="HIGH_HN", lat=48.5, lon=6.5, velocidade=2.0, tamanho_deg=10.0}]` (entre França e Alemanha) |
| `GLOBO_3D_CAIXA_LIVRE_LAT` / `_LON` | `44.5` / `6.5` (abaixo do ícone) |
| `GLOBO_3D_CAIXA_LIVRE_TEXTO` | `"Calor Extremo continua"` |
| `GLOBO_3D_CAIXA_LIVRE_COR_BOX` | `"#0077a7"` (já era a cor do jato, sem mudança) |

**Mudança de código necessária** (não é só settings): o mecanismo que reaproveita o próprio
campo shaded como guia do jato (evitando baixar Z250 à parte) era travado em
`ficha['spec']['kind'] == 'hgt250'` — generalizado para `('hgt250', 'hgt500')` em
`globo_3d_anim.py` (2 pontos), senão o jato tentaria contornar 5760 mgp dentro do campo de
Z250 (~9800–11200 mgp), onde essa isolinha nunca existe, e não desenharia nada.

Status: **concluído** — MP4+PNG+GIF gerados em `Saida/s42_GLOBO_WEATHER_CHANNEL/FORECAST/GFS/`
(1185s). Jato seguiu corretamente a isolinha 5760 mgp do próprio campo de Z500 (confirmado
visualmente no PNG da média). `settings.local.toml` já revertido ao estado anterior a este pedido.

### 2026-07-04 — s42, GEFS Z500 média, Europa, jato 5670 mgp, sem jato no PNG

> "faça através do s42 o campo da média da altura geopotencial em 500 hPa do modelo GEFS
> rodada de hoje das 00 UTC para a região da Europa, com previsão válida de 05/07 a 10/07.
> Com exceção do png, as demais saídas (gif e mp4) devem ter a corrente de jato animada
> contornando a isolinha de 5670 mgp como guia. Ícone de pressão do HN com centro em 48ºN,
> 10°O. Logo abaixo do ícone, caixa de texto livre na cor da corrente de jato escrito 'Calor
> extremo persiste na região'. Título da caixa azul: 'Heat dome setup'. Caixa cinza de datas:
> formato 05-10/07 (sem ano)."

| Setting | Valor |
|---|---|
| Script | `s42` |
| Período (via CLI, não settings) | `--data-inicial 2026-07-05 --data-final 2026-07-10` |
| `GLOBO_3D_VARIAVEIS_S42` | `["z500_abs"]` |
| Modelo | `RUN_GEFS = true`, `RUN_GFS = false` (isola só GEFS nesta rodada) |
| `RODADA` | `"00"` |
| `FORECAST_INIT` | `"2026-07-04"` (força a rodada de HOJE 00Z — sem isso o GEFS cairia no D-1 por padrão, `GEFS_FORECAST_LEAD_DAYS=35` > 16d) |
| `GLOBO_3D_GE_LAT/LON_INICIAL/FINAL` | `50.0`/`15.0` (Europa — já era o valor ativo, câmera fixa) |
| `GLOBO_3D_JET_STREAM_NIVEL` | `5670` (mgp, isolinha-guia do próprio campo Z500) |
| `GLOBO_3D_JATO_PNG` | `false` (**novo**: PNG sai sem jato; GIF/MP4 mantêm o jato animado) |
| `GLOBO_3D_ICONES_PRESSAO` | `[{tipo="HIGH_HN", lat=48.0, lon=-10.0, velocidade=2.0, tamanho_deg=10.0}]` |
| `GLOBO_3D_CAIXA_LIVRE_LAT` / `_LON` | `42.0` / `-10.0` (abaixo do ícone) |
| `GLOBO_3D_CAIXA_LIVRE_TEXTO` | `"Calor extremo persiste na região"` |
| `GLOBO_3D_CAIXA_LIVRE_COR_BOX` | `"#0077a7"` (cor do jato — já era o valor ativo) |
| `TITULO_THE_WEATHER_CHANNEL` | `"Heat dome setup"` |

**Mudanças de código necessárias** (não é só settings):
1. **Nova setting `GLOBO_3D_JATO_PNG`** (default `true`) — antes o jato era tudo-ou-nada nas 3
   saídas; agora dá pra excluir só do PNG (`_render_clip` passa `skip_jet` pro `_build_frame`,
   parâmetro que já existia internamente pro cache de fundo).
2. **Fix em `_fmt_data_br`**: o intervalo de datas no mesmo mês/ano estava saindo
   `05–10/07/2026` (com ano), mas o próprio docstring da função já documentava o formato certo
   sem ano (`'20–24/07'`) — código não batia com o que estava documentado. Corrigido — vale
   pra qualquer pedido futuro com intervalo no mesmo mês, não só este.

Status: **concluído** — MP4+PNG+GIF gerados em `Saida/s42_GLOBO_WEATHER_CHANNEL/FORECAST/GEFS/`
(923s). Confirmado visualmente no PNG: título "Heat dome setup", caixa de datas "05–10/07"
(sem ano), ícone sobre o Atlântico a oeste da Península Ibérica, SEM jato no PNG (GIF/MP4
mantiveram o jato). `settings.local.toml` já revertido ao estado anterior a este pedido.

#### Correção 2026-07-04 — MP4 deveria ser campo médio fixo (igual ao GIF), não animar a variável

> "Eu não queria uma animação da variável altura geopotencial no mp4. O que eu esperava era
> que o mp4 fosse igual o gif, mostrando a média da variável no período, com a animação do
> fluxo da corrente de jato ao longo da isolinha média de 5670 mgp, a animação do ícone de
> alta pressão, e a caixa de texto livre (mesma cor do jato) escrito 'Calor extremo persiste
> na região', abaixo do ícone. Jato/ícone/caixa não precisam de esmaecer, já aparecem direto.
> Aumente um pouco o ícone de alta pressão."

Entendimento errado da 1ª rodada: o MP4 do s42 sempre anima a variável dia a dia/sinótica —
não existia uma forma de fazer o MP4 mostrar o MESMO campo médio fixo do PNG/GIF. Precisou de
mudança de código (ver abaixo), não só settings.

| Setting | Valor |
|---|---|
| `GLOBO_3D_MP4_MEDIA_FIXA` | `true` (**novo**: MP4 usa a MÉDIA do período + câmera parada, igual ao GIF) |
| `GLOBO_3D_FADE_CAUDA` | `false` (jato/ícone/caixa aparecem direto, sem esmaecer) |
| `GLOBO_3D_ICONES_PRESSAO` | `tamanho_deg`: `10.0` → `13.0` (ícone um pouco maior) |
| `GLOBO_3D_JATO_MP4_CAUDA_SEG` | `6.0` → `15.0` (com `MP4_MEDIA_FIXA`, essa setting vira a duração do vídeo inteiro, não só uma cauda — 15s escolhido como duração razoável, sem pedido explícito) |
| (demais settings do pedido anterior) | inalteradas — `z500_abs`, GEFS hoje 00Z, região Europa, `JET_STREAM_NIVEL=5670`, `JATO_PNG=false`, ícone 48°N/10°O, caixa livre "Calor extremo persiste na região" abaixo do ícone, título "Heat dome setup" |

**Mudanças de código necessárias:**
3. **Nova setting `GLOBO_3D_MP4_MEDIA_FIXA`** — `gerar_animacao` agora calcula a MÉDIA do
   período ANTES da chamada do MP4 (antes só existia pro PNG/GIF) e, com a flag ligada, passa
   essa média + câmera fixa pro MP4 em vez da série que evolui no tempo.
4. **Fix em `_render_clip`**: o branch de MP4 "normal" ignorava totalmente o parâmetro
   `camera` (só `estatico`/`gif` respeitavam) — sempre usava `_camera_path` (voo). Agora, se
   `camera` for passado, usa esse ponto fixo, sem voo.

Status: **substituído pela rodada seguinte** (essa versão tinha 2 problemas — ver correção abaixo).

#### Correção 2026-07-04 (2) — caixa invisível, ícone sobrepondo, jato/ícone velocidade 2

> "Não resolveu ainda a questão das isolinhas translúcidas do jato estarem cortadas no
> meridiano de Greenwich. E também você ignorou meu pedido de colocar a caixa de texto livre
> abaixo do ícone! E atenção para a caixa não sobrepor o ícone. [+ depois] aumente tb para 2 a
> velocidade da corrente de jato e do ícone de alta pressão."

Diagnóstico:
- **Caixa "sumida"**: não foi ignorada — `GLOBO_3D_BG_CACHE=true` tem um caminho rápido (fundo
  congelado) que NÃO inclui a caixa de texto livre, só jato+ícones. Como o vídeo agora é campo
  fixo o tempo inteiro (`MP4_MEDIA_FIXA`), a caixa nunca era composta. Desliguei o cache pra
  esse pedido.
- **Ícone maior (13°) ficou perto demais da caixa** (só 6° de distância) — quase sobrepondo.
  Afastei mais (`CAIXA_LIVRE_LAT` 42.0 → 38.0). Confirmado via GIF de teste: sem sobreposição.
- **Isolinhas translúcidas ainda cortadas**: investigado a fundo (ver seção "Cuidados
  conhecidos" no topo deste arquivo) — é um problema DIFERENTE do corte grande já corrigido.
  Tentei uma correção (folga extra no raster) que **piorou** (virou padrão de veneziana) —
  revertida. **Ainda não resolvido.**

| Setting | Valor |
|---|---|
| `GLOBO_3D_BG_CACHE` | `true` → `false` (caixa livre não entra no cache de fundo) |
| `GLOBO_3D_CAIXA_LIVRE_LAT` | `42.0` → `38.0` (mais longe do ícone, que ficou maior) |
| `GLOBO_3D_JET_STREAM_VELOCIDADE` | `1.0` → `2.0` |
| Ícone `velocidade` | já estava `2.0`, sem mudança |

Status: **concluído** — MP4+PNG+GIF gerados em `Saida/s42_GLOBO_WEATHER_CHANNEL/FORECAST/GEFS/`
(1342s). Confirmado visualmente: PNG sem jato, ícone maior no lugar certo; GIF/MP4 com caixa
visível abaixo do ícone sem sobrepor, jato/ícone mais rápidos. `settings.local.toml` revertido.
Corte fino nas faixas translúcidas segue **pendente** (ver diagnóstico completo na seção
"Cuidados conhecidos" no topo deste arquivo — causa raiz isolada, mas fix ainda não encontrado).

#### Re-rodada 2026-07-04 — com os 4 fixes da costura do anel fechado aplicados

Mesmos settings do pedido acima (GEFS hoje 00Z, z500_abs, Europa, jato 5670 mgp, ícone
48°N/10°O, caixa abaixo do ícone, velocidade 2), rodado de novo só pra validar os 4 fixes de
código da seção "Cuidados conhecidos" (tangente na costura, resolução, fase da sequência de
texto, fechamento do laço).

Status: **concluído** — MP4+PNG+GIF gerados (1262s). Confirmado visualmente: "JET STREAM" agora
sai **perfeitamente legível em todas as repetições** ao redor do anel (antes saía
"JET"+bloco de barras+"REAM"). Resíduo bem menor remanescente, do tamanho de uma seta, ainda nas
faixas translúcidas — ver pendência detalhada acima. `settings.local.toml` revertido (incluindo
`RUN_GFS`/`RUN_GEFS`, que tinham ficado no estado do pedido anterior por engano num revert).

**Opções discutidas com o usuário pra eliminar o resíduo remanescente (nenhuma aplicada ainda,
usuário ainda decidindo):**
1. **Girar a costura da grade cíclica pra uma longitude "segura"** (ex.: 160°E, meio do
   Pacífico) em vez de deixá-la sempre em 0°/360° — `np.roll` no array antes de desenhar, sem
   mudar nada do visual. Funciona de forma simples e permanente pro s43 (região fixa) e pro s42
   com câmera fixa; precisaria recalcular por frame se um dia tiver voo de câmera longo que
   passe perto da nova costura também.
2. **Trocar o modo "drape" (raster+imshow) pelo modo "tela" (`_draw_jet_stream`, já existe no
   código) só pro jato** — desenha direto na projeção real via `transform=`, sem o
   raster+imshow intermediário que está causando o artefato de compositing. Sem problema pro
   mapa 2D (sem limbo); pro globo, só teria a desvantagem visual antiga do modo tela (elementos
   não se curvam perto da borda do disco) se a câmera enquadrar o jato bem no limbo.

### 2026-07-04 — s42, Europa, alta pressão sobre a França

> "quero um vídeo do s42 com foco no continente europeu, câmera de voo iniciando devagar no
> leste do Atlântico Norte e terminando sobre a Europa; variável hgt250 (Z250) animada em
> velocidade 2, período 05 a 10/07/2026; caixa The Weather Channel com o texto 'Alta pressão
> segue'; no último tempo da variável, esmaecer-e-aparecer a corrente de jato + ícone de alta
> pressão do HN sobre a França + caixa de texto livre do lado do ícone (mesma cor do jato) com
> 'Calor Extremo'; esses três elementos animados ficam visíveis por uns 8 segundos."

| Setting | Valor |
|---|---|
| Script | `s42` |
| `DATA_INICIAL` / `DATA_FINAL` | `2026-07-05` / `2026-07-10` |
| `GLOBO_3D_VARIAVEIS_S42` | `["z250_abs"]` |
| `GLOBO_3D_GE_LAT_INICIAL` / `GLOBO_3D_GE_LON_INICIAL` | `48.0` / `-25.0` (leste do Atlântico Norte) |
| `GLOBO_3D_GE_LAT_FINAL` / `GLOBO_3D_GE_LON_FINAL` | `50.0` / `15.0` (Europa continental) |
| `GLOBO_3D_GE_EASING` | `ease_in` (começa devagar, acelera) |
| `GLOBO_3D_GE_VELOCIDADE_VAR` | `2.0` |
| `GLOBO_3D_SO_CREDITO` | `true` |
| `TITULO_THE_WEATHER_CHANNEL` | `"Alta pressão segue"` |
| `GLOBO_3D_JATO` / `GLOBO_3D_JET_STREAM` | `true` / `true` (config já ativa, sem mudança) |
| `GLOBO_3D_FADE_CAUDA` | `true` |
| `GLOBO_3D_FADE_CAUDA_DUR_SEG` | `1.5` (fade rápido; ficam visíveis o resto da cauda) |
| `GLOBO_3D_JATO_MP4_CAUDA_SEG` | `8.0` (tempo total visível após o fade) |
| `GLOBO_3D_ICONES_PRESSAO` | `[{tipo="HIGH_HN", lat=47.0, lon=2.0, velocidade=2.0, tamanho_deg=10.0}]` (França; fade automático via `FADE_CAUDA`) |
| `GLOBO_3D_CAIXA_LIVRE` | `true` |
| `GLOBO_3D_CAIXA_LIVRE_LAT` / `_LON` | `47.0` / `12.0` (ao lado do ícone, dentro do quadro) |
| `GLOBO_3D_CAIXA_LIVRE_TEXTO` | `"Calor Extremo"` |
| `GLOBO_3D_CAIXA_LIVRE_COR_BOX` | `"#0077a7"` (mesma cor do `GLOBO_3D_JET_STREAM_COR`) |

Status: **aguardando sua confirmação antes de aplicar e rodar.**

### 2026-07-04 — Re-rodada do pedido "s42, GEFS Z500 média, Europa" com o fix definitivo da costura + caixa fixa

> "rode em background no modo forecast para a pêntada de 5-10 de julho do GEFS, conforme já
> havíamos feito hoje via requerimento_video.md" (+ depois) "pode seguir em rodar o s42 com a
> última configuração que eu havia te passado. Incluindo a caixa livre ficando fixa."

Mesma configuração do pedido "s42, GEFS Z500 média, Europa, jato 5670 mgp, sem jato no PNG"
(ver acima, já com os ajustes de ícone/caixa/velocidade da rodada de correção), rerodado agora
com o fix arquitetural da costura (grade deslocada p/ 160°E) e a caixa de texto livre sempre
fixa — nenhum settings novo além do que já estava ativo (`GLOBO_3D_FADE_CAUDA=true` já cobria a
`caixa_fixa`).

| Setting | Valor |
|---|---|
| Script | `s42` |
| Período (via CLI) | `--data-inicial 2026-07-05 --data-final 2026-07-10` |
| `GLOBO_3D_VARIAVEIS_S42` | `["z500_abs"]` |
| Modelo | `RUN_GEFS = true`, `RUN_GFS = false`, `FORECAST_INIT = "2026-07-04"` |
| `GLOBO_3D_JET_STREAM_NIVEL` | `5670` mgp |
| `GLOBO_3D_JET_STREAM_VELOCIDADE` | `2.0` |
| `GLOBO_3D_JATO_PNG` | `false` |
| `GLOBO_3D_MP4_MEDIA_FIXA` | `true` |
| `GLOBO_3D_JATO_MP4_CAUDA_SEG` | `15.0` (duração do vídeo inteiro, com media fixa) |
| `GLOBO_3D_ICONES_PRESSAO` | `[{tipo="HIGH_HN", lat=48.0, lon=-10.0, velocidade=2.0, tamanho_deg=13.0}]` |
| `GLOBO_3D_CAIXA_LIVRE_LAT` / `_LON` | `38.0` / `-10.0` |
| `GLOBO_3D_CAIXA_LIVRE_TEXTO` | `"Calor extremo persiste na região"` |
| `TITULO_THE_WEATHER_CHANNEL` | `"Heat dome setup"` |
| `GLOBO_3D_FADE_CAUDA` | `true` (já ativo — agora também mantém a caixa livre FIXA, sem esmaecer) |
| `GLOBO_3D_BG_CACHE` | `true` (deixado ligado — o bug de caixa sumindo no cache foi corrigido em código, não precisa mais desligar) |

Status: **concluído** — MP4 (125 frames + 15s efetivos), PNG e GIF gerados em
`Saida/s42_GLOBO_WEATHER_CHANNEL/FORECAST/GEFS/` (800.6s). Confirmado visualmente (frames 0, 60
e 124 do MP4): título "Heat dome setup", caixa de datas "05–10/07" (sem ano), "JET STREAM"
perfeitamente legível em todas as repetições do anel sem nenhum corte na costura, ícone HIGH
maior no lugar certo, caixa "Calor extremo persiste na região" **fixa desde o frame 0** (sem
esmaecer) até o final. `settings.local.toml` segue com esta configuração ativa (ainda não
revertido — aguardando indicação de próximo passo).

### 2026-07-04 — Sem fade-in pro jato/ícone, 2 caixas novas (Groenlândia/Rússia) e ícone/jato em novo azul

> "Você ainda manteve o Jato e a Alta pressão esmaecendo no vídeo, e não precisa! Todos ícones
> sobrepostos aos mapas precisam estar direto lá, sem ter o efeito fade-in! [...] quero adicionar
> mais duas caixas livres de texto no mapa. Uma vai estar no sul da Groenlândia na cor #14223d
> escrito em branco 'Vale no ocidente', centrada em 55.63°N, 49.39°W. A outra será idêntica (cor
> e texto 'Vale no oriente') e posicionada em 55.63°N, 49.39°E." (+ depois) "sobre os ícones, se
> vc puder mantenha os dois tons de azul" / "#09519d seria esse tom de azul para os sistemas de
> alta pressão do HS e HN" / "gerando uma versao duplicada com o azul mais escuro" / "rode
> novamente o s42 considerando a alta do hemisfério norte na cor azul nova e tb já pegando a nova
> cor da corrente de jato que defini em settings local".

**Mudanças de código necessárias** (não é só settings):
1. **`caixa_fixa` desacoplada de `GLOBO_3D_FADE_CAUDA`**: antes a caixa só ficava fixa (sem
   esmaecer) quando `FADE_CAUDA=true` — mas agora o pedido é jato/ícones aparecerem DIRETO
   (`FADE_CAUDA=false`, sem nenhum esmaecimento) E a caixa continuar fixa ao mesmo tempo. Como os
   dois estavam amarrados na mesma flag, `_caixa_fixa` virou incondicional pro s42
   (`bool(script_id == 's42')`), independente do valor de `FADE_CAUDA` — que agora só controla
   jato/ícones.
2. **Suporte a MÚLTIPLAS caixas de texto livres**: o mecanismo só suportava uma (`GLOBO_3D_CAIXA_LIVRE_*`,
   settings singulares). Generalizado pro mesmo padrão de `GLOBO_3D_ICONES_PRESSAO` (lista):
   nova setting `GLOBO_3D_CAIXAS_LIVRES_EXTRA` (lista de dicts, mesmos campos com defaults se
   omitidos) somada à caixa única existente — `ctx['caixa_livre']` (dict único) virou
   `ctx['caixas_livres']` (lista); `_draw_caixa_livre` fatorada em `_draw_uma_caixa_livre`
   (por caixa) + um loop.
3. **Recolorização dos ícones de alta pressão**: os GIFs (`Entrada/icones_pressao/*.gif`) são
   assets prontos, sem script gerador — não dá pra "regenerar" com outra cor do zero. Nova função
   `_recolor_icon_frames` reconverte só o MATIZ (hue) dos pixels azuis pra uma cor alvo,
   preservando saturação/valor (o sombreado/gradiente de tons do desenho original continua).
   Gerados 6 arquivos NOVOS e PERMANENTES com sufixo `_azul_novo` (`ALTA_HN`, `ALTA_HS`,
   `HIGH_HN`, `HIGH_HS`, `INTERROGACAO_AZUL_HORARIO`, `INTERROGACAO_AZUL_ANTIHORARIO`) na cor
   `#09519d` — os originais foram mantidos intocados. Nova setting `GLOBO_3D_ICONE_ALTA_COR`
   também criada (recolore em tempo de execução, sem precisar gerar arquivo novo), mas não usada
   neste pedido — preferiu-se referenciar o arquivo já recolorido direto via `tipo`.

| Setting | Valor |
|---|---|
| `GLOBO_3D_FADE_CAUDA` | `true` → `false` (jato/ícone aparecem direto, sem esmaecer) |
| `GLOBO_3D_ICONES_PRESSAO` | `tipo`: `"HIGH_HN"` → `"HIGH_HN_azul_novo"` (usa o GIF já recolorido) |
| `GLOBO_3D_JET_STREAM_COR` | `#0077a7` → `#09519d` (definido pelo usuário direto no settings.local.toml) |
| `GLOBO_3D_CAIXAS_LIVRES_EXTRA` | `[{lat=55.63, lon=-49.39, texto="Vale no ocidente", cor_box="#14223d", cor_texto="white"}, {lat=55.63, lon=49.39, texto="Vale no oriente", cor_box="#14223d", cor_texto="white"}]` |
| (demais settings) | inalteradas — GEFS, z500_abs, câmera Europa, `MP4_MEDIA_FIXA`, ícone/caixa "Calor extremo" |

Status: **concluído** — MP4 (125 frames), PNG e GIF regenerados em
`Saida/s42_GLOBO_WEATHER_CHANNEL/FORECAST/GEFS/` (627.1s). Confirmado visualmente (frames 0, 60,
124 do MP4 + PNG): jato, ícone e as 3 caixas de texto (incluindo as 2 novas) **totalmente
visíveis já no frame 0**, sem nenhum fade-in; ícone e corrente de jato no novo tom de azul
`#09519d` (mais escuro, com o gradiente de sombreado original preservado); "Vale no ocidente" e
"Vale no oriente" nas posições corretas. `settings.local.toml` segue com esta configuração ativa.

### 2026-07-06 — s42, GFS Z(T850) anomalia média, EUA, isolinhas Z250 abs pretas, jato HN

> "pegue a previsão do GFS das 00 UTC de hoje e faça o campo médio da anomalia da temperatura do
> ar em 850 hPa e plote com as isolinhas da altura geopotencial absoluta em 250 hPa na cor preta.
> Período DATA_INICIAL 2026-07-07 / DATA_FINAL 2026-07-21, área voltada para os Estados Unidos.
> Caixa azul de título: 'Padrão médio'. Caixa cinza: o período. Ligue tb a corrente de jato."

| Setting | Valor |
|---|---|
| Script | `s42` |
| Período (via CLI) | `--data-inicial 2026-07-07 --data-final 2026-07-21` |
| `GLOBO_3D_VARIAVEIS_S42` | `["tmp850_anom"]` (s42 aceita qualquer ficha, não só absolutas) |
| Modelo | `RUN_GFS = true` (demais false), `RODADA = "00"`, `FORECAST_INIT = "2026-07-06"` (GFS 00Z de hoje) |
| `GLOBO_3D_ISOL_HGT250_ABS` | `true` (isolinhas de Z250 absoluto sobre o campo de T850) |
| `GLOBO_3D_ISOL_HGT250_COR` | `"black"` (**novo** — isolinhas pretas; default é `#666666`) |
| `GLOBO_3D_ISOL_HGT250_INTERVALO` | `60` mgp |
| `GLOBO_3D_GE_LON/LAT_INICIAL/FINAL` | `-98.0` / `39.0` (EUA — câmera fixa, sem voo) |
| `GLOBO_3D_JATO` / `GLOBO_3D_JET_STREAM` | `true` / `true` |
| `GLOBO_3D_JATO_HEMISFERIO_NORTE` / `_SUL` | `true` / `false` (EUA = HN) |
| `GLOBO_3D_JET_STREAM_NIVEL` | `10700` mgp (mantido; confirmado com o usuário) |
| `GLOBO_3D_SO_CREDITO` | `true` (ativa a caixa azul título + cinza data do s42) |
| `TITULO_THE_WEATHER_CHANNEL` | `"Padrão médio"` (caixa azul; a cinza mostra o período automaticamente) |

**Mudanças de código necessárias** (não é só settings):
1. **Isolinhas de Z250 absoluto sobre QUALQUER campo** (antes só nas fichas nativas de Z250):
   `GLOBO_3D_ISOL_HGT250_ABS` agora reconstrói o Z250 absoluto dedicado (anom z250 + clim,
   reamostrado p/ a grade do shaded) e traça sobre `tmp850_anom` etc. — reaproveita a
   infraestrutura da corrente de jato (`globo_3d_anim.py`).
2. **Nova setting `GLOBO_3D_ISOL_HGT250_COR`** (default `#666666`): cor dessas isolinhas — este
   pedido usa `"black"`.

Status: **concluído** — MP4 + PNG + GIF gerados em `Saida/s42_GLOBO_WEATHER_CHANNEL/FORECAST/GFS/`
(632.4s). Confirmado visualmente no `s42_tmp850_anom_media.png`: área EUA/América do Norte,
anomalia de T850 média (calor no oeste dos EUA), **isolinhas de Z250 absoluto em preto**, corrente
de jato "JET STREAM" no HN, caixa azul "Padrão médio" + cinza "Jul 7–21". GFS init 2026-07-06 00Z.
`settings.local.toml` já revertido ao estado anterior a este pedido.

#### Re-rodada 2026-07-06 — ícone de alta pressão (Colorado) a partir do dia 12, data até dia 19, cauda 9s

> "A partir do dia 12, coloque o ícone da alta pressão HIGH_HN.gif esmaecendo sobre 39.75°N,
> 107.10°W. Pare a data no dia 19 ao invés de 21 de julho. Mantenha a animação do jato e da alta
> pressão no último tempo do vídeo (dia 19) por ~9 segundos."

| Setting | Valor |
|---|---|
| Período (via CLI) | `--data-inicial 2026-07-07 --data-final 2026-07-19` (data para no dia 19) |
| `GLOBO_3D_ICONES_PRESSAO` | `[{tipo="HIGH_HN", lat=39.75, lon=-107.10, velocidade=2.0, tamanho_deg=12.0, sombra=true, fade_in=true, fade_inicio=0.19, fade_duracao=0.08}]` |
| `GLOBO_3D_JATO_MP4_CAUDA_SEG` | `7.0` → `9.0` (cauda: campo congela no dia 19, jato+alta seguem animando ~9s) |
| (demais settings) | iguais ao pedido acima (GFS 00Z hoje, EUA, isolinhas Z250 pretas, jato HN, "Padrão médio") |

Cálculo do `fade_inicio`: 13 dias → base 61 frames + cauda 72 = 133 frames total. Dia 12 (índice 5)
= frame 25 → fração 25/132 ≈ **0.19**.

**Mudança de código necessária** (não é só settings): o fade-in do ícone é medido pela fração da
linha do tempo do MP4 (`_prog = f/(total_frames-1)`) — no PNG estático (`total_frames=1`, `f=0`) o
alpha zerava e o ícone sumia; no GIF (loop do campo médio) oscilava. Novo flag
`ctx['saida_estatica'] = estatico or gif` faz `_draw_icones_pressao` PULAR o fade nessas saídas
(ícone em alpha cheio, ainda girando no GIF); só o MP4 anima o fade. Vale para qualquer ícone em
qualquer globo (`globo_3d_anim.py`).

Status: **concluído** — MP4 (133 frames: 61 animação + 72 cauda de 9s) + PNG + GIF em
`Saida/s42_GLOBO_WEATHER_CHANNEL/FORECAST/GFS/` (924.4s). Verificado nos frames do MP4: Jul 10 SEM
ícone (antes do dia 12), Jul 16 COM ícone (fade completo), Jul 19 (cauda) campo congelado + jato/alta
animando; PNG da média e GIF com o ícone em alpha cheio (caixa cinza "Jul 7–19").
`settings.local.toml` já revertido ao estado anterior a este pedido.

### 2026-07-07 — s42 anomalia de T850 EUA: paleta TWC, fundo blue marble+oceano, litoral vetorial

> Sequência de ajustes (mesmo vídeo, iterado): paleta "The Weather Channel" com branco no centro;
> escala ±10 °C; fundo do s42 = oceano `#0b5292` + continente blue marble; temperatura só nos
> continentes com litoral liso (recorte vetorial); isolinhas de Z250 em branco; divisas pretas;
> ícone de alta do HN e caixa de texto na cor do jato `#1787ad`; caixa "Padrão altamente
> amplificado" em 44°N/89.82°W esmaecendo no dia 19.

| Setting | Valor |
|---|---|
| Script / período | `s42` · `--data-inicial 2026-07-07 --data-final 2026-07-19` · GFS 00Z (`FORECAST_INIT` do dia) |
| `GLOBO_3D_VARIAVEIS_S42` | `["tmp850_anom"]` |
| Paleta (master) | `temp_anom_the_weather_channel` (TWC + branco no centro), escala **±10 °C** |
| Fundo (master, s42) | `GLOBO_3D_BLUE_MARBLE=true` + `GLOBO_3D_COR_OCEANO="#0b5292"` (continente=satélite, oceano=cor) |
| `GLOBO_3D_MASCARA_OCEANO_TMP850_ANOM` | `true` (temperatura só no continente, litoral vetorial liso) |
| `GLOBO_3D_TRANSP_ATE_TMP850_ANOM` | `0.0` (sem transparência central) |
| `GLOBO_3D_ISOL_HGT250_ABS` / `_COR` | `true` / `"white"` (isolinhas de Z250 em branco) |
| `GLOBO_3D_COR_FRONTEIRAS_TMP850_ANOM` | `"black"` (+ `COASTLINE/BORDERS/STATES_LW` realçados) |
| `GLOBO_3D_JET_STREAM_COR` | `#1787ad` |
| Ícone de alta (HN) | `azul_original_HIGH_HN` (recolorido `#1787ad`) em 39.75°N/107.10°W, fade a partir do dia 12 |
| Caixa livre | `"Padrão altamente amplificado"` em 44°N/89.82°W, cor `#1787ad`, borda branca 1.0, esmaece no dia 19 |

**Mudanças de código/assets** (permanentes, versionadas): recorte vetorial da costa (`_land_clip_path`),
fundo oceano+blue-marble no s42, `GLOBO_3D_TRANSP_ATE_<VAR>`, `GLOBO_3D_COR_FRONTEIRAS_<VAR>`
generalizado, `GLOBO_3D_VMAX_TMP850_ANOM`, paleta TWC no master, 4 ícones `azul_original_*.gif`.

Status: **concluído** — o FUNDO do s42 (oceano `#0b5292` + blue marble) virou padrão no master; o
resto do render (campo/câmera/caixa/ícone) vive no `settings.local.toml` (por vídeo), ajustado por
pedido conforme este arquivo.
