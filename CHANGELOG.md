# Changelog

Todas as mudancas notaveis neste projeto serao documentadas neste arquivo.

O formato segue o [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/).

---

## [Unreleased]

### Corrigido

- `app/src/uteis/downloaders_wind200.py`: `NetCDF: HDF error` / `Can't open HDF5 attribute` ao baixar ERA5 U/V 200 hPa em paralelo (`ensure_era5_uv200_for_period`, `max_parallel=2`) — a lib HDF5 não é thread-safe e a validação/merge concorrentes de meses corrompiam o I/O e abortavam o s29. Adicionado lock de módulo (`_HDF5_IO_LOCK`) serializando todo I/O netCDF (leitura em `_extract_time_index_from_file` e merge `to_netcdf`); downloads de rede seguem paralelos. Corrigido também bug latente: `merged.load()` antes de fechar os datasets-fonte (o merge lazy referenciava arquivos já fechados)
- `scripts/s27_hovmoller_enso.py` e `scripts/s28_hovmoller_iod.py`: `AlignmentError: cannot align objects with join='override'` ao concatenar ERA5 (181 lat, 1°) com GDAS (721 lat, 0.25°) — grades com tamanhos diferentes; agora interpola arquivos GDAS para a grade de referência ERA5 antes do `xr.concat`
- `scripts/s29_walker_cell_mundo.py`: colorbar SSTA agora usa colorscale idêntico ao piso (`sst_colors` mapeado linearmente de 0→1 com `cmin=-5/cmax=+5`), eliminando a dessincronização de cores causada pelo range estendido da versão anterior; `script_version='5.43'`
- `scripts/s16_wnd250_zonal_anom_div.py`: remove contornos (`ax.contour`) sobrepostos ao shaded de magnitude de vento 250 hPa em todas as modalidades de plotagem; `script_version='1.2'`
- `scripts/s16_wnd250_zonal_anom_div.py`: paleta do modo `_pos` (anomalia positiva) alterada de amarelos/laranjas/vermelhos para nova paleta azul-rosa-cinza (`CMAP_POS_COLORS`); `script_version='1.3'`
- `scripts/s16_wnd250_zonal_anom_div.py`: continentes pintados com `#d4d4d4` via `cfeature.LAND` (zorder=1, abaixo do shaded para não cobrir a anomalia de vento sobre terra); `script_version='1.5'`
- `scripts/s16_wnd250_zonal_anom_div.py`: adicionado modo `_mag` (4ª figura por área): magnitude vento 250 hPa (sqrt(u²+v²)) + quiver divergente onde chi200<0 + anomalia OLR negativa em verde; nova utilidade `app/src/uteis/plot_wnd_speed_250.py`; `script_version='1.6'`
- `scripts/s16_wnd250_zonal_anom_div.py`: modo `_mag` refinado — `CMAP_MAG_COLORS` alterado para paleta azul-rosa-cinza idêntica ao modo `_pos`; OLR translúcido (`alpha=0.5`); adicionadas isolinhas de altura geopotencial 250 hPa (Z250) com rótulos em "placa branca" — 9960/10200 m em azul (lw=2.0) e 10440/10680 m em vermelho (lw=2.2); novos utilitários `app/src/uteis/downloaders_z250_era5.py` e `app/src/uteis/plot_z250_mean.py`; `script_version='1.7'`
- `scripts/s16_wnd250_zonal_anom_div.py`: modo `_mag` — `LEVELS_MAG` agora começa em 25 m/s (era 10); velocidades ≤24 m/s ficam transparentes (sem preenchimento via `extend='max'`); paleta de cor aplicada somente a partir de 25 m/s; `script_version='1.8'`
- `scripts/s16_wnd250_zonal_anom_div.py`: logo da Ampere com `zorder=1100` (era 500) para ficar acima das isolinhas de altura geopotencial (`zorder=900`) e do retângulo da borda (`zorder=1000`), que antes cortavam o logo

### Modificado

- `scripts/s26_chi200_anom_div_fluxo_rossby_wave.py`: cor das setas do vento divergente alterada de `paleturquoise` para `white`
- `app/src/uteis/downloaders_gdas_omega.py`: fix "the new name 'time' conflicts" — arquivos GDAS recentes têm `time` (referência) e `valid_time` simultaneamente; agora descarta `time` antes de renomear `valid_time` → `time`

### Adicionado

- `scripts/s29_walker_cell_mundo.py`: v5.37 — `WIND_VEC_LAT_MAX`: 30° → 80° (teste — vetores U/V 850 hPa cobrem ±80°); `script_version='5.37'`
- `scripts/s29_walker_cell_mundo.py`: v5.36 — `_build_land_mesh3d`: reescrita com Delaunay + grade interior 1.5° por polígono (pontos dentro do polígono a cada 1.5°); elimina slivers (aspect ratio ~5:1 vs 300:1 anterior); pula Antártida (span >350°); `simplify(0.3)` na borda; filtro antimeridiano 90°; borda do fill agora coincide com as linhas de costa 50m; `script_version='5.36'`
- `scripts/s29_walker_cell_mundo.py`: v5.35 — `_build_cartopy_land_mask`: muda resolução 110m→50m (mesma fonte das linhas Scatter3d) para alinhar borda do preenchimento LAND_SENTINEL com as linhas pretas; `script_version='5.35'`
- `scripts/s29_walker_cell_mundo.py`: v5.34 — remove `go.Mesh3d` dos continentes (triângulos fan com aspect ratio ~300:1 geravam arestas visíveis como linhas retas no WebGL); fill retorna ao LAND_SENTINEL no ocean Surface; `_extract_lines_360`: insere None onde longitures consecutivas saltam >180° em 0–360 (fix para cruzamento do meridiano 0° gerando linha reta atravessando o mapa); `script_version='5.34'`
- `scripts/s29_walker_cell_mundo.py`: v5.33 — `_build_land_mesh3d`: substitui `scipy.spatial.Delaunay` por **fan triangulation** (centroide→aresta polígono); elimina slivers de alta razão de aspecto que cruzavam o mapa como linhas retas; usa `representative_point()` como centroide interior garantido; remove dependência de `scipy.spatial`; `script_version='5.33'`
- `scripts/s29_walker_cell_mundo.py`: v5.32 — `_build_land_mesh3d`: adiciona teste 4-pontos (centróide + 3 pontos médios de aresta via `MplPath.contains_points`) para sliver triangles em concavidades (ex: Golfo da Guiné); `script_version='5.32'`
- `scripts/s29_walker_cell_mundo.py`: v5.31 — `_build_land_mesh3d`: adiciona filtro antimeridiano (`lons_360.max()-lons_360.min() > 45`); corrige import `plotly.graph_objects as go` dentro da função (erro `go is not defined`); `script_version='5.31'`
- `scripts/s29_walker_cell_mundo.py`: v5.30 — refatora `_build_land_mesh3d`: substitui `shapely.ops.triangulate` (falhava silenciosamente) por `scipy.spatial.Delaunay` (mais robusto); adiciona `part.buffer(0)` para corrigir auto-interseções antes de simplificar; usa `poly_buf.contains(Point)` com buffer de 0.05° para não excluir triângulos limítrofes; volta ao LAND_SENTINEL no ocean Surface (sem z=NaN que gerava artefato ciano escalonado); Mesh3d em z=FLOOR_Z-10 (separação de profundidade maior)
- `scripts/s29_walker_cell_mundo.py`: v5.29 — resolve z-fighting definitivo entre ocean Surface e land Mesh3d: ocean Surface agora usa `z=NaN` para células de terra (buracos geométricos — Plotly não renderiza), Mesh3d 50m preenche esses buracos em z=FLOOR_Z-5 sem competição de profundidade; colorscale do oceano simplificado para apenas cores SST (sem LAND_SENTINEL); `_build_floor_arrays` passa a retornar `land_mask_bool`; log do número de triângulos gerados no Mesh3d
- `scripts/s29_walker_cell_mundo.py`: v5.28 — unifica fonte do fill e das linhas: `_build_land_mesh3d` migrado de 110m para 50m Natural Earth com `simplify(0.1)`; agora fill (Mesh3d) e linhas de costa (Scatter3d) partem do mesmo shapefile e coincidem; todas as linhas (costa, fronteiras, estados) alteradas para preto `rgba(0,0,0,1.0)`
- `scripts/s29_walker_cell_mundo.py`: v5.27 — substitui shading rasterizado dos continentes por `go.Mesh3d` com triangulação Delaunay dos polígonos Natural Earth 110m (`_build_land_mesh3d`); bordas do continente whitesmoke agora seguem exatamente os polígonos vetoriais (sem efeito "degrau" de grade); z-stack: oceano z=FLOOR_Z(1050) → continente mesh z=FLOOR_Z-5(1045) → linhas z=FLOOR_Z-20(1030)
- `scripts/s29_walker_cell_mundo.py`: v5.26 — grade do piso refinada de 0.5° para 0.25° (resolução nativa OISSTv2), reduzindo o efeito "degrau" na borda da máscara de terra rasterizada; HTML mais pesado (~4× mais vértices na superfície do piso)
- `scripts/s29_walker_cell_mundo.py`: v5.25 — linhas de costa/fronteiras/estados movidas para `LINE_Z = FLOOR_Z - 20` (reduz z-fighting) e coloridas de branco (`rgba(255,255,255,1)` costa, tons de cinza claro para fronteiras/estados) para máxima visibilidade tanto sobre oceano colorido quanto sobre continentes whitesmoke; largura da costa aumentada para 2
- `scripts/s29_walker_cell_mundo.py`: v5.24 — corrige definitivamente continentes azuis: (1) substitui máscara baseada em NaN do OISSTv2 por máscara vetorial com polígonos Natural Earth 110m via `_build_cartopy_land_mask()` (mesma fonte que `cfeature.LAND` do cartopy) — cobre células costeiras com SST extrapolado que OISSTv2 não marca como NaN; (2) aumenta zona sentinela de 0.17% para 50% do colorscale (`LAND_SENTINEL = sst_cmin - sst_range`), eliminando falha de quantização do Plotly onde a banda anterior era menor que 1 passo de 256 cores
- `scripts/s29_walker_cell_mundo.py`: v5.23 — cópia automática do HTML gerado para `C:\Users\Pichau\Desktop\walker_cell_s29.html` (via `/mnt/c/...`) ao final da renderização em ambiente WSL; elimina a necessidade de copiar manualmente após cada execução
- `scripts/s29_walker_cell_mundo.py`: v5.22 — abordagem definitiva: substitui duas superfícies (continente+oceano) por UMA superfície única sem NaN; terra recebe valor sentinela `sst_cmin-0.01` e colorscale customizado mapeia [LAND_SENTINEL, sst_cmin) → `#f5f5f5` (whitesmoke) e [sst_cmin, sst_cmax] → cores SST normais; elimina completamente o problema de NaN rendering do Plotly 6; remove trace separado de continentes
- `scripts/s29_walker_cell_mundo.py`: v5.21 — solução definitiva para continentes azuis: usa `z=NaN` onde há terra na superfície do oceano (`z_ocean = np.where(np.isnan(ocean_sst), np.nan, FLOOR_Z)`); em Plotly, NaN no z cria **buracos geométricos** na malha da superfície — o oceano não renderiza sobre a terra, e o trace dos continentes (whitesmoke, z=FLOOR_Z) fica visível nesses buracos; continent trace adicionado primeiro, ocean trace adicionado depois com z_ocean; linhas de costa/fronteiras/estados mantidas em FLOOR_Z-1
- `scripts/s29_walker_cell_mundo.py`: v5.20 — corrige causa raiz dos continentes azuis: dataset OISSTv2 usa fill values (-9999, 1e20) nas áreas terrestres em vez de NaN; `np.isnan()` não os detectava, `land_mask` ficava todo NaN e o trace do oceano pintava a terra com cor azul (mínimo da escala SST); fix: mascarar `|sst_2d| > 50` após interpolação + log de diagnóstico `n_land`; trace dos continentes mantido em z=FLOOR_Z-0.5 (após oceano) para garantia adicional
- `scripts/s29_walker_cell_mundo.py`: v5.19 — corrige continentes azuis em Plotly 6: em Plotly 6, NaN no `surfacecolor` do trace do oceano pinta áreas terrestres com a cor mínima da escala SST (azul escuro); solução: (1) trace do oceano adicionado primeiro em z=FLOOR_Z=1050 hPa, (2) trace dos continentes adicionado depois em z=FLOOR_Z-0.5=1049.5 hPa para vencer depth buffer independente de rendering order, (3) linhas de costa/fronteiras/estados em z=FLOOR_Z-1.0=1049 hPa para ficarem acima dos continentes; `#f5f5f5` + `cmin=0.0, cmax=1.0` mantidos; fix `ADM0_A3` para estados do Brasil (61 registros); resolução das linhas de costa elevada para 50m Natural Earth; adicionadas fronteiras de países (50m) e divisas estaduais do Brasil (10m) como traces `go.Scatter3d` separados; linhas de costa com cor cinza escuro `rgba(60,60,60,0.9)` (antes branco); refatoração interna com helper genérico `_extract_lines_360()` + funções dedicadas `_extract_borders_360()` e `_extract_brazil_states_360()`; `script_version='5.17'`
- `scripts/s29_walker_cell_mundo.py`: v5.3 — substitui `go.Volume` por `go.Isosurface` (mesmo traço do script de referência, mais robusto para grades esparsas); muda coordenada z do volume e do piso de pressão negada (-hPa) para pressão positiva (+hPa); eixo z agora usa `type='log'` + `autorange='reversed'` (1000 hPa embaixo, 100 hPa em cima); `opacity=0.5`, `surface_count=7`; `script_version='5.3'`
- `scripts/s29_walker_cell_mundo.py`: novo script — Célula de Walker v5.0; visualização 3D com **Plotly** (`go.Volume` + `go.Surface` + `go.Scatter3d`): piso com SSTA OISSTv2 (oceano) + continentes verdes (`go.Surface`), linhas de costa em coordenadas 0-360 (`go.Scatter3d`), volume 3D de anomalia de omega com `go.Volume` (isosuperfícies semi-transparentes empilhadas, opacity=0.12, surface_count=21, NaN mask para valores near-zero) e taper gaussiano latitudinal criando efeito de "colunas luminosas"; Gaussian blur + interpolação para 25 níveis antes da renderização; colormap azul-âmbar; câmera e aspecto 2:1:0.55 para perspectiva sul-elevada; saída HTML interativo (`include_plotlyjs='cdn'`) + PNG via kaleido; `script_version='5.0'`; saída em `Saida/s29_WALKER_CELL_MUNDO/`
- `pyproject.toml`: `plotly>=6.0.0,<7` e `kaleido>=1.0.0` adicionados às dependências
- `app/src/uteis/downloaders_omega_era5.py`: novo downloader ERA5 `vertical_velocity` em múltiplos níveis de pressão; cache mensal; validação de período; protegido com `_HDF5_LOCK`
- `app/src/uteis/downloaders_gdas_omega.py`: novo downloader GDAS omega multi-nível via NOMADS filter (`var_VVEL`, múltiplos `lev_XXX_mb`); padrão idêntico ao `downloaders_gdas_uv850.py`; salva NetCDF diário com dim `pressure_level`
- `app/src/uteis/clim_PSL_omega_multilevel.py`: novo downloader PSL omega multi-nível; abre uma sessão Playwright única e itera os níveis; mescla os arquivos individuais em um único netCDF com dim `level` (`clim_omega_ML_*`); cache em dois níveis (por nível + arquivo mesclado)
- `app/cli/run_script.py`: s29 registrado no SCRIPTS dict
- `app/settings/settings.toml`: `RUN_S29 = true` adicionado
- `settings.local.example.toml`: `# RUN_S29 = false` adicionado

- `scripts/s28_hovmoller_iod.py`: novo script — Hovmöller IOD; cópia do s27 com domínio 5S-5N, 50E-103E (Oceano Índico, sem cruzamento do antimeridiano); nomes de arquivo `hovmoller_sst_u850_INDICO_*`; saída em `Saida/s28_MONITORAMENTO_IOD_HOVMOLLER/`
- `app/cli/run_script.py`: s28 registrado no SCRIPTS dict
- `app/settings/settings.toml`: `RUN_S28 = true` adicionado
- `settings.local.example.toml`: `# RUN_S28 = false` adicionado

- `scripts/s27_hovmoller_enso.py`: novo script — Hovmöller de Anomalia de TSM (OISSTv2, 0.25°, idêntico ao s11) e Vento Zonal 850 hPa (ERA5/GDAS híbrido + climatologia PSL u-zonal via `clim_PSL_wnd_zonal_850.get_clim_wnd_zonal_850_path`); SST shaded com `LST_ANOM_CORRETA`/`LST_SSTA_NEW_GREC`; isolinhas U850 anom (azul=negativo, vermelho=positivo, limiar 3 m/s); domínio 5S-5N, 160E-80W (cruza antimeridiano); salva PNG + NetCDF em `Saida/s27_MONITORAMENTO_ENSO_HOVMOLLER/`
- `app/cli/run_script.py`: s27 registrado no SCRIPTS dict
- `app/settings/settings.toml`: `RUN_S27 = true` adicionado
- `settings.local.example.toml`: `# RUN_S27 = false` adicionado

- `scripts/s26_chi200_anom_div_fluxo_rossby_wave.py`: novo script — chi200 shaded (verde→bege→marrom, `LinearSegmentedColormap`, ±60×10⁵ m²/s) + contornos pretos de anomalia de geopotencial 250 hPa + vetores WAF (Takaya & Nakamura 2001) + vento divergente 200 hPa onde chi200<0 entre -20° e 20° (paleturquoise com contorno preto); suporte a defasagem interativa entre chi200+vento divergente e geopotencial+WAF; cópia estrutural do s25 sem o download de OLR
- `app/cli/run_script.py`: s26 registrado no SCRIPTS dict
- `app/settings/settings.toml`: `RUN_S26 = true` adicionado
- `settings.local.example.toml`: `# RUN_S26 = false` adicionado

- `scripts/s23_sst_todas_areas_correntes_marinhas.py`: novo script — TSM shaded (OISSTv2) + streamlines de correntes marinhas de superfície (CMEMS GLOBAL_ANALYSISFORECAST_PHY_001_024, dataset `cmems_mod_glo_phy_cur_anfc_0.083deg_P1D-m`); OLR e vento 1000 hPa removidos; `copernicusmarine` adicionado ao `pyproject.toml`; credenciais `CMEMS_USERNAME`/`CMEMS_PASSWORD` adicionadas ao `.secrets_example.toml`
- `app/cli/run_script.py`: s23 registrado no SCRIPTS dict
- `app/settings/settings.toml`: `RUN_S23 = true` e `LST_AREAS_S23` adicionados

### Modificado

- `scripts/s13_sst_todas_areas.py` → renomeado para `s13_sst_todas_areas_vento_medio_1000.py`; `app/cli/run_script.py`: module e description atualizados

### Adicionado

- `scripts/s21_wnd850_anom_chi200_anom.py`: novo script — CHI200 shaded (sem contorno) + streamlines do vento anômalo 850 hPa (dimgray) + vento divergente 200 hPa apenas onde chi<0 entre -20° e 20°; áreas iguais ao s20; registrado no SCRIPTS dict e em `LST_AREAS_S21`
- `app/cli/run_script.py`: s21 registrado no SCRIPTS dict
- `app/settings/settings.toml`: `RUN_S21 = true` e `LST_AREAS_S21` adicionados
- `scripts/s07_ssta_vento850_anom.py`: novo script — anomalia de TSM (OISSTv2) sobreposta com vetores de vento anômalo 850 hPa (pipeline ERA5/GDAS + climatologia PSL); 26 áreas, sem blue marble, geração sequencial
- `app/cli/run_script.py`: s07 registrado no SCRIPTS dict e em `list_scripts()`
- `app/settings/settings.toml`: flag `RUN_S07 = true`

### Corrigido

- `scripts/s06_ssta_todas_areas.py`: substituída geração paralela (`ProcessPoolExecutor`) por sequencial — `ProcessPoolExecutor` com fork copiava o array blue_marble (96 MB) para cada worker, estourando a RAM do WSL ao rodar `--all`; qualidade da imagem restaurada para resolução original (8192×4096)

### Modificado

- `scripts/s21_wnd850_anom_chi200_anom.py` e `s22_wnd850_anom_olr_anom_div.py`: boxes oceanográficos (atlantico_tropical, iod, plot_box genérico) elevados de `zorder=100` para `zorder=300` — ficam acima do vento divergente (zorder=200) e não são cortados pelos vetores; `script_version` s21→1.5, s22→1.1
- `scripts/s22_wnd850_anom_olr_anom_div.py`: novo script — OLR shaded (BrBG_r, ±40 W/m²) + streamlines vento anômalo 850 hPa (black) + vento divergente 200 hPa onde chi200 < 0 e ±20° (white/black stroke); mesmas áreas e configurações de streamplot do s21; registrado no SCRIPTS dict e em `LST_AREAS_S22`
- `scripts/s21_wnd850_anom_chi200_anom.py`: adicionado `add_cyclic_point` ao vento 850 hPa — corrige descontinuidade das streamlines em 180° em áreas com extent cruzando o antimeridiano (ex: MJO); `script_version` → `1.4`
- `scripts/s21_wnd850_anom_chi200_anom.py`: vento divergente com preenchimento branco e contorno preto (`path_effects.withStroke`); `zorder` elevado para 200 (acima dos contornos de costa/fronteiras em 100); `script_version` → `1.3`
- `scripts/s21_wnd850_anom_chi200_anom.py`: `linewidth` das streamlines agora configurável por área via `_STREAMPLOT_LINEWIDTH` (padrão 0.5 em todas); cor das streamlines alterada para `'black'`; `script_version` → `1.2`
- `scripts/s21_wnd850_anom_chi200_anom.py`: densidade das streamlines aumentada para 2.5 em tropico/psa/pacifico_leste_america_sul/mjo/hemisferio_sul/globo/globo_3d/enso/america_sul_zom_out e para 3.0 em atlantico_tropical/amo; streamlines habilitadas em `globo_3d` (remoção do guard `if not is_polar`); `script_version` → `1.1`
- `scripts/s20_olr_wnd850_geop500_anom.py`: cor dos vetores de vento alterada de `'k'` para `'dimgray'`; `script_version` → `1.1`
- `scripts/s15_chi200_psi200_anom.py`: adicionado quiver do vento divergente restrito a chi200 < 0 na faixa -20° a 20° (`QUIVER_STEP=4`, `QUIVER_SCALE=80`); removidos contornos brancos do shaded de CHI200; `script_version` → `1.2`
- `scripts/s16_wnd250_zonal_anom_div.py`: quiver do vento divergente restrito às regiões com chi200 < 0 — carregado campo escalar `chi_anom` do `chi200.nc`, amostrado no mesmo step do quiver e aplicada máscara `chi >= 0`; `script_version` incrementado para `1.1`
- `scripts/s15_chi200_psi200_anom.py`: removida plotagem do vento divergente (quiver) — constantes `QUIVER_DEFAULTS`, `QUIVER_POR_AREA`, `UCHI_CANDIDATES`, `VCHI_CANDIDATES` e funções `_add_cyclic_uv`, `_prepare_quiver_masked`, `_get_quiver_config` deletadas; `script_version` incrementado para `1.1`
- `app/src/uteis/plot_geop250.py`: substituída concatenação em memória por acumulador streaming (`_compute_period_mean_streaming`) — processa um arquivo mensal por vez, evitando estouro de RAM no WSL para períodos longos (3–4 meses de dados globais ERA5/GDAS)
- `app/src/uteis/clim_PSL_psi200.py`: reescrito com cache por período MM-DD, logger, path via settings e bug de tupla morta corrigido (não utilizado no pipeline — s03 reutiliza `clim_PSL_wnd200.py`)
- `app/src/uteis/clim_PSL_wnd850.py`: criado pelo usuário; corrigido erro na mensagem de exceção (200→850)
- `app/src/uteis/plot_olr_wind250_anom.py`: migrado para pipeline híbrido ERA5/GDAS (250 hPa) + PSL clim + streaming; removidas importações quebradas de plot_chi200.py
- `app/src/uteis/plot_olr_wind850_anom.py`: criado seguindo mesmo padrão do 250 hPa, para anomalia de vento 850 hPa
- `scripts/s06_olr_wind_250_850_anom.py`: renomeado de s06_olr_wind250_anom.py; adicionado pipeline 850 hPa com figuras com sufixo `_250hPa.png` e `_850hPa.png`; OLR download com aria2c; try/except removido
- `app/cli/run_script.py`: s06 atualizado para novo módulo e support_files limpos

### Adicionado (s06/850hPa)

- `app/src/uteis/downloaders_wind850.py`: downloader ERA5 u/v em 850 hPa
- `app/src/uteis/downloaders_gdas_uv250.py`: downloader GDAS u/v 250mb via NOMADS
- `app/src/uteis/downloaders_gdas_uv850.py`: downloader GDAS u/v 850mb via NOMADS
- `app/src/uteis/clim_PSL_wnd250.py`: criado pelo usuário; corrigido erro na mensagem de exceção (200→250)
- `app/src/uteis/plot_rossby_waf.py`: migrado para pipeline híbrido ERA5/GDAS (hgt 250 hPa) + PSL hgt+uv 250mb + streaming accumulator; removida dependência de funções do plot_psi200.py
- `scripts/s04_fluxo_rossby_wave.py`: `script_version` incrementado para 2.0
- `app/cli/run_script.py`: removidas as 3 climatologias locais dos `support_files` do s04
- `app/src/uteis/plot_psi200.py`: migrado para pipeline híbrido ERA5 (200 hPa)/GDAS + climatologia PSL u/v 200mb (via `clim_PSL_wnd200.py`) + streaming accumulator; anomalia calculada de u/v primeiro, depois psi — mesmo padrão do s02
- `scripts/s03_psi200_anom.py`: `script_version` incrementado para 2.0
- `app/cli/run_script.py`: removidas climatologias uwnd/vwnd250 dos `support_files` do s03
- `app/src/uteis/clim_PSL_wnd200.py`: reescrito com cache por período MM-DD, logger, path via settings e uma única sessão Playwright para u e v
- `app/src/uteis/plot_chi200.py`: migrado para pipeline híbrido ERA5 (200 hPa)/GDAS + climatologia PSL + streaming accumulator (mesmo padrão do s01)
- `scripts/s02_chi200_anom.py`: removido try/except genérico em torno de `plot_chi200()`; `script_version` incrementado para 2.0

### Adicionado (s02)

- `app/src/uteis/downloaders_wind200.py`: downloader ERA5 u/v em **200 hPa** (separado do wind250 que continua em uso por s03/s04/s06)
- `app/src/uteis/downloaders_gdas_uv200.py`: downloader GDAS u/v 200mb via NOMADS Grib Filter (padrão do downloaders_gdas_hgt250)

### Removido

- Scripts de teste legados da raiz: `teste.py`, `teste00_dados_vento_SEMOP.py`, `teste01_dados_anom_geop.py`, `teste02_dados_anom_chi200.py`
- Módulos legados da API Ampere: `app/common/ampere_api_client.py`, `app/config.py`, `app/src/reanalise.py`, `app/src/prompt_data.py`, `app/src/gridlines.py`
- `AMPERE_API_BASE_URL` do `settings.toml` e bloco `AMPERE_API_USERNAME/PASSWORD` do `settings.local.example.toml`

### Adicionado

- `--force-rerun` no `run_script.py` — invalida o cache de um script especifico antes de executar (ex: `uv run python run_script.py s05 --force-rerun`), forcando o reprocessamento sem mexer no cache dos demais. Portado do `main.py` legado
- Script s06: Anomalia de OLR + linhas de corrente da anomalia do vento em 250 hPa (OLR do PSL/NOAA + vento ERA5/CDS)
- `app/src/uteis/plot_olr_wind250_anom.py` — modulo de processamento que reutiliza downloader de vento 250 hPa e climatologias uwnd/vwnd para calcular anomalia
- `RUN_S06` — flag de execucao do s06 no settings.toml
- Script s05: Anomalia de OLR (Outgoing Longwave Radiation) a partir de dados CPC Blended OLR do PSL/NOAA
- Download integrado no s05 (HTTP simples via `download_with_progress`, sem necessidade de downloader separado)
- `RUN_S05` — flag de execucao do s05 no settings.toml
- Script s04: Rossby Wave Activity Flux (Takaya & Nakamura 2001) com vetores WAF sobre anomalia de hgt 250 hPa
- `app/src/uteis/plot_rossby_waf.py` — modulo de processamento que reutiliza downloaders existentes (hgt250 + climatologias uwnd/vwnd) e calcula WAF via pacote `tnflux`
- Dependencia `tnflux` adicionada ao projeto
- `RUN_S04` — flag de execucao do s04 no settings.toml
- Script s02: anomalia de CHI200 (velocity potential) com vetores de vento divergente
- `app/src/uteis/plot_chi200.py` — modulo de download, processamento e calculo de anomalia CHI200 via scipy (Helmholtz decomposition)
- `app/src/uteis/downloaders_wind250.py` — downloader de u/v 250 hPa (ERA5) em formato NetCDF
- `FILE_CLIMATOLOGIA_UWND250` e `FILE_CLIMATOLOGIA_VWND250` — settings para climatologias de vento 250 hPa
- Climatologias uwnd250/vwnd250 adicionadas como `support_files` do s02 (download automatico via SFTP)
- `RUN_S02` — flag de execucao do s02 no settings.toml
- Interpolacao automatica da climatologia de vento para o grid do ERA5 (mesmo padrao do s01/geop250)
- Configuracao de quiver por area via `CHI200_QUIVER_POR_AREA` no settings (scale, width, step, min_mag, headwidth, headlength, color)
- Boxes ENSO (Nino 1+2, 3, 3.4, 4), IOD e TSA/TNA no s02

### Alterado

- Mensagem de interrupcao por Ctrl+C (`KeyboardInterrupt` no `@friendly_errors`) agora exibe um bloco mais visivel informando que o script foi cancelado
- `downloaders_wind250.py`: download de u/v 250 hPa migrado de GRIB para **NetCDF** — elimina dependencia de ecCodes C library (`cfgrib`)
- `downloaders_wind250.py`: removidos parametros `area` e `grid` — download global sem filtro (padrao hgt250)
- `downloaders_wind250.py`: download de u e v agora em **requisicoes CDS separadas em paralelo** — cada variavel baixa independentemente via ThreadPoolExecutor, depois mescla com `xr.merge()` (melhora significativa de velocidade)

### Removido

- `main.py` — entry point legado removido. So conhecia o s00 e usava imports proibidos pelo CLAUDE.md (`app.config`, `app.common.logger`). O entry point unico passa a ser `run_script.py` → `app/cli/run_script.py`, que ja registra s00–s06 no dict `SCRIPTS`. As funcionalidades uteis do `main.py` foram portadas (`--force-rerun`); o tratamento de Ctrl+C ja existia no `@friendly_errors`

### Corrigido

- s05/s06: arquivo `olr.day.anom.nc` (PSL/NOAA, atualizado diariamente) nao era re-baixado quando ja existia localmente, gerando mapas em branco ou com dados incorretos quando o periodo solicitado nao estava coberto pelo arquivo local cacheado. Agora o script consulta o periodo solicitado (DATA_INICIAL/DATA_FINAL) contra o arquivo local e so re-baixa se necessario; se mesmo apos download o periodo nao estiver disponivel, aborta com mensagem clara mostrando a primeira/ultima data disponivel no arquivo
- `dataset_utils.arquivo_cobre_periodo`: novo helper que verifica se um arquivo NetCDF existe e cobre completamente um intervalo `[start_date, end_date]`. Util para evitar re-downloads desnecessarios de datasets atualizados periodicamente (PSL/NOAA OLR diario, etc.)
- `dataset_utils.validar_cobertura_temporal`: novo helper que valida cobertura temporal de um dataset e levanta `RuntimeError` com mensagem clara contendo as datas disponiveis se o periodo solicitado nao estiver coberto ou tiver gaps
- `download_helper.download_with_progress`: novo parametro `max_age_hours` (Optional[float]) — checa mtime do arquivo local e re-baixa se exceder o limite. Util para datasets atualizados diariamente
- Logo do s02 deslocado para cima — substituido `fig.add_axes` + `get_position` por `AnchoredOffsetbox` (ancorado ao axes, sobrevive a `bbox_inches='tight'`)
- Faixa branca na plotagem do s01 na divisa dos hemisferios — removida conversao 0-360 e sort manual de lon, usando `sortby` + `add_cyclic_point` direto no DataArray (mesmo padrao do s00)

### Adicionado

- `app/src/uteis/plot_geop250.py` — modulo de download, processamento e calculo de anomalia geopotencial 250 hPa (ERA5)
- `FILE_CLIMATOLOGIA_GEOP250` e `REMOTE_CLIMATOLOGIA_GEOP250` — settings para climatologia geopotencial 250 hPa
- Climatologia geop250 adicionada como `support_files` do s01 (download automatico via SFTP)
- Docstrings descritivas nos scripts (`s00`, `s01`) com finalidade, dados de entrada/saida, datas de criacao e atualizacao
- `SCRIPT_ID`, `SCRIPT_NAME`, `SCRIPT_DESC` derivados de `__file__` e `__doc__` — logger, cache e header dinamicos
- `LST_AREAS_S01` no `settings.local.toml` — lista de areas de plotagem do s01 configuravel sem mexer no git
- `.hooks/` — diretorio para shell scripts de desenvolvimento (changelog-reminder.sh)
- **Refatoracao completa de infraestrutura** seguindo padroes dos projetos `usando_api_ampere` e `windx-automatico`
- Migracao de Poetry para **UV** como package manager
- `setup.sh` — setup automatizado com menu de ambiente (development/production/qa), copia configs, cria diretorios
- `setup.sh` — copia automatica de `.vscode/settings_exemplo.json` para `.vscode/settings.json`
- `.python-version` — Python 3.12.9 fixo
- `.env.example` — template para `ENV_FOR_DYNACONF`
- `.vscode/settings_exemplo.json` — template VSCode com icones e cores do projeto
- Modulo `app/shared/` com infraestrutura reutilizavel:
  - `singleton.py` — Metaclass Singleton
  - `settings_factory.py` — Dynaconf factory com Singleton
  - `logger.py` — LoggerService com Loguru (Singleton, silencia cdsapi/paramiko)
  - `sftp_client.py` — Cliente SFTP enxuto com context manager
- Modulo `app/cli/run_script.py` — CLI com argparse e dicionario SCRIPTS
- `run_script.py` — entry point wrapper na raiz
- Download automatico de arquivos de suporte (`support_files`) via SFTP antes da execucao de cada script
- Verificacao de arquivos obrigatorios (`required_files`) com `FileNotFoundError` claro antes da execucao
- Saida colorida no `--list` com status de cada script e seus requisitos (ANSI colors)
- Suporte a 3 ambientes: development (SFTP + DEBUG), qa, production
- `app/settings/.secrets_example.toml` — template para credenciais (KEY_CDS + SSH)
- `settings.local.example.toml` — template simplificado para datas e flags
- Secao `[qa]` no `settings.toml`
- Configuracoes de logging por ambiente (LEVEL_LOGGING, LOGGER_BACKTRACE, LOGGER_DIAGNOSE)
- README reescrito com diagramas Mermaid (fluxo geral, s00, s01, prioridade de settings)
- QUICKSTART atualizado para UV e novo CLI
- `GUIA-NOVOS-SCRIPTS.md` — guia completo para adicionar novos scripts ao projeto
- Skill Claude Code `.claude/skills/adicionar-novo-script.md` para assistencia ao adicionar scripts
- `app/shared/error_handler.py` — Handler global de excecoes com `@friendly_errors` e mapa `_ERROR_HINTS`
- Skill Claude Code `.claude/skills/tratamento-de-erros.md` para guia de tratamento de erros
- Secao "Tratamento de Erros" no README e GUIA-NOVOS-SCRIPTS.md
- `dados/` — diretorio dedicado para dados baixados do CDS (separado de `Entrada/` que mantem arquivos fixos)
- `DIR_DADOS` — nova setting para configurar diretorio de dados baixados por ambiente
- `.claude/rules/` — regras adaptadas do windx-automatico (code-style, security, architecture, gotchas, review, testing)
- Diagrama mermaid de separacao de diretorios (Entrada/ vs dados/ vs Saida/) no README
- `pyproject.toml` com ruff, isort, taskipy e pytest configurados (alinhado com windx-automatico)
- `.pre-commit-config.yaml` com hooks de lint, format, sort, fix e changelog-reminder
- `scripts/changelog-reminder.sh` — lembrete de atualizar CHANGELOG em cada commit
- Detalhes de conexao SFTP (IP, porta, chave) nas mensagens de log e erro
- Validacao de integridade de arquivos .nc/.grb com retry automatico (apaga + re-download 1x)
- `FILE_CLIMATOLOGIA_VENTO100M` e `REMOTE_CLIMATOLOGIA_VENTO100M` configuraveis via settings

### Alterado

- `downloaders_hgt250_ERA5.py`: download de geopotencial 250 hPa migrado de GRIB para **NetCDF** — elimina dependencia de ecCodes C library (`cfgrib`)
- `pyproject.toml` migrado de Poetry para UV (hatchling)
- Imports em todos os modulos atualizados: `app.config` → `app.shared.settings_factory`, `app.common.logger` → `app.shared.logger`
- `settings.toml` refatorado com ambientes (development/qa/production) e configuracoes de logging
- CLI simplificado: `uv run python run_script.py s00` em vez de `poetry run python main.py --script s00`
- Credenciais movidas de `settings.local.toml` para `app/settings/.secrets.toml`
- `.gitignore` atualizado para UV (uv.lock, .secrets.toml, poetry.lock)
- Downloaders e processadores agora salvam em `dados/` (DIR_DADOS) em vez de `Entrada/arquivos_nc/`
- Climatologia vento 100m agora lida do settings (`FILE_CLIMATOLOGIA_VENTO100M`) em vez de path hardcoded
- Scripts `s00` e `s01` refatorados: paths via `Path(settings.DIR_OUTPUT)`, `Path(settings.DIR_INPUT)`, `Path(settings.DIR_DADOS)`
- `s01`: diretorio de saida renomeado de `s04_GEOP250` para `s01_GEOP250`
- `s01`: `os.path` substituido por `pathlib.Path`, `raise Exception` por `raise RuntimeError from err`
- Logging nos scripts: f-strings substituidas por `%s` (lazy formatting do loguru)
- `settings.json` reformatado (whitespace consistente) e limpo (`lst_emails` removido — vinha de outro projeto)
- `s01`: `lst_areas` hardcoded movido para `settings.LST_AREAS_S01`
- `Scripts/` renomeado para `scripts/` (convencao Python lowercase)
- `s01`: referencias `s04_GEOP250` corrigidas para `s01_GEOP250` no README, QUICKSTART e list_scripts
- `s01`: descricao de requisitos no CLI atualizada (climatologia + legenda + dados em `dados/`)
- `scripts/changelog-reminder.sh` movido para `.hooks/changelog-reminder.sh`
- `REMOTE_CLIMATOLOGIA_VENTO100M`: path corrigido para `.../climatologia/` em vez de `.../Entrada/arquivos_nc/`

### Corrigido

- `s01`: logger usava `'s04'`, cache key `'s04'` e mensagens `'S04'` — corrigido para `'s01'`
- `s01`: path da legenda Atlantico usava `Path.cwd()/Entrada/` hardcoded — agora usa `settings.DIR_INPUT`

### Removido

- `database.md` das rules (nao ha banco de dados neste projeto)

- `app/logging_config.json` (substituido por LoggerService)
- Dependencia de Poetry (substituido por UV)

### Corrigido

- Logging excessivo do `cdsapi` (debug=False nos downloaders + logging.WARNING)
- Logging excessivo do `paramiko` (logging.ERROR)

---

## [0.1.0] - 2026-03-13

Commit: `6bba154`

### Adicionado

- Estrutura inicial do projeto
- Script s00: download e plotagem de vento a 100m + MSLP do ERA5
- Script s01: download e plotagem de anomalia de geopotencial em 250hPa
- Modulo `app/common/` com infraestrutura reutilizavel:
  - `logger.py` - Sistema de logging com Loguru (rotacao, retencao, thread-safe)
  - `cache_manager.py` - Cache inteligente com hash MD5 de parametros
  - `connections.py` - Conexao SSH/SFTP via Paramiko
  - `download_helper.py` - Download multi-engine com retry e barra de progresso
  - `file_loader.py` - Carregamento transparente local/remoto de arquivos
  - `dataset_utils.py` - Utilitarios xarray para NetCDF
  - `parallel_helper.py` - Processamento paralelo com ThreadPoolExecutor
- Modulo `app/src/uteis/` com downloaders e processadores ERA5
- Configuracao via Dynaconf com suporte a ambientes (development/production)
- Suporte a SSH/SFTP para acesso remoto a dados (modo development)
- `main.py` com CLI para execucao de scripts (--script, --force-download, --force-rerun, --clear-cache)
- Configuracoes de 30+ regioes geograficas para plotagem (settings.json)
- Template de configuracao local (settings.local.example.toml)
- `.gitignore` configurado para excluir dados pesados (.nc, .grb), saidas e credenciais
