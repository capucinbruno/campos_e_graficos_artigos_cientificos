# Bibliotecas padrão
import os
import time
from pathlib import Path

# Bibliotecas de terceiros
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.util import add_cyclic_point
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator, FixedLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# Módulos locais
from app.shared.settings_factory import settings
from app.shared.logger import get_logger
from app.common.dataset_utils import load_dataset
from app.common.cache_manager import check_cache_valid, save_cache_metadata
from app.src.uteis.plot_geop250 import main as plot_geop250


def main():
    # Logger específico para este script (logs/s04.log)
    logger = get_logger("s04")

    logger.info("=" * 80)
    logger.info("📊 SCRIPT S04: Anomalia GEOP250 (Geopotencial 250 hPa)")
    logger.info("=" * 80)

    # Lista de áreas (definida uma vez para cache e processamento)
    lst_areas = [
        "pacific_chile",
        "china",
        "pacifico_leste_america_sul",
        "america_sul_zom_out",
        "MDR",
        "tropico",
        "zona_zcit_atlantico",
        "brasil",
        "america_sul",
        "africa_monsoon",
        "africa",
        "mjo",
        "amo",
        "sad",
        "iod",
        "pdo",
        "tna",
        "tsa",
        "atlantico_tropical",
        "enso",
        "globo",
        "costa_brasil",
        "psa",
        "argentina",
        "estados_unidos_zoom",
        "estados_unidos",
        "hemisferio_sul",
    ]

    # Verificação de cache
    output_dir = os.path.join(f"{settings.DIR_OUTPUT}/s04_GEOP250")
    cache_params = {
        "DATA_INICIAL": settings.DATA_INICIAL,
        "DATA_FINAL": settings.DATA_FINAL,
        "areas": lst_areas,
        "script_version": "2.0",  # Incrementar se lógica de processamento mudar
    }
    output_files = [f"{output_dir}/geop250_{area}.png" for area in lst_areas]

    if check_cache_valid("s04", cache_params, output_files):
        logger.info("🎯 CACHE VÁLIDO! Execução já foi realizada com os mesmos parâmetros.")
        logger.info(f"   📅 Período: {settings.DATA_INICIAL} a {settings.DATA_FINAL}")
        logger.info(f"   📊 {len(output_files)} mapas já existem")
        logger.info(f"   📁 Diretório: {output_dir}")
        logger.info("   ⏭️  Pulando execução")
        return

    start_time = time.time()
    logger.info(f"📅 Período de análise: {settings.DATA_INICIAL} a {settings.DATA_FINAL}")
    logger.info(f"📊 Gerando {len(lst_areas)} mapas de anomalia GEOP250")
    logger.info("=" * 80)
    try:
        plot_geop250()
    except Exception as err:
        msg = "ERROR: Falha no download do PSL"
        logger.error(err)
        raise Exception(msg)

    # Cria a pasta de saída do modelo, se ainda não existir
    output_dir = os.path.join(f"{settings.DIR_OUTPUT}/s04_GEOP250")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ds1 = load_dataset(f"{settings.DIR_INPUT}/arquivos_nc/geop250.nc")


    da = ds1.sortby(ds1.lon)["hgt"].isel(time=0)
    lon = da["lon"]
    lat = da["lat"]
    hgt, lon = add_cyclic_point(da, coord=lon)

    if np.all((hgt >= -50) & (hgt <= 50)):
        levels = np.arange(-50, 53, 3)
        ticks = np.arange(-50, 55, 5)
    elif np.all((hgt >= -100) & (hgt <= 100)):
        levels = np.arange(-100, 120, 20)
        ticks = np.arange(-100, 150, 50)
    else:
        levels = np.arange(-200, 220, 20)
        ticks = np.arange(-200, 250, 50)

    # lst_areas já definido no início para cache
    for area in lst_areas:
        info_plot = settings["areas_plotagem"]
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(
            1,
            1,
            1,
            projection=ccrs.PlateCarree(central_longitude=info_plot[area]["central_longitude_mapa"]),
        )
        if info_plot[area]["plot_box"]:
            lst_boxes = info_plot[area]["lst_boxes"]
            for box in lst_boxes:
                box = patches.Rectangle(
                    (box["x_anc"], box["y_anc"]),
                    box["x_larg"],
                    box["y_larg"],
                    linewidth=box["linewidth"],
                    edgecolor=box["edgecolor"],
                    facecolor="none",
                    zorder=100,
                )

                ax.add_patch(box)

        if area == "MDR":
            lon_min_MDR, lon_max_MDR = -86, -20
            lat_min_MDR, lat_max_MDR = 10, 20

            ax.plot(
                [lon_min_MDR, lon_max_MDR, lon_max_MDR, lon_min_MDR, lon_min_MDR],
                [lat_min_MDR, lat_min_MDR, lat_max_MDR, lat_max_MDR, lat_min_MDR],
                color="black",
                linewidth=3,
                linestyle="-",
                zorder=500,
                transform=ccrs.PlateCarree(),
            )

        if area == "atlantico_tropical":
            diretorio_atual = Path.cwd()
            img_legenda_atlantic = plt.imread(f"{diretorio_atual}/Entrada/legenda_atlantic.png")
            fig.figimage(
                img_legenda_atlantic,
                125,
                614,
                zorder=3,
                alpha=1,
            )

            # Box TSA
            box1 = patches.Rectangle(
                (10, -20),
                -40,
                20,
                linewidth=3,
                edgecolor="black",
                facecolor="none",
                zorder=100,
            )  # (posição x, posição y de ancoragem) largura

            # Box TNA
            box2 = patches.Rectangle(
                (-15, 5),
                -40,
                20,
                linewidth=3,
                edgecolor="blue",
                facecolor="none",
                zorder=100,
            )  # (posição x, posição y de ancoragem) largura

            # Add the patch to the Axes
            ax.add_patch(box1)
            ax.add_patch(box2)

        if area == "iod":
            # Box Indico Ocidental
            box3 = patches.Rectangle(
                (50, -10),
                20,
                20,
                linewidth=3,
                edgecolor="black",
                facecolor="none",
                zorder=100,
            )  # (posição x, posição y de ancoragem) largura

            # Box Indico Oriental (Indonésia)
            box4 = patches.Rectangle(
                (90, -10),
                20,
                10,
                linewidth=3,
                edgecolor="black",
                facecolor="none",
                zorder=100,
            )  # (posição x, posição y de ancoragem) largura

            # Add the patch to the Axes
            ax.add_patch(box3)
            ax.add_patch(box4)

        if area == "enso":
            text1 = plt.text(66.25, -13.64, "Niño 1+2", fontsize=14, color="red", weight="bold")
            text1.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground="black"),
                path_effects.Normal(),
            ])

            text2 = plt.text(
                34.1,
                8.45,
                "Niño 3",
                fontsize=14,
                color="blue",
                weight="bold",
                path_effects=[path_effects.withSimplePatchShadow()],
            )
            text2.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground="white"),
                path_effects.Normal(),
            ])

            text3 = plt.text(
                8.6,
                -9.45,
                "Niño 3.4",
                fontsize=14,
                color="black",
                weight="bold",
                path_effects=[path_effects.withSimplePatchShadow()],
            )
            text3.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground="white"),
                path_effects.Normal(),
            ])

            text4 = plt.text(
                -22.5,
                8.45,
                "Niño 4",
                fontsize=14,
                color="m",
                weight="bold",
                path_effects=[path_effects.withSimplePatchShadow()],
            )
            text4.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground="black"),
                path_effects.Normal(),
            ])

        if area == "mjo":
            text5 = plt.text(
                -135.25,
                -4.64,
                "1",
                fontsize=50,
                color="white",
                weight="bold",
                zorder=400,
            )
            text5.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground="black"),
                path_effects.Normal(),
            ])
            text6 = plt.text(
                -115.25,
                -4.64,
                "2",
                fontsize=50,
                color="white",
                weight="bold",
                zorder=400,
            )
            text6.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground="black"),
                path_effects.Normal(),
            ])
            text7 = plt.text(
                -95.25,
                -4.64,
                "3",
                fontsize=50,
                color="white",
                weight="bold",
                zorder=400,
            )
            text7.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground="black"),
                path_effects.Normal(),
            ])
            text8 = plt.text(
                -75.25,
                -4.64,
                "4",
                fontsize=50,
                color="white",
                weight="bold",
                zorder=400,
            )
            text8.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground="black"),
                path_effects.Normal(),
            ])
            text9 = plt.text(
                -55.25,
                -4.64,
                "5",
                fontsize=50,
                color="white",
                weight="bold",
                zorder=400,
            )
            text9.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground="black"),
                path_effects.Normal(),
            ])
            text10 = plt.text(
                -35.25,
                -4.64,
                "6",
                fontsize=50,
                color="white",
                weight="bold",
                zorder=400,
            )
            text10.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground="black"),
                path_effects.Normal(),
            ])
            text11 = plt.text(
                -15.25,
                -4.64,
                "7",
                fontsize=50,
                color="white",
                weight="bold",
                zorder=400,
            )
            text11.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground="black"),
                path_effects.Normal(),
            ])
            text12 = plt.text(
                4.75,
                -4.64,
                "8",
                fontsize=50,
                color="white",
                weight="bold",
                zorder=400,
            )
            text12.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground="black"),
                path_effects.Normal(),
            ])

        # Adicionar linhas de grade (latitude e longitude)
        gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.0)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 20, "color": "black"}
        gl.ylabel_style = {"size": 20, "color": "black"}

        # Configuração específica para "hemisferio_sul"
        if area == "hemisferio_sul":
            gl.xlocator = MultipleLocator(40)
            gl.ylocator = MultipleLocator(20)

        elif area == "psa":
            gl.xlocator = MultipleLocator(40)
            gl.ylocator = MultipleLocator(20)

        elif area == "estados_unidos":
            gl.xlocator = MultipleLocator(10)
            gl.ylocator = MultipleLocator(10)

        elif area == "estados_unidos_zoom":
            gl.xlocator = MultipleLocator(10)
            gl.ylocator = MultipleLocator(5)

        elif area == "argentina":
            gl.xlocator = MultipleLocator(5)
            gl.ylocator = MultipleLocator(5)

        elif area == "brasil":
            gl.xlocator = MultipleLocator(10)
            gl.ylocator = MultipleLocator(5)

        elif area == "america_sul":
            gl.xlocator = MultipleLocator(20)
            gl.ylocator = MultipleLocator(20)

        elif area == "zona_zcit_atlantico":
            gl.xlocator = MultipleLocator(10)
            gl.ylocator = MultipleLocator(5)

        elif area == "costa_brasil":
            gl.xlocator = MultipleLocator(5)
            gl.ylocator = MultipleLocator(5)

        elif area == "enso":
            # Definindo manualmente os valores de longitude para aparecer nos rótulos
            longitudes = [
                -160,
                -140,
                -120,
                -100,
                -80,
                -60,
                0,
                20,
                40,
                60,
                80,
                100,
                120,
                140,
                150,
                160,
                170,
                180,
            ]
            gl.xlocator = FixedLocator(longitudes)
            gl.ylocator = MultipleLocator(10)
            gl.xlabel_style = {"size": 15, "color": "black"}
            gl.ylabel_style = {"size": 15, "color": "black"}

        elif area == "atlantico_tropical":
            gl.xlocator = MultipleLocator(20)
            gl.ylocator = MultipleLocator(10)

        elif area == "tsa":
            gl.xlocator = MultipleLocator(10)
            gl.ylocator = MultipleLocator(5)

        elif area == "tna":
            gl.xlocator = MultipleLocator(10)
            gl.ylocator = MultipleLocator(5)

        elif area == "globo":
            gl.xlocator = MultipleLocator(40)
            gl.ylocator = MultipleLocator(20)
            gl.xlabel_style = {"size": 15, "color": "black"}
            gl.ylabel_style = {"size": 15, "color": "black"}

        elif area == "pdo":
            gl.xlocator = MultipleLocator(20)
            gl.ylocator = MultipleLocator(10)

        elif area == "iod":
            gl.xlocator = MultipleLocator(20)
            gl.ylocator = MultipleLocator(10)

        elif area == "sad":
            gl.xlocator = MultipleLocator(20)
            gl.ylocator = MultipleLocator(10)

        elif area == "amo":
            gl.xlocator = MultipleLocator(20)
            gl.ylocator = MultipleLocator(10)

        elif area == "mjo":
            gl.xlocator = MultipleLocator(40)
            gl.ylocator = MultipleLocator(20)

        elif area == "africa":
            gl.xlocator = MultipleLocator(20)
            gl.ylocator = MultipleLocator(20)

        elif area == "africa_monsoon":
            gl.xlocator = MultipleLocator(20)
            gl.ylocator = MultipleLocator(10)

        elif area == "tropico":
            longitudes = [-160, -120, -80, -40, 0, 40, 80, 120, 160]
            gl.xlocator = FixedLocator(longitudes)
            gl.ylocator = MultipleLocator(20)
            gl.xlabel_style = {"size": 15, "color": "black"}
            gl.ylabel_style = {"size": 15, "color": "black"}

        # Limites da area de plotagem
        ax.set_xlim([info_plot[area]["lon_esq"], info_plot[area]["lon_dir"]])
        ax.set_ylim([info_plot[area]["lat_inf"], info_plot[area]["lat_sup"]])

        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=1.2)
        ax.add_feature(cfeature.LAND.with_scale("50m"), linewidth=0.5, facecolor="whitesmoke")

        if area != "china":
            ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=1.2, zorder=100)

        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=1.2, zorder=100)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=1.2, zorder=100)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), linewidth=0.5, facecolor="white")

        cmap_name = settings.LST_ANOM_CORRETA
        cmap = LinearSegmentedColormap.from_list(cmap_name, settings.LST_ANOM_CORRETA)

        im = ax.contourf(
            lon,
            lat,
            hgt,
            levels=levels,
            # cmap="RdYlBu_r",
            cmap=cmap,
            extend="both",
            transform=ccrs.PlateCarree(central_longitude=info_plot[area]["central_longitude_plot"]),
        )

        if (
            area == "enso"
            or area == "tropico"
            or area == "MDR"
            or area == "hemisferio_sul"
            or area == "psa"
        ):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("bottom", size="6%", pad=0.50, axes_class=plt.Axes)

            cbar = plt.colorbar(
                im,
                cax=cax,
                pad=0.02,
                fraction=0.02375,
                location="bottom",
                extend="both",
                orientation="horizontal",
                # ticks=settings.LST_LEVELS_SST,
                ticks=ticks,
            )

        else:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.05, axes_class=plt.Axes)

            cbar = plt.colorbar(im, cax=cax, pad=0.02, fraction=0.02375, extend="both", ticks=ticks)

        cbar.set_label(label="mgp", size=18)
        cbar.ax.tick_params(labelsize=20)

        im = ax.contour(
            lon,
            lat,
            hgt,
            # levels=levels,
            levels=levels,
            colors="white",
            linewidths=0.6,
            transform=ccrs.PlateCarree(central_longitude=info_plot[area]["central_longitude_plot"]),
        )

        # Adicionar um título com quebra de linha e variáveis
        titulo = f"Anomaly GEOP. HEIGHT 250mb (from {settings.DATA_INICIAL} to {settings.DATA_FINAL})\nSource: PSL/NOAA."
        ax.set_title(titulo, fontsize=18, loc="left")

        if area == "hemisferio_sul":
            img_logotipo_grec = plt.imread(f"{settings.DIR_INPUT}/logo_grec.png")
            fig.figimage(img_logotipo_grec, 100, 160, zorder=3, alpha=1)

        elif area == "psa":
            img_logotipo_grec = plt.imread(f"{settings.DIR_INPUT}/logo_grec.png")
            fig.figimage(img_logotipo_grec, 100, 160, zorder=3, alpha=1)

        elif area == "tropico":
            img_logotipo_grec = plt.imread(f"{settings.DIR_INPUT}/logo_grec.png")
            fig.figimage(img_logotipo_grec, 72, 145, zorder=3, alpha=1)

        elif area == "MDR":
            img_logotipo_grec = plt.imread(f"{settings.DIR_INPUT}/logo_grec.png")
            fig.figimage(img_logotipo_grec, 90, 154, zorder=3, alpha=1)

        elif area == "globo":
            img_logotipo_grec = plt.imread(f"{settings.DIR_INPUT}/logo_grec.png")
            fig.figimage(img_logotipo_grec, 76, 45, zorder=3, alpha=1)

        elif area == "mjo":
            img_logotipo_grec = plt.imread(f"{settings.DIR_INPUT}/logo_grec.png")
            fig.figimage(img_logotipo_grec, 90, 45, zorder=3, alpha=1)

        else:
            # Adicionando logotipo grec
            img_logotipo_grec = plt.imread(f"{settings.DIR_INPUT}/logo_grec.png")
            valor_x = info_plot[area]["logo_grec"]["valor_x"]
            valor_y = info_plot[area]["logo_grec"]["valor_y"]
            fig.figimage(img_logotipo_grec, valor_x, valor_y, zorder=3, alpha=1)

        # Definindo diretório onde serão salvas as figuras
        path_output = f"{settings.DIR_OUTPUT}/s04_GEOP250"
        Path(path_output).mkdir(exist_ok=True, parents=True)
        filename_fig = f"{path_output}/geop250_{area}.png"
        logger.info(f"Salvando a figura {filename_fig}")

        # Salvando as figuras
        plt.savefig(
            f"{filename_fig}",
            dpi=fig.dpi,
            bbox_inches="tight",
        )
        plt.close("all")

    # Salvar metadados do cache após execução bem-sucedida
    execution_time = time.time() - start_time
    save_cache_metadata("s04", cache_params, output_files, execution_time)
    logger.info("=" * 80)
    logger.info(f"✅ Script S04 concluído com sucesso!")
    logger.info(f"⏱️  Tempo de execução: {execution_time:.1f}s ({execution_time/60:.1f} min)")
    logger.info(f"📊 {len(output_files)} mapas gerados em: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
