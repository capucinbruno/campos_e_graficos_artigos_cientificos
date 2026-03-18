"""
Description: Adiciona Gridlines

Author:           @Bruno Capucin
Created:          2022-03-31
Copyright:        (c) Ampere Consultoria Ltda
"""

try:
    # Bibliotecas de terceiros
    import cartopy.crs as ccrs  # para trabalhar com projeções
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
    from dynaconf import Dynaconf

    settings = Dynaconf(
        envvar_prefix='AMPERE',
        settings_files=['settings.toml', '.secrets.toml'],
        environments=True,
        load_dotenv=True,
    )

except ImportError as error:
    print(error)
    print(f'error.name: {error.name}')
    print(f'error.path: {error.path}')


# Função que vai adicionar visuais aos mapas (numeros de lat, lon etc)
def add_gridlines(ax, lst_valores_lon: list, lst_valores_lat: list):
    """_Esta função tem por objetivo adicionar valores de lat
    e lon nos mapas a partir dos argumentos ax e listas com os valores de de lat e lon

    Args:
        ax (_type_): _description_
        lst_valores_lon (list): _description_
        lst_valores_lat (list): _description_
    """
    ax.set_yticks(lst_valores_lat, crs=ccrs.PlateCarree())
    ax.set_xticks(lst_valores_lon, crs=ccrs.PlateCarree())

    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
