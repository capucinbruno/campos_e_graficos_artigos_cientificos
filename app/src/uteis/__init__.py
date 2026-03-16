# Bibliotecas de terceiros
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter


def add_gridlines(ax, lst_valores_lon: list, lst_valores_lat: list):
    ax.set_yticks(lst_valores_lat, crs=ccrs.PlateCarree())
    ax.set_xticks(lst_valores_lon, crs=ccrs.PlateCarree())

    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
