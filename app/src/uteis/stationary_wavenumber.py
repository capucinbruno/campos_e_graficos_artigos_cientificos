# app/src/uteis/stationary_wavenumber.py
# -*- coding: utf-8 -*-
"""
Numero de onda estacionario de Rossby (Ks) — Hoskins & Ambrizzi (1993).

Diagnostico de "guia de onda" (waveguide): a partir do escoamento zonal medio
basico (u em um nivel, tipicamente 200 hPa), calcula o numero de onda total que
uma onda de Rossby precisaria ter para ficar ESTACIONARIA naquele ponto.

Formulacao em coordenadas de Mercator (Hoskins & Ambrizzi 1993), que evita a
singularidade de cos(phi) e revela os jatos como guias de onda:

    U_M   = u / cos(phi)                              (vento zonal de Mercator)
    beta_M = 2*Omega*cos^2(phi)/a  -  d2(U_M)/dy^2    (grad. de vort. absoluta)
    Ks    = a * sqrt(beta_M / U_M)                    (adimensional, n. de onda zonal)

onde y e a coordenada meridional de Mercator (dy = a*dphi/cos(phi)), logo
d/dy = (cos(phi)/a) d/dphi.

Interpretacao:
    - Ks real (Ks^2 > 0): onda estacionaria E POSSIVEL. Maximos locais de Ks
      marcam waveguides (nucleos de jato). A latitude onde Ks = k do forcante e
      a "latitude de retorno" (a onda e refletida).
    - Ks^2 <= 0 (ventos de leste U_M<=0, ou beta_M<=0): onda evanescente, NAO ha
      onda estacionaria possivel -> regiao mascarada nos mapas.

IMPORTANTE: Ks e propriedade do MEIO (escoamento basico), nao da onda. Diz onde
uma onda estacionaria PODE existir, nao se ELA existe. Para concluir, sobreponha
o campo da onda real (ex.: anomalia de vento meridional v'200).

Este modulo e puro NumPy e independe da fonte do vento: serve igualmente para
reanalise (ERA5/GDAS) e para dados de previsao (uso futuro em prognostico).

Referencia:
    Hoskins, B. J., & Ambrizzi, T. (1993). Rossby Wave Propagation on a Realistic
    Longitudinally Varying Flow. J. Atmos. Sci., 50(12), 1661-1671.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Constantes fisicas
EARTH_RADIUS = 6.371e6  # m
OMEGA = 7.292e-5        # s^-1 (rotacao da Terra)


@dataclass
class StationaryWavenumber:
    """Resultado do calculo de Ks (todos os campos 2D em (lat, lon), lat ascendente)."""

    ks: np.ndarray         # numero de onda estacionario (adimensional); NaN onde evanescente
    ks2: np.ndarray        # Ks^2 com sinal (negativo = evanescente) — base da mascara
    beta_m: np.ndarray     # grad. meridional de vort. absoluta de Mercator (m^-1 s^-1)
    u_m: np.ndarray        # vento zonal de Mercator (m/s)
    lat: np.ndarray        # latitudes (graus, ascendente)
    lon: np.ndarray        # longitudes (graus)

    @property
    def evanescent_mask(self) -> np.ndarray:
        """True onde NAO ha onda estacionaria possivel (Ks imaginario)."""
        return ~(self.ks2 > 0)


def _smooth_latlon(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, smooth_deg: float) -> np.ndarray:
    """Suaviza o campo basico com boxcar separavel (lat e lon) de ~smooth_deg graus.

    O calculo de Ks envolve a 2a derivada meridional de u, muito sensivel a ruido.
    Suavizar o escoamento basico antes das derivadas e pratica padrao (Hoskins &
    Ambrizzi usaram truncamento espectral; aqui usamos um boxcar equivalente leve).
    """
    if smooth_deg <= 0:
        return field

    def _boxcar(arr: np.ndarray, axis: int, coord: np.ndarray) -> np.ndarray:
        dcoord = np.abs(np.median(np.diff(coord)))
        win = max(1, int(round(smooth_deg / dcoord)))
        if win <= 1:
            return arr
        if win % 2 == 0:
            win += 1
        kernel = np.ones(win) / win
        pad = win // 2
        arr_p = np.apply_along_axis(
            lambda m: np.convolve(np.pad(m, pad, mode='edge'), kernel, mode='valid'),
            axis, arr,
        )
        return arr_p

    out = _boxcar(field, 0, lat)
    out = _boxcar(out, 1, lon)
    return out


def stationary_wavenumber(
    u_mean: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    smooth_deg: float = 5.0,
    mask_tropics_deg: float = 5.0,
    ks_max: float = 20.0,
) -> StationaryWavenumber:
    """Calcula Ks (Hoskins & Ambrizzi 1993) a partir do vento zonal medio basico.

    Parametros
    ----------
    u_mean : np.ndarray (lat, lon)
        Vento zonal medio do periodo no nivel (m/s). Use o vento TOTAL (nao a
        anomalia) — Ks descreve o escoamento basico de oeste/jato.
    lat, lon : np.ndarray
        Coordenadas em graus. `lat` sera ordenada de forma ascendente internamente.
    smooth_deg : float
        Janela (graus) da suavizacao do escoamento basico antes das derivadas.
    mask_tropics_deg : float
        Mascara |lat| < este valor (perto do equador U->0 explode Ks). 0 desativa.
    ks_max : float
        Limita Ks a este valor. Perto da linha critica U=0 (U_M->0+), Ks->infinito;
        valores acima de ~20 nao tem significado fisico util e poluem a saida.

    Retorna
    -------
    StationaryWavenumber
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    u = np.asarray(u_mean, dtype=float)

    # garante lat ascendente
    if lat[0] > lat[-1]:
        order = np.argsort(lat)
        lat = lat[order]
        u = u[order, :]

    u = _smooth_latlon(u, lat, lon, smooth_deg)

    phi = np.deg2rad(lat)
    cosphi = np.cos(phi)
    # evita divisao por zero exatamente nos polos
    cosphi = np.where(np.abs(cosphi) < 1e-6, np.nan, cosphi)
    cos2d = cosphi[:, None]

    a = EARTH_RADIUS

    # Vento zonal de Mercator
    u_m = u / cos2d

    # d/dy = (cos(phi)/a) d/dphi  (y = coordenada de Mercator)
    dudphi = np.gradient(u_m, phi, axis=0)
    du_dy = (cos2d / a) * dudphi
    d2u_dy2 = (cos2d / a) * np.gradient(du_dy, phi, axis=0)

    # beta de Mercator = grad. meridional da vorticidade absoluta
    beta_m = (2.0 * OMEGA * cos2d ** 2) / a - d2u_dy2

    # Ks^2 = a^2 * beta_M / U_M  (adimensional)
    with np.errstate(divide='ignore', invalid='ignore'):
        ks2 = (a ** 2) * beta_m / u_m

    ks = np.sqrt(np.where(ks2 > 0, ks2, np.nan))
    if ks_max > 0:
        ks = np.minimum(ks, ks_max)

    if mask_tropics_deg > 0:
        trop = np.abs(lat) < mask_tropics_deg
        ks[trop, :] = np.nan
        ks2[trop, :] = np.nan

    return StationaryWavenumber(ks=ks, ks2=ks2, beta_m=beta_m, u_m=u_m, lat=lat, lon=lon)
