# app/src/uteis/rossby_wave_source.py
# -*- coding: utf-8 -*-
"""
Fonte de Onda de Rossby (RWS) — Sardeshmukh & Hoskins (1988).

RWS = -div(v_chi * zeta_a) = -zeta_a * D - v_chi . grad(zeta_a)

onde:
    v_chi = vento divergente (irrotacional, = grad(chi))
    zeta_a = vorticidade absoluta = zeta + f
    D = divergencia do vento total

Termo forcante da equacao da vorticidade no nivel de saida da conveccao (200 hPa):
indica ONDE ondas de Rossby sao geradas. Convencao de sinal (tendencia de zeta):
    RWS > 0 -> fonte de vorticidade positiva (anticiclonica no HS / ciclonica no HN)
    RWS < 0 -> fonte de vorticidade negativa (ciclonica no HS / anticiclonica no HN)

Reusa a maquinaria de divergencia/potencial de velocidade do projeto (sem windspharm):
    - _compute_divergence (plot_chi200)
    - chi_from_wind / div_wind_from_chi (chi200_intrasazonal)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from app.src.uteis.chi200_intrasazonal import chi_from_wind, div_wind_from_chi
from app.src.uteis.plot_chi200 import _compute_divergence

EARTH_RADIUS = 6.371e6  # m
OMEGA = 7.292e-5        # s^-1


def _smooth_latlon(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, smooth_deg: float) -> np.ndarray:
    """Suaviza o campo (gaussiana separavel) — lon com wrap, lat com borda. sigma em graus.

    A RWS tem 2as derivadas; suavizar o estado basico antes e' essencial p/ um mapa
    legivel (equivale ao truncamento espectral usado na literatura).
    """
    if smooth_deg <= 0:
        return field
    s_lat = smooth_deg / abs(np.median(np.diff(lat)))
    s_lon = smooth_deg / abs(np.median(np.diff(lon)))
    out = gaussian_filter1d(field, s_lat, axis=0, mode='nearest')
    return gaussian_filter1d(out, s_lon, axis=1, mode='wrap')


def relative_vorticity(u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Vorticidade relativa zeta = (1/(a cosφ))[∂v/∂λ - ∂(u cosφ)/∂φ]. lat ascendente, lon 0..360."""
    a = EARTH_RADIUS
    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon)
    cosphi = np.cos(phi)
    cosphi_safe = np.where(np.abs(cosphi) < 1e-10, 1e-10, cosphi)

    dv_dlam = np.gradient(v, lam, axis=1)
    ucos = u * cosphi[:, None]
    ducos_dphi = np.gradient(ucos, phi, axis=0)
    return (dv_dlam - ducos_dphi) / (a * cosphi_safe[:, None])


def rossby_wave_source(
    u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray,
    smooth_deg: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula a RWS (s^-2) e o vento divergente associado.

    Parametros
    ----------
    u, v : (lat, lon) vento em 200 hPa (m/s). lat ascendente, lon 0..360.
    smooth_deg : suavizacao gaussiana (graus) do estado basico antes das derivadas.
                 Essencial p/ um mapa de RWS legivel (2as derivadas amplificam ruido).

    Retorna
    -------
    (rws, u_chi, v_chi) : RWS (s^-2) e componentes do vento divergente (m/s).
    """
    a = EARTH_RADIUS
    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon)
    cosphi = np.cos(phi)
    cosphi_safe = np.where(np.abs(cosphi) < 1e-10, 1e-10, cosphi)

    u = _smooth_latlon(np.asarray(u, dtype=float), lat, lon, smooth_deg)
    v = _smooth_latlon(np.asarray(v, dtype=float), lat, lon, smooth_deg)

    D = _compute_divergence(u, v, lat, lon)
    chi = chi_from_wind(u, v, lat, lon)
    u_chi, v_chi = div_wind_from_chi(chi, lat, lon)

    zeta = relative_vorticity(u, v, lat, lon)
    f = 2.0 * OMEGA * np.sin(phi)
    zeta_a = zeta + f[:, None]

    dza_dx = np.gradient(zeta_a, lam, axis=1) / (a * cosphi_safe[:, None])
    dza_dy = np.gradient(zeta_a, phi, axis=0) / a

    rws = -zeta_a * D - (u_chi * dza_dx + v_chi * dza_dy)
    return rws, u_chi, v_chi
