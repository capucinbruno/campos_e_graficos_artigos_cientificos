# -*- coding: utf-8 -*-
"""
Nucleo do CHI200 intrasazonal (metodo CPC de remocao de media movel).

Dado o vento anomalo diario em 200 hPa (`anom = u/v200 - LTM_diaria`), isola a
banda intrasazonal removendo a media dos N dias anteriores (default 120, padrao CPC):

    intra(t) = anom(t) - media(anom em [t-N, t-1])

e calcula o potencial de velocidade (chi200) de cada dia por inversao de Poisson
do vento divergente (reusa o solver de `plot_chi200`). A serie diaria de chi200
intrasazonal alimenta os tres produtos do s31: mapas de pentada, Hovmoller e mapa
do periodo.

Como chi e linear no vento, filtrar o vento no tempo e inverter Poisson e
equivalente a inverter Poisson a cada dia e filtrar o chi — usamos a 1a forma.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve1d

from app.src.uteis.plot_chi200 import _compute_divergence, _solve_poisson_sphere


def chi_from_wind(u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """chi (potencial de velocidade) de um campo de vento 2D, sem logs (uso em loop diario)."""
    div = _compute_divergence(u, v, lat, lon)
    return _solve_poisson_sphere(div, lat, lon)


# alias privado mantido para compatibilidade interna
_chi_from_wind = chi_from_wind


def div_wind_from_chi(chi2d: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vento divergente (u_div, v_div) associado ao potencial de velocidade chi.

    u_div = (1 / a cos φ) ∂χ/∂λ
    v_div = (1/a) ∂χ/∂φ
    """
    a = 6.371e6
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    dchi_dphi = np.gradient(chi2d, lat_r, axis=0)
    dchi_dlam = np.gradient(chi2d, lon_r, axis=1)
    cos_lat = np.where(np.abs(np.cos(lat_r)) < 1e-10, np.nan, np.cos(lat_r))
    v_div = dchi_dphi / a
    u_div = dchi_dlam / (a * cos_lat[:, None])
    return u_div, v_div


def lanczos_bandpass(
    series: np.ndarray, period_min: float, period_max: float, n: int,
) -> np.ndarray:
    """
    Filtro passa-banda de Lanczos aplicado ao eixo temporal (axis=0) de `series`.

    Isola a banda [period_min, period_max] dias.
    Bordas tratadas com mode='nearest' (adequado para uso operacional em tempo real).

    series : (..., T, lat, lon) ou (T, lat, lon)
    """
    f1 = 1.0 / period_max
    f2 = 1.0 / period_min
    k = np.arange(-n, n + 1, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        ideal = np.where(
            k == 0,
            2.0 * (f2 - f1),
            (np.sin(2 * np.pi * f2 * k) - np.sin(2 * np.pi * f1 * k)) / (np.pi * k),
        )
    weights = ideal * np.sinc(k / n)  # np.sinc(x) = sin(pi*x)/(pi*x)
    return convolve1d(series, weights, axis=0, mode='nearest')


def remove_media_movel(anom: np.ndarray, janela: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove a media movel TRAILING de `janela` dias da serie de anomalia.

    `anom`: (T, lat, lon), diario e contiguo (sem buracos).
    Retorna (intra, idx) onde:
      - intra: (T-janela, lat, lon) = anom(t) - media(anom[t-janela:t])
      - idx: indices (no eixo de tempo original) correspondentes a cada linha de intra
    """
    T = anom.shape[0]
    if T <= janela:
        raise ValueError(
            f'Serie diaria ({T} dias) menor que a janela da media movel ({janela}). '
            f'Aumente o periodo de download (precisa de >{janela} dias antes do alvo).'
        )
    out = np.empty((T - janela,) + anom.shape[1:], dtype=np.float64)
    for k, t in enumerate(range(janela, T)):
        out[k] = anom[t] - anom[t - janela:t].mean(axis=0)
    idx = np.arange(janela, T)
    return out, idx


def chi200_intrasazonal_series(
    u_anom: np.ndarray,
    v_anom: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    janela: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Serie diaria de chi200 intrasazonal e vento divergente associado.

    Retorna (chi_intra, u_div_series, v_div_series, idx):
      - chi_intra: (T-janela, lat, lon) — chi200 intrasazonal por dia
      - u_div_series, v_div_series: (T-janela, lat, lon) — vento divergente por dia
      - idx: indices no eixo de tempo original (para mapear de volta as datas)
    """
    u_is, idx = remove_media_movel(u_anom, janela)
    v_is, _ = remove_media_movel(v_anom, janela)
    chi = np.empty_like(u_is)
    u_div_ser = np.empty_like(u_is)
    v_div_ser = np.empty_like(v_is)
    for k in range(u_is.shape[0]):
        chi[k] = _chi_from_wind(u_is[k], v_is[k], lat, lon)
        u_div_ser[k], v_div_ser[k] = div_wind_from_chi(chi[k], lat, lon)
    return chi, u_div_ser, v_div_ser, idx


def _ww_filter_2d(
    data: np.ndarray,
    k_min: int, k_max: int,
    f_min: float, f_max: float,
) -> np.ndarray:
    """Filtro espectral 2D (tempo x longitude) para um modo WW numa fatia de latitude.

    data : (T, L)  —  T dias  x  L pontos de longitude
    k_min, k_max  : faixa de numero de onda zonal  (k > 0 = propagacao para leste)
    f_min, f_max  : faixa de frequencia em ciclos/dia  (f = 1 / periodo_em_dias)
    """
    from scipy.fft import fft2, ifft2

    T, L = data.shape
    spec = fft2(data)
    freqs = np.fft.fftfreq(T)      # ciclos/dia
    kwave = np.fft.fftfreq(L) * L  # numeros de onda inteiros

    # Para a convencao FFT do numpy/scipy (IFFT usa exp(+2πi(ft/T + kl/L))),
    # uma onda que PROPAGA PARA LESTE x=cos(kl/L - ft/T) tem seus coeficientes
    # espectrais no quadrante (freq < 0, kwave > 0) e no conjugado (freq > 0, kwave < 0).
    # O quadrante (freq > 0, kwave > 0) corresponde a propagacao para OESTE.
    mask = (
        (freqs[:, None] <= -f_min) & (freqs[:, None] >= -f_max) &
        (kwave[None, :] >= k_min)  & (kwave[None, :] <= k_max)
    ) | (
        (freqs[:, None] >= f_min)  & (freqs[:, None] <= f_max) &
        (kwave[None, :] <= -k_min) & (kwave[None, :] >= -k_max)
    )
    return np.real(ifft2(spec * mask))


def ww_filter_modes(
    u_intra: np.ndarray,
    v_intra: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Filtragem espectral Wheeler-Weickmann: MJO e onda de Kelvin.

    Aplica FFT 2D em (tempo x longitude) a cada latitude, zera coeficientes
    fora da banda (wavenumber, frequencia) de cada modo e reconstroi via IFFT.

    u_intra, v_intra : (T, lat, lon)  — serie intrasazonal diaria
    Retorna dict {'mjo': (u, v), 'kelvin': (u, v)} com mesma shape de entrada.

    Bandas (Wheeler & Weickmann 2001 / Wheeler & Kiladis 1999):
      MJO    : k = 1–9  (leste),  periodo 30–90 dias
      Kelvin : k = 1–14 (leste),  periodo 2.5–30 dias
    """
    # (k_min, k_max, f_min, f_max); frequencia em ciclos/dia
    BANDS: dict[str, dict] = {
        'mjo':    dict(k_min=1, k_max=9,  f_min=1 / 90, f_max=1 / 30),
        'kelvin': dict(k_min=1, k_max=14, f_min=1 / 30, f_max=2 / 5),
    }
    T, nlat, nlon = u_intra.shape
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, band in BANDS.items():
        u_f = np.empty_like(u_intra)
        v_f = np.empty_like(v_intra)
        for j in range(nlat):
            u_f[:, j, :] = _ww_filter_2d(u_intra[:, j, :], **band)
            v_f[:, j, :] = _ww_filter_2d(v_intra[:, j, :], **band)
        result[name] = (u_f, v_f)
    return result


def ww_filter_chi_modes(chi_series: np.ndarray, lat: np.ndarray) -> dict[str, np.ndarray]:
    """Filtra a serie diaria de chi intrasazonal para os modos Wheeler-Weickmann.

    Aplica o filtro espectral 2D de _ww_filter_2d diretamente ao scalar chi.

    MJO    : filtro aplicado em cada latitude independentemente (k=1-9, 30-90d).
    Kelvin : filtro aplicado sobre a MEDIA TROPICAL (|lat|<=10°), depois extendido
             meridionalmente com Gaussiana (escala 20°). Motivacao: com T~160d o
             filtro latitude-a-latitude captura ruido extra-tropical (ciclones,
             ondas de Rossby extra-tropicais com k=1-3) que nao sao ondas de Kelvin.
             Usar a media tropical isola o sinal equatorial, e a Gaussiana reproduz
             a estrutura meridional confinada ao equador caracteristica de Kelvin.

    chi_series : (T, nlat, nlon)
    lat        : (nlat,)  — graus, pode ser decrescente (sera reordenado internamente)
    Retorna {'mjo': chi_mjo, 'kelvin': chi_kel} com mesma shape de chi_series.
    """
    # Bandas adaptadas para series curtas (~120 dias de chi_intra apos janela).
    # MJO    : k=1-9,  f=1/90-1/30 CPD  (periodos 30-90d) — caixa WK99 padrao
    # Kelvin : k=1-2,  f=1/30-0.10 CPD  (periodos 10-30d)
    #   A propagacao Kelvin observada no NCICS e ~10 m/s; para k=2 isso e
    #   f~0.043 CPD (periodo 23d). Com T=122, f_min=1/30 captura a partir de
    #   bin 5 = f_real=0.041 CPD — exatamente esse sinal. k<=2 garante padroes
    #   de escala 90-180° de longitude, equivalente ao k=2 dominante no NCICS.
    BANDS: dict[str, dict] = {
        'mjo':    dict(k_min=1, k_max=9, f_min=1 / 90, f_max=1 / 30),
        'kelvin': dict(k_min=1, k_max=2, f_min=1 / 30, f_max=0.10),
    }
    T, nlat, nlon = chi_series.shape
    result: dict[str, np.ndarray] = {}

    for name, band in BANDS.items():
        if name == 'kelvin':
            # Media tropical ponderada por cos(lat) — so latitudes |lat| <= 10°
            trop = np.abs(lat) <= 10.0
            if not trop.any():
                trop = np.abs(lat) <= np.min(np.abs(lat)) + 5.0
            w = np.cos(np.deg2rad(lat[trop]))
            chi_trop = np.sum(chi_series[:, trop, :] * w[None, :, None], axis=1) / w.sum()  # (T, nlon)
            # Filtro espectral 2D na serie tropical
            chi_trop_filt = _ww_filter_2d(chi_trop, **band)  # (T, nlon)
            # Extensao meridional com Gaussiana (escala 20°): reproduz confinamento equatorial
            gauss = np.exp(-0.5 * (lat / 20.0) ** 2)  # (nlat,)
            chi_f = chi_trop_filt[:, None, :] * gauss[None, :, None]  # (T, nlat, nlon)
        else:
            chi_f = np.empty_like(chi_series)
            for j in range(nlat):
                chi_f[:, j, :] = _ww_filter_2d(chi_series[:, j, :], **band)
        result[name] = chi_f
    return result


def media_faixa_latitude(
    campo: np.ndarray, lat: np.ndarray, lat_sul: float, lat_norte: float,
) -> np.ndarray:
    """Media de `campo` (..., lat, lon) numa faixa de latitude, ponderada por cos(lat).

    Retorna array (..., lon).
    """
    lo, hi = min(lat_sul, lat_norte), max(lat_sul, lat_norte)
    mask = (lat >= lo) & (lat <= hi)
    w = np.cos(np.deg2rad(lat[mask]))
    sub = campo[..., mask, :]
    return np.sum(sub * w[:, None], axis=-2) / np.sum(w)


def agrupa_pentadas(
    chi_series: np.ndarray, dates: np.ndarray, n_pentadas: int, *extras: np.ndarray,
) -> list[tuple]:
    """
    Agrupa as ULTIMAS `n_pentadas` pentadas (blocos de 5 dias, da mais recente p/ tras)
    e tira a media de chi em cada uma.

    `extras`: arrays adicionais (mesma forma de chi_series) cuja media de pentada e retornada
    junto com chi. Cada elemento de extras aparece como campo extra na tupla de retorno.

    Retorna lista [(dia_ini, dia_fim, campo_medio, *extras_medios), ...] da mais antiga p/ a mais recente.
    """
    T = chi_series.shape[0]
    blocos = []
    for p in range(n_pentadas):
        fim = T - p * 5
        ini = fim - 5
        if ini < 0:
            break
        extras_mean = tuple(e[ini:fim].mean(axis=0) for e in extras)
        blocos.append((dates[ini], dates[fim - 1], chi_series[ini:fim].mean(axis=0)) + extras_mean)
    return list(reversed(blocos))
