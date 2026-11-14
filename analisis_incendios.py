"""
Análisis de patrones y gráficas sobre una serie diaria de actividad de incendios.

La entrada es una tabla con una fila por día de una región:
    fecha, n_arm (celdas con fuego armonizadas), frp_arm (energía total armonizada)
Los días sin incendio deben contar como cero, por eso preparar() rellena los huecos.

Cada análisis es una función que devuelve datos (DataFrame o diccionario) y cada gráfica
es otra función que recibe esos datos y devuelve una figura. La API puede llamar solo
a las primeras, y la interfaz o el CSV de descarga usan sus resultados.

Prueba rápida con datos simulados:
    python analisis_incendios.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # sin ventana, para poder guardar imágenes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import theilslopes

AZUL, NARANJA, GRIS = "#0072B2", "#D55E00", "#6b6b6b"   # par de colores seguro para daltonismo
MIN_ANIOS_REF = 5                                          # historia mínima para comparar


# ---------------------------------------------------------------- datos de prueba
def generar_serie_falsa(inicio="2000-11-01", fin="2026-11-14", semilla=1):
    """Serie diaria inventada: temporada alta en septiembre, años fuertes y poca tendencia."""
    rng = np.random.default_rng(semilla)
    dias = pd.date_range(inicio, fin, freq="D")
    doy = dias.dayofyear.to_numpy()
    temporada = 1 + 0.9 * np.sin((doy - 170) / 365 * 2 * np.pi)       # pico hacia septiembre
    efecto_anio = {a: rng.lognormal(0, 0.35) for a in range(dias.year.min(), dias.year.max() + 1)}
    for a in (2005, 2010, 2019, 2024):                                  # años excepcionales
        efecto_anio[a] *= 1.8
    f_anio = dias.year.map(efecto_anio).to_numpy()
    tendencia = 1 + 0.008 * (dias.year.to_numpy() - 2000)
    n = rng.poisson(np.clip(40 * temporada**2 * f_anio * tendencia, 0.2, None))
    frp = n * rng.gamma(4, 6, len(dias))
    return pd.DataFrame({"fecha": dias, "n_arm": n.astype(float), "frp_arm": frp})


# ---------------------------------------------------------------- utilidades
def preparar(serie):
    """Fecha como índice diario, con ceros en los días sin registro."""
    # Comprueba el esquema antes de seleccionar columnas o agrupar.
    requeridas = {"fecha", "n_arm", "frp_arm"}
    faltantes = requeridas.difference(serie.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(sorted(faltantes))}")
    if serie.empty:
        raise ValueError("La serie no contiene filas.")

    # Convierte fechas con zona horaria o sin ella a días UTC comparables.
    s = serie.loc[:, ["fecha", "n_arm", "frp_arm"]].copy()
    s["fecha"] = pd.to_datetime(s["fecha"], errors="raise", utc=True)
    s["fecha"] = s["fecha"].dt.tz_convert(None).dt.normalize()
    if s["fecha"].isna().any():
        raise ValueError("La columna fecha contiene valores vacíos o no válidos.")

    # Rechaza valores no numéricos, faltantes, infinitos o negativos.
    for col in ["n_arm", "frp_arm"]:
        s[col] = pd.to_numeric(s[col], errors="raise")
        valores = s[col].to_numpy(dtype=float)
        if not np.isfinite(valores).all() or (valores < 0).any():
            raise ValueError(f"{col} debe contener solo valores finitos y no negativos.")

    # Suma registros del mismo día y completa los días sin detecciones con cero.
    s = s.groupby("fecha")[["n_arm", "frp_arm"]].sum().sort_index()
    calendario = pd.date_range(s.index.min(), s.index.max(), freq="D")
    return s.reindex(calendario, fill_value=0.0)


# Rechaza métricas desconocidas en vez de calcular o etiquetar una columna equivocada.
def _validar_col(col):
    if col not in {"n_arm", "frp_arm"}:
        raise ValueError("col debe ser 'n_arm' o 'frp_arm'.")


def _meses_completos(s):
    """Total mensual solo de los meses cubiertos por completo (el primero y el último suelen no estarlo)."""
    mensual = s.resample("MS").sum()
    inicio, fin = s.index.min(), s.index.max()
    ok = (mensual.index >= inicio) & ((mensual.index + pd.offsets.MonthEnd(0)) <= fin)
    return mensual[ok]


def _anios_completos(s):
    inicio, fin = s.index.min(), s.index.max()
    return [a for a in range(inicio.year, fin.year + 1)
            if pd.Timestamp(a, 1, 1) >= inicio and pd.Timestamp(a, 12, 31) <= fin]


# ---------------------------------------------------------------- análisis (devuelven datos)
def estacionalidad(serie, col="n_arm"):
    """Total mensual típico: media y percentiles 10 y 90 entre años, por mes."""
    _validar_col(col)
    m = _meses_completos(preparar(serie))[col]
    g = m.groupby(m.index.month)
    return pd.DataFrame({"media": g.mean(), "p10": g.quantile(0.10), "p90": g.quantile(0.90),
                         "anios": g.size()}).rename_axis("mes")


def totales_anuales(serie, col="n_arm"):
    """Total por año completo y tendencia (pendiente de Theil-Sen, más robusta a años extremos)."""
    _validar_col(col)
    s = preparar(serie)
    anual = s[col].groupby(s.index.year).sum()
    anual = anual[anual.index.isin(_anios_completos(s))].rename("total")
    if len(anual) < MIN_ANIOS_REF:
        return anual.to_frame(), {"pendiente_por_anio": np.nan, "cambio_pct_por_decada": np.nan}
    pend, ord_, _, _ = theilslopes(anual.to_numpy(), anual.index.to_numpy())
    ref = anual.median()
    return anual.to_frame(), {"pendiente_por_anio": pend, "ordenada": ord_,
                              "cambio_pct_por_decada": 100 * pend * 10 / ref if ref else np.nan}


def anomalias_mensuales(serie, col="n_arm"):
    """
    Anomalía de cada mes contra el promedio de ese mismo mes en los demás años (sin incluir el propio).
    Devuelve valor, referencia, anomalía (diferencia) y z (en desviaciones estándar).
    """
    _validar_col(col)
    m = _meses_completos(preparar(serie))[col]
    x = m.groupby([m.index.year, m.index.month]).sum().unstack()          # años x meses
    cnt = x.notna().sum()
    s1, s2 = x.sum(), (x ** 2).sum()
    n_ex = cnt - 1
    suma_ex = s1 - x
    media_ex = suma_ex.div(n_ex.where(n_ex > 0))
    var_ex = (s2 - x ** 2 - suma_ex ** 2 / n_ex.where(n_ex > 0)).div(
        (n_ex - 1).where(n_ex > 1)
    )
    std_ex = np.sqrt(var_ex.clip(lower=0))
    ok = n_ex >= MIN_ANIOS_REF
    out = pd.DataFrame({
        "valor": x.stack(),
        "referencia": media_ex.where(ok, np.nan).stack(),
        "std_ref": std_ex.where(ok, np.nan).stack(),
    })
    out["anomalia"] = out["valor"] - out["referencia"]
    out["z"] = out["anomalia"] / out["std_ref"].replace(0, np.nan)
    out.index.names = ["anio", "mes"]
    return out.reset_index()


def acumulado_temporada(serie, anio, col="n_arm"):
    """
    Curva acumulada del año elegido contra los demás años completos.
    Devuelve las curvas, las bandas p10/p50/p90 y el lugar del año a la fecha de corte.
    """
    _validar_col(col)
    s = preparar(serie)
    d = s[col].to_frame("v")
    d["anio"] = d.index.year
    # Alinea años bisiestos y no bisiestos en un calendario común de 366 días.
    ajuste = (~d.index.is_leap_year) & (d.index.month > 2)
    d["doy"] = d.index.dayofyear + ajuste.astype(int)
    por_dia = d.groupby(["anio", "doy"])["v"].sum().unstack(fill_value=0.0)
    por_dia = por_dia.reindex(columns=range(1, 367), fill_value=0.0)
    acum = por_dia.cumsum(axis=1)

    ref_anios = [a for a in _anios_completos(s) if a != anio]
    if len(ref_anios) < MIN_ANIOS_REF or anio not in acum.index:
        raise ValueError("No hay suficientes años de referencia para comparar.")
    ref = acum.loc[ref_anios]

    corte = int(d.loc[d["anio"] == anio, "doy"].max())
    actual = acum.loc[anio, :corte]
    en_corte = ref[corte]
    lugar = 1 + int((en_corte > actual.iloc[-1]).sum())
    return {
        "actual": actual,
        "p10": ref.quantile(0.10), "p50": ref.quantile(0.50), "p90": ref.quantile(0.90),
        "corte_doy": corte,
        "lugar": lugar, "de": len(ref_anios) + 1,
        "percentil": 100 * float((en_corte <= actual.iloc[-1]).mean()),
        "vs_mediana_pct": 100 * (actual.iloc[-1] / ref[corte].median() - 1) if ref[corte].median() else np.nan,
    }


# ---------------------------------------------------------------- gráficas (devuelven figuras)
def _titulo_col(col):
    _validar_col(col)
    return "Celdas con fuego (armonizado)" if col == "n_arm" else "FRP total armonizado (MW)"


def fig_calendario(serie, col="n_arm", cmap="viridis"):
    """Calendario de actividad: año (filas) contra semana del año (columnas)."""
    _validar_col(col)
    s = preparar(serie)
    d = s[col].to_frame("v")
    d["anio"], d["sem"] = d.index.year, np.minimum((d.index.dayofyear - 1) // 7, 51)
    m = d.groupby(["anio", "sem"])["v"].sum().unstack()
    fig, ax = plt.subplots(figsize=(11, 0.28 * len(m) + 1.6))
    valores = m.to_numpy()
    vmax = float(np.nanmax(valores)) if valores.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    im = ax.imshow(valores, aspect="auto", cmap=cmap,
                   norm=matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=vmax))
    ax.set_yticks(range(len(m)), m.index)
    meses = ["E", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    ax.set_xticks([round(i * 52 / 12) for i in range(12)], meses)
    ax.set_title(f"Calendario de actividad: {_titulo_col(col)}", loc="left")
    fig.colorbar(im, ax=ax, label="Total semanal", pad=0.01)
    fig.tight_layout()
    return fig


def fig_estacionalidad(est, col="n_arm"):
    _validar_col(col)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = est.index
    ax.bar(x, est["media"], color=AZUL, alpha=0.85, label="Promedio")
    ax.vlines(x, est["p10"], est["p90"], color=GRIS, lw=2, label="Rango p10 a p90")
    ax.set_xticks(x, ["E", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_ylabel("Total mensual")
    ax.set_title(f"Estacionalidad: {_titulo_col(col)}", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def fig_anomalias(anom, anio, col="n_arm"):
    _validar_col(col)
    a = anom[anom["anio"] == anio].set_index("mes").reindex(range(1, 13))
    colores = np.where(a["z"].fillna(0) >= 0, NARANJA, AZUL)
    fig, ax = plt.subplots(figsize=(8, 4))
    presentes = a["z"].dropna()
    ax.bar(presentes.index, presentes, color=colores[presentes.index - 1])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(1, 13), ["E", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_ylabel("Desviaciones estándar vs. otros años")
    ax.set_title(f"Anomalía mensual {anio}: {_titulo_col(col)}", loc="left")
    fig.tight_layout()
    return fig


def fig_totales_anuales(anual, tend, col="n_arm"):
    _validar_col(col)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(anual.index, anual["total"], color=AZUL, alpha=0.85)
    if np.isfinite(tend["pendiente_por_anio"]):
        xs = np.array([anual.index.min(), anual.index.max()])
        ax.plot(xs, tend["ordenada"] + tend["pendiente_por_anio"] * xs, color=NARANJA, lw=2,
                label=f"Tendencia: {tend['cambio_pct_por_decada']:+.1f}% por década")
        ax.legend(frameon=False)
    ax.set_ylabel("Total anual")
    ax.set_title(f"Totales anuales: {_titulo_col(col)}", loc="left")
    fig.tight_layout()
    return fig


def fig_acumulado(res, anio, col="n_arm"):
    _validar_col(col)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = res["p50"].index
    ax.fill_between(x, res["p10"], res["p90"], color=GRIS, alpha=0.25, label="Años anteriores (p10 a p90)")
    ax.plot(x, res["p50"], color=GRIS, lw=1.5, label="Mediana de años anteriores")
    ax.plot(res["actual"].index, res["actual"], color=NARANJA, lw=2.5, label=str(anio))
    ax.set_xlabel("Día del año")
    ax.set_ylabel("Acumulado")
    ax.set_title(f"Temporada {anio}: {_titulo_col(col)}, lugar {res['lugar']} de {res['de']} a la fecha "
                 f"({res['vs_mediana_pct']:+.0f}% vs. mediana)", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- demostración
def main(salida="figuras_demo", anio=2026):
    out = Path(salida)
    out.mkdir(parents=True, exist_ok=True)
    serie = generar_serie_falsa()

    est = estacionalidad(serie)
    anual, tend = totales_anuales(serie)
    anom = anomalias_mensuales(serie)
    acum = acumulado_temporada(serie, anio)

    print("Estacionalidad (media mensual):\n", est["media"].round(0).to_string())
    print("\nTendencia:", {k: round(float(v), 2) for k, v in tend.items()})
    print(f"\nTemporada {anio}: lugar {acum['lugar']} de {acum['de']}, "
          f"percentil {acum['percentil']:.0f}, {acum['vs_mediana_pct']:+.0f}% vs. mediana")

    fig_calendario(serie).savefig(out / "calendario.png", dpi=130)
    fig_estacionalidad(est).savefig(out / "estacionalidad.png", dpi=130)
    fig_anomalias(anom, anio - 1).savefig(out / "anomalias.png", dpi=130)
    fig_totales_anuales(anual, tend).savefig(out / "totales_anuales.png", dpi=130)
    fig_acumulado(acum, anio).savefig(out / "acumulado.png", dpi=130)
    anom.to_csv(out / "anomalias_mensuales.csv", index=False)
    print("\nFiguras en", out.resolve())


if __name__ == "__main__":
    main()
