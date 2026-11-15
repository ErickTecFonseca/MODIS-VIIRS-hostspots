"""
Análisis de patrones y gráficas sobre una serie diaria de actividad de incendios.

Flujo completo:
    parquet o CSV de puntos -> leer_puntos -> limpiar_puntos -> puntos_a_celdas -> serie_region
    -> análisis y gráficas

La entrada de los análisis es una tabla con una fila por día de una región:
    fecha, n_arm (celdas con fuego armonizadas), frp_arm (energía total armonizada)
Los días sin incendio deben contar como cero, por eso serie_region() y preparar() rellenan los huecos.

Cada análisis es una función que devuelve datos (DataFrame o diccionario) y cada gráfica
es otra función que recibe esos datos y devuelve una figura. La API puede llamar solo
a las primeras, y la interfaz o el CSV de descarga usan sus resultados.

Pruebas rápidas:
    python analisis_incendiosV3.py                                      # datos simulados
    python analisis_incendiosV3.py --modis modis.parquet --viirs viirs.csv --pais Peru
    python analisis_incendiosV3.py --puntos viirs.csv --sensor VIIRS --pais Peru

Las fuentes separadas quedan etiquetadas para calcular su proporción de celdas-día.
No se aplica calibración VIIRS: hasta definirla, los resultados combinados se reportan
como no armonizados.
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # sin ventana, para poder guardar imágenes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import theilslopes

AZUL, NARANJA, GRIS = "#00F2FF", "#FB730A", "#6b6b6b"   # par de colores seguro para daltonismo
MIN_ANIOS_REF = 5                                          # historia mínima para comparar


# ---------------------------------------------------------------- datos de prueba
def generar_serie_falsa(inicio="2000-11-01", fin="2026-11-14", semilla=1):
    """Serie diaria inventada: temporada alta en septiembre, años fuertes y poca tendencia."""
    rng = np.random.default_rng(semilla)
    dias = pd.date_range(inicio, fin, freq="D")
    doy = dias.day_of_year.to_numpy()
    temporada = 1 + 0.9 * np.sin((doy - 170) / 365 * 2 * np.pi)       # pico hacia septiembre
    efecto_anio = {a: rng.lognormal(0, 0.35) for a in range(dias.year.min(), dias.year.max() + 1)}
    for a in (2005, 2010, 2019, 2024):                                  # años excepcionales
        if a in efecto_anio:
            efecto_anio[a] *= 1.8
    f_anio = dias.year.map(efecto_anio).to_numpy()
    tendencia = 1 + 0.008 * (dias.year.to_numpy() - 2000)
    n = rng.poisson(np.clip(40 * temporada**2 * f_anio * tendencia, 0.2, None))
    frp = n * rng.gamma(4, 6, len(dias))
    return pd.DataFrame({"fecha": dias, "n_arm": n.astype(float), "frp_arm": frp})


# ---------------------------------------------------------------- de puntos a serie diaria
COLUMNAS_PUNTOS = ["latitude", "longitude", "acq_date", "confidence", "type", "frp", "country",
                   "sensor", "instrument"]
CONFIANZA_LETRAS = {"n", "h", "nominal", "high"}      # VIIRS: nominal y alta


def leer_puntos(ruta, columnas=COLUMNAS_PUNTOS):
    """
    Lee solo las columnas que se usan. En el parquet de ejemplo, latitud y longitud en float64
    ocupan casi dos tercios del archivo, así que conviene no cargar nada de más.
    """
    ruta = str(ruta)
    if ruta.endswith(".parquet"):
        # Se piden solo las columnas que existen; si falta una obligatoria, limpiar_puntos lo dice.
        # Antes un except general escondía cualquier error (archivo dañado, falta de type, etc.).
        try:
            import pyarrow.parquet as pq
            disponibles = set(pq.read_schema(ruta).names)
            columnas = [c for c in columnas if c in disponibles]
        except ImportError:
            pass
        return pd.read_parquet(ruta, columns=list(columnas))
    return pd.read_csv(ruta, usecols=lambda c: c in columnas, dtype={"confidence": str})


def limpiar_puntos(df, conf_min=30):
    """
    Aplica la misma regla de calidad para los dos sensores:
    confianza de 30 o más (MODIS) o nominal y alta (VIIRS), solo incendios de vegetación (type 0).
    """
    faltantes = {"latitude", "longitude", "acq_date", "confidence", "frp"}.difference(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas: {', '.join(sorted(faltantes))}")

    if pd.api.types.is_numeric_dtype(df["confidence"]):
        ok = df["confidence"] >= conf_min
    else:
        c = df["confidence"].astype(str).str.strip().str.lower()
        ok = (pd.to_numeric(c, errors="coerce") >= conf_min) | c.isin(CONFIANZA_LETRAS)

    latitud = pd.to_numeric(df["latitude"], errors="coerce")
    longitud = pd.to_numeric(df["longitude"], errors="coerce")
    frp = pd.to_numeric(df["frp"], errors="coerce")
    # acq_date ya es fecha UTC; se quita la zona para que coincida con las demás tablas
    fechas = pd.to_datetime(df["acq_date"], errors="coerce", utc=True).dt.tz_convert(None)
    ok &= latitud.between(-90, 90).fillna(False)
    ok &= longitud.between(-180, 180).fillna(False)
    ok &= np.isfinite(frp.to_numpy(dtype=float))
    ok &= fechas.notna()
    if "type" in df.columns:
        # Un type vacío casi siempre significa que ese archivo no traía la columna (al unir MODIS
        # con un VIIRS sin type, todo VIIRS quedaba en NaN y se descartaba). Solo se excluye
        # lo que trae un type distinto de 0.
        tipo = pd.to_numeric(df["type"], errors="coerce")
        ok &= tipo.isna() | tipo.eq(0)

    # Se asigna por posición y se reinicia el índice: si varios CSV se unieron sin
    # ignore_index, el índice repetido puede romper la alineación en algunas versiones de pandas.
    m = ok.to_numpy()
    d = df.loc[m].reset_index(drop=True)
    d["latitude"] = latitud.to_numpy()[m]
    d["longitude"] = longitud.to_numpy()[m]
    d["acq_date"] = fechas.to_numpy()[m]
    d["frp"] = np.clip(frp.to_numpy()[m], 0, None)       # el parquet trae algún -0.0
    return d


def indice_celda(valor, escala=100):
    """
    floor(valor * escala) sin errores de punto flotante: 40.12 * 100 da 4011.9999... y caía en la
    celda 4011. En el parquet de MODIS 2011 pasaba con unos 2,600 puntos por eje.
    """
    v = np.round(np.asarray(valor, dtype=float) * escala, 6)
    return np.floor(v).astype("int32")


def normalizar_sensor(serie):
    """Lleva cualquier etiqueta (MODIS, modis, VIIRS-SNPP, instrument de FIRMS) a MODIS, VIIRS o DESCONOCIDO."""
    t = serie.astype("string").fillna("").str.strip().str.upper()
    return pd.Series(np.select([t.str.contains("VIIRS"), t.str.contains("MODIS")],
                               ["VIIRS", "MODIS"], "DESCONOCIDO"), index=serie.index)


def puntos_a_celdas(df):
    """
    Una fila por celda de 0.01 grados y día (con país, si existe).
    La celda se identifica con índices enteros: ci = floor(lat*100), cj = floor(lon*100).
    """
    d = pd.DataFrame({
        "fecha": pd.to_datetime(df["acq_date"]).dt.normalize().to_numpy(),
        "ci": indice_celda(df["latitude"]),
        "cj": indice_celda(df["longitude"]),
        "frp": df["frp"].to_numpy(),
    })
    llaves = ["fecha", "ci", "cj"]
    if "country" in df.columns:
        # Sin esto, las filas sin país se perdían en el groupby sin avisar
        d["pais"] = df["country"].astype("string").fillna("Desconocido").astype(str).to_numpy()
        llaves.insert(0, "pais")
    origen = "sensor" if "sensor" in df.columns else ("instrument" if "instrument" in df.columns else None)
    if origen:
        d["sensor"] = normalizar_sensor(df[origen]).to_numpy()
        llaves.insert(0, "sensor")
    return d.groupby(llaves, as_index=False, dropna=False).agg(puntos=("frp", "size"), frp=("frp", "sum"))


def top_paises(celdas, n=10):
    """Países con más celdas-día de fuego: sirve para el selector de la app."""
    if "pais" not in celdas.columns:
        raise ValueError("La tabla de celdas no tiene columna pais.")
    claves = [c for c in ("pais", "fecha", "ci", "cj") if c in celdas.columns]
    datos = celdas.drop_duplicates(claves) if len(claves) == 4 else celdas
    return (datos.groupby("pais").size().sort_values(ascending=False).head(n)
            .rename("celdas_dia").reset_index())


def tramo_cubierto(celdas):
    """
    Periodo continuo con datos. Un mes calendario sin ninguna detección en toda la tabla se toma
    como falta de archivo, no como ausencia de fuego (en MODIS 2011 no hubo un solo día del año
    sin detecciones en el mundo, y en un país la racha más larga sin fuego fue de 11 días).
    Si hay huecos, se usa el tramo continuo más largo (el más reciente si empatan) y se avisa.
    """
    fechas = pd.to_datetime(celdas["fecha"])
    meses = pd.PeriodIndex(fechas.dt.to_period("M").unique()).sort_values()
    todos = pd.period_range(meses.min(), meses.max(), freq="M")
    huecos = todos.difference(meses)
    inicio, fin = fechas.min().normalize(), fechas.max().normalize()
    if len(huecos) == 0:
        return inicio, fin
    cubierto = todos.isin(meses)
    tramos, ini = [], None
    for i, ok in enumerate(cubierto):
        if ok and ini is None:
            ini = i
        if (not ok or i == len(cubierto) - 1) and ini is not None:
            fin_i = i if ok else i - 1
            tramos.append((fin_i - ini + 1, fin_i, ini))
            ini = None
    largo, fin_i, ini = max(tramos)
    a = max(todos[ini].start_time, inicio)
    b = min(todos[fin_i].end_time.normalize(), fin)
    warnings.warn(f"Hay {len(huecos)} mes(es) sin ninguna detección en los datos ({huecos[0]} a {huecos[-1]}); "
                  f"se tratan como meses sin archivo y se usa el tramo {a.date()} a {b.date()}.")
    return a, b


def seleccionar_celdas(celdas, pais=None, bbox=None):
    """
    Celdas de una región, listas para contar:
    - con MODIS y VIIRS juntos, cada día usa un solo sensor (VIIRS desde su primera fecha, MODIS antes);
      sin esto, una celda vista por los dos contaba doble.
    - una celda en la frontera de dos países cuenta una vez al no filtrar por país (el FRP se suma).
    - se recorta al tramo con datos (ver tramo_cubierto).
    """
    c = celdas
    if pais is not None:
        if "pais" not in c.columns:
            raise ValueError("La tabla no tiene país; usa bbox.")
        c = c[c["pais"] == pais]
        if c.empty:
            ejemplos = ", ".join(sorted(celdas["pais"].unique())[:5])
            raise ValueError(f"No hay detecciones para '{pais}'. Los nombres van como en la columna "
                             f"country, con guiones bajos (por ejemplo: {ejemplos}).")
    if bbox is not None:
        lat0, lat1, lon0, lon1 = map(float, bbox)
        if not (-90 <= lat0 < lat1 <= 90 and -180 <= lon0 < lon1 <= 180):
            raise ValueError("bbox debe ser (lat_min, lat_max, lon_min, lon_max) con min < max "
                             "y dentro de los rangos válidos (no cruza el meridiano 180).")
        i0, i1 = indice_celda([lat0, lat1])
        j0, j1 = indice_celda([lon0, lon1])
        c = c[(c["ci"] >= i0) & (c["ci"] < i1) & (c["cj"] >= j0) & (c["cj"] < j1)]

    if "sensor" in celdas.columns:
        sens = celdas["sensor"]
        if (sens == "VIIRS").any() and (sens == "MODIS").any():
            inicio_viirs = celdas.loc[sens == "VIIRS", "fecha"].min()
            usar = np.where(c["fecha"] >= inicio_viirs, "VIIRS", "MODIS")
            c = c[(c["sensor"] == usar) | ~c["sensor"].isin(["MODIS", "VIIRS"])]

    llaves = [k for k in ("sensor", "fecha", "ci", "cj") if k in c.columns]
    c = c.groupby(llaves, as_index=False).agg(frp=("frp", "sum"))
    if "sensor" in c.columns:
        # si quedan sensores desconocidos junto a uno conocido en la misma celda-día, cuenta una vez
        prioridad = c["sensor"].map({"VIIRS": 0, "MODIS": 1}).fillna(2)
        c = (c.assign(_p=prioridad).sort_values("_p")
              .drop_duplicates(["fecha", "ci", "cj"]).drop(columns="_p"))

    a, b = tramo_cubierto(celdas)
    c = c[(c["fecha"] >= a) & (c["fecha"] <= b)]
    if c.empty:
        raise ValueError("La región no tiene detecciones en el periodo con datos.")
    return c.reset_index(drop=True), (a, b)


def serie_region(celdas, pais=None, bbox=None):
    """
    Serie diaria de una región (país o caja lat/lon) con ceros en los días sin fuego.
    bbox = (lat_min, lat_max, lon_min, lon_max). Devuelve fecha, n_arm y frp_arm.
    Aquí n_arm cuenta celdas-día sin calibrar: el factor MODIS/VIIRS se aplica antes, en la calibración.
    """
    c, (a, b) = seleccionar_celdas(celdas, pais, bbox)
    # El calendario se completa con el periodo que cubren los datos, no con el de la región,
    # si no una región cuyo primer fuego fue en febrero perdería enero.
    cobertura = pd.date_range(a, b, freq="D")
    s = c.groupby("fecha").agg(n_arm=("ci", "size"), frp_arm=("frp", "sum"))
    s = s.reindex(cobertura, fill_value=0.0).rename_axis("fecha").reset_index()
    s["n_arm"] = s["n_arm"].astype(float)
    return s


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


def indicador_confianza(serie, celdas=None):
    """Resume años completos y porcentaje de celdas-día por sensor; no infiere calibración."""
    s = preparar(serie)
    anios = _anios_completos(s)
    resultado = {
        "anios_completos": len(anios),
        "primer_anio_completo": anios[0] if anios else None,
        "ultimo_anio_completo": anios[-1] if anios else None,
        "porcentaje_modis": None,
        "porcentaje_viirs": None,
        "porcentaje_otros": None,
        "unidad_porcentaje": "celdas-dia",
        "calibracion_viirs": "No definida",
    }
    if celdas is None or "sensor" not in celdas.columns or celdas.empty:
        return resultado

    sensores = (celdas["sensor"].astype("string").fillna("Desconocido")
                .str.strip().str.upper())
    conteos = sensores.value_counts()
    total = int(conteos.sum())
    if total:
        resultado["porcentaje_modis"] = 100 * float(conteos.get("MODIS", 0)) / total
        resultado["porcentaje_viirs"] = 100 * float(conteos.get("VIIRS", 0)) / total
        resultado["porcentaje_otros"] = 100 * float(
            total - conteos.get("MODIS", 0) - conteos.get("VIIRS", 0)
        ) / total
    return resultado


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
    # Meses con pocos años de referencia quedan sin comparación (se anulan por columna)
    pocos = n_ex < MIN_ANIOS_REF
    media_ex.loc[:, pocos] = np.nan
    std_ex.loc[:, pocos] = np.nan
    out = pd.DataFrame({
        "valor": x.stack(),
        "referencia": media_ex.stack(),
        "std_ref": std_ex.stack(),
    })
    out["anomalia"] = out["valor"] - out["referencia"]
    out["z"] = out["anomalia"] / out["std_ref"].replace(0, np.nan)
    out.index.names = ["anio", "mes"]
    return out.dropna(subset=["valor"]).reset_index()      # quita meses sin datos (antes o después de la serie)


def acumulado_temporada(serie, anio, col="n_arm", celdas=None):
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
    d["doy"] = d.index.day_of_year + ajuste.astype(int)
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
    valor_actual = actual.iloc[-1]
    menores = int((en_corte < valor_actual).sum())
    iguales = int((en_corte == valor_actual).sum())
    percentil = 100 * (menores + 0.5 * iguales) / len(en_corte)
    return {
        "actual": actual,
        "p10": ref.quantile(0.10), "p50": ref.quantile(0.50), "p90": ref.quantile(0.90),
        "corte_doy": corte,
        "lugar": lugar, "de": len(ref_anios) + 1,
        "percentil": percentil,
        "semaforo": semaforo_temporada(percentil),
        "historia_anios": len(_anios_completos(s)),
        "confianza": indicador_confianza(serie, celdas),
        "vs_mediana_pct": 100 * (actual.iloc[-1] / ref[corte].median() - 1) if ref[corte].median() else np.nan,
    }


def semaforo_temporada(percentil):
    """
    Clasifica el percentil de actividad a la fecha:
    por debajo de lo normal (<25), normal (25 a 75), por encima de lo normal (75 a 90), excepcional (>90).
    Antes, cualquier año arriba de la mediana salía "por encima" (la mitad de los años) y uno muy bajo
    salía "normal".
    """
    if not np.isfinite(percentil) or not 0 <= percentil <= 100:
        raise ValueError("El percentil debe ser un número finito entre 0 y 100.")
    if percentil > 90:
        return "excepcional"
    if percentil > 75:
        return "por encima de lo normal"
    if percentil >= 25:
        return "normal"
    return "por debajo de lo normal"


def _semana_anio(indice):
    dias_anio = np.where(indice.is_leap_year, 366, 365)
    return (indice.day_of_year - 1) * 52 // dias_anio + 1


def fenologia_temporadas(serie, col="n_arm"):
    """Fechas 10/90% y semanas pico en 52 semanas normalizadas; marca años parciales."""
    _validar_col(col)
    s = preparar(serie)
    completos = set(_anios_completos(s))
    filas = []
    for anio, valores in s[col].groupby(s.index.year):
        fechas = valores.index
        total = float(valores.sum())
        semanas = _semana_anio(fechas)
        if total <= 0:
            filas.append({
                "anio": int(anio), "total": total, "anio_completo": anio in completos,
                "inicio_fecha": pd.NaT, "inicio_semana": pd.NA,
                "pico_fecha": pd.NaT, "pico_semana": pd.NA,
                "fin_fecha": pd.NaT, "fin_semana": pd.NA,
            })
            continue

        acumulado = valores.cumsum().to_numpy()
        inicio_pos = int(np.flatnonzero(acumulado >= 0.10 * total)[0])
        fin_pos = int(np.flatnonzero(acumulado >= 0.90 * total)[0])
        por_semana = valores.groupby(semanas).sum()
        pico_semana = int(por_semana.idxmax())
        posiciones_pico = np.flatnonzero(semanas == pico_semana)
        pico_pos = posiciones_pico[int(np.argmax(valores.to_numpy()[posiciones_pico]))]
        pico_fecha = fechas[pico_pos]
        filas.append({
            "anio": int(anio), "total": total, "anio_completo": anio in completos,
            "inicio_fecha": fechas[inicio_pos], "inicio_semana": int(semanas[inicio_pos]),
            "pico_fecha": pico_fecha, "pico_semana": pico_semana,
            "fin_fecha": fechas[fin_pos], "fin_semana": int(semanas[fin_pos]),
        })
    return pd.DataFrame(filas)


def ranking_extremos(serie, top=10, frecuencia="dia"):
    """Top de días o semanas por celdas con fuego; semanas van de lunes a domingo."""
    if isinstance(top, bool) or not isinstance(top, (int, np.integer)) or top <= 0:
        raise ValueError("top debe ser un entero positivo.")
    if frecuencia not in {"dia", "semana"}:
        raise ValueError("frecuencia debe ser 'dia' o 'semana'.")
    s = preparar(serie)
    if frecuencia == "dia":
        eventos = s.loc[s["n_arm"] > 0].reset_index(names="fecha")
        return (eventos.sort_values(["n_arm", "frp_arm", "fecha"],
                                    ascending=[False, False, True]).head(top).reset_index(drop=True))

    semanal = s.resample("W-MON", label="left", closed="left").sum()
    semanal = semanal.loc[semanal["n_arm"] > 0].reset_index(names="semana_inicio")
    semanal["semana_fin"] = semanal["semana_inicio"] + pd.Timedelta(days=6)
    return (semanal.sort_values(["n_arm", "frp_arm", "semana_inicio"],
                                ascending=[False, False, True]).head(top).reset_index(drop=True))


def intensidad_media(serie):
    """FRP media por celda-día en el periodo; NaN si no hubo actividad."""
    s = preparar(serie)
    celdas = float(s["n_arm"].sum())
    return float(s["frp_arm"].sum() / celdas) if celdas else np.nan


def mapa_recurrencia(celdas):
    """Años distintos con fuego por celda; conserva país, si está disponible."""
    requeridas = {"fecha", "ci", "cj"}
    faltantes = requeridas.difference(celdas.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas de celdas: {', '.join(sorted(faltantes))}")
    if celdas.empty:
        raise ValueError("La tabla de celdas está vacía.")
    d = celdas.loc[:, [c for c in ("pais", "ci", "cj", "fecha") if c in celdas.columns]].copy()
    d["fecha"] = pd.to_datetime(d["fecha"], errors="raise", utc=True).dt.tz_convert(None)
    d["anio"] = d["fecha"].dt.year
    llaves = [c for c in ("pais", "ci", "cj") if c in d.columns]
    resultado = (d.groupby(llaves, dropna=False)
                 .agg(anios_con_fuego=("anio", "nunique"), primera_fecha=("fecha", "min"),
                      ultima_fecha=("fecha", "max"))
                 .reset_index())
    return (resultado.sort_values(["anios_con_fuego"] + llaves,
                                  ascending=[False] + [True] * len(llaves))
            .reset_index(drop=True))


def _curva_acumulada_anual(serie, anio, col):
    s = preparar(serie)
    d = s.loc[s.index.year == anio, col]
    if d.empty:
        raise ValueError(f"No hay datos para el año {anio}.")
    ajuste = (~d.index.is_leap_year) & (d.index.month > 2)
    doy = d.index.day_of_year + ajuste.astype(int)
    diaria = d.groupby(doy).sum().reindex(range(1, 367), fill_value=0.0)
    acumulado = diaria.cumsum()
    inicio = int(doy.min())
    fin = int(doy.max())
    acumulado.loc[(acumulado.index < inicio) | (acumulado.index > fin)] = np.nan
    return acumulado


def comparar_temporadas(serie_a, anio_a, serie_b, anio_b, col="n_arm"):
    """Alinea curvas acumuladas de dos regiones o años en un calendario común de 366 días."""
    _validar_col(col)
    a = _curva_acumulada_anual(serie_a, anio_a, col)
    b = _curva_acumulada_anual(serie_b, anio_b, col)
    return pd.DataFrame({"dia": range(1, 367), "acumulado_a": a.to_numpy(), "acumulado_b": b.to_numpy()})


# ---------------------------------------------------------------- gráficas (devuelven figuras)
def _titulo_col(col):
    _validar_col(col)
    return "Celdas con fuego" if col == "n_arm" else "FRP total (MW)"


def _matriz_calendario(serie, col):
    s = preparar(serie)
    d = s[col].to_frame("v")
    d["anio"], d["sem"] = d.index.year, _semana_anio(d.index) - 1
    # Semanas fuera del periodo con datos quedan vacías (blanco), no como cero fuego
    return d.groupby(["anio", "sem"])["v"].sum().unstack().reindex(columns=range(52))


def _dibujar_calendario(ax, matriz, cmap):
    valores = matriz.to_numpy()
    vmax = float(np.nanmax(valores)) if valores.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    im = ax.imshow(valores, aspect="auto", cmap=cmap,
                   norm=matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=vmax))
    ax.set_yticks(range(len(matriz)), matriz.index)
    meses = ["E", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    ax.set_xticks([round(i * 52 / 12) for i in range(12)], meses)
    return im


def fig_calendario(serie, col="n_arm", cmap="viridis"):
    """Calendario de actividad: año (filas) contra semana del año (columnas)."""
    _validar_col(col)
    m = _matriz_calendario(serie, col)
    fig, ax = plt.subplots(figsize=(11, 0.28 * len(m) + 1.6))
    im = _dibujar_calendario(ax, m, cmap)
    ax.set_title(f"Calendario de actividad: {_titulo_col(col)}", loc="left")
    fig.colorbar(im, ax=ax, label="Total semanal", pad=0.01)
    fig.tight_layout()
    return fig


def fig_serie_diaria(serie, col="n_arm", ventana=7):
    """Serie diaria con media móvil: útil cuando hay pocos años para comparar."""
    _validar_col(col)
    v = preparar(serie)[col]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(v.index, v.to_numpy(), color=AZUL, alpha=0.55, width=1.0, label="Diario")
    ax.plot(v.index, v.rolling(ventana, center=True, min_periods=1).mean(), color=NARANJA, lw=2,
            label=f"Media de {ventana} días")
    ax.set_ylabel("Total diario")
    ax.set_title(f"Serie diaria: {_titulo_col(col)}", loc="left")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return fig


def fig_estacionalidad(est, col="n_arm"):
    _validar_col(col)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = est.index
    ax.bar(x, est["media"], color=AZUL, alpha=0.85, label="Promedio")
    ax.vlines(x, est["p10"], est["p90"], color=GRIS, lw=2, label="Rango p10 a p90")
    ax.set_xticks(range(1, 13), ["E", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
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
    _dibujar_curva_acumulada(ax, res, anio, col)
    fig.tight_layout()
    return fig


def _dibujar_curva_acumulada(ax, res, anio, col):
    x = res["p50"].index
    ax.fill_between(x, res["p10"], res["p90"], color=GRIS, alpha=0.25, label="Años anteriores (p10 a p90)")
    ax.plot(x, res["p50"], color=GRIS, lw=1.5, label="Mediana de años anteriores")
    ax.plot(res["actual"].index, res["actual"], color=NARANJA, lw=2.5, label=str(anio))
    ax.set_xlabel("Día del año")
    ax.set_ylabel("Acumulado")
    vs = res["vs_mediana_pct"]
    vs_txt = f"{vs:+.0f}% vs. mediana" if np.isfinite(vs) else "sin mediana de referencia"
    ax.set_title(f"Temporada {anio}: {_titulo_col(col)}, {res['semaforo']} "
                 f"(lugar {res['lugar']} de {res['de']}; {vs_txt})", loc="left")
    ax.legend(frameon=False)


def fig_comparar_temporadas(comparacion, etiqueta_a="Región A", etiqueta_b="Región B", col="n_arm"):
    """Dibuja dos curvas acumuladas ya alineadas por día de temporada."""
    _validar_col(col)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(comparacion["dia"], comparacion["acumulado_a"], color=AZUL, lw=2.2, label=etiqueta_a)
    ax.plot(comparacion["dia"], comparacion["acumulado_b"], color=NARANJA, lw=2.2, label=etiqueta_b)
    ax.set_xlabel("Día del año (alineado a 366 días)")
    ax.set_ylabel("Acumulado")
    ax.set_title(f"Comparación de temporadas: {_titulo_col(col)}", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def fig_reporte_region(serie, anio, region="Región", celdas=None, col="n_arm"):
    """Reporte de una página: semáforo, calendario, curva acumulada y cifras del periodo."""
    _validar_col(col)
    resultado = acumulado_temporada(serie, anio, col, celdas)
    s = preparar(serie)
    datos_anio = s.loc[s.index.year == anio]
    if datos_anio.empty:
        raise ValueError(f"No hay datos para el año {anio}.")

    estado = resultado["semaforo"]
    colores = {"por debajo de lo normal": "#4575b4", "normal": "#2ca25f",
               "por encima de lo normal": "#f0ad4e", "excepcional": "#d73027"}
    total_celdas = float(datos_anio["n_arm"].sum())
    total_frp = float(datos_anio["frp_arm"].sum())
    frp_medio = intensidad_media(datos_anio.reset_index(names="fecha"))

    fig = plt.figure(figsize=(11.7, 8.3), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[0.8, 2.2, 1.8])
    ax_head = fig.add_subplot(grid[0, :])
    ax_head.axis("off")
    ax_head.text(0.01, 0.82, f"Reporte de incendios — {region} — {anio}",
                 fontsize=17, weight="bold", transform=ax_head.transAxes)
    ax_head.text(0.01, 0.37, f"SEMÁFORO: {estado.upper()}  |  Percentil {resultado['percentil']:.0f}",
                 fontsize=13, weight="bold", color="white", transform=ax_head.transAxes,
                 bbox={"boxstyle": "round,pad=0.45", "facecolor": colores[estado], "edgecolor": "none"})

    ax_cal = fig.add_subplot(grid[1, 0])
    im = _dibujar_calendario(ax_cal, _matriz_calendario(serie, col), "viridis")
    ax_cal.set_title("Calendario semanal", loc="left")
    fig.colorbar(im, ax=ax_cal, label="Total semanal", pad=0.01)
    ax_curve = fig.add_subplot(grid[1, 1])
    _dibujar_curva_acumulada(ax_curve, resultado, anio, col)
    ax_curve.set_title(f"Curva acumulada — {estado}", loc="left")

    ax_stats = fig.add_subplot(grid[2, :])
    ax_stats.axis("off")
    metricas = (
        ("Celdas-día", f"{total_celdas:,.0f}"),
        ("FRP total (MW)", f"{total_frp:,.1f}"),
        ("FRP media por celda (MW)", f"{frp_medio:,.2f}" if np.isfinite(frp_medio) else "Sin actividad"),
    )
    for i, (etiqueta, valor) in enumerate(metricas):
        x = 0.03 + i * 0.32
        ax_stats.text(x, 0.66, etiqueta, fontsize=11, color=GRIS, transform=ax_stats.transAxes)
        ax_stats.text(x, 0.36, valor, fontsize=19, weight="bold", transform=ax_stats.transAxes)
    return fig


def exportar_reporte(serie, ruta, anio, region="Región", celdas=None, col="n_arm"):
    """Exporta el reporte a PDF o imagen según la extensión de ruta y cierra la figura."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig = fig_reporte_region(serie, anio, region, celdas, col)
    try:
        fig.savefig(ruta, bbox_inches="tight")
    finally:
        plt.close(fig)


# ---------------------------------------------------------------- ejecución
def guardar(fig, ruta, dpi=130):
    """Guarda y cierra la figura. Sin cerrar, pyplot las retiene y en una API la memoria crece con cada consulta."""
    fig.savefig(ruta, dpi=dpi)
    plt.close(fig)


def _leer_fuentes(puntos=None, modis=None, viirs=None, sensor=None):
    if puntos and (modis or viirs):
        raise ValueError("Usa --puntos o las rutas --modis/--viirs, no ambos formatos a la vez.")
    if sensor is not None and not puntos:
        raise ValueError("--sensor solo se usa junto con --puntos.")
    if sensor is not None and sensor.upper() not in {"MODIS", "VIIRS"}:
        raise ValueError("sensor debe ser MODIS o VIIRS.")

    if puntos:
        datos = leer_puntos(puntos)
        if sensor is not None:
            datos["sensor"] = sensor.upper()
        return datos

    fuentes = []
    for nombre, ruta in (("MODIS", modis), ("VIIRS", viirs)):
        if ruta:
            datos = leer_puntos(ruta)
            datos["sensor"] = nombre
            fuentes.append(datos)
    return pd.concat(fuentes, ignore_index=True) if fuentes else None


def main(salida="figuras_demo", anio=2026, puntos=None, pais=None, modis=None, viirs=None, sensor=None):
    if pais is not None and not puntos:
        if not (modis or viirs):
            raise ValueError("--pais requiere proporcionar --puntos, --modis o --viirs.")

    datos_puntos = _leer_fuentes(puntos, modis, viirs, sensor)
    out = Path(salida)
    out.mkdir(parents=True, exist_ok=True)
    celdas_region = None
    if datos_puntos is not None:
        celdas = puntos_a_celdas(limpiar_puntos(datos_puntos))
        if pais is None and "pais" in celdas.columns:
            print("Países con más actividad:\n", top_paises(celdas, 5).to_string(index=False))
        serie = serie_region(celdas, pais=pais)
        with warnings.catch_warnings():          # el aviso de huecos ya salió en serie_region
            warnings.simplefilter("ignore")
            celdas_region, _ = seleccionar_celdas(celdas, pais=pais)
        nombre_region = pais or "Región"
        anio = int(serie["fecha"].max().year)
    else:
        serie = generar_serie_falsa()
        nombre_region = "Región simulada"

    n_anios = len(_anios_completos(preparar(serie)))
    print(f"\nDías: {len(serie)} | años completos: {n_anios}")
    guardar(fig_calendario(serie), out / "calendario.png")
    guardar(fig_serie_diaria(serie), out / "serie_diaria.png")

    confianza = indicador_confianza(serie, celdas_region)
    print(f"Historia completa: {confianza['anios_completos']} años.")
    if confianza["porcentaje_modis"] is None:
        print("Composición MODIS/VIIRS: no disponible.")
    else:
        print(f"Origen de celdas-día: MODIS {confianza['porcentaje_modis']:.1f}%, "
              f"VIIRS {confianza['porcentaje_viirs']:.1f}% (sin calibración definida), "
              f"otros {confianza['porcentaje_otros']:.1f}%.")
        if confianza["porcentaje_viirs"] > 0:
            print("Aviso: los totales con VIIRS son datos sin calibración; no deben interpretarse como armonizados.")

    fenologia_temporadas(serie).to_csv(out / "fenologia_temporadas.csv", index=False)
    ranking_extremos(serie, top=10, frecuencia="dia").to_csv(out / "dias_extremos.csv", index=False)
    ranking_extremos(serie, top=10, frecuencia="semana").to_csv(out / "semanas_extremas.csv", index=False)
    if celdas_region is not None:
        mapa_recurrencia(celdas_region).to_csv(out / "recurrencia_celdas.csv", index=False)
    media_frp = intensidad_media(serie)
    if np.isfinite(media_frp):
        print(f"FRP medio por celda-día: {media_frp:.2f} MW.")

    if n_anios < MIN_ANIOS_REF:
        print(f"Con {n_anios} año(s) completo(s) no se calculan estacionalidad, tendencia ni anomalías "
              f"(se necesitan al menos {MIN_ANIOS_REF}).")
        mensual = _meses_completos(preparar(serie))
        print("\nTotal mensual de celdas con fuego:\n", mensual["n_arm"].round(0).astype(int).to_string())
        print("\nFiguras en", out.resolve())
        return

    est = estacionalidad(serie)
    anual, tend = totales_anuales(serie)
    anom = anomalias_mensuales(serie)
    anios_completos = _anios_completos(preparar(serie))
    anio_anomalia = max(anios_completos)
    anios_referencia = [a for a in anios_completos if a != anio]

    print("Estacionalidad (media mensual):\n", est["media"].round(0).to_string())
    print("\nTendencia:", {k: round(float(v), 2) for k, v in tend.items()})

    guardar(fig_estacionalidad(est), out / "estacionalidad.png")
    guardar(fig_totales_anuales(anual, tend), out / "totales_anuales.png")
    anom_anio = anom[anom["anio"] == anio_anomalia]
    if anom_anio["z"].notna().any():
        guardar(fig_anomalias(anom, anio_anomalia), out / "anomalias.png")
    else:
        print(f"\nNo se genera anomalía para {anio_anomalia}: se necesitan al menos "
              f"{MIN_ANIOS_REF} años completos de referencia.")

    if len(anios_referencia) >= MIN_ANIOS_REF:
        acum = acumulado_temporada(serie, anio, celdas=celdas_region)
        print(f"\nTemporada {anio}: lugar {acum['lugar']} de {acum['de']}, "
              f"percentil {acum['percentil']:.0f} ({acum['semaforo']}), "
              f"{acum['vs_mediana_pct']:+.0f}% vs. mediana")
        guardar(fig_acumulado(acum, anio), out / "acumulado.png")
        exportar_reporte(serie, out / "reporte_region.pdf", anio, nombre_region, celdas_region)
    else:
        print(f"\nNo se genera comparación acumulada para {anio}: hay {len(anios_referencia)} "
              f"años completos de referencia; se necesitan {MIN_ANIOS_REF}.")
    anom.to_csv(out / "anomalias_mensuales.csv", index=False)
    print("\nFiguras en", out.resolve())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Análisis de patrones de incendios.")
    ap.add_argument("--puntos", help="parquet o CSV de puntos (si se omite, usa datos simulados)")
    ap.add_argument("--modis", help="CSV o parquet de detecciones MODIS")
    ap.add_argument("--viirs", help="CSV o parquet de detecciones VIIRS")
    ap.add_argument("--sensor", choices=["MODIS", "VIIRS"],
                    help="etiqueta el archivo indicado en --puntos cuando no trae columna sensor")
    ap.add_argument("--pais", help="nombre del país tal como viene en la columna country")
    ap.add_argument("--salida", default="figuras_demo")
    a = ap.parse_args()
    if a.pais is not None and not (a.puntos or a.modis or a.viirs):
        ap.error("--pais requiere proporcionar --puntos, --modis o --viirs.")
    if a.puntos and (a.modis or a.viirs):
        ap.error("Usa --puntos o las rutas --modis/--viirs, no ambos formatos a la vez.")
    if a.sensor and not a.puntos:
        ap.error("--sensor solo se usa junto con --puntos.")
    main(salida=a.salida, puntos=a.puntos, pais=a.pais, modis=a.modis, viirs=a.viirs, sensor=a.sensor)
