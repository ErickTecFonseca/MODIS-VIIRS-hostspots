"""
Armoniza conteos de focos activos VIIRS con MODIS por zona de 5 grados y mes.

Cada CSV de entrada debe tener una fila por sensor, fecha y celda de 0.01 grados,
con las columnas date, ci, cj y frp. ci y cj son indices enteros: floor(lat*100)
y floor(lon*100). Los datos deben estar filtrados con el mismo criterio.


Ejemplo:
python calibracion.py --modis modis.csv --viirs viirs.csv \
    --anio-inicio 2013 --anio-fin 2019 --salida resultados
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Estas columnas son las llaves para ubicar una observacion espacio-temporal.
LLAVES_CELDA_DIA = ["date", "ci", "cj"]
LLAVES_ZONA_MES = ["zi", "zj", "mes"]
COLUMNAS_REQUERIDAS = {"date", "ci", "cj", "frp"}


# Prepara un CSV y falla temprano si no tiene el formato esperado.
def preparar_datos(df, sensor="sensor"):
    # Comprueba que las columnas basicas existan antes de procesar.
    faltantes = COLUMNAS_REQUERIDAS.difference(df.columns)
    if faltantes:
        raise ValueError(
            f"{sensor}: faltan columnas requeridas: {', '.join(sorted(faltantes))}"
        )
    if df.empty:
        raise ValueError(f"{sensor}: el archivo no contiene filas.")

    # Trabaja sobre una copia para no modificar el DataFrame recibido.
    datos = df.loc[:, ["date", "ci", "cj", "frp"]].copy()

    # Convierte las fechas a UTC, elimina la zona horaria y conserva solo el dia.
    fechas = pd.to_datetime(datos["date"], errors="raise", utc=True)
    datos["date"] = fechas.dt.tz_convert(None).dt.normalize()
    if datos["date"].isna().any():
        raise ValueError(f"{sensor}: date contiene fechas vacias o no validas.")

    # Convierte los indices de celda y FRP a numeros y rechaza valores no validos.
    for columna in ["ci", "cj", "frp"]:
        datos[columna] = pd.to_numeric(datos[columna], errors="raise")
        valores = datos[columna].to_numpy(dtype=float)
        if not np.isfinite(valores).all():
            raise ValueError(f"{sensor}: la columna {columna} contiene nulos o infinitos.")

    # ci y cj deben ser indices enteros, no coordenadas decimales sin convertir.
    for columna in ["ci", "cj"]:
        valores = datos[columna].to_numpy(dtype=float)
        if not np.equal(valores, np.floor(valores)).all():
            raise ValueError(
                f"{sensor}: {columna} debe contener indices enteros de celdas de 0.01 grados."
            )
        datos[columna] = datos[columna].astype("int64")

    # FRP no puede ser negativa; cero es valido si el producto de origen lo permite.
    if (datos["frp"] < 0).any():
        raise ValueError(f"{sensor}: FRP no puede contener valores negativos.")

    # Evita contar dos veces una misma celda activa en el mismo dia.
    duplicados = datos.duplicated(LLAVES_CELDA_DIA, keep=False)
    if duplicados.any():
        ejemplo = datos.loc[duplicados, LLAVES_CELDA_DIA].head(1).to_dict("records")[0]
        raise ValueError(
            f"{sensor}: hay celdas-día duplicadas. Consolida date, ci y cj antes "
            f"de calibrar. Ejemplo: {ejemplo}"
        )

    # Devuelve las filas ordenadas para facilitar la inspeccion y las uniones.
    return datos.sort_values(LLAVES_CELDA_DIA).reset_index(drop=True)


# Subpaso 2: asigna cada indice de 0.01 grados a una zona de 5 grados y a un mes.
def zona_mes(df):
    # Copia para que agregar columnas no altere los datos de entrada.
    datos = df.copy()

    # Quinientas celdas de 0.01 grados forman una zona de 5 grados.
    datos["zi"] = datos["ci"] // 500
    datos["zj"] = datos["cj"] // 500

    # Extrae el mes de la fecha ya validada y normalizada.
    datos["mes"] = datos["date"].dt.month
    return datos


# Subpaso 3: suma celdas-dia activas y FRP por ano, zona y mes.
def conteos(df):
    # Requiere el formato canonico para que cada fila represente una celda-dia.
    datos = preparar_datos(df, sensor="tabla")

    # Deriva la zona, el mes y el ano de cada observacion.
    datos = zona_mes(datos)
    datos["anio"] = datos["date"].dt.year

    # Como cada celda-dia es unica, el numero de filas es el conteo de actividad.
    return (
        datos.groupby(["anio", "zi", "zj", "mes"], as_index=False)
        .agg(n=("ci", "size"), frp=("frp", "sum"))
        .sort_values(["anio", "zi", "zj", "mes"])
        .reset_index(drop=True)
    )


# Subpasos 4 y 5: calcula M/V por zona y mes y lo suaviza hacia el factor global.
def factores(cm, cv, anio_ini, anio_fin, col="n", a=1000):
    # Verifica que la metrica y el intervalo de calibracion sean validos.
    if col not in {"n", "frp"}:
        raise ValueError("col debe ser 'n' para conteos o 'frp' para energia FRP.")
    if anio_ini > anio_fin:
        raise ValueError("anio_ini no puede ser mayor que anio_fin.")
    if not np.isfinite(a) or a < 0:
        raise ValueError("a debe ser un numero finito mayor o igual a cero.")

    # Conserva solo los anos acordados para entrenar los factores.
    modis = cm.loc[cm["anio"].between(anio_ini, anio_fin)]
    viirs = cv.loc[cv["anio"].between(anio_ini, anio_fin)]
    if modis.empty or viirs.empty:
        raise ValueError("No hay datos de ambos sensores dentro del periodo de calibracion.")

    # Suma cada metrica por zona y mes en todo el periodo de calibracion.
    m = modis.groupby(LLAVES_ZONA_MES)[col].sum().rename("M")
    v = viirs.groupby(LLAVES_ZONA_MES)[col].sum().rename("V")

    # Une zonas presentes en cualquiera de los sensores; una ausencia equivale a cero.
    tabla = m.to_frame().join(v, how="outer").fillna(0).reset_index()

    # Obtiene el factor global por mes para estabilizar zonas con pocas observaciones.
    globales = tabla.groupby("mes")[["M", "V"]].sum()
    meses_sin_viirs = globales.index[globales["V"] <= 0].tolist()
    if meses_sin_viirs:
        raise ValueError(
            "No se puede calcular factor global: VIIRS no tiene actividad en "
            f"los meses {meses_sin_viirs} del periodo de calibracion."
        )
    fg = (globales["M"] / globales["V"]).rename("fg")

    # Incorpora el factor global de cada mes a las filas por zona.
    tabla = tabla.merge(fg, left_on="mes", right_index=True, how="left")

    # Aplica el suavizado; a debe expresarse en las mismas unidades que col.
    denominador = tabla["V"] + a
    tabla["factor"] = tabla["fg"]
    con_datos = denominador > 0
    tabla.loc[con_datos, "factor"] = (
        tabla.loc[con_datos, "M"] + a * tabla.loc[con_datos, "fg"]
    ) / denominador.loc[con_datos]

    # No recorta factores automaticamente: un limite ocultaria problemas de datos.
    if not np.isfinite(tabla["factor"].to_numpy(dtype=float)).all():
        raise ValueError("Se produjo un factor no finito; revisa los datos de entrada.")

    # Devuelve factores zonales y globales, ambos necesarios al aplicar la calibracion.
    factores_zonales = tabla[LLAVES_ZONA_MES + ["M", "V", "factor"]]
    return factores_zonales, fg


# Subpaso 6: agrega VIIRS a celdas de 0.1 grados y aplica el factor correspondiente.
def aplicar(viirs, fac, fg, col="n"):
    # Permite calibrar conteos y FRP con sus propios factores.
    if col not in {"n", "frp"}:
        raise ValueError("col debe ser 'n' para conteos o 'frp' para energia FRP.")

    # Valida VIIRS y asegura que solo haya una fila por celda fina y dia.
    datos = preparar_datos(viirs, sensor="VIIRS")

    # Cada diez indices finos se forma una celda de 0.1 grados.
    datos["ci"] = datos["ci"] // 10
    datos["cj"] = datos["cj"] // 10

    # Suma las celdas finas dentro de cada celda gruesa y dia.
    diario = (
        datos.groupby(LLAVES_CELDA_DIA, as_index=False)
        .agg(n=("frp", "size"), frp=("frp", "sum"))
    )

    # Calcula zona y mes con la rejilla gruesa, que tiene 50 celdas por zona de 5 grados.
    diario["zi"] = diario["ci"] // 50
    diario["zj"] = diario["cj"] // 50
    diario["mes"] = diario["date"].dt.month

    # Busca primero el factor de zona; las zonas sin datos usan el factor global mensual.
    diario = diario.merge(
        fac[LLAVES_ZONA_MES + ["factor"]],
        how="left",
        on=LLAVES_ZONA_MES,
        validate="many_to_one",
    )
    diario["factor"] = diario["factor"].fillna(diario["mes"].map(fg))

    # Falla explicitamente si tampoco existe un factor global para alguno de los meses.
    if diario["factor"].isna().any():
        meses = sorted(diario.loc[diario["factor"].isna(), "mes"].unique().tolist())
        raise ValueError(
            f"No hay factores zonales ni globales para los meses {meses}; "
            "amplia el periodo de calibracion o revisa los datos."
        )

    # Multiplica la metrica seleccionada por el factor y conserva tambien la original.
    diario[f"{col}_arm"] = diario[col] * diario["factor"]
    return diario.sort_values(LLAVES_CELDA_DIA).reset_index(drop=True)


# Subpaso 7: compara MODIS, VIIRS crudo y VIIRS calibrado por ano y mes.
def validar(cm, cv, fac, fg, anio_fin, col="n", anio_ini=None):
    # Comprueba la metrica y define el primer ano del periodo de calibracion.
    if col not in {"n", "frp"}:
        raise ValueError("col debe ser 'n' para conteos o 'frp' para energia FRP.")
    if anio_ini is None:
        anio_ini = int(cm["anio"].min())
    if anio_ini > anio_fin:
        raise ValueError("anio_ini no puede ser mayor que anio_fin.")

    # Mantiene solo las columnas necesarias y une zonas presentes en ambos sensores.
    modis = cm[["anio", "zi", "zj", "mes", col]].rename(columns={col: f"{col}_m"})
    viirs = cv[["anio", "zi", "zj", "mes", col]].rename(columns={col: f"{col}_v"})
    tabla = modis.merge(
        viirs,
        how="outer",
        on=["anio", "zi", "zj", "mes"],
        validate="one_to_one",
    )

    # Una zona con actividad en un sensor y ninguna en el otro se compara contra cero.
    tabla[[f"{col}_m", f"{col}_v"]] = tabla[
        [f"{col}_m", f"{col}_v"]
    ].fillna(0)

    # Asigna factor zonal y, cuando falte, el factor global del mes.
    tabla = tabla.merge(
        fac[LLAVES_ZONA_MES + ["factor"]],
        how="left",
        on=LLAVES_ZONA_MES,
        validate="many_to_one",
    )
    tabla["factor"] = tabla["factor"].fillna(tabla["mes"].map(fg))
    if tabla["factor"].isna().any():
        meses = sorted(tabla.loc[tabla["factor"].isna(), "mes"].unique().tolist())
        raise ValueError(f"No hay factores para validar los meses {meses}.")

    # Calcula el valor VIIRS ajustado para cada ano, zona y mes.
    tabla["viirs_cal"] = tabla[f"{col}_v"] * tabla["factor"]

    # Separa los datos usados para calibrar de los anos posteriores de prueba.
    tabla["muestra"] = np.select(
        [
            tabla["anio"].between(anio_ini, anio_fin),
            tabla["anio"] > anio_fin,
        ],
        ["calibracion", "prueba"],
        default="fuera_calibracion",
    )

    # Suma zonas para obtener la comparacion mensual nacional o regional.
    mensual = (
        tabla.groupby(["muestra", "anio", "mes"], as_index=False)
        .agg(
            modis=(f"{col}_m", "sum"),
            viirs=(f"{col}_v", "sum"),
            viirs_cal=("viirs_cal", "sum"),
        )
        .sort_values(["anio", "mes"])
        .reset_index(drop=True)
    )
    return mensual


# Calcula correlacion y sesgo porcentual a partir de la tabla mensual de validar().
def metricas_validacion(mensual):
    # Las filas previas al entrenamiento se conservan en la tabla, pero no son prueba.
    datos = mensual[mensual["muestra"].isin(["calibracion", "prueba"])]
    resultados = []

    # Resume por separado el periodo de entrenamiento y el de prueba.
    for nombre, grupo in datos.groupby("muestra", sort=False):
        total_modis = grupo["modis"].sum()
        sesgo_crudo = (
            100 * (grupo["viirs"].sum() - total_modis) / total_modis
            if total_modis != 0
            else np.nan
        )
        sesgo_cal = (
            100 * (grupo["viirs_cal"].sum() - total_modis) / total_modis
            if total_modis != 0
            else np.nan
        )
        resultados.append(
            {
                "muestra": nombre,
                "meses": len(grupo),
                "correlacion": grupo["modis"].corr(grupo["viirs_cal"]),
                "sesgo_crudo_pct": sesgo_crudo,
                "sesgo_calibrado_pct": sesgo_cal,
            }
        )

    # Devuelve un resumen tabular que tambien se puede guardar como CSV.
    return pd.DataFrame(
        resultados,
        columns=[
            "muestra",
            "meses",
            "correlacion",
            "sesgo_crudo_pct",
            "sesgo_calibrado_pct",
        ],
    )


# Guarda los factores globales con el mes como columna para poder exportarlos a CSV.
def _globales_a_tabla(fg):
    return fg.rename_axis("mes").reset_index()


# Ejecuta el flujo completo desde dos CSV y guarda resultados listos para inspeccion.
def main():
    # Define los argumentos de entrada para poder ejecutar el script desde terminal.
    parser = argparse.ArgumentParser(
        description="Calibra VIIRS contra MODIS y exporta las tablas resultantes."
    )
    parser.add_argument("--modis", required=True, help="Ruta del CSV filtrado de MODIS.")
    parser.add_argument("--viirs", required=True, help="Ruta del CSV filtrado de VIIRS.")
    parser.add_argument(
        "--anio-inicio",
        type=int,
        default=2013,
        help="Primer ano del periodo usado para calcular factores (default: 2013).",
    )
    parser.add_argument(
        "--anio-fin",
        type=int,
        default=2019,
        help="Ultimo ano del periodo usado para calcular factores (default: 2019).",
    )
    parser.add_argument(
        "--suavizado-conteos",
        type=float,
        default=1000,
        help="Peso a del suavizado de conteos; se expresa en conteos.",
    )
    parser.add_argument(
        "--suavizado-frp",
        type=float,
        default=0,
        help=(
            "Peso a del suavizado FRP, en unidades FRP; default 0 porque "
            "su escala depende del producto."
        ),
    )
    parser.add_argument(
        "--salida",
        default="resultados_calibracion",
        help="Carpeta donde se escribiran los CSV de salida.",
    )
    args = parser.parse_args()

    # Lee ambos archivos y normaliza fechas, indices, valores y orden.
    modis = preparar_datos(pd.read_csv(args.modis), sensor="MODIS")
    viirs = preparar_datos(pd.read_csv(args.viirs), sensor="VIIRS")

    # Resume ambos sensores por ano, zona y mes.
    cm = conteos(modis)
    cv = conteos(viirs)

    # Calcula factores independientes para numero de celdas-dia y para FRP.
    fac_n, fg_n = factores(
        cm, cv, args.anio_inicio, args.anio_fin, col="n", a=args.suavizado_conteos
    )
    fac_frp, fg_frp = factores(
        cm, cv, args.anio_inicio, args.anio_fin, col="frp", a=args.suavizado_frp
    )

    # Armoniza las dos metricas de VIIRS en celdas de 0.1 grados.
    armonizado_n = aplicar(viirs, fac_n, fg_n, col="n")
    armonizado_frp = aplicar(viirs, fac_frp, fg_frp, col="frp")
    columnas_frp = LLAVES_CELDA_DIA + ["frp_arm"]
    armonizado = armonizado_n.merge(
        armonizado_frp[columnas_frp],
        how="one_to_one",
        on=LLAVES_CELDA_DIA,
        validate="one_to_one",
    )

    # Genera comparaciones mensuales y metricas para conteos y FRP.
    validacion_n = validar(
        cm, cv, fac_n, fg_n, args.anio_fin, col="n", anio_ini=args.anio_inicio
    )
    validacion_frp = validar(
        cm, cv, fac_frp, fg_frp, args.anio_fin, col="frp", anio_ini=args.anio_inicio
    )
    metricas_n = metricas_validacion(validacion_n)
    metricas_frp = metricas_validacion(validacion_frp)

    # Crea la carpeta de resultados si todavia no existe.
    carpeta_salida = Path(args.salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    # Exporta tablas intermedias, factores, series mensuales y resultados armonizados.
    cm.to_csv(carpeta_salida / "conteos_modis.csv", index=False)
    cv.to_csv(carpeta_salida / "conteos_viirs.csv", index=False)
    fac_n.to_csv(carpeta_salida / "factores_conteos.csv", index=False)
    _globales_a_tabla(fg_n).to_csv(
        carpeta_salida / "factores_globales_conteos.csv", index=False
    )
    fac_frp.to_csv(carpeta_salida / "factores_frp.csv", index=False)
    _globales_a_tabla(fg_frp).to_csv(
        carpeta_salida / "factores_globales_frp.csv", index=False
    )
    armonizado.to_csv(carpeta_salida / "viirs_armonizado_01grados.csv", index=False)
    validacion_n.to_csv(carpeta_salida / "validacion_conteos_mensual.csv", index=False)
    validacion_frp.to_csv(carpeta_salida / "validacion_frp_mensual.csv", index=False)
    metricas_n.to_csv(carpeta_salida / "metricas_validacion_conteos.csv", index=False)
    metricas_frp.to_csv(carpeta_salida / "metricas_validacion_frp.csv", index=False)

    # Muestra donde quedaron los resultados y las metricas calculadas.
    print(f"Resultados guardados en: {carpeta_salida.resolve()}")
    print("\nValidacion de conteos:")
    print(metricas_n.to_string(index=False))
    print("\nValidacion de FRP:")
    print(metricas_frp.to_string(index=False))


# Ejecuta el flujo solo al llamar el archivo directamente, no al importarlo.
if __name__ == "__main__":
    main()
