import json
from pathlib import Path

import duckdb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="FireScope API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    
    allow_headers=["*"],
)

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed"


@app.get("/")
def home():
    return {"message": "FireScope API funcionando"}


@app.get("/fires")  # <- faltaba este decorador: sin él la ruta nunca llamaba a get_fires
def get_fires(
    year: int,
    sensor: str,
    month: int | None = Query(None, description="1-12. Si se omite, devuelve todo el año"),
    country: str | None = None,
    limit: int = 1000,
    offset: int = 0,
):
    sensor = sensor.upper()

    # Validar parámetros
    if sensor not in ["MODIS", "VIIRS"]:
        raise HTTPException(400, "Invalid sensor")

    if month is not None and not 1 <= month <= 12:
        raise HTTPException(400, "Invalid month")

    if not 2000 <= year <= 2025:
        raise HTTPException(400, "Invalid year")

    if sensor == "VIIRS" and year < 2018:
        raise HTTPException(400, "VIIRS unavailable for this year")

    if not 1 <= limit <= 10000 or offset < 0:
        raise HTTPException(400, "Invalid pagination")

    file_path = DATA_PATH / sensor / f"year={year}" / "data.parquet"
    if not file_path.exists():
        raise HTTPException(404, "Dataset not found")

    # Rango de fechas: todo el año o solo un mes
    if month is None:
        start_date, end_date = f"{year}-01-01", f"{year + 1}-01-01"
    else:
        start_date = f"{year}-{month:02d}-01"
        end_date = f"{year + 1}-01-01" if month == 12 else f"{year}-{month + 1:02d}-01"

    where = "WHERE acq_date >= ? AND acq_date < ?"
    params = [str(file_path), start_date, end_date]
    if country:
        where += " AND country = ?"
        params.append(country)

    # Fecha como texto para que el JSON sea limpio
    select = "SELECT * REPLACE (CAST(acq_date AS VARCHAR) AS acq_date)"

    with duckdb.connect() as con:
        total = con.execute(
            f"SELECT COUNT(*) FROM read_parquet(?) {where}", params
        ).fetchone()[0]

        # ORDER BY para que la paginación sea estable entre páginas
        result = con.execute(
            f"{select} FROM read_parquet(?) {where} ORDER BY acq_date, acq_time LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchdf()

    # to_json convierte NaN en null (to_dict dejaría NaN y rompería el JSON)
    records = json.loads(result.to_json(orient="records"))

    return {
        "sensor": sensor,
        "year": year,
        "month": month,
        "total": total,
        "count": len(records),
        "limit": limit,
        "offset": offset,
        "data": records,
    }
