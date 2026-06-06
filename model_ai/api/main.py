"""
УБ Гал түймрийн эрсдэлийн прогноз — REST API
=============================================
Эхлүүлэх: model_ai/ директороос
  uvicorn api.main:app --reload --port 8000

Баримт бичиг:
  http://localhost:8000/docs
"""
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from api import data_loader, predictor
from api.auth import require_api_key
from api.models import (
    DistrictSummary,
    DistrictsResponse,
    MetricsResponse,
    PredictRequest,
    PredictResult,
    RiskRecord,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    data_loader.startup()
    predictor.startup()
    yield


app = FastAPI(
    title="УБ Гал түймрийн эрсдэлийн API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─── Нийтийн endpoint-ууд ────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": len(predictor.models) == 3}


@app.get("/districts", response_model=DistrictsResponse)
def districts():
    return {"districts": data_loader.get_districts()}


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    return data_loader.get_metrics()


@app.get("/risk", response_model=list[RiskRecord])
def risk(
    year:  int = Query(2025, ge=2015, le=2030),
    month: int | None = Query(None, ge=1, le=12),
):
    """Бүх хороодын эрсдэлийн оноо. month заахгүй бол 12 сар бүгдийг буцаана."""
    rows = data_loader.get_records()
    if month is not None:
        rows = [r for r in rows if r["month"] == month]
    return rows


@app.get("/risk/{district}/{khoroo}", response_model=list[RiskRecord])
def risk_khoroo(district: str, khoroo: int):
    """Нэг хорооны бүх сарын эрсдэл (12 мөр)."""
    rows = [
        r for r in data_loader.get_records()
        if r["district"] == district and r["khoroo"] == khoroo
    ]
    if not rows:
        raise HTTPException(status_code=404, detail="Хороо олдсонгүй")
    return sorted(rows, key=lambda r: r["month"])


@app.get("/rank", response_model=list[RiskRecord])
def rank(
    month:  int = Query(..., ge=1, le=12),
    cause:  str = Query("ilgal", pattern="^(ilgal|tsakhilgaan|yandan)$"),
    limit:  int = Query(40, ge=1, le=204),
):
    """Эрсдэлээр эрэмбэлсэн хороодын жагсаалт."""
    rows = [r for r in data_loader.get_records() if r["month"] == month]
    key  = f"risk_{cause}"
    rows.sort(key=lambda r: r[key], reverse=True)
    return rows[:limit]


@app.get("/summary/{month}", response_model=list[DistrictSummary])
def summary(month: int):
    """Дүүрэг бүрийн дундаж эрсдэл (тухайн сард)."""
    if not 1 <= month <= 12:
        raise HTTPException(status_code=422, detail="Сар 1-12 байх ёстой")
    rows = [r for r in data_loader.get_records() if r["month"] == month]
    agg: dict[str, dict] = {}
    for r in rows:
        d = r["district"]
        if d not in agg:
            agg[d] = {"district": d, "month": month,
                      "risk_ilgal": 0.0, "risk_tsakhilgaan": 0.0, "risk_yandan": 0.0, "_n": 0}
        agg[d]["risk_ilgal"]       += r["risk_ilgal"]
        agg[d]["risk_tsakhilgaan"] += r["risk_tsakhilgaan"]
        agg[d]["risk_yandan"]      += r["risk_yandan"]
        agg[d]["_n"] += 1
    result = []
    for v in agg.values():
        n = v.pop("_n")
        v["risk_ilgal"]       = round(v["risk_ilgal"] / n, 4)
        v["risk_tsakhilgaan"] = round(v["risk_tsakhilgaan"] / n, 4)
        v["risk_yandan"]      = round(v["risk_yandan"] / n, 4)
        result.append(v)
    return sorted(result, key=lambda r: r["district"])


# ─── Нууц шаардах endpoint-ууд ───────────────────────────────────────────────

@app.post("/predict", response_model=list[PredictResult],
          dependencies=[Depends(require_api_key)])
def predict(body: PredictRequest):
    """Ирээдүйн сарын таамаглал (загвар шууд ажиллуулна). X-API-Key шаардана."""
    reqs = [{"district": k.district, "khoroo": k.khoroo,
              "year": body.year, "month": body.month}
            for k in body.khoroo_ids]
    return predictor.predict(reqs)


@app.post("/admin/reload", dependencies=[Depends(require_api_key)])
def admin_reload():
    """dashboard_data.json-г дахин ачаална (pipeline дуусмагц дуудна)."""
    data_loader.reload()
    return {"status": "reloaded"}
