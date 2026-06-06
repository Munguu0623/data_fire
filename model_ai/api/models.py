from pydantic import BaseModel
from typing import Any


class RiskRecord(BaseModel):
    district: str
    khoroo: int
    month: int
    pop: int
    risk_ilgal: float
    risk_tsakhilgaan: float
    risk_yandan: float
    n_ilgal: int
    n_tsakhilgaan: int
    n_yandan: int


class MetricItem(BaseModel):
    cause: str
    baseline: float
    pr_auc: float
    roc_auc: float
    recall_top30: float
    recall_top50: float


class MetricsResponse(BaseModel):
    metrics: list[MetricItem]
    importance: dict[str, Any]
    meta: dict[str, Any]


class DistrictsResponse(BaseModel):
    districts: list[str]


class DistrictSummary(BaseModel):
    district: str
    month: int
    risk_ilgal: float
    risk_tsakhilgaan: float
    risk_yandan: float


class KhorooId(BaseModel):
    district: str
    khoroo: int


class PredictRequest(BaseModel):
    year: int
    month: int
    khoroo_ids: list[KhorooId]


class PredictResult(BaseModel):
    district: str
    khoroo: int
    year: int
    month: int
    risk_ilgal: float
    risk_tsakhilgaan: float
    risk_yandan: float
