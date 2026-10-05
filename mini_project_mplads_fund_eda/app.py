import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.model import CompletionRiskModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mplads_app")

app = FastAPI(title="MPLADS Fund & Risk Intelligence", version="1.0.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_PATH = os.path.join(BASE_DIR, "models", "completion_risk_v1.pkl")
PROCESSED_DF_PATH = os.path.join(BASE_DIR, "data", "processed", "df_cleaned.csv")
CAT_SUMMARY_PATH = os.path.join(BASE_DIR, "data", "processed", "cat_summary.json")

# In-memory fast cache
DATA_CACHE = {
    "df": None,
    "category_summary": None,
    "model": None,
    "state_medians": {}
}

def load_data_and_model():
    if DATA_CACHE["df"] is not None:
        return

    # 1. Load Data
    if os.path.exists(PROCESSED_DF_PATH):
        df = pd.read_csv(PROCESSED_DF_PATH)
    else:
        from main import build_final_dataframe
        df = build_final_dataframe()

    # 2. Load Model
    if os.path.exists(MODEL_PATH):
        model = CompletionRiskModel.load(MODEL_PATH)
    else:
        model = CompletionRiskModel(model="logistic_regression")
        model.train(df)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model.save(MODEL_PATH)

    # 3. Ensure risk score exists
    if "risk_score" not in df.columns:
        try:
            proba = model.predict_proba(df)
            df["risk_score"] = np.round(proba[:, 1] * 100, 1)
        except Exception:
            df["risk_score"] = (df["at_risk"] * 100.0).round(1)

    # 4. State Medians
    state_medians = df.groupby("state")["utilization_rate"].median().to_dict()

    # 5. Category Summary
    if os.path.exists(CAT_SUMMARY_PATH):
        with open(CAT_SUMMARY_PATH, "r", encoding="utf-8") as f:
            cat_summary = json.load(f)
    else:
        cat_summary = []

    DATA_CACHE["df"] = df
    DATA_CACHE["model"] = model
    DATA_CACHE["state_medians"] = state_medians
    DATA_CACHE["category_summary"] = cat_summary
    logger.info(f"Loaded {len(df)} MP records into memory cache.")

@app.on_event("startup")
def startup():
    load_data_and_model()

# Ensure data is ready for every request (Serverless Cold-Start Guard)
def get_cache():
    if DATA_CACHE["df"] is None:
        load_data_and_model()
    return DATA_CACHE

class PredictRequest(BaseModel):
    state: str
    allocated_amount: float
    total_sanction_amount: float
    sanctioned_work_count: int
    total_disbursed_amount: float
    completed_work_count: int

# --- API Endpoints ---

@app.get("/api/overview")
def get_overview():
    cache = get_cache()
    df = cache["df"]
    
    total_allocated = float(df["allocated_amount"].sum())
    total_sanctioned = float(df["total_sanction_amount"].sum())
    total_disbursed = float(df["total_disbursed_amount"].sum())
    total_mps = int(len(df))
    at_risk_count = int(df["at_risk"].sum())
    zero_activity_count = int(df["has_no_activity"].sum())
    orphan_count = int(df["has_orphan_completions"].sum())
    
    avg_utilization = float(df["utilization_rate"].mean())
    median_utilization = float(df["utilization_rate"].median())
    total_backlog = float(df["sanctioned_backlog"].sum())

    return {
        "total_mps": total_mps,
        "total_allocated": total_allocated,
        "total_sanctioned": total_sanctioned,
        "total_disbursed": total_disbursed,
        "total_backlog": total_backlog,
        "avg_utilization_rate": avg_utilization,
        "median_utilization_rate": median_utilization,
        "at_risk_count": at_risk_count,
        "at_risk_percent": round((at_risk_count / total_mps) * 100, 1) if total_mps else 0,
        "zero_activity_count": zero_activity_count,
        "orphan_count": orphan_count,
        "states_count": int(df["state"].nunique())
    }

@app.get("/api/states")
def get_states_analytics():
    cache = get_cache()
    df = cache["df"]

    states_df = df[df["state"].notna() & (df["state"] != "")].groupby("state").agg(
        mp_count=("honble_members_of_parliament", "count"),
        total_allocated=("allocated_amount", "sum"),
        total_sanctioned=("total_sanction_amount", "sum"),
        total_disbursed=("total_disbursed_amount", "sum"),
        avg_utilization=("utilization_rate", "mean"),
        median_utilization=("utilization_rate", "median"),
        avg_backlog=("sanctioned_backlog", "mean"),
        at_risk_count=("at_risk", "sum")
    ).reset_index()

    states_df["at_risk_pct"] = (states_df["at_risk_count"] / states_df["mp_count"]) * 100
    states_df = states_df.sort_values(by="avg_utilization", ascending=False)
    
    return states_df.to_dict(orient="records")

@app.get("/api/mps")
def get_mps_directory(
    search: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    risk_filter: Optional[str] = Query("all"),
    sort_by: Optional[str] = Query("utilization_rate"),
    sort_order: Optional[str] = Query("desc"),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=200)
):
    cache = get_cache()
    res = cache["df"]

    if state and state != "All":
        res = res[res["state"] == state]

    if search:
        s = search.strip().lower()
        res = res[
            res["honble_members_of_parliament"].astype(str).str.lower().str.contains(s, na=False) |
            res["constituency"].astype(str).str.lower().str.contains(s, na=False)
        ]

    if risk_filter == "at_risk":
        res = res[res["at_risk"] == 1]
    elif risk_filter == "on_track":
        res = res[res["at_risk"] == 0]

    ascending = (sort_order.lower() == "asc")
    if sort_by in res.columns:
        res = res.sort_values(by=sort_by, ascending=ascending)

    total_count = len(res)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated = res.iloc[start_idx:end_idx]

    cols = [
        "honble_members_of_parliament", "state", "constituency",
        "allocated_amount", "total_sanction_amount", "sanctioned_work_count",
        "total_disbursed_amount", "completed_work_count",
        "utilization_rate", "sanctioned_backlog", "completion_ratio",
        "has_no_activity", "has_orphan_completions", "at_risk", "risk_score"
    ]
    records = paginated[cols].fillna("").to_dict(orient="records")

    return {
        "total": total_count,
        "page": page,
        "limit": limit,
        "total_pages": (total_count + limit - 1) // limit,
        "data": records
    }

@app.get("/api/categories")
def get_categories():
    cache = get_cache()
    return cache["category_summary"]

@app.post("/api/predict")
def predict_risk(req: PredictRequest):
    cache = get_cache()
    model = cache["model"]
    state_medians = cache["state_medians"]

    utilization_rate = req.total_disbursed_amount / req.allocated_amount if req.allocated_amount > 0 else 0.0
    sanctioned_backlog = req.total_sanction_amount - req.total_disbursed_amount
    completion_ratio = req.completed_work_count / req.sanctioned_work_count if req.sanctioned_work_count > 0 else 0.0
    has_no_activity = int(req.sanctioned_work_count == 0 and req.completed_work_count == 0)
    has_orphan_completions = int(req.sanctioned_work_count == 0 and req.completed_work_count > 0)

    input_df = pd.DataFrame([{
        "state": req.state,
        "allocated_amount": req.allocated_amount,
        "total_sanction_amount": req.total_sanction_amount,
        "sanctioned_work_count": req.sanctioned_work_count,
        "total_disbursed_amount": req.total_disbursed_amount,
        "completed_work_count": req.completed_work_count,
        "sanctioned_backlog": sanctioned_backlog,
        "completion_ratio": completion_ratio,
        "has_no_activity": has_no_activity,
        "has_orphan_completions": has_orphan_completions
    }])

    try:
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        risk_probability = float(probabilities[1] * 100)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    state_median = state_medians.get(req.state, 0.0)

    risk_factors = []
    if utilization_rate < state_median:
        risk_factors.append(f"Utilization ({utilization_rate:.1%}) is below {req.state} state median ({state_median:.1%}).")
    if sanctioned_backlog > 50000000:
        risk_factors.append(f"High unreleased sanction backlog: ₹{sanctioned_backlog / 1e7:.2f} Cr.")
    if req.sanctioned_work_count > 0 and completion_ratio < 0.3:
        risk_factors.append(f"Low completion ratio ({completion_ratio:.1%}): Majority of sanctioned works remain incomplete.")
    if has_no_activity:
        risk_factors.append("Zero sanctioned and zero completed works recorded.")
    if has_orphan_completions:
        risk_factors.append("Orphan completion anomaly: Disbursed works recorded without active matching sanctions.")

    if not risk_factors:
        risk_factors.append("Performance metrics are healthy and tracking above the regional baseline.")

    return {
        "at_risk": prediction,
        "risk_status": "AT RISK" if prediction == 1 else "ON TRACK",
        "risk_probability": round(risk_probability, 1),
        "metrics": {
            "utilization_rate": round(utilization_rate, 4),
            "utilization_pct": round(utilization_rate * 100, 2),
            "sanctioned_backlog": sanctioned_backlog,
            "completion_ratio": round(completion_ratio, 4),
            "state_median_utilization": round(state_median, 4),
            "state_median_pct": round(state_median * 100, 2)
        },
        "risk_factors": risk_factors
    }

os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(index_file)
