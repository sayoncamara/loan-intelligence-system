"""
FastAPI wrapper around core ML logic.
Exposes prediction as a JSON API with auto-generated Swagger docs at /docs.
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from typing import Literal, Optional, List

import core

app = FastAPI(
    title="Loan Intelligence API",
    description=(
        "Credit risk assessment API. Combines XGBoost prediction, "
        "SHAP explainability, and RAG-grounded rejection explanations. "
        "Try the interactive docs below."
    ),
    version="1.0.0",
)


# --- Pydantic models ---

class LoanApplication(BaseModel):
    loan_amnt: float = Field(..., ge=1000, le=40000, description="Loan amount in USD")
    term: Literal["36 months", "60 months"]
    int_rate: float = Field(..., ge=5.0, le=30.0, description="Interest rate (%)")
    installment: float = Field(..., ge=50, le=1500, description="Monthly installment (USD)")
    grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    sub_grade: str = Field(..., description="E.g. A1, B3, D5")
    annual_inc: float = Field(..., ge=10000, le=300000)
    dti: float = Field(..., ge=0.0, le=60.0, description="Debt-to-income ratio (%)")
    emp_length: Literal[
        "< 1 year", "1 year", "2 years", "3 years", "4 years",
        "5 years", "6 years", "7 years", "8 years", "9 years",
        "10+ years", "Unknown",
    ]
    fico: int = Field(..., ge=580, le=850, description="FICO credit score")
    home_ownership: Literal["RENT", "MORTGAGE", "OWN", "OTHER"]
    purpose: Literal[
        "debt_consolidation", "credit_card", "home_improvement",
        "other", "major_purchase", "small_business", "medical",
        "moving", "vacation", "wedding", "house", "renewable_energy",
        "educational",
    ]
    open_acc: int = Field(..., ge=0, le=40)
    pub_rec: int = Field(..., ge=0, le=10)
    revol_bal: float = Field(..., ge=0, le=100000)
    revol_util: float = Field(..., ge=0.0, le=100.0)
    total_acc: int = Field(..., ge=0, le=60)
    mort_acc: int = Field(..., ge=0, le=10)
    pub_rec_bankruptcies: int = Field(..., ge=0, le=5)

    model_config = {
        "json_schema_extra": {
            "example": {
                "loan_amnt": 10000,
                "term": "36 months",
                "int_rate": 13.0,
                "installment": 300,
                "grade": "C",
                "sub_grade": "C3",
                "annual_inc": 60000,
                "dti": 20.0,
                "emp_length": "3 years",
                "fico": 700,
                "home_ownership": "RENT",
                "purpose": "debt_consolidation",
                "open_acc": 8,
                "pub_rec": 0,
                "revol_bal": 15000,
                "revol_util": 50.0,
                "total_acc": 20,
                "mort_acc": 0,
                "pub_rec_bankruptcies": 0,
            }
        }
    }


class FeatureContribution(BaseModel):
    feature: str
    value: float
    shap: float


class PredictionResponse(BaseModel):
    decision: Literal["APPROVED", "DENIED"]
    default_risk: float = Field(..., description="Predicted default probability (0-1)")
    base_value: float = Field(..., description="SHAP base value (model's average output)")
    top_features: List[FeatureContribution] = Field(
        ..., description="Top 10 features by absolute SHAP contribution"
    )
    explanation: Optional[str] = Field(
        None, description="RAG-generated explanation (only present if DENIED)"
    )
    explanation_error: Optional[str] = Field(
        None, description="Set if RAG explanation failed (prediction still valid)"
    )


# --- Startup: pre-warm model + RAG chain ---

@app.on_event("startup")
async def warmup():
    """Pre-load model and RAG chain so first real request isn't slow."""
    try:
        core.get_model()
        core.get_explainer()
        # RAG chain warmup is optional — it costs an OpenAI call, skip on startup
        # to avoid failing startup if key is missing. First /predict with DENIED will load it.
    except Exception as e:
        print(f"[warmup] Warning: {e}")


# --- Endpoints ---

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to interactive Swagger docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    """Lightweight health check — does not touch the model."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(application: LoanApplication):
    """
    Submit a loan application and receive:
    - Approval decision
    - Default probability
    - Top 10 features by SHAP contribution
    - If denied: a policy-grounded natural-language explanation
    """
    raw = application.model_dump()
    # Split single fico into the two fields the model expects
    fico = raw.pop("fico")
    raw["fico_range_low"] = fico
    raw["fico_range_high"] = fico + 4

    try:
        result = core.predict(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return result