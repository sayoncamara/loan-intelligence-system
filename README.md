# Loan Intelligence System

An end-to-end credit risk assessment system combining **XGBoost** predictions, **SHAP** explainability, and a **RAG-powered LLM** for policy-grounded rejection explanations.

**[Live Demo](https://loan-intelligence-system-6vbtpbunxh7yvf5neajhva.streamlit.app/)**

---

## What It Does

1. **Predicts** loan approval/rejection using XGBoost trained on 100K Lending Club records
2. **Explains** each decision with SHAP waterfall plots showing which features drove the prediction
3. **Generates** human-readable rejection explanations grounded in lending policy documents via RAG (LangChain + FAISS + GPT-3.5-turbo)

---

## Architecture

```
         User enters loan application
                    |
              XGBoost Model
           (trained on 100K records)
                    |
            ┌───────┴───────┐
            v               v
        Approved         Rejected
            |               |
        SHAP waterfall   SHAP waterfall
        (transparency)    + RAG pipeline
                          (LangChain + FAISS + GPT-3.5)
                              |
                        Policy-grounded
                        rejection explanation
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML Model | XGBoost |
| Explainability | SHAP (waterfall plots) |
| RAG Pipeline | LangChain + FAISS + OpenAI GPT-3.5-turbo |
| Frontend | Streamlit |
| Data | Lending Club (100K records) |
| Deployment | Streamlit Cloud + GitHub CI/CD |

---

## Quickstart

### Prerequisites
- Python 3.10+
- OpenAI API key

### Setup

```bash
git clone https://github.com/sayoncamara/loan-intelligence-system.git
cd loan-intelligence-system
pip install -r requirements.txt
```

Create a `.env` file:
```
OPENAI_API_KEY=sk-your-key-here
```

### Run

```bash
python -m streamlit run app.py
```

---

## Key Features

- **Per-applicant SHAP explanations** — not just global feature importance, but individual waterfall plots showing exactly why each applicant was approved or rejected
- **RAG-grounded rejections** — rejection letters cite actual lending policy documents, not hallucinated reasons
- **Real-time predictions** — enter loan details and get an instant decision with full transparency

---

## Author

**Sayon Camara** — MSc Business Administration (Finance & Banking), KU Leuven
