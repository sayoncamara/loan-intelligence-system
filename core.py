"""
Core ML logic — framework-agnostic.
Used by both app.py (Streamlit) and api.py (FastAPI).
"""
import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import shap

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Module-level singletons (lazy-loaded)
_model = None
_qa_chain = None
_explainer = None


def get_model():
    global _model
    if _model is None:
        _model = XGBClassifier()
        _model.load_model("xgboost_loan_model.json")
    return _model


def get_explainer():
    global _explainer
    if _explainer is None:
        _explainer = shap.TreeExplainer(get_model())
    return _explainer


def get_qa_chain():
    global _qa_chain
    if _qa_chain is None:
        policy_files = [
            "policies/credit_score_policy.txt",
            "policies/dti_policy.txt",
            "policies/loan_purpose_policy.txt",
        ]
        documents = []
        for file in policy_files:
            loader = TextLoader(file)
            documents.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(api_key=API_KEY)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        llm = ChatOpenAI(api_key=API_KEY, model="gpt-3.5-turbo", temperature=0)

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a loan officer assistant explaining why a loan application "
                "was denied. Use the following policy documents to explain the decision "
                "clearly and professionally to the applicant.\n\n"
                "Policy context:\n{context}\n\n"
                "Provide a clear, empathetic explanation referencing specific policies. "
                "Give the applicant actionable advice on how to improve their chances.",
            ),
            ("human", "{question}"),
        ])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        _qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    return _qa_chain


def prepare_input(data: dict) -> pd.DataFrame:
    """Feature engineering — converts raw input dict to model-ready DataFrame."""
    data = data.copy()
    data["term"] = 36 if data["term"] == "36 months" else 60

    emp_map = {
        "Unknown": 0, "< 1 year": 0.5, "1 year": 1, "2 years": 2,
        "3 years": 3, "4 years": 4, "5 years": 5, "6 years": 6,
        "7 years": 7, "8 years": 8, "9 years": 9, "10+ years": 10,
    }
    data["emp_length"] = emp_map.get(data["emp_length"], 0)

    grade_map = {"A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1}
    data["grade"] = grade_map.get(data["grade"], 4)

    sub_grades = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]
    sub_grade_map = {sg: i for i, sg in enumerate(reversed(sub_grades), 1)}
    data["sub_grade"] = sub_grade_map.get(data["sub_grade"], 18)

    home_ownership_cols = [
        "home_ownership_MORTGAGE", "home_ownership_NONE",
        "home_ownership_OTHER", "home_ownership_OWN",
        "home_ownership_RENT",
    ]
    purpose_cols = [
        "purpose_credit_card", "purpose_debt_consolidation",
        "purpose_educational", "purpose_home_improvement",
        "purpose_house", "purpose_major_purchase", "purpose_medical",
        "purpose_moving", "purpose_other", "purpose_renewable_energy",
        "purpose_small_business", "purpose_vacation", "purpose_wedding",
    ]

    for col in home_ownership_cols + purpose_cols:
        data[col] = 0

    home_col = f"home_ownership_{data['home_ownership']}"
    purpose_col = f"purpose_{data['purpose']}"
    if home_col in data:
        data[home_col] = 1
    if purpose_col in data:
        data[purpose_col] = 1

    del data["home_ownership"]
    del data["purpose"]

    column_order = [
        "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
        "annual_inc", "dti", "emp_length", "fico_range_low", "fico_range_high",
        "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
        "mort_acc", "pub_rec_bankruptcies",
        "home_ownership_MORTGAGE", "home_ownership_NONE",
        "home_ownership_OTHER", "home_ownership_OWN", "home_ownership_RENT",
        "purpose_credit_card", "purpose_debt_consolidation",
        "purpose_educational", "purpose_home_improvement",
        "purpose_house", "purpose_major_purchase", "purpose_medical",
        "purpose_moving", "purpose_other", "purpose_renewable_energy",
        "purpose_small_business", "purpose_vacation", "purpose_wedding",
    ]

    return pd.DataFrame([data])[column_order]


def predict(raw_input: dict) -> dict:
    """Run full pipeline: predict + SHAP + (if denied) RAG explanation."""
    model = get_model()
    X_input = prepare_input(raw_input)

    prob = float(model.predict_proba(X_input)[0][1])
    denied = prob > 0.5

    # SHAP values
    explainer = get_explainer()
    shap_values = explainer.shap_values(X_input)
    feature_contributions = [
        {"feature": col, "value": float(X_input.iloc[0][col]), "shap": float(shap_values[0][i])}
        for i, col in enumerate(X_input.columns)
    ]
    feature_contributions.sort(key=lambda x: abs(x["shap"]), reverse=True)

    result = {
        "decision": "DENIED" if denied else "APPROVED",
        "default_risk": prob,
        "base_value": float(np.asarray(explainer.expected_value).flatten()[0]),
        "top_features": feature_contributions[:10],
    }

    if denied:
        try:
            qa_chain = get_qa_chain()
            question = (
                f"A loan application has been DENIED. Applicant details: "
                f"Loan amount: ${raw_input['loan_amnt']:,}, Term: {raw_input['term']}, "
                f"Credit grade: {raw_input['grade']} ({raw_input['sub_grade']}), "
                f"FICO: {raw_input['fico_range_low']}, "
                f"Annual income: ${raw_input['annual_inc']:,}, DTI: {raw_input['dti']}%, "
                f"Employment: {raw_input['emp_length']}, Purpose: {raw_input['purpose']}. "
                f"Why was this denied and how can the applicant improve?"
            )
            result["explanation"] = qa_chain.invoke(question)
        except Exception as e:
            result["explanation"] = None
            result["explanation_error"] = str(e)

    return result