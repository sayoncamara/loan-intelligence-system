import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from dotenv import load_dotenv
import os
import shap
import matplotlib.pyplot as plt

# RAG imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# --- CONFIG ---
st.set_page_config(page_title="Loan Intelligence System", page_icon="🏦", layout="wide")

# --- LOAD ENV ---
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model('xgboost_loan_model.json')
    return model

# --- LOAD RAG CHAIN ---
@st.cache_resource
def load_rag_chain():
policy_files = [
    'policies/credit_score_policy.txt',
    'policies/dti_policy.txt',
    'policies/loan_purpose_policy.txt'
]
    documents = []
    for file in policy_files:
        loader = TextLoader(file)
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0)
    prompt_template = """You are a loan officer assistant explaining why a loan application 
was denied. Use the following policy documents to explain the decision clearly and 
professionally to the applicant.

Policy context:
{context}

Loan application details and question:
{question}

Provide a clear, empathetic explanation referencing specific policies. 
Give the applicant actionable advice on how to improve their chances."""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# --- FEATURE ENGINEERING ---
def prepare_input(data):
    # Encode term
    data['term'] = 36 if data['term'] == '36 months' else 60

    # Encode emp_length
    emp_map = {'Unknown': 0, '< 1 year': 0.5, '1 year': 1, '2 years': 2,
               '3 years': 3, '4 years': 4, '5 years': 5, '6 years': 6,
               '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10}
    data['emp_length'] = emp_map.get(data['emp_length'], 0)

    # Encode grade
    grade_map = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
    data['grade'] = grade_map.get(data['grade'], 4)

    # Encode sub_grade
    sub_grades = [f'{g}{n}' for g in 'ABCDEFG' for n in range(1, 6)]
    sub_grade_map = {sg: i for i, sg in enumerate(reversed(sub_grades), 1)}
    data['sub_grade'] = sub_grade_map.get(data['sub_grade'], 18)

    # All possible one-hot columns
    home_ownership_cols = ['home_ownership_MORTGAGE', 'home_ownership_NONE',
                       'home_ownership_OTHER', 'home_ownership_OWN',
                       'home_ownership_RENT']
    purpose_cols = ['purpose_credit_card', 'purpose_debt_consolidation',
                    'purpose_educational', 'purpose_home_improvement',
                    'purpose_house', 'purpose_major_purchase', 'purpose_medical',
                    'purpose_moving', 'purpose_other', 'purpose_renewable_energy',
                    'purpose_small_business', 'purpose_vacation', 'purpose_wedding']

    # Set all to 0 first
    for col in home_ownership_cols + purpose_cols:
        data[col] = 0

    # Set the selected ones to 1
    home_col = f"home_ownership_{data['home_ownership']}"
    purpose_col = f"purpose_{data['purpose']}"
    if home_col in data:
        data[home_col] = 1
    if purpose_col in data:
        data[purpose_col] = 1

    # Remove original categorical columns
    del data['home_ownership']
    del data['purpose']

    # Define exact column order from training
    column_order = [
    'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
    'annual_inc', 'dti', 'emp_length', 'fico_range_low', 'fico_range_high',
    'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
    'mort_acc', 'pub_rec_bankruptcies',
    'home_ownership_MORTGAGE', 'home_ownership_NONE', 
    'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT',
    'purpose_credit_card', 'purpose_debt_consolidation',
    'purpose_educational', 'purpose_home_improvement',
    'purpose_house', 'purpose_major_purchase', 'purpose_medical',
    'purpose_moving', 'purpose_other', 'purpose_renewable_energy',
    'purpose_small_business', 'purpose_vacation', 'purpose_wedding'
]

    return pd.DataFrame([data])[column_order]

# --- UI ---
st.title("🏦 Loan Intelligence System")
st.markdown("Enter loan application details to get an instant decision with explanation.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Loan Details")
    loan_amnt = st.number_input("Loan Amount ($)", 1000, 40000, 10000)
    term = st.selectbox("Loan Term", ["36 months", "60 months"])
    int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 13.0)
    installment = st.number_input("Monthly Installment ($)", 50, 1500, 300)
    purpose = st.selectbox("Loan Purpose", [
        "debt_consolidation", "credit_card", "home_improvement",
        "other", "major_purchase", "small_business", "medical",
        "moving", "vacation", "wedding", "house", "renewable_energy", "educational"
    ])

with col2:
    st.subheader("Applicant Details")
    grade = st.selectbox("Credit Grade", ["A", "B", "C", "D", "E", "F", "G"])
    sub_grade = st.selectbox("Sub Grade", [f'{g}{n}' for g in 'ABCDEFG' for n in range(1, 6)])
    annual_inc = st.number_input("Annual Income ($)", 10000, 300000, 60000)
    dti = st.slider("Debt-to-Income Ratio (%)", 0.0, 60.0, 20.0)
    emp_length = st.selectbox("Employment Length", [
        '< 1 year', '1 year', '2 years', '3 years', '4 years',
        '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years', 'Unknown'
    ])
    fico = st.slider("FICO Credit Score", 580, 850, 700)
    home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    open_acc = st.number_input("Open Credit Accounts", 0, 40, 8)
    pub_rec = st.number_input("Public Records", 0, 10, 0)
    revol_bal = st.number_input("Revolving Balance ($)", 0, 100000, 15000)
    revol_util = st.slider("Revolving Utilization (%)", 0.0, 100.0, 50.0)
    total_acc = st.number_input("Total Credit Accounts", 0, 60, 20)
    mort_acc = st.number_input("Mortgage Accounts", 0, 10, 0)
    pub_rec_bankruptcies = st.number_input("Bankruptcies", 0, 5, 0)

# --- PREDICT ---
if st.button("🔍 Assess Loan Application", type="primary"):
    model = load_model()
    qa_chain = load_rag_chain()

    input_data = {
        'loan_amnt': loan_amnt, 'term': term, 'int_rate': int_rate,
        'installment': installment, 'grade': grade, 'sub_grade': sub_grade,
        'annual_inc': annual_inc, 'dti': dti, 'emp_length': emp_length,
        'fico_range_low': fico, 'fico_range_high': fico + 4,
        'open_acc': open_acc, 'pub_rec': pub_rec, 'revol_bal': revol_bal,
        'revol_util': revol_util, 'total_acc': total_acc, 'mort_acc': mort_acc,
        'pub_rec_bankruptcies': pub_rec_bankruptcies,
        'home_ownership': home_ownership, 'purpose': purpose
    }

    X_input = prepare_input(input_data.copy())
    prob = model.predict_proba(X_input)[0][1]
    prediction = int(prob > 0.5)

    st.divider()

    if prediction == 0:
        st.success(f"✅ LOAN APPROVED — Default Risk: {prob:.1%}")
        st.balloons()
    else:
        st.error(f"❌ LOAN DENIED — Default Risk: {prob:.1%}")

        # RAG explanation
        with st.spinner("Generating explanation..."):
            question = f"""
            A loan application has been DENIED. Applicant details:
            - Loan amount: ${loan_amnt:,}, Term: {term}
            - Credit grade: {grade} ({sub_grade}), FICO: {fico}
            - Annual income: ${annual_inc:,}, DTI: {dti}%
            - Employment: {emp_length}, Purpose: {purpose}
            Why was this denied and how can the applicant improve?
            """
            response = qa_chain.invoke({"query": question})
            st.subheader("📋 Explanation")
            st.write(response['result'])

        # SHAP chart
        with st.spinner("Generating SHAP explanation..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_input)
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_input.iloc[0],
                feature_names=X_input.columns.tolist()
            ), show=False)
            st.pyplot(fig)