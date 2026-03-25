-- Mart model: final training-ready table
-- All features encoded, no nulls, ready for XGBoost

select
    loan_amnt,
    term,
    int_rate,
    installment,
    grade_encoded as grade,
    annual_inc,
    dti,
    emp_length_years as emp_length,
    fico_range_low,
    fico_range_high,
    open_acc,
    pub_rec,
    revol_bal,
    revol_util,
    total_acc,
    mort_acc,
    pub_rec_bankruptcies,
    home_ownership_mortgage,
    home_ownership_none,
    home_ownership_other,
    home_ownership_own,
    home_ownership_rent,
    purpose_credit_card,
    purpose_debt_consolidation,
    purpose_educational,
    purpose_home_improvement,
    purpose_house,
    purpose_major_purchase,
    purpose_medical,
    purpose_moving,
    purpose_other,
    purpose_renewable_energy,
    purpose_small_business,
    purpose_vacation,
    purpose_wedding,
    target
from {{ ref('int_loan_features') }}
where loan_amnt > 0
  and annual_inc > 0
  and dti >= 0
