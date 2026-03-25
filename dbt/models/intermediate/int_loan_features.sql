-- Intermediate model: feature engineering for ML training
-- Encodes categoricals, maps grades, creates target variable

with base as (
    select * from {{ ref('stg_loans') }}
),

featured as (
    select
        loan_amnt,
        term_months as term,
        int_rate,
        installment,

        -- Grade encoding (A=7 best, G=1 worst)
        case grade
            when 'A' then 7 when 'B' then 6 when 'C' then 5
            when 'D' then 4 when 'E' then 3 when 'F' then 2
            when 'G' then 1 else 4
        end as grade_encoded,

        annual_inc,
        dti,

        -- Employment length encoding
        case
            when emp_length like '%10+%' then 10
            when emp_length like '%9%' then 9
            when emp_length like '%8%' then 8
            when emp_length like '%7%' then 7
            when emp_length like '%6%' then 6
            when emp_length like '%5%' then 5
            when emp_length like '%4%' then 4
            when emp_length like '%3%' then 3
            when emp_length like '%2%' then 2
            when emp_length like '%1%' then 1
            else 0
        end as emp_length_years,

        fico_range_low,
        fico_range_high,
        open_acc,
        coalesce(pub_rec, 0) as pub_rec,
        revol_bal,
        coalesce(revol_util, 0.0) as revol_util,
        total_acc,
        coalesce(mort_acc, 0) as mort_acc,
        coalesce(pub_rec_bankruptcies, 0) as pub_rec_bankruptcies,

        -- One-hot: home ownership
        case when home_ownership = 'MORTGAGE' then 1 else 0 end as home_ownership_mortgage,
        case when home_ownership = 'NONE' then 1 else 0 end as home_ownership_none,
        case when home_ownership = 'OTHER' then 1 else 0 end as home_ownership_other,
        case when home_ownership = 'OWN' then 1 else 0 end as home_ownership_own,
        case when home_ownership = 'RENT' then 1 else 0 end as home_ownership_rent,

        -- One-hot: purpose
        case when purpose = 'credit_card' then 1 else 0 end as purpose_credit_card,
        case when purpose = 'debt_consolidation' then 1 else 0 end as purpose_debt_consolidation,
        case when purpose = 'educational' then 1 else 0 end as purpose_educational,
        case when purpose = 'home_improvement' then 1 else 0 end as purpose_home_improvement,
        case when purpose = 'house' then 1 else 0 end as purpose_house,
        case when purpose = 'major_purchase' then 1 else 0 end as purpose_major_purchase,
        case when purpose = 'medical' then 1 else 0 end as purpose_medical,
        case when purpose = 'moving' then 1 else 0 end as purpose_moving,
        case when purpose = 'other' then 1 else 0 end as purpose_other,
        case when purpose = 'renewable_energy' then 1 else 0 end as purpose_renewable_energy,
        case when purpose = 'small_business' then 1 else 0 end as purpose_small_business,
        case when purpose = 'vacation' then 1 else 0 end as purpose_vacation,
        case when purpose = 'wedding' then 1 else 0 end as purpose_wedding,

        -- Target variable
        case when loan_status = 'Charged Off' then 1 else 0 end as target

    from base
)

select * from featured
