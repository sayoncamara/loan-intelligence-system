-- Staging model: clean and validate raw Lending Club data
-- Source: CSV loaded into DuckDB via seed or external table

with raw as (
    select * from read_csv_auto('data/lending_club.csv')
),

cleaned as (
    select
        loan_amnt,
        case
            when term like '%36%' then 36
            when term like '%60%' then 60
            else null
        end as term_months,
        int_rate,
        installment,
        grade,
        sub_grade,
        annual_inc,
        dti,
        emp_length,
        fico_range_low,
        fico_range_high,
        open_acc,
        pub_rec,
        revol_bal,
        revol_util,
        total_acc,
        mort_acc,
        pub_rec_bankruptcies,
        home_ownership,
        purpose,
        loan_status
    from raw
    where loan_amnt is not null
      and annual_inc is not null
      and annual_inc > 0
      and loan_status in ('Fully Paid', 'Charged Off')
)

select * from cleaned
