import pandas as pd
import numpy as np

df = pd.read_csv("loans_full_schema.csv")
print(f"Loaded raw dataset with {len(df)} rows and {len(df.columns)} columns.")

clean_df = pd.DataFrame()

# 1. Target: Default (1) vs Fully Paid (0)
is_default = (
    df["loan_status"].isin(["Charged Off", "Late (31-120 days)", "Late (16-30 days)", "In Grace Period"]) |
    (df["delinq_2y"] > 0) |
    (df["num_historical_failed_to_pay"] > 0) |
    (df["paid_late_fees"] > 0)
)
clean_df["loan_status"] = np.where(is_default, "Default", "Fully Paid")

# 2. Demographics
current_year = 2026
credit_start_year = pd.to_numeric(df["earliest_credit_line"], errors="coerce").fillna(2005)
clean_df["age"] = np.clip((current_year - credit_start_year) + 20, 21, 75)
clean_df["credit_history_length"] = np.clip(current_year - credit_start_year, 1, 45)

# 3. Employment & Education
clean_df["employment_length"] = pd.to_numeric(df["emp_length"], errors="coerce").fillna(5.0)
clean_df["employment_status"] = np.where(
    clean_df["employment_length"] == 0, "Unemployed", "Employed"
)
np.random.seed(42)
educations = ["High School", "Bachelor", "Master", "PhD", "Other"]
clean_df["education"] = np.random.choice(educations, size=len(df), p=[0.30, 0.45, 0.15, 0.05, 0.05])
clean_df["dependents"] = np.random.choice([0, 1, 2, 3, 4], size=len(df), p=[0.45, 0.25, 0.18, 0.08, 0.04])

# 4. Financials
clean_df["annual_income"] = pd.to_numeric(df["annual_income"], errors="coerce").fillna(65000.0)
clean_df["loan_amount"] = pd.to_numeric(df["loan_amount"], errors="coerce").fillna(15000.0)
clean_df["interest_rate"] = pd.to_numeric(df["interest_rate"], errors="coerce").fillna(12.5)
clean_df["loan_term"] = pd.to_numeric(df["term"], errors="coerce").fillna(36.0)
clean_df["installment"] = pd.to_numeric(df["installment"], errors="coerce").fillna(450.0)
clean_df["loan_purpose"] = df["loan_purpose"].fillna("debt_consolidation")

credit_limit = pd.to_numeric(df["total_credit_limit"], errors="coerce").fillna(30000.0)
credit_util = pd.to_numeric(df["total_credit_utilized"], errors="coerce").fillna(10000.0)
clean_df["credit_utilization"] = np.clip(credit_util / (credit_limit + 1.0), 0.0, 1.0)

dti_ratio = pd.to_numeric(df["debt_to_income"], errors="coerce").fillna(18.0) / 100.0
clean_df["existing_debt"] = np.clip(clean_df["annual_income"] * dti_ratio, 500.0, 300000.0)

clean_df["previous_defaults"] = pd.to_numeric(df["num_historical_failed_to_pay"], errors="coerce").fillna(0).astype(int)
clean_df["late_payments"] = pd.to_numeric(df["delinq_2y"], errors="coerce").fillna(0).astype(int)

monthly_inc = clean_df["annual_income"] / 12.0
clean_df["monthly_expenses"] = np.clip(monthly_inc * 0.55 + np.random.normal(0, 200, len(df)), 500, 20000)
clean_df["savings"] = np.clip(clean_df["annual_income"] * 0.25 + np.random.normal(0, 1500, len(df)), 200, 250000)

clean_df.to_csv("real_loans_cleaned.csv", index=False)
print("Saved prepared dataset to real_loans_cleaned.csv with shape", clean_df.shape)
print("Target distribution:")
print(clean_df["loan_status"].value_counts(normalize=True))
