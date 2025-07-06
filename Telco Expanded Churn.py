#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
import os
from IPython.display import display

# ── CONFIGURE THESE PATHS ──────────────────────────────────────────────
churn_path = "/Users/davidsolis/generated/Telco_Churn_Data.csv"
supp_path  = "/Users/davidsolis/generated/supplemental"
# ───────────────────────────────────────────────────────────────────────

# 1. Load churn dataset
churn = pd.read_csv(churn_path)
churn['TotalCharges'] = pd.to_numeric(churn['TotalCharges'], errors='coerce')
churn['Churn_flag']   = churn['Churn'].map({'Yes': 1, 'No': 0})

# 2. Load supplemental tables
customers = pd.read_csv(os.path.join(supp_path, "customers.csv"))
usage     = pd.read_csv(os.path.join(supp_path, "usage_events.csv"))
billing   = pd.read_csv(os.path.join(supp_path, "billing_transactions.csv"))
tickets   = pd.read_csv(os.path.join(supp_path, "support_tickets.csv"))
surveys   = pd.read_csv(os.path.join(supp_path, "customer_surveys.csv"))
contracts = pd.read_csv(os.path.join(supp_path, "contracts.csv"))

# 3. Pre-process support tickets
tickets['opened_at'] = pd.to_datetime(tickets['opened_at'])
tickets['closed_at'] = pd.to_datetime(tickets['closed_at'])
tickets['resolution_hours'] = (tickets['closed_at'] - tickets['opened_at']) \
                              .dt.total_seconds() / 3600
tickets['first_contact_resolution'] = tickets['first_contact_resolution'].map({'Y':1,'N':0})

# 4. Compute supplemental metrics
usage_metrics   = usage.groupby('customerID')['usage_amount'] \
                       .mean().rename('AvgUsage')
billing_metrics = billing.groupby('customerID')['late_fee_flag'] \
                         .mean().rename('LatePaymentRate')
ticket_metrics  = tickets.groupby('customerID').agg(
    AvgResolutionTime=('resolution_hours','mean'),
    FCR=('first_contact_resolution','mean')
)

# 5. Compute survey metrics
surveys['promoter']  = (surveys['nps_category']=='Promoter').astype(int)
surveys['detractor'] = (surveys['nps_category']=='Detractor').astype(int)
survey_metrics = surveys.groupby('customerID').apply(
    lambda g: pd.Series({
        'CSAT': g['csat_score'].mean(),
        'NPS': 100*(g['promoter'].sum()/len(g)
                   - g['detractor'].sum()/len(g))
    })
).reset_index()

# 6. Merge everything
df = (churn
      .merge(customers, on='customerID', how='left')
      .merge(contracts[['customerID','contract_type']], on='customerID', how='left')
      .merge(usage_metrics,   on='customerID', how='left')
      .merge(billing_metrics, on='customerID', how='left')
      .merge(ticket_metrics,  on='customerID', how='left')
      .merge(survey_metrics,  on='customerID', how='left')
)

# 7. Compute enriched KPIs
summary = {
    'Customer_Churn_Rate (%)': [df['Churn_flag'].mean() * 100],
    'Avg_CLTV (USD)':           [(df['tenure'] * df['MonthlyCharges']).mean()],
    'Avg_CSAT':                 [df['CSAT'].mean()],
    'NPS (%)':                  [df['NPS'].mean()],
    'Avg_Resolution_Time (hrs)': [df['AvgResolutionTime'].mean()],
    'FCR (%)':                  [df['FCR'].mean() * 100],
    'Avg_Usage':                [df['AvgUsage'].mean()],
    'Late_Payment_Rate (%)':    [df['LatePaymentRate'].mean() * 100]
}
summary_df = pd.DataFrame(summary)

# 8. Display the results
display(summary_df)


# In[ ]:




