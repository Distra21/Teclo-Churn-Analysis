#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from IPython.display import display

# 1. Load data
df = pd.read_csv('/Users/davidsolis/Desktop/Portfolio Projects/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2. Prepare data
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['Churn_flag'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['CLTV'] = df['tenure'] * df['MonthlyCharges']

# 3. KPI calculations
customer_churn_rate = df['Churn_flag'].mean() * 100
avg_cltv = df['CLTV'].mean()

# 4. Simple churn prediction model
features = ['tenure', 'MonthlyCharges', 'TotalCharges']
X = df[features].fillna(0)
y = df['Churn_flag']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 5. Model performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# 6. Summarize results
summary_df = pd.DataFrame({
    'Metric': [
        'Customer_Churn_Rate',
        'Avg_CLTV',
        'Model_Accuracy',
        'Model_Precision',
        'Model_Recall'
    ],
    'Value': [
        round(customer_churn_rate, 2),
        round(avg_cltv, 2),
        round(accuracy, 2),
        round(precision, 2),
        round(recall, 2)
    ]
})

# 7. Display in Jupyter
display(summary_df)


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt

# Prepare the summary data
data = {
    'Customer_Churn_Rate': 26.54,
    'Avg_CLTV': 2279.58,
    'Model_Accuracy': 0.79,
    'Model_Precision': 0.66,
    'Model_Recall': 0.45
}

metrics = list(data.keys())
values = list(data.values())

# Create card-style visualization
fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4))
for ax, metric, value in zip(axes, metrics, values):
    ax.axis('off')  # Hide axes
    ax.text(0.5, 0.65, metric.replace('_', ' '), ha='center', va='center', fontsize=12)
    ax.text(0.5, 0.35, value, ha='center', va='center', fontsize=20)
    # Draw a rectangle border for the card
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, linewidth=1))

plt.tight_layout()
plt.show()


# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# Load and prepare data
df = pd.read_csv('/Users/davidsolis/Desktop/Portfolio Projects/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['Churn_flag'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 1. Descriptive Statistics for Key Numericals
num_stats = df[['tenure', 'MonthlyCharges', 'TotalCharges']].describe()
display(num_stats)

# 2. Churn Rate by Customer Segment
contract_churn = df.groupby('Contract')['Churn_flag'].mean() * 100
internet_churn = df.groupby('InternetService')['Churn_flag'].mean() * 100
payment_churn = df.groupby('PaymentMethod')['Churn_flag'].mean() * 100

# Display segment-level churn rates
display(contract_churn.rename('Churn Rate %').reset_index())
display(internet_churn.rename('Churn Rate %').reset_index())
display(payment_churn.rename('Churn Rate %').reset_index())

# 3. Average Monthly Charges and Tenure by Churn Status
avg_metrics = df.groupby('Churn').agg({
    'MonthlyCharges': 'mean',
    'tenure': 'mean'
}).rename(columns={
    'MonthlyCharges': 'AvgMonthlyCharges',
    'tenure': 'AvgTenure'
})
display(avg_metrics)

# 4. Visualizations

# 4.1 Churn Rate by Contract Type
plt.figure()
contract_churn.plot.bar()
plt.ylabel('Churn Rate (%)')
plt.title('Churn Rate by Contract Type')
plt.tight_layout()
plt.show()

# 4.2 Churn Rate by Internet Service
plt.figure()
internet_churn.plot.bar()
plt.ylabel('Churn Rate (%)')
plt.title('Churn Rate by Internet Service')
plt.tight_layout()
plt.show()

# 4.3 Churn Rate by Payment Method
plt.figure()
payment_churn.plot.bar()
plt.ylabel('Churn Rate (%)')
plt.title('Churn Rate by Payment Method')
plt.tight_layout()
plt.show()

# 4.4 Correlation Matrix for Numeric Drivers
corr = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn_flag']].corr()
plt.figure()
plt.imshow(corr, aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=45)
plt.yticks(range(len(corr)), corr.index)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()


# In[ ]:




