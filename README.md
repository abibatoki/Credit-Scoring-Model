# 🏦 Credit Scoring Model for ABC Bank

> 🚀 **Best Model Accuracy | Threshold: 0.25 | XGBoost: 82%**

This project focuses on building a credit scoring model for a fictional bank called **ABC Bank**. The goal was to help the bank identify customers who are likely to default on their loans and support smarter lending decisions that improve profitability. This project demonstrates how data science can directly inform business decisions, particularly when balancing risk and profit.

---
## 📦 Code and Resources Used

**Python Version:** 3.10

**Libraries:**
- pandas, numpy
- scikit-learn, imblearn
- xgboost, random forest
- seaborn, matplotlib
- Streamlit (for dashboard — coming soon)
-  💡 **Interpretation of model probabilities and profitability analysis** was inspired by this YouTube video:  
  [Credit Risk Modelling | Interpretation and Business Recommendation]([https://www.youtube.com/watch?v=xyz123abc](https://www.youtube.com/watch?v=8jzvzRo3Ij0&t=1271s)  

---

## 📁 Dataset Description

There are 30 variables and 3000 observations on historical loan applicants and their repayment behaviour. The target variable indicates whether a customer is a **good loan (0)** or **bad loan (1)**. I also included a separate file in this repository that describes the dataset in detail. 

<img width="600" height="200" alt="image" src="https://github.com/user-attachments/assets/f2864e8c-8399-46db-9132-65a75e5ab51d" />

---

## ⚙️ Model Building and Evaluation

### Steps Taken:

1. **Data Cleaning**
   - Corrected column typos (e.g., “banruptcy” → “bankruptcy”)
   - Encoded categorical variables and scaled numerical ones
   - Balanced the dataset using SMOTE

2. **Model Training**
   - Started with logistic regression as a baseline
   - Improved performance with Random Forest and GridSearchCV
   - Final model used **XGBoost with threshold tuning**

3. **Threshold Tuning**
   - The default threshold (0.5) was too high for business needs
   - I lowered it to **0.25** to capture more defaults and minimize financial loss

4. **Feature Selection**
   - Used XGBoost to extract the **top 10 most important features** identified by the model
   - The features align well with domain expectations and provide clear indicators of borrower risk
   - Most are related to delinquency history, credit utilization, and recent credit-seeking behavior, all of which correlate with a higher likelihood of default (Target = 1). 

### 🔍 Top 10 Features Driving Loan Default

| Feature         | Description                                                              | Business Insight & Link to Target                                         |
|----------------|---------------------------------------------------------------------------|----------------------------------------------------------------------------|
| TLDel60Cnt24    | Number of trade lines 60+ days overdue (past 24 months)                  | Recent delinquency → Strong signal of **default**                         |
| TLDel3060Cnt24  | Number of trade lines 30–60 or 60+ days overdue (past 24 months)         | Early payment trouble → Likely **defaulters**                             |
| TLCnt03         | Number of trade lines opened in last 3 months                            | Many new accounts → Possible **credit shopping** / **instability**        |
| CollectCnt      | Number of accounts sent to collections                                   | Direct evidence of past **default behavior**                              |
| TL75UtilCnt     | Number of trade lines with >75% utilization                              | **Credit overuse** → Higher financial stress and **default** risk         |
| TLDel60Cnt      | Number of trade lines currently 60+ days late                            | **Ongoing delinquencies** → Major red flag for lenders                    |
| InqTimeLast     | Time since last credit inquiry                                           | Very recent inquiries → Indicates **credit hunger**                       |
| InqFinanceCnt24 | Number of finance-related inquiries (past 24 months)                     | High value → Frequent financing need → May be **overleveraged**           |
| InqCnt06        | Total inquiries over past 6 months                                       | Frequent inquiries → **Aggressive credit-seeking behavior**              |
| BankruptcyInd   | Bankruptcy history (Yes/No flag)                                         | Known **default** indicator → Directly tied to target = 1                 |

---

## 📊 Model Performance

- The XGBoost model with a threshold of 0.25 gave the best trade-off between recall and profitability.
- AUC-ROC score showed strong model separation between good and bad loans.
- Classification metrics showed significant improvement over the baseline.

<img width="2354" height="1756" alt="roc_curve" src="https://github.com/user-attachments/assets/736f22e8-e2a1-4dfe-b067-8446ed5c0fb1" />

---

### 📊 ROC Curve Interpretation

The ROC curve illustrates the model’s ability to distinguish between good and bad loans across all probability thresholds.

- **Curve Shape**:  
  The curve rises steeply toward the top-left corner, indicating that the model does a good job of identifying most bad loans (high sensitivity) while minimizing false positives (wrongly labeling good loans as bad).

- **Baseline Comparison**:  
  The diagonal dashed line represents random guessing. Since the ROC curve lies well above it, the model clearly performs better than chance.

- **Implied AUC**:  
  While the exact AUC value isn't shown here, the curve’s shape suggests strong discriminatory power, likely between 0.75 and 0.80. This implies that:
  
  > *There’s a 75–80% chance the model assigns a higher risk score to a bad customer than to a good one.*

---

### 🏦 Business Implication

This confirms that the model is reliable for separating risky borrowers from safe ones. This is a critical requirement for making informed lending decisions in subprime markets. By fine-tuning the threshold (e.g., 0.25 in this case), ABC Bank can optimize profits ($100 per good loan) while minimizing losses ($500 per bad loan).

---

## 📈 Profitability Analysis

To assess how this model performs in business terms, I grouped customers into deciles based on predicted probability.

- **Deciles 1–4** resulted in **financial losses**
- **Deciles 5–10** generated **positive profit**
- Interestingly, **Decile 2** generated more loss than Decile 1, despite Decile 1 being riskier by probability. This confirms the need to evaluate models with both statistical and business metrics.

<img width="2953" height="1456" alt="profit_to_business_by_decile" src="https://github.com/user-attachments/assets/117a9d05-5a16-4be4-9700-c0458bedd9a3" />

---

### 📉 Sensitivity and Specificity by Decile

This chart compares **Sensitivity (Cumulative Good %)** and **Specificity (Bad Avoided %)** across customer deciles, ordered by predicted probability of being a good loan.

<img width="2953" height="1454" alt="sensitivity_specificity" src="https://github.com/user-attachments/assets/68dace99-a699-4421-8c6a-6e8f5d9ad677" />

- **Sensitivity (Blue Line)**:  
  This line rises steadily from left to right. It tells us how well the model captures good loans. By the 5th decile, the model has already identified about 50% of the good customers, showing that most low-risk customers are ranked in the higher deciles.

- **Specificity (Orange Line)**:  
  This line falls sharply, indicating that fewer bad customers are present in the later deciles. High specificity at lower deciles (especially Decile 1 and 2) confirms that the model has effectively pushed the riskiest customers to the bottom.

---

### 🏦 Business Implication

This decile-based view provides strong evidence of model **segmentation power**:

- Customers in the top deciles (e.g., 8–10) are largely **good loans** — ideal candidates for approval.
- The bottom deciles (e.g., 1–3) are where most **bad loans** are concentrated and should be approached with caution or rejected.

This insight allows ABC Bank to tailor credit decisions by **prioritizing customers in deciles with high sensitivity and low default risk**, aligning model performance with the profitability goals of minimizing $500 losses while capturing $100 profits.

---

## 💡 Recommendations

- **Avoid lending to Deciles 1–4**: These customer segments pose the highest risk and lowest profitability.
- **Target Deciles 5–10** for loan approvals: They offer a safer balance of risk and return.
- **Use threshold tuning carefully**: The 0.25 threshold aligned well with ABC Bank’s profit-loss structure.
- **Retrain regularly**: The model should be updated with fresh data to remain accurate.
- **Use as decision-support tool**: While the model helps flag risk, human oversight should still be part of the loan review process.

---

Thanks for checking out this project!  
Feel free to reach out if you’d like to collaborate or ask questions about the methodology or insights.





