# Methodology

## 1. Research Question

**Primary question.** On the 1,000,000-record customer churn dataset, does a gradient-boosted tree classifier rank customers by churn risk better than a regularised logistic regression baseline, as measured by PR-AUC (average precision) on a held-out test set, and which customer-experience signals (satisfaction, complaints, service calls, contract type) account for the difference?

**Operational sub-question.** If a retention team can only contact the 10% of customers with the highest predicted risk, how many of the true churners does each model capture (recall@10%), and how much better is that than contacting customers at random (lift@10%)?

**Hypotheses.**
- H0: The gradient-boosted model and the logistic regression have the same PR-AUC on the test set.
- H1: The gradient-boosted model has a higher PR-AUC. This is judged with a paired bootstrap of the test set, so a difference is only called meaningful if its 95% interval excludes zero.

## 2. Dataset Description

**Source.** Kaggle, *Customer Churn Prediction Dataset – 1M* (`isandeep06/customer-churn-prediction-dataset-1m`), file `customer_churn_1M.csv`.

**Size.** 1,000,000 rows and 32 columns: one identifier, 30 predictors and the target. No duplicate customer IDs or duplicate rows were found.

**Target.** `churn` (1 = customer churned, 0 = retained). About 9.9% of customers (roughly 99,000) are churners, so the problem is moderately imbalanced (about 9 retained customers for every churner).

| Group | Columns |
|---|---|
| Identifier / date | `customer_id`, `signup_date` |
| Demographics | `age`, `gender`, `annual_income`, `education`, `marital_status`, `dependents`, `senior_citizen`, `credit_score` |
| Contract and billing | `tenure`, `contract`, `payment_method`, `paperless_billing`, `monthlycharges`, `totalcharges`, `late_payments` |
| Services | `num_services`, `has_phone_service`, `has_internet_service`, `has_online_security`, `has_online_backup`, `has_device_protection`, `has_tech_support`, `has_streaming_tv`, `has_streaming_movies` |
| Behaviour and experience | `customer_satisfaction`, `num_complaints`, `num_service_calls`, `avg_monthly_gb`, `days_since_last_interaction` |

**Missing values.** Five columns contain missing values: `avg_monthly_gb` (5.0%), `credit_score` (4.0%), `annual_income` (3.0%), `num_complaints` (3.0%) and `customer_satisfaction` (2.0%). The churn rate is about 9.9% both for rows with and without a missing value in each of these columns, so the missingness does not appear to carry information about the target.

**Limitations and biases identified during profiling.**
1. **Likely synthetic data.** Demographic fields (age, gender, income, education, marital status, credit score) show essentially no relationship with churn, and `signup_date` is unrelated to `tenure`. Conclusions are about this dataset's structure and should not be assumed to transfer to a real customer base.
2. **Internal inconsistencies.** `num_services` agrees with the sum of the individual service flags in only 12% of rows, about 129,000 customers (12.9%) have an add-on service flagged without internet service, and `totalcharges` differs from `tenure × monthlycharges` by roughly -35% to +10% for most customers. These columns are therefore treated as separate, noisy signals rather than exact functions of each other.
3. **Implausible outliers.** About 980 customers have `monthlycharges` between roughly 257 and 855 (the median is about 85) and about 3,000 have `avg_monthly_gb` above 300.
4. **Skewed categories.** Month-to-month contracts make up only about 2% of customers, so estimates for that group are noisier.
5. **Weak overall signal.** The strongest relationships with churn are contract type, satisfaction, complaints and service calls; most other features are close to uninformative. High ceilings on accuracy-type metrics should not be expected.
6. **Snapshot data.** There is a single row per customer with no time dimension, so the study measures association at one point in time, not forecasting, and leakage from the future cannot be fully excluded without a data dictionary.

## 3. Data Cleaning Plan

Stateless rules are applied to the whole table before splitting. Anything that learns a statistic from the data (clipping limits, medians) is fitted on the training split only, inside the model pipeline, to avoid leakage.

1. **Load and audit.** Check shape, dtypes, missing counts, duplicate IDs and duplicate rows, and the class balance.
2. **Drop the identifier.** Remove `customer_id` after the duplicate check, since it carries no predictive meaning.
3. **Fix formatting.** Strip whitespace and lower-case all text categories; convert `paperless_billing` from Yes/No to 1/0; parse `signup_date` as a timestamp and derive `signup_year` and `signup_month`, then drop the raw string.
4. **Apply domain validity rules.** Values outside plausible ranges (age outside 18–100, tenure below 1, satisfaction outside 1–10, credit score outside 300–850, any negative count, charge or usage) are set to missing. The number of violations is printed; none are expected.
5. **Check logical consistency.** Count rows that report an add-on service (for example online security) without internet service.
6. **Split.** Stratified 70% train / 10% validation / 20% test with a fixed seed (42). The test set is not touched until the final evaluation.
7. **Handle outliers (train-fitted).** Winsorise `monthlycharges`, `totalcharges`, `avg_monthly_gb` and `annual_income` at the 0.5th and 99.5th percentiles of the training data. Rows are kept, because the extreme values are rare and removal would bias the sample.
8. **Impute missing values (train-fitted).** Replace missing values in the five affected columns with the training median and add a binary `<column>_missing` indicator for each, so the model can use the missingness pattern if it matters.

## 4. Feature Engineering Plan

All features are built inside a scikit-learn pipeline so that the identical transformation is applied to validation and test data.

**Derived variables**
- `avg_billed_per_month` = `totalcharges` / `tenure`
- `billing_ratio` = `totalcharges` / (`tenure` × `monthlycharges`), clipped to [0, 2], to capture discounts or billing irregularities
- `charge_to_income` = 12 × `monthlycharges` / `annual_income`
- `n_addons` = count of the six add-on service flags (security, backup, device protection, tech support, streaming TV, streaming movies)
- `support_contacts` = `num_complaints` + `num_service_calls`
- `complaints_per_month` and `calls_per_month` = counts divided by `tenure`
- `low_satisfaction` = 1 if `customer_satisfaction` ≤ 4
- `inactive_90d` = 1 if `days_since_last_interaction` > 90
- `tenure_bucket` = ordinal bands (1–6, 7–12, 13–24, 25–48, 49–72 months)
- `log_totalcharges`, `log_avg_monthly_gb`, `log_annual_income` (log1p) to reduce right skew for the linear model

**Encodings**
- Ordinal encoding with a natural order: `education` (high school < college < bachelor < master < PhD) and `contract` (month-to-month < one year < two year, stored as term length in months).
- One-hot encoding for unordered categories: `gender`, `marital_status`, `payment_method` (unknown categories ignored at prediction time).
- Standardisation (zero mean, unit variance) of numeric features for logistic regression only; tree models do not need scaling.

**Class imbalance.** No synthetic oversampling is used. With about 99,000 positive cases, the models have ample minority examples. Imbalance is handled by (a) selecting the decision threshold on the validation set and (b) treating class weighting as a tunable hyperparameter for logistic regression.

## 5. Models

| Model | Role | Why it is appropriate |
|---|---|---|
| **Logistic Regression** (L2-regularised) | Interpretable baseline | Fast and stable at 1M rows; produces calibrated probabilities and readable coefficients that a retention team can interpret; a strong benchmark in the churn literature, and a fair test of whether non-linear structure actually helps. Tuned over regularisation strength `C` and class weighting. |
| **Histogram Gradient Boosting** (`HistGradientBoostingClassifier`) | Main model | A LightGBM-style algorithm that captures non-linear effects and interactions (for example, low satisfaction combined with many service calls and a short contract) without manual specification, handles outliers and skew natively, and trains quickly on a million rows with early stopping. Boosted trees are the consistently strongest family in the reviewed churn studies. Tuned over learning rate, leaf count, depth, minimum leaf size and L2 regularisation. |

Hyperparameters are tuned with stratified 3-fold cross-validation on a 100,000-row subsample of the training split, scoring PR-AUC, and the best configuration is then refitted on the full training split. A trivial prevalence-only classifier is reported as a floor for comparison.

## 6. Evaluation Metrics

| Metric | Role | Justification |
|---|---|---|
| **PR-AUC (average precision)** | **Primary** | Summarises the precision–recall trade-off across all thresholds and is far more informative than ROC-AUC when positives are only about 10% of the data (Saito & Rehmsmeier, 2015). The no-skill baseline equals the churn prevalence. |
| **F1-score** at a validation-tuned threshold | **Primary (decision-level)** | Balances precision (not wasting retention offers) and recall (not missing churners) in one number; the threshold is chosen on the validation set and then frozen. |
| Precision and recall | Secondary | Show the business trade-off directly: missed churners cost lost lifetime value, while false alarms cost incentive spend. |
| **Recall@top-10% and lift@top-10%** | Secondary (business) | Mirrors a budget-limited campaign that contacts only the highest-risk decile (Verbeke et al., 2012); lift is the factor by which this beats contacting customers at random. |
| ROC-AUC | Secondary | Threshold-free ranking quality, included for comparability with other studies. |
| Brier score | Secondary | Measures the quality of the predicted probabilities, which matters if scores feed an expected-value calculation. |

Accuracy is deliberately not used for model selection, because a classifier that predicts "no churn" for everyone would already score about 90%. Uncertainty is quantified with a 100-resample paired bootstrap of the test set, giving 95% intervals for each model's PR-AUC and for the difference between models.
