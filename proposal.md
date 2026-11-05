# Project Proposal

## Research question
On a 1,000,000-customer dataset, does a gradient-boosted tree model rank customers by churn risk better than a regularised logistic regression, measured by PR-AUC on a held-out test set, and which customer signals (contract, satisfaction, complaints, service calls) drive the risk?

## Dataset
- **Link:** https://www.kaggle.com/datasets/isandeep06/customer-churn-prediction-dataset-1m
- **Source:** Kaggle dataset by `isandeep06`; profiling shows it is very likely synthetic (injected outliers, inconsistent service counts, demographics unrelated to churn).
- **Size and target:** 1,000,000 rows, 32 columns; binary target `churn` with about 9.9% positives.
- **Technique:** supervised classification (covered in class). The dataset is bundled in the repo as `data/customer_churn_1M.csv.gz`.

## Scan of existing work
1. **Ahmad, Jafar & Aljoumaa (2019), "Customer churn prediction in telecom using machine learning in big data platform", Journal of Big Data 6:28.** The authors compared Decision Tree, Random Forest, Gradient Boosted Machine and XGBoost on nine months of SyriaTel customer data. XGBoost was best with an AUC of 93.3%, and adding social-network features lifted AUC from 84% to 93.3%, so feature engineering mattered as much as the algorithm. (https://arxiv.org/abs/1904.00690)
2. **Saito & Rehmsmeier (2015), "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets", PLoS ONE 10(3): e0118432.** They show that ROC curves can look good on imbalanced data because easy negatives dominate, while precision-recall curves reflect the performance a practitioner actually sees. This justifies PR-AUC as the primary metric for a dataset with only about 10% churners. (https://doi.org/10.1371/journal.pone.0118432)

## Plan
Clean and engineer features, train logistic regression and gradient boosting, evaluate on PR-AUC, F1 and top-10% lift. The full plan is in `methodology.md` and the results are in `experiment.ipynb`.
