# Customer Churn Prediction (1M customers)

Research design and experiment for predicting customer churn on the Kaggle dataset
[*Customer Churn Prediction Dataset – 1M*](https://www.kaggle.com/datasets/isandeep06/customer-churn-prediction-dataset-1m).

**Research question:** on 1,000,000 customers, does gradient boosting beat logistic regression on PR-AUC, and which signals drive churn risk? (details in `proposal.md`)

## Repository contents

| File | Description |
|---|---|
| `literature_review.md` | Four sources on churn prediction, each summarised in 2–3 sentences |
| `methodology.md` | Research question, dataset description and limitations, cleaning plan, feature-engineering plan, models, evaluation metrics |
| `experiment.ipynb` | Full experiment with outputs saved: cleaning, feature pipeline, two tuned models, evaluation, findings |
| `data/customer_churn_1M.csv.gz` | The 1,000,000-row dataset, gzip-compressed (40 MB; the 170 MB raw CSV exceeds GitHub's 100 MB file limit) |

## Results (test set, 200,000 customers, 9.9% churn)

| Model | PR-AUC | ROC-AUC | F1 | Precision | Recall | Recall@top 10% | Lift@top 10% |
|---|---|---|---|---|---|---|---|
| No-skill baseline | 0.099 | 0.500 | – | – | – | 0.100 | 1.0 |
| Logistic Regression | 0.203 | 0.685 | 0.269 | 0.188 | 0.472 | 0.245 | 2.45 |
| Gradient Boosting (HistGradientBoosting) | 0.206 | 0.685 | 0.269 | 0.199 | 0.412 | 0.247 | 2.47 |

Gradient boosting is only marginally better (PR-AUC +0.004, 95% bootstrap interval excludes zero). Contract type, customer satisfaction, complaints and service calls are the main drivers. The dataset appears to be synthetic, so these results should not be assumed to transfer to real customers. See the last cell of the notebook for the full discussion.

## How to run

```bash
git clone https://github.com/qaiumsust2003-hue/Customer-Churn-Prediction-Dataset_-1M.git
cd Customer-Churn-Prediction-Dataset_-1M
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install pandas numpy scikit-learn matplotlib jupyter
jupyter lab experiment.ipynb                            # then Run > Run All Cells
```

The notebook reads `data/customer_churn_1M.csv.gz` directly, so no extra download is needed. A full run takes a few minutes on a laptop. Tested with Python 3.13, pandas 3.0, scikit-learn 1.9.

## Workflow

Work was done on an `experiment` branch with incremental commits and merged into `main` through a Pull Request.
