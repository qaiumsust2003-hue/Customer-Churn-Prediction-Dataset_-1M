# Literature Review: Customer Churn Prediction

**Project:** Customer Churn Prediction (Kaggle "Customer Churn Prediction Dataset – 1M")
**Purpose:** Identify what prior work says about modelling approaches, feature engineering and evaluation for churn problems with a rare positive class, and use it to shape the experiment in this repository.

---

## 1. Ahmad, Jafar & Aljoumaa (2019): Churn prediction on a big-data platform

**Citation:** Ahmad, A. K., Jafar, A., & Aljoumaa, K. (2019). *Customer churn prediction in telecom using machine learning in big data platform.* Journal of Big Data, 6, Article 28. ([preprint](https://arxiv.org/abs/1904.00690))

**Summary:** The authors built a churn model for the telecom operator SyriaTel on a Spark platform, turning nine months of raw customer records into a large feature table and comparing four tree-based learners: Decision Tree, Random Forest, Gradient Boosted Machine and XGBoost. Models were scored with AUC, and XGBoost performed best, reaching an AUC of 93.3%. Adding social-network-analysis features built from customer interaction graphs raised AUC from 84% to 93.3%, which shows that feature engineering mattered as much as the choice of algorithm.

**Relevance:** It supports using a gradient-boosted tree model against a simpler baseline and investing effort in engineered features.

---

## 2. Verbeke, Dejaeger, Martens, Hur & Baesens (2012): Profit-driven churn evaluation

**Citation:** Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012). *New insights into churn prediction in the telecommunication sector: A profit driven data mining approach.* European Journal of Operational Research, 218(1), 211–229. ([record](https://econpapers.repec.org/RePEc:eee:ejores:v:218:y:2012:i:1:p:211-229))

**Summary:** The paper points out that churn models are normally judged with purely statistical measures even though the real goal is the profit of a retention campaign, so the wrong model can end up being selected. The authors define a profit-centric performance measure that finds the maximum profit achievable by contacting the best fraction of customers, ranked by predicted churn probability. They report that selecting a model, and a contact fraction, with this measure produces a significant profit gain compared with selecting on statistical measures.

**Relevance:** It motivates evaluating the models on a "top-X% of customers contacted" basis (recall and lift at a fixed campaign size) in addition to the usual classification scores.

---

## 3. Saito & Rehmsmeier (2015): Precision-recall versus ROC on imbalanced data

**Citation:** Saito, T., & Rehmsmeier, M. (2015). *The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets.* PLoS ONE, 10(3), e0118432. ([article](https://doi.org/10.1371/journal.pone.0118432))

**Summary:** The authors compare ROC plots and precision-recall (PR) plots as ways of judging binary classifiers when negatives greatly outnumber positives. They argue that ROC curves can look reassuring because specificity is dominated by the many easy negatives, whereas PR curves show what share of predicted positives are truly positive and therefore reflect the performance a practitioner will actually experience. They conclude that PR plots are the more informative choice on imbalanced data and that studies relying on ROC alone deserve caution.

**Relevance:** With only about 10% churners in the dataset, this is the justification for making PR-AUC (average precision) the primary metric and reporting ROC-AUC only as a secondary one.

---

## 4. Sikri, Jameel, Idrees & Kaur (2024): Class balancing and model comparison

**Citation:** Sikri, A., Jameel, R., Idrees, S. M., & Kaur, H. (2024). *Enhancing customer retention in telecom industry with machine learning driven churn prediction.* Scientific Reports, 14, Article 13097. ([article](https://doi.org/10.1038/s41598-024-63750-0))

**Summary:** Working with a 19-feature Kaggle telecom churn dataset, the authors propose a "ratio-based" balancing step, tested at five class ratios and compared with ordinary over- and under-sampling. They trained eight classifiers, from Perceptron, Naive Bayes and Logistic Regression up to Gradient Boosting and XGBoost, scored them on accuracy, precision, recall and F1, and used a rank-aggregation step to pick winners. Ensemble methods beat single models, with XGBoost best (an F1 of about 60 on the untouched imbalanced data and a reported 89.6% accuracy at a 75:25 ratio), although the paper's tables contain inconsistencies and its claim that ratio balancing is clearly superior is only partly supported.

**Relevance:** It reinforces the ensemble-versus-linear comparison, and its mixed evidence on resampling is one reason this project handles imbalance through threshold tuning and class weighting instead of synthetic oversampling.

---

## Synthesis

- **Model choice:** Gradient-boosted trees are consistently reported as the strongest family on tabular churn data (Ahmad et al.; Sikri et al.), so the experiment pairs one with a regularised logistic regression as an interpretable baseline.
- **Features matter:** The largest single improvement in the reviewed work came from richer features rather than a different algorithm (Ahmad et al.), which justifies the dedicated feature-engineering stage.
- **Evaluation should match the decision:** Accuracy is misleading at about 10% prevalence; PR-AUC, F1 at a tuned threshold, and campaign-style metrics such as recall and lift in the top decile reflect how a retention team would actually use the scores (Saito & Rehmsmeier; Verbeke et al.).
- **Caution with resampling and reported numbers:** Evidence for synthetic balancing is mixed and some published tables are internally inconsistent (Sikri et al.), so every threshold and hyperparameter here is chosen on validation data only and all results are reported on an untouched test set.

## References

1. Ahmad, A. K., Jafar, A., & Aljoumaa, K. (2019). Customer churn prediction in telecom using machine learning in big data platform. *Journal of Big Data, 6*, 28.
2. Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012). New insights into churn prediction in the telecommunication sector: A profit driven data mining approach. *European Journal of Operational Research, 218*(1), 211–229.
3. Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE, 10*(3), e0118432.
4. Sikri, A., Jameel, R., Idrees, S. M., & Kaur, H. (2024). Enhancing customer retention in telecom industry with machine learning driven churn prediction. *Scientific Reports, 14*, 13097.
