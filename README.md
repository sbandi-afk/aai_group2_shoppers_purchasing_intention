# Online Shoppers Purchasing Intention — Team 2 Final Project

A complete, reproducible workflow to explore, model, and evaluate purchasing intention using the UCI Online Shoppers Purchasing Intention Dataset. The project covers data cleaning, exploratory analysis, clustering to discover behavior segments, and multiple classification models (Logistic Regression, SVM, Random Forest, XGBoost, and MLP), followed by model comparison and a soft voting ensemble. Evaluation centers on recall and F1 to address class imbalance.

Dataset: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

## Repository contents
- AAI501_Final_Project_Team2_v6_final.ipynb — main end-to-end notebook (data prep, EDA, clustering, modeling, evaluation)
- MSAAI-501-Final-Project-Report-Team-2.pdf — written report with results and discussion
- README.md — this project overview and usage guide

## Project description
Predict whether an online shopping session results in a purchase (Revenue = True) from behavioral and contextual features across 12,330 sessions. We handle mixed numeric/categorical data, class imbalance, and evaluate multiple models. Clustering is used to uncover behavioral segments linked to conversion.

Key features include:
- Session behavior: Administrative, Informational, ProductRelated and durations
- Engagement: PageValues, SpecialDay
- Technical: Browser, OperatingSystems, Region, TrafficType
- Temporal: Month, Weekend
- Target: Revenue (boolean)

## Setup / installation
Tested with Python 3.10+ on Windows.

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install jupyter numpy pandas scikit-learn xgboost matplotlib seaborn imbalanced-learn
```
Optional:
- JupyterLab: `pip install jupyterlab`
- If your platform needs a specific xgboost: `pip install xgboost==1.7.6`

## Usage
Open and run the notebook end-to-end:

```bash
jupyter notebook AAI501_Final_Project_Team2_v6_final.ipynb
```

- Run cells in order. The notebook downloads or loads the UCI dataset, preprocesses, trains models, and reports metrics.
- Keep the provided random_state values and stratified splits to reproduce results.
- Outputs include plots (EDA, clustering, ROC/PR curves, confusion matrices) and printed metric tables.

## Notebook highlights
- Introduction and objectives
  - States the goal: predict Revenue (True/False) and explore behavioral segments
  - Notes class imbalance and evaluation focus on recall/F1
- Data loading
  - Fetch metadata and dataset details from UCI (repository URL, feature descriptions)
  - Load CSV into pandas; display shape, head, dtypes
- Data quality checks
  - Check missing values (none expected per UCI), duplicates, and drop duplicates if found
  - Review numeric ranges and categorical cardinality; verify target distribution (~15% positives)
- Data type conversion
  - Convert categorical-like columns (Month, VisitorType, Weekend) to appropriate types
  - Ensure boolean/strings are consistently represented before encoding
- Exploratory Data Analysis (EDA)
  - Univariate histograms for numeric features (durations, PageValues, rates)
  - Stacked bar charts for categorical features vs. Revenue
  - Correlation heatmap among numeric variables; pairwise patterns
  - Revenue uplift by Month, Weekend, TrafficType, Region
- Feature relevance (optional)
  - Mutual information computed using one-hot encoded categoricals to gauge signal
- Train/validation/test split
  - Stratified split to preserve class ratios; fixed random_state for reproducibility
- Preprocessing pipeline
  - ColumnTransformer: OneHotEncoder(drop='first', handle_unknown='ignore') for categoricals; StandardScaler() for numerics
  - Fit on train only; transform validation/test to avoid leakage
- Clustering and visualization
  - KMeans on scaled behavioral subset; choose k using inertia (elbow) and silhouette score
  - t-SNE used to visualize session clusters in 2D; analyze cluster-level Revenue propensity
- Model training and tuning
  - LogisticRegression(max_iter=1000, solver='liblinear'); tune C and class_weight/threshold
  - SVC(probability=True, random_state=42); tune kernel, C, gamma
  - RandomForestClassifier() with class_weight grid; tune n_estimators, max_depth, min_samples_leaf
  - XGBClassifier(use_label_encoder=False, eval_metric='logloss'); tune max_depth, learning_rate, n_estimators, subsample
  - MLPClassifier(max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20); tune hidden_layer_sizes, alpha
  - Cross-validation on training set; re-instantiate with best params and refit on full train
- Ensembling and calibration
  - Construct VotingClassifier(estimators=best_estimators_list, voting='soft') from top models
  - Calibrate probabilities where needed; tune decision thresholds via validation PR curve
- Final evaluation
  - Report on held-out test: Recall, Precision, F1, ROC-AUC, Balanced Accuracy, Accuracy
  - Produce confusion matrices and optionally calibration curves

## Results

- Voting Classifier (soft):
  - Recall: 0.7749, Precision: 0.5759, F1: 0.6607, ROC-AUC: 0.8892, Balanced Acc: 0.8345, Accuracy: 0.8754
- MLPClassifier:
  - Recall: 0.8377, Precision: 0.5229, F1: 0.6439, ROC-AUC: 0.8880, Balanced Acc: 0.8479, Accuracy: 0.8549
- SVC:
  - Recall: 0.7644, Precision: 0.5489, F1: 0.6390, ROC-AUC: 0.8576, Balanced Acc: 0.8239, Accuracy: 0.8648
- Logistic Regression:
  - Recall: 0.7277, Precision: 0.5673, F1: 0.6376, ROC-AUC: 0.8806, Balanced Acc: 0.8124, Accuracy: 0.8705
- RandomForestClassifier:
  - Recall: 0.7330, Precision: 0.5556, F1: 0.6321, ROC-AUC: 0.8796, Balanced Acc: 0.8121, Accuracy: 0.8664
- XGBClassifier:
  - Recall: 0.6649, Precision: 0.5546, F1: 0.6048, ROC-AUC: 0.8517, Balanced Acc: 0.7829, Accuracy: 0.8639

### Observations from analysis
- Class imbalance and thresholding
  - Revenue ≈ 15% creates a strong accuracy baseline; optimizing decision thresholds on validation data materially improved recall and F1 over default 0.5.
  - Models tuned for higher recall increased false positives slightly, which is acceptable for conversion-focused scenarios.
- Model trade-offs
  - MLP achieved the highest recall (0.84) but with lower precision; SVC and Logistic Regression offered more balanced precision/recall.
  - Tree ensembles (RandomForest, XGB) were competitive but slightly behind on recall; they contributed complementary signals to the ensemble.
- Ensemble behavior and calibration
  - The soft-voting ensemble delivered the best F1 (0.66) by blending diverse decision boundaries (linear, kernel, neural, tree-based).
  - Probability calibration and per-model threshold tuning improved ensemble voting quality and class separation.
- Feature signals and directionality (EDA + model explainability)
  - Strong positive signals: PageValues, ProductRelated and ProductRelated_Duration, SpecialDay proximity, Weekend, Returning Visitor.
  - Negative signals: high ExitRates and BounceRates correlate with no purchase.
  - Temporal patterns: certain months (e.g., late-year seasonality) show higher conversion rates; Region/Browser effects are modest.
- Clustering insights
  - K-Means revealed high-engagement segments (high PageValues and long product durations) with 3–5x Revenue rate vs. low-engagement clusters.
  - Mid-engagement clusters benefited most from model-based thresholding, where small probability gains flipped outcomes to positive.
- Error analysis
  - False negatives commonly had moderate PageValues but elevated ExitRates; false positives showed high engagement patterns without checkout completion.
  - Balanced Accuracy ≥ 0.78 across models indicates improved coverage of both classes despite imbalance.
- Practical implications
  - Use ensemble scores to prioritize retargeting; relax thresholds for high-lifetime-value campaigns to maximize recall.
  - Target high-engagement clusters with timely nudges (cart reminders, discounts), especially on weekends and high-PageValue sessions.
  - Reduce ExitRates by surfacing checkout CTAs earlier for mid-engagement users.

## Implementation details
Based on searching the notebook source, we found these exact configurations:
- Preprocessing
  - ColumnTransformer with:
    - OneHotEncoder(drop='first', handle_unknown='ignore') applied to categorical columns (Browser, OperatingSystems, Region, TrafficType, Month, VisitorType, Weekend)
    - StandardScaler() applied to numeric columns (Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues, SpecialDay)
  - Fit on training only; transform validation/test to avoid leakage
- Clustering and visualization
  - KMeans on scaled numeric/categorical-encoded features; k chosen via elbow and silhouette
  - t-SNE used for 2D visualization of clusters
- Models (constructors used before tuning)
  - LogisticRegression(max_iter=1000, solver='liblinear')
  - SVC(probability=True, random_state=42)
  - RandomForestClassifier() with class_weight candidates ['balanced', 'balanced_subsample'] explored during search
  - XGBClassifier(use_label_encoder=False, eval_metric='logloss')
  - MLPClassifier(max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20)
- Hyperparameter search
  - Per-model GridSearchCV/RandomizedSearchCV; final models re-instantiated with Best Parameters and refit
- Ensemble
  - VotingClassifier(estimators=best_estimators_list, voting='soft') combining the tuned top models

## Key code snippets
- ColumnTransformer (encode + scale)
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

categorical_pipeline = Pipeline([
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])
numeric_pipeline = Pipeline([
    ('scaler', StandardScaler())
])
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, num_cols),
    ('cat', categorical_pipeline, cat_cols)
])
```

- KMeans and t-SNE visualization
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

# choose k via elbow/silhouette
inertias, silhouettes = [], []
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

# t-SNE for 2D visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
emb = tsne.fit_transform(X_scaled)
```

- Model search and ensemble
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier

models = [
    { 'name': 'LogisticRegression', 'model': LogisticRegression(max_iter=1000, solver='liblinear'), 'param_grid': {...} },
    { 'name': 'SVC', 'model': SVC(probability=True, random_state=42), 'param_grid': {...} },
    { 'name': 'RandomForestClassifier', 'model': RandomForestClassifier(), 'param_grid': {...} },
    { 'name': 'XGBClassifier', 'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss'), 'param_grid': {...} },
    { 'name': 'MLPClassifier', 'model': MLPClassifier(max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20), 'param_grid': {...} },
]
# Fit CV per model, collect best estimators
voting_clf = VotingClassifier(estimators=best_estimators_list, voting='soft')
```

- Threshold tuning on validation set (optimize F1/recall)
```python
from sklearn.metrics import precision_recall_curve, f1_score

proba = model.predict_proba(X_val)[:, 1]
prec, rec, th = precision_recall_curve(y_val, proba)
# choose threshold by best F1
best_idx = max(range(len(th)), key=lambda i: f1_score(y_val, (proba >= th[i]).astype(int)))
best_threshold = th[best_idx]
```

## Dependencies
- Python 3.10+
- jupyter
- numpy
- pandas
- scikit-learn
- xgboost
- matplotlib
- seaborn
- imbalanced-learn

If you need exact versions, capture your environment with:
```
pip freeze > requirements.txt
```

## License
No explicit license is included in the notebook. Code here is for academic purposes. Refer to the UCI dataset license for data usage terms.

## Citation
Sakar, C.O., Polat, S., Katircioglu, H.K., & Korkmaz, S. (2018). Online Shoppers Purchasing Intention Dataset. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

Abdullah-All-Tanvir, Khandokar, I. A., Islam, A. M., Islam, S., & Shatabda, S. (2023). A gradient boosting classifier for purchase intention prediction of online shoppers. Heliyon, 9(4), e15163. https://doi.org/10.1016/j.heliyon.2023.e15163

Adhikari, J. (2023). Online shoppers' purchase intention using ensemble learning approach. International Journal of Next-Generation Computing, 14(4). https://doi.org/10.47164/ijngc.v14i4.1065

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, 321-357. https://doi.org/10.1613/jair.953

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785-794). ACM. https://doi.org/10.1145/2939672.2939785

Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273-297. https://doi.org/10.1007/BF00994018

MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. In L. M. Le Cam & J. Neyman (Eds.), Proceedings of the fifth Berkeley symposium on mathematical statistics and probability (Vol. 1, pp. 281-297). University of California Press.
