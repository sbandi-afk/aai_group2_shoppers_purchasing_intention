# Predicting Online Shoppers Purchasing Intention

## Overview
The rapid expansion of e-commerce has necessitated the development of predictive models
to understand customer behavior and optimize conversion rates. This project aimed to predict
the purchasing intention of online shoppers using the "Online Shoppers Purchasing Intention
Dataset" from the UCI Machine Learning Repository. The study addressed the challenge of a
highly imbalanced dataset, where only 15.5% of sessions resulted in a transaction. The
methodology employed a robust pipeline including Exploratory Data Analysis (EDA), feature
engineering via K-Means clustering, and Synthetic Minority Over-sampling Technique
(SMOTE) to address class imbalance. Five distinct algorithms—Logistic Regression, Random
Forest, Support Vector Machine (SVM), XGBoost, and Multilayer Perceptron (MLP)—were
trained and optimized using GridSearch, a HyperParameter Tuning technique. A Voting
Classifier ensemble was subsequently developed to aggregate predictions. The Voting
Classifier achieved the highest stability and performance, yielding an F1-score of 0.66 and an
Area Under the Curve (ROC-AUC) of 0.89 on the test set. Key findings indicated that
PageValues and ExitRates were the most significant predictors of revenue. These results
demonstrate that ensemble learning, combined with unsupervised clustering features,
provides a reliable framework for real-time prediction of purchasing intention of online
shoppers.

## Objectives
- **Predict Purchasing Intention**: Accurately classify whether a shopping session will result in a transaction (Revenue).
- **Behavioral Analysis**: Use clustering to identify distinct shopper segments based on engagement and activity.
- **Model Comparison**: Evaluate multiple machine learning algorithms to find the most effective predictor.
- **Handling Imbalance**: Address the class imbalance inherent in conversion data using appropriate metrics (Recall, F1) and techniques (SMOTE, Class Weights).

## Methods
- **Clustering**: K-Means clustering was employed to discover natural groupings of user sessions, revealing high-engagement vs. low-engagement segments.
- **Classification Models**: The following supervised learning models were trained and tuned:
  - Logistic Regression in Python
  - Support Vector Machine (SVM)
  - Random Forest Classifier
  - XGBoost Classifier
  - Multilayer Perceptron (MLP)
- **Ensemble Learning**: A Soft Voting Classifier combined the top-performing models to improve generalization and robustness.

## Data Preparation
- **Source**: [UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).
- **Cleaning**: Checked for and removed duplicate sessions. Verified no missing values were present.
- **Feature Engineering**:
  - **Encoding**: Categorical variables (e.g., Month, VisitorType) were transformed using One-Hot Encoding.
  - **Scaling**: Numeric features (e.g., Duration, PageValues) were standardized using StandardScaler.
  - **Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique) and class weighting strategies were used to support training on the minority class (Revenue=True).
  - Key features include:
    - Session behavior: Administrative, Informational, ProductRelated and durations
    - Engagement: PageValues, SpecialDay
    - Technical: Browser, OperatingSystems, Region, TrafficType
    - Temporal: Month, Weekend
    - Target: Revenue (boolean)

## Exploratory Data Analysis
- **Univariate Analysis**: Histograms and distributions for features like `PageValues`, `ExitRates`, and session durations.
- **Bivariate Analysis**: Stacked bar charts showing conversion rates across different Categories (Month, TrafficType).
- **Correlation**: Heatmaps revealed strong positive correlations between `PageValues` and Revenue, and negative correlations with `ExitRates`.
- **Seasonality**: Observed higher traffic and conversion rates in months like November and May.

## Results
The models were evaluated on a held-out test set. The **Soft Voting Classifier** provided the best overall balance of F1-score and accuracy.

| Model | Recall | Precision | F1 Score | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Voting Classifier (Soft)** | **0.7749** | **0.5759** | **0.6607** | **0.8754** |
| MLP Classifier | 0.8377 | 0.5229 | 0.6439 | 0.8549 |
| SVC | 0.7644 | 0.5489 | 0.6390 | 0.8648 |
| Logistic Regression | 0.7277 | 0.5673 | 0.6376 | 0.8705 |
| Random Forest | 0.7330 | 0.5556 | 0.6321 | 0.8664 |
| XGBoost | 0.6649 | 0.5546 | 0.6048 | 0.8639 |

## Analysis

## Repository Structure
- `AAI501_Final_Project_Team2_v6_final.ipynb`: The primary Jupyter Notebook containing the full end-to-end workflow (Data Loading, EDA, Modeling, Evaluation).
- `MSAAI-501-Final-Project-Report-Team-2-final.pdf`: The comprehensive project report detailing methodology, experiments, and conclusions.
- `README.md`: This file, providing an overview and guide to the project.

## How to Run
### Prerequisites
Ensure you have Python 3.10+ installed. Install the required dependencies:

```bash
pip install jupyter numpy pandas scikit-learn xgboost matplotlib seaborn imbalanced-learn ucimlrepo
```

### Execution
1.  Clone this repository.
2.  Navigate to the project directory.
3.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook AAI501_Final_Project_Team2_v6_final.ipynb
    ```
4.  Run all cells sequentially to reproduce the analysis and results.


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

## References

Sakar, C.O., Polat, S., Katircioglu, H.K., & Korkmaz, S. (2018). Online Shoppers Purchasing Intention Dataset. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

Abdullah-All-Tanvir, Khandokar, I. A., Islam, A. M., Islam, S., & Shatabda, S. (2023). A gradient boosting classifier for purchase intention prediction of online shoppers. Heliyon, 9(4), e15163. https://doi.org/10.1016/j.heliyon.2023.e15163

Adhikari, J. (2023). Online shoppers' purchase intention using ensemble learning approach. International Journal of Next-Generation Computing, 14(4). https://doi.org/10.47164/ijngc.v14i4.1065

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, 321-357. https://doi.org/10.1613/jair.953

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785-794). ACM. https://doi.org/10.1145/2939672.2939785

Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273-297. https://doi.org/10.1007/BF00994018

MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. In L. M. Le Cam & J. Neyman (Eds.), Proceedings of the fifth Berkeley symposium on mathematical statistics and probability (Vol. 1, pp. 281-297). University of California Press.

## Appendix

## YouTube Link : https://youtu.be/Kbz0VpzSZkk

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

## License
No explicit license is included in the notebook. Code here is for academic purposes. Refer to the UCI dataset license for data usage terms.



