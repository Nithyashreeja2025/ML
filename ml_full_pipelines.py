"""
MACHINE LEARNING FULL PIPELINES — BEGINNER CHEATSHEET
====================================================
Each section is a separate runnable example with:
- Step headings and comments
- Preprocessing in a Pipeline (prevents data leakage)
- Train/test split
- Fit and prediction
- Evaluation
- Inference and conclusion

Install once:
    pip install numpy pandas scikit-learn matplotlib xgboost

Run the file:
    python ml_full_pipelines.py

Notes:
1. Classification examples use sklearn's Breast Cancer dataset (binary target).
2. Regression examples use sklearn's Diabetes dataset (numeric target).
3. K-Means/PCA examples use Iris features (unsupervised; target labels are not used
   to train the models).
4. XGBoost is optional. If it is not installed, that section is skipped.
5. Inference/conclusion are based on the metrics printed by the program. Do not
   claim a model is "best" without comparing its test/CV scores.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score
)

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier,
    AdaBoostClassifier, GradientBoostingClassifier,
    BaggingRegressor, RandomForestRegressor,
    AdaBoostRegressor, GradientBoostingRegressor,
    StackingClassifier, StackingRegressor
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ============================================================
# SHARED DATA LOADING FUNCTIONS
# ============================================================

def get_classification_data():
    """Step 1: Load a built-in binary classification dataset."""
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target
    # Step 2: Split before fitting preprocessing, so test data is kept unseen.
    return train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )


def get_regression_data():
    """Step 1: Load a built-in regression dataset."""
    data = load_diabetes(as_frame=True)
    X = data.data
    y = data.target
    # Step 2: Split before fitting preprocessing.
    return train_test_split(X, y, test_size=0.20, random_state=42)


def classification_pipeline(model, scale=True):
    """Step 3: Build preprocessing + model as one pipeline."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def regression_pipeline(model, scale=True):
    """Step 3: Build preprocessing + model as one pipeline."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def run_classifier(name, model, scale=True):
    """
    Full classification pipeline:
    load -> split -> preprocess -> train -> predict -> evaluate -> inference.
    """
    print("\n" + "=" * 72)
    print(f"ALGORITHM: {name}")
    print("=" * 72)

    # Step 1 & 2: Load data and create train/test sets.
    X_train, X_test, y_train, y_test = get_classification_data()
    print("Step 1-2: Dataset loaded and split.")
    print(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)}")

    # Step 3: Impute missing values and optionally scale features.
    pipe = classification_pipeline(model, scale=scale)
    print("Step 3: Preprocessing pipeline created.")

    # Step 4: Train the algorithm using training data only.
    pipe.fit(X_train, y_train)
    print("Step 4: Model training completed.")

    # Step 5: Predict labels for unseen test data.
    y_pred = pipe.predict(X_test)
    print("Step 5: Predictions completed.")

    # Step 6: Evaluate.
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print("\nRESULTS")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification report:\n", classification_report(
        y_test, y_pred, zero_division=0
    ))

    # Step 7: Inference and conclusion (based on actual printed scores).
    print("INFERENCE:")
    print(f"The model correctly classified {acc * 100:.2f}% of test samples.")
    print(f"Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}.")
    print("CONCLUSION:")
    if acc >= 0.90 and f1 >= 0.90:
        print("Performance is strong on this test split; validate with cross-validation.")
    elif acc >= 0.75:
        print("Performance is moderate/good; inspect precision, recall and errors.")
    else:
        print("Performance may need improvement; tune parameters or inspect features.")
    print("Do not compare models from different datasets/splits as if the comparison were fair.")
    return pipe, {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def run_regressor(name, model, scale=True):
    """
    Full regression pipeline:
    load -> split -> preprocess -> train -> predict -> evaluate -> inference.
    """
    print("\n" + "=" * 72)
    print(f"ALGORITHM: {name}")
    print("=" * 72)

    # Step 1 & 2: Load regression data and split.
    X_train, X_test, y_train, y_test = get_regression_data()
    print("Step 1-2: Dataset loaded and split.")
    print(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)}")

    # Step 3: Preprocessing.
    pipe = regression_pipeline(model, scale=scale)
    print("Step 3: Preprocessing pipeline created.")

    # Step 4: Train.
    pipe.fit(X_train, y_train)
    print("Step 4: Model training completed.")

    # Step 5: Predict.
    y_pred = pipe.predict(X_test)
    print("Step 5: Predictions completed.")

    # Step 6: Evaluate.
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print("\nRESULTS")
    print(f"MAE : {mae:.3f}")
    print(f"MSE : {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²  : {r2:.3f}")

    # Optional plot: actual vs predicted.
    plt.figure()
    plt.scatter(y_test, y_pred)
    low = min(float(np.min(y_test)), float(np.min(y_pred)))
    high = max(float(np.max(y_test)), float(np.max(y_pred)))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.title(f"{name}: Actual vs Predicted")
    plt.tight_layout()
    plt.show()

    # Step 7: Inference and conclusion.
    print("INFERENCE:")
    print(f"Predictions have an average absolute error (MAE) of {mae:.3f}.")
    print(f"R²={r2:.3f}; closer to 1 is generally better.")
    print("CONCLUSION:")
    if r2 >= 0.60:
        print("The model explains a useful portion of target variation on this test split.")
    elif r2 >= 0.20:
        print("The model has limited/moderate explanatory performance; compare alternatives.")
    else:
        print("The model has weak performance on this split; inspect features and assumptions.")
    print("Lower MAE/RMSE and higher R² are preferred when comparing on the same split.")
    return pipe, {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


# ============================================================
# PART A — CLASSIFICATION ALGORITHMS
# Run one function call at a time if you want to study separately.
# ============================================================

def logistic_regression():
    # Step 1: Import/choose Logistic Regression.
    # Step 2-3: Split, impute missing values and standardize features.
    # Step 4: Train a classifier.
    # Step 5-6: Predict and evaluate with accuracy, precision, recall and F1.
    return run_classifier(
        "Logistic Regression",
        LogisticRegression(max_iter=2000, random_state=42),
        scale=True
    )


def decision_tree_classifier():
    # Trees split the data using feature rules; scaling is usually unnecessary.
    return run_classifier(
        "Decision Tree Classifier",
        DecisionTreeClassifier(max_depth=5, random_state=42),
        scale=False
    )


def gaussian_naive_bayes():
    # GaussianNB models each feature with a Gaussian distribution per class.
    return run_classifier("Gaussian Naive Bayes", GaussianNB(), scale=False)


def multinomial_naive_bayes():
    # MultinomialNB requires non-negative feature values. The selected dataset
    # contains non-negative measurements, so it can be used for this demonstration.
    return run_classifier("Multinomial Naive Bayes", MultinomialNB(), scale=False)


def bernoulli_naive_bayes():
    # BernoulliNB is designed for binary-valued features; it binarizes numeric
    # features internally at threshold 0 by default, so treat this as a demonstration.
    return run_classifier("Bernoulli Naive Bayes", BernoulliNB(), scale=False)


def knn_classifier():
    # KNN predicts using the nearest training examples; scaling is important.
    return run_classifier(
        "K-Nearest Neighbors",
        KNeighborsClassifier(n_neighbors=5),
        scale=True
    )


def svm_classifier():
    # RBF SVM can learn nonlinear boundaries; scaling is important.
    return run_classifier(
        "Support Vector Machine (SVC)",
        SVC(kernel="rbf", C=1.0, gamma="scale"),
        scale=True
    )


def bagging_classifier():
    # Bagging trains many base estimators on resampled data and combines predictions.
    return run_classifier(
        "Bagging Classifier",
        BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=42),
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        scale=False
    )


def random_forest_classifier():
    # Random Forest averages many decision trees to improve generalization.
    return run_classifier(
        "Random Forest Classifier",
        RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        ),
        scale=False
    )


def adaboost_classifier():
    # AdaBoost increases attention to examples previous weak learners got wrong.
    return run_classifier(
        "AdaBoost Classifier",
        AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42),
        scale=False
    )


def gradient_boosting_classifier():
    # Gradient Boosting builds trees sequentially to reduce previous errors.
    return run_classifier(
        "Gradient Boosting Classifier",
        GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        ),
        scale=False
    )


def stacking_classifier():
    # Stacking combines different base models using a final estimator.
    estimators = [
        ("lr", LogisticRegression(max_iter=2000, random_state=42)),
        ("tree", DecisionTreeClassifier(max_depth=4, random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ]
    model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000),
        cv=5,
        n_jobs=-1
    )
    # Scaling in the outer pipeline helps LR/KNN; tree models do not require it.
    return run_classifier("Stacking Classifier", model, scale=True)


# ============================================================
# PART B — REGRESSION ALGORITHMS
# ============================================================

def linear_regression():
    # Linear Regression fits a linear relationship between features and target.
    return run_regressor("Linear Regression", LinearRegression(), scale=True)


def ridge_regression():
    # Ridge uses L2 regularization to reduce large coefficients.
    return run_regressor("Ridge Regression", Ridge(alpha=1.0), scale=True)


def lasso_regression():
    # Lasso uses L1 regularization and can shrink some coefficients to zero.
    return run_regressor("Lasso Regression", Lasso(alpha=0.1), scale=True)


def decision_tree_regressor():
    # Decision Tree Regression predicts values from learned feature-based regions.
    return run_regressor(
        "Decision Tree Regressor",
        DecisionTreeRegressor(max_depth=4, random_state=42),
        scale=False
    )


def svr_regressor():
    # SVR fits a margin-based regression function; scaling is important.
    return run_regressor("Support Vector Regression (SVR)", SVR(kernel="rbf"), scale=True)


def bagging_regressor():
    # Bagging averages predictions from multiple trees trained on resampled data.
    return run_regressor(
        "Bagging Regressor",
        BaggingRegressor(
            estimator=DecisionTreeRegressor(random_state=42),
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        scale=False
    )


def random_forest_regressor():
    # Random Forest Regression averages many tree predictions.
    return run_regressor(
        "Random Forest Regressor",
        RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        scale=False
    )


def adaboost_regressor():
    # AdaBoost Regressor combines weak regressors sequentially.
    return run_regressor(
        "AdaBoost Regressor",
        AdaBoostRegressor(n_estimators=100, learning_rate=0.5, random_state=42),
        scale=False
    )


def gradient_boosting_regressor():
    # Gradient Boosting Regressor fits sequential trees to reduce residual errors.
    return run_regressor(
        "Gradient Boosting Regressor",
        GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=42),
        scale=False
    )


def stacking_regressor():
    # Stacking combines different regressors with a final regression model.
    estimators = [
        ("linear", LinearRegression()),
        ("tree", DecisionTreeRegressor(max_depth=4, random_state=42)),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ]
    model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1
    )
    return run_regressor("Stacking Regressor", model, scale=True)


# ============================================================
# PART C — XGBOOST (OPTIONAL)
# ============================================================

def xgboost_classifier():
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("XGBoost is not installed. Run: pip install xgboost")
        return None
    return run_classifier(
        "XGBoost Classifier",
        XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=42
        ),
        scale=False
    )


def xgboost_regressor():
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("XGBoost is not installed. Run: pip install xgboost")
        return None
    return run_regressor(
        "XGBoost Regressor",
        XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42
        ),
        scale=False
    )


# ============================================================
# PART D — UNSUPERVISED LEARNING: K-MEANS
# ============================================================

def kmeans_clustering():
    print("\n" + "=" * 72)
    print("ALGORITHM: K-Means Clustering")
    print("=" * 72)

    # Step 1: Load feature data. We intentionally do not use target labels.
    data = load_iris(as_frame=True)
    X = data.data
    print("Step 1: Iris feature dataset loaded; labels are not used for fitting.")

    # Step 2: Impute and scale because K-Means uses distances.
    prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    X_scaled = prep.fit_transform(X)
    print("Step 2: Missing values handled and features standardized.")

    # Step 3: Use the elbow method to compare candidate cluster counts.
    k_values = range(2, 7)
    inertias = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    plt.figure()
    plt.plot(list(k_values), inertias, marker="o")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia / WCSS")
    plt.title("K-Means Elbow Method")
    plt.tight_layout()
    plt.show()
    print("Step 3: Elbow plot displayed; choose K near the bend, not automatically.")

    # Step 4: Fit the selected K. K=3 is used here for demonstration.
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    print("Step 4: K-Means model fitted with K=3.")

    # Step 5: Evaluate clustering with silhouette score.
    sil = silhouette_score(X_scaled, labels)
    print(f"Silhouette score: {sil:.3f} (closer to 1 generally means better-separated clusters).")
    print("Cluster counts:\n", pd.Series(labels).value_counts().sort_index())

    # Step 6: Visualize clusters after projecting to two principal components.
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    plt.figure()
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("K-Means Clusters Visualized with PCA")
    plt.tight_layout()
    plt.show()

    # Inference and conclusion.
    print("INFERENCE:")
    print(f"K-Means assigned all {len(labels)} samples to one of 3 clusters.")
    print(f"The silhouette score is {sil:.3f}.")
    print("CONCLUSION:")
    if sil >= 0.50:
        print("Clusters are reasonably well separated for this representation.")
    elif sil >= 0.25:
        print("Some cluster structure exists, but clusters may overlap.")
    else:
        print("Clusters overlap considerably; try another K or inspect features.")
    return kmeans, labels


# ============================================================
# PART E — PCA DIMENSIONALITY REDUCTION
# ============================================================

def pca_dimensionality_reduction():
    print("\n" + "=" * 72)
    print("ALGORITHM: Principal Component Analysis (PCA)")
    print("=" * 72)

    # Step 1: Load feature data only; PCA does not need target labels.
    data = load_iris(as_frame=True)
    X = data.data
    print(f"Step 1: Loaded {X.shape[0]} rows and {X.shape[1]} original features.")

    # Step 2: Impute and scale before PCA.
    prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    X_scaled = prep.fit_transform(X)
    print("Step 2: Missing values handled and features standardized.")

    # Step 3: Reduce to two principal components.
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print("Step 3: PCA fitted and data transformed to 2 components.")

    # Step 4: Report explained variance.
    ratios = pca.explained_variance_ratio_
    total = ratios.sum()
    print("Explained variance ratio:", np.round(ratios, 4))
    print(f"Variance retained by 2 components: {total * 100:.2f}%")

    # Step 5: Visualize the lower-dimensional representation.
    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA: Iris Data Reduced to 2 Components")
    plt.tight_layout()
    plt.show()

    # Inference and conclusion.
    print("INFERENCE:")
    print(f"Data was reduced from {X.shape[1]} features to 2 components.")
    print(f"The two components retain {total * 100:.2f}% of the variance.")
    print("CONCLUSION:")
    if total >= 0.80:
        print("Two components retain most of the variation, with dimensionality reduced.")
    else:
        print("Some information is lost; use more components if higher variance retention is needed.")
    return pca, X_pca


# ============================================================
# RUN ONE ALGORITHM AT A TIME
# Uncomment exactly ONE line below, then run this file.
# Example: logistic_regression()
# ============================================================

if __name__ == "__main__":
    # ---- Classification: uncomment ONE ----
    # logistic_regression()
    # decision_tree_classifier()
    # gaussian_naive_bayes()
    # multinomial_naive_bayes()
    # bernoulli_naive_bayes()
    # knn_classifier()
    # svm_classifier()
    # bagging_classifier()
    # random_forest_classifier()
    # adaboost_classifier()
    # gradient_boosting_classifier()
    # stacking_classifier()
    # xgboost_classifier()

    # ---- Regression: uncomment ONE ----
    # linear_regression()
    # ridge_regression()
    # lasso_regression()
    # decision_tree_regressor()
    # svr_regressor()
    # bagging_regressor()
    # random_forest_regressor()
    # adaboost_regressor()
    # gradient_boosting_regressor()
    # stacking_regressor()
    # xgboost_regressor()

    # ---- Unsupervised learning: uncomment ONE ----
    # kmeans_clustering()
    # pca_dimensionality_reduction()

    print("Choose one algorithm by uncommenting its function call at the bottom.")
