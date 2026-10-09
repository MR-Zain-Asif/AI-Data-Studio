"""ML Model Builder — Advanced ML with Polynomial, Overfitting Detection, CV, Comparison, Scaling, Learning Curve, Prediction."""

import time
import numpy as np
import pandas as pd
from typing import Any, Optional


class MLModeler:
    """Handles auto-detection, training, overfitting detection, comparison, and prediction."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.model = None
        self.model_type = None  # 'regression' or 'classification'
        self.model_name = None
        self.target_column = None
        self.feature_columns = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.training_time = 0.0
        self.scores = {}
        self.feature_importances = {}
        self.poly_degree = 1
        self.scaler_type = None  # None, 'standard', 'minmax'
        self.scaler = None
        self._poly_scaler = None  # auto-scaler applied before polynomial features
        self.poly_transformer = None
        self.overfitting_info = {}
        self.cv_scores = {}
        self.train_score = None
        self.test_score = None
        self._target_encoder = None
        self._prepared_X = None  # raw X before poly/scaling for prediction
        self._feature_medians = {}

    # ── Auto-detect target column and model type ─────────────────
    def detect(self, target_col: Optional[str] = None) -> dict[str, Any]:
        """Detect suitable target column and model type from the dataset."""
        numeric_cols = []
        categorical_cols = []

        for col in self.df.columns:
            series = self.df[col].dropna()
            if series.empty:
                continue
            if pd.api.types.is_numeric_dtype(series):
                numeric_cols.append(col)
            elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
                unique_count = series.nunique()
                if 2 <= unique_count <= 20:
                    categorical_cols.append(col)

        suggestions = []

        if target_col and target_col in self.df.columns:
            series = self.df[target_col].dropna()
            if pd.api.types.is_numeric_dtype(series):
                unique_count = series.nunique()
                if unique_count <= 10 and unique_count >= 2:
                    suggestions.append({
                        "target": target_col,
                        "model_type": "classification",
                        "model_name": "Logistic Regression",
                        "reason": f"'{target_col}' has {unique_count} unique numeric categories",
                        "confidence": "high"
                    })
                else:
                    suggestions.append({
                        "target": target_col,
                        "model_type": "regression",
                        "model_name": "Linear Regression",
                        "reason": f"'{target_col}' is continuous numeric ({unique_count} unique values)",
                        "confidence": "high"
                    })
            else:
                unique_count = series.nunique()
                if 2 <= unique_count <= 20:
                    suggestions.append({
                        "target": target_col,
                        "model_type": "classification",
                        "model_name": "Logistic Regression",
                        "reason": f"'{target_col}' is categorical with {unique_count} classes",
                        "confidence": "high"
                    })

            return {
                "suggestions": suggestions,
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "total_rows": len(self.df),
                "total_columns": len(self.df.columns),
            }

        # Auto-detect heuristics
        target_hints = ['target', 'label', 'class', 'output', 'result', 'price', 'salary',
                        'income', 'score', 'rating', 'prediction', 'y', 'status', 'category',
                        'survived', 'approved', 'churn']

        for col in self.df.columns:
            col_lower = col.strip().lower()
            for hint in target_hints:
                if hint in col_lower:
                    series = self.df[col].dropna()
                    if pd.api.types.is_numeric_dtype(series):
                        unique_count = series.nunique()
                        if unique_count <= 10 and unique_count >= 2:
                            suggestions.append({
                                "target": col, "model_type": "classification",
                                "model_name": "Logistic Regression",
                                "reason": f"'{col}' matches target pattern and has {unique_count} categories",
                                "confidence": "high"
                            })
                        else:
                            suggestions.append({
                                "target": col, "model_type": "regression",
                                "model_name": "Linear Regression",
                                "reason": f"'{col}' matches target pattern and is continuous numeric",
                                "confidence": "high"
                            })
                    elif col in categorical_cols:
                        suggestions.append({
                            "target": col, "model_type": "classification",
                            "model_name": "Logistic Regression",
                            "reason": f"'{col}' matches target pattern and is categorical",
                            "confidence": "high"
                        })
                    break

        if not suggestions:
            last_col = self.df.columns[-1]
            series = self.df[last_col].dropna()
            if pd.api.types.is_numeric_dtype(series):
                unique_count = series.nunique()
                if unique_count <= 10 and unique_count >= 2:
                    suggestions.append({
                        "target": last_col, "model_type": "classification",
                        "model_name": "Logistic Regression",
                        "reason": f"Last column '{last_col}' has {unique_count} categories",
                        "confidence": "medium"
                    })
                elif unique_count > 10:
                    suggestions.append({
                        "target": last_col, "model_type": "regression",
                        "model_name": "Linear Regression",
                        "reason": f"Last column '{last_col}' is continuous numeric",
                        "confidence": "medium"
                    })

        for col in categorical_cols:
            if not any(s["target"] == col for s in suggestions):
                suggestions.append({
                    "target": col, "model_type": "classification",
                    "model_name": "Logistic Regression",
                    "reason": f"'{col}' is categorical with {self.df[col].dropna().nunique()} classes",
                    "confidence": "low"
                })

        for col in numeric_cols:
            if not any(s["target"] == col for s in suggestions) and self.df[col].dropna().nunique() > 10:
                suggestions.append({
                    "target": col, "model_type": "regression",
                    "model_name": "Linear Regression",
                    "reason": f"'{col}' is continuous numeric",
                    "confidence": "low"
                })

        confidence_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda s: confidence_order.get(s["confidence"], 3))
        suggestions = suggestions[:5]

        return {
            "suggestions": suggestions,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
        }

    # ── Prepare data ─────────────────────────────────────────────
    def _prepare_data(self, target_col: str) -> tuple:
        """Prepare features and target with encoding and cleaning."""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        df = self.df.copy()
        df = df.dropna(subset=[target_col])

        if len(df) < 4:
            raise ValueError(f"Not enough valid data rows ({len(df)} after removing missing target values). Need at least 4 rows.")

        y = df[target_col]
        X = df.drop(columns=[target_col])

        # Remove useless columns
        cols_to_drop = []
        for col in X.columns:
            series = X[col].dropna()
            if series.empty or series.nunique() <= 1:
                cols_to_drop.append(col)
                continue
            if not pd.api.types.is_numeric_dtype(series):
                if series.nunique() > 50 or series.nunique() / len(series) > 0.5:
                    cols_to_drop.append(col)
        X = X.drop(columns=cols_to_drop)

        if X.empty or len(X.columns) == 0:
            raise ValueError("No suitable feature columns found after preprocessing.")

        # Encode all non-numeric columns
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                le = LabelEncoder()
                X[col] = X[col].fillna('__MISSING__')
                X[col] = le.fit_transform(X[col].astype(str))

        # Fill numeric NaN with median
        for col in X.columns:
            if X[col].isna().any():
                median_val = X[col].median()
                self._feature_medians[col] = float(median_val) if pd.notna(median_val) else 0
                X[col] = X[col].fillna(self._feature_medians[col])
            else:
                self._feature_medians[col] = float(X[col].median()) if not X[col].empty else 0

        # Encode target if non-numeric
        target_encoder = None
        if not pd.api.types.is_numeric_dtype(y):
            target_encoder = LabelEncoder()
            y = pd.Series(target_encoder.fit_transform(y.astype(str)), index=y.index)
        else:
            y = pd.to_numeric(y, errors='coerce')
            if y.isna().any():
                med_y = float(y.median()) if not y.dropna().empty else 0.0
                y = y.fillna(med_y)
        self._target_encoder = target_encoder

        X = X.select_dtypes(include=[np.number])
        if X.empty:
            raise ValueError("No numeric features available after encoding.")

        self.feature_columns = X.columns.tolist()

        # Dynamic test split for small datasets
        if len(df) <= 8:
            test_size = 1  # 1 test sample for very small dataset
        elif len(df) < 30:
            test_size = 0.25
        else:
            test_size = 0.2

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        return X_train, X_test, y_train, y_test, target_encoder


    # ── Apply polynomial features ────────────────────────────────
    def _apply_polynomial(self, X_train, X_test, degree):
        """Apply polynomial feature transformation with auto-scaling for stability."""
        if degree <= 1:
            return X_train, X_test

        from sklearn.preprocessing import PolynomialFeatures, StandardScaler

        # Limit features to avoid explosion (max 10 input features for poly)
        max_features_for_poly = 10
        if X_train.shape[1] > max_features_for_poly:
            # Use first N features (consistent order)
            cols = X_train.columns[:max_features_for_poly].tolist()
            X_train = X_train[cols]
            X_test = X_test[cols]
            self.feature_columns = cols

        # Auto-scale before polynomial to prevent numerical instability
        # (large values squared/cubed cause ill-conditioned matrices)
        if not hasattr(self, '_poly_scaler') or self._poly_scaler is None:
            self._poly_scaler = StandardScaler()
            cols = X_train.columns
            idx_train, idx_test = X_train.index, X_test.index
            X_train = pd.DataFrame(self._poly_scaler.fit_transform(X_train), columns=cols, index=idx_train)
            X_test = pd.DataFrame(self._poly_scaler.transform(X_test), columns=cols, index=idx_test)

        self.poly_transformer = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)

        X_train_poly = self.poly_transformer.fit_transform(X_train)
        X_test_poly = self.poly_transformer.transform(X_test)

        feature_names = self.poly_transformer.get_feature_names_out(self.feature_columns)
        X_train_poly = pd.DataFrame(X_train_poly, columns=feature_names, index=X_train.index)
        X_test_poly = pd.DataFrame(X_test_poly, columns=feature_names, index=X_test.index)

        return X_train_poly, X_test_poly

    # ── Apply scaling ────────────────────────────────────────────
    def _apply_scaling(self, X_train, X_test, scaler_type):
        """Apply feature scaling."""
        if not scaler_type or scaler_type == 'none':
            return X_train, X_test

        if scaler_type == 'standard':
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        else:
            return X_train, X_test

        cols = X_train.columns
        X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train), columns=cols, index=X_train.index)
        X_test_scaled = pd.DataFrame(self.scaler.transform(X_test), columns=cols, index=X_test.index)

        return X_train_scaled, X_test_scaled

    def _detect_target_type(self, col: str) -> str:
        """Robustly detect if target is regression or classification."""
        if col not in self.df.columns:
            return "regression"
        series = self.df[col].dropna()
        if series.empty:
            return "regression"
        if not pd.api.types.is_numeric_dtype(series):
            return "classification"

        if pd.api.types.is_float_dtype(series):
            if not np.all(series % 1 == 0):
                return "regression"

        unique_cnt = series.nunique()
        total_cnt = len(series)

        if unique_cnt <= 5 and (unique_cnt / total_cnt) <= 0.5:
            return "classification"

        return "regression"

    # ── Train Single Model ─────────────────────────────────────────
    def train(self, target_col: str, model_type: Optional[str] = None,
              degree: int = 1, scaler_type: Optional[str] = None,
              algorithm: Optional[str] = None, C: float = 1.0,
              alpha: float = 1.0, max_depth: int = 10,
              n_estimators: int = 100, n_neighbors: int = 5) -> dict[str, Any]:
        """Full pipeline: prepare -> poly -> scale -> train -> evaluate -> return JSON dict."""
        self.target_column = target_col
        self.poly_degree = degree
        self.scaler_type = scaler_type
        self.algorithm = algorithm
        self.C = C
        self.alpha = alpha
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.n_neighbors = n_neighbors

        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")

        # Auto-detect model type
        if not model_type or model_type == "auto":
            model_type = self._detect_target_type(target_col)
        self.model_type = model_type

        # Prepare data
        X_train, X_test, y_train, y_test, target_encoder = self._prepare_data(target_col)
        self._prepared_X = X_train.copy()

        # Apply polynomial features
        X_train, X_test = self._apply_polynomial(X_train, X_test, degree)

        # Apply scaling
        X_train, X_test = self._apply_scaling(X_train, X_test, scaler_type)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # Train with fallback from classification to regression if target is continuous
        start_time = time.time()
        try:
            if self.model_type == "regression":
                self._train_regression(X_train, y_train, algorithm=algorithm, alpha=alpha, max_depth=max_depth, n_estimators=n_estimators, n_neighbors=n_neighbors)
            else:
                self._train_classification(X_train, y_train, algorithm=algorithm, C=C, max_depth=max_depth, n_estimators=n_estimators, n_neighbors=n_neighbors)
        except ValueError as ve:
            if "Unknown label type" in str(ve) or "continuous" in str(ve):
                # Fallback to regression
                self.model_type = "regression"
                self._train_regression(X_train, y_train, algorithm=algorithm, alpha=alpha, max_depth=max_depth, n_estimators=n_estimators, n_neighbors=n_neighbors)
            else:
                raise

        self.training_time = round(time.time() - start_time, 3)

        # Predict & score
        self.y_pred = self.model.predict(X_test)
        y_train_pred = self.model.predict(X_train)

        if self.model_type == "regression":
            self.scores = self._regression_scores(y_test, self.y_pred)
            self.train_score = self._regression_scores(y_train, y_train_pred)
        else:
            self.scores = self._classification_scores(y_test, self.y_pred, target_encoder)
            self.train_score = self._classification_scores(y_train, y_train_pred, target_encoder)

        # Overfitting detection
        self.overfitting_info = self._detect_overfitting()

        # Cross-validation
        self.cv_scores = self._cross_validate(X_train, y_train)

        # Feature importances
        self.feature_importances = self._get_feature_importances()

        return self._build_results(target_encoder)

    def _train_regression(self, X_train, y_train, algorithm: Optional[str] = None,
                          alpha: float = 1.0, max_depth: int = 10,
                          n_estimators: int = 100, n_neighbors: int = 5):
        alg = (algorithm or "").lower().strip()
        if alg in ("rf", "random_forest", "random forest", "random forest regressor"):
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            self.model_name = f"Random Forest Regressor (n_estimators={n_estimators}, max_depth={max_depth})"
        elif alg in ("gb", "gradient_boosting", "gradient boosting", "gradient boosting regressor"):
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            self.model_name = f"Gradient Boosting Regressor (n_estimators={n_estimators})"
        elif alg in ("et", "extra_trees", "extra trees", "extra trees regressor"):
            from sklearn.ensemble import ExtraTreesRegressor
            self.model = ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            self.model_name = f"Extra Trees Regressor (n_estimators={n_estimators})"
        elif alg in ("ada", "adaboost", "adaboost regressor"):
            from sklearn.ensemble import AdaBoostRegressor
            self.model = AdaBoostRegressor(n_estimators=n_estimators, random_state=42)
            self.model_name = "AdaBoost Regressor"
        elif alg in ("dt", "decision_tree", "decision tree", "decision tree regressor"):
            from sklearn.tree import DecisionTreeRegressor
            self.model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
            self.model_name = f"Decision Tree Regressor (max_depth={max_depth})"
        elif alg in ("ridge", "ridge regression"):
            from sklearn.linear_model import Ridge
            self.model = Ridge(alpha=alpha, random_state=42)
            self.model_name = f"Ridge Regression (alpha={alpha})"
        elif alg in ("lasso", "lasso regression"):
            from sklearn.linear_model import Lasso
            self.model = Lasso(alpha=alpha, max_iter=2000, random_state=42)
            self.model_name = f"Lasso Regression (alpha={alpha})"
        elif alg in ("elasticnet", "elastic_net", "elastic net"):
            from sklearn.linear_model import ElasticNet
            self.model = ElasticNet(alpha=alpha, random_state=42)
            self.model_name = f"ElasticNet Regression (alpha={alpha})"
        elif alg in ("svm", "svr", "support vector regressor"):
            from sklearn.svm import SVR
            self.model = SVR()
            self.model_name = "Support Vector Regression (SVR)"
        elif alg in ("linear_svr", "linear svr"):
            from sklearn.svm import LinearSVR
            self.model = LinearSVR(dual="auto", max_iter=2000, random_state=42)
            self.model_name = "Linear Support Vector Regression"
        elif alg in ("knn", "kneighbors", "k-nearest neighbors"):
            from sklearn.neighbors import KNeighborsRegressor
            self.model = KNeighborsRegressor(n_neighbors=n_neighbors)
            self.model_name = f"K-Nearest Neighbors Regressor (k={n_neighbors})"
        elif alg in ("mlp", "neural_network", "neural_net", "mlp regressor"):
            from sklearn.neural_network import MLPRegressor
            self.model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
            self.model_name = "Multi-layer Perceptron (Neural Network)"
        elif alg in ("huber", "huber regressor"):
            from sklearn.linear_model import HuberRegressor
            self.model = HuberRegressor(max_iter=1000)
            self.model_name = "Huber Regressor"
        elif alg in ("bayesian_ridge", "bayesian ridge"):
            from sklearn.linear_model import BayesianRidge
            self.model = BayesianRidge()
            self.model_name = "Bayesian Ridge Regression"
        elif alg in ("sgd", "sgd_regressor", "sgd regressor"):
            from sklearn.linear_model import SGDRegressor
            self.model = SGDRegressor(random_state=42)
            self.model_name = "SGD Regressor"
        elif alg in ("xgboost", "xgb"):
            try:
                from xgboost import XGBRegressor
                self.model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, verbosity=0)
                self.model_name = "XGBoost Regressor"
            except Exception:
                from sklearn.ensemble import GradientBoostingRegressor
                self.model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                self.model_name = "Gradient Boosting Regressor (Fallback for XGBoost)"
        elif alg in ("lightgbm", "lgb"):
            try:
                from lightgbm import LGBMRegressor
                self.model = LGBMRegressor(n_estimators=n_estimators, random_state=42, verbose=-1)
                self.model_name = "LightGBM Regressor"
            except Exception:
                from sklearn.ensemble import GradientBoostingRegressor
                self.model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                self.model_name = "Gradient Boosting Regressor (Fallback for LightGBM)"
        elif alg in ("catboost", "cat"):
            try:
                from catboost import CatBoostRegressor
                self.model = CatBoostRegressor(iterations=n_estimators, verbose=0, random_state=42)
                self.model_name = "CatBoost Regressor"
            except Exception:
                from sklearn.ensemble import GradientBoostingRegressor
                self.model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                self.model_name = "Gradient Boosting Regressor (Fallback for CatBoost)"
        else:
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression()
            self.model_name = f"{'Polynomial' if self.poly_degree > 1 else 'Linear'} Regression (degree={self.poly_degree})" if self.poly_degree > 1 else "Linear Regression"
            
        self.model.fit(X_train, y_train)

    def _train_classification(self, X_train, y_train, algorithm: Optional[str] = None,
                              C: float = 1.0, max_depth: int = 10,
                              n_estimators: int = 100, n_neighbors: int = 5):
        n_classes = y_train.nunique()
        alg = (algorithm or "").lower().strip()
        if alg in ("rf", "random_forest", "random forest", "random forest classifier"):
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            self.model_name = f"Random Forest Classifier (n_estimators={n_estimators}, max_depth={max_depth})"
        elif alg in ("gb", "gradient_boosting", "gradient boosting", "gradient boosting classifier"):
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            self.model_name = f"Gradient Boosting Classifier (n_estimators={n_estimators})"
        elif alg in ("et", "extra_trees", "extra trees", "extra trees classifier"):
            from sklearn.ensemble import ExtraTreesClassifier
            self.model = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            self.model_name = f"Extra Trees Classifier (n_estimators={n_estimators})"
        elif alg in ("ada", "adaboost", "adaboost classifier"):
            from sklearn.ensemble import AdaBoostClassifier
            self.model = AdaBoostClassifier(n_estimators=n_estimators, random_state=42)
            self.model_name = "AdaBoost Classifier"
        elif alg in ("dt", "decision_tree", "decision tree", "decision tree classifier"):
            from sklearn.tree import DecisionTreeClassifier
            self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            self.model_name = f"Decision Tree Classifier (max_depth={max_depth})"
        elif alg in ("svm", "svc", "support vector classifier"):
            from sklearn.svm import SVC
            self.model = SVC(C=C, probability=True, random_state=42)
            self.model_name = f"Support Vector Classifier (C={C})"
        elif alg in ("linear_svc", "linear svc"):
            from sklearn.svm import LinearSVC
            self.model = LinearSVC(C=C, dual="auto", max_iter=2000, random_state=42)
            self.model_name = f"Linear Support Vector Classifier (C={C})"
        elif alg in ("knn", "kneighbors", "k-nearest neighbors"):
            from sklearn.neighbors import KNeighborsClassifier
            self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
            self.model_name = f"K-Nearest Neighbors Classifier (k={n_neighbors})"
        elif alg in ("naive_bayes", "gnb", "gaussian_nb", "gaussian naive bayes"):
            from sklearn.naive_bayes import GaussianNB
            self.model = GaussianNB()
            self.model_name = "Gaussian Naive Bayes"
        elif alg in ("bernoulli_nb", "bnb", "bernoulli naive bayes"):
            from sklearn.naive_bayes import BernoulliNB
            self.model = BernoulliNB()
            self.model_name = "Bernoulli Naive Bayes"
        elif alg in ("mlp", "neural_network", "neural_net", "mlp classifier"):
            from sklearn.neural_network import MLPClassifier
            self.model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
            self.model_name = "Multi-layer Perceptron (Neural Network)"
        elif alg in ("ridge_classifier", "ridge classifier", "ridge"):
            from sklearn.linear_model import RidgeClassifier
            self.model = RidgeClassifier(alpha=C, random_state=42)
            self.model_name = "Ridge Classifier"
        elif alg in ("sgd", "sgd_classifier", "sgd classifier"):
            from sklearn.linear_model import SGDClassifier
            self.model = SGDClassifier(random_state=42)
            self.model_name = "SGD Classifier"
        elif alg in ("xgboost", "xgb"):
            try:
                from xgboost import XGBClassifier
                self.model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, eval_metric="logloss", verbosity=0)
                self.model_name = "XGBoost Classifier"
            except Exception:
                from sklearn.ensemble import GradientBoostingClassifier
                self.model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                self.model_name = "Gradient Boosting Classifier (Fallback for XGBoost)"
        elif alg in ("lightgbm", "lgb"):
            try:
                from lightgbm import LGBMClassifier
                self.model = LGBMClassifier(n_estimators=n_estimators, random_state=42, verbose=-1)
                self.model_name = "LightGBM Classifier"
            except Exception:
                from sklearn.ensemble import GradientBoostingClassifier
                self.model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                self.model_name = "Gradient Boosting Classifier (Fallback for LightGBM)"
        elif alg in ("catboost", "cat"):
            try:
                from catboost import CatBoostClassifier
                self.model = CatBoostClassifier(iterations=n_estimators, verbose=0, random_state=42)
                self.model_name = "CatBoost Classifier"
            except Exception:
                from sklearn.ensemble import GradientBoostingClassifier
                self.model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                self.model_name = "Gradient Boosting Classifier (Fallback for CatBoost)"
        else:
            from sklearn.linear_model import LogisticRegression
            multi_class = 'multinomial' if n_classes > 2 else 'auto'
            self.model = LogisticRegression(C=C, max_iter=1000, random_state=42, multi_class=multi_class, solver='lbfgs')
            self.model_name = f"Logistic Regression (C={C})" if C != 1.0 else "Logistic Regression"
            
        self.model.fit(X_train, y_train)

    # ── Overfitting detection ────────────────────────────────────
    def _detect_overfitting(self) -> dict[str, Any]:
        """Detect overfitting by comparing train vs test scores."""
        if self.model_type == "regression":
            train_r2 = self.train_score.get("r2_score", 0)
            test_r2 = self.scores.get("r2_score", 0)
            gap = train_r2 - test_r2

            if test_r2 < 0:
                return {
                    "status": "critical",
                    "level": "danger",
                    "title": "Model Useless!",
                    "message": f"Test R² = {test_r2:.4f} (negative). Model predictions worse than mean. Features ya target column change karein.",
                    "suggestion": "Different target column try karein ya features badlein",
                    "train_score": round(train_r2, 4),
                    "test_score": round(test_r2, 4),
                    "gap": round(gap, 4),
                }
            elif train_r2 > 0.95 and test_r2 < 0.7:
                return {
                    "status": "overfitting",
                    "level": "danger",
                    "title": "Overfitting Detected!",
                    "message": f"Train R² = {train_r2:.4f} but Test R² = {test_r2:.4f}. Model ne training data memorize kar li hai.",
                    "suggestion": "Polynomial degree kam karein, ya zyada data collect karein",
                    "train_score": round(train_r2, 4),
                    "test_score": round(test_r2, 4),
                    "gap": round(gap, 4),
                }
            elif gap > 0.2:
                return {
                    "status": "possible_overfitting",
                    "level": "warning",
                    "title": "Possible Overfitting",
                    "message": f"Train R² ({train_r2:.4f}) aur Test R² ({test_r2:.4f}) mein {gap:.2f} ka gap hai.",
                    "suggestion": "Polynomial degree kam karein ya regularization try karein",
                    "train_score": round(train_r2, 4),
                    "test_score": round(test_r2, 4),
                    "gap": round(gap, 4),
                }
            elif train_r2 < 0.5 and test_r2 < 0.5:
                return {
                    "status": "underfitting",
                    "level": "warning",
                    "title": "Underfitting Detected",
                    "message": f"Train R² = {train_r2:.4f}, Test R² = {test_r2:.4f}. Model data ko properly capture nahi kar raha.",
                    "suggestion": "Polynomial degree badhaein ya zyada features add karein",
                    "train_score": round(train_r2, 4),
                    "test_score": round(test_r2, 4),
                    "gap": round(gap, 4),
                }
            else:
                return {
                    "status": "good_fit",
                    "level": "success",
                    "title": "Good Fit!",
                    "message": f"Train R² = {train_r2:.4f}, Test R² = {test_r2:.4f}. Model stable hai!",
                    "suggestion": "",
                    "train_score": round(train_r2, 4),
                    "test_score": round(test_r2, 4),
                    "gap": round(gap, 4),
                }
        else:
            # Classification
            train_acc = self.train_score.get("accuracy", 0)
            test_acc = self.scores.get("accuracy", 0)
            gap = train_acc - test_acc

            if train_acc > 0.95 and test_acc < 0.7:
                return {
                    "status": "overfitting",
                    "level": "danger",
                    "title": "Overfitting Detected!",
                    "message": f"Train Accuracy = {train_acc:.2%} but Test Accuracy = {test_acc:.2%}.",
                    "suggestion": "Features kam karein ya regularization badhaein",
                    "train_score": round(train_acc, 4),
                    "test_score": round(test_acc, 4),
                    "gap": round(gap, 4),
                }
            elif gap > 0.15:
                return {
                    "status": "possible_overfitting",
                    "level": "warning",
                    "title": "Possible Overfitting",
                    "message": f"Train Accuracy ({train_acc:.2%}) aur Test Accuracy ({test_acc:.2%}) mein {gap:.2%} ka gap hai.",
                    "suggestion": "Model simplify karein",
                    "train_score": round(train_acc, 4),
                    "test_score": round(test_acc, 4),
                    "gap": round(gap, 4),
                }
            elif train_acc < 0.5 and test_acc < 0.5:
                return {
                    "status": "underfitting",
                    "level": "warning",
                    "title": "Underfitting Detected",
                    "message": f"Train = {train_acc:.2%}, Test = {test_acc:.2%}. Model properly learn nahi kar raha.",
                    "suggestion": "Zyada features add karein ya different model try karein",
                    "train_score": round(train_acc, 4),
                    "test_score": round(test_acc, 4),
                    "gap": round(gap, 4),
                }
            else:
                return {
                    "status": "good_fit",
                    "level": "success",
                    "title": "Good Fit!",
                    "message": f"Train = {train_acc:.2%}, Test = {test_acc:.2%}. Model stable hai!",
                    "suggestion": "",
                    "train_score": round(train_acc, 4),
                    "test_score": round(test_acc, 4),
                    "gap": round(gap, 4),
                }

    # ── Cross-validation ─────────────────────────────────────────
    def _cross_validate(self, X, y) -> dict:
        """Run 5-fold cross-validation."""
        from sklearn.model_selection import cross_val_score

        scoring = 'r2' if self.model_type == 'regression' else 'accuracy'
        try:
            n_splits = min(5, len(X))
            if n_splits < 2:
                return {"mean": 0, "std": 0, "scores": [], "stable": False}

            cv_scores = cross_val_score(self.model, X, y, cv=n_splits, scoring=scoring)
            mean_score = float(np.mean(cv_scores))
            std_score = float(np.std(cv_scores))

            return {
                "mean": round(mean_score, 4),
                "std": round(std_score, 4),
                "scores": [round(float(s), 4) for s in cv_scores],
                "stable": std_score < 0.1,
                "mean_percentage": round(mean_score * 100, 2),
            }
        except Exception:
            return {"mean": 0, "std": 0, "scores": [], "stable": False}

    # ── Score calculations ───────────────────────────────────────
    def _regression_scores(self, y_true, y_pred) -> dict:
        import math
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        try:
            r2 = float(r2_score(y_true, y_pred))
            if math.isnan(r2) or math.isinf(r2):
                r2 = 0.0
        except Exception:
            r2 = 0.0
        try:
            mae = float(mean_absolute_error(y_true, y_pred))
            if math.isnan(mae) or math.isinf(mae):
                mae = 0.0
        except Exception:
            mae = 0.0
        try:
            mse = float(mean_squared_error(y_true, y_pred))
            if math.isnan(mse) or math.isinf(mse):
                mse = 0.0
        except Exception:
            mse = 0.0
        rmse = float(math.sqrt(mse)) if mse >= 0 else 0.0
        return {
            "r2_score": round(r2, 4),
            "r2": round(r2, 4),
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "r2_percentage": round(r2 * 100, 2),
        }

    def _classification_scores(self, y_true, y_pred, target_encoder=None) -> dict:
        import math
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        n_classes = len(set(y_true) | set(y_pred))
        average = 'binary' if n_classes == 2 else 'weighted'
        try:
            accuracy = float(accuracy_score(y_true, y_pred))
            if math.isnan(accuracy) or math.isinf(accuracy):
                accuracy = 0.0
        except Exception:
            accuracy = 0.0
        try:
            precision = float(precision_score(y_true, y_pred, average=average, zero_division=0))
            if math.isnan(precision) or math.isinf(precision):
                precision = 0.0
        except Exception:
            precision = 0.0
        try:
            recall = float(recall_score(y_true, y_pred, average=average, zero_division=0))
            if math.isnan(recall) or math.isinf(recall):
                recall = 0.0
        except Exception:
            recall = 0.0
        try:
            f1 = float(f1_score(y_true, y_pred, average=average, zero_division=0))
            if math.isnan(f1) or math.isinf(f1):
                f1 = 0.0
        except Exception:
            f1 = 0.0
        try:
            cm = confusion_matrix(y_true, y_pred).tolist()
        except Exception:
            cm = []
        class_labels = sorted(list(set(y_true) | set(y_pred)))
        if target_encoder:
            try:
                class_labels_str = target_encoder.inverse_transform(class_labels).tolist()
            except Exception:
                class_labels_str = [str(c) for c in class_labels]
        else:
            class_labels_str = [str(c) for c in class_labels]
        return {
            "accuracy": round(accuracy, 4),
            "accuracy_percentage": round(accuracy * 100, 2),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "f1": round(f1, 4),
            "confusion_matrix": cm,
            "class_labels": class_labels_str,
        }

    def _get_feature_importances(self) -> dict:
        importances = {}
        feature_names = self.X_train.columns.tolist() if hasattr(self.X_train, 'columns') else self.feature_columns

        if hasattr(self.model, "feature_importances_"):
            fi = self.model.feature_importances_
            if isinstance(fi, np.ndarray):
                for i, col in enumerate(feature_names):
                    if i < len(fi):
                        importances[col] = round(float(fi[i]), 4)
        elif hasattr(self.model, "coef_"):
            coeffs = self.model.coef_
            if isinstance(coeffs, np.ndarray):
                if coeffs.ndim == 1:
                    for i, col in enumerate(feature_names):
                        if i < len(coeffs):
                            importances[col] = round(float(abs(coeffs[i])), 4)
                else:
                    avg_coeffs = np.mean(np.abs(coeffs), axis=0)
                    for i, col in enumerate(feature_names):
                        if i < len(avg_coeffs):
                            importances[col] = round(float(avg_coeffs[i]), 4)

        if importances:
            max_imp = max(importances.values()) if importances.values() else 1
            if max_imp > 0:
                importances = {k: round(v / max_imp * 100, 2) for k, v in importances.items()}

        importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
        return importances

    # ── Build final results ──────────────────────────────────────
    def _build_results(self, target_encoder=None) -> dict[str, Any]:
        result = {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "algorithm": self.algorithm or ("linear_regression" if self.model_type == "regression" else "logistic_regression"),
            "hyperparameters": {
                "C": getattr(self, "C", 1.0),
                "alpha": getattr(self, "alpha", 1.0),
                "max_depth": getattr(self, "max_depth", 10),
                "n_estimators": getattr(self, "n_estimators", 100),
                "n_neighbors": getattr(self, "n_neighbors", 5)
            },
            "target_column": self.target_column,
            "feature_columns": self.feature_columns,
            "num_features": len(self.feature_columns),
            "poly_degree": self.poly_degree,
            "scaler_type": self.scaler_type or "none",
            "train_size": len(self.X_train),
            "test_size": len(self.X_test),
            "training_time_seconds": self.training_time,
            "scores": self.scores,
            "train_scores": self.train_score,
            "feature_importances": self.feature_importances,
            "overfitting": self.overfitting_info,
            "cv_scores": self.cv_scores,
            "feature_medians": self._feature_medians,
        }

        if self.model_type == "regression":
            y_true_list = self.y_test.values.tolist()
            y_pred_list = self.y_pred.tolist()
            limit = min(200, len(y_true_list))
            result["scatter_data"] = {
                "actual": [round(float(v), 4) for v in y_true_list[:limit]],
                "predicted": [round(float(v), 4) for v in y_pred_list[:limit]],
            }
            residuals = [round(float(a - p), 4) for a, p in zip(y_true_list[:limit], y_pred_list[:limit])]
            result["residuals_data"] = {
                "predicted": [round(float(v), 4) for v in y_pred_list[:limit]],
                "residuals": residuals,
            }
            if len(y_true_list) > 0:
                min_val = min(min(y_true_list), min(y_pred_list))
                max_val = max(max(y_true_list), max(y_pred_list))
                result["regression_line"] = {"min": round(float(min_val), 4), "max": round(float(max_val), 4)}

        return result

    # ── Multi-model comparison ───────────────────────────────────
    def compare_models(self, target_col: str, scaler_type: Optional[str] = None) -> dict[str, Any]:
        """Train multiple models and compare their performance."""
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.preprocessing import LabelEncoder

        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")

        # Detect model type
        is_classification = (self._detect_target_type(target_col) == "classification")

        # Prepare data once
        X_train, X_test, y_train, y_test, target_encoder = self._prepare_data(target_col)

        # Apply scaling if requested
        X_train_s, X_test_s = self._apply_scaling(X_train.copy(), X_test.copy(), scaler_type)

        results = []

        if not is_classification:
            # Regression models
            from sklearn.linear_model import LinearRegression, Ridge, Lasso
            from sklearn.preprocessing import StandardScaler, PolynomialFeatures
            from sklearn.metrics import r2_score, mean_absolute_error

            from sklearn.ensemble import (
                RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
            )
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.svm import SVR
            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.neural_network import MLPRegressor
            from sklearn.linear_model import HuberRegressor, ElasticNet

            models = [
                ("Linear Regression", LinearRegression()),
                ("Ridge Regression", Ridge(alpha=1.0, random_state=42)),
                ("Lasso Regression", Lasso(alpha=0.1, max_iter=2000, random_state=42)),
                ("ElasticNet", ElasticNet(alpha=0.1, random_state=42)),
                ("Decision Tree", DecisionTreeRegressor(max_depth=10, random_state=42)),
                ("Random Forest", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
                ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
                ("Extra Trees", ExtraTreesRegressor(n_estimators=100, random_state=42)),
                ("AdaBoost", AdaBoostRegressor(n_estimators=50, random_state=42)),
                ("Support Vector (SVR)", SVR()),
                ("K-Nearest Neighbors (KNN)", KNeighborsRegressor(n_neighbors=5)),
                ("Neural Network (MLP)", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400, random_state=42)),
                ("Huber Regressor", HuberRegressor(max_iter=1000)),
            ]

            for name, model in models:
                try:
                    start = time.time()
                    if "Polynomial" in name:
                        deg = 2 if "2" in name else 3
                        max_f = min(10, X_train_s.shape[1])
                        cols = X_train_s.columns[:max_f].tolist()
                        X_tr_sub = X_train_s[cols].copy()
                        X_te_sub = X_test_s[cols].copy()

                        poly_scaler = StandardScaler()
                        X_tr_sub = pd.DataFrame(poly_scaler.fit_transform(X_tr_sub), columns=cols, index=X_tr_sub.index)
                        X_te_sub = pd.DataFrame(poly_scaler.transform(X_te_sub), columns=cols, index=X_te_sub.index)

                        poly = PolynomialFeatures(degree=deg, include_bias=False)
                        X_tr_p = poly.fit_transform(X_tr_sub)
                        X_te_p = poly.transform(X_te_sub)
                        m = LinearRegression()
                        m.fit(X_tr_p, y_train)
                        y_pred = m.predict(X_te_p)
                        y_train_pred = m.predict(X_tr_p)
                    else:
                        model.fit(X_train_s, y_train)
                        y_pred = model.predict(X_test_s)
                        y_train_pred = model.predict(X_train_s)
                        m = model

                    train_time = round(time.time() - start, 3)
                    train_r2 = round(float(r2_score(y_train, y_train_pred)), 4)
                    test_r2 = round(float(r2_score(y_test, y_pred)), 4)
                    mae = round(float(mean_absolute_error(y_test, y_pred)), 4)
                    gap = round(train_r2 - test_r2, 4)

                    overfit = "danger" if gap > 0.3 else ("warning" if gap > 0.15 else "good")

                    try:
                        if "Polynomial" in name:
                            cv = cross_val_score(m, X_tr_p, y_train, cv=min(5, max(2, len(y_train))), scoring='r2')
                        else:
                            cv = cross_val_score(m, X_train_s, y_train, cv=min(5, max(2, len(y_train))), scoring='r2')
                        cv_mean = round(float(np.mean(cv)), 4)
                    except Exception:
                        cv_mean = None

                    results.append({
                        "name": name,
                        "test_score": test_r2,
                        "train_score": train_r2,
                        "mae": mae,
                        "cv_score": cv_mean,
                        "training_time": train_time,
                        "overfit_level": overfit,
                        "gap": gap,
                    })
                except Exception as e:
                    results.append({"name": name, "error": str(e)})
        else:
            # Classification models
            from sklearn.linear_model import LogisticRegression, RidgeClassifier
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.ensemble import (
                RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
            )
            from sklearn.svm import SVC
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.naive_bayes import GaussianNB
            from sklearn.neural_network import MLPClassifier
            from sklearn.metrics import accuracy_score, f1_score

            n_classes = y_train.nunique()
            avg = 'binary' if n_classes == 2 else 'weighted'

            models = [
                ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
                ("Decision Tree", DecisionTreeClassifier(random_state=42, max_depth=10)),
                ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)),
                ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, random_state=42)),
                ("Extra Trees", ExtraTreesClassifier(n_estimators=100, random_state=42)),
                ("AdaBoost", AdaBoostClassifier(n_estimators=50, random_state=42)),
                ("Support Vector (SVC)", SVC(probability=True, random_state=42)),
                ("K-Nearest Neighbors (KNN)", KNeighborsClassifier(n_neighbors=5)),
                ("Gaussian Naive Bayes", GaussianNB()),
                ("Neural Network (MLP)", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=42)),
                ("Ridge Classifier", RidgeClassifier(random_state=42)),
            ]

            for name, model in models:
                try:
                    start = time.time()
                    model.fit(X_train_s, y_train)
                    y_pred = model.predict(X_test_s)
                    y_train_pred = model.predict(X_train_s)
                    train_time = round(time.time() - start, 3)

                    train_acc = round(float(accuracy_score(y_train, y_train_pred)), 4)
                    test_acc = round(float(accuracy_score(y_test, y_pred)), 4)
                    f1 = round(float(f1_score(y_test, y_pred, average=avg, zero_division=0)), 4)
                    gap = round(train_acc - test_acc, 4)

                    overfit = "danger" if gap > 0.2 else ("warning" if gap > 0.1 else "good")

                    try:
                        cv = cross_val_score(model, X_train_s, y_train, cv=min(5, len(y_train)), scoring='accuracy')
                        cv_mean = round(float(np.mean(cv)), 4)
                    except Exception:
                        cv_mean = None

                    results.append({
                        "name": name,
                        "test_score": test_acc,
                        "train_score": train_acc,
                        "f1_score": f1,
                        "cv_score": cv_mean,
                        "training_time": train_time,
                        "overfit_level": overfit,
                        "gap": gap,
                    })
                except Exception as e:
                    results.append({"name": name, "error": str(e)})

        # Find best model
        valid_results = [r for r in results if "error" not in r]
        if valid_results:
            best = max(valid_results, key=lambda r: r["test_score"])
            best["is_best"] = True

        return {
            "model_type": "classification" if is_classification else "regression",
            "target_column": target_col,
            "results": results,
            "metric": "accuracy" if is_classification else "r2_score",
        }

    # ── Learning curve ───────────────────────────────────────────
    def generate_learning_curve(self, target_col: str, degree: int = 1,
                                scaler_type: Optional[str] = None) -> dict[str, Any]:
        """Generate learning curve data."""
        from sklearn.model_selection import learning_curve

        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")

        X_train, X_test, y_train, y_test, _ = self._prepare_data(target_col)

        # Combine back for learning curve
        X_all = pd.concat([X_train, X_test])
        y_all = pd.concat([y_train, y_test])

        # Apply poly if needed
        if degree > 1:
            X_all_poly, _ = self._apply_polynomial(X_all.copy(), X_all.copy(), degree)
            X_all = X_all_poly

        # Apply scaling
        X_all, _ = self._apply_scaling(X_all.copy(), X_all.copy(), scaler_type)

        # Use cloned trained model if available
        from sklearn.base import clone
        scoring = 'r2' if self.model_type == 'regression' else 'accuracy'
        if self.model is not None:
            model = clone(self.model)
        else:
            # Fallback
            series = self.df[target_col].dropna()
            if pd.api.types.is_numeric_dtype(series) and series.nunique() > 10:
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
            else:
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(max_iter=1000, random_state=42)

        train_sizes_frac = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        n_cv = min(5, len(X_all) // 2)
        if n_cv < 2:
            n_cv = 2

        try:
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_all, y_all,
                train_sizes=train_sizes_frac,
                cv=n_cv,
                scoring=scoring,
                random_state=42,
                n_jobs=1,
            )

            return {
                "train_sizes": train_sizes.tolist(),
                "train_scores_mean": np.mean(train_scores, axis=1).tolist(),
                "train_scores_std": np.std(train_scores, axis=1).tolist(),
                "val_scores_mean": np.mean(val_scores, axis=1).tolist(),
                "val_scores_std": np.std(val_scores, axis=1).tolist(),
                "metric": scoring,
            }
        except Exception as e:
            return {"error": str(e)}

    # ── Prediction ───────────────────────────────────────────────
    def predict_single(self, feature_values: dict) -> dict[str, Any]:
        """Predict a single row from user-provided feature values."""
        if self.model is None:
            raise ValueError("No model trained yet. Train a model first.")

        # Build feature vector
        X = pd.DataFrame([feature_values])

        # Ensure correct columns
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = self._feature_medians.get(col, 0)

        X = X[self.feature_columns]

        # Apply poly scaler (auto-scaling before polynomial)
        if hasattr(self, '_poly_scaler') and self._poly_scaler is not None:
            X = pd.DataFrame(self._poly_scaler.transform(X), columns=X.columns)

        # Apply polynomial
        if self.poly_transformer is not None:
            X = pd.DataFrame(
                self.poly_transformer.transform(X),
                columns=self.poly_transformer.get_feature_names_out(self.feature_columns)
            )

        # Apply scaling
        if self.scaler is not None:
            X = pd.DataFrame(self.scaler.transform(X), columns=X.columns)

        prediction = self.model.predict(X)
        result = {"prediction": float(prediction[0])}

        # Decode if classification
        if self._target_encoder is not None:
            try:
                result["prediction_label"] = str(self._target_encoder.inverse_transform([int(prediction[0])])[0])
            except Exception:
                result["prediction_label"] = str(prediction[0])
        elif self.model_type == "classification":
            result["prediction_label"] = str(int(prediction[0]))

        # Confidence for classification
        if self.model_type == "classification" and hasattr(self.model, 'predict_proba'):
            try:
                proba = self.model.predict_proba(X)
                result["confidence"] = round(float(np.max(proba)), 4)
                result["probabilities"] = {str(c): round(float(p), 4) for c, p in
                                           zip(self.model.classes_, proba[0])}
            except Exception:
                pass

        return result
