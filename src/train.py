import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def build_pipeline(model):
    num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

def train_and_evaluate():
    mlflow.set_experiment("Heart-Disease-Prediction")
    df = pd.read_csv("data/heart_disease.csv")
    
    target_col = 'num'
    X = df.drop(columns=[target_col])
    y = df[target_col].values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    }

    best_auc = 0.0
    best_pipeline = None

    for name, clf in models.items():
        with mlflow.start_run(run_name=name):
            pipeline = build_pipeline(clf)

            # Cross-validation
            cv_results = cross_validate(
                pipeline, X_train, y_train, cv=5,
                scoring=['accuracy', 'precision', 'recall', 'roc_auc']
            )

            # Fit on full train split
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)

            # Log metrics to MLflow
            mlflow.log_param("model_name", name)
            mlflow.log_metric("cv_mean_roc_auc", np.mean(cv_results['test_roc_auc']))
            mlflow.log_metric("test_accuracy", acc)
            mlflow.log_metric("test_precision", prec)
            mlflow.log_metric("test_recall", rec)
            mlflow.log_metric("test_roc_auc", auc)

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
            )

            if auc > best_auc:
                best_auc = auc
                best_pipeline = pipeline

    # Export best model pipeline
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipeline, "models/model_pipeline.joblib")
    print(f"Best model exported to models/model_pipeline.joblib with Test AUC: {best_auc:.4f}")

if __name__ == "__main__":
    train_and_evaluate()