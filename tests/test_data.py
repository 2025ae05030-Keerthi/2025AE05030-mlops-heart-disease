import os
import pandas as pd
import pytest
from src.train import build_pipeline
from sklearn.linear_model import LogisticRegression

DATA_PATH = "data/heart_disease.csv"

@pytest.fixture
def dataset():
    # If not present locally, ensure download has been run
    if not os.path.exists(DATA_PATH):
        pytest.skip(f"Dataset file not found at {DATA_PATH}. Run src/download_data.py first.")
    return pd.read_csv(DATA_PATH)

def test_dataset_columns(dataset):
    expected_cols = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
        'ca', 'thal', 'num'
    ]
    for col in expected_cols:
        assert col in dataset.columns, f"Missing expected column: {col}"

def test_target_is_binary(dataset):
    unique_targets = set(dataset['num'].unique())
    assert unique_targets.issubset({0, 1}), f"Target contains non-binary values: {unique_targets}"

def test_pipeline_transformation_shape(dataset):
    X = dataset.drop(columns=['num'])
    y = dataset['num'].values.ravel()
    
    pipeline = build_pipeline(LogisticRegression())
    pipeline.fit(X[:50], y[:50])
    
    # Test inference on a single row
    sample = X.iloc[[0]]
    prediction = pipeline.predict(sample)
    proba = pipeline.predict_proba(sample)
    
    assert len(prediction) == 1
    assert proba.shape == (1, 2)