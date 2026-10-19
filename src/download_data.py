import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

def download_data():
    os.makedirs("data", exist_ok=True)
    # UCI ID 45 is the Heart Disease dataset
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    
    # Target in raw UCI dataset is 0 (healthy) and 1..4 (disease levels).
    # Convert to binary target: 0 = absent, 1 = present
    y_binary = (y > 0).astype(int)
    
    df = pd.concat([X, y_binary], axis=1)
    df.to_csv("data/heart_disease.csv", index=False)
    print("Dataset saved to data/heart_disease.csv")

if __name__ == "__main__":
    download_data()