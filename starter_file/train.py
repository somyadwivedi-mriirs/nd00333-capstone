import os
import joblib
import argparse
from azureml.core import Run, Dataset

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

def train():
    
    # Get variable hyperparameters as arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
    
    args = parser.parse_args()
        
    # Get run context from AzureML
    run = Run.get_context()
    ws = run.experiment.workspace
    
    # Read Data
    df = Dataset.get_by_name(ws, 'heart-disease-kaggle').to_pandas_dataframe()
    
    X, y = df.drop('DEATH_EVENT', axis=1), df['DEATH_EVENT']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define Classifier and Train
    clf = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    
    # Log metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    run.log('Accuracy', np.float(accuracy))
    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    
    # Dump the model
    
    os.makedirs('outputs', exist_ok=True)
    
    model_path = "outputs/model.joblib")
    
    joblib.dump(clf, model_path)
    

if __name__ == "__main__":
    train()