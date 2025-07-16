import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn import svm

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def main():
    # Load in data
    X, y = joblib.load('./data/processed/avg_mfcc_data.pkl')

    # Split data into testing and training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Create a pipeline of SVM and define parameter we use to gridsearch with
    pipe = make_pipeline(StandardScaler(), svm.SVC(random_state=3, probability=False))
    param_grid = [{'svc__C': [2, 4, 6], 
                    'svc__kernel': ['rbf', 'linear'],
                    'svc__class_weight': ['balanced', None]}]

    # GridSearchCV tuning
    gs = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='accuracy', refit=True, cv=3, verbose=3, n_jobs=-1)
    gs.fit(X_train, y_train)
    best_model = gs.best_estimator_

    print(f"\nBest Params: {gs.best_params_}")
    print(f"CV Accuracy: {cross_val_score(best_model, X_train, y_train, cv=3).mean():.4f}")

    # Save model
    joblib.dump(best_model, './models/best_svm_model.pkl')

    print(f"Saved model and test set.")

    return 0

if __name__ == "__main__":
    main()
