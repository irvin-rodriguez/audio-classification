from model_utils import *

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import joblib

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn import svm

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def main():
    df_mfcc = pd.read_pickle("./data/processed/mfcc_data.pkl")  # path to data

    # Split and prepare
    X_train, X_test, y_train, y_test = split_data(df_mfcc, num_mfcc=25, test_size=0.2, random_state=42)

    # Save training and test set for future use with training/testing other models
    joblib.dump((X_train, y_train), './data/processed/saved_train_set.pkl')
    joblib.dump((X_test, y_test), './data/processed/saved_test_set.pkl')

    # Create a pipeline of SVM and define parameter we use to gridsearch with
    pipe = make_pipeline(StandardScaler(), svm.SVC(random_state=3, probability=False))
    param_grid = [{'svc__C': [4], 
                    'svc__kernel': ['rbf'],
                    'svc__class_weight': ['balanced']}]

    # GridSearchCV tuning
    gs = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='accuracy', refit=True, cv=3, verbose=3, n_jobs=-1)
    gs.fit(X_train, y_train)
    best_model = gs.best_estimator_

    print(f"\nBest Params: {gs.best_params_}")
    print(f"CV Accuracy: {cross_val_score(best_model, X_train, y_train, cv=3).mean():.4f}")

    # Save model
    joblib.dump(best_model, './models/best_svm_model.pkl')

    print(f"Saved model and test set. Ready for comparison script.")

    return 0

if __name__ == "__main__":
    main()
