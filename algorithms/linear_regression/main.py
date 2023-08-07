#!/usr/bin/python3.8


import pandas as pd

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression as sk_lr

from algorithms.linear_regression.cl_linear_regression import LinearRegression as my_lr




if __name__ == "__main__":
    
    # settings
    LR = 0.01
    N_ITER = 1000
    LS_INTERCEPT = [True, False]
    N_FEATURES = 8
    BIAS_DATA = 5
    
    # generate fake dataset
    X, y = make_regression(
        n_samples=1000, 
        n_features=N_FEATURES, 
        random_state=2023,
        bias=BIAS_DATA
    )

    data = pd.concat(objs=[pd.DataFrame(X), pd.Series(y)], axis=1)
    ls_features = [f"feature_{i}" for i in range(N_FEATURES)]
    label = ["label"]
    data.columns = ls_features + label

    for intercept in LS_INTERCEPT:
        # my
        my_model = my_lr(learning_rate=LR, num_iterations=N_ITER, fit_intercept=intercept)
        my_model.fit(data[ls_features], data[label])
        my_score = my_model.score(data[ls_features], data[label])
        
        # sklearn
        sk_model = sk_lr(fit_intercept=intercept)
        sk_model.fit(data[ls_features], data[label])
        sk_score = sk_model.score(data[ls_features], data[label])
        
        print(f"my_coefs: {my_model.coef_} ; intercept: {my_model.intercept_}")
        print(f"sk_coefs: {sk_model.coef_} ; intercept: {sk_model.intercept_}")
        print(f"***INTERCEPT={intercept} => {my_score=} vs {sk_score=}***\n")  
