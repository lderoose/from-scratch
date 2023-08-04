#!/usr/bin/python3.8


import pandas as pd

from tqdm import tqdm
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor as sk_dtr

from algorithms.decision_tree_regressor.cl_decision_tree_regressor import DecisionTreeRegressor as my_dtr



if __name__ == "__main__":
    
    # settings
    LS_MAX_DEPTH = [2, 3, 4]
    CRITERION = "squared_error"
    
    # generate fake dataset
    X, y = make_regression(
        n_samples=1000, 
        n_features=2, 
        random_state=2023
    )

    data = pd.concat(objs=[pd.DataFrame(X), pd.Series(y)], axis=1)
    ls_features = ["feature_0", "feature_1"]
    label = ["label"]
    data.columns = ls_features + label
    
    for max_depth in tqdm(LS_MAX_DEPTH):
        # my
        my_clf = my_dtr(criterion=CRITERION, max_depth=max_depth)
        my_clf.fit(X=data[ls_features], y=data[label])
        my_score = my_clf.score(X=data[ls_features], y=data[label])
        
        # display elements of the tree
        for node in my_clf.tree["nodes"]:
            print(node)
        for leaf in my_clf.tree["leaves"]:
            print(leaf)
        
        # sklearn
        sk_clf = sk_dtr(criterion=CRITERION, max_depth=max_depth)
        sk_clf_fitted = sk_clf.fit(X=data[ls_features], y=data[label])
        sk_score = sk_clf_fitted.score(X=data[ls_features], y=data[label])
        
        print(f"***MAX_DEPTH={max_depth} => {my_score=} vs {sk_score=}***\n")  
 