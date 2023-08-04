#!/usr/bin/python3.8


import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier as sk_dtc

from algorithms.decision_tree_classifier.cl_decision_tree_classifier import DecisionTreeClassifier as my_dtc




if __name__ == "__main__":
    
    # settings
    LS_MAX_DEPTH = [2, 3, 4]
    CRITERION = "entropy"
    
    # generate fake dataset
    X, y = make_classification(
        n_samples=1000, 
        n_features=2, 
        n_redundant=0, 
        n_classes=2, 
        flip_y=0,
        random_state=2023
    )

    data = pd.concat(objs=[pd.DataFrame(X), pd.Series(y)], axis=1)
    ls_features = ["feature_0", "feature_1"]
    label = ["label"]
    data.columns = ls_features + label

    idx_0 = data[data.label == 0].index
    idx_1 = data[data.label == 1].index

    # # display
    # plt.figure(figsize=(8, 8))
    # plt.scatter(data.loc[idx_0, "feature_0"], data.loc[idx_0, "feature_1"], c="darkblue", label="label_0")
    # plt.scatter(data.loc[idx_1, "feature_0"], data.loc[idx_1, "feature_1"], c="darkgreen", label="label_1")
    # plt.legend()
    # plt.show()
    
    for max_depth in tqdm(LS_MAX_DEPTH):
        # my
        my_clf = my_dtc(criterion=CRITERION, max_depth=max_depth)
        my_clf.fit(X=data[ls_features], y=data[label])
        my_score = my_clf.score(X=data[ls_features], y=data[label])
        
        # display elements of the tree
        for node in my_clf.tree["nodes"]:
            print(node)
        for leaf in my_clf.tree["leaves"]:
            print(leaf)
        
        # sklearn
        sk_clf = sk_dtc(criterion="entropy", max_depth=max_depth)
        sk_clf_fitted = sk_clf.fit(X=data[ls_features], y=data[label])
        sk_score = sk_clf_fitted.score(X=data[ls_features], y=data[label])
        
        print(f"***MAX_DEPTH={max_depth} => {my_score=} vs {sk_score=}***\n")
     