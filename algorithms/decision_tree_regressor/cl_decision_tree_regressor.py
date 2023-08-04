#!/usr/bin/python3.8


from queue import SimpleQueue

import numpy as np

from algorithms.decision_tree_regressor.cl_node import Node
from algorithms.decision_tree_regressor.cl_leaf import Leaf



class DecisionTreeRegressor:
    
    def __init__(self, criterion="squared_error", max_depth=3):
        assert criterion == "squared_error", f"only criterion 'squared_error' is implemented."
        assert isinstance(max_depth, int), f"expected type int for parameter max_depth, get type '{type(max_depth)}' instead."
        assert (max_depth >= -1), f"expected max_depth >= -1, get '{max_depth}' instead."
        self.criterion = criterion
        self.f_criterion = self.mseCriterion
        self.max_depth = max_depth
        
        self.tree = None
        return None
    
    @staticmethod
    def mseCriterion(y):
        mse = 0
        if len(y) > 0:
            y_bar = np.mean(y)
            mse = np.sum((y - y_bar)**2) / len(y)
        return mse
    
    def __applySplit(self, X, y, feature, v):
        left_X, right_X = X[X[feature] < v], X[X[feature] >= v]
        left_y, right_y = y.loc[left_X.index], y.loc[right_X.index]
        return left_X, right_X, left_y, right_y
    
    def _criterionFromSplit(self, X, y, feature, v):
        left_X, right_X, left_y, right_y = self.__applySplit(X, y, feature, v)
        n_left, n_right = len(left_X), len(right_X)
        left_criterion, right_criterion = self.f_criterion(left_y.values.ravel()), self.f_criterion(right_y.values.ravel())
        criterion = (n_left*left_criterion + n_right*right_criterion) / (n_left + n_right)
        return criterion
    
    def _findBestSplit(self, X, y):
        ls_features = X.columns
        best_split = {
            "feature": None,
            "value": None,
            "criterion": np.inf
        }
        for feature in ls_features:
            for v in X[feature]:
                criterion = self._criterionFromSplit(X, y, feature, v)
                if criterion < best_split["criterion"]:
                    # update split
                    best_split["feature"] = feature
                    best_split["value"] = v
                    best_split["criterion"] = criterion
        return best_split
    
    def fit(self, X, y):
        # incomplete due to missing child and split data (feature, value, criterion)
        q_incomplete_nodes = SimpleQueue()
        
        # the first task
        root_node = Node(
            id=0,
            X=X.copy(), 
            y=y.copy(),
            parent=None,
            depth=0
        )
        q_incomplete_nodes.put(root_node) 
        
        n_leaves = -1
        ls_nodes = []
        ls_leaves = []
        
        while not q_incomplete_nodes.empty():
                        
            node = q_incomplete_nodes.get()        
            best_split = self._findBestSplit(node.X.copy(), node.y.copy())
            
            id_child_node_a = (2*node.id) + 1
            id_child_node_b = (2*node.id) + 2
            
            # complete node
            node.feature = best_split["feature"]
            node.value = best_split["value"]
            node.criterion = best_split["criterion"]
            node.child = [("node", id_child_node_a), ("node", id_child_node_b)]
            
            # maj
            current_depth = node.depth + 1
            
            # perform the split
            left_X, right_X, left_y, right_y = self.__applySplit(
                node.X.copy(), 
                node.y.copy(), 
                node.feature, 
                node.value
            )
            
            # create incomplete child nodes (can be a leaf if max_depth constraint met)
            incomplete_child_node_a = Node(
                id=id_child_node_a,  
                X=left_X.copy(),
                y=left_y.copy(),
                parent=node.id,
                depth=node.depth+1,
            )
            incomplete_child_node_b = Node(
                id=id_child_node_b,  
                X=right_X.copy(),
                y=right_y.copy(),
                parent=node.id,
                depth=node.depth+1,
            )   
            
            # max_depth constraint
            if current_depth >= self.max_depth:
                
                # assign label to the leaves by majority label
                label_a = 0
                if not incomplete_child_node_a.y.empty:
                    label_a = np.mean(incomplete_child_node_a.y.values)
                label_b = 0 
                if not incomplete_child_node_b.y.empty:
                    label_b = np.mean(incomplete_child_node_b.y.values)
                
                # create and save leaves
                leaf_a = Leaf(id=n_leaves+1, label=label_a, node=incomplete_child_node_a)
                leaf_b = Leaf(id=n_leaves+2, label=label_b, node=incomplete_child_node_b)
                ls_leaves.append(leaf_a)
                ls_leaves.append(leaf_b)
                
                node.child = [("leaf", leaf_a.id), ("leaf", leaf_b.id)]
                
                n_leaves += 2
                
            else:  
                # append incomplete nodes to the queue
                q_incomplete_nodes.put(incomplete_child_node_a)
                q_incomplete_nodes.put(incomplete_child_node_b)  
                
            # save filled parent node
            ls_nodes.append(node)   
        
        # fill tree   
        self.tree = {"nodes": ls_nodes, "leaves": ls_leaves}
        return None
    
    def _predict_one(self, x):
        id_node = 0
        leaf_met = False
        
        while not leaf_met:
            
            current_node = self.tree["nodes"][id_node]
            idx_feature = int(current_node.feature.strip("feature_"))
            idx_child = 0 if x[idx_feature] < current_node.value else 1
            next_elem = current_node.child[idx_child]

            if next_elem[0] == "leaf":
                id_leaf = next_elem[1]
                leaf_met = True
                
            else:
                id_node = next_elem[1]
                
        pred_label = self.tree["leaves"][id_leaf].label
        return pred_label
    
    def predict(self, X):
        return X.apply(lambda x: self._predict_one(x), axis=1)
    
    def score(self, X, y):
        y_bar = np.mean(y["label"])
        y_pred = self.predict(X)
        u = np.sum((y["label"] - y_pred)** 2)
        v = np.sum((y["label"] - y_bar)** 2)
        r2 = 1 - (u/v)
        return r2
