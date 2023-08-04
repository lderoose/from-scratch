#!/usr/bin/python3.8


import copy


class Node:
        
    def __init__(self, id=None, X=None, y=None, feature=None, value=None, criterion=None, child=None, parent=None, depth=-1):
        self.id = id
        self.X = X
        self.y = y
        self.feature = feature
        self.value = value
        self.criterion = criterion
        self.child = child
        self.parent = parent
        self.depth = depth
        
        self.root = False
        if self.id == 0:
            self.root = True
        return None
    
    def __repr__(self):
        return f"Node_{self.id}[({self.feature}, {self.value}) ; root={self.root} ; parent={self.parent}]"
    
    def copy(self):
        return copy.deepcopy(self)