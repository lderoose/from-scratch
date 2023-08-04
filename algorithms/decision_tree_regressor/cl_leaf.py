#!/usr/bin/python3.8



class Leaf:
    
    def __init__(self, id, label, node):
        self.id = id
        self.label = label
        self.depth = node.depth
        self.parent_node = node.parent
        self.n_samples = len(node.y)
        return None
    
    def __repr__(self):
        return f"Leaf_{self.id}[n_samples: {self.n_samples} ; label={self.label} ; parent={self.parent_node}]"
