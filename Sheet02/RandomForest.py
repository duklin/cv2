from Tree import DecisionTree
import numpy as np
import json


class Forest():
    def __init__(self, patches=[], labels=[], tree_param=[], n_trees=1):

        self.patches, self.labels = patches, labels
        self.tree_param = tree_param
        self.ntrees = n_trees
        self.trees = []
        for i in range(n_trees):
            self.trees.append(DecisionTree(self.patches, self.labels, self.tree_param))

    # Function to create ensemble of trees
    # provide your implementation
    # Should return a trained forest with n_trees
    def create_forest(self):
        pass

    # Function to apply the trained Random Forest on a test image
    # provide your implementation
    # should return class for every pixel in the test image
    def test(self, I):
        pass

    # feel free to add any helper functions
