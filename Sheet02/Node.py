import numpy as np

class Node():
    def __init__(self):
        self.type = 'None'
        self.leftChild = -1
        self.rightChild = -1
        self.feature = {'color': -1, 'pixel_location': [-1, -1], 'th': -1}
        self.probabilities = []

    # Function to create a new split node
    def create_SplitNode(self, leftchild, rightchild, feature):
        self.type = ["SplitNode"]
        self.leftChild = leftchild
        self.rightChild = rightchild
        self.feature = feature

    # Function to create a new leaf node
    def create_leafNode(self, labels, classes):
        self.type = ["LeafNode"]
        self.probabilities = np.bincount(labels, minlength=len(classes)) / len(labels)