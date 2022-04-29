import numpy as np
from Node import Node
# from Sampler import PatchSampler


class DecisionTree():
    def __init__(self, patches, labels, tree_param):

        self.patches, self.labels = patches, labels
        self.depth = tree_param['depth']
        self.pixel_locations = tree_param['pixel_locations']
        self.random_color_values = tree_param['random_color_values']
        self.no_of_thresholds = tree_param['no_of_thresholds']
        self.minimum_patches_at_leaf = tree_param['minimum_patches_at_leaf']
        self.classes = tree_param['classes']
        self.nodes = []

    # Function to train the tree
    # provide your implementation
    # should return a trained tree with provided tree param
    def train(self):
        pass

    # Function to predict probabilities for single image
    # provide your implementation
    # should return predicted class for every pixel in the test image
    def predict(self, I):
        pass

    # Function to get feature response for a random color and pixel location
    # provide your implementation
    # should return feature response for all input patches
    def getFeatureResponse(self, patches, feature):
        pass

    # Function to get left/right split given feature responses and a threshold
    # provide your implementation
    # should return left/right split
    def getsplit(self, responses, threshold):
        pass

    # Function to get a random pixel location
    # provide your implementation
    # should return a random location inside the patch
    def generate_random_pixel_location(self):
        pass

    # Function to compute entropy over incoming class labels
    # provide your implementation
    def compute_entropy(self, labels):
        pass

    # Function to measure information gain for a given split
    # provide your implementation
    def get_information_gain(self, Entropyleft, Entropyright, EntropyAll, Nall, Nleft, Nright):
        pass

    # Function to get the best split for given patches with labels
    # provide your implementation
    # should return left,right split, color, pixel location and threshold
    def best_split(self, patches, labels):
        pass

    # feel free to add any helper functions

