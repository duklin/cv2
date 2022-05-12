import numpy as np
from Node import Node
# from Sampler import PatchSampler


class DecisionTree():
    def __init__(self, patches, labels, tree_param: dict):
        self.patches, self.labels = patches, labels
        self.depth = tree_param['depth']
        self.pixel_locations = tree_param['pixel_locations']
        self.random_color_values = tree_param['random_color_values']
        self.no_of_thresholds = tree_param['no_of_thresholds']
        self.minimum_patches_at_leaf = tree_param['minimum_patches_at_leaf']
        self.classes = tree_param['classes']
        self.root_node = Node()

    # Function to train the tree
    # should return a trained tree with provided tree param
    def train(self):
        self.split_node(self.root_node, self.patches, self.labels, 0)

    # Function to predict probabilities for single image
    # should return predicted class for every pixel in the test image
    def predict(self, I, patch_size):
        p1 = patch_size // 2
        p2 = (patch_size + 1) // 2
        image = np.pad(I, ((p1, p2 - 1), (p1, p2 - 1), (0, 0)), 'constant', constant_values=0)
        h, w, _ = image.shape
        segmap = np.zeros((I.shape[0], I.shape[1]))
        for i, r in enumerate(range(p1, h - p2 + 1)):
            for j, c in enumerate(range(p1, w - p2 + 1)):
                node = self.root_node
                patch = image[r - p1:r + p2, c - p1:c + p2]
                while True:
                    if node.type == ["SplitNode"]:
                        feature = node.feature
                        x, y = feature['pixel_location']
                        c = feature['color']
                        th = feature['th']
                        node = node.leftChild if patch[x, y, c] < th else node.rightChild
                    else:
                        segmap[i, j] = np.argmax(node.probabilities)
                        break
        return segmap.astype(np.int64)
                    
    # Function to get feature response for a random color and pixel location
    # should return feature response for all input patches
    def getFeatureResponse(self, patches, feature):
        x, y = feature["pixel_location"]
        c = feature["color"]
        return patches[:, x, y, c]

    # Function to get left/right split given feature responses and a threshold
    # should return left/right split
    def getsplit(self, responses, threshold):
        return np.where(responses < threshold, "l", "r")

    # Function to get a random pixel location
    # should return a random location inside the patch
    def generate_random_pixel_location(self):
        return [np.random.random_integers(0, 15), np.random.random_integers(0, 15)]

    # Function to compute entropy over incoming class labels
    # provide your implementation
    def compute_entropy(self, labels):
        p = np.bincount(labels) / len(labels)
        entropy = -np.sum([c * np.log(c) for c in p if c != 0])
        return entropy
        
    # Function to measure information gain for a given split
    # provide your implementation
    def get_information_gain(self, Entropyleft, Entropyright, EntropyAll, Nall, Nleft, Nright):
        return EntropyAll - ((Nleft / Nall) * Entropyleft + (Nright / Nall) * Entropyright)

    # Function to get the best split for given patches with labels
    # should return left,right split, color, pixel location and threshold
    def best_split(self, patches, labels):
        thresholds = np.linspace(10, 245, self.no_of_thresholds)
        EntropyAll = self.compute_entropy(labels)
        feature = {'color': -1, 'pixel_location': [-1, -1], 'th': -1}
        best_gain = 0
        for _ in range(self.pixel_locations):
            pixel = self.generate_random_pixel_location()
            feature['pixel_location'] = pixel
            for c in self.random_color_values:
                feature['color'] = c
                responses = self.getFeatureResponse(patches, feature)
                for th in thresholds:
                    split = self.getsplit(responses, th)
                    left_labels = labels[split == 'l']
                    right_labels = labels[split == 'r']
                    Entropyleft = self.compute_entropy(left_labels)
                    Entropyright = self.compute_entropy(right_labels)
                    gain = self.get_information_gain(Entropyleft, Entropyright, EntropyAll, len(labels), len(left_labels), len(right_labels))
                    if gain >= best_gain:
                        best_gain = gain
                        best_feature = {'color': c, 'pixel_location': pixel, 'th': th}
                        best_split = np.copy(split)
        
        return best_split, best_feature

    def split_node(self, node, patches, labels, depth):
        print("depth: ", depth)
        if depth == self.depth:
            node.create_leafNode(labels, self.classes)
            return
        split, feature = self.best_split(patches, labels)
        left_patches, left_labels = patches[split == 'l'], labels[split == 'l']
        right_patches, right_labels = patches[split == 'r'], labels[split == 'r']
        if len(left_patches) > self.minimum_patches_at_leaf and len(right_patches) > self.minimum_patches_at_leaf:
            leftChild = Node()
            rightChild = Node()
            node.create_SplitNode(leftChild, rightChild, feature)
            self.split_node(leftChild, left_patches, left_labels, depth + 1)
            self.split_node(rightChild, right_patches, right_labels, depth + 1)
        else:
            node.create_leafNode(labels, self.classes)
            return
