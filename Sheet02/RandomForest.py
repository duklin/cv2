import numpy as np

from Tree import DecisionTree


class Forest:
    def __init__(self, patches=[], labels=[], tree_param=[], n_trees=1):

        self.patches, self.labels = patches, labels
        self.tree_param = tree_param
        self.ntrees = n_trees
        self.trees = []
        # Bagging for training data for each tree
        for i in range(n_trees):
            idx = np.arange(len(patches))
            np.random.shuffle(idx)
            idx = idx[: len(patches) // 2]
            self.trees.append(
                DecisionTree(self.patches[idx], self.labels[idx], self.tree_param)
            )

    # Function to create ensemble of trees
    # Should return a trained forest with n_trees
    def create_forest(self):
        for i, tree in enumerate(self.trees):
            print(f"Training tree {i} from the forest")
            tree.train()

    # Function to apply the trained Random Forest on a test image
    # should return class for every pixel in the test image
    def test(self, I, patch_size):
        # Take vote of each tree's prediction
        segmaps = np.dstack([tree.predict(I, patch_size) for tree in self.trees])
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=2, arr=segmaps
        )
