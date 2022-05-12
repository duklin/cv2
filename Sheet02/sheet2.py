from RandomForest import Forest
from Tree import DecisionTree

from Sampler import PatchSampler
import numpy as np
import cv2
import json


def readData(root_folder: str, data_ascii_file: str):
    data = np.loadtxt(root_folder + data_ascii_file, str)
    num_images, num_classes = data[0].astype(np.int16)
    images = []
    segmaps = []
    for i in range(num_images):
        image = cv2.imread(root_folder + data[i + 1, 0])
        images.append(image)
        segmap = cv2.imread(root_folder + data[i + 1, 1], 0)
        segmaps.append(segmap)
    
    return num_images, num_classes, images, segmaps

def main():
    # Read data
    train_n, train_k, train_imgs, train_segmaps = readData("images/", "train_images.txt")
    test_n, test_k, test_imgs, test_segmap = readData("images/", "test_images.txt")
    
    # Sample patches from training data
    patch_size = 16
    train_classes = np.arange(train_k)
    train_patch_sampler = PatchSampler(train_imgs, train_segmaps, train_classes, patch_size)
    train_patch_sampler.extractpatches()


    tree_params = {"depth": 15,
                   "pixel_locations": 100,
                   "random_color_values": np.random.random_integers(0, 2, 10),
                   "no_of_thresholds": 50,
                   "minimum_patches_at_leaf": 20,
                   "classes": train_classes}

    n_trees = 5
    forest = Forest(train_patch_sampler.training_patches, train_patch_sampler.training_patches_labels, tree_params, n_trees)
    forest.create_forest()

    # tree = DecisionTree(train_patch_sampler.training_patches, train_patch_sampler.training_patches_labels, tree_params)
    # tree.train()

    for img in test_imgs:
        segmap = forest.test(img, patch_size)
        colored_segmap = np.zeros_like(img)
        for i in range(colored_segmap.shape[0]):
            for j in range(colored_segmap.shape[1]):
                colored_segmap[i, j] = train_patch_sampler.class_colors[segmap[i, j]]
        cv2.imshow('prediction', colored_segmap)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
