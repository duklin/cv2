import cv2
import numpy as np

from RandomForest import Forest
from Sampler import PatchSampler
from Tree import DecisionTree


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
    train_n, train_k, train_imgs, train_segmaps = readData(
        "images/", "train_images.txt"
    )
    test_n, test_k, test_imgs, test_segmap = readData("images/", "test_images.txt")

    # Sample patches from training data
    patch_size = 16
    train_classes = np.arange(train_k)
    train_patch_sampler = PatchSampler(
        train_imgs, train_segmaps, train_classes, patch_size
    )
    train_patch_sampler.extractpatches()

    tree_params = {
        "depth": 15,
        "pixel_locations": 100,
        "random_color_values": np.random.randint(0, 3, 10),
        "no_of_thresholds": 50,
        "minimum_patches_at_leaf": 20,
        "classes": train_classes,
    }

    # Random Forest
    n_trees = 5
    forest = Forest(
        train_patch_sampler.training_patches,
        train_patch_sampler.training_patches_labels,
        tree_params,
        n_trees,
    )
    forest.create_forest()

    # Decision Tree
    tree = DecisionTree(
        train_patch_sampler.training_patches,
        train_patch_sampler.training_patches_labels,
        tree_params,
    )
    tree.train()

    # Inference
    for img in test_imgs:
        segmap_tree = tree.predict(img, patch_size)
        segmap_forest = forest.test(img, patch_size)
        color_segmap_tree = np.zeros_like(img)
        color_segmap_forest = np.zeros_like(img)
        for i in range(color_segmap_tree.shape[0]):
            for j in range(color_segmap_tree.shape[1]):
                color_segmap_tree[i, j] = train_patch_sampler.class_colors[
                    segmap_tree[i, j]
                ]
                color_segmap_forest[i, j] = train_patch_sampler.class_colors[
                    segmap_forest[i, j]
                ]
        # cv2.imshow('prediction-tree', color_segmap_tree)
        # cv2.imshow('prediction-forest', color_segmap_forest)
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    cv2.imwrite("results/img_12_tree.bmp", color_segmap_tree)
    cv2.imwrite("results/img_12_forest.bmp", color_segmap_forest)


if __name__ == "__main__":
    main()
