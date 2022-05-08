from RandomForest import Forest
from Sampler import PatchSampler
import numpy as np
import cv2
import json

def read_data(root_folder: str, data_ascii_file: str):
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
    train_n, train_k, train_imgs, train_segmaps = read_data("images/", "train_images.txt")
    test_n, test_k, test_imgs, test_segmap = read_data("images/", "test_images.txt")
    
    # Sample patches from training data
    patch_size = 16
    train_classes = np.arange(train_k)
    train_patch_sampler = PatchSampler(train_imgs, train_segmaps, train_classes, patch_size)
    train_patch_sampler.extractpatches()

if __name__ == "__main__":
    main()
