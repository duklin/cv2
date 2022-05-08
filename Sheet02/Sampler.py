import numpy as np

class PatchSampler():
    def __init__(self, train_images_list, gt_segmentation_maps_list, classes_colors, patch_size):

        self.train_images_list = train_images_list
        self.gt_segmentation_maps_list = gt_segmentation_maps_list
        self.class_colors = classes_colors
        self.patch_size = patch_size

    # Function for sampling patches for each class
    # provide your implementation
    # should return extracted patches with labels
    def extractpatches(self):
        p1 = self.patch_size // 2
        p2 = (self.patch_size + 1) // 2
        self.training_patches = []
        self.training_patches_labels = []
        for image, segmap in zip(self.train_images_list, self.gt_segmentation_maps_list):
            h, w, _ = image.shape
            num_patches = int((w * h) / 4)
            rows = np.random.randint(p1, h - p2 + 1, num_patches)
            cols = np.random.randint(p1, w - p2 + 1, num_patches)
            for r, c in zip(rows, cols):
                patch = image[r - p1:r + p2, c - p1: c + p2]
                self.training_patches.append(patch)
                self.training_patches_labels.append(segmap[r, c])