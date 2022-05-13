import numpy as np


class PatchSampler:
    def __init__(
        self,
        train_images_list: list,
        gt_segmentation_maps_list: list,
        classes_colors: np.ndarray,
        patch_size: int,
    ):

        self.train_images_list = train_images_list
        self.gt_segmentation_maps_list = gt_segmentation_maps_list
        self.class_colors = {
            0: [0, 0, 0],
            1: [0, 0, 255],
            2: [255, 0, 0],
            3: [0, 255, 0],
        }
        self.patch_size = patch_size

    # Function for sampling patches for each class
    # provide your implementation
    # should return extracted patches with labels
    def extractpatches(self):
        p1 = self.patch_size // 2
        p2 = (self.patch_size + 1) // 2
        self.training_patches = []
        self.training_patches_labels = []
        for image, segmap in zip(
            self.train_images_list, self.gt_segmentation_maps_list
        ):
            num_patches = int((image.shape[0] * image.shape[1]) / 3)
            image = np.pad(
                image,
                ((p1, p2 - 1), (p1, p2 - 1), (0, 0)),
                "constant",
                constant_values=0,
            )
            h, w, _ = image.shape
            rows = np.random.randint(p1, h - p2 + 1, num_patches)
            cols = np.random.randint(p1, w - p2 + 1, num_patches)
            for r, c in zip(rows, cols):
                patch = image[r - p1 : r + p2, c - p1 : c + p2]
                self.training_patches.append(patch)
                self.training_patches_labels.append(segmap[r - p1, c - p1])
        self.training_patches = np.array(self.training_patches)
        self.training_patches_labels = np.array(self.training_patches_labels)

        # Balancing dataset
        min_label_count = np.min(np.bincount(self.training_patches_labels))
        balanced_patches = []
        balanced_labels = []
        for label in np.unique(self.training_patches_labels):
            idx = np.where(self.training_patches_labels == label)[0]
            balanced_patches.append(self.training_patches[idx[:min_label_count]])
            balanced_labels.append(self.training_patches_labels[idx[:min_label_count]])

        balanced_patches = np.array(balanced_patches)
        balanced_labels = np.array(balanced_labels)
        idx = np.arange(len(balanced_labels))
        np.random.shuffle(idx)
        self.training_patches = balanced_patches[idx].reshape(
            -1, self.patch_size, self.patch_size, 3
        )
        self.training_patches_labels = balanced_labels[idx].reshape(-1)
