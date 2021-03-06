import cv2
import numpy as np


class CustomHogDetector:

    # Some constants that you will be using in your implementation
    detection_width = 64  # the crop width dimension
    detection_height = 128  # the crop height dimension
    window_stride = 32  # the stride size
    scaleFactors = (
        1.05  # scale each patch down by this factor, feel free to try other values
    )
    # You may play with different values for these two theshold values below as well
    hit_threshold = 0.55  # detections above this threshold are counted as positive.
    overlap_threshold = 0.2  # if the overlap between two detections is above this threshold, eliminate the one with the lower confidence score.

    def __init__(self, trained_svm_name):
        self.svm = cv2.ml.SVM_load(trained_svm_name)
        self.hog = cv2.HOGDescriptor()

    def create_pyramid(self, image):
        """
        Create the image pyramid

        No blurring is done intentially since the paper
        mentions that the blurring worsens the results (6.7 Discussion).

        Returns:
        list of images, the first element is equal to the original image"""
        pyramid = [image.copy()]
        while True:
            resized = cv2.resize(
                pyramid[-1],
                dsize=None,
                fx=1 / CustomHogDetector.scaleFactors,
                fy=1 / CustomHogDetector.scaleFactors,
                interpolation=cv2.INTER_CUBIC,
            )
            if (
                resized.shape[0] >= CustomHogDetector.detection_height
                and resized.shape[1] >= CustomHogDetector.detection_width
            ):
                pyramid.append(resized)
            else:
                break
        return pyramid

    def slide_window(self, image):
        """
        Extract HOG descriptors across the image
        Keep the left upper (i, j) pair for each descriptor

        Returns:
        descriptors : ndarray (num of descriptors, 3780)
        locations : ndarray (num of descriptors, 2)
        """
        descriptors = []
        locations = []
        height, width, _ = image.shape
        i = 0
        while i <= height - CustomHogDetector.detection_height:
            j = 0
            while j <= width - CustomHogDetector.detection_width:
                patch = image[
                    i : i + CustomHogDetector.detection_height,
                    j : j + CustomHogDetector.detection_width,
                    :,
                ]
                descriptor = self.hog.compute(patch)
                descriptors.append(descriptor)
                locations.append([i, j])
                j += CustomHogDetector.window_stride
            i += CustomHogDetector.window_stride
        descriptors = np.array(descriptors)
        locations = np.array(locations, dtype=np.int64)
        return descriptors, locations

    def create_feature_pyramid(self, image):
        """
        Create pyramid of HOG descriptors.

        Returns:
        ftr_pyramid : list of descriptors for each scaled image
        location_pyramid : list of descriptor locations for each scaled image
        """
        pyramid = self.create_pyramid(image)
        ftr_pyramid, location_pyramid = [], []
        for im in pyramid:
            features, locations = self.slide_window(im)
            ftr_pyramid.append(features)
            location_pyramid.append(locations)
        return ftr_pyramid, location_pyramid

    def detect(self, image, nms=True):
        """
        Perform multi-scale human detection using HOG Descriptors

        Returns:
        detections : ndarray (num of detections, (x, y, w, h))
        conf_scores : ndarray (num of detections, )
        """
        ftr_pyramid, location_pyramid = self.create_feature_pyramid(image)
        # keep the predictions of each scale along with their location and confidence score
        predictions = []
        for i, (features, locations) in enumerate(zip(ftr_pyramid, location_pyramid)):
            pred = self.svm.predict(features, flags=cv2.ml.StatModel_RAW_OUTPUT)[1]
            pred = (
                -pred
            )  # for some reason, the positives have negative distance to the margin
            loc = locations[np.flatnonzero(pred > CustomHogDetector.hit_threshold)]
            conf_scores = pred[pred > CustomHogDetector.hit_threshold]
            if loc.size:
                predictions.append((i, loc, conf_scores))

        detections = []
        conf_scores = []
        # normalize (flatten) the predictions at different scale
        for i, loc, scores in predictions:
            scale = CustomHogDetector.scaleFactors**i
            loc = loc * scale
            wh = CustomHogDetector.detection_height * scale
            ww = CustomHogDetector.detection_width * scale
            for j, l in enumerate(loc):
                detections.append([l[1], l[0], ww, wh])  # x y width height
                conf_scores.append(scores[j])
        detections = np.array(detections, dtype=np.int64)
        conf_scores = np.array(conf_scores)

        if nms:
            conf_scores_copy = conf_scores.copy()
            candidate_idxs = list(range(detections.shape[0]))
            survivor_idxs = []
            removed_idxs = []
            while len(survivor_idxs) + len(removed_idxs) < len(candidate_idxs):
                best_idx = conf_scores_copy.argmax()
                survivor_idxs.append(best_idx)
                conf_scores_copy[best_idx] = 0
                for idx in candidate_idxs:
                    if idx != best_idx and idx not in removed_idxs:
                        overlap = self.overlap(detections[best_idx], detections[idx])
                        if overlap >= CustomHogDetector.overlap_threshold:
                            removed_idxs.append(idx)
                            conf_scores_copy[idx] = 0

            detections = detections.take(survivor_idxs, axis=0)
            conf_scores = conf_scores.take(survivor_idxs)

        return detections, conf_scores

    def overlap(self, d1, d2):
        """Compute Intersection over Union"""
        x1, y1, w1, h1 = d1
        x2, y2, w2, h2 = d2
        x1_intersect = max(x1, x2)
        y1_intersect = max(y1, y2)
        x2_intersect = min(x1 + w1, x2 + w2)
        y2_intersect = min(y1 + h1, y2 + h2)
        intersect_area = max(0, x2_intersect - x1_intersect + 1) * max(
            0, y2_intersect - y1_intersect + 1
        )
        d1_area = w1 * h1
        d2_area = w2 * h2
        iou = intersect_area / (d1_area + d2_area - intersect_area)
        return iou
