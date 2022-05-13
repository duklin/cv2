# TODO: This class should which Implements the following functionality:
# - opencv HOGDescriptor combined with a sliding window
# - perform detection at multiple scales, i.e. you need to scale the extracted patches when performing the detection
# - non maximum suppression: eliminate detections using non-maximum-suppression based on the overlap area


class CustomHogDetector:

    # Some constants that you will be using in your implementation
    detection_width = 64  # the crop width dimension
    detection_height = 128  # the crop height dimension
    window_stride = 32  # the stride size
    scaleFactors = [
        1.2
    ]  # scale each patch down by this factor, feel free to try other values
    # You may play with different values for these two theshold values below as well
    hit_threshold = 0  # detections above this threshold are counted as positive.
    overlap_threshold = 0.3  # if the overlap between two detections is above this threshold, eliminate the one with the lower confidence score.

    def __init__(self, trained_svm_name):
        # load the trained SVM from file trained_svm_name
        pass
