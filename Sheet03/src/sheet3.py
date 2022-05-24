import os
import random

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from cv2 import HOGDescriptor

from custom_hog_detector import CustomHogDetector

# Global constants

# crop/patch dimensions for the training samples
width = 64
height = 128

num_negative_samples = 10  # number of negative samples per image
train_hog_path = "train_hog_descs.dat"  # the file to which you save the HOG descriptors of every patch
train_labels_path = (
    "labels_train.dat"  # the file to which you save the labels of the training data
)


my_svm_filename = "my_pretrained_svm.dat"  # the file to which you save the trained svm

# data paths
test_images_1 = "data/task_1_testImages"
path_train_2 = "data/task_2_3_data/train"
path_test_2 = "data/task_2_3_data/test"

# ***********************************************************************************
# draw a bounding box in a given image
# Parameters:
# im: The image on which you want to draw the bounding boxes
# detections: the bounding box of the detections (people)
# returns None


def drawBoundingBox(im, detections):
    for x, y, w, h in detections:
        cv.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)


def center_crop(im):
    rows, cols, _ = im.shape
    return im[
        rows // 2 - height // 2 : rows // 2 + height // 2,
        cols // 2 - width // 2 : cols // 2 + width // 2,
        :,
    ]


def compute_hog_descriptors(filenames, transform, num_patches=1):
    hog = cv.HOGDescriptor()
    descriptors = []
    for filename in filenames:
        im = cv.imread(filename)
        if transform == "center_crop":
            cropped = center_crop(im)
            descriptors.append(hog.compute(cropped))
        elif transform == "random_crop":
            rows, cols, _ = im.shape
            ys = np.random.randint(0, rows - height, size=num_patches)
            xs = np.random.randint(0, cols - width, size=num_patches)
            for x, y in zip(xs, ys):
                descriptors.append(hog.compute(im[y : y + height, x : x + width, :]))
    descriptors = np.array(descriptors)
    return descriptors


def task1():
    print("Task 1 - OpenCV HOG")

    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

    filelist = os.path.join(test_images_1, "filenames.txt")
    filenames = open(filelist).read().splitlines()
    for filename in filenames:
        filename = os.path.join(test_images_1, os.path.basename(filename))
        im = cv.imread(filename)
        detections, w = hog.detectMultiScale(
            im, winStride=(8, 8), padding=(32, 32), scale=1.05
        )
        if isinstance(detections, np.ndarray):
            drawBoundingBox(
                im, detections[w > 0.55]
            )  # empirically decided confidence value
        cv.imshow("task1", im)
        ch = cv.waitKey()
        if ch == 27:
            break

    cv.destroyAllWindows()


def task2():
    print("Task 2 - Extract HOG features")

    random.seed()
    np.random.seed()

    filelist_train_pos = os.path.join(path_train_2, "filenamesTrainPos.txt")
    filelist_train_neg = os.path.join(path_train_2, "filenamesTrainNeg.txt")

    # compute hog descriptors for positive train samples
    filenames = open(filelist_train_pos, "r").read().splitlines()
    filenames = [os.path.join(path_train_2, "pos", fn) for fn in filenames]
    pos_features = compute_hog_descriptors(filenames, transform="center_crop")
    pos_labels = np.ones(pos_features.shape[0], dtype=np.int32)

    # compute hog descriptors for negative train samples
    filenames = open(filelist_train_neg, "r").read().splitlines()
    filenames = [os.path.join(path_train_2, "neg", fn) for fn in filenames]
    neg_features = compute_hog_descriptors(
        filenames, transform="random_crop", num_patches=num_negative_samples
    )
    neg_labels = np.ones(neg_features.shape[0], dtype=np.int32) * -1

    # put features from pos and neg samples in one ndarray (same for labels)
    features = np.concatenate((pos_features, neg_features), axis=0)
    labels = np.concatenate((pos_labels, neg_labels), axis=0)

    features.dump(train_hog_path)
    labels.dump(train_labels_path)


def task3():
    print("Task 3 - Train SVM and predict confidence values")

    train_hog_features = np.load(train_hog_path, allow_pickle=True)
    train_labels = np.load(train_labels_path, allow_pickle=True)

    c_values = [0.01, 1, 100]

    # train and save the 3 SVMs
    for c in c_values:
        svm = cv.ml.SVM_create()
        svm.setType(cv.ml.SVM_C_SVC)
        svm.setKernel(cv.ml.SVM_LINEAR)
        svm.setC(c)
        svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
        svm.train(train_hog_features, cv.ml.ROW_SAMPLE, train_labels)
        svm.save(f"{c=}_{my_svm_filename}")

    # compute hog descriptors for positive test samples
    filelist_testPos = os.path.join(path_test_2, "filenamesTestPos.txt")
    test_pos_filenames = open(filelist_testPos, "r").read().splitlines()
    test_pos_filenames = [
        os.path.join(path_test_2, "pos", fn) for fn in test_pos_filenames
    ]
    test_pos_features = compute_hog_descriptors(
        test_pos_filenames, transform="center_crop"
    )

    # compute hog descriptors for negative test samples (10 patches per image)
    filelist_testNeg = os.path.join(path_test_2, "filenamesTestNeg.txt")
    test_neg_filenames = open(filelist_testNeg, "r").read().splitlines()
    test_neg_filenames = [
        os.path.join(path_test_2, "neg", fn) for fn in test_neg_filenames
    ]
    test_neg_features = compute_hog_descriptors(
        test_neg_filenames, transform="random_crop", num_patches=num_negative_samples
    )

    # load SVMs
    svm_ensemble = []
    for c in c_values:
        svm_ensemble.append(cv.ml.SVM_load(f"{c=}_{my_svm_filename}"))

    svm_ensemble[1].save(my_svm_filename)

    pos_predicted = np.zeros((test_pos_features.shape[0], 1))
    neg_predicted = np.zeros((test_neg_features.shape[0], 1))
    for svm in svm_ensemble:
        pos_predicted += svm.predict(test_pos_features)[1]
        neg_predicted += svm.predict(test_neg_features)[1]

    pos_predicted = np.sign(pos_predicted)
    neg_predicted = np.sign(neg_predicted)
    tp = np.count_nonzero(pos_predicted == 1)
    fp = np.count_nonzero(neg_predicted == 1)
    fn = np.count_nonzero(pos_predicted == -1)

    precision = tp / (fp + tp)
    recall = tp / (fn + tp)

    # write class predictions for every test sample
    with open("task3_test_results.txt", "w") as fout:
        for i, filename in enumerate(test_pos_filenames):
            fout.write(f"{filename}, {pos_predicted[i]}\n")

        for i, filename in enumerate(test_neg_filenames):
            neg_sample_pred = neg_predicted[
                i * num_negative_samples : (i + 1) * num_negative_samples
            ]
            fout.write(
                f"{filename}, {neg_sample_pred.reshape((num_negative_samples,))}\n"
            )

        fout.write(f"\nPrecision: {precision:.2}, Recall: {recall:.2}")

    # write confidence scores for every train sample
    filelist_train_pos = os.path.join(path_train_2, "filenamesTrainPos.txt")
    filelist_train_neg = os.path.join(path_train_2, "filenamesTrainNeg.txt")
    pos_train_filenames = open(filelist_train_pos, "r").read().splitlines()
    pos_train_filenames = [
        os.path.join(path_train_2, "pos", fn) for fn in pos_train_filenames
    ]
    neg_train_filenames = open(filelist_train_neg, "r").read().splitlines()
    neg_train_filenames = [
        os.path.join(path_train_2, "neg", fn) for fn in neg_train_filenames
    ]
    conf_scores = []
    for svm in svm_ensemble:
        conf_scores.append(
            svm.predict(train_hog_features, flags=cv.ml.StatModel_RAW_OUTPUT)[1]
        )
    conf_scores = np.array(conf_scores).squeeze()

    # the confidence scores for every positive train sample represent the raw output of the three different SVMs
    # because for every negative train sample there are 10 patches and 3 confidence scores per patch,
    # there are 30 confidence scores for every negative train sample
    pos_train_num = len(pos_train_filenames)
    with open("task3_train_results.txt", "w") as fout:
        for i, fn in enumerate(pos_train_filenames):
            fout.write(f"{fn}, {conf_scores[:,i].reshape(-1)}\n")
        for i, fn in enumerate(neg_train_filenames):
            fout.write(
                f"{fn}, {conf_scores[:,pos_train_num+i:pos_train_num+i+num_negative_samples].reshape(-1)}\n"
            )

    # from the plot we notice that for c=0.01, the distances to the margin is smaller compared to c=1 or c=100
    _, axes = plt.subplots(nrows=3, ncols=1, squeeze=True)
    for i, c in enumerate(c_values):
        pos_predicted = conf_scores[i, :pos_train_num]
        neg_predicted = conf_scores[i, pos_train_num:]
        axes[i].set_title(f"c={c}")
        axes[i].scatter(
            pos_predicted, np.zeros(pos_predicted.shape[0]), s=3, label=f"positive"
        )
        axes[i].scatter(
            neg_predicted, np.zeros(neg_predicted.shape[0]), s=3, label=f"negative"
        )
        axes[i].legend()

    plt.show()


def task5():

    print("Task 5 - Eliminating redundant Detections")

    # TODO: Write your own custom class myHogDetector
    # Note: compared with the previous tasks, this task requires more coding

    my_detector = CustomHogDetector(my_svm_filename)

    # TODO Apply your HOG detector on the same test images as used in task 1 and display the results

    filelist = os.path.join(test_images_1, "filenames.txt")
    filenames = open(filelist).read().splitlines()
    for filename in filenames:
        filename = os.path.join(test_images_1, os.path.basename(filename))
        im = cv.imread(filename)
        detections_no_nms, _ = my_detector.detect(im, nms=False)
        detections_nms, _ = my_detector.detect(im, nms=True)
        im_no_nms = im.copy()
        im_nms = im.copy()
        drawBoundingBox(im_no_nms, detections_no_nms)
        drawBoundingBox(im_nms, detections_nms)

        cv.imshow("Without NMS", im_no_nms)
        cv.imshow("With NMS", im_nms)
        ch = cv.waitKey()
        if ch == 27:
            break

    print("Done!")
    cv.destroyAllWindows()


if __name__ == "__main__":

    # Task 1 - OpenCV HOG
    task1()

    # Task 2 - Extract HOG Features
    task2()

    # Task 3 - Train SVM
    task3()

    # Task 5 - Multiple Detections
    task5()
