import os
import random

import cv2 as cv
import numpy as np

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


my_svm_filename = (
    "../my_pretrained_svm.dat"  # the file to which you save the trained svm
)

# data paths
test_images_1 = "data/task_1_testImages/"
path_train_2 = "data/task_2_3_data/train/"
path_test_2 = "data/task_2_3_data/test/"

# ***********************************************************************************
# draw a bounding box in a given image
# Parameters:
# im: The image on which you want to draw the bounding boxes
# detections: the bounding box of the detections (people)
# returns None


def drawBoundingBox(im, detections):
    for x, y, w, h in detections:
        cv.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)


def task1():
    print("Task 1 - OpenCV HOG")

    # Load images
    filelist = os.path.join(test_images_1, "filenames.txt")
    filenames = open(filelist).read().splitlines()
    images = []
    for filename in filenames:
        filename = os.path.join(test_images_1, os.path.basename(filename))
        im = cv.imread(filename)
        images.append(im)

    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

    for im in images:
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

    # Load image names

    filelist_train_pos = os.path.join(path_train_2, "filenamesTrainPos.txt")
    filelist_train_neg = os.path.join(path_train_2, "filenamesTrainNeg.txt")

    hog = cv.HOGDescriptor()

    filenames = open(filelist_train_pos, "r").read().splitlines()
    w, h = 64, 128
    pos_features = []
    for filename in filenames:
        filename = os.path.join(path_train_2, "pos", filename)
        im = cv.imread(filename)
        rows, cols, _ = im.shape
        cropped = im[
            rows // 2 - h // 2 : rows // 2 + h // 2,
            cols // 2 - w // 2 : cols // 2 + w // 2,
            :,
        ]
        pos_features.append(hog.compute(cropped))
    pos_features = np.array(pos_features)
    pos_labels = np.ones(pos_features.shape[0], dtype=np.int32)

    neg_features = []
    filenames = open(filelist_train_neg, "r").read().splitlines()
    for filename in filenames:
        filename = os.path.join(path_train_2, "neg", filename)
        im = cv.imread(filename)
        rows, cols, _ = im.shape
        ys = np.random.randint(0, rows - h, size=10)
        xs = np.random.randint(0, cols - w, size=10)
        for x, y in zip(xs, ys):
            neg_features.append(hog.compute(im[y : y + h, x : x + w, :]))
    neg_features = np.array(neg_features)
    neg_labels = np.ones(neg_features.shape[0], dtype=np.int32) * -1

    features = np.concatenate((pos_features, neg_features), axis=0)
    labels = np.concatenate((pos_labels, neg_labels), axis=0)

    features.dump(train_hog_path)
    labels.dump(train_labels_path)


def task3():
    print("Task 3 - Train SVM and predict confidence values")
    # TODO Create 3 SVMs with different C values, train them with the training data and save them
    # then use them to classify the test images and save the results

    filelist_testPos = path_test_2 + "filenamesTestPos.txt"
    filelist_testNeg = path_test_2 + "filenamesTestNeg.txt"

    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

    train_hog_features = np.load(train_hog_path, allow_pickle=True)
    train_labels = np.load(train_labels_path, allow_pickle=True)
    svm.train(train_hog_features, cv.ml.ROW_SAMPLE, train_labels)


def task5():

    print("Task 5 - Eliminating redundant Detections")

    # TODO: Write your own custom class myHogDetector
    # Note: compared with the previous tasks, this task requires more coding

    my_detector = Custom_Hog_Detector(my_svm_filename)

    # TODO Apply your HOG detector on the same test images as used in task 1 and display the results

    print("Done!")
    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == "__main__":

    # Task 1 - OpenCV HOG
    # task1()

    # Task 2 - Extract HOG Features
    # task2()

    # Task 3 - Train SVM
    task3()

    # Task 5 - Multiple Detections
    # task5()
