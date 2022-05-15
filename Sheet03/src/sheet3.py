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
train_hog_path = "../train_hog_descs.dat"  # the file to which you save the HOG descriptors of every patch
train_labels = (
    "../labels_train.dat"  # the file to which you save the labels of the training data
)
my_svm_filename = (
    "../my_pretrained_svm.dat"  # the file to which you save the trained svm
)

# data paths
test_images_1 = "data/task_1_testImages/"
path_train_2 = "../task_2_3_Data/01Train/"
path_test_2 = "../task_2_3_Data/02Test/"

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
    filenames = open(filelist).readlines()
    images = []
    for filename in filenames:
        filename = os.path.join(
            test_images_1, os.path.basename(filename).replace("\n", "")
        )
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

    filelist_train_pos = path_train_2 + "filenamesTrainPos.txt"
    filelist_train_neg = path_train_2 + "filenamesTrainNeg.txt"
    # TODO: Create a HOG descriptor object to extract the features from the set of positive and negative samples

    # positive samples: Get a crop of size 64*128 at the center of the image then extract its HOG features
    # negative samples: Sample 10 crops from each negative sample at random and then extract their HOG features
    # In total you should have  (x+10*y) training samples represented as HOG features(x=number of positive images, y=number of negative images),
    # save them and their labels in the path train_hog_path and train_labels in order to load them in section 3


def task3():
    print("Task 3 - Train SVM and predict confidence values")
    # TODO Create 3 SVMs with different C values, train them with the training data and save them
    # then use them to classify the test images and save the results

    filelist_testPos = path_test_2 + "filenamesTestPos.txt"
    filelist_testNeg = path_test_2 + "filenamesTestNeg.txt"


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
    task1()

    # Task 2 - Extract HOG Features
    # task2()

    # Task 3 - Train SVM
    # task3()

    # Task 5 - Multiple Detections
    # task5()
