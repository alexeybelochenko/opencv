import numpy as np
import matplotlib.pyplot as plt 
import os
import cv2
import random, pickle

PATH = os.getcwd()
data_path = PATH + '/dataset'
data_dir_list = os.listdir(data_path)
IMG_SIZE = 128

num_classes = len([name for name in os.listdir(data_path)])

CATEGORIES = []
for dataset in data_dir_list:
    img_list = os.path.join(data_path + '/',dataset)
    CATEGORIES.append(dataset)
    for img in os.listdir(img_list):
        img_array = cv2.imread(os.path.join(img_list, img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap='gray')


training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(data_path + '/',dataset)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(img_list):
            img_array = cv2.imread(os.path.join(img_list, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])

create_training_data()
random.shuffle(training_data)

X = []
Y = []

for features, label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_out = open('x.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('y.pickle', 'wb')
pickle.dump(Y, pickle_out)
pickle_out.close()