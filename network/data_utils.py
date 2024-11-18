import pickle

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import rcParams


# функція для завантаження даних для тренування/тестування мережі
def get_raw_data_set(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f, encoding='latin1')


# функція для попередньої обробки даних
# та приведення їх у необхідний формат для навчання мережі
def preprocess_data(raw_data, num_class):
    # дістаємо дані
    x_data = raw_data['data']
    # перетворюємо дані
    x_data = x_data.reshape(len(x_data), 3, 32, 32).transpose(0, 2, 3, 1)
    # Зміна масштабу шляхом ділення кожного пікселя зображення на 255
    x_data = x_data / 255.
    # дістаємо конкретні надписи (ярлики)
    y_data = raw_data['fine_labels']
    # перетворюємо написи
    y_data = keras.utils.to_categorical(y_data, num_class)
    return x_data, y_data


def exploring_data(trainData, testData, metaData):
    # type of items in each file
    for item in trainData:
        print(item, type(trainData[item]))
    print(len(trainData['data']))
    print(len(trainData['data'][0]))
    # There are 50000 images in the training dataset and each image is a 3 channel 32 * 32 pixel image (32 * 32 * 3 = 3072).
    print(np.unique(trainData['fine_labels']))
    # There are 100 different fine labels for the images (0 to 99).
    print(np.unique(trainData['coarse_labels']))
    # There are 10 different coarse labels for the images (0 to 9).
    print(trainData['batch_label'])
    print(len(trainData['filenames']))
    # Meta file has a dictionary of fine labels and coarse labels.
    # storing coarse labels along with its number code in a dataframe
    category = pd.DataFrame(metaData['coarse_label_names'], columns=['SuperClass'])
    # The above list shows coarse label number and name, which we are denoting as categories.
    # storing fine labels along with its number code in a dataframe
    subCategory = pd.DataFrame(metaData['fine_label_names'], columns=['SubClass'])
    # The above list shows fine label number and name, which we are denoting as subcategories.
    X_train = trainData['data']
    ### Image Transformation for Tensorflow (Keras) and Convolutional Neural Networks
    # 4D array input for building the CNN model using Keras
    X_train = X_train.reshape(len(X_train), 3, 32, 32).transpose(0, 2, 3, 1)
    # ## Exploring the Images in the Dataset
    # generating a random number to display a random image from the dataset along with the label's number and name
    rcParams['figure.figsize'] = 2, 2
    imageId = np.random.randint(0, len(X_train))
    plt.imshow(X_train[imageId])
    plt.axis('off')
    print("Image number selected : {}".format(imageId))
    print("Shape of image : {}".format(X_train[imageId].shape))
    print("Image category number: {}".format(trainData['coarse_labels'][imageId]))
    print("Image category name: {}".format(category.iloc[trainData['coarse_labels'][imageId]][0].capitalize()))
    print("Image subcategory number: {}".format(trainData['fine_labels'][imageId]))
    print("Image subcategory name: {}".format(subCategory.iloc[trainData['fine_labels'][imageId]][0].capitalize()))
    # 16 random images to display at a time along with their true labels
    rcParams['figure.figsize'] = 8, 8
    num_row = 4
    num_col = 4
    # to get 4 * 4 = 16 images together
    imageId = np.random.randint(0, len(X_train), num_row * num_col)
    fig, axes = plt.subplots(num_row, num_col)
    plt.suptitle('Images with True Labels', fontsize=18)
    for i in range(0, num_row):
        for j in range(0, num_col):
            k = (i * num_col) + j
            axes[i, j].imshow(X_train[imageId[k]])
            axes[i, j].set_title(subCategory.iloc[trainData['fine_labels'][imageId[k]]][0].capitalize())
            axes[i, j].axis('off')
