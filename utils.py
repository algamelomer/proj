import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import random
import tensorflow as tf

from tensorflow.keras.models import Sequential                          #type: ignore
from tensorflow.keras.layers import Convolution2D, Flatten, Dense       #type: ignore
from tensorflow.keras.optimizers import Adam                            #type: ignore

def getName(filePath):
    return os.path.basename(filePath)

def importDataInfo(path):
    columns = ['Center', 'Steering', 'Throttle']
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names=columns)
    data['Center'] = data['Center'].apply(getName)
    print('Total Images Imported', data.shape[0])
    return data

def balanceData(data, display=True):
    nBin = 31
    samplesPerBin = 500
    hist, bins = np.histogram(data['Steering'], nBin)
    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.06)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
        plt.show()

    removeindexList = []
    for j in range(nBin):
        binDataList = []
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j + 1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeindexList.extend(binDataList)

    print('Removed Images:', len(removeindexList))
    data.drop(data.index[removeindexList], inplace=True)
    print('Remaining Images:', len(data))
    if display:
        hist, _ = np.histogram(data['Steering'], (nBin))
        plt.bar(center, hist, width=0.06)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
        plt.show()

    return data

def loadData(path, data):
    imagesPath = []
    steerings = []
    throttles = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        imagesPath.append(f'{path}/IMG/{indexed_data.iloc[0]}')
        steerings.append(float(indexed_data.iloc[1]))
        throttles.append(float(indexed_data.iloc[2]))

    imagesPath = np.asarray(imagesPath)
    steerings = np.asarray(steerings)
    throttles = np.asarray(throttles)
    return imagesPath, steerings, throttles

def augmentImage(imgPath, steering, throttle):
    img = mpimg.imread(imgPath)
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.2, 1.2))
        img = brightness.augment_image(img)
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
        throttle = -throttle
    return img, steering, throttle

def preProcess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img

def batchGen(imagesPath, dataList, batchSize, trainFlag):
    while True:
        imgBatch = []
        labelBatch = []

        for i in range(batchSize):
            index = random.randint(0, len(imagesPath) - 1)
            if trainFlag:
                img, steering, throttle = augmentImage(imagesPath[index], dataList[index][0], dataList[index][1])
            else:
                img = mpimg.imread(imagesPath[index])
                steering, throttle = dataList[index]
            img = preProcess(img)
            imgBatch.append(img)
            labelBatch.append([steering, throttle])
        
        yield np.asarray(imgBatch), np.asarray(labelBatch)


def createModel():
    model = Sequential()

    model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))

    model.add(Dense(2))  # Two outputs: steering and throttle

    model.compile(Adam(learning_rate=0.0001), loss='mse')
    return model
