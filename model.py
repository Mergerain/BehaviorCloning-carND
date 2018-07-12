#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 15:37:36 2018

@author: wei
"""

import csv
import cv2
import numpy as np
from scipy.misc import imread

lines = []
with open ('/Users/wei/Desktop/Three-Loops/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
correction = 0.2
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename  = source_path.split('/')[-1]
        current_path = '/Users/wei/Desktop/Three-Loops/IMG/'+filename
        image = imread(current_path).astype(np.float32)
        images.append(image[80:140][:])
        measurement = float(line[3])
        if i ==0:
            measurements.append(measurement)
        if i ==1:
            measurements.append(measurement+correction)
        if i ==2:
            measurements.append(measurement-correction)
#create the array for both original images and the floped images
#X_train = np.array(images)
#y_train = np.array(measurements)

augmented_images, augmented_measurements = [],[]
for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras import optimizers
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D,Dropout

model = Sequential()
model.add(Lambda(lambda x:x/255.0-0.5, input_shape=(60,320,3)))
model.add(Convolution2D(12,5,5,activation = "relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(24,5,5,activation = "relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(48,5,5,activation = "relu"))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer='adam')
model.fit(X_train,y_train,validation_split = 0.2,shuffle = True,nb_epoch=2)
model.save('model.h5')