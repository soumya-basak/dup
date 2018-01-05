import csv
import cv2
import numpy as np
import pandas as pd
import os

lines = []
with open('../data/driving_log.csv') as csvFile:
	reader = csv.reader(csvFile)
	
	for line in reader:
		lines.append(line)
		
images = []
measurements = []
for line in lines:
	for i in range(3):
		source_path = line[0]
		filename = source_path.split('/')[-1]
		current_path = '../data/IMG/' + filename
		image = cv2.imread(current_path)
		images.append(image)
		measurement = float(line[3])
		measurements.append(measurement)
	
augmented_images = []
augmented_measurements = []

for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image,1)
    flipped_measurement = measurement * -1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
print(X_train.shape)
print(y_train.shape)
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
#from keras.models import load_model

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
#model.add(Flatten(input_shape=(160,320,3)))

model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
#print(X_train[0])

model.compile(loss='mse',optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle= True, nb_epoch=5)

model.save('model.h5')

	 
