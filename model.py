import os
import csv

samples = []
# Use this with data provided by Udacity
with open('./data.bak/driving_log.csv') as csvfile:
# Use this for own data
#with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, load_img
# Multiply by 3 because we get data from 3 different angles
train_samples, validation_samples = train_test_split(samples*3, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=256):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:				
				for i in range (3):
					# Use this with data provided by Udacity
					name = './data.bak/IMG/'+batch_sample[i].split('/')[-1]
					# Use this for own data
					#name = batch_sample[i].split('/')[-1]
					image = load_img(name)
					image = img_to_array(image)
					images.append(image)
					image_flipped = np.fliplr(image)
					images.append(image_flipped)
				center_angle = float(batch_sample[3])
				correction = 0.2
				left_angle = center_angle + correction
				right_angle = center_angle - correction
				angles.append(center_angle)
				angles.append(-center_angle)
				angles.append(left_angle)
				angles.append(-left_angle)
				angles.append(right_angle)
				angles.append(-right_angle)

			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

ch, row, col = 3, 80, 320  # Trimmed image format

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5)

model.save('model.h5')