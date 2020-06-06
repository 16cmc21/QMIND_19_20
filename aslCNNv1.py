# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 19:54:05 2019

@author: Colin Cumming
"""
#image processing
import numpy as np
import cv2 
import os
from random import shuffle
from tqdm import tqdm

#for data manipulation
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split

#for CNN
import keras
from keras import optimizers
from keras.models import Sequential 
from keras.layers import Conv2D,MaxPooling2D, Dense,Flatten, Dropout
from keras.utils import np_utils
from keras.optimizers import SGD

#for gpu support for training model
from tensorflow.python.client import device_lib
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

TRAIN_DIR = 'C:/QMIND_19_20/training_images'
TEST_DIR = 'C:/QMIND_19_20/testing_images'

IMG_SIZE = 50
LR = 1e-3

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

num_to_alpha = dict(zip(range(0,26), ALPHABET))


MODEL_NAME = 'asl-{}-{}.model'.format(LR, '6conv-basic')

char_to_int = dict((c, i) for i, c in enumerate(ALPHABET))
int_to_char = dict((i, c) for i, c in enumerate(ALPHABET))


def label_img(img):
    word_label = img.split('.')[0]
    word_label = word_label.rstrip('0123456789')
    label = [char_to_int[char] for char in word_label] #turn letter into a number representation
    onehot_encoded = list()
    for value in label:
    	letter = [0 for _ in range(len(ALPHABET))]
    	letter[value] = 1 #1 for letter, 0 for all other letters, creating a one-hotted array letter representation
    	onehot_encoded.append(letter)
    return onehot_encoded
    #provides a label onehotted for letters of the alphabet
    
def create_train_data(): #function used for reading in images, processing into numpy arrays (OpenCV) and adding image labels
    training_data=[]
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data) #shuffle before returning to mix up data, not reading in all A, then all B, ...
    np.save('train_data.npy', training_data)
    return training_data

def process_test_data(): #function used to process test data, dont need to label
    testing_data = []
    img_num = 0
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        img_num = img_num + 1
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

#train_data = create_train_data()
#test_data = process_test_data()

train_data = np.load('train_data.npy')
test_data = np.load('test_data.npy')

#scikit learn train test split is used to split up data into two datasets, 75% for training, 25% for testing
train, test = train_test_split(train_data, test_size=0.25, random_state=42)

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #get image arrays from training data
Y = [i[1] for i in train] #get image labels from training data
Y = np.concatenate(Y, axis=0)
test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #same as above but for testing data
test_y = [i[1] for i in test] 
test_y = np.concatenate(test_y, axis=0)

''' Beginning of CNN layer definition '''
classifier = Sequential()
classifier.add(Conv2D(filters=32, kernel_size=(3,3),strides=(1,1),padding='same', #note: input shape is shape of image, 1D since greyscale
                      input_shape=(IMG_SIZE,IMG_SIZE,1),activation='relu', 
                      data_format='channels_last'))

classifier.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Dense(1024, activation = 'relu')) #create fully connected CNN layer
classifier.add(Dropout(0.5))
classifier.add(Flatten()) #reduce to 1D to allow 1x26 array output
classifier.add(Dense(26, activation = 'softmax')) #connected layer with 26 outputs, for alphabet
''' End of CNN layer definition '''

classifier.summary()

#compile CNN layers and train model
sgd = optimizers.SGD(lr=LR, decay=1e-6, momentum=0.9, nesterov=True) #hyperparameters for sgd optimizer, could try out adam opt as well
classifier.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
print("[INFO] training model...")
history = classifier.fit(X, Y, validation_data=(test_x, test_y), epochs=3, batch_size=8, verbose=1)
accuracy = classifier.evaluate(x=test_x,y=test_y,batch_size=8)
print("Accuracy: ",accuracy[1])
classifier.save('CNNmodel.h5') #save model for later use, will be called in to program for using webcam and doing live processing


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, color='red', label='Training Loss')
plt.plot(epochs, val_loss, color='green', label='Validation Loss')
plt.title('Model Loss Values')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, color='red', label='Training acc')
plt.plot(epochs, val_acc, color='green', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()




        
    
    

    

    
    