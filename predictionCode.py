# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 13:41:19 2020

@author: Colin Cumming
"""

from keras.models import load_model
import cv2
import numpy as np
import os
import time

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
videos = ['mp4', 'mov']
images = ['jpeg', 'jpg', 'png']
int_to_char = dict((i, c) for i, c in enumerate(ALPHABET)) #int to char dictionary

path_to_image = "C:/QMIND_19_20/flask_uploads"

model = load_model('CNNmodel.h5')
print(model.layers[0].input_shape)


#this code is as an exmaple of getting the image/video from the front end
empty = True
while(empty == True):
    if len(os.listdir(path_to_image)) != 0:
        print("Not empty")
        empty=False
#once a video is received, the image/video is opened
time.sleep(1)    
input_file = path_to_image + '/' + os.listdir(path_to_image)[0]
print(input_file)
string = input_file.split('.')

classes = []
guesses = [0]*26
frame_count = 0
guess = 0


if (string[1] in videos):
    cap = cv2.VideoCapture(input_file)
    if (cap.isOpened() == False):
        print("Error")
    while (cap.isOpened()):
        while frame_count < 75: #will need to play around with this value, currently set to 2.5 seconds at 30 fps
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, (50,50))
                frame = np.reshape(frame, [1,50,50,1])
                guess = (model.predict_classes(frame)[0]) #get guess value 0-25
                guesses[guess] = guesses[guess] + 1 #increment guess count
                frame_count = frame_count + 1
        classes.append(np.argmax(guesses))
        guesses = [0]*26 #reset guesses
        frame_count = 0
    cap.release()
elif(string[1] in images):   
    img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(50,50))
    img = np.reshape(img, [1,50,50,1])
    print(img.shape)
    print(model.predict_classes(img)[0])
    classes.append(model.predict_classes(img)[0])
else:
    print("incorrect file format")

labels = [int_to_char[x] for x in classes]

print(labels)

outputString = ''.join(str(e) for e in labels)

with open("C:/QMIND_19_20/inputText/output.txt", "w") as textFile:
    textFile.write(outputString)

