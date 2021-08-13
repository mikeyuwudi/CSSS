# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 14:56:00 2021

@author: xinyu
"""
import os
import numpy as np
from PIL import Image
import cv2
import pickle

# Wherever this current file is saved, looking for where that path is on system
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(BASE_DIR, "images/cvMike")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

# face recognizer
recog = cv2.face.LBPHFaceRecognizer_create()

X, Y = [], []
current_id = 0
label_ids = {}

# each image file can have any name, and the label is just the folder name
# https://www.geeksforgeeks.org/os-walk-python/
for root, dirs, files in os.walk(img_dir):
    for file in files:
        if file.endswith("PNG") or file.endswith("jpg"):
            path = os.path.join(root, file)

            #label = os.path.basename(root).replace(" ", "-").lower()
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
                        
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]    
            
            pil_image = Image.open(path).convert("L")  # "L" turns it into grayscale
            
            # resize images for training (perhaps not the best way)
            size = (500, 500)
            final_img = pil_image.resize(size, Image.ANTIALIAS)
            
            img_array = np.array(final_img, "uint8")
            
            faces = face_cascade.detectMultiScale(img_array, scaleFactor = 1.5, minNeighbors = 5)
            
            for (x,y,w,h) in faces:
                roi = img_array[y:y+h, x:x+w]
                X.append(roi)
                Y.append(id_)

            # X.append(img_array)
            # Y.append(id_)
# print(label_ids)

with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)      

print(len(X), len(Y))

# recog.train(X, np.array(Y)) 
# recog.save("trainner.yml")           
            
            
            
            
            
            
            
            
            
            
            
            
            