# -*- coding: utf-8 -*-
"""
@author: xinyu
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F



# face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(BASE_DIR, "images/dlMike")

# shape 0 depends on how many images for training
X_tr = np.empty([117, 250001])
current_id = 0
label_ids = {}
current_row = 0
size = (500, 500)

for root, dirs, files in os.walk(img_dir):
    for file in files:        
        if file.endswith("PNG") or file.endswith("jpg"):
            path = os.path.join(root, file)
            pil_image = Image.open(path).convert("L")  # "L" turns it into grayscale            
                        
            final_img = pil_image.resize(size, Image.ANTIALIAS)    
            
            img_array = np.array(final_img)                     
            
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            
            # faces = face_cascade.detectMultiScale(img_array, scaleFactor = 1.5, minNeighbors = 5)
            
            # for (x,y,w,h) in faces:
                # roi = img_array[y:y+h, x:x+w]
                # roi_flat = roi.flatten()
                # X_tr[current_row, :] = np.array(list([id_]) + list(roi_flat), np.uint8)
            X_tr[current_row, :] = np.array(list([id_]) + list(img_array.flatten()), np.uint8)
            current_row += 1 
      
# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper-parameters
num_epoch = 10
batch_size = 1
lr = 0.001

# conv net
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 11, 1, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 40, 11, 1, 5)
        self.conv3 = nn.Conv2d(40, 80, 11, 1, 5)
        self.pool2 = nn.MaxPool2d(5, 5)
        self.conv4 = nn.Conv2d(80, 320, 11, 1, 5)
        self.conv8 = nn.Conv2d(320, 640, 11, 1, 5)
        # Here is a link for nn.Linear()
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # for FC layer demo, see Lec12. P.26
        self.fc1 = nn.Linear(640 * 5 * 5, 320)
        self.fc2 = nn.Linear(320, 160)
        self.fc3 = nn.Linear(160, 80)
        self.fc4 = nn.Linear(80, 10)
        self.fc5 = nn.Linear(10, 2)
    
    def forward(self, x):
        out = self.pool1(F.relu(self.conv1(x)))
        out = self.pool1(F.relu(self.conv2(out)))
        out = self.pool2(F.relu(self.conv3(out)))
        out = self.pool2(F.relu(self.conv4(out)))
        out = F.relu(self.conv8(out))
        # see above link for the shape details about the input & output
        out = out.view(-1, 640 * 5 * 5)   # should be batch_size * __
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = self.fc5(out)
        
        return out
        # y_h = torch.sigmoid(out)       
        # return y_h
    
model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
# criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(num_epoch):
    for i in range(X_tr.shape[0]):
        images = torch.from_numpy(X_tr[i,1:].reshape(1, 1, 500, 500)).float().to(device)
        labels = torch.from_numpy(X_tr[i,[0]]).long().to(device)
        # Forward pass
        outputs = model(images)

        loss = criterion(outputs, labels)
        # print(loss)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
print(label_ids)        
torch.save(model, "torchModel.pth")           
            
            
            
            
