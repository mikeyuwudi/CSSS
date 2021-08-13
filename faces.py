# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 03:00:11 2021

@author: xinyu
"""
import numpy as np
import cv2
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import smtplib
from email.message import EmailMessage
import time

# email config
msg = EmailMessage()
account = "example@gmail.com"
password = "passcode"
msg['Subject'] = 'TEST SUBJECT'
msg['From'] = account
msg['To'] = account

server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
server.login(account, password)



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
        # return y_h.reshape(1)

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
# eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
# smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

recog = cv2.face.LBPHFaceRecognizer_create()
recog.read("trainner.yml")

labels = {}
with open("labels.pickle", "rb") as f:
    labels = pickle.load(f)
    labels = {v:k for k,v in labels.items()}

cap = cv2.VideoCapture(0)

size = (500, 500)
model = torch.load("./torchModel.pth")
model.eval()
# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

color_CV = (255, 255, 255)
color_DL = (255, 0, 0)
font = cv2.FONT_HERSHEY_SIMPLEX
stroke = 2

DL = True
EMAIL = False 

sample, correct = 0, 0
while True:                           
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # frame (and gray) is already a NumPy array
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
    for (x, y, w, h) in faces:   
        sample += 1
        # region of interest
        roi_gray = gray[y:y+h, x:x+w]  # (ycord_start, ycord_end), same thing for x
        # roi_color = frame[y:y+h, x:x+w]
        
        # recognizer using cv2        
        if not DL:
            id_, conf = recog.predict(roi_gray)
            print(conf)
            if conf <= 95 and conf >= 75:
                # print(id_, labels[id_])                
                name = 'mike_cv'  
            else: 
                name = "unknown"
                # 1 stands for font size
            # if name == 'cvmike': correct += 1    
            # if name == 'unknown': correct += 1
            cv2.putText(frame, name, (x,y), font, 1, color_CV, stroke, cv2.LINE_AA) 
        else:
            face_r = cv2.resize(roi_gray, size) 
            
            img = torch.from_numpy(face_r.reshape(1,1,500,500)).float().to(device)
            outputs = model(img)
            # _, predicted = torch.max(outputs, 1)
                     
            print(outputs) 
            if outputs[0][1].item() - outputs[0][0].item() <= 6.5:
                name = 'mike_torch'
            else:
                name = 'unknown'
            cv2.putText(frame, name, (x,y), font, 1, color_DL, stroke, cv2.LINE_AA)
            
        if EMAIL:
            msg.set_content(name + " just saw ur screen at " + time.ctime())
            server.send_message(msg)
            EMAIL = False
            time.sleep(300)
            EMAIL = True      
        
        # draw a rectangle
        color = (0, 0, 255) # BGR 0-255
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        
        # subitems = smile_cascade.detectMultiScale(roi_gray)
        # for (subx, suby, subw, subh) in subitems:
            # cv2.rectangle(roi_color, (subx, suby), (subx+subw, suby+subh), (0,255,0), 2)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'): break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

server.quit()
print(f'accu: {correct*100/sample}%')
