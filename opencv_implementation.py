#import required libraries
import keras
from keras.models import load_model
import cv2
import numpy as np
import operator
model=load_model("path/to/your/directory/which/has/model-file/my_model1.h5")

#initialise the camera 
cap=cv2.VideoCapture(0)

#declaring the categorical output
categories={0:"call_me",1:"finger_cross",2:"okay",3:"paper",4:"peace",5:"rock",6:"rock_on",7:"scissor",8:"thumbs",9:"up"}

#initialise infinite loop
while True:
       _,frame=cap.read() #reading video from a camera
       frame=cv2.flip(frame,1) 
       x1=int(0.5*frame.shape[1]) #to create a ROI on the screen
       y1=10
       x2=frame.shape[1]-10
       y2=int(0.5*frame.shape[1])

       cv2.rectangle(frame,(x1-1,y1-1),(x2-1,y2-1),(255,0,0),1)
       roi=frame[y1:y2,x1:x2]
       roi=cv2.resize(roi,(240,200)) #resize the image
       roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY) #converting BGR to gray scale image
       _,test_image=cv2.threshold(roi,120,255,cv2.THRESH_BINARY_INV) #binary inverse thresholded image
       cv2.imshow("test",test_image) 
       result=model.predict(test_image.reshape(1,200,240,1)) #making prediction from the loaded model
      
       prediction={
                    "call_me":result[0][0],   
                    "finger_cross":result[0][1],
                    "okay":result[0][2],
                    "paper":result[0][3],
                    "peace":result[0][4],
                    "rock":result[0][5],
                    "rock_on":result[0][6],
                    "scissor":result[0][7],
                    "thumbs":result[0][8],
                    "up":result[0][9],
           }
       prediction=sorted(prediction.items(),key=operator.itemgetter(1),reverse=True) #sorting of prediction on the basis of max accuracy
       cv2.putText(frame,prediction[0][0],(x1+100,y2+30),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),3) #showning text on the screen
       cv2.imshow("Frame",frame)

       key=cv2.waitKey(10)
       if key & 0xFF == 27: #press esc for break
           break

cap.release() #switch off camera
cv2.destroyAllWindows() #destroy camera windows
