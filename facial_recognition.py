import cv2
import numpy as np
import face_recognition

anushka_img = face_recognition.load_image_file('images/an.jpeg')
anushka_img = cv2.cvtColor(steve_img,cv2.COLOR_BGR2RGB)
test_img = face_recognition.load_image_file('images/ans.jpeg')
test_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(anushka_img)[0]
encodeAnushka = face_recognition.face_encodings(anushka_img)[0]
print(faceLoc)
cv2.rectangle(anushka_img,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(test_img)[0]
encodeTest = face_recognition.face_encodings(test_img)[0]
cv2.rectangle(test_img,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeAnushka],encodeTest)
faceDis = face_recognition.face_distance([encodeAnushka],encodeTest)
print(results,faceDis)
cv2.putText(test_img,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)

cv2.imshow('Anushka Raj',anushka_img)
cv2.imshow('Anushka Test',test_img)
cv2.waitKey(0)
