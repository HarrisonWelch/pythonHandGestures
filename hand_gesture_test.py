# hand_gesture_test.py
# Harrison Welch

import numpy as np
import cv2

if __name__ == "__main__":
  print("cv2 version = " + str(cv2.__version__))
  print("np version = " + str(np.__version__))
  print("MAIN STUFF")

  while(True):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
      img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
      roi_gray = gray[y:y+h, x:x+w]
      roi_color = img[y:y+h, x:x+w]
      eyes = eye_cascade.detectMultiScale(roi_gray)
      for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

  cv2.imshow('img',img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

# print("HEY THIS WORKS")