# hand_gesture_test.py
# Harrison Welch

import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# img = cv2.imread('sachin.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

if __name__ == "__main__":
  print("cv2 version = " + str(cv2.__version__))
  print("np version = " + str(np.__version__))
  print("MAIN STUFF")

  while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Our operations on the frame come here
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
      frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
      roi_gray = gray[y:y+h, x:x+w]
      roi_color = frame[y:y+h, x:x+w]
      eyes = eye_cascade.detectMultiScale(roi_gray)
      for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    # display 
    cv2.imshow('frame',frame)

    # Display the gray frame
    cv2.imshow('gray',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # cv2.imshow('img',img)
  # cv2.waitKey(0)
  cv2.destroyAllWindows()

# print("HEY THIS WORKS")
