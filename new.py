import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# img = cv2.imread('B.jpg')
# To capture video from webcam
webcam = cv2.VideoCapture(0)

while True:
    
    successful_frame_read, frame = webcam.read()
    frame = cv2.flip(frame, 1)
    
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(128, 256), randrange(128, 256), randrange(128, 256) ), 2)
    
    cv2.imshow('Clever Programmer Face Detector', frame)

    # To prevent the image from closing
    key = cv2.waitKey(1)
    if key==81 or key==113:
        break

webcam.release()


print("Code Completed")