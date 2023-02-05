#Photo
import cv2

face_cascade = cv2.CascadeClassifier(r"class/haarcascade_frontalface_default.xml")

img = cv2.imread(r"photos/bilim.jpg") #photo name
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray,1.1,8)

for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w, y+h),(0,255,0),3)

cv2.imshow('Image',img)
cv2.waitKey(0)

"""

#Cam
import cv2

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(0)

while (True):
	ret, frame = video.read()

	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = faceCascade.detectMultiScale(gray_frame, scaleFactor=1.2)

	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2) #Blue, Green, Red -BGR

	cv2.imshow("Goruntu", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

video.release()
cv2.destroyAllWindows()

"""

"""
#Video
import cv2

face_cascade = cv2.CascadeClassifier(r"class/haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(r"photos/video.mp4")
while cap.isOpened():
    _,img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(gray,1.1,8)

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w, y+h),(0,255,0),3)

    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

"""