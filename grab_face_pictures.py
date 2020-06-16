#grab_face_pictures.py

import time, cv2, os

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

print("[INFO] starting webcam")
vs = cv2.VideoCapture(0)
time.sleep(2.0)
total = 0

face_name = input("Enter name: ")
IMG_SAVE_PATH = 'dataset'
IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH, face_name)

try:
    os.mkdir(IMG_SAVE_PATH)
except FileExistsError:
    pass

try:
    os.mkdir(IMG_CLASS_PATH)
except:
    print("Directory already exists")
    print("\nAll images will be gathered")

while True:
	ret, frame = vs.read()
	if not ret:
		continue
	scale_percent = 100
	width = 400 
	height = 400
	dm = (width, height)
	frame = cv2.resize(frame, dm, interpolation = cv2.INTER_AREA)

	rects = detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
		scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

	for (x,y,w,h) in rects:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,255), 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("c"):
		p = os.path.sep.join((IMG_CLASS_PATH, "{}.png".format(str(total).zfill(5))))
		cv2.imwrite(p, frame)
		total += 1
	elif key == ord("q"):
		break

print("[INFO] {} face images stored".format(total))
vs.release()
cv2.destroyAllWindows()
