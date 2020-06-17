import cv2 as cv

haar_cascades = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

video = cv.VideoCapture(0)
video.set(cv.CAP_PROP_FRAME_WIDTH, 1024)
video.set(cv.CAP_PROP_FRAME_HEIGHT, 576)

while True:
	ret, frame = video.read()
	if frame is None:
		break

	frame = cv.flip(frame, 1)
	grayscaled_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	faces = haar_cascades.detectMultiScale(grayscaled_frame, 1.1, 4)

	# frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
	# low_hsv = (0, 0.23 * 255, 0)
	# hight_hsv = (0.5 * 255, 0.68 * 255, 255)
	# new_frame = cv.inRange(frame_HSV, low_hsv, hight_hsv)
    
	copy_frame = frame.copy()
	for x, y, w, h in faces:
		cv.rectangle(copy_frame, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=10)

	roi = frame[y:y+h, x:x+w]
	cv.imshow('roi', roi)

	cv.imshow('video stream', copy_frame)

	key = cv.waitKey(30)
	if key == ord('q') or key == 27:
		break

video.release()