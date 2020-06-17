import cv2
import numpy as np

ESC_KEY = 27

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1280,720))

while True:
	ret, frame = cam.read()

	out.write(frame)

	cv2.imshow('Video Stream', frame)

	key = cv2.waitKey(1)
	if key == ESC_KEY or not ret:
		break

cam.release()
out.release()
cv2.destroyAllWindows()