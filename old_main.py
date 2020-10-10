# todo:
# - add/parse arguments
# - face detection for getting skin color


# img = cv2.imread('/home/img/python.png', cv2.IMREAD_UNCHANGED) todo: do this for video cam

import cv2
import numpy as np

bgSubtractor = None
blurAmount = 40
roi_width = 0.5
roi_height = 1

def extractSkin(frame):
	frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	lowerSkinHSV = np.array([0, 48, 80], np.uint8)
	upperSkinHSV = np.array([20, 255, 255], np.uint8)
	mask = cv2.inRange(frameHSV, lowerSkinHSV, upperSkinHSV)
	cv2.imshow('hsv mask', mask)
	frame = cv2.bitwise_and(frame, frame, mask = mask)
	return frame

def removeBackground(frame):
	if bgSubtractor:
		mask = bgSubtractor.apply(frame, learningRate = 0)
		kernel = np.ones((3,3), np.uint8)
		mask = cv2.erode(mask, kernel, iterations = 1)
		cv2.imshow('mask', mask)
		frame = cv2.bitwise_and(frame, frame, mask = mask)
	return frame

video = cv2.VideoCapture(0)

video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def main():
	global bgSubtractor
	while video.isOpened():
		ret, frame = video.read()

		if not ret:
			break

		frame = cv2.bilateralFilter(frame, 5, 50, 100)
		frame = cv2.flip(frame, 1) # flip horizontally
		cv2.rectangle(frame, (int(roi_width * frame.shape[1]), 0), (frame.shape[1], int(roi_height * frame.shape[0])), (0, 255, 0), 5)
		# skinFrame = extractSkin(frame)
		# cv2.imshow('skin frame', skinFrame)

		if bgSubtractor:
			roi = frame[0:int(roi_height * frame.shape[0]), int(roi_width * frame.shape[1]):frame.shape[1]]
			hand = removeBackground(roi)
			cv2.imshow('hand', hand)
			hand = cv2.GaussianBlur(hand, (5,5), sigmaX = 0)
			cv2.imshow('blured hand', hand)

		cv2.imshow('video', frame)

		key = cv2.waitKey(30)
		if key == 27: # ESC
			video.release()
			cv2.destroyAllWindows()
			break
		elif key == 190: # F1
			bgSubtractor = cv2.createBackgroundSubtractorMOG2(history = 0, varThreshold = 50, detectShadows = True)
		# elif key != -1:
		# 	print(key)

	# on_video('samples/hand_and_face.avi', keepWindows = True)
	# on_video('samples/hand_on_ceiling.avi', keepWindows = True)
	# on_video('samples/hand_on_wall.avi')

if __name__ == "__main__":
	main()