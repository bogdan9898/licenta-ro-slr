import numpy as np
import cv2
# from sklearn.cluster import KMeans
import tensorflow as tf
from kmeanstf import TunnelKMeansTF as KMeans
from matplotlib import pyplot as plt
import argparse
import os
import time
from threading import Thread
from time import time
from queue import Queue

print("\033[92mPID: {0}\033[0m".format(os.getpid()))

windows = Queue(-1)

# config = tf.compat.v1.ConfigProto()
# print(config)
# print(tf.config.list_physical_devices("GPU"))
# exit(0)

# from tensorflow.python.client import device_lib as dev_lib
# print (dev_lib.list_local_devices())
# exit(0)

def draw_text(frame, text, font=cv2.FONT_HERSHEY_PLAIN, scale=4, thickness=3):
	frame = frame.copy()
	text_size = cv2.getTextSize(text, font, scale, thickness)[0]
	x = round((frame.shape[1] - text_size[0]) / 2)
	y = round((frame.shape[0] + text_size[1]) / 2)
	cv2.putText(frame, text, (x, y), font, scale, (0, 0, 255), thickness, cv2.LINE_AA)
	return frame

def extract_face(frame):
	# HAAR cascades
	frame = frame.copy()
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame_gray = cv2.equalizeHist(frame_gray)

	faces = face_cascade.detectMultiScale(frame_gray)
	if len(faces) <= 0:
		return None

	x,y,w,h = faces[0]
	face_roi = frame[y:y+h, x:x+w]
	return face_roi
	# cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4, cv2.LINE_AA)
	# return frame

def draw_classified_colors(classified_colors):
	w = 50
	h = 50*classified_colors.shape[0]
	frame = np.zeros((h, w, 4), dtype=np.uint8)
	i = 0
	for color in classified_colors:
		cv2.rectangle(frame, (0, 50*i), (w, 50*(i+1)), color.numpy().tolist(), -1)
		i += 1
	return frame

def classify_colors(frame, n_clusters=3):
	frame = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = frame.reshape((frame.shape[0]*frame.shape[1]), 3)
	classifier = KMeans(n_clusters=n_clusters, random_state=0)
	classifier.fit(frame)
	# print("labels:")
	# print(classifier.labels_)
	# print("cluster centers:")
	# print(classifier.cluster_centers_)
	
	colors_plot = draw_classified_colors(classifier.cluster_centers_)
	windows.put(('colors plot', colors_plot))
	
	# info = getColorInformation(classifier.labels_, classifier.cluster_centers_, False)

	# print("Color Information")
	# # prety_print_data(info)

	# # Show in the dominant color as bar
	# print("Color Bar")
	# color_bar = plotColorBar(info)
	# cv2.imshow('colors', color_bar)
	# plt.subplot(3, 1, 3)
	# plt.axis("off")
	# plt.imshow(colour_bar)
	# plt.title("Color Bar")

def extract_skin(frame):
	# HSV extraction
	frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	lower_threshold = np.array((0, 48, 80))
	upper_threshold = np.array((20, 255, 255))

	skin_mask = cv2.inRange(frame_hsv, lower_threshold, upper_threshold)
	skin_mask = cv2.GaussianBlur(skin_mask, (3,3), 0)
	result = cv2.bitwise_and(frame, frame, mask=skin_mask)

	cv2.imshow('skin', result)

def thread_work(frame):
	new_frame = extract_face(frame)
	if new_frame is None:
		frame = draw_text(frame, 'No face detected')
	else:
		frame = new_frame
		classify_colors(frame, 20)
	windows.put(('face', frame))

parser = argparse.ArgumentParser(description='Sign language classifier')
parser.add_argument('--face-cascade', help='Path to face haarcascade file', default='haarcascades/haarcascade_frontalface_alt.xml')
parser.add_argument('--video', help='Path to video file or camera device number', default=0)
args = parser.parse_args()
face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(cv2.samples.findFile(args.face_cascade)):
	print('An error occured while loading face cascade')
	exit(0)

cap = cv2.VideoCapture(args.video)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened:
	print('Video stream cannot be opened')
	exit(0)

prev_time = time()
while True:
	current_time = time()

	ret, frame = cap.read()
	if ret is None or frame is None:
		print('Video stream reached EOL')
		break

	frame = cv2.flip(frame, 1) # flip horizontally
	cv2.imshow('video stream', frame)

	print(f'{prev_time} -> {current_time}')
	if current_time - prev_time > 1:
		print('running thread')
		prev_time = current_time
		Thread(target=thread_work, args=(frame,)).start()

	while not windows.empty():
		window_to_show = windows.get()
		cv2.imshow(window_to_show[0], window_to_show[1])

	key = cv2.waitKey(30)
	if key == 27: # ESC
		cap.release()
		cv2.destroyAllWindows()
		break