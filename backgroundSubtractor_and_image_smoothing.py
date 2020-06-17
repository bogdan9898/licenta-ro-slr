import cv2
import numpy as np

MAX_TRACKBAR_VALUE = 10

history = 10
varThreshold = 20

apply_morph_open = False
apply_morph_close = False
morph_open_value = 4
morph_close_value = 4
morph_open_iterations = 2
morph_close_iterations = 2

control_window_name = 'Control Window'
input_window_name = 'Input'
output_window_name = 'Output'

history_name= 'History'
varThreshold_name = 'Var threshold'
apply_morph_open_name = 'Apply morph open'
apply_morph_close_name = 'Apply morph close'
morph_open_value_name = 'Morph open value'
morph_close_value_name = 'Morph close value'
morph_open_iterations_name = 'Morph open iterations'
morph_close_iterations_name = 'Morph close iterations'

def generate_subtractor():
	global bg_subtractor
	bg_subtractor = cv2.createBackgroundSubtractorMOG2(history = history, varThreshold = varThreshold, detectShadows = False)

def history_callback(value):
	print('history_callback: ' + str(value))
	global history
	history = value
	generate_subtractor()

def varThreshold_callback(value):
	print('varThreshold_callback: ' + str(value))
	global varThreshold
	varThreshold = value
	generate_subtractor()

def apply_morph_open_callback(value):
	print('apply_morph_open_callback: ' + str(value))
	global apply_morph_open
	apply_morph_open = value

def apply_morph_close_callback(value):
	print('apply_morph_close_callback: ' + str(value))
	global apply_morph_close
	apply_morph_close = value

def morph_open_value_callback(value):
	print('morph_open_value_callback: ' + str(value))
	global morph_open_value
	morph_open_value = value

def morph_close_value_callback(value):
	print('morph_close_value_callback: ' + str(value))
	global morph_close_value
	morph_close_value = value

def morph_open_iterations_callback(value):
	print('morph_open_iterations_callback:' + str(value))
	global morph_open_iterations
	morph_open_iterations = value

def morph_close_iterations_callback(value):
	print('morph_close_iterations_callback: ' + str(value))
	global morph_close_iterations
	morph_close_iterations = value

cv2.namedWindow(control_window_name)

cv2.createTrackbar(history_name, control_window_name, history, 100, history_callback)
cv2.createTrackbar(varThreshold_name, control_window_name, varThreshold, 100, varThreshold_callback)

cv2.createTrackbar(apply_morph_open_name, control_window_name, apply_morph_open, 1, apply_morph_open_callback)
cv2.createTrackbar(apply_morph_close_name, control_window_name, apply_morph_close, 1, apply_morph_close_callback)

cv2.createTrackbar(morph_open_value_name, control_window_name, morph_open_value, MAX_TRACKBAR_VALUE, morph_open_value_callback)
cv2.createTrackbar(morph_close_value_name, control_window_name, morph_close_value, MAX_TRACKBAR_VALUE, morph_close_value_callback)

cv2.createTrackbar(morph_open_iterations_name, control_window_name, morph_open_iterations, MAX_TRACKBAR_VALUE, morph_open_iterations_callback)
cv2.createTrackbar(morph_close_iterations_name, control_window_name, morph_close_iterations, MAX_TRACKBAR_VALUE, morph_close_iterations_callback)

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# bg_subtractor = cv2.createBackgroundSubtractorMOG2(varThreshold = varThreshold, detectShadows = False)
generate_subtractor()

while True:
	ret, frame = video.read()

	if ret == False:
		break

	mask = bg_subtractor.apply(frame)
	# print('ifs: ' + str(apply_morph_open) + ' ' + str(apply_morph_close))
	if apply_morph_open:
		kernel = np.ones((morph_open_value, morph_open_value), np.uint8)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = morph_open_iterations)
	if apply_morph_close:
		kernel = np.ones((morph_close_value, morph_close_value), np.uint8)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = morph_close_iterations)

	cv2.imshow(input_window_name, frame)
	cv2.imshow(output_window_name, mask)

	if cv2.waitKey(30) == 27:
		break

video.release()
cv2.destroyAllWindows()