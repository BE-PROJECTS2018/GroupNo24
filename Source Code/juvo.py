import numpy as np
import cv2
import tensorflow as tf
import imutils
import pyttsx3
import math
import os
import speech_recognition as sr

from sklearn.externals import joblib
from scipy.spatial import distance as dist
from imutils.video import FPS
from utils import label_map_util
from utils import visualization_utils as vis_util

def find_angle(aX, aY, bX, bY):
	if(bX-aX != 0):
		m = abs((bY-aY)/(bX-aX))
		z = 90-math.degrees(math.atan(m))
		return int(z)
	elif bY-aY == 0:
		return 90
	else:
		return 0

def left_or_right(k, centX, input):
	if k == 0:
		engine.say(input + "is not in frame")
		engine.runAndWait()
	elif k==1:
		if (centX[j] < 187):
			engine.say(input + "is at the left")
		elif (centX[j] < 222):
			engine.say(input + "is at the center")
		else:
			engine.say(input + "is at the right")
		engine.runAndWait()
	else:
		for j in range(k):
			if (centX[j] < 187):
				engine.say(input + str(j+1) + "is at the left")
			elif (centX[j] < 222):
				engine.say(input + str(j+1) + "is at the center")
			else:
				engine.say(input + str(j+1) + "is at the right")
			engine.runAndWait()
		
def distance_calculation(k, centX, centY, center, reg):
	if k == 0:
		engine.say(input + "is not in frame")
		engine.runAndWait()
	else:
		for j in range(k):
			D = dist.euclidean((center[0], center[1]), (centX[j], centY[j]))
			angle = find_angle(center[0], center[1], centX[j], centY[j])
			D = D*0.0265
			D = int(reg.predict(np.reshape([D, angle],(1,-1))))
			print(angle, " Degrees")
			print(D, " Centimeters")
			D = str(D)
			angle = str(angle)
			engine.say("At" + D + " centimeters and" + angle + "degrees")
			engine.runAndWait()

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
PATH_TO_CKPT = 'ssd_mobilenet.pb'
PATH_TO_LABELS = 'mscoco_label_map.pbtxt'
NUM_CLASSES = 90

print("[INFO] loading model...")
detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)
fps = FPS().start()
engine = pyttsx3.init()
r = sr.Recognizer()
reg = joblib.load('gradient_boosting_model.pkl')
with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		while True:
			ret, frame = cap.read()
			frame = imutils.resize(frame, width=400)
			image_np_expanded = np.expand_dims(frame, axis=0)
			ch = cv2.waitKey(1)
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
			scores = detection_graph.get_tensor_by_name('detection_scores:0')
			classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')
			(boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})
			vis_util.visualize_boxes_and_labels_on_image_array(frame, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, use_normalized_coordinates=True)
			if ch == ord('y'):
				objects = vis_util.objects_in_frame(np.squeeze(scores), np.squeeze(classes))
				for i in objects:
					engine.say("There is" + category_index[int(objects[i])]['name'])
					engine.runAndWait()
			if ch == ord('s'):
				with sr.Microphone() as source:
					audio = r.listen(source)
				try:
					input = r.recognize_google(audio)
					objects = vis_util.objects_in_frame(np.squeeze(scores), np.squeeze(classes))
					flag = False
					for i in objects:
						if category_index[int(objects[i])]['name'] == input:
							flag = True
							break
					if flag:
						engine.say("Yes! There is" + input)
						engine.runAndWait()
						idx = int(objects[i])
						while True:
							val, map_frame = cap.read()
							map_frame = imutils.resize(map_frame, width=400)
							img_np_expanded = np.expand_dims(map_frame, axis=0)
							ch2 = cv2.waitKey(1)
							img_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
							boxes_arr = detection_graph.get_tensor_by_name('detection_boxes:0')
							scores_arr = detection_graph.get_tensor_by_name('detection_scores:0')
							classes_arr = detection_graph.get_tensor_by_name('detection_classes:0')
							num_detections_arr = detection_graph.get_tensor_by_name('num_detections:0')
							(boxes_arr, scores_arr, classes_arr, num_detections_arr) = sess.run([boxes_arr, scores_arr, classes_arr, num_detections_arr], feed_dict={img_tensor: img_np_expanded})
							k, centX, centY = vis_util.visualize_only_mapped_boxes_on_image_array(map_frame, idx, np.squeeze(boxes_arr), np.squeeze(classes_arr).astype(np.int32), np.squeeze(scores_arr), category_index, use_normalized_coordinates=True)
							cv2.circle(map_frame, (centX[0], centY[0]), 3,(0, 255, 255), 2)
							#glove detection
							frame_to_thresh = cv2.cvtColor(map_frame, cv2.COLOR_BGR2HSV)
							thresh = cv2.inRange(frame_to_thresh, (30, 80, 35), (65, 255, 166)) # set min & max HSV values
							kernel = np.ones((5,5),np.uint8)
							mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
							mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
							cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
							center = None
							if len(cnts) > 0:
								c = max(cnts, key=cv2.contourArea)
								((x, y), radius) = cv2.minEnclosingCircle(c)
								M = cv2.moments(c)
								center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
								cv2.circle(map_frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
								cv2.circle(map_frame, center, 3, (0, 0, 255), -1)
								cv2.putText(map_frame,"centroid", (center[0]+10,center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 255),1)
								cv2.putText(map_frame,"("+str(center[0])+","+str(center[1])+")", (center[0]+10,center[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 255),1)
							if ch2 == ord("x"):
								left_or_right(k, centX, input)
							if ch2 == ord("d"):
								if len(cnts) > 0:
									distance_calculation(k, centX, centY, center, reg)
								else:
									engine.say("Hand is not in frame")
									engine.runAndWait()
							cv2.imshow('Mapped',map_frame)
							if ch2 == ord('m'):
								cv2.destroyWindow("Mapped")
								break
					else:
						engine.say("There is no" + input)
						engine.runAndWait()
				except:
					engine.say("Couldn't hear you!")
					engine.runAndWait()
			cv2.imshow('Input',frame)
			if ch == ord('q'):
				break
			fps.update()
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()				
cap.release()