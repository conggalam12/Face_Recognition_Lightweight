# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os
def rotate_point(point,rotation_matrix):
    # Convert coordinates to NumPy arrays for convenience
    point = np.array(point)

    # Add a row [0, 0, 1] to represent homogeneous coordinates
    point_homogeneous = np.append(point, 1)

    # Apply the rotation matrix
    rotated_point_homogeneous = np.dot(rotation_matrix, point_homogeneous)

    # Convert back to coordinates without homogeneous coordinates
    rotated_point = rotated_point_homogeneous[:2]

    return tuple(map(int, rotated_point))
def rotate(right_eye, left_eye, image):
    midpoint = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
    angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]) * 180 / np.pi
    midpoint = tuple(midpoint)
    rotation_matrix = cv2.getRotationMatrix2D(midpoint, angle, scale=1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    return rotated_image , rotation_matrix

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/ipcteam/congnt/face/clean_data/shape_predictor_68_face_landmarks.dat')

folder_origin = "/home/ipcteam/congnt/face/data_face_origin/VN-celeb"

folder_new = "/home/ipcteam/congnt/face/face_recognition/data/data_VN/img"
count = 0 
path_log = '/home/ipcteam/congnt/face/clean_data/log.log'
if os.path.exists(path_log):
	os.remove(path_log)
total_folder = len(os.listdir(folder_origin))
for folder in os.listdir(folder_origin):
	path_folder = os.path.join(folder_origin,folder)
	for img in os.listdir(path_folder):
		path_img = os.path.join(path_folder,img)
		image = cv2.imread(path_img)
		image = imutils.resize(image, width=500)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		if not os.path.exists(os.path.join(folder_new,folder)) :
			os.mkdir(os.path.join(folder_new,folder))
		path_new_img = os.path.join(folder_new,folder,img)
		# detect faces in the grayscale image
		rects = detector(gray, 1)
		if len(rects) !=1 :
			continue
		# loop over the face detections
		for (i, rect) in enumerate(rects):

			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			# 20 : top left eye 
			# 25 : top right eye
			# 1 : left face
			# 17 : right face
			# 9 : bottom face
			right_eye = shape[25]
			left_eye = shape[20]
			rotate_img,rotate_matrix = rotate(right_eye,left_eye,image)
			x_max = shape[0][0]
			y_max = shape[0][1]
			x_min = shape[0][0]
			y_min = shape[0][1]
			for (x,y) in shape:
				(x,y) = rotate_point((x,y),rotate_matrix)
				if x_max < x:
					x_max = x
				elif x_min > x:
					x_min = x
				elif y_max < y:
					y_max = y
				elif y_min > y:
					y_min = y
				# cv2.circle(rotate_img,(x,y),2,(0,255,0),-1)
			crop_img = rotate_img[y_min-40:y_max+40,x_min-20:x_max+20]
			(a,b,c) = crop_img.shape
			if a < 112 or b < 112:
				continue
			cv2.imwrite(path_new_img,crop_img)
	count+=1
	with open('/home/ipcteam/congnt/face/clean_data/log.log','a') as file:
		file.write("Done folder {}/{}\n".format(count,total_folder))
	print("Done folder {} - {}/{}".format(folder,count,total_folder))
