# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os

def rotate_point(point, rotation_matrix):
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
    return rotated_image, rotation_matrix

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
    path_folder = os.path.join(folder_origin, folder)
    for img in os.listdir(path_folder):
        path_img = os.path.join(path_folder, img)
        image = cv2.imread(path_img)
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if not os.path.exists(os.path.join(folder_new, folder)):
            os.mkdir(os.path.join(folder_new, folder))
        path_new_img = os.path.join(folder_new, folder, img)
        # detect faces in the grayscale image
        rects = detector(gray, 1)
        if len(rects) != 1:
            continue
        # loop over the face detections
        
        shape = predictor(gray, rects[0])
        lm = face_utils.shape_to_np(shape)
        lm_eye_left = lm[36:42]  # left-clockwise
        lm_eye_right = lm[42:48]  # left-clockwise
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        lm_mouth_outer = lm[48:60]
        center_mouth = np.mean(lm_mouth_outer, axis=0)
        rotated_image, rotation_matrix = rotate(eye_right, eye_left, image)
        # rotate point to new point
        eye_left = rotate_point(eye_left, rotation_matrix)
        eye_right = rotate_point(eye_right, rotation_matrix)
        center_mouth = rotate_point(center_mouth, rotation_matrix)
        mouth_rec_left = (eye_left[0], center_mouth[1])
        mouth_rec_right = (eye_right[0], center_mouth[1])
        top_left = (int(eye_left[0] * 0.65), int(eye_left[1] * 0.65))
        bottom_right = (int(mouth_rec_right[0] * 1.2), int(mouth_rec_right[1] * 1.2))
        x1, y1 = top_left
        x2, y2 = bottom_right
        crop_img = rotated_image[y1:y2, x1:x2]
        (a, b, c) = crop_img.shape
        if a < 112 or b < 112:
            continue
        cv2.imwrite(path_new_img, crop_img)
    count += 1
    with open('/home/ipcteam/congnt/face/clean_data/log.log', 'a') as file:
        file.write("Done folder {}/{}\n".format(count, total_folder))
    print("Done folder {} - {}/{}".format(folder, count, total_folder))
