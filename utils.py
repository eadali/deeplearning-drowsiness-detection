from scipy.spatial import distance as dist
import numpy as np
from imutils import face_utils
import cv2


def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear


def detect_sleeping(landmarks, ear_threshold):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEAR = eye_aspect_ratio(landmarks[lStart:lEnd])
    rightEAR = eye_aspect_ratio(landmarks[rStart:rEnd])
    return max(leftEAR,rightEAR) < ear_threshold


def draw_detection(frame, rect, landmarks):
    point1 = rect[:2]
    point2 = rect[2:]
    color = (160, 32, 240)
    width=6
    cv2.rectangle(frame, point1, point2, color, width) 
    draw_eyes(frame, landmarks)
    draw_jaw(frame, landmarks)
    draw_nose(frame, landmarks)
    draw_eyebrows(frame, landmarks)
    draw_merge(frame, landmarks)


def draw_eyes(frame, landmarks):
    color = (231,221,153)
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    cv2.drawContours(frame, [landmarks[lStart:lEnd]], -1, color, 1) 
    cv2.drawContours(frame, [landmarks[rStart:rEnd]], -1, color, 1)


def draw_eyebrows(frame, landmarks):
    color = (231,221,153)
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
    for i in range(lStart, lEnd-1):
        cv2.line(frame, landmarks[i], landmarks[i+1], color)
    for i in range(rStart, rEnd-1):
        cv2.line(frame, landmarks[i], landmarks[i+1], color)
    

def draw_jaw(frame, landmarks):
    color = (231,221,153)
    (Start, End) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
    for i in range(Start, End-1):
        cv2.line(frame, landmarks[i], landmarks[i+1], color)

def draw_nose(frame, landmarks):
    color = (231,221,153)
    for i in range(27, 30):
        cv2.line(frame, landmarks[i], landmarks[i+1], color)

def draw_merge(frame, landmarks):
    color = (231,221,153)
    cv2.line(frame, landmarks[27], landmarks[39], color)
    cv2.line(frame, landmarks[27], landmarks[42], color)
    cv2.line(frame, landmarks[0], landmarks[17], color)
    cv2.line(frame, landmarks[16], landmarks[26], color)
    cv2.line(frame, landmarks[21], landmarks[27], color)
    cv2.line(frame, landmarks[22], landmarks[27], color)

def draw_decision(frame, rect, is_sleeping):
    print(is_sleeping)
    text = 'Sleeping' if is_sleeping else 'Not Sleeping'
    font = cv2.FONT_HERSHEY_SIMPLEX 
    point1 = (rect[0]-3, rect[1]-30)
    point2 = (rect[0]+150, rect[1])
    color = (160, 32, 240)
    cv2.rectangle(frame, point1, point2, color, -1) 
    color = (0, 0, 0) 
    point = (rect[0]+2, rect[1]-8)
    fontScale = 0.7
    thickness = 1
    cv2.putText(frame, text, point, font, fontScale, color, thickness, cv2.LINE_AA) 
        
    