from facenet_pytorch import MTCNN
import dlib
import numpy as np
import cv2
import torch


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


class ConvolutionalNeuralNetwork:
    def __init__(self, device, landmark_path):
        # Load pytorch face detection model
        device = torch.device(device)
        print('Running on device: {}'.format(device))
        self.face_detector = MTCNN(keep_all=True, device=device)
        # Load dlib landmark detection model
        self.landmark_detector = dlib.shape_predictor(landmark_path)

    def __call__(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        boxes, _ = self.face_detector.detect(rgb)
        rect = None
        shape = None
        if boxes is not None:
            rect = boxes[0].astype(int)
            rect[:2] = rect[:2] - 10
            rect[2:] = rect[2:] + 10
            drect = dlib.rectangle(rect[0], rect[1], rect[2], rect[3])
            shape = self.landmark_detector(gray, drect)
            shape = shape_to_np(shape)
        return rect, shape 
