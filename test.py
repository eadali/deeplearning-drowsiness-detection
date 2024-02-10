import dlib
import cv2 
import torch
from models import ConvolutionalNeuralNetwork
from imutils import face_utils
from utils import detect_sleeping
from utils import draw_detection, draw_decision


# Parameters
ear_threshold = 0.2
image_path = './data/open.png'
# image_path = './data/closed.png'
landmark_path = "shape_predictor_68_face_landmarks.dat"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load model
cnn = ConvolutionalNeuralNetwork(device, landmark_path)

# Read image
bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Detect face
rect, landmarks = cnn(bgr)
frame = bgr.copy()
if rect is not None:
    is_sleeping = detect_sleeping(landmarks, ear_threshold)
    
    draw_detection(frame, rect, landmarks)
    draw_decision(frame, rect, is_sleeping)

# Display the resulting frame 
cv2.imshow('frame', frame) 
cv2.waitKey(0)
cv2.destroyAllWindows()




