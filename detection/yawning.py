import cv2
import numpy as np
from collections import deque
import dlib

class YawningDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.yawn_counter = 0
        self.yawn_history = deque(maxlen=30)  # Track past MARs
        self.yawn_detected = False

    def _get_mouth_aspect_ratio(self, landmarks):
        A = np.linalg.norm(landmarks[62] - landmarks[66])  # Vertical
        B = np.linalg.norm(landmarks[63] - landmarks[65])
        C = np.linalg.norm(landmarks[60] - landmarks[64])  # Horizontal
        mar = (A + B) / (2.0 * C)
        return mar

    def _get_lip_distance(self, landmarks):
        top_lip = np.mean([landmarks[62], landmarks[63], landmarks[64]], axis=0)
        bottom_lip = np.mean([landmarks[66], landmarks[67], landmarks[65]], axis=0)
        return np.linalg.norm(top_lip - bottom_lip)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        for face in faces:
            shape = self.predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])

            mar = self._get_mouth_aspect_ratio(landmarks)
            lip_distance = self._get_lip_distance(landmarks)

            self.yawn_history.append(mar)
            avg_mar = np.mean(self.yawn_history) if len(self.yawn_history) >= 10 else 0.5

            if mar > avg_mar * 1.6 or lip_distance > 15:
                self.yawn_counter += 1
            else:
                self.yawn_counter = 0

            if self.yawn_counter > 15:
                self.yawn_detected = True
            else:
                self.yawn_detected = False

        return self.yawn_detected, frame
