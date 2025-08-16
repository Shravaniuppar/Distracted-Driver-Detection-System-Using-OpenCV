# hand_off_wheel.py
import cv2
#import numpy as np
import mediapipe as mp

class HandOffWheelDetector:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(min_detection_confidence=0.7)
        self.hand_off_counter = 0
        self.HAND_OFF_THRESH = 20  # Number of consecutive frames

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        hand_on_wheel = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    # Define your steering wheel region here
                    if self.is_in_steering_wheel_region(x, y, frame):
                        hand_on_wheel = True
                        break
                if hand_on_wheel:
                    break

        if not hand_on_wheel:
            self.hand_off_counter += 1
        else:
            self.hand_off_counter = 0

        hand_off = self.hand_off_counter > self.HAND_OFF_THRESH
        if hand_off:
            cv2.putText(frame, "Hands Off Wheel!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return hand_off, frame

    def is_in_steering_wheel_region(self, x, y, frame):
        # Define the region corresponding to the steering wheel
        # For example, a rectangle in the lower center of the frame
        h, w = frame.shape[:2]
        return (w//3 < x < 2*w//3) and (2*h//3 < y < h)
