import cv2
import mediapipe as mp

class PhoneUsageDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_detection
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6)
        self.face = self.mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_hands = self.hands.process(frame_rgb)
        results_face = self.face.process(frame_rgb)

        phone_detected = False

        if results_hands.multi_hand_landmarks and results_face.detections:
            for detection in results_face.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                face_x = int(bboxC.xmin * iw)
                face_y = int(bboxC.ymin * ih)
                face_w = int(bboxC.width * iw)
                face_h = int(bboxC.height * ih)
                face_box = (face_x, face_y, face_x + face_w, face_y + face_h)

                for hand_landmarks in results_hands.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        hand_x = int(lm.x * iw)
                        hand_y = int(lm.y * ih)
                        if (face_box[0] < hand_x < face_box[2]) and (face_box[1] < hand_y < face_box[3]):
                            phone_detected = True
                            break
                    if phone_detected:
                        break

        return phone_detected, frame
