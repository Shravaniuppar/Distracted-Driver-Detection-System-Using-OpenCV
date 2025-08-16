# head_pose.py
import cv2
import numpy as np
import mediapipe as mp

class HeadPoseDetector:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        self.FACE_3D = np.array([
            (0.0, 0.0, 0.0),    # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),   # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)   # Right mouth corner
        ])
        self.THRESHOLD = 25  # degrees

    def detect(self, frame):
        img_h, img_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = face_landmarks.landmark

            image_points = np.array([
                (landmarks[1].x * img_w, landmarks[1].y * img_h),     # Nose tip
                (landmarks[152].x * img_w, landmarks[152].y * img_h), # Chin
                (landmarks[263].x * img_w, landmarks[263].y * img_h), # Left eye left corner
                (landmarks[33].x * img_w, landmarks[33].y * img_h),   # Right eye right corner
                (landmarks[287].x * img_w, landmarks[287].y * img_h), # Left Mouth corner
                (landmarks[57].x * img_w, landmarks[57].y * img_h)    # Right mouth corner
            ], dtype="double")

            focal_length = img_w
            center = (img_w / 2, img_h / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")

            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.FACE_3D, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
                proj_matrix = np.hstack((rvec_matrix, translation_vector))
                euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

                pitch, yaw, roll = [angle[0] for angle in euler_angles]

                if abs(pitch) > self.THRESHOLD or abs(yaw) > self.THRESHOLD:
                    cv2.putText(frame, "Distracted Head Pose", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    return True, frame

        return False, frame
