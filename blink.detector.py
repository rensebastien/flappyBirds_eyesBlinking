import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

class BlinkDetector:
    def __init__(self, shape_predictor_path='assets/shape_predictor_68_face_landmarks.dat', ear_thresh=0.22, consec_frames=3):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor_path)
        self.EYE_AR_THRESH = ear_thresh
        self.EYE_AR_CONSEC_FRAMES = consec_frames
        self.COUNTER = 0

        self.LEFT_EYE_POINTS = list(range(42, 48))
        self.RIGHT_EYE_POINTS = list(range(36, 42))

    @staticmethod
    def eye_aspect_ratio(eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def detect_blink(self, frame):
        """Returns True if blink detected in this frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        for rect in rects:
            landmarks = np.matrix([[p.x, p.y] for p in self.predictor(frame, rect).parts()])
            left_eye = landmarks[self.LEFT_EYE_POINTS]
            right_eye = landmarks[self.RIGHT_EYE_POINTS]

            ear_left = self.eye_aspect_ratio(left_eye)
            ear_right = self.eye_aspect_ratio(right_eye)
            ear_avg = (ear_left + ear_right) / 2.0

            if ear_avg < self.EYE_AR_THRESH:
                self.COUNTER += 1
            else:
                if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                    self.COUNTER = 0
                    return True
                self.COUNTER = 0
        return False
