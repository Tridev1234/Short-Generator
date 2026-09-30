import cv2
import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_detection

def get_face_centers(frame):
    h, w, _ = frame.shape
    faces = []
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = detector.process(rgb)
        if res.detections:
            for d in res.detections:
                box = d.location_data.relative_bounding_box
                cx = int((box.xmin + box.width / 2) * w)
                cy = int((box.ymin + box.height / 2) * h)
                faces.append((cx, cy, int(box.width * w), int(box.height * h)))
    return faces

def crop_to_vertical(frame, faces):
    h, w, _ = frame.shape
    target_w = int(h * (9 / 16))

    if len(faces) <= 1:
        cx = faces[0][0] if len(faces) == 1 else w // 2
        x1 = max(0, min(w - target_w, cx - target_w // 2))
        return frame[:, x1:x1 + target_w]
    else:
        # Two speakers -> Stacked split-screen
        faces_sorted = sorted(faces, key=lambda f: f[0])
        f1, f2 = faces_sorted[0], faces_sorted[1]
        sub_h = h // 2
        sub_w = int(sub_h * (9 / 16))
        x1_a = max(0, min(w - sub_w, f1[0] - sub_w // 2))
        x1_b = max(0, min(w - sub_w, f2[0] - sub_w // 2))
        top = cv2.resize(frame[:, x1_a:x1_a + sub_w], (target_w, sub_h))
        bottom = cv2.resize(frame[:, x1_b:x1_b + sub_w], (target_w, sub_h))
        return np.vstack([top, bottom])
